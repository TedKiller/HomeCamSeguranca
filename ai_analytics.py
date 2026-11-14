"""Pipeline de Visão Computacional e IA em tempo real para CCTV (ONNX DirectML / CPU).

Executa detecção e rastreamento multiobjeto com predição de movimento (Pessoas, Animais,
Carros, Motos e Bicicletas), classificação de gênero e cores predominantes das roupas
(parte superior e inferior) para pessoas, ALPR (placas brasileiras antiga e Mercosul),
estimativa perceptual CIELAB/HSV de cor e classificação visual Multi-View de modelo veicular.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import queue
import re
import threading
import time
import unicodedata

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_PROMPTS_NPZ = BASE_DIR / "assets" / "mobileclip_prompts.npz"

COCO_PERSON_ID = 0
COCO_BICYCLE_ID = 1
COCO_CAR_ID = 2
COCO_MOTORCYCLE_ID = 3
COCO_BUS_ID = 5
COCO_TRUCK_ID = 7
COCO_ANIMAL_LABELS: dict[int, str] = {
    14: "Pássaro",
    15: "Gato",
    16: "Cachorro",
    17: "Cavalo",
    18: "Ovelha",
    19: "Vaca",
    20: "Animal",
    21: "Animal",
    22: "Animal",
    23: "Animal",
}

PLATE_MERCOSUL_RE = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
PLATE_ANTIGA_RE = re.compile(r"^[A-Z]{3}[0-9]{4}$")

_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "A": "4"}

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


@dataclass
class DetectionItem:
    """Elemento detectado e rastreado em um quadro do fluxo de vídeo."""

    track_id: int
    category: str  # "Pessoa", "Animal", "Carro", "Moto", "Bicicleta"
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    sub_label: str = ""
    plate: str = ""
    color: str = ""
    approx_model: str = ""
    upper_color: str = ""
    lower_color: str = ""


@dataclass
class VehicleRecord:
    """Registro consolidado para a tabela em tempo real de veículos (Carros e Motos)."""

    track_id: int
    timestamp: str
    vehicle_type: str  # "Carro" ou "Moto"
    plate: str = "Lendo..."
    plate_conf: float = 0.0
    color: str = "Indefinida"
    approx_model: str = "Analisando..."
    model_conf: float = 0.0
    camera_name: str = ""
    plate_votes: Counter[str] = field(default_factory=Counter, repr=False)
    color_votes: dict[str, float] = field(default_factory=dict, repr=False)
    model_scores: dict[str, float] = field(default_factory=dict, repr=False)


@dataclass
class PersonRecord:
    """Registro consolidado para a tabela em tempo real de pessoas e vestuário."""

    track_id: int
    timestamp: str
    person_type: str = "Indeterminado"  # "Homem", "Mulher" ou "Indeterminado"
    clothing_colors: str = "Analisando..."
    upper_color: str = "Indefinida"
    lower_color: str = "Indefinida"
    gender_conf: float = 0.0
    camera_name: str = ""
    gender_votes: dict[str, float] = field(default_factory=dict, repr=False)
    upper_color_votes: dict[str, float] = field(default_factory=dict, repr=False)
    lower_color_votes: dict[str, float] = field(default_factory=dict, repr=False)


def format_clothing_colors(upper_color: str, lower_color: str) -> str:
    """Formata o resumo compacto de cor predominante superior/inferior para tabela e CSV."""
    valid_u = bool(upper_color and upper_color != "Indefinida")
    valid_l = bool(lower_color and lower_color not in ("Indefinida", "Não visível"))
    if valid_u and valid_l:
        return f"Sup: {upper_color} | Inf: {lower_color}"
    if valid_u:
        suffix = " | Inf: Não visível" if lower_color == "Não visível" else ""
        return f"Sup: {upper_color}{suffix}"
    if valid_l:
        return f"Inf: {lower_color}"
    return "Analisando..."


def _accumulate_weighted_vote(
    votes: dict[str, float],
    label: str,
    weight: float,
) -> tuple[str, float]:
    """Acumula evidência ponderada por qualidade de quadro e retorna `(vencedor, participação)`."""
    if not label:
        return ("", 0.0)
    votes[label] = votes.get(label, 0.0) + max(0.05, float(weight))
    total = sum(votes.values())
    if total <= 0.0:
        return (label, 0.0)
    best_label = max(votes.items(), key=lambda item: item[1])[0]
    return (best_label, votes[best_label] / total)


def _crop_quality_weight(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
    """Calcula peso de confiabilidade de um recorte pelo tamanho e centralidade no quadro."""
    x1, y1, x2, y2 = bbox
    area_ratio = max(1, (x2 - x1) * (y2 - y1)) / float(max(1, frame_w * frame_h))
    weight = max(0.45, min(2.5, 0.55 + area_ratio * 22.0))
    # Penaliza objetos parcialmente cortados nas bordas do frame
    if x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2:
        weight *= 0.65
    return weight


@dataclass
class AISummary:
    """Resumo instantâneo das contagens e registros do frame atual."""

    people_total: int = 0
    men_count: int = 0
    women_count: int = 0
    animals_count: int = 0
    cars_count: int = 0
    motos_count: int = 0
    bicycles_count: int = 0
    last_plate: str = ""
    backend_name: str = "Inativo"


def _ascii_overlay_text(text: str) -> str:
    """Remove acentuação para renderização limpa com fontes Hershey do OpenCV."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def create_ort_session(onnx_path: Path | str, prefer_gpu: bool = True) -> tuple[object, str]:
    """Cria uma sessão ONNX Runtime priorizando DirectML (GPU Windows) com fallback para CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list[str] = []
    if prefer_gpu and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    active = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
    return sess, active


def ensure_ai_models(
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Garante o download local dos modelos ONNX necessários para detecção, gênero e veículo."""
    import requests
    from huggingface_hub import hf_hub_download

    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    yolo_target = root / "yolov10s" / "onnx" / "model.onnx"
    if not yolo_target.exists():
        log("[IA] Baixando detector YOLOv10s ONNX (~28 MB)...")
        hf_hub_download("onnx-community/yolov10s", "onnx/model.onnx", local_dir=str(root / "yolov10s"))

    ff_target = root / "fairface" / "onnx" / "model_fp16.onnx"
    if not ff_target.exists():
        log("[IA] Baixando modelo de gênero FairFace ONNX (~164 MB)...")
        hf_hub_download(
            "onnx-community/fairface_gender_image_detection-ONNX",
            "onnx/model_fp16.onnx",
            local_dir=str(root / "fairface"),
        )

    yunet_target = root / "face" / "face_detection_yunet_2023mar.onnx"
    if not yunet_target.exists():
        log("[IA] Baixando detector facial YuNet (~230 KB)...")
        yunet_target.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(YUNET_URL, timeout=30)
        resp.raise_for_status()
        yunet_target.write_bytes(resp.content)

    mclip_target = root / "mobileclip_s2" / "onnx" / "vision_model.onnx"
    if not mclip_target.exists():
        log("[IA] Baixando classificador visual MobileCLIP-S2 (~136 MB)...")
        hf_hub_download("Xenova/mobileclip_s2", "onnx/vision_model.onnx", local_dir=str(root / "mobileclip_s2"))

    return {
        "yolo": yolo_target,
        "fairface": ff_target,
        "yunet": yunet_target,
        "mobileclip_vision": mclip_target,
        "prompts_npz": DEFAULT_PROMPTS_NPZ,
    }


def _try_repair_plate_window(window: str) -> str | None:
    if len(window) != 7:
        return None
    chars = list(window)
    for idx in (0, 1, 2):
        chars[idx] = _TO_LETTER.get(chars[idx], chars[idx])
    for idx in (3, 5, 6):
        chars[idx] = _TO_DIGIT.get(chars[idx], chars[idx])

    candidate = "".join(chars)
    if PLATE_MERCOSUL_RE.match(candidate):
        return candidate
    if PLATE_ANTIGA_RE.match(candidate):
        return f"{candidate[:3]}-{candidate[3:]}"
    return None


def normalize_brazilian_plate(raw_text: str) -> str | None:
    """Normaliza leitura OCR e valida placa brasileira Antiga (AAA-0000) ou Mercosul (AAA0A00)."""
    if not raw_text:
        return None
    stripped = re.sub(r"^(?:BRASIL|MERCOSUL)\b[\s\-:]*|^BR[\s\-:]+", "", raw_text.upper().strip())
    cleaned = re.sub(r"[^A-Z0-9]", "", stripped)
    if len(cleaned) < 7:
        return None

    # 1) Prioriza correspondência exata sem substituições (evita falsos positivos com prefixos tipo "BR")
    for start in range(len(cleaned) - 6):
        win = cleaned[start : start + 7]
        if PLATE_MERCOSUL_RE.match(win):
            return win
        if PLATE_ANTIGA_RE.match(win):
            return f"{win[:3]}-{win[3:]}"

    # 2) Fallback com reparo posicional de ambiguidade OCR (O/0, I/1, S/5, B/8, A/4)
    start_offsets = list(range(len(cleaned) - 6))
    if cleaned.startswith("BR") and len(cleaned) >= 9:
        start_offsets = [2] + [i for i in start_offsets if i != 2]
    for start in start_offsets:
        cand = _try_repair_plate_window(cleaned[start : start + 7])
        if cand is not None:
            return cand
    return None


def is_ir_frame(frame_bgr: np.ndarray) -> bool:
    """Detecta no quadro inteiro se a câmera está em modo P&B/Infravermelho noturno."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    sample = (
        cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        if max(h, w) > 64
        else frame_bgr
    )
    b, g, r = cv2.split(sample)
    diff_rg = float(np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16))))
    diff_gb = float(np.mean(np.abs(g.astype(np.int16) - b.astype(np.int16))))
    return diff_rg < 2.2 and diff_gb < 2.2


def _classify_roi_color(
    roi_bgr: np.ndarray,
    *,
    is_vehicle: bool = True,
    suppress_skin: bool = False,
    apply_center_weight: bool = True,
    allow_ir: bool | None = None,
) -> str:
    """Classifica cor predominante via espaço perceptual CIELAB + HSV com votação espacial ponderada."""
    if roi_bgr is None or roi_bgr.size == 0 or min(roi_bgr.shape[:2]) < 8:
        return "Indefinida"

    rh, rw = roi_bgr.shape[:2]
    max_dim = max(rh, rw)
    if max_dim > 96:
        scale = 96.0 / float(max_dim)
        roi_work = cv2.resize(
            roi_bgr,
            (max(8, int(round(rw * scale))), max(8, int(round(rh * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        roi_work = roi_bgr

    b, g, r = cv2.split(roi_work)
    diff_rg = float(np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16))))
    diff_gb = float(np.mean(np.abs(g.astype(np.int16) - b.astype(np.int16))))
    if diff_rg < 2.2 and diff_gb < 2.2:
        if allow_ir is True:
            return "P&B (Noturno IR)"
        if allow_ir is None and 55.0 <= float(np.mean(g)) <= 195.0:
            return "P&B (Noturno IR)"

    wh, ww = roi_work.shape[:2]
    hsv = cv2.cvtColor(roi_work, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(roi_work, cv2.COLOR_BGR2Lab).reshape(-1, 3)

    if apply_center_weight:
        yy = np.linspace(-1.0, 1.0, wh, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, ww, dtype=np.float32)[None, :]
        weights = np.exp(-1.15 * (xx * xx + yy * yy)).reshape(-1)
    else:
        weights = np.ones(wh * ww, dtype=np.float32)

    h_vals = hsv[:, 0]
    s_vals = hsv[:, 1]
    v_vals = hsv[:, 2]
    l_lab = lab[:, 0].astype(np.float32)
    a_off = lab[:, 1].astype(np.float32) - 128.0
    b_off = lab[:, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a_off, b_off)

    # Filtra sombras/frestas escuras e reflexos estourados preservando carros/roupas pretas ou brancas
    dark_ratio = float(np.mean(v_vals < 42))
    bright_ratio = float(np.mean((v_vals > 246) & (chroma < 8.0)))

    mask = np.ones(len(hsv), dtype=bool)
    if dark_ratio < 0.65:
        mask &= v_vals >= 42
    if bright_ratio < 0.50:
        mask &= ~((v_vals > 246) & (chroma < 8.0))

    if suppress_skin and np.any(mask):
        ycrcb = cv2.cvtColor(roi_work, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        cr = ycrcb[:, 1]
        cb = ycrcb[:, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (l_lab > 70)
        non_skin = mask & (~skin_mask)
        if float(np.sum(weights[non_skin])) >= 0.32 * float(np.sum(weights[mask])):
            mask = non_skin

    if not np.any(mask):
        mask = np.ones(len(hsv), dtype=bool)

    mh = h_vals[mask]
    ms = s_vals[mask]
    mv = v_vals[mask]
    ml = l_lab[mask]
    mc = chroma[mask]
    mw = weights[mask]

    # Região acromática real pelo croma perceptual LAB + guarda para ruído de sombra em baixa luz
    achrom = (mc < 12.5) | ((ms < 34) & (mc < 20.0)) | ((mv < 50) & (ms < 88) & (mc < 17.5))
    silver_name = "Prata" if is_vehicle else "Cinza Claro"

    scores: dict[str, float] = {}

    def _add(label: str, cond: np.ndarray) -> None:
        if np.any(cond):
            scores[label] = scores.get(label, 0.0) + float(np.sum(mw[cond]))

    _add("Preto", achrom & (ml < 66))
    _add("Cinza / Grafite", achrom & (ml >= 66) & (ml < 134))
    _add(silver_name, achrom & (ml >= 134) & (ml < 194))
    _add("Branco", achrom & (ml >= 194))

    chrom = ~achrom
    if np.any(chrom):
        red_zone = chrom & ((mh < 10) | (mh >= 166))
        dark_red = red_zone & (ml < 92) & (mc < 52)
        _add("Vinho / Roxo", dark_red)
        _add("Vermelho", red_zone & (~dark_red))

        orange_zone = chrom & (mh >= 10) & (mh < 23)
        if not is_vehicle:
            beige = orange_zone & (ml > 158) & (mc < 32)
            _add("Bege / Creme", beige)
            _add("Laranja / Marrom", orange_zone & (~beige))
        else:
            _add("Laranja / Marrom", orange_zone)

        _add("Amarelo / Dourado", chrom & (mh >= 23) & (mh < 36))
        _add("Verde", chrom & (mh >= 36) & (mh < 84))
        _add("Azul", chrom & (mh >= 84) & (mh < 133))

        purple_zone = chrom & (mh >= 133) & (mh < 166)
        if not is_vehicle:
            pink = purple_zone & (ml > 138)
            _add("Rosa / Roxo", pink)
            _add("Vinho / Roxo", purple_zone & (~pink))
        else:
            _add("Vinho / Roxo", purple_zone)

    if not scores:
        return "Indefinida"
    return max(scores.items(), key=lambda item: item[1])[0]


def estimate_vehicle_color(
    vehicle_crop: np.ndarray,
    vehicle_type: str = "Carro",
    *,
    allow_ir: bool | None = None,
) -> str:
    """Estima a cor predominante da lataria/tanque ignorando vidros escuros, sombras e reflexos."""
    if vehicle_crop is None or vehicle_crop.size == 0 or min(vehicle_crop.shape[:2]) < 24:
        return "Indefinida"

    h, w = vehicle_crop.shape[:2]
    if vehicle_type == "Moto":
        # Foca na região central do tanque/carenagem, evitando pneu inferior e capacete superior
        roi = vehicle_crop[int(h * 0.34) : int(h * 0.70), int(w * 0.25) : int(w * 0.75)]
        if roi.size == 0 or min(roi.shape[:2]) < 12:
            roi = vehicle_crop[int(h * 0.25) : int(h * 0.76), int(w * 0.18) : int(w * 0.82)]
    else:
        # Carro/Utilitário: faixa abaixo do para-brisa até acima da sombra dos pneus
        roi = vehicle_crop[int(h * 0.40) : int(h * 0.78), int(w * 0.18) : int(w * 0.82)]

    return _classify_roi_color(
        roi,
        is_vehicle=True,
        suppress_skin=False,
        apply_center_weight=True,
        allow_ir=allow_ir,
    )


def estimate_clothing_colors(
    person_crop: np.ndarray,
    *,
    allow_ir: bool = False,
) -> tuple[str, str, str]:
    """Extrai cor predominante da vestimenta superior e inferior de uma pessoa rastreada."""
    if person_crop is None or person_crop.size == 0:
        return ("Indefinida", "Indefinida", "Analisando...")

    h, w = person_crop.shape[:2]
    if h < 40 or w < 20:
        return ("Indefinida", "Indefinida", "Analisando...")

    # Torso central (20%..50% da altura) evitando cabeça/cabelo acima e braços nas bordas
    upper_roi = person_crop[int(h * 0.20) : int(h * 0.50), int(w * 0.24) : int(w * 0.76)]
    upper_color = _classify_roi_color(
        upper_roi,
        is_vehicle=False,
        suppress_skin=True,
        apply_center_weight=True,
        allow_ir=allow_ir,
    )

    # Calça/bermuda/saia (54%..85% da altura) apenas quando o enquadramento mostra corpo inferior
    aspect = h / float(max(1, w))
    if h >= 60 and aspect >= 1.35:
        lower_roi = person_crop[int(h * 0.54) : int(h * 0.85), int(w * 0.24) : int(w * 0.76)]
        lower_color = _classify_roi_color(
            lower_roi,
            is_vehicle=False,
            suppress_skin=True,
            apply_center_weight=True,
            allow_ir=allow_ir,
        )
    else:
        lower_color = "Não visível"

    return (upper_color, lower_color, format_clothing_colors(upper_color, lower_color))


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Calcula a interseção sobre união (IoU) entre duas caixas `(x1, y1, x2, y2)`."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / float(area_a + area_b - inter)


@dataclass
class _TrackState:
    track_id: int
    category: str
    bbox: tuple[int, int, int, int]
    last_seen: float
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0


class ClassAwareIoUTracker:
    """Rastreador multiobjeto com predição linear de velocidade isolado por classe."""

    def __init__(self, iou_threshold: float = 0.25, max_age_sec: float = 2.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_sec = max_age_sec
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    @staticmethod
    def _predict_bbox(tr: _TrackState, ts: float) -> tuple[int, int, int, int]:
        dt = min(1.5, max(0.0, ts - tr.last_seen))
        dx = int(round(tr.vx * dt))
        dy = int(round(tr.vy * dt))
        return (tr.bbox[0] + dx, tr.bbox[1] + dy, tr.bbox[2] + dx, tr.bbox[3] + dy)

    def update(
        self,
        detections: Sequence[tuple[str, tuple[int, int, int, int], float]],
        now: float | None = None,
    ) -> list[tuple[int, str, tuple[int, int, int, int], float]]:
        """Associa detecções atuais `(categoria, bbox, conf)` a `track_id`s persistentes."""
        ts = time.monotonic() if now is None else now
        expired = [tid for tid, tr in self._tracks.items() if ts - tr.last_seen > self.max_age_sec]
        for tid in expired:
            del self._tracks[tid]

        matched_tracks: set[int] = set()
        results: list[tuple[int, str, tuple[int, int, int, int], float]] = []

        # Associa usando maior IoU entre caixa observada e caixa predita pela velocidade do objeto
        candidate_pairs: list[tuple[float, int, int]] = []
        for d_idx, (cat, bbox, _conf) in enumerate(detections):
            dcx = (bbox[0] + bbox[2]) * 0.5
            dcy = (bbox[1] + bbox[3]) * 0.5
            for tid, tr in self._tracks.items():
                if tr.category != cat:
                    continue
                pred_box = self._predict_bbox(tr, ts)
                iou = max(compute_iou(bbox, tr.bbox), compute_iou(bbox, pred_box))
                score = iou
                if iou < self.iou_threshold:
                    pcx = (pred_box[0] + pred_box[2]) * 0.5
                    pcy = (pred_box[1] + pred_box[3]) * 0.5
                    diag = float(np.hypot(max(1, pred_box[2] - pred_box[0]), max(1, pred_box[3] - pred_box[1])))
                    dist_norm = float(np.hypot(dcx - pcx, dcy - pcy)) / max(1.0, diag)
                    if iou >= 0.12 and dist_norm <= 0.40:
                        score = max(iou, self.iou_threshold + (0.40 - dist_norm) * 0.1)
                if score >= self.iou_threshold:
                    candidate_pairs.append((score, d_idx, tid))
        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        det_to_track: dict[int, int] = {}
        for _score, d_idx, tid in candidate_pairs:
            if d_idx in det_to_track or tid in matched_tracks:
                continue
            det_to_track[d_idx] = tid
            matched_tracks.add(tid)

        for d_idx, (cat, bbox, conf) in enumerate(detections):
            if d_idx in det_to_track:
                tid = det_to_track[d_idx]
                tr = self._tracks[tid]
                old = tr.bbox
                dt = max(0.05, ts - tr.last_seen)
                old_cx = (old[0] + old[2]) * 0.5
                old_cy = (old[1] + old[3]) * 0.5
                new_cx = (bbox[0] + bbox[2]) * 0.5
                new_cy = (bbox[1] + bbox[3]) * 0.5
                inst_vx = (new_cx - old_cx) / dt
                inst_vy = (new_cy - old_cy) / dt
                tr.vx = tr.vx * 0.55 + inst_vx * 0.45
                tr.vy = tr.vy * 0.55 + inst_vy * 0.45

                smoothed = (
                    int(old[0] * 0.35 + bbox[0] * 0.65),
                    int(old[1] * 0.35 + bbox[1] * 0.65),
                    int(old[2] * 0.35 + bbox[2] * 0.65),
                    int(old[3] * 0.35 + bbox[3] * 0.65),
                )
                tr.bbox = smoothed
                tr.last_seen = ts
                tr.hits += 1
                results.append((tid, cat, smoothed, conf))
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _TrackState(track_id=tid, category=cat, bbox=bbox, last_seen=ts)
                results.append((tid, cat, bbox, conf))

        return results


def letterbox_bgr(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona com preenchimento proporcional para entrada de detectores YOLO."""
    h, w = image.shape[:2]
    scale = min(size / float(h), size / float(w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[dh : dh + nh, dw : dw + nw] = resized
    return canvas, scale, dw, dh


class YOLOv10ONNXDetector:
    """Detector de Pessoas, Animais e Veículos via YOLOv10s ONNX Runtime."""

    def __init__(self, onnx_path: Path | str, conf_threshold: float = 0.38) -> None:
        self.conf_threshold = conf_threshold
        self.sess, self.provider = create_ort_session(onnx_path)
        self._input_name = self.sess.get_inputs()[0].name  # type: ignore[attr-defined]

    def detect(
        self,
        frame: np.ndarray,
        detect_people: bool = True,
        detect_animals: bool = True,
        detect_vehicles: bool = True,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        h, w = frame.shape[:2]
        canvas, scale, dw, dh = letterbox_bgr(frame, 640)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        raw = self.sess.run(None, {self._input_name: blob})[0][0]  # type: ignore[attr-defined]
        rows = raw[raw[:, 4] >= self.conf_threshold]

        detections: list[tuple[str, tuple[int, int, int, int], float]] = []
        for row in rows:
            conf = float(row[4])
            cls_id = int(row[5])
            if cls_id == COCO_PERSON_ID:
                if not detect_people:
                    continue
                cat = "Pessoa"
            elif cls_id == COCO_CAR_ID or (cls_id in (COCO_TRUCK_ID, COCO_BUS_ID) and conf >= max(self.conf_threshold, 0.44)):
                if not detect_vehicles:
                    continue
                cat = "Carro"
            elif cls_id == COCO_MOTORCYCLE_ID:
                if not detect_vehicles:
                    continue
                cat = "Moto"
            elif cls_id == COCO_BICYCLE_ID:
                if not detect_vehicles:
                    continue
                cat = "Bicicleta"
            elif cls_id in COCO_ANIMAL_LABELS:
                if not detect_animals:
                    continue
                cat = f"Animal ({COCO_ANIMAL_LABELS[cls_id]})"
            else:
                continue

            x1 = max(0, min(w - 1, int((float(row[0]) - dw) / scale)))
            y1 = max(0, min(h - 1, int((float(row[1]) - dh) / scale)))
            x2 = max(x1 + 1, min(w, int((float(row[2]) - dw) / scale)))
            y2 = max(y1 + 1, min(h, int((float(row[3]) - dh) / scale)))
            detections.append((cat, (x1, y1, x2, y2), conf))
        return detections


class MobileCLIPVisualClassifier:
    """Classificador visual via embeddings MobileCLIP-S2 ONNX (Modelo de Veículo e Fallback Gênero)."""

    def __init__(
        self,
        vision_onnx_path: Path | str,
        prompts_npz_path: Path | str = DEFAULT_PROMPTS_NPZ,
    ) -> None:
        self.sess, self.provider = create_ort_session(vision_onnx_path)
        self._input_name = self.sess.get_inputs()[0].name  # type: ignore[attr-defined]
        data = np.load(str(prompts_npz_path), allow_pickle=True)
        self.car_labels = [str(x) for x in data["car_labels"]]
        self.car_embs = np.asarray(data["car_embs"], dtype=np.float32)
        self.moto_labels = [str(x) for x in data["moto_labels"]]
        self.moto_embs = np.asarray(data["moto_embs"], dtype=np.float32)
        self.gender_labels = [str(x) for x in data["gender_labels"]]
        self.gender_embs = np.asarray(data["gender_embs"], dtype=np.float32)

    def encode_image(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Codifica o recorte preservando a proporção real do objeto via letterbox 256x256."""
        padded, _scale, _dw, _dh = letterbox_bgr(bgr_crop, size=256)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        emb = self.sess.run(None, {self._input_name: blob})[0][0]  # type: ignore[attr-defined]
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-8)

    def encode_image_multiview(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Combina recorte original + espelhado (TTA leve) para estabilizar classificação de ângulo."""
        vec = self.encode_image(bgr_crop)
        if min(bgr_crop.shape[:2]) >= 56:
            vec_flip = self.encode_image(cv2.flip(bgr_crop, 1))
            avg = 0.65 * vec + 0.35 * vec_flip
            norm = float(np.linalg.norm(avg))
            return avg / max(norm, 1e-8)
        return vec

    def predict_vehicle_model(self, vehicle_crop: np.ndarray, vehicle_type: str) -> tuple[str, float]:
        if vehicle_crop.size == 0 or min(vehicle_crop.shape[:2]) < 32:
            return ("Analisando...", 0.0)
        labels = self.moto_labels if vehicle_type == "Moto" else self.car_labels
        embs = self.moto_embs if vehicle_type == "Moto" else self.car_embs
        vec = self.encode_image_multiview(vehicle_crop)
        sims = embs @ vec
        scaled = (sims - float(np.max(sims))) * 30.0
        probs = np.exp(scaled) / float(np.sum(np.exp(scaled)))
        best = int(np.argmax(sims))
        return (labels[best], float(probs[best]))

    def predict_person_gender_fallback(self, person_crop: np.ndarray) -> tuple[str, float]:
        if person_crop.size == 0 or person_crop.shape[0] < 64 or person_crop.shape[1] < 28:
            return ("Indeterminado", 0.0)
        vec = self.encode_image(person_crop)
        sims = self.gender_embs @ vec
        margin = abs(float(sims[0]) - float(sims[1]))
        if margin < 0.020:
            return ("Indeterminado", 0.0)
        best = int(np.argmax(sims))
        return (self.gender_labels[best], min(0.88, 0.56 + margin * 4.5))


class FaceGenderClassifier:
    """Classifica Homem / Mulher priorizando rosto detectado (YuNet + FairFace) e fallback corporal."""

    def __init__(
        self,
        yunet_path: Path | str,
        fairface_onnx_path: Path | str,
        clip_fallback: MobileCLIPVisualClassifier | None = None,
    ) -> None:
        self._yunet_lock = threading.Lock()
        yunet_bytes = Path(yunet_path).read_bytes()
        # Carrega via buffer binário para suportar caminhos Windows com acentuação (ex.: "segurança")
        self.detector = cv2.FaceDetectorYN.create("onnx", yunet_bytes, b"", (320, 320), 0.52, 0.3, 5)
        self.ff_sess, self.provider = create_ort_session(fairface_onnx_path)
        self._ff_input = self.ff_sess.get_inputs()[0].name  # type: ignore[attr-defined]
        self.clip_fallback = clip_fallback

    def predict(self, person_crop: np.ndarray) -> tuple[str, float]:
        if person_crop.size == 0 or person_crop.shape[0] < 36 or person_crop.shape[1] < 24:
            return ("Indeterminado", 0.0)

        h, w = person_crop.shape[:2]
        upper = person_crop[: max(24, int(h * 0.58)), :]
        uh, uw = upper.shape[:2]

        with self._yunet_lock:
            self.detector.setInputSize((uw, uh))
            _ok, faces = self.detector.detect(upper)

        if faces is not None and len(faces) > 0:
            best_face = max(faces, key=lambda f: float(f[14]))
            fx, fy, fw, fh = map(int, best_face[:4])
            fscore = float(best_face[14])
            if fscore >= 0.52 and fw >= 20 and fh >= 20:
                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.25)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(uw, fx + fw + pad_x)
                y2 = min(uh, fy + fh + pad_y)
                face_crop = upper[y1:y2, x1:x2]
                if face_crop.size > 0:
                    rgb = cv2.resize(
                        cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                        (224, 224),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
                    blob = np.transpose(arr, (2, 0, 1))[None, ...]
                    logits = self.ff_sess.run(None, {self._ff_input: blob})[0][0]  # type: ignore[attr-defined]
                    exp_l = np.exp(logits - np.max(logits))
                    probs = exp_l / np.sum(exp_l)
                    # config.json FairFace: {"0": "Female", "1": "Male"}
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    if conf >= 0.65:
                        return ("Mulher" if idx == 0 else "Homem", conf)

        if self.clip_fallback is not None:
            return self.clip_fallback.predict_person_gender_fallback(person_crop)
        return ("Indeterminado", 0.0)


class FastALPRAdapter:
    """Adaptador ALPR de dois estágios (YOLOv9 placa + CCT OCR) com normalização Brasil/Mercosul."""

    def __init__(self, prefer_gpu: bool = True) -> None:
        from fast_alpr import ALPR

        providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
            detector_providers=providers,
            ocr_providers=providers,
        )

    @staticmethod
    def _score_conf(raw_conf: object) -> float:
        if isinstance(raw_conf, (list, tuple, np.ndarray)) and len(raw_conf) > 0:
            return float(np.mean([float(x) for x in raw_conf]))
        try:
            return float(raw_conf or 0.0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    def read_plate(self, vehicle_crop: np.ndarray) -> tuple[str, float]:
        if vehicle_crop.size == 0 or min(vehicle_crop.shape[:2]) < 40:
            return ("", 0.0)

        attempts = [vehicle_crop]
        if vehicle_crop.shape[1] < 240:
            attempts.append(cv2.resize(vehicle_crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC))

        best_plate = ""
        best_conf = 0.0
        for img in attempts:
            try:
                results = self.alpr.predict(img)
            except Exception:
                continue
            for item in results:
                if item.ocr is None or not item.ocr.text:
                    continue
                normalized = normalize_brazilian_plate(item.ocr.text)
                if not normalized:
                    continue
                conf = self._score_conf(item.ocr.confidence)
                if conf > best_conf:
                    best_plate = normalized
                    best_conf = conf
            if best_plate and best_conf >= 0.60:
                break

        return (best_plate, best_conf)


def draw_ai_overlays(frame: np.ndarray, detections: Sequence[DetectionItem]) -> np.ndarray:
    """Desenha caixas e rótulos de detecção no quadro de pré-visualização."""
    if not detections:
        return frame
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        if det.category == "Pessoa":
            color = (34, 197, 94)  # Verde
        elif det.category == "Carro":
            color = (59, 130, 246)  # Azul (RGB -> BGR abaixo)
        elif det.category == "Moto":
            color = (249, 115, 22)  # Laranja
        elif det.category == "Bicicleta":
            color = (234, 179, 8)  # Amarelo
        else:
            color = (6, 182, 212)  # Ciano para animais

        bgr = (color[2], color[1], color[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

        text = _ascii_overlay_text(det.label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        pill_y1 = max(0, y1 - th - 10)
        pill_y2 = max(th + 10, y1)
        pill_x2 = min(w, x1 + tw + 10)
        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), bgr, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 5, pill_y2 - 5),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


class CCTVAnalyticsPipeline:
    """Orquestra detecção contínua, tracking e classificação secundária assíncrona."""

    def __init__(
        self,
        detector: object | None = None,
        alpr: object | None = None,
        gender_clf: object | None = None,
        vehicle_clf: object | None = None,
        csv_path: Path | str | None = None,
        people_csv_path: Path | str | None = None,
        save_csv: bool = True,
        analysis_fps: float = 6.0,
        secondary_cooldown_sec: float = 1.2,
        on_update: Callable[[AISummary, list[VehicleRecord]], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self._detector = detector
        self._alpr = alpr
        self._gender_clf = gender_clf
        self._vehicle_clf = vehicle_clf
        self.csv_path = Path(csv_path) if csv_path else None
        if people_csv_path:
            self.people_csv_path: Path | None = Path(people_csv_path)
        elif self.csv_path is not None:
            self.people_csv_path = self.csv_path.with_name("pessoas_detectadas.csv")
        else:
            self.people_csv_path = None
        self.save_csv = bool(save_csv)
        self._min_interval = 1.0 / max(1.0, float(analysis_fps))
        self._secondary_cooldown = max(0.0, float(secondary_cooldown_sec))
        self.on_update = on_update
        self.on_log = on_log

        self.detect_people_gender = True
        self.detect_animals = True
        self.detect_vehicles_alpr = True
        self._active_camera_name = "Camera"

        self._tracker = ClassAwareIoUTracker()
        self._frame_queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        self._lock = threading.Lock()
        self._ready = False
        self._loading = False
        self._backend_name = "Não carregado"
        self._last_secondary_run: dict[int, float] = {}
        self._vehicles: dict[int, VehicleRecord] = {}
        self._history_rows: list[VehicleRecord] = []
        self._people: dict[int, PersonRecord] = {}
        self._people_history_rows: list[PersonRecord] = []
        self._last_detections: list[DetectionItem] = []
        self._summary = AISummary(backend_name=self._backend_name)
        self._last_csv_write = 0.0

        if self._detector is not None:
            self._ready = True
            self._backend_name = getattr(self._detector, "provider", "Personalizado")
            self._summary.backend_name = self._backend_name

        if self.csv_path is not None and self.csv_path.exists():
            self._load_csv_history()
        if self.people_csv_path is not None and self.people_csv_path.exists():
            self._load_people_csv_history()

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def is_loading(self) -> bool:
        with self._lock:
            return self._loading

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    def configure(
        self,
        *,
        analysis_fps: float | None = None,
        detect_people_gender: bool | None = None,
        detect_animals: bool | None = None,
        detect_vehicles_alpr: bool | None = None,
        save_csv: bool | None = None,
        csv_path: Path | str | None = None,
        people_csv_path: Path | str | None = None,
    ) -> None:
        if analysis_fps is not None:
            self._min_interval = 1.0 / max(1.0, float(analysis_fps))
        if detect_people_gender is not None:
            self.detect_people_gender = bool(detect_people_gender)
        if detect_animals is not None:
            self.detect_animals = bool(detect_animals)
        if detect_vehicles_alpr is not None:
            self.detect_vehicles_alpr = bool(detect_vehicles_alpr)
        if save_csv is not None:
            self.save_csv = bool(save_csv)
        if csv_path is not None:
            self.csv_path = Path(csv_path)
            if people_csv_path is None and self.people_csv_path is None:
                self.people_csv_path = self.csv_path.with_name("pessoas_detectadas.csv")
        if people_csv_path is not None:
            self.people_csv_path = Path(people_csv_path)

    def load_default_models(self, models_dir: Path | str = DEFAULT_MODELS_DIR) -> bool:
        """Baixa (se necessário) e inicializa os modelos ONNX DirectML/CPU em memória."""
        with self._lock:
            if self._ready:
                return True
            if self._loading:
                return False
            self._loading = True

        try:
            paths = ensure_ai_models(models_dir, on_log=self.on_log)
            self._log("[IA] Carregando detector YOLOv10s, FairFace, MobileCLIP-S2 e Fast-ALPR...")
            detector = YOLOv10ONNXDetector(paths["yolo"])
            clip_clf = MobileCLIPVisualClassifier(paths["mobileclip_vision"], paths["prompts_npz"])
            gender_clf = FaceGenderClassifier(paths["yunet"], paths["fairface"], clip_fallback=clip_clf)
            prefer_gpu = "Dml" in detector.provider
            alpr = FastALPRAdapter(prefer_gpu=prefer_gpu)

            with self._lock:
                self._detector = detector
                self._vehicle_clf = clip_clf
                self._gender_clf = gender_clf
                self._alpr = alpr
                self._backend_name = "GPU DirectML (RX 580)" if prefer_gpu else detector.provider
                self._summary.backend_name = self._backend_name
                self._ready = True
            self._log(f"[IA] Motores carregados com sucesso ({self._backend_name}).")
            return True
        except Exception as err:
            self._log(f"[IA] Falha ao carregar modelos: {err}")
            return False
        finally:
            with self._lock:
                self._loading = False

    def start(self) -> None:
        """Inicia a thread assíncrona de inferência em tempo real."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Para a thread de IA e grava CSV pendente de veículos e pessoas."""
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        if self.save_csv:
            self._write_all_csv_snapshots()
        with self._lock:
            self._last_detections = []

    def submit_frame(self, frame: np.ndarray, camera_name: str = "Camera") -> None:
        """Envia o frame mais recente descartando anteriores caso o pipeline esteja ocupado."""
        if not self.is_ready or self._stop.is_set():
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait((frame.copy(), camera_name))
        except queue.Full:
            pass

    def get_overlay_detections(self) -> list[DetectionItem]:
        with self._lock:
            return list(self._last_detections)

    def get_vehicle_records(self) -> list[VehicleRecord]:
        with self._lock:
            combined = list(self._history_rows) + list(self._vehicles.values())
            return list(reversed(combined))

    def get_person_records(self) -> list[PersonRecord]:
        with self._lock:
            combined = list(self._people_history_rows) + list(self._people.values())
            return list(reversed(combined))

    def get_summary(self) -> AISummary:
        with self._lock:
            return AISummary(
                people_total=self._summary.people_total,
                men_count=self._summary.men_count,
                women_count=self._summary.women_count,
                animals_count=self._summary.animals_count,
                cars_count=self._summary.cars_count,
                motos_count=self._summary.motos_count,
                bicycles_count=self._summary.bicycles_count,
                last_plate=self._summary.last_plate,
                backend_name=self._summary.backend_name,
            )

    def clear_vehicle_records(self) -> None:
        with self._lock:
            self._vehicles.clear()
            self._history_rows.clear()
            self._summary.last_plate = ""
        if self.save_csv and self.csv_path is not None:
            self._write_csv_snapshot()
        if self.on_update:
            self.on_update(self.get_summary(), self.get_vehicle_records())

    def clear_person_records(self) -> None:
        with self._lock:
            self._people.clear()
            self._people_history_rows.clear()
        if self.save_csv and self.people_csv_path is not None:
            self._write_people_csv_snapshot()
        if self.on_update:
            self.on_update(self.get_summary(), self.get_vehicle_records())

    def clear_all_records(self) -> None:
        with self._lock:
            self._vehicles.clear()
            self._history_rows.clear()
            self._people.clear()
            self._people_history_rows.clear()
            self._last_secondary_run.clear()
            self._summary.last_plate = ""
        if self.save_csv:
            self._write_all_csv_snapshots()
        if self.on_update:
            self.on_update(self.get_summary(), self.get_vehicle_records())

    def export_csv(self, target_path: Path | str) -> Path:
        out = Path(target_path)
        self._write_csv_file(out, self.get_vehicle_records())
        return out

    def export_people_csv(self, target_path: Path | str) -> Path:
        out = Path(target_path)
        self._write_people_csv_file(out, self.get_person_records())
        return out

    def process_frame(self, frame: np.ndarray, camera_name: str = "Camera") -> tuple[list[DetectionItem], bool]:
        """Executa detecção, tracking e classificadores secundários em um único frame BGR."""
        if self._detector is None or frame is None or frame.size == 0:
            return ([], False)

        now = time.monotonic()
        h, w = frame.shape[:2]
        frame_ir = is_ir_frame(frame)
        raw_dets = self._detector.detect(  # type: ignore[attr-defined]
            frame,
            detect_people=self.detect_people_gender,
            detect_animals=self.detect_animals,
            detect_vehicles=self.detect_vehicles_alpr,
        )
        tracked = self._tracker.update(raw_dets, now=now)

        items: list[DetectionItem] = []
        people_total = men_count = women_count = animals_count = cars_count = motos_count = bicycles_count = 0
        table_dirty = False
        immediate_csv = False

        for track_id, category, (x1, y1, x2, y2), conf in tracked:
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            crop = frame[y1:y2, x1:x2]
            quality = _crop_quality_weight((x1, y1, x2, y2), w, h)

            if category == "Pessoa":
                people_total += 1
                with self._lock:
                    if track_id not in self._people:
                        self._people[track_id] = PersonRecord(
                            track_id=track_id,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            camera_name=camera_name,
                        )
                        table_dirty = True
                        immediate_csv = True

                if self.detect_people_gender and self._can_run_secondary(track_id, now):
                    if self._update_person_attributes(track_id, crop, quality=quality, frame_ir=frame_ir):
                        table_dirty = True

                with self._lock:
                    prec = self._people[track_id]
                    if prec.person_type == "Homem":
                        men_count += 1
                    elif prec.person_type == "Mulher":
                        women_count += 1
                    parts = [f"Pessoa #{track_id}"]
                    if prec.person_type != "Indeterminado":
                        parts[0] = f"Pessoa #{track_id}: {prec.person_type}"
                    if prec.clothing_colors and prec.clothing_colors != "Analisando...":
                        parts.append(prec.clothing_colors)
                    lbl = " | ".join(parts)
                    items.append(
                        DetectionItem(
                            track_id=track_id,
                            category="Pessoa",
                            label=lbl,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            sub_label=prec.person_type,
                            upper_color=prec.upper_color,
                            lower_color=prec.lower_color,
                        )
                    )

            elif category.startswith("Animal"):
                animals_count += 1
                items.append(DetectionItem(track_id, "Animal", f"{category} #{track_id}", (x1, y1, x2, y2), conf))

            elif category == "Bicicleta":
                bicycles_count += 1
                items.append(DetectionItem(track_id, "Bicicleta", f"Bicicleta #{track_id}", (x1, y1, x2, y2), conf))

            elif category in ("Carro", "Moto"):
                if category == "Carro":
                    cars_count += 1
                else:
                    motos_count += 1

                with self._lock:
                    is_new = track_id not in self._vehicles
                    if is_new:
                        self._vehicles[track_id] = VehicleRecord(
                            track_id=track_id,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            vehicle_type=category,
                            camera_name=camera_name,
                        )
                        table_dirty = True
                        immediate_csv = True
                    prev_plate = self._vehicles[track_id].plate

                if self.detect_vehicles_alpr and self._can_run_secondary(track_id, now):
                    if self._update_vehicle_attributes(
                        track_id, crop, category, quality=quality, frame_ir=frame_ir
                    ):
                        table_dirty = True
                        with self._lock:
                            cur_plate = self._vehicles[track_id].plate
                        if cur_plate != prev_plate and cur_plate != "Lendo...":
                            immediate_csv = True

                with self._lock:
                    rec = self._vehicles[track_id]
                    parts = [f"{category} #{track_id}"]
                    if rec.plate and rec.plate != "Lendo...":
                        parts.append(rec.plate)
                    if rec.color and rec.color != "Indefinida":
                        parts.append(rec.color)
                    if rec.approx_model and rec.approx_model != "Analisando...":
                        parts.append(rec.approx_model.split(" — ")[0])
                    lbl = " | ".join(parts)
                    items.append(
                        DetectionItem(
                            track_id=track_id,
                            category=category,
                            label=lbl,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            plate=rec.plate,
                            color=rec.color,
                            approx_model=rec.approx_model,
                        )
                    )

        with self._lock:
            self._last_detections = items
            self._summary = AISummary(
                people_total=people_total,
                men_count=men_count,
                women_count=women_count,
                animals_count=animals_count,
                cars_count=cars_count,
                motos_count=motos_count,
                bicycles_count=bicycles_count,
                last_plate=self._summary.last_plate,
                backend_name=self._backend_name,
            )

        if table_dirty and self.save_csv:
            if immediate_csv or time.monotonic() - self._last_csv_write >= 2.0:
                self._write_all_csv_snapshots()

        return (items, table_dirty)

    def _run_loop(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            try:
                frame, cam_name = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.monotonic()
            if now - last_ts < self._min_interval:
                continue
            last_ts = now

            try:
                _items, _dirty = self.process_frame(frame, camera_name=cam_name)
                if self.on_update:
                    self.on_update(self.get_summary(), self.get_vehicle_records())
            except Exception as err:
                self._log(f"[IA] Erro no processamento do quadro: {err}")

    def _can_run_secondary(self, track_id: int, now: float) -> bool:
        last = self._last_secondary_run.get(track_id, -999.0)
        if now - last < self._secondary_cooldown:
            return False
        self._last_secondary_run[track_id] = now
        return True

    def _update_person_attributes(
        self,
        track_id: int,
        crop: np.ndarray,
        quality: float = 1.0,
        frame_ir: bool = False,
    ) -> bool:
        g_label = "Indeterminado"
        g_conf = 0.0
        if self._gender_clf is not None:
            g_label, g_conf = self._gender_clf.predict(crop)  # type: ignore[attr-defined]

        upper_c, lower_c, _ = estimate_clothing_colors(crop, allow_ir=frame_ir)

        changed = False
        with self._lock:
            prec = self._people[track_id]
            if g_label in ("Homem", "Mulher") and g_conf > 0.0:
                best_g, share = _accumulate_weighted_vote(prec.gender_votes, g_label, g_conf * quality)
                if best_g and (best_g != prec.person_type or share > prec.gender_conf):
                    prec.person_type = best_g
                    prec.gender_conf = max(prec.gender_conf, g_conf)
                    changed = True

            if upper_c != "Indefinida":
                u_weight = quality * (0.35 if upper_c.startswith("P&B") else 1.0)
                best_u, _ = _accumulate_weighted_vote(prec.upper_color_votes, upper_c, u_weight)
                if best_u and best_u != prec.upper_color:
                    prec.upper_color = best_u
                    changed = True

            if lower_c != "Indefinida":
                l_weight = quality * (0.25 if lower_c == "Não visível" else (0.35 if lower_c.startswith("P&B") else 1.0))
                best_l, _ = _accumulate_weighted_vote(prec.lower_color_votes, lower_c, l_weight)
                if best_l and best_l != prec.lower_color:
                    prec.lower_color = best_l
                    changed = True

            new_clothing = format_clothing_colors(prec.upper_color, prec.lower_color)
            if new_clothing != prec.clothing_colors:
                prec.clothing_colors = new_clothing
                changed = True

        return changed

    def _update_vehicle_attributes(
        self,
        track_id: int,
        crop: np.ndarray,
        vehicle_type: str,
        quality: float = 1.0,
        frame_ir: bool | None = None,
    ) -> bool:
        plate = ""
        plate_conf = 0.0
        if self._alpr is not None:
            plate, plate_conf = self._alpr.read_plate(crop)  # type: ignore[attr-defined]

        color = estimate_vehicle_color(crop, vehicle_type=vehicle_type, allow_ir=frame_ir)

        model_name = ""
        model_conf = 0.0
        if self._vehicle_clf is not None:
            model_name, model_conf = self._vehicle_clf.predict_vehicle_model(crop, vehicle_type)  # type: ignore[attr-defined]

        changed = False
        with self._lock:
            rec = self._vehicles[track_id]
            if color != "Indefinida":
                c_weight = quality * (0.25 if color.startswith("P&B") else 1.0)
                best_color, _ = _accumulate_weighted_vote(rec.color_votes, color, c_weight)
                if best_color and best_color != rec.color:
                    rec.color = best_color
                    changed = True

            if model_name and model_name != "Analisando..." and model_conf >= 0.12:
                best_model, share = _accumulate_weighted_vote(rec.model_scores, model_name, quality * model_conf)
                if best_model and (best_model != rec.approx_model or share > rec.model_conf):
                    rec.approx_model = best_model
                    rec.model_conf = max(rec.model_conf, share)
                    changed = True

            if plate:
                rec.plate_votes[plate] += 1
                best_plate, _cnt = rec.plate_votes.most_common(1)[0]
                if best_plate != rec.plate or plate_conf > rec.plate_conf:
                    rec.plate = best_plate
                    rec.plate_conf = max(rec.plate_conf, plate_conf)
                    self._summary.last_plate = best_plate
                    changed = True

        return changed

    def _load_csv_history(self) -> None:
        assert self.csv_path is not None
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                loaded: list[VehicleRecord] = []
                for idx, row in enumerate(reader, start=1):
                    ts = (row.get("Timestamp") or "").strip()
                    vtype = (row.get("Tipo de Veículo") or row.get("Tipo") or "").strip()
                    if not ts or not vtype:
                        continue
                    loaded.append(
                        VehicleRecord(
                            track_id=-idx,
                            timestamp=ts,
                            vehicle_type=vtype,
                            plate=(row.get("Placa") or "Lendo...").strip(),
                            color=(row.get("Cor") or "Indefinida").strip(),
                            approx_model=(row.get("Modelo Aproximado") or "Não identificado").strip(),
                            camera_name=(row.get("Câmera") or "").strip(),
                        )
                    )
                self._history_rows = loaded[-500:]
        except Exception:
            self._history_rows = []

    def _load_people_csv_history(self) -> None:
        assert self.people_csv_path is not None
        try:
            with self.people_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                loaded: list[PersonRecord] = []
                for idx, row in enumerate(reader, start=1):
                    ts = (row.get("Timestamp") or "").strip()
                    ptype = (row.get("Tipo (Homem/Mulher)") or row.get("Tipo") or "").strip()
                    if not ts or not ptype:
                        continue
                    upper_c = (row.get("Cor Superior") or "Indefinida").strip()
                    lower_c = (row.get("Cor Inferior") or "Indefinida").strip()
                    clothes = (row.get("Cor das Roupas") or "").strip() or format_clothing_colors(upper_c, lower_c)
                    loaded.append(
                        PersonRecord(
                            track_id=-idx,
                            timestamp=ts,
                            person_type=ptype,
                            clothing_colors=clothes,
                            upper_color=upper_c,
                            lower_color=lower_c,
                            camera_name=(row.get("Câmera") or "").strip(),
                        )
                    )
                self._people_history_rows = loaded[-500:]
        except Exception:
            self._people_history_rows = []

    def _write_all_csv_snapshots(self) -> None:
        self._last_csv_write = time.monotonic()
        if self.csv_path is not None:
            self._write_csv_snapshot()
        if self.people_csv_path is not None:
            self._write_people_csv_snapshot()

    def _write_csv_snapshot(self) -> None:
        if self.csv_path is None:
            return
        self._last_csv_write = time.monotonic()
        records_asc = list(reversed(self.get_vehicle_records()))[-1000:]
        self._write_csv_file(self.csv_path, records_asc)

    def _write_people_csv_snapshot(self) -> None:
        if self.people_csv_path is None:
            return
        self._last_csv_write = time.monotonic()
        records_asc = list(reversed(self.get_person_records()))[-1000:]
        self._write_people_csv_file(self.people_csv_path, records_asc)

    @staticmethod
    def _write_csv_rows(target: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerow(headers)
                writer.writerows(rows)
            os.replace(tmp, target)
        except OSError:
            pass

    @classmethod
    def _write_csv_file(cls, target: Path, records: Sequence[VehicleRecord]) -> None:
        cls._write_csv_rows(
            target,
            ["Timestamp", "Tipo de Veículo", "Placa", "Cor", "Modelo Aproximado", "Câmera"],
            [
                [
                    rec.timestamp,
                    rec.vehicle_type,
                    rec.plate,
                    rec.color,
                    rec.approx_model,
                    rec.camera_name,
                ]
                for rec in records
            ],
        )

    @classmethod
    def _write_people_csv_file(cls, target: Path, records: Sequence[PersonRecord]) -> None:
        cls._write_csv_rows(
            target,
            ["Timestamp", "Tipo (Homem/Mulher)", "Cor das Roupas", "Cor Superior", "Cor Inferior", "Câmera"],
            [
                [
                    rec.timestamp,
                    rec.person_type,
                    rec.clothing_colors,
                    rec.upper_color,
                    rec.lower_color,
                    rec.camera_name,
                ]
                for rec in records
            ],
        )
