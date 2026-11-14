"""Pipeline de Restauração de Textura, Redução de Ruído e Super-Resolução (AI Upscaling).

Projetado para operar em tempo real (ROIs / preview / ALPR) ou pós-processamento
de clipes gravados, utilizando OpenCV e ONNX Runtime (DirectML / CUDA / CPU)
com inferência em blocos (tiling) para evitar estourar VRAM em GPUs de 4–8 GB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import threading

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


def restore_texture_and_denoise(
    frame_bgr: np.ndarray,
    *,
    denoise_strength: float = 0.45,
    sharpen_amount: float = 0.45,
    clahe_clip: float = 1.6,
) -> np.ndarray:
    """Remove ruído de sensor/compressão preservando micro-texturas (sem aspecto 'lavado').

    Separa luminância (Y) e crominância (Cr/Cb): limpa ruído colorido na croma,
    decompõe a luminância em camada base + detalhe de alta frequência com coring
    suave (remove grão mantendo fibras/bordas/caracteres) e aplica micro-contraste.
    """
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return frame_bgr

    strength = max(0.0, min(1.0, float(denoise_strength)))
    sharp = max(0.0, min(1.5, float(sharpen_amount)))

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

    # 1) Limpeza suave de ruído cromático (bloqueio H.264/HEVC e baixa luz)
    if strength > 0.05:
        k_chroma = 3 if strength < 0.6 else 5
        cr_ch = cv2.GaussianBlur(cr_ch, (k_chroma, k_chroma), 0.8 + strength)
        cb_ch = cv2.GaussianBlur(cb_ch, (k_chroma, k_chroma), 0.8 + strength)

    # 2) Separação frequência base vs. micro-textura na luminância
    y_f = y_ch.astype(np.float32)
    if strength > 0.0:
        sigma_color = 10.0 + 26.0 * strength
        base_y = cv2.bilateralFilter(y_ch, d=5, sigmaColor=sigma_color, sigmaSpace=4.5).astype(np.float32)
        detail = y_f - base_y
        # Coring suave: atenua grão fino (< noise_floor), preserva e realça bordas/texturas reais
        noise_floor = 2.0 + 5.5 * strength
        abs_d = np.abs(detail)
        keep_ratio = np.clip((abs_d - noise_floor) / max(1.0, noise_floor * 1.5), 0.0, 1.0)
        detail_gain = 1.0 + 0.55 * sharp
        y_clean = base_y + detail * keep_ratio * detail_gain
    else:
        y_clean = y_f

    y_u8 = np.clip(y_clean, 0.0, 255.0).astype(np.uint8)

    # 3) Micro-contraste local amortecido (evita lavar sombras/tecidos escuros)
    if clahe_clip > 0.1:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        y_clahe = clahe.apply(y_u8)
        y_u8 = cv2.addWeighted(y_u8, 0.72, y_clahe, 0.28, 0.0)

    restored = cv2.cvtColor(cv2.merge([y_u8, cr_ch, cb_ch]), cv2.COLOR_YCrCb2BGR)

    # 4) Nitidez final controlada em bordas
    if sharp > 0.05:
        blur = cv2.GaussianBlur(restored, (0, 0), sigmaX=1.0)
        restored = cv2.addWeighted(restored, 1.0 + 0.38 * sharp, blur, -0.38 * sharp, 0.0)

    return restored


def _select_onnx_providers() -> list[str]:
    if ort is None:
        return []
    avail = set(ort.get_available_providers())
    ordered = [
        p
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        if p in avail
    ]
    return ordered or ["CPUExecutionProvider"]


class SuperResolutionEnhancer:
    """Motor híbrido de Super-Resolução ONNX (Real-ESRGAN/SwinIR) + fallback analítico rápido."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        scale: int = 2,
        tile_size: int = 256,
        tile_pad: int = 16,
        denoise_strength: float = 0.45,
        sharpen_amount: float = 0.45,
    ) -> None:
        self.scale = max(1, min(4, int(scale)))
        self.tile_size = max(64, int(tile_size))
        self.tile_pad = max(0, int(tile_pad))
        self.denoise_strength = float(denoise_strength)
        self.sharpen_amount = float(sharpen_amount)

        self._lock = threading.Lock()
        self._session = None
        self._input_name = ""
        self._backend_label = "OpenCV Analítico (Lanczos4 + Textura)"
        self.model_path = Path(model_path) if model_path else None

        if self.model_path and self.model_path.exists() and ort is not None:
            self._load_onnx_model(self.model_path)

    @property
    def backend_name(self) -> str:
        return self._backend_label

    @property
    def has_ai_model(self) -> bool:
        return self._session is not None

    def _load_onnx_model(self, path: Path) -> None:
        if ort is None:
            return
        try:
            providers = _select_onnx_providers()
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(path), sess_options=sess_opts, providers=providers)
            self._session = sess
            self._input_name = sess.get_inputs()[0].name
            active = sess.get_providers()[0] if sess.get_providers() else "CPU"
            self._backend_label = f"ONNX SR ({active})"
        except Exception:
            self._session = None
            self._backend_label = "OpenCV Analítico (Lanczos4 + Textura)"

    def _infer_tile_onnx(self, tile_bgr: np.ndarray) -> np.ndarray:
        assert self._session is not None
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None, ...]  # NCHW
        out = self._session.run(None, {self._input_name: inp})[0]
        out_img = np.transpose(np.clip(out[0], 0.0, 1.0), (1, 2, 0))
        out_u8 = (out_img * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

    def _upscale_tiled_onnx(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        # Detecta fator real de escala a partir de um bloco pequeno se necessário
        sample = self._infer_tile_onnx(frame_bgr[: min(h, 64), : min(w, 64)])
        actual_scale = max(1, round(sample.shape[0] / float(min(h, 64))))
        out_h, out_w = h * actual_scale, w * actual_scale
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        ts = self.tile_size
        pad = self.tile_pad
        for y in range(0, h, ts):
            for x in range(0, w, ts):
                x1, y1 = x, y
                x2, y2 = min(w, x + ts), min(h, y + ts)
                px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                px2, py2 = min(w, x2 + pad), min(h, y2 + pad)

                tile = frame_bgr[py1:py2, px1:px2]
                sr_tile = self._infer_tile_onnx(tile)

                ox1 = (x1 - px1) * actual_scale
                oy1 = (y1 - py1) * actual_scale
                ox2 = ox1 + (x2 - x1) * actual_scale
                oy2 = oy1 + (y2 - y1) * actual_scale

                output[y1 * actual_scale : y2 * actual_scale, x1 * actual_scale : x2 * actual_scale] = sr_tile[
                    oy1:oy2, ox1:ox2
                ]
        return output

    def _upscale_analytic(self, frame_bgr: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        up = cv2.resize(frame_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        return restore_texture_and_denoise(
            up,
            denoise_strength=max(0.15, self.denoise_strength * 0.5),
            sharpen_amount=min(1.2, self.sharpen_amount * 1.25),
            clahe_clip=1.2,
        )

    def enhance_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        upscale: bool = False,
        scale: int | None = None,
    ) -> np.ndarray:
        """Aplica restauração de textura em tempo real e upscaling opcional."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        cleaned = restore_texture_and_denoise(
            frame_bgr,
            denoise_strength=self.denoise_strength,
            sharpen_amount=self.sharpen_amount,
        )
        if not upscale:
            return cleaned

        target_scale = max(1, min(4, int(scale or self.scale)))
        if target_scale == 1:
            return cleaned

        if self._session is not None:
            with self._lock:
                sr = self._upscale_tiled_onnx(cleaned)
            if sr.shape[0] != cleaned.shape[0] * target_scale:
                sr = cv2.resize(
                    sr,
                    (cleaned.shape[1] * target_scale, cleaned.shape[0] * target_scale),
                    interpolation=cv2.INTER_AREA if sr.shape[0] > cleaned.shape[0] * target_scale else cv2.INTER_LANCZOS4,
                )
            return sr

        return self._upscale_analytic(cleaned, target_scale)

    def enhance_roi(self, crop_bgr: np.ndarray, *, scale: int = 2) -> np.ndarray:
        """Realça recortes pequenos (placas ALPR, rostos, tecidos) com super-resolução."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        return self.enhance_frame(crop_bgr, upscale=True, scale=scale)


def enhance_video_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    enhancer: SuperResolutionEnhancer | None = None,
    upscale: bool = False,
    scale: int = 2,
    max_frames: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Pós-processa um vídeo gravado (.mp4), aplicando restauração de textura e/ou SR."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Vídeo de origem não encontrado: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src}")

    eng = enhancer or SuperResolutionEnhancer(scale=scale)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if fps <= 0.1 or fps > 120.0:
        fps = 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    processed = 0
    out_w = 0
    out_h = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out_frame = eng.enhance_frame(frame, upscale=upscale, scale=scale)
            if writer is None:
                out_h, out_w = out_frame.shape[:2]
                dst.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))
                if not writer.isOpened():
                    raise RuntimeError(f"Falha ao criar arquivo de saída: {dst}")
            writer.write(out_frame)
            processed += 1
            if progress_cb is not None:
                progress_cb(processed, total_frames)
            if max_frames is not None and processed >= int(max_frames):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        "input": str(src),
        "output": str(dst),
        "frames": processed,
        "width": out_w,
        "height": out_h,
        "fps": fps,
        "backend": eng.backend_name,
    }
