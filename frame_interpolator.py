"""Pipeline de Interpolação de Quadros (Frame Interpolation) para >= 30 FPS constantes.

Suporta três modos integrados:
1. RIFE ONNX (GPU DirectML/CUDA/TensorRT ou CPU) com detecção de corte de cena;
2. Fluxo Óptico Bidirecional (OpenCV DIS + remap) sem depender de modelo externo;
3. FFmpeg nativo (`minterpolate` com `aobmc` + `bidir` + `scd`) preservando áudio 100% sincronizado.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


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


def _is_scene_cut(frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 0.42) -> bool:
    """Detecta corte brusco de cena via correlação de histograma HSV para evitar ghosting."""
    ha = cv2.calcHist([cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return corr < (1.0 - threshold)


def _warp_with_flow(img: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[:, :, 0] * float(scale)
    map_y = grid_y + flow[:, :, 1] * float(scale)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def optical_flow_interpolate_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    t: float = 0.5,
    *,
    scene_cut_threshold: float = 0.42,
) -> np.ndarray:
    """Gera quadro intermediário em `t in (0, 1)` usando fluxo óptico bidirecional (DIS)."""
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.02:
        return frame_a.copy()
    if t >= 0.98:
        return frame_b.copy()
    if _is_scene_cut(frame_a, frame_b, threshold=scene_cut_threshold):
        return frame_a.copy() if t < 0.5 else frame_b.copy()

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_ab = dis.calc(gray_a, gray_b, None)
    flow_ba = dis.calc(gray_b, gray_a, None)

    # Remapeamento bidirecional para t (backward warp aproximado via fluxo inverso)
    warped_a = _warp_with_flow(frame_a, flow_ba, scale=t)
    warped_b = _warp_with_flow(frame_b, flow_ab, scale=(1.0 - t))
    return cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0.0)


class RIFEInterpolator:
    """Interpolador neural RIFE via ONNX Runtime com fallback para Fluxo Óptico DIS."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._session = None
        self._input_names: list[str] = []
        self._backend_label = "OpenCV DIS Optical Flow"
        self.model_path = Path(model_path) if model_path else None

        if self.model_path and self.model_path.exists() and ort is not None:
            try:
                providers = _select_onnx_providers()
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
                self._session = sess
                self._input_names = [inp.name for inp in sess.get_inputs()]
                active = sess.get_providers()[0] if sess.get_providers() else "CPU"
                self._backend_label = f"RIFE ONNX ({active})"
            except Exception:
                self._session = None

    @property
    def backend_name(self) -> str:
        return self._backend_label

    def interpolate_pair(self, frame_a: np.ndarray, frame_b: np.ndarray, t: float = 0.5) -> np.ndarray:
        if _is_scene_cut(frame_a, frame_b):
            return frame_a.copy() if t < 0.5 else frame_b.copy()

        if self._session is None:
            return optical_flow_interpolate_pair(frame_a, frame_b, t=t)

        # Pad múltiplo de 32 exigido pelo RIFE
        h, w = frame_a.shape[:2]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32

        def _prep(img: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            padded = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
            return np.transpose(padded, (2, 0, 1))[None, ...]

        img0 = _prep(frame_a)
        img1 = _prep(frame_b)
        if len(self._input_names) == 1:
            feed = {self._input_names[0]: np.concatenate([img0, img1], axis=1)}
        elif len(self._input_names) == 2:
            feed = {self._input_names[0]: img0, self._input_names[1]: img1}
        else:
            t_arr = np.array([[float(t)]], dtype=np.float32)
            feed = {self._input_names[0]: img0, self._input_names[1]: img1, self._input_names[2]: t_arr}

        out = self._session.run(None, feed)[0]
        pred = np.transpose(np.clip(out[0, :, :h, :w], 0.0, 1.0), (1, 2, 0))
        return cv2.cvtColor((pred * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


class LiveFrameInterpolator:
    """Interpolador de fluxo óptico em proxy reduzido com cache de grade para streaming >= 30 FPS."""

    def __init__(self, max_flow_width: int = 400, scene_cut_threshold: float = 0.42) -> None:
        self.max_flow_width = max(128, int(max_flow_width))
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_grid(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._grid_cache.get((h, w))
        if cached is None:
            cached = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_cache[(h, w)] = cached
        return cached

    def interpolate_intermediate_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        subframes: int = 2,
    ) -> list[np.ndarray]:
        """Retorna `subframes` quadros intermediários entre `frame_a` e `frame_b`."""
        n = max(0, min(4, int(subframes)))
        if n == 0 or frame_a is None or frame_b is None or frame_a.shape != frame_b.shape:
            return []
        if _is_scene_cut(frame_a, frame_b, threshold=self.scene_cut_threshold):
            return []

        h, w = frame_a.shape[:2]
        if w > self.max_flow_width:
            scale_x = float(w) / float(self.max_flow_width)
            fh = max(32, int(round(h / scale_x)))
            fw = self.max_flow_width
            scale_y = float(h) / float(fh)
            ga = cv2.cvtColor(cv2.resize(frame_a, (fw, fh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(cv2.resize(frame_b, (fw, fh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            flow_ab_s = self._dis.calc(ga, gb, None)
            flow_ba_s = self._dis.calc(gb, ga, None)
            flow_ab = cv2.resize(flow_ab_s, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_ba = cv2.resize(flow_ba_s, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_ab[:, :, 0] *= scale_x
            flow_ab[:, :, 1] *= scale_y
            flow_ba[:, :, 0] *= scale_x
            flow_ba[:, :, 1] *= scale_y
        else:
            ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            flow_ab = self._dis.calc(ga, gb, None)
            flow_ba = self._dis.calc(gb, ga, None)

        grid_x, grid_y = self._get_grid(h, w)
        results: list[np.ndarray] = []
        for idx in range(1, n + 1):
            t = idx / float(n + 1)
            warped_a = cv2.remap(
                frame_a,
                grid_x + flow_ba[:, :, 0] * t,
                grid_y + flow_ba[:, :, 1] * t,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_b = cv2.remap(
                frame_b,
                grid_x + flow_ab[:, :, 0] * (1.0 - t),
                grid_y + flow_ab[:, :, 1] * (1.0 - t),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            results.append(cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0.0))
        return results


def build_ffmpeg_minterpolate_cmd(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: int = 30,
    crf: int = 17,
    preset: str = "medium",
    enhance: bool = False,
    ffmpeg_exe: str | None = None,
) -> list[str]:
    """Monta comando FFmpeg com `minterpolate` (AOBMC + bidir + detecção de corte) e áudio sincronizado."""
    exe = ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
    fps = max(30, int(target_fps))
    interp_filter = (
        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
        f"me_mode=bidir:me=epzs:vsbmc=1:scd=fdiff:scd_threshold=10.0"
    )
    vf = f"hqdn3d=1.2:1.2:3.0:3.0,unsharp=5:5:0.75:3:3:0.0,{interp_filter}" if enhance else interp_filter
    return [
        exe,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def interpolate_video_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: int = 30,
    crf: int = 17,
    preset: str = "medium",
    enhance: bool = False,
) -> dict[str, object]:
    """Executa interpolação de quadros via FFmpeg preservando duração e sincronia de áudio."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fps = max(30, int(target_fps))

    cmd = build_ffmpeg_minterpolate_cmd(
        src, dst, target_fps=fps, crf=crf, preset=preset, enhance=enhance
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg minterpolate falhou ({proc.returncode}): {proc.stderr[-500:]}")

    return {"input": str(src), "output": str(dst), "target_fps": fps, "backend": "FFmpeg minterpolate (AOBMC/bidir)"}


def interpolate_video_python(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: float = 30.0,
    model_path: str | Path | None = None,
    crf: int = 17,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Eleva vídeo para `>= 30 FPS` via RIFE ONNX / DIS Optical Flow e remuxa áudio original via FFmpeg."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if src_fps <= 0.1 or src_fps > 240.0:
        src_fps = 15.0
    out_fps = max(30.0, float(target_fps), src_fps)

    frames: list[np.ndarray] = []
    while True:
        ok, frm = cap.read()
        if not ok or frm is None:
            break
        frames.append(frm)
    cap.release()

    if not frames:
        raise RuntimeError("O vídeo de entrada não contém quadros legíveis.")

    h, w = frames[0].shape[:2]
    duration_sec = len(frames) / src_fps
    out_count = max(1, int(round(duration_sec * out_fps)))
    interp = RIFEInterpolator(model_path=model_path)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp:
        raw_video = Path(tmp) / "video_only.mp4"
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Falha ao criar vídeo temporário: {raw_video}")

        try:
            for k in range(out_count):
                t_sec = k / out_fps
                src_pos = min(len(frames) - 1.0, t_sec * src_fps)
                idx0 = int(src_pos)
                idx1 = min(len(frames) - 1, idx0 + 1)
                alpha = src_pos - idx0
                if idx0 == idx1 or alpha <= 0.02:
                    out_frm = frames[idx0]
                elif alpha >= 0.98:
                    out_frm = frames[idx1]
                else:
                    out_frm = interp.interpolate_pair(frames[idx0], frames[idx1], t=alpha)
                writer.write(out_frm)
                if progress_cb is not None:
                    progress_cb(k + 1, out_count)
        finally:
            writer.release()

        # Re-codifica em H.264 de alta qualidade e remuxa a faixa de áudio original 100% sincronizada
        mux_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(raw_video),
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        proc = subprocess.run(mux_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Falha ao finalizar MP4/áudio ({proc.returncode}): {proc.stderr[-500:]}")

    return {
        "input": str(src),
        "output": str(dst),
        "src_fps": src_fps,
        "target_fps": out_fps,
        "src_frames": len(frames),
        "out_frames": out_count,
        "backend": interp.backend_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interpola vídeo para >= 30 FPS constantes preservando áudio.")
    parser.add_argument("input", help="Arquivo de vídeo de entrada")
    parser.add_argument("output", help="Arquivo MP4 de saída")
    parser.add_argument("--fps", type=int, default=30, help="FPS de saída mínimo (>= 30)")
    parser.add_argument("--method", choices=("ffmpeg", "python"), default="ffmpeg", help="Motor de interpolação")
    parser.add_argument("--model", default="", help="Caminho opcional para modelo RIFE .onnx")
    parser.add_argument("--crf", type=int, default=17, help="Qualidade x264 CRF (menor = melhor)")
    args = parser.parse_args(argv)

    if args.method == "ffmpeg":
        info = interpolate_video_ffmpeg(args.input, args.output, target_fps=args.fps, crf=args.crf)
    else:
        info = interpolate_video_python(
            args.input, args.output, target_fps=args.fps, model_path=args.model or None, crf=args.crf
        )
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
