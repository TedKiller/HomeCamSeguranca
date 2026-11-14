"""Gravação contínua segmentada (FFmpeg), gravação por movimento (OpenCV) e snapshots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np

from config_manager import sanitize_filename_part
from xiongmai_netip import XiongmaiAnnexBReader, parse_dvrip_url

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def build_recording_dir(recordings_root: str | Path, camera_name: str, when: datetime | None = None) -> Path:
    """Retorna e garante o diretório `<raiz>/<camera>/<YYYY-MM-DD>` para gravações."""
    dt = when or datetime.now()
    cam_part = sanitize_filename_part(camera_name, "Camera")
    day_part = dt.strftime("%Y-%m-%d")
    target = Path(recordings_root) / cam_part / day_part
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_snapshot(frame: np.ndarray, recordings_root: str | Path, camera_name: str) -> Path:
    """Salva uma foto JPEG do quadro atual na pasta da câmera e retorna o caminho."""
    now = datetime.now()
    out_dir = build_recording_dir(recordings_root, camera_name, now)
    cam_part = sanitize_filename_part(camera_name, "Camera")
    filename = f"{cam_part}_foto_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    out_path = out_dir / filename
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Falha ao codificar snapshot JPEG para {out_path}")
    out_path.write_bytes(encoded.tobytes())
    return out_path


def is_valid_mp4_file(path: Path | str) -> bool:
    """Valida rapidamente a estrutura de átomos ISO/MP4 (`moov` e mídia gravada)."""
    target = Path(path)
    try:
        file_size = target.stat().st_size
        if file_size <= 64:
            return False
        has_moov = False
        has_media = False
        with target.open("rb") as fh:
            pos = 0
            boxes = 0
            while pos + 8 <= file_size and boxes < 1000:
                fh.seek(pos)
                header = fh.read(8)
                if len(header) < 8:
                    break
                box_size, box_type = struct.unpack(">I4s", header)
                header_len = 8
                if box_size == 1:
                    ext = fh.read(8)
                    if len(ext) < 8:
                        break
                    (box_size,) = struct.unpack(">Q", ext)
                    header_len = 16
                elif box_size == 0:
                    box_size = file_size - pos
                if box_size < header_len:
                    break

                if box_type == b"moov" and box_size > 16:
                    has_moov = True
                elif box_type == b"moof" and box_size > 16:
                    has_media = True
                elif box_type == b"mdat" and box_size > 16:
                    has_media = True

                if has_moov and has_media:
                    return True

                next_pos = pos + box_size
                if next_pos <= pos or next_pos > file_size:
                    break
                pos = next_pos
                boxes += 1
        return has_moov and has_media
    except OSError:
        return False


def _segment_muxer_args(segment_seconds: int, out_pattern: str) -> list[str]:
    """Argumentos FFmpeg para segmentação contínua em MP4 fragmentado tolerante a quedas."""
    return [
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-segment_format",
        "mp4",
        "-segment_format_options",
        "movflags=+frag_keyframe+empty_moov+default_base_moof",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        out_pattern,
    ]


class MotionDetector:
    """Detector de movimento leve por média móvel ponderada de fundo em escala de cinza."""

    def __init__(self, sensitivity: float = 0.015, warmup_frames: int = 8) -> None:
        self.sensitivity = max(0.001, min(0.5, float(sensitivity)))
        self.warmup_frames = max(0, int(warmup_frames))
        self._bg: np.ndarray | None = None
        self._frame_count = 0

    def reset(self) -> None:
        """Reinicia o modelo de fundo (chamar ao trocar de câmera)."""
        self._bg = None
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        """Analisa um frame BGR e retorna `(movimento_detectado, fracao_alterada)`."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._bg is None or self._bg.shape != blurred.shape:
            self._bg = blurred.astype("float32")
            self._frame_count = 1
            return False, 0.0

        self._frame_count += 1
        cv2.accumulateWeighted(blurred, self._bg, 0.08)
        diff = cv2.absdiff(blurred, cv2.convertScaleAbs(self._bg))
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        changed_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)

        if self._frame_count <= self.warmup_frames:
            return False, changed_ratio
        return changed_ratio >= self.sensitivity, changed_ratio


class VideoRecorder:
    """Gerencia gravação contínua (FFmpeg) ou por movimento/USB (OpenCV) com rotação por segmento."""

    def __init__(
        self,
        recordings_root: str | Path,
        on_segment_ready: Callable[[Path, str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root)
        self.on_segment_ready = on_segment_ready
        self.on_log = on_log

        self._lock = threading.Lock()
        self._active = False
        self._mode = "idle"  # "ffmpeg" | "opencv" | "idle"
        self._camera_name = "Camera"
        self._segment_seconds = 300
        self._motion_only = False
        self._motion_post_seconds = 12
        self._source: str | int = ""

        # Estado FFmpeg
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._watch_root: Path | None = None
        self._file_prefix = ""
        self._emitted_files: set[Path] = set()

        # Estado OpenCV
        self._detector = MotionDetector()
        self._writer: cv2.VideoWriter | None = None
        self._writer_path: Path | None = None
        self._writer_started_at = 0.0
        self._last_motion_at = 0.0
        self._motion_now = False
        self._fps = 15.0
        self._recording_started_at = 0.0
        self._recording_wall_start = 0.0
        self._active_file_seen: tuple[str, float] | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._active

    @property
    def motion_detected_now(self) -> bool:
        with self._lock:
            return self._motion_now

    def get_active_recording_stats(self) -> tuple[str, int, float, int]:
        """Retorna `(arquivo_atual, tamanho_bytes, segundos_decorridos, duracao_segmento)`."""
        with self._lock:
            active = self._active
            mode = self._mode
            watch_root = self._watch_root
            prefix = self._file_prefix
            seg_secs = self._segment_seconds
            motion_only = self._motion_only
            writer_path = self._writer_path
            writer_start = self._writer_started_at
            rec_start = self._recording_started_at
            wall_start = self._recording_wall_start

        if not active:
            return ("", 0, 0.0, seg_secs)

        now_mono = time.monotonic()
        if mode == "opencv":
            if writer_path is not None and writer_path.exists():
                try:
                    size = writer_path.stat().st_size
                except OSError:
                    size = 0
                return (writer_path.name, size, max(0.0, now_mono - writer_start), seg_secs)
            label = "Aguardando movimento..." if motion_only else "Iniciando vídeo..."
            return (label, 0, 0.0, seg_secs)

        if watch_root is not None and watch_root.exists():
            newest_path: Path | None = None
            newest_mtime = -1.0
            newest_size = 0
            for candidate in watch_root.rglob(f"{prefix}_*.mp4"):
                try:
                    st = candidate.stat()
                except OSError:
                    continue
                if st.st_mtime >= wall_start - 2.0 and (st.st_mtime, candidate.name) > (newest_mtime, newest_path.name if newest_path else ""):
                    newest_mtime = st.st_mtime
                    newest_path = candidate
                    newest_size = st.st_size
            if newest_path is not None:
                with self._lock:
                    if self._active_file_seen is None or self._active_file_seen[0] != newest_path.name:
                        self._active_file_seen = (newest_path.name, now_mono)
                        first_seen = now_mono
                    else:
                        first_seen = self._active_file_seen[1]
                return (newest_path.name, newest_size, max(0.0, now_mono - first_seen), seg_secs)

        return ("Iniciando arquivo MP4...", 0, max(0.0, now_mono - rec_start), seg_secs)

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    def _emit_ready(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self._emitted_files:
            return
        try:
            st = path.stat()
        except OSError:
            return
        if not path.is_file() or st.st_size <= 0:
            return
        if path.suffix.lower() == ".mp4" and not is_valid_mp4_file(path):
            return
        self._emitted_files.add(resolved)
        self._log(f"Gravação concluída: {path.name}")
        if self.on_segment_ready:
            self.on_segment_ready(path, self._camera_name)

    def start(
        self,
        source: str | int,
        camera_name: str,
        segment_minutes: int = 5,
        motion_only: bool = False,
        motion_sensitivity: float = 0.015,
        motion_post_seconds: int = 12,
        fps: float = 15.0,
    ) -> None:
        """Inicia a gravação da câmera selecionada."""
        self.stop()

        cam_name = camera_name or "Camera"
        probe_dir = build_recording_dir(self.recordings_root, cam_name)
        probe_file = probe_dir / f".homecam_rec_probe_{threading.get_ident()}"
        probe_file.write_bytes(b"")
        probe_file.unlink(missing_ok=True)

        with self._lock:
            self._camera_name = cam_name
            self._file_prefix = sanitize_filename_part(self._camera_name, "Camera")
            self._segment_seconds = max(10, int(segment_minutes) * 60)
            self._motion_only = bool(motion_only)
            self._motion_post_seconds = max(3, int(motion_post_seconds))
            self._source = source
            self._fps = max(5.0, min(30.0, float(fps or 15.0)))
            self._emitted_files.clear()
            self._stop_event.clear()
            self._detector = MotionDetector(sensitivity=motion_sensitivity)
            self._motion_now = False
            self._last_motion_at = 0.0
            self._recording_started_at = time.monotonic()
            self._recording_wall_start = time.time()
            self._active_file_seen = None

            try:
                use_ffmpeg = isinstance(source, str) and not motion_only
                if use_ffmpeg:
                    self._mode = "ffmpeg"
                    self._watch_root = self.recordings_root / self._file_prefix
                    self._watch_root.mkdir(parents=True, exist_ok=True)
                    target_loop = (
                        self._run_dvrip_ffmpeg_loop
                        if source.strip().lower().startswith("dvrip://")
                        else self._run_ffmpeg_loop
                    )
                    self._ffmpeg_thread = threading.Thread(
                        target=target_loop,
                        daemon=True,
                    )
                    self._active = True
                    self._ffmpeg_thread.start()
                else:
                    self._mode = "opencv"
                    self._active = True
                    self._log(
                        "Gravação por movimento iniciada."
                        if motion_only
                        else "Gravação contínua via webcam USB iniciada."
                    )
            except Exception:
                self._active = False
                self._mode = "idle"
                raise

    def stop(self) -> None:
        """Encerra a gravação ativa e finaliza o segmento aberto com segurança."""
        closed_cv_path: Path | None = None
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop_event.set()
            mode = self._mode
            self._mode = "idle"
            proc = self._ffmpeg_proc
            thread = self._ffmpeg_thread
            is_dvrip = str(self._source).strip().lower().startswith("dvrip://")
            closed_cv_path = self._close_cv_writer_locked()

        if closed_cv_path is not None:
            self._emit_ready(closed_cv_path)

        if mode == "ffmpeg":
            if not is_dvrip and proc is not None and proc.poll() is None:
                self._stop_ffmpeg_process(proc)
            if thread is not None and thread.is_alive():
                thread.join(timeout=15.0)
            self._flush_remaining_ffmpeg_files()

    def feed_frame(self, frame: np.ndarray) -> bool:
        """Alimenta um frame BGR capturado na GUI (usado em modo USB ou movimento).

        Retorna True se movimento foi detectado neste frame.
        """
        closed_paths: list[Path] = []
        with self._lock:
            if not self._active:
                return False

            detected, _ratio = self._detector.update(frame)
            self._motion_now = detected

            if self._mode != "opencv":
                return detected

            now = time.monotonic()
            if self._motion_only:
                if detected:
                    if self._writer is None:
                        self._log("Movimento detectado! Iniciando clipe...")
                        self._open_cv_writer_locked(frame, tag="mov")
                    self._last_motion_at = now
                if self._writer is not None:
                    self._writer.write(frame)
                    if (now - self._last_motion_at) > self._motion_post_seconds:
                        closed = self._close_cv_writer_locked()
                        if closed:
                            closed_paths.append(closed)
                    elif (now - self._writer_started_at) >= self._segment_seconds:
                        closed = self._close_cv_writer_locked()
                        if closed:
                            closed_paths.append(closed)
                        if detected:
                            self._open_cv_writer_locked(frame, tag="mov")
            else:
                if self._writer is None:
                    self._open_cv_writer_locked(frame, tag="usb")
                self._writer.write(frame)
                if (now - self._writer_started_at) >= self._segment_seconds:
                    closed = self._close_cv_writer_locked()
                    if closed:
                        closed_paths.append(closed)
                    self._open_cv_writer_locked(frame, tag="usb")

        for path in closed_paths:
            self._emit_ready(path)
        return detected

    def _open_cv_writer_locked(self, frame: np.ndarray, tag: str) -> None:
        now = datetime.now()
        try:
            out_dir = build_recording_dir(self.recordings_root, self._camera_name, now)
        except OSError as err:
            self._log(f"Não foi possível criar pasta de gravação ({err}).")
            return
        stem = f"{self._file_prefix}_{tag}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        out_path = out_dir / f"{stem}.mp4"
        counter = 1
        while out_path.exists():
            out_path = out_dir / f"{stem}_{counter:02d}.mp4"
            counter += 1
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            self._log(f"Não foi possível criar arquivo de gravação {out_path.name}.")
            return
        self._writer = writer
        self._writer_path = out_path
        self._writer_started_at = time.monotonic()

    def _close_cv_writer_locked(self) -> Path | None:
        writer = self._writer
        path = self._writer_path
        self._writer = None
        self._writer_path = None
        if writer is not None:
            writer.release()
        return path

    def _ensure_day_folders(self) -> None:
        now = datetime.now()
        build_recording_dir(self.recordings_root, self._camera_name, now)
        build_recording_dir(self.recordings_root, self._camera_name, now + timedelta(days=1))

    def _close_proc_pipes(self, proc: subprocess.Popen) -> None:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    def _build_ffmpeg_command(self, include_audio: bool = True, copy_video: bool = True) -> list[str]:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._ensure_day_folders()
        assert self._watch_root is not None
        out_pattern = str(self._watch_root / "%Y-%m-%d" / f"{self._file_prefix}_%Y-%m-%d_%H-%M-%S.mp4")

        cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y"]
        src_str = str(self._source)
        if src_str.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        cmd.extend(["-i", src_str])
        cmd.extend(["-c:v", "copy" if copy_video else "mpeg4"])
        if include_audio:
            cmd.extend(["-c:a", "aac"])
        else:
            cmd.append("-an")
        cmd.extend(_segment_muxer_args(self._segment_seconds, out_pattern))
        return cmd

    def _run_ffmpeg_loop(self) -> None:
        variants = [
            (True, True),
            (False, True),
            (False, False),
        ]
        variant_idx = 0
        self._log(f"Gravação contínua iniciada (segmentos de {self._segment_seconds // 60 or 1} min).")

        while not self._stop_event.is_set():
            include_audio, copy_video = variants[variant_idx]
            cmd = self._build_ffmpeg_command(include_audio=include_audio, copy_video=copy_video)
            started_at = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=_CREATE_NO_WINDOW,
            )
            with self._lock:
                self._ffmpeg_proc = proc

            try:
                while proc.poll() is None and not self._stop_event.is_set():
                    self._ensure_day_folders()
                    self._scan_completed_ffmpeg_segments()
                    self._stop_event.wait(1.0)

                self._scan_completed_ffmpeg_segments()
                if self._stop_event.is_set():
                    break

                is_network_stream = str(self._source).lower().startswith(("rtsp://", "http://", "https://"))
                if proc.returncode == 0 and not is_network_stream:
                    self._flush_remaining_ffmpeg_files()
                    with self._lock:
                        self._active = False
                        self._mode = "idle"
                    break

                elapsed = time.monotonic() - started_at
                stderr_text = ""
                if proc.stderr is not None:
                    try:
                        stderr_text = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        pass
            finally:
                if proc.poll() is None:
                    self._stop_ffmpeg_process(proc)
                self._close_proc_pipes(proc)

            if proc.returncode != 0 and elapsed < 5.0 and variant_idx + 1 < len(variants):
                variant_idx += 1
                self._log("Ajustando parâmetros de áudio/vídeo para compatibilidade com a câmera...")
                continue

            if stderr_text:
                last_line = stderr_text.splitlines()[-1]
                self._log(f"Conexão de gravação interrompida ({last_line}). Reconectando em 4s...")
            else:
                self._log("Fluxo de gravação encerrado inesperadamente. Reconectando em 4s...")
            self._stop_event.wait(4.0)

        with self._lock:
            self._ffmpeg_proc = None

    def _run_dvrip_ffmpeg_loop(self) -> None:
        self._log(f"Gravação contínua NetIP iniciada (segmentos de {self._segment_seconds // 60 or 1} min).")
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        ip, port, user, pwd, ch, st = parse_dvrip_url(str(self._source))

        while not self._stop_event.is_set():
            proc: subprocess.Popen | None = None
            proc_lock = threading.Lock()
            reader = XiongmaiAnnexBReader(ip, port, user, pwd, ch, st, timeout_sec=6.0)

            def _on_chunk(codec: str, chunk: bytes) -> None:
                nonlocal proc
                with proc_lock:
                    if self._stop_event.is_set():
                        return
                    if proc is None:
                        self._ensure_day_folders()
                        assert self._watch_root is not None
                        out_pattern = str(
                            self._watch_root / "%Y-%m-%d" / f"{self._file_prefix}_%Y-%m-%d_%H-%M-%S.mp4"
                        )
                        cmd = [
                            exe,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-use_wallclock_as_timestamps",
                            "1",
                            "-f",
                            codec,
                            "-i",
                            "pipe:0",
                            "-c:v",
                            "copy",
                            *_segment_muxer_args(self._segment_seconds, out_pattern),
                        ]
                        proc = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            creationflags=_CREATE_NO_WINDOW,
                        )
                        with self._lock:
                            self._ffmpeg_proc = proc
                    if proc.poll() is None and proc.stdin is not None:
                        try:
                            proc.stdin.write(chunk)
                            proc.stdin.flush()
                        except OSError:
                            pass

            started_ok = reader.start(_on_chunk)
            if not started_ok or reader.last_error:
                reader.stop()
                if not self._stop_event.is_set():
                    self._log(f"Falha na gravação NetIP ({reader.last_error or 'sem resposta'}). Reconectando em 4s...")
                    self._stop_event.wait(4.0)
                continue

            try:
                while not self._stop_event.is_set() and not reader.last_error:
                    self._ensure_day_folders()
                    self._scan_completed_ffmpeg_segments()
                    with proc_lock:
                        if proc is not None and proc.poll() is not None:
                            break
                    self._stop_event.wait(1.0)
            finally:
                reader.stop()
                with proc_lock:
                    cur_proc = proc
                    proc = None
                ffmpeg_err = ""
                if cur_proc is not None:
                    self._stop_ffmpeg_process(cur_proc, pipe_input=True)
                    if cur_proc.stderr is not None:
                        try:
                            ffmpeg_err = cur_proc.stderr.read().decode("utf-8", errors="ignore").strip()
                        except Exception:
                            pass
                    self._close_proc_pipes(cur_proc)

            self._scan_completed_ffmpeg_segments()
            if self._stop_event.is_set():
                break
            if ffmpeg_err:
                last_err = ffmpeg_err.splitlines()[-1]
                self._log(f"Fluxo de gravação NetIP interrompido ({last_err}). Reconectando em 4s...")
            else:
                self._log("Fluxo de gravação NetIP interrompido. Reconectando em 4s...")
            self._stop_event.wait(4.0)

        with self._lock:
            self._ffmpeg_proc = None

    def _stop_ffmpeg_process(self, proc: subprocess.Popen, pipe_input: bool = False) -> None:
        is_pipe = pipe_input or str(self._source).strip().lower().startswith("dvrip://")
        try:
            if proc.stdin is not None:
                if not is_pipe:
                    proc.stdin.write(b"q\n")
                    proc.stdin.flush()
                else:
                    proc.stdin.close()
            proc.wait(timeout=8.0)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()

    def _scan_completed_ffmpeg_segments(self) -> None:
        if self._watch_root is None or not self._watch_root.exists():
            return
        files = sorted(self._watch_root.rglob(f"{self._file_prefix}_*.mp4"))
        if len(files) <= 1:
            return
        now = time.time()
        for candidate in files[:-1]:
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if now - mtime >= 2.0:
                self._emit_ready(candidate)

    def _flush_remaining_ffmpeg_files(self) -> None:
        if self._watch_root is None or not self._watch_root.exists():
            return
        for candidate in sorted(self._watch_root.rglob(f"{self._file_prefix}_*.mp4")):
            if candidate.resolve() in self._emitted_files:
                continue
            if is_valid_mp4_file(candidate):
                self._emit_ready(candidate)
            else:
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                if mtime >= self._recording_wall_start - 2.0:
                    self._log(
                        f"Aviso: {candidate.name} ficou incompleto/sem índice MP4 e não foi enviado ao Drive."
                    )
