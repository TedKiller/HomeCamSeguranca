"""Motor NVR 24/7 headless desacoplado da GUI Desktop (gravação, IA, Drive, DVR e PTZ)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import ctypes
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
import shutil
import sys
import threading
import time

import cv2
import numpy as np

from ai_analytics import (
    AISummary,
    CCTVAnalyticsPipeline,
    PersonRecord,
    VehicleRecord,
    draw_ai_overlays,
)
from camera_discovery import (
    mask_rtsp_url,
    open_video_capture,
    resolve_best_connection_source,
)
from config_manager import (
    AppConfig,
    CameraConfig,
    ConfigManager,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PENDING_PATH,
    resolve_writable_recordings_dir,
    sanitize_filename_part,
)
from dvr_playback import (
    DVREventStore,
    DVRTimelineIndex,
    EVENT_TYPE_LABELS,
    TimelineEvent,
    TimelineSegment,
)
from frame_interpolator import LiveFrameInterpolator, interpolate_video_ffmpeg
from gdrive_uploader import GoogleDriveUploader
from image_enhancer import SuperResolutionEnhancer
from ptz_control import create_ptz_backend
from recorder import VideoRecorder, is_valid_mp4_file

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_PTZ_DIRS = frozenset(
    {"up", "down", "left", "right", "left_up", "right_up", "left_down", "right_down", "zoom_in", "zoom_out"}
)


class WindowsPowerGuard:
    """Impede suspensão automática do PC/rede/GPU durante operação 24/7 (permite desligar o monitor)."""

    def __init__(
        self,
        keepalive_sec: float = 30.0,
        set_state_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.keepalive_sec = max(5.0, float(keepalive_sec))
        self._set_state_fn = set_state_fn
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._last_flags = 0

    @property
    def is_supported(self) -> bool:
        return self._set_state_fn is not None or sys.platform == "win32"

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def _call_execution_state(self, flags: int) -> int:
        if self._set_state_fn is not None:
            return int(self._set_state_fn(int(flags)))
        if sys.platform != "win32":
            return 0
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(int(flags))
            return int(res or 0)
        except Exception:
            return 0

    def _apply_keepawake(self) -> bool:
        desired = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        res = self._call_execution_state(desired)
        if res != 0:
            self._last_flags = desired
            return True
        fallback = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        res2 = self._call_execution_state(fallback)
        self._last_flags = fallback if res2 != 0 else 0
        return res2 != 0

    def _keepalive_loop(self) -> None:
        while not self._stop_event.wait(self.keepalive_sec):
            self._apply_keepawake()
        self._call_execution_state(ES_CONTINUOUS)

    def acquire(self) -> bool:
        """Ativa prevenção de suspensão do sistema mantendo o desligamento de tela livre."""
        if not self.is_supported:
            return False
        with self._lock:
            if self._active:
                return True
            ok = self._apply_keepawake()
            self._active = bool(ok)
            if ok:
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._keepalive_loop,
                    name="HomeCamPowerGuard",
                    daemon=True,
                )
                self._thread.start()
            return ok

    def release(self) -> None:
        """Libera a trava de energia do processo com segurança."""
        with self._lock:
            was_active = self._active
            self._active = False
            thread = self._thread
            self._thread = None
            self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        if was_active:
            self._call_execution_state(ES_CONTINUOUS)

    def __enter__(self) -> "WindowsPowerGuard":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def record_ai_summary_events(
    dvr_events: DVREventStore,
    camera_name: str,
    summary: AISummary,
    person_records: list[PersonRecord] | None = None,
) -> None:
    """Registra eventos de IA (pessoas/vestuário, animais, veículos e placas) na linha do tempo DVR."""
    if summary.people_total > 0:
        clothes_hint = ""
        if person_records and person_records[0].clothing_colors != "Analisando...":
            clothes_hint = f" [{person_records[0].clothing_colors}]"
        dvr_events.record_event(
            camera_name,
            "ai_person",
            f"Pessoas: {summary.people_total} (H:{summary.men_count} | M:{summary.women_count}){clothes_hint}",
            min_interval_sec=4.0,
        )
    if summary.animals_count > 0:
        dvr_events.record_event(
            camera_name,
            "ai_animal",
            f"Animais: {summary.animals_count}",
            min_interval_sec=4.0,
        )
    if summary.cars_count > 0 or summary.motos_count > 0:
        dvr_events.record_event(
            camera_name,
            "ai_vehicle",
            f"Veículos: {summary.cars_count} carro(s), {summary.motos_count} moto(s)",
            min_interval_sec=4.0,
        )
    if summary.last_plate:
        dvr_events.record_event(
            camera_name,
            "ai_plate",
            f"Placa ALPR: {summary.last_plate}",
            min_interval_sec=15.0,
            dedupe_key=summary.last_plate,
        )


def recover_orphaned_recordings(
    rec_root: Path,
    uploader: GoogleDriveUploader,
    *,
    drive_enabled: bool,
    auto_upload: bool,
    delete_after_upload: bool,
    should_stop: Callable[[], bool] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Recupera gravações válidas pendentes de sessões anteriores e isola MP4s corrompidos."""
    if not rec_root.exists() or not rec_root.is_dir():
        return (0, 0)

    now = time.time()
    recovered = 0
    quarantined = 0
    drive_auto = bool(drive_enabled and auto_upload)
    skip_dirs = {"_corrompidos", "_playback_cache", "_timeline_events"}

    for candidate in sorted(rec_root.rglob("*")):
        if should_stop is not None and should_stop():
            break
        if any(part in skip_dirs for part in candidate.parts) or not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix not in (".mp4", ".jpg"):
            continue
        try:
            st = candidate.stat()
        except OSError:
            continue

        age = now - st.st_mtime
        if age < 15.0:
            continue

        if suffix == ".mp4" and not is_valid_mp4_file(candidate):
            if age >= 60.0:
                quarantine_dir = rec_root / "_corrompidos"
                try:
                    quarantine_dir.mkdir(parents=True, exist_ok=True)
                    dest = quarantine_dir / candidate.name
                    counter = 1
                    while dest.exists():
                        dest = quarantine_dir / f"{candidate.stem}_{counter:02d}{candidate.suffix}"
                        counter += 1
                    shutil.move(str(candidate), str(dest))
                    quarantined += 1
                    if on_log:
                        on_log(
                            f"[Gravação] Arquivo antigo sem índice MP4 movido para _corrompidos: {candidate.name}"
                        )
                except OSError:
                    pass
            continue

        if not drive_auto:
            continue
        if not delete_after_upload and age > 48 * 3600:
            continue
        if uploader.enqueue_file(candidate):
            recovered += 1

    if recovered > 0 and on_log:
        on_log(f"[Drive] {recovered} gravação(ões) anterior(es) recuperada(s) na fila de envio.")
    return (recovered, quarantined)


class HeadlessNVRService:
    """Serviço autônomo 24/7 para captura, gravação contínua, IA, eventos DVR, Drive e PTZ."""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        *,
        recordings_root: Path | str | None = None,
        pending_path: Path | str = DEFAULT_PENDING_PATH,
        auto_connect: bool = True,
        auto_record: bool | None = None,
        enable_ai: bool | None = None,
        load_ai_async: bool = True,
        power_guard: WindowsPowerGuard | None = None,
        uploader: GoogleDriveUploader | None = None,
        recorder: VideoRecorder | None = None,
        dvr_events: DVREventStore | None = None,
        ai_pipeline: CCTVAnalyticsPipeline | None = None,
        source_resolver: Callable[[CameraConfig], str | int] | None = None,
        capture_factory: Callable[[str | int, int], object] | None = None,
        ptz_factory: Callable[[CameraConfig, str | int], object] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.config_manager = config_manager or ConfigManager(DEFAULT_CONFIG_PATH)
        self.config: AppConfig = self.config_manager.load()

        if recordings_root is not None:
            resolved_dir, _ = resolve_writable_recordings_dir(recordings_root, prefer_drive_subdir=False)
            self.config.recording.recordings_dir = str(resolved_dir)
            self.config.ai.csv_log_path = str(resolved_dir / "veiculos_detectados.csv")
            self.config.ai.people_csv_log_path = str(resolved_dir / "pessoas_detectadas.csv")

        self.recordings_root = Path(self.config.recording.recordings_dir)
        self.auto_connect = bool(auto_connect)
        self.auto_record = bool(self.config.recording.auto_start_recording) if auto_record is None else bool(auto_record)
        self.enable_ai = bool(self.config.ai.enabled) if enable_ai is None else bool(enable_ai)
        self.load_ai_async = bool(load_ai_async)

        self._external_log = on_log
        self._logs: deque[str] = deque(maxlen=250)
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._ai_init_thread: threading.Thread | None = None
        self._download_lock = threading.Lock()
        self._file_download_locks: dict[str, threading.Lock] = {}

        self._source_resolver = source_resolver or resolve_best_connection_source
        self._capture_factory = capture_factory or (lambda src, timeout_ms: open_video_capture(src, timeout_ms=timeout_ms))
        self._ptz_factory = ptz_factory or create_ptz_backend
        self.power_guard = power_guard or WindowsPowerGuard()

        self.uploader = uploader or GoogleDriveUploader(
            pending_path=pending_path,
            credentials_path=self.config.drive.credentials_path,
            token_path=self.config.drive.token_path,
            root_folder_name=self.config.drive.root_folder_name,
            delete_after_upload=self.config.drive.delete_after_upload,
            on_log=lambda msg: self.post_log(f"[Drive] {msg}"),
            sync_mode=self.config.drive.sync_mode,
            local_drive_dir=self.config.drive.local_drive_dir,
        )
        self.recorder = recorder or VideoRecorder(
            recordings_root=self.recordings_root,
            on_segment_ready=self._on_segment_ready,
            on_log=lambda msg: self.post_log(f"[Gravação] {msg}"),
        )
        self.dvr_events = dvr_events or DVREventStore(self.recordings_root / "_timeline_events")
        self.ai_pipeline = ai_pipeline or CCTVAnalyticsPipeline(
            csv_path=self.config.ai.csv_log_path,
            people_csv_path=self.config.ai.people_csv_log_path,
            save_csv=self.config.ai.save_csv_log,
            analysis_fps=self.config.ai.analysis_fps,
            on_update=self._on_ai_update,
            on_log=self.post_log,
        )
        if self.ai_pipeline.on_update is None:
            self.ai_pipeline.on_update = self._on_ai_update
        self.ai_pipeline.configure(
            analysis_fps=self.config.ai.analysis_fps,
            detect_people_gender=self.config.ai.detect_people_gender,
            detect_animals=self.config.ai.detect_animals,
            detect_vehicles_alpr=self.config.ai.detect_vehicles_alpr,
            save_csv=self.config.ai.save_csv_log,
            csv_path=self.config.ai.csv_log_path,
            people_csv_path=self.config.ai.people_csv_log_path,
        )

        self._started_at: float | None = None
        self._camera_connected = False
        self._connection_state = "stopped"
        self._current_fps = 15.0
        self._motion_now = False
        self._frame_seq = 0
        self._latest_frame: np.ndarray | None = None
        self._last_frame_wall: float = 0.0
        self._cached_jpeg_key: tuple[int, bool, bool, int] | None = None
        self._cached_jpeg_bytes: bytes | None = None
        self._frame_cond = threading.Condition()
        self.enhancer = SuperResolutionEnhancer()

        self._ptz_lock = threading.Lock()
        self._ptz_client: object | None = None
        self._ptz_backend_label: str = ""
        self._active_source: str | int | None = None
        self._recording_started_for_source: str | int | None = None

    def post_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self._state_lock:
            self._logs.append(line)
        if self._external_log is not None:
            self._external_log(line)

    @property
    def selected_camera(self) -> CameraConfig:
        return self.config_manager.get_selected_camera(self.config)

    @property
    def is_running(self) -> bool:
        return self._started_at is not None and not self._stop_event.is_set()

    def _on_segment_ready(self, file_path: Path, camera_name: str) -> None:
        if self.config.drive.enabled and self.config.drive.auto_upload:
            if self.uploader.enqueue_file(file_path, camera_name=camera_name):
                self.post_log(f"[Drive] Na fila para backup: {file_path.name}")

    def _on_ai_update(self, summary: AISummary, _vehicles: list[VehicleRecord]) -> None:
        record_ai_summary_events(
            self.dvr_events,
            self.selected_camera.name,
            summary,
            self.ai_pipeline.get_person_records(),
        )

    def _init_ai_worker(self) -> None:
        if self._stop_event.is_set():
            return
        if not self.ai_pipeline.is_ready:
            ok = self.ai_pipeline.load_default_models()
            if not ok or self._stop_event.is_set():
                self.post_log("[IA] Modelos ONNX indisponíveis no modo headless.")
                return
        self.ai_pipeline.start()
        self.post_log(f"[IA] Pipeline 24/7 ativo ({self.ai_pipeline.get_summary().backend_name}).")

    def _recover_worker(self) -> None:
        recover_orphaned_recordings(
            self.recordings_root,
            self.uploader,
            drive_enabled=self.config.drive.enabled,
            auto_upload=self.config.drive.auto_upload,
            delete_after_upload=self.config.drive.delete_after_upload,
            should_stop=self._stop_event.is_set,
            on_log=self.post_log,
        )

    def start(self) -> None:
        """Inicia o serviço autônomo NVR 24/7 (energia, recuperação, Uploader, IA e captura)."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()

        if self.power_guard.acquire():
            self.post_log("[Energia] WindowsPowerGuard ativo: PC/GPU/Rede 24/7 (monitor livre para desligar).")

        self.uploader.start_worker()
        threading.Thread(target=self._recover_worker, name="HomeCamOrphanRecover", daemon=True).start()

        if self.enable_ai:
            if self.load_ai_async:
                self._ai_init_thread = threading.Thread(
                    target=self._init_ai_worker,
                    name="HomeCamAIInit",
                    daemon=True,
                )
                self._ai_init_thread.start()
            else:
                self._init_ai_worker()

        if self.auto_connect:
            self._connection_state = "connecting"
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="HomeCamCapture24h",
                daemon=True,
            )
            self._capture_thread.start()
        else:
            self._connection_state = "standby"

        self.post_log(f"[NVR 24/7] Serviço iniciado para câmera '{self.selected_camera.name}'.")

    def stop(self, wait_uploads_sec: float = 0.0) -> None:
        """Encerra captura, gravação MP4 ativa, IA (flush CSV), Drive e trava de energia."""
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=8.0)
        self._capture_thread = None

        self.recorder.stop()
        self.ai_pipeline.stop()

        if wait_uploads_sec > 0:
            deadline = time.monotonic() + float(wait_uploads_sec)
            while self.uploader.get_pending_count() > 0 and time.monotonic() < deadline:
                self.uploader.trigger_sync()
                time.sleep(0.25)

        self.uploader.stop_worker()
        self._close_ptz_client()
        self.power_guard.release()

        with self._state_lock:
            self._camera_connected = False
            self._connection_state = "stopped"
        self._started_at = None
        self.post_log("[NVR 24/7] Serviço encerrado com segurança.")

    def _close_ptz_client(self) -> None:
        with self._ptz_lock:
            client = self._ptz_client
            self._ptz_client = None
            self._ptz_backend_label = ""
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass

    def _init_ptz_async(self, cam: CameraConfig, source: str | int) -> None:
        def worker() -> None:
            try:
                client = self._ptz_factory(cam, source)
            except Exception:
                client = None
            with self._ptz_lock:
                old = self._ptz_client
                self._ptz_client = client
                self._ptz_backend_label = str(getattr(client, "backend_label", "") or "")
            if old is not None and old is not client and hasattr(old, "close"):
                try:
                    old.close()
                except Exception:
                    pass

        threading.Thread(target=worker, name="HomeCamPTZInit", daemon=True).start()

    def _ensure_recording_for_source(self, cam: CameraConfig, source: str | int, fps: float) -> None:
        if not self.auto_record:
            return
        if self.recorder.is_recording and self._recording_started_for_source == source:
            return
        try:
            self.recorder.recordings_root = self.recordings_root
            self.recorder.start(
                source=source,
                camera_name=cam.name,
                segment_minutes=self.config.recording.segment_minutes,
                motion_only=self.config.recording.motion_only,
                motion_sensitivity=self.config.recording.motion_sensitivity,
                motion_post_seconds=self.config.recording.motion_post_seconds,
                fps=fps,
            )
            self._recording_started_for_source = source
            self.post_log(f"[Gravação] Gravação 24/7 iniciada ({cam.name}).")
        except Exception as err:
            self.post_log(f"[Gravação] Falha ao iniciar gravação automática: {err}")

    def _capture_loop(self) -> None:
        cam = self.selected_camera
        cam_name = cam.name
        last_source: object = None
        fallback_logged = False

        while not self._stop_event.is_set():
            source = self._source_resolver(cam)
            if (
                not fallback_logged
                and isinstance(source, str)
                and source.lower().startswith("dvrip://")
                and "xiongmai netip" not in cam.brand.lower()
            ):
                fallback_logged = True
                self.post_log(f"RTSP indisponível; usando Xiongmai NetIP ({mask_rtsp_url(source)}).")

            self._active_source = source
            cap = self._capture_factory(source, 5000)
            if not getattr(cap, "isOpened", lambda: False)():
                try:
                    cap.release()
                except Exception:
                    pass
                with self._state_lock:
                    self._camera_connected = False
                    self._connection_state = "reconnecting"
                if self.auto_record and isinstance(source, str) and not self.config.recording.motion_only:
                    self._ensure_recording_for_source(cam, source, self._current_fps)
                self.post_log(f"Falha ao abrir câmera '{cam_name}'. Nova tentativa em 4s...")
                self._stop_event.wait(4.0)
                continue

            if last_source != source:
                last_source = source
                self._init_ptz_async(cam, source)

            try:
                raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
            except Exception:
                raw_fps = 15.0
            fps = raw_fps if 0 < raw_fps <= 120 else 15.0

            with self._state_lock:
                self._current_fps = fps
                self._camera_connected = True
                self._connection_state = "live"

            self.post_log(f"[Captura] Fluxo ao vivo ativo: {cam_name}")
            self._ensure_recording_for_source(cam, source, fps)

            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.post_log("[Captura] Sinal perdido. Reconectando...")
                        with self._state_lock:
                            self._camera_connected = False
                            self._connection_state = "reconnecting"
                        break
                    self.process_live_frame(frame, camera_name=cam_name)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass

            if not self._stop_event.is_set():
                self._stop_event.wait(3.0)

    def process_live_frame(self, frame: np.ndarray, camera_name: str | None = None) -> None:
        """Processa um quadro BGR ao vivo (gravação, detecção de movimento, IA e buffer MJPEG)."""
        cam_name = camera_name or self.selected_camera.name
        motion = False
        if self.recorder.is_recording:
            motion = self.recorder.feed_frame(frame)
            if motion:
                self.dvr_events.record_event(
                    cam_name,
                    "motion",
                    "Movimento detectado",
                    min_interval_sec=4.0,
                )

        if self.enable_ai and self.ai_pipeline.is_ready:
            self.ai_pipeline.submit_frame(frame, camera_name=cam_name)

        frame_copy = frame.copy()
        now_wall = time.time()
        with self._frame_cond:
            self._latest_frame = frame_copy
            self._last_frame_wall = now_wall
            self._motion_now = bool(motion)
            self._frame_seq += 1
            self._frame_cond.notify_all()

    def encode_frame_jpeg(
        self,
        frame: np.ndarray,
        *,
        with_overlays: bool = True,
        quality: int = 80,
        enhance: bool = False,
    ) -> bytes | None:
        """Codifica um quadro BGR em JPEG aplicando nitidez e overlays opcionais."""
        if frame is None or frame.size == 0:
            return None
        render = frame
        if enhance:
            render = self.enhancer.enhance_frame(render, upscale=False)
        if with_overlays and self.enable_ai and self.config.ai.show_overlays:
            dets = self.ai_pipeline.get_overlay_detections()
            if dets:
                render = draw_ai_overlays(render, dets)
        ok, encoded = cv2.imencode(
            ".jpg", render, [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, int(quality)))]
        )
        return encoded.tobytes() if ok else None

    def get_latest_jpeg(
        self,
        with_overlays: bool = True,
        quality: int = 80,
        enhance: bool = False,
    ) -> bytes | None:
        """Codifica em JPEG o quadro mais recente (com cache por sequência para múltiplos clientes)."""
        with self._frame_cond:
            seq = self._frame_seq
            frame = self._latest_frame
            cached_key = self._cached_jpeg_key
            cached_bytes = self._cached_jpeg_bytes

        if frame is None:
            return None

        key = (seq, bool(with_overlays), bool(enhance), int(quality))
        if cached_key == key and cached_bytes is not None:
            return cached_bytes

        jpg_bytes = self.encode_frame_jpeg(
            frame, with_overlays=with_overlays, quality=quality, enhance=enhance
        )
        if jpg_bytes is None:
            return None
        with self._frame_cond:
            if self._frame_seq == seq:
                self._cached_jpeg_key = key
                self._cached_jpeg_bytes = jpg_bytes
        return jpg_bytes

    def wait_next_frame(self, last_seq: int, timeout: float = 2.0) -> tuple[int, np.ndarray | None]:
        """Aguarda um novo frame bruto além de `last_seq` e retorna `(novo_seq, frame_bgr)`."""
        with self._frame_cond:
            if self._frame_seq == last_seq and not self._stop_event.is_set():
                self._frame_cond.wait(timeout=max(0.1, float(timeout)))
            return (self._frame_seq, self._latest_frame)

    def wait_next_jpeg(
        self,
        last_seq: int,
        timeout: float = 2.0,
        with_overlays: bool = True,
        enhance: bool = False,
    ) -> tuple[int, bytes | None]:
        """Aguarda um novo frame além de `last_seq` e retorna `(novo_seq, jpeg_bytes)`."""
        with self._frame_cond:
            if self._frame_seq == last_seq and not self._stop_event.is_set():
                self._frame_cond.wait(timeout=max(0.1, float(timeout)))
            seq = self._frame_seq
        return (seq, self.get_latest_jpeg(with_overlays=with_overlays, enhance=enhance))

    def get_status_payload(self) -> dict[str, object]:
        """Retorna estado completo do motor 24/7 para a API REST / Portal Web."""
        rec_file, rec_bytes, rec_elapsed, rec_seg_secs = self.recorder.get_active_recording_stats()
        summary = self.ai_pipeline.get_summary()
        cam = self.selected_camera
        with self._state_lock:
            uptime = (time.monotonic() - self._started_at) if self._started_at is not None else 0.0
            connected = self._camera_connected
            conn_state = self._connection_state
            fps = self._current_fps
            logs = list(self._logs)[-80:]
        with self._frame_cond:
            frame_seq = self._frame_seq
            last_frame_wall = self._last_frame_wall
            motion_now = self._motion_now
        with self._ptz_lock:
            ptz_available = self._ptz_client is not None
            ptz_backend = self._ptz_backend_label

        return {
            "service_running": self.is_running,
            "uptime_sec": round(uptime, 1),
            "platform": sys.platform,
            "power_guard_active": self.power_guard.is_active,
            "camera": {
                "name": cam.name,
                "brand": cam.brand,
                "connected": connected,
                "state": conn_state,
                "fps": round(fps, 1),
                "frame_count": frame_seq,
                "last_frame_at": last_frame_wall,
                "motion_now": motion_now,
            },
            "recording": {
                "active": self.recorder.is_recording,
                "current_file": rec_file,
                "size_bytes": rec_bytes,
                "elapsed_sec": round(rec_elapsed, 1),
                "segment_sec": rec_seg_secs,
                "root_dir": str(self.recordings_root),
            },
            "drive": {
                "enabled": self.config.drive.enabled,
                "sync_mode": self.config.drive.sync_mode,
                "ready": self.uploader.is_ready(),
                "account": self.uploader.cached_account_email,
                "pending_count": self.uploader.get_pending_count(),
                "delete_after_upload": self.config.drive.delete_after_upload,
            },
            "ai": {
                "enabled": self.enable_ai,
                "ready": self.ai_pipeline.is_ready,
                "loading": self.ai_pipeline.is_loading,
                "summary": asdict(summary),
            },
            "ptz": {
                "available": ptz_available,
                "backend": ptz_backend,
            },
            "logs": logs,
        }

    def get_vehicles_payload(self, limit: int = 200) -> list[dict[str, object]]:
        records = self.ai_pipeline.get_vehicle_records()[: max(1, min(1000, int(limit)))]
        return [
            {
                "track_id": int(r.track_id),
                "timestamp": r.timestamp,
                "vehicle_type": r.vehicle_type,
                "plate": r.plate,
                "plate_conf": round(float(r.plate_conf), 3),
                "color": r.color,
                "approx_model": r.approx_model,
                "model_conf": round(float(r.model_conf), 3),
                "camera_name": r.camera_name,
            }
            for r in records
        ]

    def get_people_payload(self, limit: int = 200) -> list[dict[str, object]]:
        records = self.ai_pipeline.get_person_records()[: max(1, min(1000, int(limit)))]
        return [
            {
                "track_id": int(r.track_id),
                "timestamp": r.timestamp,
                "person_type": r.person_type,
                "clothing_colors": r.clothing_colors,
                "upper_color": r.upper_color,
                "lower_color": r.lower_color,
                "gender_conf": round(float(r.gender_conf), 3),
                "camera_name": r.camera_name,
            }
            for r in records
        ]

    def get_detections_payload(self) -> dict[str, object]:
        """Retorna as detecções ativas com coordenadas normalizadas (0..1) para overlay HTML."""
        with self._frame_cond:
            frame = self._latest_frame
        if frame is not None and len(frame.shape) >= 2:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        else:
            frame_w, frame_h = 1280, 720
        dets = self.ai_pipeline.get_overlay_detections() if self.enable_ai else []
        inv_w = 1.0 / max(1, frame_w)
        inv_h = 1.0 / max(1, frame_h)
        items = []
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            nx = max(0.0, min(1.0, x1 * inv_w))
            ny = max(0.0, min(1.0, y1 * inv_h))
            nw = max(0.005, min(1.0 - nx, (x2 - x1) * inv_w))
            nh = max(0.005, min(1.0 - ny, (y2 - y1) * inv_h))
            items.append(
                {
                    "track_id": int(det.track_id),
                    "category": det.category,
                    "label": det.label,
                    "x": round(nx, 5),
                    "y": round(ny, 5),
                    "w": round(nw, 5),
                    "h": round(nh, 5),
                    "confidence": round(float(det.confidence), 3),
                    "plate": det.plate,
                    "color": det.color,
                    "approx_model": det.approx_model,
                }
            )
        return {
            "frame_width": frame_w,
            "frame_height": frame_h,
            "show_overlays": bool(self.config.ai.show_overlays),
            "detections": items,
        }

    def get_timeline_dates(self, camera_name: str | None = None, include_cloud: bool = True) -> list[str]:
        cam = camera_name or self.selected_camera.name
        return DVRTimelineIndex.discover_dates(
            self.recordings_root,
            cam,
            uploader=self.uploader if include_cloud else None,
            include_cloud=bool(include_cloud),
        )

    def get_day_timeline(
        self,
        date_str: str,
        camera_name: str | None = None,
        include_cloud: bool = True,
    ) -> dict[str, object]:
        cam = camera_name or self.selected_camera.name
        if not _DATE_RE.match(date_str):
            raise ValueError(f"Data inválida (esperado AAAA-MM-DD): {date_str}")

        segs, events = DVRTimelineIndex.build_day_timeline(
            self.recordings_root,
            cam,
            date_str,
            event_store=self.dvr_events,
            csv_path=self.config.ai.csv_log_path if self.config.ai.save_csv_log else None,
            uploader=self.uploader if include_cloud else None,
            include_cloud=bool(include_cloud),
            fallback_segment_sec=max(60.0, float(self.config.recording.segment_minutes) * 60.0),
        )

        def _seg_dict(seg: TimelineSegment) -> dict[str, object]:
            return {
                "filename": seg.filename,
                "camera_name": seg.camera_name,
                "date_str": seg.date_str,
                "source_kind": seg.source_kind,
                "drive_file_id": seg.drive_file_id,
                "start_sec": round(seg.start_sec, 2),
                "end_sec": round(seg.end_sec, 2),
                "duration_sec": round(seg.duration_sec, 2),
                "fps": round(seg.fps, 2),
                "frame_count": seg.frame_count,
                "is_motion_clip": seg.is_motion_clip,
            }

        def _event_dict(ev: TimelineEvent) -> dict[str, object]:
            return {
                "date_str": ev.date_str,
                "sec_of_day": round(ev.sec_of_day, 2),
                "end_sec": round(ev.end_sec, 2),
                "event_type": ev.event_type,
                "event_label": EVENT_TYPE_LABELS.get(ev.event_type, ev.event_type),
                "label": ev.label,
            }

        return {
            "camera_name": cam,
            "date_str": date_str,
            "segments": [_seg_dict(s) for s in segs],
            "events": [_event_dict(e) for e in events],
        }

    def resolve_video_path(
        self,
        date_str: str,
        filename: str,
        camera_name: str | None = None,
        drive_file_id: str = "",
    ) -> Path:
        """Resolve localmente (ou baixa sob demanda do Drive para cache) um segmento MP4 seguro."""
        if not _DATE_RE.match(date_str):
            raise ValueError("Parâmetro 'date' inválido.")
        clean_name = Path(filename).name
        if not clean_name or clean_name != filename or not clean_name.lower().endswith(".mp4"):
            raise ValueError("Nome de arquivo de vídeo inválido.")

        cam = camera_name or self.selected_camera.name
        cam_part = sanitize_filename_part(cam, "Camera")

        local_candidate = self.recordings_root / cam_part / date_str / clean_name
        if local_candidate.exists() and local_candidate.is_file() and is_valid_mp4_file(local_candidate):
            return local_candidate

        if self.config.drive.sync_mode == "folder" and self.config.drive.local_drive_dir:
            folder_candidate = (
                Path(self.config.drive.local_drive_dir)
                / sanitize_filename_part(self.config.drive.root_folder_name, "HomeCamSeguranca")
                / cam_part
                / date_str
                / clean_name
            )
            if folder_candidate.exists() and folder_candidate.is_file() and is_valid_mp4_file(folder_candidate):
                return folder_candidate

        cache_candidate = DVRTimelineIndex.playback_cache_dir(self.recordings_root) / cam_part / date_str / clean_name
        if cache_candidate.exists() and cache_candidate.is_file() and is_valid_mp4_file(cache_candidate):
            return cache_candidate

        file_id = drive_file_id.strip()
        if not file_id:
            for item in self.uploader.list_remote_recordings(cam, date_str):
                if str(item.get("name") or "") == clean_name:
                    file_id = str(item.get("id") or "").strip()
                    break

        if not file_id:
            raise FileNotFoundError(f"Gravação não encontrada localmente nem no Drive: {clean_name}")

        lock_key = f"{cam_part}/{date_str}/{clean_name}"
        with self._download_lock:
            per_file_lock = self._file_download_locks.setdefault(lock_key, threading.Lock())
        with per_file_lock:
            if cache_candidate.exists() and is_valid_mp4_file(cache_candidate):
                return cache_candidate
            return self.uploader.download_remote_file(file_id, cache_candidate)

    def export_interpolated_video(
        self,
        date_str: str,
        filename: str,
        *,
        camera_name: str | None = None,
        drive_file_id: str = "",
        target_fps: int = 30,
        enhance: bool = False,
    ) -> Path:
        """Gera versão interpolada (>= 30 FPS constantes) e opcionalmente restaurada de um clipe."""
        src = self.resolve_video_path(
            date_str, filename, camera_name=camera_name, drive_file_id=drive_file_id
        )
        fps = max(30, min(60, int(target_fps)))
        cam = camera_name or self.selected_camera.name
        cam_part = sanitize_filename_part(cam, "Camera")
        suffix = f"_{fps}fps{'_enh' if enhance else ''}.mp4"
        out_path = (
            DVRTimelineIndex.playback_cache_dir(self.recordings_root)
            / "_export_30fps"
            / cam_part
            / date_str
            / f"{src.stem}{suffix}"
        )
        lock_key = f"export/{cam_part}/{date_str}/{out_path.name}"
        with self._download_lock:
            per_file_lock = self._file_download_locks.setdefault(lock_key, threading.Lock())
        with per_file_lock:
            if out_path.exists() and is_valid_mp4_file(out_path):
                return out_path
            interpolate_video_ffmpeg(src, out_path, target_fps=fps, enhance=enhance)
        return out_path

    def ptz_move(self, direction: str, duration_sec: float = 0.35, speed: int | None = None) -> tuple[bool, str]:
        """Executa um passo direcional PTZ seguro remotamente pela API Web."""
        clean_dir = direction.strip().lower()
        if clean_dir not in _VALID_PTZ_DIRS:
            return (False, f"Direção PTZ inválida: {direction}")
        with self._ptz_lock:
            client = self._ptz_client
        if client is None:
            return (False, "PTZ não disponível ou câmera ainda desconectada.")
        spd = int(speed or self.selected_camera.ptz_speed or 5)
        try:
            ok = bool(client.move_start(clean_dir, speed=spd))
            time.sleep(max(0.08, min(1.5, float(duration_sec))))
            client.move_stop(clean_dir)
            return (ok, f"Movimento PTZ '{clean_dir}' enviado." if ok else "Câmera recusou comando PTZ.")
        except Exception as err:
            return (False, f"Erro PTZ: {err}")

    def ptz_home(self) -> tuple[bool, str]:
        """Retorna a câmera PTZ ao preset inicial configurado."""
        with self._ptz_lock:
            client = self._ptz_client
        if client is None:
            return (False, "PTZ não disponível.")
        preset_id = int(self.selected_camera.ptz_home_preset or 100)
        try:
            ok = bool(client.goto_preset(preset_id))
            return (ok, f"Retornando ao Ponto Fixo #{preset_id}." if ok else "Falha ao chamar Ponto Fixo.")
        except Exception as err:
            return (False, f"Erro PTZ Home: {err}")

    def trigger_drive_sync(self) -> int:
        self.uploader.trigger_sync()
        return self.uploader.get_pending_count()

