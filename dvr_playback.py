"""Motor profissional de reprodução DVR/NVR, linha do tempo 24h, eventos e zoom digital."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import queue
import re
import struct
import threading
import time

import cv2
import numpy as np

from config_manager import sanitize_filename_part
from recorder import is_valid_mp4_file

SUPPORTED_SPEEDS: tuple[float, ...] = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)

TIMELINE_WINDOW_PRESETS_SEC: dict[str, float] = {
    "24h": 86400.0,
    "6h": 21600.0,
    "2h": 7200.0,
    "1h": 3600.0,
    "15m": 900.0,
}

TIMELINE_COLORS: dict[str, str] = {
    "continuous": "#2563eb",   # Azul: gravação contínua local
    "drive_cloud": "#0891b2",  # Ciano: gravação no Google Drive
    "motion": "#f59e0b",       # Âmbar: detecção de movimento
    "ai_person": "#22c55e",    # Verde: IA Pessoa
    "ai_animal": "#a855f7",    # Roxo: IA Animal
    "ai_vehicle": "#ef4444",   # Vermelho: IA Veículo/Moto
    "ai_plate": "#f43f5e",     # Coral: Placa ALPR reconhecida
}

EVENT_TYPE_LABELS: dict[str, str] = {
    "motion": "Movimento",
    "ai_person": "IA: Pessoa",
    "ai_animal": "IA: Animal",
    "ai_vehicle": "IA: Veículo",
    "ai_plate": "IA: Placa (ALPR)",
}

_FILENAME_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(?:\.[^.]+)?$")
_DATE_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SKIPPED_DIR_NAMES = frozenset({"_corrompidos", "_playback_cache", "_timeline_events"})

_META_CACHE_LOCK = threading.Lock()
_META_CACHE: dict[tuple[str, int, int], tuple[float, float, int]] = {}


@dataclass
class TimelineSegment:
    """Bloco contínuo de vídeo gravado posicionado na linha do tempo de 24 horas."""

    segment_id: str
    camera_name: str
    date_str: str
    filename: str
    source_kind: str  # "local" | "drive_folder" | "drive_cloud"
    local_path: Path | None
    drive_file_id: str = ""
    file_size: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    fps: float = 15.0
    frame_count: int = 0
    is_motion_clip: bool = False


@dataclass
class TimelineEvent:
    """Marcador de evento (movimento ou alerta de IA) sobre a linha do tempo."""

    camera_name: str
    date_str: str
    sec_of_day: float
    end_sec: float
    event_type: str  # "motion" | "ai_person" | "ai_animal" | "ai_vehicle" | "ai_plate"
    label: str
    details: str = ""

    @property
    def color(self) -> str:
        return TIMELINE_COLORS.get(self.event_type, "#e2e8f0")


@dataclass
class PlaybackFrameState:
    """Estado entregue à UI sempre que um quadro de reprodução é atualizado."""

    frame_bgr: np.ndarray | None
    day_sec: float
    date_str: str
    camera_name: str
    filename: str
    frame_index: int
    total_frames: int
    fps: float
    speed: float
    playing: bool
    source_kind: str


def format_day_seconds(sec: float) -> str:
    """Formata segundos desde `00:00:00` no padrão `HH:MM:SS`."""
    clamped = max(0, min(86399, int(round(sec))))
    hh = clamped // 3600
    mm = (clamped % 3600) // 60
    ss = clamped % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def datetime_to_day_seconds(dt: datetime) -> float:
    """Converte um `datetime` para segundos decorridos no respectivo dia."""
    return float(dt.hour * 3600 + dt.minute * 60 + dt.second) + (dt.microsecond / 1_000_000.0)


def parse_segment_start_datetime(filename: str, fallback_mtime: float | None = None) -> datetime | None:
    """Extrai data/hora inicial do padrão `<Cam>_YYYY-MM-DD_HH-MM-SS.mp4` ou do mtime."""
    match = _FILENAME_TS_RE.search(filename)
    if match:
        date_part, hh, mm, ss = match.groups()
        try:
            return datetime.strptime(f"{date_part} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    if fallback_mtime is not None:
        try:
            return datetime.fromtimestamp(fallback_mtime)
        except (OSError, OverflowError, ValueError):
            return None
    return None


def _read_mp4_mvhd_duration(path: Path) -> float:
    """Lê a duração do átomo ISO BMFF `moov/mvhd` como fallback leve sem decodificar quadros."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fh:
            pos = 0
            while pos + 8 <= file_size:
                fh.seek(pos)
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                bsize, btype = struct.unpack(">I4s", hdr)
                if bsize < 8:
                    break
                if btype == b"moov":
                    moov_end = min(file_size, pos + bsize)
                    sub = pos + 8
                    while sub + 8 <= moov_end:
                        fh.seek(sub)
                        shdr = fh.read(8)
                        if len(shdr) < 8:
                            break
                        ssize, stype = struct.unpack(">I4s", shdr)
                        if ssize < 8:
                            break
                        if stype == b"mvhd" and ssize >= 32:
                            version = fh.read(1)[0]
                            fh.read(3)  # flags
                            if version == 1 and ssize >= 40:
                                _ct, _mt, timescale, duration = struct.unpack(">QQIQ", fh.read(28))
                            else:
                                _ct, _mt, timescale, duration = struct.unpack(">IIII", fh.read(16))
                            if timescale > 0 and duration > 0:
                                return float(duration) / float(timescale)
                            return 0.0
                        sub += ssize
                    break
                pos += bsize
    except OSError:
        return 0.0
    return 0.0


def probe_mp4_metadata(path: Path, fallback_start: datetime | None = None) -> tuple[float, float, int]:
    """Retorna `(duracao_seg, fps, total_quadros)` com cache por assinatura de arquivo."""
    target = Path(path)
    try:
        st = target.stat()
    except OSError:
        return (0.0, 15.0, 0)

    if st.st_size <= 64 or not is_valid_mp4_file(target):
        return (0.0, 15.0, 0)

    cache_key = (str(target.resolve()), int(st.st_size), int(st.st_mtime_ns))
    with _META_CACHE_LOCK:
        cached = _META_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fps = 15.0
    frame_count = 0
    cap = cv2.VideoCapture(str(target), cv2.CAP_FFMPEG)
    try:
        if cap.isOpened():
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 2.0 <= raw_fps <= 120.0:
                fps = raw_fps
            raw_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if raw_fc > 0:
                frame_count = raw_fc
    finally:
        cap.release()

    duration = float(frame_count) / fps if frame_count > 0 else 0.0
    if duration <= 0.0:
        duration = _read_mp4_mvhd_duration(target)
    if duration <= 0.0 and fallback_start is not None:
        duration = max(1.0, min(7200.0, st.st_mtime - fallback_start.timestamp()))
    if duration <= 0.0:
        duration = max(1.0, float(st.st_size) / 90_000.0)
    if frame_count <= 0:
        frame_count = max(1, int(round(duration * fps)))

    result = (duration, fps, frame_count)
    with _META_CACHE_LOCK:
        if len(_META_CACHE) > 2000:
            _META_CACHE.clear()
        _META_CACHE[cache_key] = result
    return result


class DVREventStore:
    """Armazena e carrega marcações leves de movimento e alertas IA por câmera e data."""

    def __init__(self, events_dir: Path | str) -> None:
        self.events_dir = Path(events_dir)
        self._lock = threading.Lock()
        self._last_recorded: dict[tuple[str, str], tuple[float, str]] = {}

    def _file_for(self, camera_name: str, date_str: str) -> Path:
        cam_part = sanitize_filename_part(camera_name, "Camera")
        return self.events_dir / f"{date_str}_{cam_part}.jsonl"

    def record_event(
        self,
        camera_name: str,
        event_type: str,
        label: str,
        *,
        when: datetime | None = None,
        min_interval_sec: float = 3.0,
        dedupe_key: str = "",
    ) -> bool:
        """Grava um evento na linha do tempo respeitando intervalo mínimo anti-ruído."""
        dt = when or datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        sec = datetime_to_day_seconds(dt)
        now_mono = time.monotonic()

        throttle_key = (sanitize_filename_part(camera_name, "Camera"), event_type)
        with self._lock:
            last = self._last_recorded.get(throttle_key)
            if last is not None:
                last_ts, last_dedupe = last
                if dedupe_key and dedupe_key != last_dedupe:
                    pass
                elif (now_mono - last_ts) < min_interval_sec:
                    return False
            self._last_recorded[throttle_key] = (now_mono, dedupe_key)

            payload = {
                "camera": camera_name,
                "date": date_str,
                "sec": round(sec, 2),
                "type": event_type,
                "label": label,
            }
            path = self._file_for(camera_name, date_str)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                return True
            except OSError:
                return False

    def load_events(
        self,
        camera_name: str,
        date_str: str,
        csv_path: Path | str | None = None,
    ) -> list[TimelineEvent]:
        """Carrega eventos do dia combinando JSONL em tempo real e tabela ALPR CSV."""
        events: list[TimelineEvent] = []
        target_cam_norm = sanitize_filename_part(camera_name, "Camera").casefold()

        jsonl_path = self._file_for(camera_name, date_str)
        if jsonl_path.exists():
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sec = max(0.0, min(86399.0, float(raw.get("sec") or 0.0)))
                    etype = str(raw.get("type") or "motion")
                    lbl = str(raw.get("label") or EVENT_TYPE_LABELS.get(etype, etype))
                    events.append(
                        TimelineEvent(
                            camera_name=camera_name,
                            date_str=date_str,
                            sec_of_day=sec,
                            end_sec=min(86400.0, sec + 4.0),
                            event_type=etype,
                            label=lbl,
                        )
                    )
            except OSError:
                pass

        if csv_path is not None:
            csv_file = Path(csv_path)
            if csv_file.exists() and csv_file.is_file():
                try:
                    with csv_file.open("r", encoding="utf-8-sig", newline="") as fh:
                        reader = csv.DictReader(fh, delimiter=";")
                        for row in reader:
                            ts = (row.get("Timestamp") or "").strip()
                            if not ts.startswith(date_str):
                                continue
                            row_cam = (row.get("Câmera") or "").strip()
                            if row_cam and sanitize_filename_part(row_cam, "Camera").casefold() != target_cam_norm:
                                continue
                            try:
                                dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                continue
                            vtype = (row.get("Tipo de Veículo") or "Veículo").strip()
                            plate = (row.get("Placa") or "").strip()
                            color = (row.get("Cor") or "").strip()
                            model = (row.get("Modelo Aproximado") or "").strip()
                            has_plate = bool(plate and plate != "Lendo...")
                            etype = "ai_plate" if has_plate else "ai_vehicle"
                            parts = [vtype]
                            if has_plate:
                                parts.append(f"Placa {plate}")
                            if color and color != "Indefinida":
                                parts.append(color)
                            if model and model != "Analisando...":
                                parts.append(model.split(" — ")[0])
                            sec = datetime_to_day_seconds(dt)
                            events.append(
                                TimelineEvent(
                                    camera_name=camera_name,
                                    date_str=date_str,
                                    sec_of_day=sec,
                                    end_sec=min(86400.0, sec + 5.0),
                                    event_type=etype,
                                    label=" | ".join(parts),
                                )
                            )
                except OSError:
                    pass

        events.sort(key=lambda ev: ev.sec_of_day)
        return events


class DVRTimelineIndex:
    """Indexa segmentos gravados (locais, Drive Desktop e Google Drive nuvem) em blocos de 24h."""

    @staticmethod
    def playback_cache_dir(recordings_dir: Path | str) -> Path:
        return Path(recordings_dir) / "_playback_cache"

    @classmethod
    def discover_dates(
        cls,
        recordings_dir: Path | str,
        camera_name: str,
        uploader: object | None = None,
        include_cloud: bool = False,
    ) -> list[str]:
        """Retorna todas as datas (`YYYY-MM-DD`) com gravações ou eventos da câmera."""
        rec_root = Path(recordings_dir)
        cam_part = sanitize_filename_part(camera_name, "Camera")
        dates: set[str] = set()

        cam_dir = rec_root / cam_part
        if cam_dir.exists() and cam_dir.is_dir():
            for child in cam_dir.iterdir():
                if child.is_dir() and _DATE_STR_RE.match(child.name):
                    if any(p.is_file() and p.suffix.lower() == ".mp4" for p in child.iterdir()):
                        dates.add(child.name)

        events_dir = rec_root / "_timeline_events"
        if events_dir.exists() and events_dir.is_dir():
            suffix = f"_{cam_part}.jsonl"
            for child in events_dir.iterdir():
                if child.is_file() and child.name.endswith(suffix) and len(child.name) >= 10:
                    dpart = child.name[:10]
                    if _DATE_STR_RE.match(dpart):
                        dates.add(dpart)

        if include_cloud and uploader is not None:
            list_dates_fn = getattr(uploader, "list_remote_dates", None)
            if callable(list_dates_fn):
                for d in list_dates_fn(camera_name):
                    if _DATE_STR_RE.match(str(d)):
                        dates.add(str(d))

        return sorted(dates, reverse=True)

    @classmethod
    def build_day_timeline(
        cls,
        recordings_dir: Path | str,
        camera_name: str,
        date_str: str,
        *,
        event_store: DVREventStore | None = None,
        csv_path: Path | str | None = None,
        uploader: object | None = None,
        include_cloud: bool = False,
        fallback_segment_sec: float = 300.0,
    ) -> tuple[list[TimelineSegment], list[TimelineEvent]]:
        """Constrói segmentos contínuos e eventos cronológicos para uma data (`YYYY-MM-DD`)."""
        rec_root = Path(recordings_dir)
        cam_part = sanitize_filename_part(camera_name, "Camera")
        cache_day_dir = cls.playback_cache_dir(rec_root) / cam_part / date_str

        segments_by_name: dict[str, TimelineSegment] = {}

        local_day_dir = rec_root / cam_part / date_str
        if local_day_dir.exists() and local_day_dir.is_dir():
            for mp4 in sorted(local_day_dir.iterdir(), key=lambda p: p.name):
                if any(part in _SKIPPED_DIR_NAMES for part in mp4.parts):
                    continue
                if not mp4.is_file() or mp4.suffix.lower() != ".mp4":
                    continue
                seg = cls._segment_from_local_file(mp4, camera_name, date_str, source_kind="local")
                if seg is not None:
                    segments_by_name[mp4.name] = seg

        if include_cloud and uploader is not None:
            list_rec_fn = getattr(uploader, "list_remote_recordings", None)
            if callable(list_rec_fn):
                for entry in list_rec_fn(camera_name, date_str):
                    fname = str(entry.get("name") or "")
                    if not fname or fname in segments_by_name:
                        continue
                    drive_local = entry.get("local_path")
                    if drive_local:
                        seg = cls._segment_from_local_file(
                            Path(str(drive_local)),
                            camera_name,
                            date_str,
                            source_kind="drive_folder",
                        )
                        if seg is not None:
                            segments_by_name[fname] = seg
                        continue

                    cached_mp4 = cache_day_dir / fname
                    if cached_mp4.exists() and cached_mp4.is_file() and is_valid_mp4_file(cached_mp4):
                        seg = cls._segment_from_local_file(
                            cached_mp4,
                            camera_name,
                            date_str,
                            source_kind="drive_cloud",
                        )
                        if seg is not None:
                            seg.drive_file_id = str(entry.get("id") or "")
                            segments_by_name[fname] = seg
                        continue

                    start_dt = parse_segment_start_datetime(fname)
                    if start_dt is None:
                        mod_iso = str(entry.get("modified_time") or "")
                        if mod_iso:
                            try:
                                start_dt = datetime.fromisoformat(mod_iso.replace("Z", "+00:00")).astimezone()
                            except ValueError:
                                start_dt = None
                    if start_dt is None:
                        continue
                    fsize = int(entry.get("size") or 0)
                    est_dur = max(12.0, min(fallback_segment_sec, float(fsize) / 85_000.0)) if fsize > 0 else fallback_segment_sec
                    start_sec = datetime_to_day_seconds(start_dt)
                    end_sec = min(86400.0, start_sec + est_dur)
                    segments_by_name[fname] = TimelineSegment(
                        segment_id=f"cloud:{entry.get('id') or fname}",
                        camera_name=camera_name,
                        date_str=date_str,
                        filename=fname,
                        source_kind="drive_cloud",
                        local_path=None,
                        drive_file_id=str(entry.get("id") or ""),
                        file_size=fsize,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=max(1.0, end_sec - start_sec),
                        fps=15.0,
                        frame_count=max(1, int(est_dur * 15)),
                        is_motion_clip="_mov_" in fname.lower(),
                    )

        segments = sorted(segments_by_name.values(), key=lambda s: (s.start_sec, s.filename))
        for idx in range(len(segments) - 1):
            if segments[idx].end_sec > segments[idx + 1].start_sec:
                segments[idx].end_sec = max(segments[idx].start_sec + 0.5, segments[idx + 1].start_sec)

        events: list[TimelineEvent] = []
        if event_store is not None:
            events = event_store.load_events(camera_name, date_str, csv_path=csv_path)

        # Marca clipes `_mov_` como eventos de movimento na timeline caso não haja JSONL naquele instante
        for seg in segments:
            if not seg.is_motion_clip:
                continue
            has_near = any(
                ev.event_type == "motion" and abs(ev.sec_of_day - seg.start_sec) <= 5.0
                for ev in events
            )
            if not has_near:
                events.append(
                    TimelineEvent(
                        camera_name=camera_name,
                        date_str=date_str,
                        sec_of_day=seg.start_sec,
                        end_sec=seg.end_sec,
                        event_type="motion",
                        label=f"Clipe por Movimento ({seg.filename})",
                    )
                )

        events.sort(key=lambda ev: ev.sec_of_day)
        return segments, events

    @staticmethod
    def _segment_from_local_file(
        path: Path,
        camera_name: str,
        date_str: str,
        source_kind: str,
    ) -> TimelineSegment | None:
        try:
            st = path.stat()
        except OSError:
            return None
        start_dt = parse_segment_start_datetime(path.name, fallback_mtime=st.st_mtime)
        if start_dt is None:
            return None
        duration, fps, frame_count = probe_mp4_metadata(path, fallback_start=start_dt)
        if duration <= 0.0:
            return None
        start_sec = datetime_to_day_seconds(start_dt)
        end_sec = min(86400.0, start_sec + duration)
        return TimelineSegment(
            segment_id=str(path.resolve()),
            camera_name=camera_name,
            date_str=date_str,
            filename=path.name,
            source_kind=source_kind,
            local_path=path,
            file_size=int(st.st_size),
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration,
            fps=fps,
            frame_count=frame_count,
            is_motion_clip="_mov_" in path.name.lower(),
        )


class DigitalZoomViewport:
    """Controla zoom digital interativo ancorado no cursor, panorâmica e mini-mapa PiP."""

    ZOOM_MIN = 1.0
    ZOOM_MAX = 10.0

    def __init__(self) -> None:
        self.zoom: float = 1.0
        self.pan_cx: float = 0.5
        self.pan_cy: float = 0.5

    def reset(self) -> None:
        self.zoom = 1.0
        self.pan_cx = 0.5
        self.pan_cy = 0.5

    def _clamp_pan(self) -> None:
        half = 0.5 / self.zoom
        self.pan_cx = max(half, min(1.0 - half, self.pan_cx))
        self.pan_cy = max(half, min(1.0 - half, self.pan_cy))

    @staticmethod
    def _fitted_box(disp_w: int, disp_h: int, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
        if disp_w <= 0 or disp_h <= 0 or frame_w <= 0 or frame_h <= 0:
            return (0.0, 0.0, max(1.0, float(disp_w)), max(1.0, float(disp_h)))
        scale = min(disp_w / float(frame_w), disp_h / float(frame_h))
        fit_w = frame_w * scale
        fit_h = frame_h * scale
        ox = (disp_w - fit_w) * 0.5
        oy = (disp_h - fit_h) * 0.5
        return (ox, oy, fit_w, fit_h)

    def get_source_roi(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Retorna `(x1, y1, x2, y2)` na resolução original do frame para o zoom/pan atual."""
        self._clamp_pan()
        if self.zoom <= 1.001:
            return (0, 0, frame_w, frame_h)
        roi_w = max(16, int(round(frame_w / self.zoom)))
        roi_h = max(16, int(round(frame_h / self.zoom)))
        cx = self.pan_cx * frame_w
        cy = self.pan_cy * frame_h
        x1 = max(0, min(frame_w - roi_w, int(round(cx - roi_w * 0.5))))
        y1 = max(0, min(frame_h - roi_h, int(round(cy - roi_h * 0.5))))
        return (x1, y1, min(frame_w, x1 + roi_w), min(frame_h, y1 + roi_h))

    def set_zoom_at_point(
        self,
        new_zoom: float,
        cursor_x: float,
        cursor_y: float,
        disp_w: int,
        disp_h: int,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Ajusta o zoom mantendo exatamente o detalhe sob `(cursor_x, cursor_y)` estável."""
        clamped = max(self.ZOOM_MIN, min(self.ZOOM_MAX, float(new_zoom)))
        if abs(clamped - 1.0) < 0.03:
            self.reset()
            return

        ox, oy, fit_w, fit_h = self._fitted_box(disp_w, disp_h, frame_w, frame_h)
        nx = max(0.0, min(1.0, (cursor_x - ox) / max(1.0, fit_w)))
        ny = max(0.0, min(1.0, (cursor_y - oy) / max(1.0, fit_h)))

        old_w_norm = 1.0 / self.zoom
        old_x0 = self.pan_cx - 0.5 * old_w_norm
        old_y0 = self.pan_cy - 0.5 * old_w_norm
        target_x = old_x0 + nx * old_w_norm
        target_y = old_y0 + ny * old_w_norm

        self.zoom = clamped
        new_w_norm = 1.0 / self.zoom
        self.pan_cx = target_x - (nx - 0.5) * new_w_norm
        self.pan_cy = target_y - (ny - 0.5) * new_w_norm
        self._clamp_pan()

    def pan_by_pixels(
        self,
        dx_px: float,
        dy_px: float,
        disp_w: int,
        disp_h: int,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Desloca a panorâmica proporcionalmente ao arrasto do mouse na imagem ampliada."""
        if self.zoom <= 1.01:
            return
        _ox, _oy, fit_w, fit_h = self._fitted_box(disp_w, disp_h, frame_w, frame_h)
        self.pan_cx -= (dx_px / max(1.0, fit_w)) / self.zoom
        self.pan_cy -= (dy_px / max(1.0, fit_h)) / self.zoom
        self._clamp_pan()

    def render(
        self,
        frame_bgr: np.ndarray,
        target_w: int,
        target_h: int,
        *,
        draw_minimap: bool = True,
    ) -> np.ndarray:
        """Recorta a região de zoom diretamente na imagem original e renderiza o viewport + PiP."""
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.get_source_roi(fw, fh)
        crop = frame_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        scale = min(max(64, target_w) / float(max(1, cw)), max(48, target_h) / float(max(1, ch)))
        out_w = max(1, int(round(cw * scale)))
        out_h = max(1, int(round(ch * scale)))
        interp = cv2.INTER_AREA if (out_w < cw and out_h < ch) else cv2.INTER_LINEAR
        rendered = cv2.resize(crop, (out_w, out_h), interpolation=interp)

        if draw_minimap and self.zoom > 1.02 and out_w >= 260 and out_h >= 160:
            self._overlay_minimap_pip(rendered, frame_bgr, (x1, y1, x2, y2), fw, fh)
        return rendered

    def _overlay_minimap_pip(
        self,
        canvas_bgr: np.ndarray,
        full_bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        fw: int,
        fh: int,
    ) -> None:
        oh, ow = canvas_bgr.shape[:2]
        pip_w = max(96, min(170, ow // 5))
        pip_h = max(54, int(round(pip_w * fh / float(max(1, fw)))))
        if pip_h + 24 >= oh:
            return
        thumb = cv2.resize(full_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = roi
        rx1 = int(round((x1 / float(fw)) * (pip_w - 1)))
        ry1 = int(round((y1 / float(fh)) * (pip_h - 1)))
        rx2 = int(round((x2 / float(fw)) * (pip_w - 1)))
        ry2 = int(round((y2 / float(fh)) * (pip_h - 1)))
        cv2.rectangle(thumb, (rx1, ry1), (rx2, ry2), (255, 220, 40), 2)

        px = ow - pip_w - 10
        py = oh - pip_h - 10
        cv2.rectangle(canvas_bgr, (px - 2, py - 2), (px + pip_w + 2, py + pip_h + 2), (24, 32, 48), -1)
        canvas_bgr[py : py + pip_h, px : px + pip_w] = thumb

        badge = f"ZOOM {self.zoom:.1f}x"
        cv2.rectangle(canvas_bgr, (px - 2, py - 22), (px + 96, py - 2), (15, 23, 42), -1)
        cv2.putText(
            canvas_bgr,
            badge,
            (px + 4, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (103, 232, 249),
            1,
            cv2.LINE_AA,
        )


class DVRPlaybackEngine:
    """Reprodutor DVR multi-segmento assíncrono com velocidades ±1x..±16x, quadro a quadro e cache GOP."""

    def __init__(
        self,
        *,
        recordings_root: Path | str,
        uploader: object | None = None,
        on_frame: Callable[[PlaybackFrameState], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        max_cache_bytes: int = 160 * 1024 * 1024,
        backward_batch_size: int = 30,
    ) -> None:
        self.recordings_root = Path(recordings_root)
        self.uploader = uploader
        self.on_frame = on_frame
        self.on_state = on_state
        self.max_cache_bytes = max(32 * 1024 * 1024, int(max_cache_bytes))
        self.backward_batch_size = max(8, int(backward_batch_size))

        self._segments: list[TimelineSegment] = []
        self._events: list[TimelineEvent] = []
        self._seg_idx = -1
        self._frame_idx = 0
        self._frame_accum = 0.0
        self._speed = 1.0
        self._playing = False

        self._cap: cv2.VideoCapture | None = None
        self._cap_path: Path | None = None
        self._cap_pos = -1

        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._last_frame: np.ndarray | None = None
        self._state_lock = threading.Lock()

        self._cmd_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def is_playing(self) -> bool:
        with self._state_lock:
            return self._playing

    @property
    def speed(self) -> float:
        with self._state_lock:
            return self._speed

    def get_current_frame(self) -> np.ndarray | None:
        with self._state_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def set_recordings_root(self, root: Path | str) -> None:
        self.recordings_root = Path(root)

    def load_timeline(
        self,
        segments: Sequence[TimelineSegment],
        events: Sequence[TimelineEvent] = (),
        *,
        auto_seek_sec: float | None = None,
    ) -> None:
        self._cmd_queue.put(("load", (list(segments), list(events), auto_seek_sec)))

    def play(self) -> None:
        self._cmd_queue.put(("play", None))

    def pause(self) -> None:
        self._cmd_queue.put(("pause", None))

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        self._cmd_queue.put(("set_speed", float(speed)))

    def step_speed(self, direction: int) -> float:
        """Seleciona o próximo nível de velocidade para frente (`+1`) ou ré (`-1`)."""
        with self._state_lock:
            cur = self._speed
        if cur in SUPPORTED_SPEEDS:
            idx = SUPPORTED_SPEEDS.index(cur)
        else:
            idx = SUPPORTED_SPEEDS.index(1.0)
        new_idx = max(0, min(len(SUPPORTED_SPEEDS) - 1, idx + (1 if direction >= 0 else -1)))
        new_speed = SUPPORTED_SPEEDS[new_idx]
        self.set_speed(new_speed)
        return new_speed

    def seek_to_day_second(self, sec_of_day: float) -> None:
        self._cmd_queue.put(("seek_sec", float(sec_of_day)))

    def step_frame(self, delta: int) -> None:
        self._cmd_queue.put(("step_frame", int(delta)))

    def jump_to_adjacent_event(self, direction: int, allowed_types: set[str] | None = None) -> None:
        self._cmd_queue.put(("jump_event", (1 if direction >= 0 else -1, allowed_types)))

    def stop(self) -> None:
        self._stop.set()
        self._cmd_queue.put(("quit", None))
        if self._worker.is_alive():
            self._worker.join(timeout=2.5)
        self._close_capture()
        self._clear_cache()

    def _emit_state(self, message: str) -> None:
        if self.on_state:
            self.on_state(message)

    def _emit_frame(self, frame: np.ndarray | None, seg: TimelineSegment | None, frame_idx: int) -> None:
        with self._state_lock:
            self._last_frame = None if frame is None else frame.copy()
        if seg is None:
            day_sec = 0.0
            date_str = ""
            cam_name = ""
            fname = ""
            total_frames = 0
            fps = 15.0
            kind = ""
        else:
            fps = max(1.0, seg.fps)
            total_frames = max(1, seg.frame_count)
            day_sec = min(seg.end_sec, seg.start_sec + (frame_idx / fps))
            date_str = seg.date_str
            cam_name = seg.camera_name
            fname = seg.filename
            kind = seg.source_kind

        if self.on_frame:
            self.on_frame(
                PlaybackFrameState(
                    frame_bgr=frame,
                    day_sec=day_sec,
                    date_str=date_str,
                    camera_name=cam_name,
                    filename=fname,
                    frame_index=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    speed=self._speed,
                    playing=self._playing,
                    source_kind=kind,
                )
            )

    def _clear_cache(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0

    def _put_cache(self, path_str: str, idx: int, frame: np.ndarray) -> None:
        key = (path_str, idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        nbytes = int(frame.nbytes)
        self._cache[key] = frame
        self._cache_bytes += nbytes
        while self._cache_bytes > self.max_cache_bytes and len(self._cache) > 1:
            _old_key, old_frame = self._cache.popitem(last=False)
            self._cache_bytes -= int(old_frame.nbytes)

    def _close_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._cap_path = None
            self._cap_pos = -1

    def _open_capture(self, path: Path) -> cv2.VideoCapture | None:
        resolved = path.resolve()
        if self._cap is not None and self._cap_path == resolved and self._cap.isOpened():
            return self._cap
        self._close_capture()
        cap = cv2.VideoCapture(str(resolved), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        self._cap = cap
        self._cap_path = resolved
        self._cap_pos = 0
        return cap

    def _ensure_local_segment(self, seg: TimelineSegment) -> Path | None:
        if seg.local_path is not None and seg.local_path.exists():
            return seg.local_path
        if seg.source_kind != "drive_cloud" or not seg.drive_file_id or self.uploader is None:
            return None

        dl_fn = getattr(self.uploader, "download_remote_file", None)
        if not callable(dl_fn):
            return None

        cam_part = sanitize_filename_part(seg.camera_name, "Camera")
        dest = DVRTimelineIndex.playback_cache_dir(self.recordings_root) / cam_part / seg.date_str / seg.filename
        self._emit_state(f"Baixando clipe do Google Drive: {seg.filename}...")
        try:
            downloaded = dl_fn(seg.drive_file_id, dest)
            local_p = Path(downloaded)
            dur, fps, fc = probe_mp4_metadata(local_p)
            seg.local_path = local_p
            if dur > 0:
                seg.duration_sec = dur
                seg.end_sec = min(86400.0, seg.start_sec + dur)
                seg.fps = fps
                seg.frame_count = fc
            self._emit_state(f"Pronto (cache Google Drive): {seg.filename}")
            return local_p
        except Exception as err:
            self._emit_state(f"Falha ao baixar {seg.filename} do Drive: {err}")
            return None

    def _read_exact_frame(
        self,
        seg: TimelineSegment,
        frame_idx: int,
        *,
        fill_back_batch: bool = False,
    ) -> np.ndarray | None:
        local_p = self._ensure_local_segment(seg)
        if local_p is None:
            return None

        p_str = str(local_p.resolve())
        clamped_idx = max(0, min(max(0, seg.frame_count - 1), int(frame_idx)))
        cached = self._cache.get((p_str, clamped_idx))
        if cached is not None:
            self._cache.move_to_end((p_str, clamped_idx))
            return cached

        cap = self._open_capture(local_p)
        if cap is None:
            return None

        # Avanço sequencial ou pequeno salto usa `grab()` sem buscar keyframe do início do GOP
        if self._cap_pos == clamped_idx:
            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap_pos = clamped_idx + 1
                self._put_cache(p_str, clamped_idx, frame)
                return frame
        elif 0 < (clamped_idx - self._cap_pos) <= 4:
            cur = self._cap_pos
            while cur < clamped_idx:
                if not cap.grab():
                    break
                cur += 1
            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap_pos = clamped_idx + 1
                self._put_cache(p_str, clamped_idx, frame)
                return frame

        # Em passos para trás ou ré, pré-decodifica em lote o bloco GOP anterior num único seek
        if fill_back_batch:
            batch_start = max(0, clamped_idx - self.backward_batch_size + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
            target_frame: np.ndarray | None = None
            cur = batch_start
            while cur <= clamped_idx:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                self._put_cache(p_str, cur, fr)
                if cur == clamped_idx:
                    target_frame = fr
                cur += 1
            self._cap_pos = cur
            if target_frame is not None:
                return target_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        self._cap_pos = clamped_idx + 1
        self._put_cache(p_str, clamped_idx, frame)
        return frame

    def _find_segment_for_sec(self, sec_of_day: float) -> tuple[int, int, bool]:
        if not self._segments:
            return (-1, 0, False)
        last_seg_idx = len(self._segments) - 1
        for idx, seg in enumerate(self._segments):
            in_bounds = (
                (sec_of_day < seg.end_sec)
                if idx < last_seg_idx
                else (sec_of_day <= seg.end_sec)
            )
            if seg.start_sec <= sec_of_day and in_bounds:
                local_idx = int(round((sec_of_day - seg.start_sec) * max(1.0, seg.fps)))
                return (idx, max(0, min(max(0, seg.frame_count - 1), local_idx)), True)
        # Se clicou numa lacuna sem vídeo, informa que não há exata, mas aponta o clipe mais próximo à frente ou último
        for idx, seg in enumerate(self._segments):
            if seg.start_sec > sec_of_day:
                return (idx, 0, False)
        return (last_seg_idx, 0, False)

    def _sync_playing_state(self, playing: bool | None = None, speed: float | None = None) -> None:
        with self._state_lock:
            if playing is not None:
                self._playing = playing
            if speed is not None:
                self._speed = speed

    def _worker_loop(self) -> None:
        last_tick = time.monotonic()
        while not self._stop.is_set():
            timeout = 0.015 if self._playing else 0.15
            cmds: list[tuple[str, object]] = []
            try:
                cmds.append(self._cmd_queue.get(timeout=timeout))
                while True:
                    cmds.append(self._cmd_queue.get_nowait())
            except queue.Empty:
                pass

            # Coalesce múltiplos `seek_sec` consecutivos durante arrasto na linha do tempo
            coalesced: list[tuple[str, object]] = []
            last_seek_idx = -1
            for kind, payload in cmds:
                if kind == "seek_sec":
                    if last_seek_idx >= 0:
                        coalesced[last_seek_idx] = (kind, payload)
                    else:
                        last_seek_idx = len(coalesced)
                        coalesced.append((kind, payload))
                else:
                    coalesced.append((kind, payload))

            for kind, payload in coalesced:
                if kind == "quit":
                    return
                self._handle_command(kind, payload)
                last_tick = time.monotonic()

            if self._playing and 0 <= self._seg_idx < len(self._segments):
                now = time.monotonic()
                seg = self._segments[self._seg_idx]
                eff_fps = max(1.0, seg.fps)
                display_interval = 1.0 / min(28.0, max(10.0, eff_fps * min(4.0, abs(self._speed))))
                elapsed = now - last_tick
                if elapsed < display_interval:
                    continue
                last_tick = now
                self._advance_playback(min(0.25, elapsed))

    def _handle_command(self, kind: str, payload: object) -> None:
        if kind == "load":
            segs, evs, auto_sec = payload  # type: ignore[misc]
            self._close_capture()
            self._clear_cache()
            self._segments = list(segs)
            self._events = list(evs)
            self._sync_playing_state(playing=False)
            if not self._segments:
                self._seg_idx = -1
                self._frame_idx = 0
                self._frame_accum = 0.0
                self._emit_frame(None, None, 0)
                self._emit_state("Nenhuma gravação encontrada nesta data.")
                return
            target_sec = self._segments[0].start_sec if auto_sec is None else float(auto_sec)
            self._seek_to_sec(target_sec)
            return

        if kind == "play":
            if not self._segments:
                return
            if self._seg_idx < 0:
                self._seg_idx = 0
                self._frame_idx = 0
            self._frame_accum = float(self._frame_idx)
            self._sync_playing_state(playing=True)
            seg = self._segments[self._seg_idx]
            fr = self._read_exact_frame(seg, self._frame_idx, fill_back_batch=self._speed < 0)
            self._emit_frame(fr, seg, self._frame_idx)
            self._emit_state(f"Reproduzindo {seg.filename} ({self._speed:+.0f}x)")
            return

        if kind == "pause":
            self._sync_playing_state(playing=False)
            if 0 <= self._seg_idx < len(self._segments):
                seg = self._segments[self._seg_idx]
                self._emit_frame(self.get_current_frame(), seg, self._frame_idx)
                self._emit_state(f"Pausado em {format_day_seconds(seg.start_sec + self._frame_idx / max(1.0, seg.fps))}")
            return

        if kind == "set_speed":
            spd = float(payload)
            if spd == 0.0:
                spd = 1.0
            self._sync_playing_state(speed=spd)
            if 0 <= self._seg_idx < len(self._segments):
                seg = self._segments[self._seg_idx]
                self._emit_frame(self.get_current_frame(), seg, self._frame_idx)
                self._emit_state(f"Velocidade ajustada para {spd:+.0f}x")
            return

        if kind == "seek_sec":
            self._seek_to_sec(float(payload))
            return

        if kind == "step_frame":
            self._step_exact_frame(int(payload))
            return

        if kind == "jump_event":
            direction, allowed = payload  # type: ignore[misc]
            self._jump_to_event(int(direction), allowed)

    def _seek_to_sec(self, sec_of_day: float) -> None:
        if not self._segments:
            self._emit_state("Nenhuma gravação carregada.")
            return
        s_idx, f_idx, exact = self._find_segment_for_sec(sec_of_day)
        seg = self._segments[s_idx]
        self._seg_idx = s_idx
        self._frame_idx = f_idx
        self._frame_accum = float(f_idx)
        fr = self._read_exact_frame(seg, f_idx, fill_back_batch=self._speed < 0)
        self._emit_frame(fr, seg, f_idx)
        if exact:
            self._emit_state(f"Posicionado em {format_day_seconds(sec_of_day)} — {seg.filename}")
        else:
            self._emit_state(
                f"Sem gravação exatamente às {format_day_seconds(sec_of_day)}; saltou para {seg.filename}."
            )

    def _step_exact_frame(self, delta: int) -> None:
        if not (0 <= self._seg_idx < len(self._segments)):
            return
        self._sync_playing_state(playing=False)
        target_idx = self._frame_idx + (1 if delta >= 0 else -1)
        seg = self._segments[self._seg_idx]

        if target_idx >= max(1, seg.frame_count):
            if self._seg_idx + 1 < len(self._segments):
                self._seg_idx += 1
                seg = self._segments[self._seg_idx]
                target_idx = 0
            else:
                target_idx = max(0, seg.frame_count - 1)
        elif target_idx < 0:
            if self._seg_idx - 1 >= 0:
                self._seg_idx -= 1
                seg = self._segments[self._seg_idx]
                target_idx = max(0, seg.frame_count - 1)
            else:
                target_idx = 0

        self._frame_idx = target_idx
        self._frame_accum = float(target_idx)
        fr = self._read_exact_frame(seg, target_idx, fill_back_batch=delta < 0)
        self._emit_frame(fr, seg, target_idx)
        self._emit_state(f"Quadro a quadro: frame {target_idx + 1}/{max(1, seg.frame_count)}")

    def _jump_to_event(self, direction: int, allowed_types: set[str] | None) -> None:
        filtered = [
            ev for ev in self._events if allowed_types is None or ev.event_type in allowed_types
        ]
        if not filtered or not self._segments:
            self._emit_state("Nenhum evento correspondente nesta data.")
            return
        cur_sec = 0.0
        if 0 <= self._seg_idx < len(self._segments):
            s = self._segments[self._seg_idx]
            cur_sec = s.start_sec + self._frame_idx / max(1.0, s.fps)

        chosen: TimelineEvent | None = None
        if direction >= 0:
            for ev in filtered:
                if ev.sec_of_day > cur_sec + 1.0:
                    chosen = ev
                    break
            if chosen is None:
                chosen = filtered[0]
        else:
            for ev in reversed(filtered):
                if ev.sec_of_day < cur_sec - 1.0:
                    chosen = ev
                    break
            if chosen is None:
                chosen = filtered[-1]

        self._seek_to_sec(chosen.sec_of_day)
        self._emit_state(f"Evento [{format_day_seconds(chosen.sec_of_day)}]: {chosen.label}")

    def _advance_playback(self, elapsed: float) -> None:
        seg = self._segments[self._seg_idx]
        fps = max(1.0, seg.fps)
        self._frame_accum += self._speed * fps * elapsed
        next_idx = int(round(self._frame_accum))
        if next_idx == self._frame_idx:
            return

        if self._speed > 0 and next_idx >= max(1, seg.frame_count):
            if self._seg_idx + 1 < len(self._segments):
                self._seg_idx += 1
                seg = self._segments[self._seg_idx]
                next_idx = 0
                self._frame_accum = 0.0
            else:
                self._frame_idx = max(0, seg.frame_count - 1)
                self._sync_playing_state(playing=False)
                self._emit_frame(self.get_current_frame(), seg, self._frame_idx)
                self._emit_state("Fim das gravações do dia.")
                return
        elif self._speed < 0 and next_idx < 0:
            if self._seg_idx - 1 >= 0:
                self._seg_idx -= 1
                seg = self._segments[self._seg_idx]
                next_idx = max(0, seg.frame_count - 1)
                self._frame_accum = float(next_idx)
            else:
                self._frame_idx = 0
                self._frame_accum = 0.0
                self._sync_playing_state(playing=False)
                fr = self._read_exact_frame(seg, 0)
                self._emit_frame(fr, seg, 0)
                self._emit_state("Início das gravações do dia.")
                return

        self._frame_idx = next_idx
        use_batch = self._speed < 0 and abs(self._speed) <= 2.0
        fr = self._read_exact_frame(seg, next_idx, fill_back_batch=use_batch)
        if fr is not None:
            self._emit_frame(fr, seg, next_idx)
