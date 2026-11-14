"""Configuração persistente e proteção local de credenciais para o HomeCam Segurança."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import uuid

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_RECORDINGS_DIR = BASE_DIR / "gravacoes"
DEFAULT_SECRETS_DIR = BASE_DIR / "secrets"
DEFAULT_PENDING_PATH = BASE_DIR / "pending_uploads.json"

_DPAPI_PREFIX = "dpapi:"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _make_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_ubyte]]:
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    blob = _DataBlob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buf


def _read_and_free_blob(blob: _DataBlob) -> bytes:
    try:
        return bytes(ctypes.string_at(blob.pbData, blob.cbData))
    finally:
        kernel32 = ctypes.windll.kernel32
        kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel32.LocalFree.restype = wintypes.HLOCAL
        kernel32.LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))


def protect_secret(plaintext: str) -> str:
    """Protege uma senha com Windows DPAPI (usuário atual do Windows).

    Em outros sistemas ou caso a API falhe, preserva o texto sem interromper o salvamento.
    """
    if not plaintext:
        return ""
    if sys.platform != "win32":
        return plaintext
    try:
        raw = plaintext.encode("utf-8")
        in_blob, _keepalive = _make_blob(raw)
        out_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            return plaintext
        encrypted = _read_and_free_blob(out_blob)
        return _DPAPI_PREFIX + base64.b64encode(encrypted).decode("ascii")
    except Exception:
        return plaintext


def unprotect_secret(stored: str) -> str:
    """Descriptografa uma senha salva com DPAPI ou devolve valores legados em texto claro."""
    if not stored:
        return ""
    if not stored.startswith(_DPAPI_PREFIX):
        return stored
    if sys.platform != "win32":
        return ""
    try:
        encrypted = base64.b64decode(stored[len(_DPAPI_PREFIX) :])
        in_blob, _keepalive = _make_blob(encrypted)
        out_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            return ""
        return _read_and_free_blob(out_blob).decode("utf-8")
    except Exception:
        return ""


def sanitize_filename_part(value: str, fallback: str = "Camera") -> str:
    """Remove caracteres inválidos para nomes de pastas e arquivos no Windows e no Drive."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (value or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or fallback


def find_virtual_drive_subdir(path: Path | str) -> Path | None:
    """Detecta subpasta real ('Meu Drive' / 'My Drive') dentro da raiz virtual do Google Drive."""
    try:
        base = Path(path)
        for name in ("Meu Drive", "My Drive"):
            sub = base / name
            if sub.exists() and sub.is_dir():
                return sub
    except OSError:
        return None
    return None


def is_directory_writable(path: Path | str) -> bool:
    """Testa se é possível criar subpasta e arquivo dentro do diretório informado."""
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        probe_dir = target / f".homecam_probe_{os.getpid()}"
        probe_dir.mkdir(exist_ok=True)
        try:
            probe_file = probe_dir / "probe.tmp"
            probe_file.write_bytes(b"ok")
            probe_file.unlink(missing_ok=True)
        finally:
            probe_dir.rmdir()
        return True
    except OSError:
        return False


def normalize_drive_folder_dir(raw_path: str | Path) -> tuple[str, bool]:
    """Ajusta caminhos apontando para raiz virtual do Drive (ex.: H:\\ -> H:\\Meu Drive)."""
    text = str(raw_path or "").strip()
    if not text:
        return "", False
    p = Path(text)
    sub = find_virtual_drive_subdir(p)
    if sub is not None and not is_directory_writable(p) and is_directory_writable(sub):
        return str(sub), True
    return text, False


def resolve_writable_recordings_dir(
    raw_path: str | Path,
    *,
    prefer_drive_subdir: bool = False,
    fallback: Path = DEFAULT_RECORDINGS_DIR,
) -> tuple[Path, bool]:
    """Garante uma pasta de gravações realmente gravável, evitando raízes virtuais bloqueadas."""
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if is_directory_writable(candidate):
            return candidate, False
        if prefer_drive_subdir:
            sub = find_virtual_drive_subdir(candidate)
            if sub is not None:
                drive_rec = sub / "HomeCamGravacoes"
                if is_directory_writable(drive_rec):
                    return drive_rec, True
    return fallback, True


@dataclass
class CameraConfig:
    """Parâmetros de conexão de uma câmera monitorada."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Câmera Principal"
    brand: str = "Xiongmai / General / ICSee / XMEye"
    ip: str = ""
    rtsp_port: int = 554
    onvif_port: int = 8899
    username: str = "admin"
    password: str = ""
    channel: int = 1
    stream_type: str = "main"
    custom_rtsp_url: str = ""
    usb_index: int = 0
    source_url_override: str = ""
    serial_number: str = ""
    ptz_speed: int = 5
    ptz_home_preset: int = 100
    ptz_auto_return_enabled: bool = False
    ptz_auto_return_seconds: int = 30


@dataclass
class RecordingConfig:
    """Preferências de gravação local e detecção de movimento."""

    recordings_dir: str = str(DEFAULT_RECORDINGS_DIR)
    segment_minutes: int = 5
    motion_only: bool = False
    motion_sensitivity: float = 0.015
    motion_post_seconds: int = 12
    auto_start_recording: bool = False
    auto_connect_on_startup: bool = False


@dataclass
class DriveConfig:
    """Preferências de sincronização com o Google Drive."""

    enabled: bool = False
    sync_mode: str = "oauth"  # "oauth" (API direta) ou "folder" (Pasta Google Drive Desktop)
    local_drive_dir: str = ""
    credentials_path: str = str(DEFAULT_SECRETS_DIR / "credentials.json")
    token_path: str = str(DEFAULT_SECRETS_DIR / "token.json")
    root_folder_name: str = "HomeCamSeguranca"
    delete_after_upload: bool = False
    auto_upload: bool = True


@dataclass
class AIConfig:
    """Preferências de Visão Computacional e Reconhecimento de Veículos (ALPR) e Pessoas."""

    enabled: bool = False
    show_overlays: bool = True
    analysis_fps: float = 6.0
    detect_people_gender: bool = True
    detect_animals: bool = True
    detect_vehicles_alpr: bool = True
    save_csv_log: bool = True
    csv_log_path: str = str(DEFAULT_RECORDINGS_DIR / "veiculos_detectados.csv")
    people_csv_log_path: str = str(DEFAULT_RECORDINGS_DIR / "pessoas_detectadas.csv")


@dataclass
class WebConfig:
    """Preferências do servidor HTTP/REST e Portal Web remoto."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""


@dataclass
class AppConfig:
    """Configuração raiz do aplicativo."""

    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    selected_camera_id: str = ""
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _filter_dataclass_fields(cls: type, data: dict) -> dict:
    valid_keys = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in data.items() if k in valid_keys}


def _looks_like_windows_path(raw: str) -> bool:
    text = (raw or "").strip()
    return (len(text) >= 2 and text[1] == ":" and text[0].isalpha()) or text.startswith("\\\\")


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Aplica variáveis de ambiente opcionais para execução headless/Docker/cloud."""
    env_rec = (os.environ.get("HOMECAM_RECORDINGS_DIR") or "").strip()
    if env_rec:
        config.recording.recordings_dir = env_rec
    elif os.name != "nt" and _looks_like_windows_path(config.recording.recordings_dir):
        config.recording.recordings_dir = str(DEFAULT_RECORDINGS_DIR)

    if os.name != "nt":
        if _looks_like_windows_path(config.drive.credentials_path):
            config.drive.credentials_path = str(
                DEFAULT_SECRETS_DIR / (PureWindowsPath(config.drive.credentials_path).name or "credentials.json")
            )
        if _looks_like_windows_path(config.drive.token_path):
            config.drive.token_path = str(
                DEFAULT_SECRETS_DIR / (PureWindowsPath(config.drive.token_path).name or "token.json")
            )
        if _looks_like_windows_path(config.drive.local_drive_dir):
            config.drive.local_drive_dir = ""

    env_creds = (os.environ.get("HOMECAM_DRIVE_CREDENTIALS") or "").strip()
    if env_creds:
        config.drive.credentials_path = env_creds
    env_token = (os.environ.get("HOMECAM_DRIVE_TOKEN") or "").strip()
    if env_token:
        config.drive.token_path = env_token

    use_drive_oauth = config.drive.enabled and config.drive.sync_mode == "oauth"
    resolved_rec_dir, _ = resolve_writable_recordings_dir(
        config.recording.recordings_dir,
        prefer_drive_subdir=not use_drive_oauth,
        fallback=DEFAULT_RECORDINGS_DIR,
    )
    config.recording.recordings_dir = str(resolved_rec_dir)

    env_veh_csv = (os.environ.get("HOMECAM_VEHICLES_CSV") or "").strip()
    if env_veh_csv:
        config.ai.csv_log_path = env_veh_csv
    elif env_rec or (os.name != "nt" and _looks_like_windows_path(config.ai.csv_log_path)):
        config.ai.csv_log_path = str(resolved_rec_dir / "veiculos_detectados.csv")

    env_ppl_csv = (os.environ.get("HOMECAM_PEOPLE_CSV") or "").strip()
    if env_ppl_csv:
        config.ai.people_csv_log_path = env_ppl_csv
    elif env_rec or (os.name != "nt" and _looks_like_windows_path(config.ai.people_csv_log_path)):
        config.ai.people_csv_log_path = str(resolved_rec_dir / "pessoas_detectadas.csv")

    env_pwd = os.environ.get("HOMECAM_CAM_PASSWORD")
    if env_pwd:
        for cam in config.cameras:
            cam.password = env_pwd

    selected = ConfigManager.get_selected_camera(config)
    env_src = (os.environ.get("HOMECAM_SOURCE_URL") or os.environ.get("HOMECAM_RTSP_URL") or "").strip()
    if env_src:
        selected.source_url_override = env_src

    env_web_host = (os.environ.get("HOMECAM_WEB_HOST") or "").strip()
    if env_web_host:
        config.web.host = env_web_host
    env_web_port = (os.environ.get("HOMECAM_WEB_PORT") or "").strip()
    if env_web_port:
        try:
            port_val = int(env_web_port)
            if 1 <= port_val <= 65535:
                config.web.port = port_val
        except ValueError:
            pass
    env_web_token = os.environ.get("HOMECAM_WEB_TOKEN")
    if env_web_token is not None and env_web_token != "":
        config.web.auth_token = env_web_token

    return config


class ConfigManager:
    """Carrega e salva a configuração do aplicativo em JSON com escrita atômica."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)

    def load(self) -> AppConfig:
        """Carrega a configuração do disco ou cria uma configuração padrão."""
        if not self.config_path.exists():
            cfg = AppConfig()
            cfg.selected_camera_id = cfg.cameras[0].id
            return _apply_env_overrides(cfg)
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = AppConfig()
            cfg.selected_camera_id = cfg.cameras[0].id
            return _apply_env_overrides(cfg)

        cameras: list[CameraConfig] = []
        for item in raw.get("cameras", []):
            if isinstance(item, dict):
                filtered = _filter_dataclass_fields(CameraConfig, item)
                filtered["password"] = unprotect_secret(str(filtered.get("password", "")))
                cameras.append(CameraConfig(**filtered))
        if not cameras:
            cameras = [CameraConfig()]

        rec_raw = raw.get("recording", {})
        recording = (
            RecordingConfig(**_filter_dataclass_fields(RecordingConfig, rec_raw))
            if isinstance(rec_raw, dict)
            else RecordingConfig()
        )

        drive_raw = raw.get("drive", {})
        drive = (
            DriveConfig(**_filter_dataclass_fields(DriveConfig, drive_raw))
            if isinstance(drive_raw, dict)
            else DriveConfig()
        )
        normalized_drive_dir, _ = normalize_drive_folder_dir(drive.local_drive_dir)
        drive.local_drive_dir = normalized_drive_dir

        ai_raw = raw.get("ai", {})
        ai = (
            AIConfig(**_filter_dataclass_fields(AIConfig, ai_raw))
            if isinstance(ai_raw, dict)
            else AIConfig()
        )

        web_raw = raw.get("web", {})
        web = (
            WebConfig(**_filter_dataclass_fields(WebConfig, web_raw))
            if isinstance(web_raw, dict)
            else WebConfig()
        )

        selected_id = str(raw.get("selected_camera_id", ""))
        if not any(cam.id == selected_id for cam in cameras):
            selected_id = cameras[0].id

        cfg = AppConfig(
            cameras=cameras,
            selected_camera_id=selected_id,
            recording=recording,
            drive=drive,
            ai=ai,
            web=web,
        )
        return _apply_env_overrides(cfg)

    def save(self, config: AppConfig) -> None:
        """Persiste a configuração criptografando as senhas com DPAPI."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_cams: list[dict] = []
        for cam in config.cameras:
            cam_dict = asdict(cam)
            cam_dict["password"] = protect_secret(cam.password)
            serialized_cams.append(cam_dict)

        payload = {
            "selected_camera_id": config.selected_camera_id,
            "cameras": serialized_cams,
            "recording": asdict(config.recording),
            "drive": asdict(config.drive),
            "ai": asdict(config.ai),
            "web": asdict(config.web),
        }

        tmp_path = self.config_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.config_path)

    @staticmethod
    def get_selected_camera(config: AppConfig) -> CameraConfig:
        """Retorna a câmera selecionada atualmente ou a primeira cadastrada."""
        for cam in config.cameras:
            if cam.id == config.selected_camera_id:
                return cam
        if not config.cameras:
            cam = CameraConfig()
            config.cameras.append(cam)
            config.selected_camera_id = cam.id
            return cam
        config.selected_camera_id = config.cameras[0].id
        return config.cameras[0]

    @staticmethod
    def upsert_camera(config: AppConfig, camera: CameraConfig) -> None:
        """Atualiza uma câmera existente ou adiciona uma nova e a seleciona."""
        for idx, existing in enumerate(config.cameras):
            if existing.id == camera.id:
                config.cameras[idx] = camera
                config.selected_camera_id = camera.id
                return
        config.cameras.append(camera)
        config.selected_camera_id = camera.id

    @staticmethod
    def delete_camera(config: AppConfig, camera_id: str) -> None:
        """Remove uma câmera pelo ID garantindo que reste ao menos um perfil."""
        config.cameras = [c for c in config.cameras if c.id != camera_id]
        if not config.cameras:
            config.cameras = [CameraConfig()]
        if not any(c.id == config.selected_camera_id for c in config.cameras):
            config.selected_camera_id = config.cameras[0].id

