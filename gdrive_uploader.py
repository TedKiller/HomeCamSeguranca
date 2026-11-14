"""Autenticação OAuth2 e fila persistente de envio resumível para o Google Drive."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import threading

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from config_manager import (
    DEFAULT_PENDING_PATH,
    DEFAULT_SECRETS_DIR,
    find_virtual_drive_subdir,
    normalize_drive_folder_dir,
    sanitize_filename_part,
)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DRIVE_SETUP_GUIDE_PT = (
    "Como conectar sua conta do Google Drive (passo a passo):\n\n"
    "1. Acesse o Google Cloud Console (https://console.cloud.google.com/) com sua conta Google.\n"
    "2. Crie um projeto novo (ex.: 'HomeCam Seguranca').\n"
    "3. No menu 'APIs e Serviços > Biblioteca', pesquise por 'Google Drive API' e clique em ATIVAR.\n"
    "4. Em 'Tela de permissão OAuth', escolha tipo 'Externo', preencha o nome do app e adicione "
    "seu próprio e-mail em 'Usuários de teste'.\n"
    "5. Em 'APIs e Serviços > Credenciais', clique em '+ CRIAR CREDENCIAIS > ID do cliente OAuth'.\n"
    "6. Em 'Tipo de aplicativo', escolha 'App para computador' (Desktop app) e clique em Criar.\n"
    "7. Baixe o arquivo JSON gerado, volte aqui na aba 'Google Drive' e selecione esse arquivo "
    "(credentials.json), depois clique em 'Conectar Conta Google Drive'."
)


@dataclass
class UploadJob:
    """Item pendente na fila persistente de envio para o Google Drive."""

    local_path: str
    camera_name: str
    date_folder: str
    attempts: int = 0
    last_error: str = ""


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _guess_camera_and_date(path: Path) -> tuple[str, str]:
    parent_name = path.parent.name
    if _DATE_DIR_RE.match(parent_name):
        date_folder = parent_name
        cam_name = path.parent.parent.name or "Camera"
    else:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        cam_name = path.parent.name or "Camera"
    return sanitize_filename_part(cam_name, "Camera"), date_folder


def validate_oauth_client_json(path: Path | str) -> tuple[bool, str]:
    """Valida se um arquivo JSON corresponde a um cliente OAuth ('installed' ou 'web')."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False, f"Arquivo não encontrado: {p}"
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as err:
        return False, f"Arquivo JSON inválido ({err})."
    if not isinstance(data, dict):
        return False, "Formato JSON inválido para credencial OAuth."
    if data.get("type") == "service_account":
        return (
            False,
            "Este arquivo é de 'Conta de serviço'. Para uso pessoal no Google Drive, crie um "
            "'ID do cliente OAuth' > tipo 'App para computador' (Desktop app).",
        )
    section = data.get("installed") or data.get("web")
    if isinstance(section, dict) and section.get("client_id") and section.get("client_secret"):
        return True, "Credencial OAuth válida."
    return (
        False,
        "O JSON selecionado não contém 'installed.client_id' / 'client_secret'. "
        "Baixe o JSON de 'ID do cliente OAuth' ('App para computador') no Google Cloud Console.",
    )


def detect_google_drive_desktop_dirs() -> list[Path]:
    """Localiza pastas do Google Drive para computador montadas no sistema."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        try:
            mask = ctypes.windll.kernel32.GetLogicalDrives()
        except Exception:
            mask = 0
        for idx in range(3, 26):  # D: até Z:
            if mask & (1 << idx):
                letter = chr(ord("A") + idx)
                for sub in ("Meu Drive", "My Drive"):
                    candidates.append(Path(f"{letter}:\\{sub}"))
    for sub in ("Google Drive", "Meu Drive", "My Drive"):
        candidates.append(Path.home() / sub)

    found: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    found.append(p)
        except OSError:
            continue
    return found


class GoogleDriveUploader:
    """Envia gravações e snapshots para o Google Drive (API OAuth ou pasta sincronizada)."""

    def __init__(
        self,
        pending_path: Path | str = DEFAULT_PENDING_PATH,
        credentials_path: Path | str = DEFAULT_SECRETS_DIR / "credentials.json",
        token_path: Path | str = DEFAULT_SECRETS_DIR / "token.json",
        root_folder_name: str = "HomeCamSeguranca",
        delete_after_upload: bool = False,
        on_log: Callable[[str], None] | None = None,
        on_queue_change: Callable[[int], None] | None = None,
        service_factory: Callable[[object], object] | None = None,
        sync_mode: str = "oauth",
        local_drive_dir: Path | str = "",
    ) -> None:
        self.pending_path = Path(pending_path)
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.root_folder_name = sanitize_filename_part(root_folder_name, "HomeCamSeguranca")
        self.delete_after_upload = bool(delete_after_upload)
        self.sync_mode = "folder" if sync_mode == "folder" else "oauth"
        norm_dir, _ = normalize_drive_folder_dir(local_drive_dir)
        self.local_drive_dir: Path | None = Path(norm_dir) if norm_dir else None
        self.on_log = on_log
        self.on_queue_change = on_queue_change
        self._service_factory = service_factory or (
            lambda creds: build("drive", "v3", credentials=creds, cache_discovery=False)
        )

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._folder_cache: dict[tuple[str, str], str] = {}
        self._creds: Credentials | None = None
        self._account_email = ""
        self.history_path = self.pending_path.with_name(f"{self.pending_path.stem}_uploaded.json")
        self._jobs: list[UploadJob] = self._load_pending()
        self._uploaded_history: dict[str, list[int]] = self._load_uploaded_history()

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    def _notify_queue(self) -> None:
        if self.on_queue_change:
            with self._lock:
                count = len(self._jobs)
            self.on_queue_change(count)

    def configure(
        self,
        credentials_path: Path | str,
        token_path: Path | str,
        root_folder_name: str,
        delete_after_upload: bool,
        sync_mode: str = "oauth",
        local_drive_dir: Path | str = "",
    ) -> None:
        """Atualiza caminhos de credenciais e política de armazenamento no Drive."""
        new_root = sanitize_filename_part(root_folder_name, "HomeCamSeguranca")
        new_token = Path(token_path)
        norm_dir, _ = normalize_drive_folder_dir(local_drive_dir)
        with self._lock:
            if new_root != self.root_folder_name:
                self._folder_cache.clear()
            if new_token != self.token_path:
                self._creds = None
                self._account_email = ""
                self._folder_cache.clear()
            self.credentials_path = Path(credentials_path)
            self.token_path = new_token
            self.root_folder_name = new_root
            self.delete_after_upload = bool(delete_after_upload)
            self.sync_mode = "folder" if sync_mode == "folder" else "oauth"
            self.local_drive_dir = Path(norm_dir) if norm_dir else None

    def is_folder_ready(self) -> bool:
        """Indica se a pasta local do Google Drive para computador está disponível."""
        with self._lock:
            target = self.local_drive_dir
        return (
            target is not None
            and target.exists()
            and target.is_dir()
            and find_virtual_drive_subdir(target) is None
        )

    def is_ready(self) -> bool:
        """Indica se o destino configurado (OAuth ou pasta sincronizada) está pronto para enviar."""
        if self.sync_mode == "folder":
            return self.is_folder_ready()
        return self.is_authenticated()

    def has_credentials_file(self) -> bool:
        """Indica se o arquivo `credentials.json` OAuth existe no caminho configurado."""
        return self.credentials_path.exists() and self.credentials_path.is_file()

    def has_saved_token(self) -> bool:
        """Indica rapidamente (sem chamadas de rede) se há token salvo localmente."""
        return (self._creds is not None and self._creds.valid) or self.token_path.exists()

    @property
    def cached_account_email(self) -> str:
        """Retorna o último e-mail autenticado em cache sem bloquear em I/O."""
        return self._account_email

    def load_saved_credentials(self) -> Credentials | None:
        """Carrega o token salvo e renova automaticamente quando expirado."""
        if self._creds is not None and self._creds.valid:
            return self._creds

        if not self.token_path.exists():
            return None

        try:
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        except Exception:
            return None

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_token(creds)
            except Exception:
                return None

        if creds and creds.valid:
            self._creds = creds
            return creds
        return None

    def is_authenticated(self) -> bool:
        """Verifica se existe uma sessão OAuth válida do Google Drive."""
        return self.load_saved_credentials() is not None

    def _save_token(self, creds: Credentials) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.token_path.with_suffix(".tmp")
        tmp_path.write_text(creds.to_json(), encoding="utf-8")
        os.replace(tmp_path, self.token_path)

    def get_account_email(self) -> str:
        """Retorna o e-mail da conta Google autenticada, se disponível."""
        if self._account_email:
            return self._account_email
        creds = self.load_saved_credentials()
        if not creds:
            return ""
        try:
            service = self._service_factory(creds)
            about = service.about().get(fields="user(emailAddress,displayName)").execute()
            user = about.get("user", {})
            email = str(user.get("emailAddress") or user.get("displayName") or "")
            self._account_email = email
            return email
        except Exception:
            return ""

    def authenticate_interactive(self) -> tuple[bool, str]:
        """Abre o navegador para autorizar acesso ao Google Drive e salva o token."""
        if not self.has_credentials_file():
            return (
                False,
                f"Arquivo credentials.json não encontrado em:\n{self.credentials_path}\n\n"
                "Baixe o arquivo OAuth ('App para computador') no Google Cloud Console.",
            )

        valid_json, validation_msg = validate_oauth_client_json(self.credentials_path)
        if not valid_json:
            return False, validation_msg

        existing = self.load_saved_credentials()
        if existing is not None:
            email = self.get_account_email()
            self._wake_event.set()
            return True, f"Conectado ao Google Drive ({email or 'sessão ativa'})."

        try:
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
            self._save_token(creds)
            self._creds = creds
            self._account_email = ""
            email = self.get_account_email()
            self._wake_event.set()
            self._log(f"Conta Google Drive conectada: {email or 'autenticada'}")
            return True, f"Conta conectada com sucesso ({email or 'autenticada'})!"
        except Exception as err:
            return False, f"Falha na autenticação com o Google Drive: {err}"

    def disconnect(self) -> None:
        """Remove o token local salvo e encerra a sessão do Google Drive."""
        with self._lock:
            self._creds = None
            self._account_email = ""
            self._folder_cache.clear()
        if self.token_path.exists():
            try:
                self.token_path.unlink()
            except OSError:
                pass
        self._log("Conta Google Drive desconectada.")

    def _load_pending(self) -> list[UploadJob]:
        if not self.pending_path.exists():
            return []
        try:
            raw = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        jobs: list[UploadJob] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("local_path"):
                    jobs.append(
                        UploadJob(
                            local_path=str(item["local_path"]),
                            camera_name=str(item.get("camera_name") or "Camera"),
                            date_folder=str(item.get("date_folder") or datetime.now().strftime("%Y-%m-%d")),
                            attempts=int(item.get("attempts") or 0),
                            last_error=str(item.get("last_error") or ""),
                        )
                    )
        return jobs

    def _atomic_write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)

    def _load_uploaded_history(self) -> dict[str, list[int]]:
        if not self.history_path.exists():
            return {}
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        cleaned: dict[str, list[int]] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, list) and len(v) == 2:
                try:
                    cleaned[k] = [int(v[0]), int(v[1])]
                except (TypeError, ValueError):
                    continue
        return cleaned

    def _save_pending_locked(self) -> None:
        self._atomic_write_json(self.pending_path, [asdict(job) for job in self._jobs])

    def _record_uploaded_locked(self, path: Path) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        self._uploaded_history[str(path.resolve())] = [int(st.st_size), int(st.st_mtime_ns)]
        if len(self._uploaded_history) > 2000:
            overflow = len(self._uploaded_history) - 2000
            for old_key in list(self._uploaded_history.keys())[:overflow]:
                self._uploaded_history.pop(old_key, None)
        self._atomic_write_json(self.history_path, self._uploaded_history)

    def was_uploaded(self, local_path: Path | str) -> bool:
        """Indica se exatamente esta versão do arquivo local já foi enviada ao Drive."""
        path = Path(local_path).resolve()
        try:
            st = path.stat()
        except OSError:
            return False
        sig = [int(st.st_size), int(st.st_mtime_ns)]
        with self._lock:
            return self._uploaded_history.get(str(path)) == sig

    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def get_pending_jobs(self) -> list[UploadJob]:
        with self._lock:
            return [UploadJob(**asdict(j)) for j in self._jobs]

    def enqueue_file(
        self,
        local_path: Path | str,
        camera_name: str | None = None,
        date_folder: str | None = None,
        force: bool = False,
    ) -> bool:
        """Adiciona um vídeo ou foto à fila persistente de envio ao Drive."""
        path = Path(local_path).resolve()
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return False
        if not force and self.was_uploaded(path):
            return False

        inferred_cam, inferred_date = _guess_camera_and_date(path)
        clean_cam = sanitize_filename_part(camera_name or inferred_cam, "Camera")
        clean_date = date_folder or inferred_date

        with self._lock:
            if any(Path(j.local_path).resolve() == path for j in self._jobs):
                return False
            self._jobs.append(
                UploadJob(
                    local_path=str(path),
                    camera_name=clean_cam,
                    date_folder=clean_date,
                )
            )
            self._save_pending_locked()

        self._notify_queue()
        self._wake_event.set()
        return True

    def enqueue_existing_folder(self, root_dir: Path | str) -> int:
        """Enfileira gravações e fotos já existentes em uma pasta local."""
        root = Path(root_dir)
        if not root.exists():
            return 0
        added = 0
        for ext in ("*.mp4", "*.jpg"):
            for candidate in sorted(root.rglob(ext)):
                if self.enqueue_file(candidate):
                    added += 1
        return added

    def start_worker(self) -> None:
        """Inicia a thread de envio em segundo plano."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def stop_worker(self) -> None:
        """Para a thread de envio em segundo plano."""
        self._stop_event.set()
        self._wake_event.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def trigger_sync(self) -> None:
        """Acorda o worker imediatamente para tentar enviar itens pendentes."""
        self._wake_event.set()

    def _find_folder_id(self, service: object, name: str, parent_id: str | None = None) -> str | None:
        cache_key = (parent_id or "root", name)
        with self._lock:
            cached = self._folder_cache.get(cache_key)
        if cached:
            return cached

        escaped = _escape_drive_query_value(name)
        query_parts = [
            f"name='{escaped}'",
            "mimeType='application/vnd.google-apps.folder'",
            "trashed=false",
            f"'{parent_id}' in parents" if parent_id else "'root' in parents",
        ]
        resp = (
            service.files()
            .list(q=" and ".join(query_parts), spaces="drive", fields="files(id, name)", pageSize=5)
            .execute()
        )
        files = resp.get("files", [])
        if not files:
            return None
        folder_id = str(files[0]["id"])
        with self._lock:
            self._folder_cache[cache_key] = folder_id
        return folder_id

    def _ensure_folder(self, service: object, name: str, parent_id: str | None = None) -> str:
        existing = self._find_folder_id(service, name, parent_id=parent_id)
        if existing:
            return existing

        metadata: dict[str, object] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]
        created = service.files().create(body=metadata, fields="id").execute()
        folder_id = str(created["id"])

        cache_key = (parent_id or "root", name)
        with self._lock:
            self._folder_cache[cache_key] = folder_id
        return folder_id

    def list_remote_dates(self, camera_name: str) -> list[str]:
        """Lista datas (`YYYY-MM-DD`) com gravações desta câmera no Google Drive."""
        cam_part = sanitize_filename_part(camera_name, "Camera")
        if self.sync_mode == "folder":
            with self._lock:
                base = self.local_drive_dir
                root_name = self.root_folder_name
            if base is None:
                return []
            cam_dir = base / root_name / cam_part
            if not cam_dir.exists() or not cam_dir.is_dir():
                return []
            return sorted(
                (p.name for p in cam_dir.iterdir() if p.is_dir() and _DATE_DIR_RE.match(p.name)),
                reverse=True,
            )

        creds = self.load_saved_credentials()
        if creds is None:
            return []
        try:
            service = self._service_factory(creds)
            root_id = self._find_folder_id(service, self.root_folder_name, parent_id=None)
            if not root_id:
                return []
            cam_id = self._find_folder_id(service, cam_part, parent_id=root_id)
            if not cam_id:
                return []
            resp = (
                service.files()
                .list(
                    q=(
                        f"'{cam_id}' in parents and "
                        "mimeType='application/vnd.google-apps.folder' and trashed=false"
                    ),
                    spaces="drive",
                    fields="files(id, name)",
                    pageSize=200,
                )
                .execute()
            )
            dates = [
                str(item["name"])
                for item in resp.get("files", [])
                if _DATE_DIR_RE.match(str(item.get("name") or ""))
            ]
            return sorted(dates, reverse=True)
        except Exception as err:
            self._log(f"Aviso ao listar datas no Google Drive: {err}")
            return []

    def list_remote_recordings(self, camera_name: str, date_str: str) -> list[dict[str, object]]:
        """Lista vídeos `.mp4` de uma câmera/data no Google Drive (pasta local ou API OAuth)."""
        cam_part = sanitize_filename_part(camera_name, "Camera")
        if self.sync_mode == "folder":
            with self._lock:
                base = self.local_drive_dir
                root_name = self.root_folder_name
            if base is None:
                return []
            day_dir = base / root_name / cam_part / date_str
            if not day_dir.exists() or not day_dir.is_dir():
                return []
            out: list[dict[str, object]] = []
            for p in sorted(day_dir.iterdir(), key=lambda x: x.name):
                if p.is_file() and p.suffix.lower() == ".mp4":
                    try:
                        st = p.stat()
                        out.append(
                            {
                                "id": "",
                                "name": p.name,
                                "size": int(st.st_size),
                                "mtime": float(st.st_mtime),
                                "local_path": str(p),
                            }
                        )
                    except OSError:
                        continue
            return out

        creds = self.load_saved_credentials()
        if creds is None:
            return []
        try:
            service = self._service_factory(creds)
            root_id = self._find_folder_id(service, self.root_folder_name, parent_id=None)
            if not root_id:
                return []
            cam_id = self._find_folder_id(service, cam_part, parent_id=root_id)
            if not cam_id:
                return []
            date_id = self._find_folder_id(service, date_str, parent_id=cam_id)
            if not date_id:
                return []

            items: list[dict[str, object]] = []
            page_token: str | None = None
            while True:
                kwargs: dict[str, object] = {
                    "q": f"'{date_id}' in parents and trashed=false",
                    "spaces": "drive",
                    "fields": "nextPageToken, files(id, name, size, modifiedTime)",
                    "pageSize": 250,
                }
                if page_token:
                    kwargs["pageToken"] = page_token
                resp = service.files().list(**kwargs).execute()
                for entry in resp.get("files", []):
                    name = str(entry.get("name") or "")
                    if not name.lower().endswith(".mp4"):
                        continue
                    items.append(
                        {
                            "id": str(entry.get("id") or ""),
                            "name": name,
                            "size": int(entry.get("size") or 0),
                            "modified_time": str(entry.get("modifiedTime") or ""),
                            "local_path": None,
                        }
                    )
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            items.sort(key=lambda row: str(row["name"]))
            return items
        except Exception as err:
            self._log(f"Aviso ao listar gravações no Google Drive ({date_str}): {err}")
            return []

    def download_remote_file(self, file_id: str, dest_path: Path | str) -> Path:
        """Baixa atomicamente um arquivo do Google Drive (API v3) para cache local de playback."""
        target = Path(dest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        creds = self.load_saved_credentials()
        if creds is None:
            raise RuntimeError("Conta Google Drive não conectada para baixar gravação.")
        service = self._service_factory(creds)
        request = service.files().get_media(fileId=file_id)
        tmp = target.with_suffix(".mp4.tmp")
        try:
            with tmp.open("wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=4 * 1024 * 1024)
                done = False
                while not done:
                    if self._stop_event.is_set():
                        raise InterruptedError("Download cancelado no encerramento.")
                    _status, done = downloader.next_chunk()
            os.replace(tmp, target)
            return target
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _upload_job(self, service: object, job: UploadJob) -> str:
        path = Path(job.local_path)
        root_id = self._ensure_folder(service, self.root_folder_name, parent_id=None)
        cam_id = self._ensure_folder(service, job.camera_name, parent_id=root_id)
        date_id = self._ensure_folder(service, job.date_folder, parent_id=cam_id)

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"

        media = MediaFileUpload(str(path), mimetype=mime_type, chunksize=5 * 1024 * 1024, resumable=True)
        try:
            request = service.files().create(
                body={"name": path.name, "parents": [date_id]},
                media_body=media,
                fields="id, name",
            )
            response = None
            while response is None:
                if self._stop_event.is_set():
                    raise InterruptedError("Envio interrompido pelo encerramento do aplicativo.")
                _status, response = request.next_chunk()

            file_id = str(response.get("id", ""))
            if not file_id:
                raise RuntimeError("Google Drive não retornou o ID do arquivo enviado.")
            return file_id
        finally:
            try:
                stream = media.stream()
                if stream is not None and not getattr(stream, "closed", True):
                    stream.close()
            except Exception:
                pass

    def _sync_job_to_folder(self, job: UploadJob) -> Path:
        """Copia o arquivo de forma atômica para a pasta sincronizada do Google Drive."""
        with self._lock:
            base_dir = self.local_drive_dir
            root_name = self.root_folder_name
        if base_dir is None or not base_dir.exists() or not base_dir.is_dir():
            raise RuntimeError("Pasta local do Google Drive indisponível.")

        src = Path(job.local_path)
        dest_dir = base_dir / root_name / job.camera_name / job.date_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if src.resolve() == dest.resolve():
            return dest

        tmp_dest = dest_dir / f".{src.name}.tmp"
        shutil.copy2(src, tmp_dest)
        if tmp_dest.stat().st_size != src.stat().st_size:
            raise OSError("Tamanho do arquivo copiado difere do arquivo original.")
        os.replace(tmp_dest, dest)
        return dest

    def process_next_pending(self) -> bool:
        """Processa um único item pendente da fila. Retorna True se enviou ou removeu um item."""
        with self._lock:
            if not self._jobs:
                return False
            job = self._jobs[0]
            mode = self.sync_mode

        path = Path(job.local_path)
        if not path.exists() or path.stat().st_size <= 0:
            with self._lock:
                if self._jobs and self._jobs[0].local_path == job.local_path:
                    self._jobs.pop(0)
                    self._save_pending_locked()
            self._notify_queue()
            return True

        dest_path: Path | None = None
        summary = ""
        try:
            if mode == "folder":
                if not self.is_folder_ready():
                    return False
                dest_path = self._sync_job_to_folder(job)
                summary = f"Copiado para pasta do Google Drive: {dest_path}"
            else:
                creds = self.load_saved_credentials()
                if creds is None:
                    return False
                service = self._service_factory(creds)
                file_id = self._upload_job(service, job)
                summary = f"Enviado ao Google Drive: {job.camera_name}/{job.date_folder}/{path.name} (ID {file_id})"
        except Exception as err:
            with self._lock:
                if self._jobs and self._jobs[0].local_path == job.local_path:
                    self._jobs[0].attempts += 1
                    self._jobs[0].last_error = str(err)
                    self._save_pending_locked()
            self._log(f"Erro ao enviar {path.name} para o Drive (tentativa {job.attempts + 1}): {err}")
            return False

        with self._lock:
            if self._jobs and self._jobs[0].local_path == job.local_path:
                self._jobs.pop(0)
                self._save_pending_locked()

        self._notify_queue()
        self._log(summary)

        should_delete = self.delete_after_upload and (dest_path is None or dest_path.resolve() != path.resolve())
        deleted = False
        if should_delete:
            try:
                path.unlink()
                deleted = True
                self._log(f"Arquivo local removido após backup verificado: {path.name}")
            except OSError as err:
                self._log(f"Aviso: enviado ao Drive, mas não foi possível apagar local ({err}).")
        if not deleted:
            with self._lock:
                self._record_uploaded_locked(path)
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            while not self._stop_event.is_set():
                if self.get_pending_count() == 0 or not self.is_ready():
                    break
                processed = self.process_next_pending()
                if not processed:
                    # Aguarda antes de tentar novamente em caso de falha de rede
                    self._stop_event.wait(10.0)
            self._wake_event.wait(timeout=15.0)
