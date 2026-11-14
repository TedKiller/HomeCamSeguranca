"""Cliente nativo para câmeras Xiongmai / iCSee / XMEye (protocolo NetIP / DVRIP TCP 34567)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit

import cv2
import imageio_ffmpeg
import numpy as np

XM_DEFAULT_PORT = 34567
XM_WATCH_PRESET_ID = 100

_MSG_LOGIN_REQ = 1000
_MSG_KEEPALIVE_REQ = 1006
_MSG_SYSINFO_REQ = 1020
_MSG_CONFIG_SET = 1040
_MSG_CONFIG_GET = 1042
_MSG_SYSTEM_ABILITY = 1360
_MSG_PTZ_CONTROL = 1400
_MSG_MONITOR_REQ = 1410
_MSG_MONITOR_DATA = 1412
_MSG_MONITOR_CLAIM = 1413

PTZ_DIRECTIONS: dict[str, str] = {
    "up": "DirectionUp",
    "down": "DirectionDown",
    "left": "DirectionLeft",
    "right": "DirectionRight",
    "left_up": "DirectionLeftUp",
    "right_up": "DirectionRightUp",
    "left_down": "DirectionLeftDown",
    "right_down": "DirectionRightDown",
    "zoom_in": "ZoomTile",
    "zoom_out": "ZoomWide",
}

_SOFIA_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

_RET_ERRORS = {
    105: "Senha incorreta na câmera.",
    106: "Usuário bloqueado ou sem permissão.",
    113: "Usuário não encontrado na câmera.",
    203: "Senha incorreta na câmera.",
    205: "Usuário não encontrado na câmera.",
}


@dataclass
class XiongmaiProbeResult:
    """Resultado do diagnóstico direto via Xiongmai NetIP (porta 34567)."""

    ok: bool
    message: str
    width: int = 0
    height: int = 0
    fps: float = 12.0
    model: str = ""
    serial: str = ""
    codec: str = "hevc"


@dataclass
class PTZPresetItem:
    """Ponto fixo / preset armazenado na câmera."""

    id: int
    name: str = ""


@dataclass
class DetectTrackSettings:
    """Configuração de auto-rastreamento e tempo de retorno no firmware Xiongmai."""

    supported: bool = False
    watch_point_supported: bool = False
    enable: bool = False
    return_time: int = 30
    sensitivity: int = 1


@dataclass
class PTZStatusSnapshot:
    """Estado consolidado de capacidades PTZ, presets e rastreamento da câmera."""

    supported_ptz: bool = True
    presets: list[PTZPresetItem] | None = None
    detect_track: DetectTrackSettings | None = None


def xm_sofia_hash(password: str) -> str:
    """Calcula o hash MD5 Sofia de 8 caracteres exigido no login NetIP/DVRIP."""
    digest = hashlib.md5(password.encode("utf-8")).digest()
    return "".join(
        _SOFIA_CHARS[(digest[2 * idx] + digest[2 * idx + 1]) % len(_SOFIA_CHARS)]
        for idx in range(8)
    )


def build_dvrip_url(
    ip: str,
    port: int = XM_DEFAULT_PORT,
    username: str = "admin",
    password: str = "",
    channel: int = 1,
    stream_type: str = "main",
) -> str:
    """Monta URL interna `dvrip://` para conexão direta à porta NetIP 34567."""
    clean_ip = ip.strip()
    if not clean_ip:
        return ""
    auth = ""
    if username:
        u = quote(username, safe="")
        auth = f"{u}:{quote(password, safe='')}@" if password else f"{u}@"
    st = "main" if stream_type == "main" else "sub"
    return f"dvrip://{auth}{clean_ip}:{int(port)}?channel={max(1, int(channel))}&stream={st}"


def parse_dvrip_url(url: str) -> tuple[str, int, str, str, int, str]:
    """Extrai `(ip, port, username, password, channel_1based, stream_type)` de uma URL `dvrip://`."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").strip()
    port = int(parts.port or XM_DEFAULT_PORT)
    username = unquote(parts.username) if parts.username else "admin"
    password = unquote(parts.password) if parts.password else ""
    qs = parse_qs(parts.query)
    try:
        channel = max(1, int(qs.get("channel", ["1"])[0]))
    except ValueError:
        channel = 1
    stream_type = qs.get("stream", ["main"])[0].lower()
    if stream_type not in ("main", "sub"):
        stream_type = "main"
    return host, port, username, password, channel, stream_type


def pack_xm_packet(session_id: int, seq: int, msg_id: int, payload_obj: dict) -> bytes:
    """Empacota um quadro NetIP/DVRIP (cabeçalho binário 20 bytes + corpo JSON)."""
    payload = (json.dumps(payload_obj, separators=(",", ":")) + "\x0a\x00").encode("utf-8")
    header = struct.pack(
        "<BBBBIIBBHI",
        0xFF,
        0x00,
        0x00,
        0x00,
        session_id,
        seq,
        0x00,
        0x00,
        msg_id,
        len(payload),
    )
    return header + payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise EOFError("Conexão encerrada pela câmera.")
        buf.extend(chunk)
    return bytes(buf)


def recv_xm_packet(sock: socket.socket) -> tuple[int, int, int, bytes]:
    """Lê um quadro NetIP completo e retorna `(session_id, seq, msg_id, payload)`."""
    header = _recv_exact(sock, 20)
    _head, _ver, _r1, _r2, session_id, seq, _tot, _cur, msg_id, length = struct.unpack(
        "<BBBBIIBBHI", header
    )
    if length > 16 * 1024 * 1024:
        raise ValueError(f"Pacote NetIP inválido ({length} bytes).")
    body = _recv_exact(sock, length) if length else b""
    return session_id, seq, msg_id, body


def decode_xm_json(body: bytes) -> dict:
    """Decodifica a carga JSON terminada em nulo de pacotes de controle NetIP."""
    text = body.rstrip(b"\x00\r\n ").decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    return dict(json.loads(text))


def send_xm_command(
    sock: socket.socket, session_id: int, seq: int, msg_id: int, payload_obj: dict
) -> dict:
    """Envia comando NetIP JSON e lê a resposta JSON correspondente."""
    sock.sendall(pack_xm_packet(session_id, seq, msg_id, payload_obj))
    _sid, _rseq, _rmid, body = recv_xm_packet(sock)
    return decode_xm_json(body)


def extract_xm_video_nalus(stream_buf: bytearray) -> tuple[str | None, bytes]:
    """Extrai bytes Annex-B H.264/H.265 puros removendo cabeçalhos privados Xiongmai."""
    out = bytearray()
    detected_codec: str | None = None

    while len(stream_buf) >= 8:
        idx = stream_buf.find(b"\x00\x00\x01")
        if idx < 0:
            stream_buf.clear()
            break
        if idx > 0:
            del stream_buf[:idx]
        if len(stream_buf) < 8:
            break

        tag = stream_buf[3]
        if tag == 0xFC:  # I-frame (cabeçalho de 16 bytes)
            if len(stream_buf) < 16:
                break
            codec_byte = stream_buf[4]
            if codec_byte == 3:
                detected_codec = "hevc"
            elif codec_byte == 2:
                detected_codec = "h264"
            frame_len = struct.unpack_from("<I", stream_buf, 12)[0]
            if frame_len > 4_000_000:
                del stream_buf[:4]
                continue
            if len(stream_buf) < 16 + frame_len:
                break
            out.extend(stream_buf[16 : 16 + frame_len])
            del stream_buf[: 16 + frame_len]
        elif tag in (0xFD, 0xFE):  # P/B-frame (cabeçalho de 8 bytes)
            frame_len = struct.unpack_from("<I", stream_buf, 4)[0]
            if frame_len > 4_000_000:
                del stream_buf[:4]
                continue
            if len(stream_buf) < 8 + frame_len:
                break
            out.extend(stream_buf[8 : 8 + frame_len])
            del stream_buf[: 8 + frame_len]
        elif tag in (0xFA, 0xF9):  # Áudio G.711 ou metadados (cabeçalho de 8 bytes)
            frame_len = struct.unpack_from("<H", stream_buf, 6)[0]
            if len(stream_buf) < 8 + frame_len:
                break
            del stream_buf[: 8 + frame_len]
        else:
            del stream_buf[:4]

    return detected_codec, bytes(out)


def _open_control_session(
    ip: str,
    port: int,
    username: str,
    password: str,
    timeout: float,
    fetch_details: bool = True,
) -> tuple[socket.socket, int, str, dict, float]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, int(port)))
    try:
        login_req = {
            "EncryptType": "MD5",
            "LoginType": "DVRIP-Web",
            "PassWord": xm_sofia_hash(password),
            "UserName": username or "admin",
        }
        login_resp = send_xm_command(sock, 0, 0, _MSG_LOGIN_REQ, login_req)
        ret = int(login_resp.get("Ret", 0) or 0)
        if ret != 100:
            err_msg = _RET_ERRORS.get(ret, f"Falha no login NetIP (código Ret={ret}).")
            raise PermissionError(err_msg)

        sess_str = str(login_resp.get("SessionID") or "0x0")
        sid = int(sess_str, 16) if sess_str.lower().startswith("0x") else int(sess_str or 0)

        sys_info: dict = {}
        fps = 12.0
        if fetch_details:
            try:
                sys_resp = send_xm_command(
                    sock, sid, 1, _MSG_SYSINFO_REQ, {"Name": "SystemInfo", "SessionID": sess_str}
                )
                if int(sys_resp.get("Ret", 0) or 0) == 100 and isinstance(sys_resp.get("SystemInfo"), dict):
                    sys_info = sys_resp["SystemInfo"]
            except Exception:
                pass

            try:
                enc_resp = send_xm_command(
                    sock, sid, 2, _MSG_CONFIG_GET, {"Name": "Simplify.Encode", "SessionID": sess_str}
                )
                enc_list = enc_resp.get("Simplify.Encode")
                if isinstance(enc_list, list) and enc_list:
                    main_fmt = enc_list[0].get("MainFormat", {}).get("Video", {})
                    if main_fmt.get("FPS"):
                        fps = float(main_fmt["FPS"])
            except Exception:
                pass

        return sock, sid, sess_str, sys_info, fps
    except Exception:
        sock.close()
        raise


def _monitor_param(channel: int, stream_type: str) -> dict:
    xm_stream = "Main" if stream_type == "main" else "Extra1"
    return {
        "Channel": max(0, int(channel) - 1),
        "CombinMode": "NONE",
        "StreamType": xm_stream,
        "TransMode": "TCP",
    }


def probe_xiongmai_netip(
    ip: str,
    port: int = XM_DEFAULT_PORT,
    username: str = "admin",
    password: str = "",
    channel: int = 1,
    stream_type: str = "main",
    timeout: float = 6.0,
) -> XiongmaiProbeResult:
    """Conecta à câmera via NetIP 34567, autentica e decodifica o 1º quadro real de vídeo."""
    ctrl_sock: socket.socket | None = None
    media_sock: socket.socket | None = None
    try:
        ctrl_sock, sid, sess_str, sys_info, fps = _open_control_session(
            ip, port, username, password, timeout
        )
        model = str(sys_info.get("DeviceModel") or "")
        serial = str(sys_info.get("SerialNo") or "")

        media_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        media_sock.settimeout(timeout)
        media_sock.connect((ip, int(port)))

        param = _monitor_param(channel, stream_type)
        claim_resp = send_xm_command(
            media_sock,
            sid,
            1,
            _MSG_MONITOR_CLAIM,
            {"Name": "OPMonitor", "OPMonitor": {"Action": "Claim", "Parameter": param}, "SessionID": sess_str},
        )
        if int(claim_resp.get("Ret", 0) or 0) != 100:
            return XiongmaiProbeResult(
                ok=False,
                message="Login aceito, mas a câmera recusou abrir o canal de vídeo NetIP.",
                model=model,
                serial=serial,
            )

        start_resp = send_xm_command(
            ctrl_sock,
            sid,
            3,
            _MSG_MONITOR_REQ,
            {"Name": "OPMonitor", "OPMonitor": {"Action": "Start", "Parameter": param}, "SessionID": sess_str},
        )
        if int(start_resp.get("Ret", 0) or 0) != 100:
            return XiongmaiProbeResult(
                ok=False,
                message="Falha ao iniciar fluxo de vídeo NetIP na câmera.",
                model=model,
                serial=serial,
            )

        raw_buf = bytearray()
        annex_b = bytearray()
        codec: str | None = None
        chunks_seen = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not (codec is not None and chunks_seen >= 2 and len(annex_b) >= 4096):
            _rsid, _seq, mid, mdata = recv_xm_packet(media_sock)
            if mid == _MSG_MONITOR_DATA:
                raw_buf.extend(mdata)
                c, chunk = extract_xm_video_nalus(raw_buf)
                if c:
                    codec = c
                if chunk:
                    annex_b.extend(chunk)
                    chunks_seen += 1

        try:
            ctrl_sock.sendall(
                pack_xm_packet(
                    sid,
                    4,
                    _MSG_MONITOR_REQ,
                    {"Name": "OPMonitor", "OPMonitor": {"Action": "Stop", "Parameter": param}, "SessionID": sess_str},
                )
            )
        except OSError:
            pass

        if not annex_b:
            return XiongmaiProbeResult(
                ok=False,
                message="Câmera conectou na porta 34567, mas não enviou quadros de vídeo.",
                model=model,
                serial=serial,
            )

        width, height = _probe_annexb_dimensions(bytes(annex_b), codec or "hevc")
        desc_parts = [p for p in (model, f"Serial {serial}" if serial else "") if p]
        extra = f" ({', '.join(desc_parts)})" if desc_parts else ""
        res_str = f"{width}x{height}" if width and height else "vídeo ativo"
        return XiongmaiProbeResult(
            ok=True,
            message=f"Conectado via Xiongmai NetIP (porta {port}){extra}! {res_str} (~{fps:.0f} FPS).",
            width=width,
            height=height,
            fps=fps,
            model=model,
            serial=serial,
            codec=codec or "hevc",
        )
    except PermissionError as err:
        return XiongmaiProbeResult(ok=False, message=str(err))
    except OSError as err:
        return XiongmaiProbeResult(
            ok=False,
            message=f"Não foi possível conectar em {ip}:{port} via NetIP ({err}).",
        )
    finally:
        for s in (media_sock, ctrl_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


def _probe_annexb_dimensions(data: bytes, codec: str) -> tuple[int, int]:
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        codec,
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "bmp",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=data,
        capture_output=True,
        timeout=8.0,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.stdout:
        img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    return 0, 0


def find_xiongmai_by_serial(
    serial_no: str,
    local_ip: str,
    username: str = "admin",
    password: str = "",
    port: int = XM_DEFAULT_PORT,
    timeout: float = 0.35,
) -> tuple[str, dict] | None:
    """Procura na sub-rede local a câmera NetIP que possui o SerialNo informado."""
    target_serial = (serial_no or "").strip().lower()
    if not target_serial:
        return None
    cidr = f"{local_ip.rsplit('.', 1)[0]}.0/24"
    try:
        network = ipaddress.IPv4Network(cidr, strict=False)
    except ValueError:
        return None

    def _check_host(host_ip: str) -> tuple[str, dict] | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        is_open = False
        try:
            is_open = s.connect_ex((host_ip, port)) == 0
        except OSError:
            return None
        finally:
            s.close()
        if not is_open:
            return None

        try:
            ctrl, _sid, _sess, sys_info, _fps = _open_control_session(
                host_ip, port, username, password, timeout=3.0
            )
            ctrl.close()
            if str(sys_info.get("SerialNo") or "").strip().lower() == target_serial:
                return host_ip, sys_info
        except Exception:
            pass
        return None

    hosts = [str(h) for h in network.hosts() if str(h) != local_ip]
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(_check_host, h) for h in hosts]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                return res
    return None


class XiongmaiAnnexBReader:
    """Lê fluxo contínuo de vídeo Annex-B H.264/H.265 de uma câmera Xiongmai NetIP."""

    def __init__(
        self,
        ip: str,
        port: int = XM_DEFAULT_PORT,
        username: str = "admin",
        password: str = "",
        channel: int = 1,
        stream_type: str = "main",
        timeout_sec: float = 5.0,
    ) -> None:
        self.ip = ip
        self.port = int(port)
        self.username = username
        self.password = password
        self.channel = channel
        self.stream_type = stream_type
        self.timeout_sec = timeout_sec

        self.fps = 12.0
        self.codec = "hevc"
        self.last_error = ""

        self._stop_event = threading.Event()
        self._codec_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctrl_sock: socket.socket | None = None
        self._media_sock: socket.socket | None = None

    def start(self, on_chunk: Callable[[str, bytes], None]) -> bool:
        """Inicia leitura em background e aguarda conexão + detecção do codec."""
        self._stop_event.clear()
        self._codec_ready.clear()
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, args=(on_chunk,), daemon=True)
        self._thread.start()
        return self._codec_ready.wait(timeout=self.timeout_sec + 2.0)

    def stop(self) -> None:
        """Solicita parada do fluxo e fecha os sockets NetIP."""
        self._stop_event.set()
        for sock in (self._media_sock, self._ctrl_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self, on_chunk: Callable[[str, bytes], None]) -> None:
        ctrl_sock: socket.socket | None = None
        media_sock: socket.socket | None = None
        sid = 0
        sess_str = "0x0"
        param = _monitor_param(self.channel, self.stream_type)

        try:
            ctrl_sock, sid, sess_str, _sys_info, fps = _open_control_session(
                self.ip, self.port, self.username, self.password, self.timeout_sec
            )
            self.fps = fps
            self._ctrl_sock = ctrl_sock

            media_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            media_sock.settimeout(max(5.0, self.timeout_sec))
            media_sock.connect((self.ip, self.port))
            self._media_sock = media_sock

            claim = send_xm_command(
                media_sock,
                sid,
                1,
                _MSG_MONITOR_CLAIM,
                {"Name": "OPMonitor", "OPMonitor": {"Action": "Claim", "Parameter": param}, "SessionID": sess_str},
            )
            if int(claim.get("Ret", 0) or 0) != 100:
                raise RuntimeError("Câmera recusou Claim de vídeo NetIP.")

            start = send_xm_command(
                ctrl_sock,
                sid,
                3,
                _MSG_MONITOR_REQ,
                {"Name": "OPMonitor", "OPMonitor": {"Action": "Start", "Parameter": param}, "SessionID": sess_str},
            )
            if int(start.get("Ret", 0) or 0) != 100:
                raise RuntimeError("Câmera recusou Start de vídeo NetIP.")

            raw_buf = bytearray()
            last_keepalive = time.monotonic()
            seq_ka = 10

            while not self._stop_event.is_set():
                now = time.monotonic()
                if now - last_keepalive >= 20.0:
                    last_keepalive = now
                    seq_ka += 1
                    try:
                        ctrl_sock.sendall(
                            pack_xm_packet(
                                sid, seq_ka, _MSG_KEEPALIVE_REQ, {"Name": "KeepAlive", "SessionID": sess_str}
                            )
                        )
                    except OSError:
                        break

                _rsid, _seq, mid, mdata = recv_xm_packet(media_sock)
                if mid != _MSG_MONITOR_DATA:
                    continue
                raw_buf.extend(mdata)
                c, chunk = extract_xm_video_nalus(raw_buf)
                if c:
                    self.codec = c
                if chunk:
                    if not self._codec_ready.is_set():
                        self._codec_ready.set()
                    on_chunk(self.codec, chunk)
        except Exception as err:
            self.last_error = str(err)
        finally:
            if ctrl_sock is not None:
                try:
                    ctrl_sock.sendall(
                        pack_xm_packet(
                            sid,
                            999,
                            _MSG_MONITOR_REQ,
                            {
                                "Name": "OPMonitor",
                                "OPMonitor": {"Action": "Stop", "Parameter": param},
                                "SessionID": sess_str,
                            },
                        )
                    )
                except OSError:
                    pass
            for s in (media_sock, ctrl_sock):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            self._media_sock = None
            self._ctrl_sock = None
            self._codec_ready.set()


class XiongmaiVideoCapture:
    """Adaptador compatível com `cv2.VideoCapture` para fluxo ao vivo Xiongmai NetIP."""

    def __init__(self, dvrip_url: str, timeout_sec: float = 6.0) -> None:
        ip, port, user, pwd, ch, st = parse_dvrip_url(dvrip_url)
        self._reader = XiongmaiAnnexBReader(ip, port, user, pwd, ch, st, timeout_sec=timeout_sec)
        self._opened = False
        self._stopped = False
        self._proc: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None

        self._cond = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._read_seq = 0
        self._width = 0
        self._height = 0

        self._start_pipeline(timeout_sec)

    def _start_pipeline(self, timeout_sec: float) -> None:
        def _on_chunk(codec: str, chunk: bytes) -> None:
            proc = self._proc
            if proc is None:
                proc = self._launch_decoder(codec)
                if proc is None:
                    return
            if proc.poll() is not None or proc.stdin is None:
                return
            try:
                proc.stdin.write(chunk)
                proc.stdin.flush()
            except OSError:
                pass

        ready = self._reader.start(_on_chunk)
        if not ready or self._reader.last_error:
            self.release()
            return

        with self._cond:
            if self._latest_frame is None and not self._stopped:
                self._cond.wait(timeout=min(timeout_sec, 4.0))
            self._opened = (self._proc is not None and self._proc.poll() is None) or (self._latest_frame is not None)

    def _launch_decoder(self, codec: str) -> subprocess.Popen | None:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-probesize",
            "65536",
            "-analyzeduration",
            "150000",
            "-f",
            codec,
            "-i",
            "pipe:0",
            "-f",
            "image2pipe",
            "-vcodec",
            "bmp",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        self._proc = proc
        self._stdout_thread = threading.Thread(target=self._read_bmp_stdout, args=(proc,), daemon=True)
        self._stdout_thread.start()
        return proc

    def _read_bmp_stdout(self, proc: subprocess.Popen) -> None:
        buf = bytearray()
        try:
            assert proc.stdout is not None
            while not self._stopped:
                chunk = proc.stdout.read1(131072) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                while len(buf) >= 6:
                    if buf[:2] != b"BM":
                        idx = buf.find(b"BM")
                        if idx < 0:
                            buf.clear()
                            break
                        del buf[:idx]
                        continue
                    size = struct.unpack_from("<I", buf, 2)[0]
                    if size < 54 or size > 32 * 1024 * 1024:
                        del buf[:2]
                        continue
                    if len(buf) < size:
                        break
                    img = cv2.imdecode(np.frombuffer(bytes(buf[:size]), np.uint8), cv2.IMREAD_COLOR)
                    del buf[:size]
                    if img is not None:
                        h, w = img.shape[:2]
                        with self._cond:
                            self._width = w
                            self._height = h
                            self._latest_frame = img
                            self._frame_seq += 1
                            self._cond.notify_all()
        finally:
            with self._cond:
                self._cond.notify_all()

    def isOpened(self) -> bool:
        """Indica se o fluxo ao vivo NetIP está ativo."""
        if self._stopped or not self._opened:
            return False
        if self._proc is not None and self._proc.poll() is not None:
            return False
        return True

    def get(self, prop_id: int) -> float:
        """Retorna propriedades compatíveis com `cv2.CAP_PROP_*` (FPS, largura, altura)."""
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._reader.fps or 12.0)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Retorna o próximo quadro BGR decodificado mais recente."""
        with self._cond:
            deadline = time.monotonic() + 4.0
            while not self._stopped and self._frame_seq == self._read_seq:
                if self._reader.last_error:
                    return False, None
                if self._proc is not None and self._proc.poll() is not None:
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=remaining)

            if self._stopped or self._latest_frame is None:
                return False, None
            self._read_seq = self._frame_seq
            return True, self._latest_frame.copy()

    def release(self) -> None:
        """Encerra leitura NetIP, decodificador FFmpeg e fecha descritores."""
        if self._stopped:
            return
        self._stopped = True
        self._opened = False
        with self._cond:
            self._cond.notify_all()
        self._reader.stop()
        proc = self._proc
        self._proc = None
        if proc is not None:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._stdout_thread is not None and self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=2.0)


class XiongmaiPTZClient:
    """Cliente PTZ, Presets e Rastreamento (DetectTrack) via protocolo Xiongmai NetIP/DVRIP."""

    supports_firmware_tracking = True
    backend_label = "Xiongmai NetIP (34567)"

    def __init__(
        self,
        ip: str,
        port: int = XM_DEFAULT_PORT,
        username: str = "admin",
        password: str = "",
        channel: int = 1,
        timeout: float = 3.5,
    ) -> None:
        self.ip = ip.strip()
        self.port = int(port or XM_DEFAULT_PORT)
        self.username = username or "admin"
        self.password = password or ""
        self._channel = max(0, int(channel) - 1)
        self.timeout = float(timeout)

        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._sid = 0
        self._sess_str = "0x0"
        self._seq = 1
        self._last_direction = "DirectionLeft"

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._sid = 0
        self._sess_str = "0x0"
        self._seq = 1

    def close(self) -> None:
        """Fecha o socket de controle NetIP em cache."""
        with self._lock:
            self._close_locked()

    def _ensure_session_locked(self) -> None:
        if self._sock is not None:
            return
        sock, sid, sess_str, _sys_info, _fps = _open_control_session(
            self.ip,
            self.port,
            self.username,
            self.password,
            timeout=self.timeout,
            fetch_details=False,
        )
        self._sock = sock
        self._sid = sid
        self._sess_str = sess_str
        self._seq = 1

    def _request(self, msg_id: int, payload_builder: Callable[[str], dict]) -> dict:
        with self._lock:
            last_exc: Exception | None = None
            for _attempt in range(2):
                try:
                    self._ensure_session_locked()
                    assert self._sock is not None
                    seq = self._seq
                    self._seq += 1
                    payload = payload_builder(self._sess_str)
                    return send_xm_command(self._sock, self._sid, seq, msg_id, payload)
                except (OSError, EOFError, ValueError, json.JSONDecodeError) as exc:
                    last_exc = exc
                    self._close_locked()
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Falha desconhecida ao enviar comando NetIP.")

    def _send_ptz_command(
        self,
        command: str,
        *,
        preset: int = 65535,
        step: int = 0,
        menu_opts: str = "Enter",
        pattern: str = "SetBegin",
    ) -> bool:
        resp = self._request(
            _MSG_PTZ_CONTROL,
            lambda sess: {
                "Name": "OPPTZControl",
                "SessionID": sess,
                "OPPTZControl": {
                    "Command": command,
                    "Parameter": {
                        "AUX": {"Number": 0, "Status": "Off"},
                        "Channel": self._channel,
                        "MenuOpts": menu_opts,
                        "Pattern": pattern,
                        "Preset": int(preset),
                        "Step": int(step),
                        "Tour": 0,
                    },
                },
            },
        )
        return int(resp.get("Ret", 0) or 0) == 100

    def move_start(self, direction: str, speed: int = 5) -> bool:
        """Inicia movimento contínuo PTZ ou zoom até `move_stop`."""
        xm_cmd = PTZ_DIRECTIONS.get(direction.strip().lower())
        if not xm_cmd:
            return False
        spd = max(1, min(8, int(speed)))
        self._last_direction = xm_cmd
        return self._send_ptz_command(
            xm_cmd,
            preset=65535,
            step=spd,
            menu_opts="Enter",
            pattern="SetBegin",
        )

    def move_stop(self, direction: str = "") -> bool:
        """Para o movimento PTZ enviando `Exit`/`SetEnd` com `Preset=-1` na direção ativa."""
        xm_cmd = PTZ_DIRECTIONS.get(direction.strip().lower(), "") or self._last_direction or "DirectionLeft"
        return self._send_ptz_command(
            xm_cmd,
            preset=-1,
            step=0,
            menu_opts="Exit",
            pattern="SetEnd",
        )

    def set_preset(self, preset_id: int) -> bool:
        """Salva a posição atual da câmera no índice de preset informado."""
        pid = max(1, min(255, int(preset_id)))
        return self._send_ptz_command("SetPreset", preset=pid, step=0)

    def goto_preset(self, preset_id: int) -> bool:
        """Move a câmera para o preset salvo."""
        pid = max(1, min(255, int(preset_id)))
        return self._send_ptz_command("GotoPreset", preset=pid, step=0)

    def clear_preset(self, preset_id: int) -> bool:
        """Remove o preset salvo do índice informado."""
        pid = max(1, min(255, int(preset_id)))
        return self._send_ptz_command("ClearPreset", preset=pid, step=0)

    def get_presets(self) -> list[PTZPresetItem]:
        """Consulta `Uart.PTZPreset` e retorna os pontos pré-definidos ordenados por ID."""
        resp = self._request(
            _MSG_CONFIG_GET,
            lambda sess: {"Name": "Uart.PTZPreset", "SessionID": sess},
        )
        raw = resp.get("Uart.PTZPreset")
        if not isinstance(raw, list) or not raw:
            return []
        if isinstance(raw[0], list):
            items = raw[self._channel] if self._channel < len(raw) else raw[0]
        else:
            items = raw
        if not isinstance(items, list):
            return []

        by_id: dict[int, PTZPresetItem] = {}
        for entry in items:
            if not isinstance(entry, dict) or "Id" not in entry:
                continue
            if entry.get("Enable") is False:
                continue
            try:
                pid = int(entry["Id"])
            except (TypeError, ValueError):
                continue
            name = str(entry.get("PresetName") or "").strip()
            by_id[pid] = PTZPresetItem(id=pid, name=name)
        return [by_id[k] for k in sorted(by_id)]

    def _get_other_functions(self) -> dict:
        try:
            resp = self._request(
                _MSG_SYSTEM_ABILITY,
                lambda sess: {"Name": "SystemFunction", "SessionID": sess},
            )
            sys_fn = resp.get("SystemFunction")
            if isinstance(sys_fn, dict):
                other = sys_fn.get("OtherFunction")
                if isinstance(other, dict):
                    return other
        except Exception:
            pass
        return {}

    def _get_detect_track_raw(self) -> tuple[bool, dict]:
        try:
            resp = self._request(
                _MSG_CONFIG_GET,
                lambda sess: {"Name": "Detect.DetectTrack", "SessionID": sess},
            )
            ok = int(resp.get("Ret", 0) or 0) == 100
            raw = resp.get("Detect.DetectTrack")
            if isinstance(raw, list) and raw:
                raw = raw[self._channel] if self._channel < len(raw) else raw[0]
            if isinstance(raw, dict):
                return ok, raw
        except Exception:
            pass
        return False, {}

    def _parse_detect_track(self, other_fn: dict, ok: bool, cfg: dict) -> DetectTrackSettings:
        has_cfg = ok and bool(cfg) and "Enable" in cfg
        supported = bool(other_fn.get("SupportDetectTrack", False)) or has_cfg
        watch_supported = bool(other_fn.get("SupportSetDetectTrackWatchPoint", False))
        enable = bool(int(cfg.get("Enable", 0) or 0)) if has_cfg else False
        try:
            return_time = max(1, int(cfg.get("ReturnTime", 30) or 30)) if has_cfg else 30
        except (TypeError, ValueError):
            return_time = 30
        try:
            sens_val = cfg.get("Sensitivity", 1)
            sensitivity = max(0, min(2, int(1 if sens_val is None else sens_val))) if has_cfg else 1
        except (TypeError, ValueError):
            sensitivity = 1
        return DetectTrackSettings(
            supported=supported,
            watch_point_supported=watch_supported,
            enable=enable,
            return_time=return_time,
            sensitivity=sensitivity,
        )

    def get_detect_track(self) -> DetectTrackSettings:
        """Consulta as capacidades e configuração atual de `Detect.DetectTrack` na câmera."""
        other_fn = self._get_other_functions()
        ok, cfg = self._get_detect_track_raw()
        return self._parse_detect_track(other_fn, ok, cfg)

    def set_detect_track(self, enable: bool, return_time: int, sensitivity: int) -> bool:
        """Atualiza `Detect.DetectTrack` (habilitar rastreamento, tempo de retorno e sensibilidade)."""
        _ok, existing = self._get_detect_track_raw()
        payload_cfg = dict(existing) if isinstance(existing, dict) else {}
        payload_cfg["Enable"] = 1 if enable else 0
        payload_cfg["ReturnTime"] = max(5, min(3600, int(return_time)))
        payload_cfg["Sensitivity"] = max(0, min(2, int(sensitivity)))

        resp = self._request(
            _MSG_CONFIG_SET,
            lambda sess: {
                "Name": "Detect.DetectTrack",
                "SessionID": sess,
                "Detect.DetectTrack": payload_cfg,
            },
        )
        return int(resp.get("Ret", 0) or 0) == 100

    def get_status_snapshot(self) -> PTZStatusSnapshot:
        """Retorna snapshot único com presets salvos e status de auto-rastreamento."""
        other_fn = self._get_other_functions()
        ok, cfg = self._get_detect_track_raw()
        presets = self.get_presets()
        supported_ptz = bool(other_fn.get("SupportPTZDirectionControl", True))
        detect_track = self._parse_detect_track(other_fn, ok, cfg)
        return PTZStatusSnapshot(
            supported_ptz=supported_ptz,
            presets=presets,
            detect_track=detect_track,
        )

