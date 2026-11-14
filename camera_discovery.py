"""Descoberta automática de câmeras (ONVIF / rede local), perfis RTSP e teste de conexão."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import os
import socket
import urllib.error
from urllib.parse import parse_qs, quote, unquote, urlsplit, urlunsplit
import urllib.request
import uuid
import xml.etree.ElementTree as ET

import cv2

from config_manager import CameraConfig
from xiongmai_netip import (
    XM_DEFAULT_PORT,
    XiongmaiVideoCapture,
    build_dvrip_url,
    parse_dvrip_url,
    probe_xiongmai_netip,
)

_WS_DISCOVERY_MULTICAST = ("239.255.255.250", 3702)
_DEFAULT_SCAN_PORTS = (554, 8899, 8000, 37777, 34567, 2020)

_BRAND_SPECS: dict[str, dict[str, object]] = {
    "Xiongmai NetIP / iCSee (Porta 34567)": {
        "rtsp_port": XM_DEFAULT_PORT,
        "onvif_port": XM_DEFAULT_PORT,
        "builder": lambda ip, port, user, pwd, ch, st, _custom: build_dvrip_url(
            ip, port, user, pwd, ch, st
        ),
    },
    "Xiongmai / General / ICSee / XMEye": {
        "rtsp_port": 554,
        "onvif_port": 8899,
        "builder": lambda ip, port, user, pwd, ch, st, _custom: (
            f"rtsp://{ip}:{port}/user={quote(user, safe='')}"
            f"&password={quote(pwd, safe='')}"
            f"&channel={ch}&stream={'0' if st == 'main' else '1'}.sdp"
        ),
    },
    "Intelbras / Dahua": {
        "rtsp_port": 554,
        "onvif_port": 80,
        "builder": lambda ip, port, user, pwd, ch, st, _custom: (
            f"rtsp://{_format_userinfo(user, pwd)}{ip}:{port}"
            f"/cam/realmonitor?channel={ch}&subtype={'0' if st == 'main' else '1'}"
        ),
    },
    "Hikvision": {
        "rtsp_port": 554,
        "onvif_port": 80,
        "builder": lambda ip, port, user, pwd, ch, st, _custom: (
            f"rtsp://{_format_userinfo(user, pwd)}{ip}:{port}"
            f"/Streaming/Channels/{ch}0{'1' if st == 'main' else '2'}"
        ),
    },
    "TP-Link Tapo": {
        "rtsp_port": 554,
        "onvif_port": 2020,
        "builder": lambda ip, port, user, pwd, _ch, st, _custom: (
            f"rtsp://{_format_userinfo(user, pwd)}{ip}:{port}"
            f"/{'stream1' if st == 'main' else 'stream2'}"
        ),
    },
    "Genérico RTSP": {
        "rtsp_port": 554,
        "onvif_port": 80,
        "builder": lambda ip, port, user, pwd, ch, st, custom: _build_generic_rtsp(
            ip, port, user, pwd, ch, st, custom
        ),
    },
    "Webcam USB": {
        "rtsp_port": 554,
        "onvif_port": 80,
        "builder": lambda *_args: "",
    },
}

BRAND_NAMES: tuple[str, ...] = tuple(_BRAND_SPECS.keys())


def _format_userinfo(username: str, password: str) -> str:
    if not username:
        return ""
    u = quote(username, safe="")
    if password:
        return f"{u}:{quote(password, safe='')}@"
    return f"{u}@"


def _inject_rtsp_credentials(url: str, username: str, password: str) -> str:
    url = url.strip()
    if not url or not username:
        return url
    parts = urlsplit(url)
    if parts.scheme.lower() != "rtsp" or parts.username:
        return url
    host = parts.hostname or ""
    if not host:
        return url
    netloc = f"{_format_userinfo(username, password)}{host}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _build_generic_rtsp(
    ip: str,
    port: int,
    user: str,
    pwd: str,
    channel: int,
    stream_type: str,
    custom_url: str,
) -> str:
    if custom_url.strip():
        return _inject_rtsp_credentials(custom_url, user, pwd)
    stream_seg = "main" if stream_type == "main" else "sub"
    return (
        f"rtsp://{_format_userinfo(user, pwd)}{ip}:{port}"
        f"/h264/ch{channel}/{stream_seg}/av_stream"
    )


def get_brand_defaults(brand: str) -> tuple[int, int]:
    """Retorna `(porta_rtsp_padrao, porta_onvif_padrao)` para uma marca."""
    spec = _BRAND_SPECS.get(brand, _BRAND_SPECS["Genérico RTSP"])
    return int(spec["rtsp_port"]), int(spec["onvif_port"])


def build_rtsp_url(
    brand: str,
    ip: str,
    port: int = 554,
    username: str = "",
    password: str = "",
    channel: int = 1,
    stream_type: str = "main",
    custom_url: str = "",
) -> str:
    """Monta a URL RTSP a partir da marca selecionada e dos parâmetros da câmera."""
    spec = _BRAND_SPECS.get(brand, _BRAND_SPECS["Genérico RTSP"])
    builder = spec["builder"]
    clean_ip = ip.strip()
    if not clean_ip and not custom_url.strip():
        return ""
    return builder(clean_ip, int(port), username, password, max(1, int(channel)), stream_type, custom_url)


def resolve_camera_source(cam: CameraConfig) -> str | int:
    """Resolve a fonte efetiva de captura (índice USB, `dvrip://` ou URL RTSP) de uma câmera."""
    if cam.brand == "Webcam USB":
        return max(0, int(cam.usb_index))
    if cam.source_url_override.strip():
        return cam.source_url_override.strip()
    if cam.brand == "Xiongmai / General / ICSee / XMEye" and int(cam.rtsp_port) == XM_DEFAULT_PORT and cam.ip.strip():
        return build_dvrip_url(
            ip=cam.ip,
            port=cam.rtsp_port,
            username=cam.username,
            password=cam.password,
            channel=cam.channel,
            stream_type=cam.stream_type,
        )
    return build_rtsp_url(
        brand=cam.brand,
        ip=cam.ip,
        port=cam.rtsp_port,
        username=cam.username,
        password=cam.password,
        channel=cam.channel,
        stream_type=cam.stream_type,
        custom_url=cam.custom_rtsp_url,
    )


def _rtsp_to_dvrip_fallback_if_available(url: str, timeout: float = 1.5) -> str | None:
    """Se RTSP estiver fechado no host e NetIP (34567) aberto, converte para `dvrip://`."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() != "rtsp" or not parts.hostname:
        return None
    host = parts.hostname
    rtsp_port = parts.port or 554
    if _check_tcp_port(host, rtsp_port, timeout=timeout):
        return None
    if not _check_tcp_port(host, XM_DEFAULT_PORT, timeout=timeout):
        return None

    user = unquote(parts.username) if parts.username else "admin"
    pwd = unquote(parts.password) if parts.password else ""
    ch = 1
    st = "main"
    path_query = parts.path.lstrip("/")
    if "user=" in path_query:
        parsed_xm = parse_qs(path_query)
        user = parsed_xm.get("user", [user])[0]
        pwd = parsed_xm.get("password", [pwd])[0]
        try:
            ch = max(1, int(parsed_xm.get("channel", ["1"])[0]))
        except ValueError:
            ch = 1
        stream_raw = parsed_xm.get("stream", ["0.sdp"])[0]
        st = "sub" if stream_raw.startswith("1") else "main"
    return build_dvrip_url(host, XM_DEFAULT_PORT, user, pwd, ch, st)


def resolve_best_connection_source(cam: CameraConfig) -> str | int:
    """Resolve a fonte e aplica fallback automático RTSP -> Xiongmai NetIP 34567 se necessário."""
    src = resolve_camera_source(cam)
    if isinstance(src, str) and src.lower().startswith("rtsp://"):
        fallback = _rtsp_to_dvrip_fallback_if_available(src)
        if fallback:
            return fallback
    return src


def mask_rtsp_url(url: str) -> str:
    """Mascara senhas em URLs RTSP para exibição segura em logs e interface."""
    if not url or "://" not in url:
        return url
    masked = url
    parts = urlsplit(masked)
    if parts.password is not None and parts.hostname:
        user = parts.username or ""
        netloc = f"{user}:***@{parts.hostname}"
        if parts.port:
            netloc += f":{parts.port}"
        masked = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    if "password=" in masked:
        before, after = masked.split("password=", 1)
        if "&" in after:
            _, rest = after.split("&", 1)
            masked = f"{before}password=***&{rest}"
        else:
            masked = f"{before}password=***"
    return masked


@dataclass
class DiscoveredCamera:
    """Câmera encontrada na rede local via ONVIF ou varredura TCP."""

    ip: str
    source: str
    name: str = ""
    manufacturer: str = ""
    xaddr: str = ""
    onvif_port: int = 0
    rtsp_port: int = 554
    open_ports: list[int] = field(default_factory=list)
    suggested_brand: str = "Genérico RTSP"

    def label(self) -> str:
        """Formata um rótulo legível para seleção na interface gráfica."""
        title = self.name or self.manufacturer or self.suggested_brand
        ports_str = ",".join(str(p) for p in self.open_ports) if self.open_ports else str(self.onvif_port or 80)
        return f"{self.ip} — {title} ({self.source}; portas: {ports_str})"


@dataclass
class ConnectionTestResult:
    """Resultado detalhado do teste de conexão com a câmera."""

    ok: bool
    message: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    resolved_source: str = ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _guess_brand_from_info(text: str, ports: list[int]) -> str:
    lowered = text.lower()
    if 34567 in ports and 554 not in ports and "onvif" not in lowered:
        return "Xiongmai NetIP / iCSee (Porta 34567)"
    if any(k in lowered for k in ("xiongmai", "xmeye", "icsee", "general", "h.264", "netip")) or (
        34567 in ports or 8899 in ports
    ):
        return "Xiongmai / General / ICSee / XMEye"
    if any(k in lowered for k in ("intelbras", "dahua")) or 37777 in ports:
        return "Intelbras / Dahua"
    if "hikvision" in lowered or 8000 in ports:
        return "Hikvision"
    if any(k in lowered for k in ("tapo", "tp-link")) or 2020 in ports:
        return "TP-Link Tapo"
    return "Genérico RTSP"


def parse_onvif_probe_match(xml_bytes: bytes, sender_ip: str) -> DiscoveredCamera | None:
    """Interpreta uma resposta XML WS-Discovery ProbeMatches de câmera ONVIF."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    xaddrs_text = ""
    scopes_text = ""
    for elem in root.iter():
        name = _local_name(elem.tag)
        if name == "XAddrs" and elem.text and not xaddrs_text:
            xaddrs_text = elem.text.strip()
        elif name == "Scopes" and elem.text and not scopes_text:
            scopes_text = elem.text.strip()

    xaddr = xaddrs_text.split()[0] if xaddrs_text else ""
    if not xaddr and not sender_ip:
        return None

    parsed_url = urlsplit(xaddr) if xaddr else None
    host_ip = (parsed_url.hostname if parsed_url and parsed_url.hostname else sender_ip).strip()
    onvif_port = parsed_url.port if parsed_url and parsed_url.port else (80 if xaddr else 8899)

    cam_name = ""
    manufacturer = ""
    for token in scopes_text.split():
        decoded = unquote(token)
        if "/name/" in decoded:
            cam_name = decoded.split("/name/", 1)[-1].replace("_", " ").strip()
        elif "/hardware/" in decoded and not manufacturer:
            manufacturer = decoded.split("/hardware/", 1)[-1].strip()
        elif "/manufacturer/" in decoded:
            manufacturer = decoded.split("/manufacturer/", 1)[-1].strip()

    ports = [onvif_port] if onvif_port else []
    suggested = _guess_brand_from_info(f"{cam_name} {manufacturer} {scopes_text}", ports)
    return DiscoveredCamera(
        ip=host_ip,
        source="ONVIF",
        name=cam_name,
        manufacturer=manufacturer,
        xaddr=xaddr,
        onvif_port=onvif_port,
        rtsp_port=554,
        open_ports=ports,
        suggested_brand=suggested,
    )


def _build_ws_discovery_probe(probe_type: str) -> bytes:
    msg_id = f"urn:uuid:{uuid.uuid4()}"
    envelope = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl" '
        'xmlns:tds="http://www.onvif.org/ver10/device/wsdl">'
        "<e:Header>"
        f"<w:MessageID>{msg_id}</w:MessageID>"
        "<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
        "<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>"
        "</e:Header>"
        "<e:Body><d:Probe>"
        f"<d:Types>{probe_type}</d:Types>"
        "</d:Probe></e:Body>"
        "</e:Envelope>"
    )
    return envelope.encode("utf-8")


def discover_onvif_devices(timeout: float = 2.5) -> list[DiscoveredCamera]:
    """Envia broadcast/multicast WS-Discovery ONVIF na rede local e lista câmeras."""
    found: dict[str, DiscoveredCamera] = {}
    local_ip = get_local_ipv4()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        except OSError:
            pass
        sock.settimeout(0.4)
        for ptype in ("dn:NetworkVideoTransmitter", "tds:Device"):
            sock.sendto(_build_ws_discovery_probe(ptype), _WS_DISCOVERY_MULTICAST)

        deadline = datetime.now(timezone.utc).timestamp() + timeout
        while datetime.now(timezone.utc).timestamp() < deadline:
            try:
                data, (sender_ip, _) = sock.recvfrom(8192)
            except socket.timeout:
                continue
            cam = parse_onvif_probe_match(data, sender_ip)
            if cam and cam.ip not in found:
                found[cam.ip] = cam
    except OSError:
        pass
    finally:
        sock.close()
    return sorted(found.values(), key=lambda c: tuple(int(p) for p in c.ip.split(".") if p.isdigit()) or (0,))


def get_local_ipv4() -> str:
    """Obtém o endereço IPv4 local ativo na LAN."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "192.168.1.10"
    finally:
        sock.close()


def _check_tcp_port(ip: str, port: int, timeout: float) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _probe_host_ports(ip: str, ports: tuple[int, ...], timeout: float) -> list[int]:
    return [p for p in ports if _check_tcp_port(ip, p, timeout)]


def scan_lan_cameras(
    ports: tuple[int, ...] = _DEFAULT_SCAN_PORTS,
    timeout: float = 0.35,
    max_workers: int = 80,
    subnet_cidr: str | None = None,
) -> list[DiscoveredCamera]:
    """Varre a sub-rede local /24 procurando portas típicas de câmeras IP."""
    local_ip = get_local_ipv4()
    cidr = subnet_cidr or f"{local_ip.rsplit('.', 1)[0]}.0/24"
    try:
        network = ipaddress.IPv4Network(cidr, strict=False)
    except ValueError:
        return []

    hosts = [str(host) for host in network.hosts() if str(host) != local_ip]
    results: list[DiscoveredCamera] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_probe_host_ports, ip, ports, timeout): ip for ip in hosts}
        for future in as_completed(future_map):
            ip = future_map[future]
            open_ports = future.result()
            if open_ports:
                if XM_DEFAULT_PORT in open_ports and 554 not in open_ports:
                    rtsp_port = XM_DEFAULT_PORT
                    onvif_port = XM_DEFAULT_PORT
                else:
                    rtsp_port = 554
                    onvif_port = next((p for p in (8899, 2020, 8000, 80) if p in open_ports), 80)
                results.append(
                    DiscoveredCamera(
                        ip=ip,
                        source="LAN Scan",
                        onvif_port=onvif_port,
                        rtsp_port=rtsp_port,
                        open_ports=open_ports,
                        suggested_brand=_guess_brand_from_info("", open_ports),
                    )
                )
    return sorted(results, key=lambda c: tuple(int(p) for p in c.ip.split(".") if p.isdigit()) or (0,))


def discover_all_cameras(onvif_timeout: float = 2.0, scan_timeout: float = 0.35) -> list[DiscoveredCamera]:
    """Combina descoberta ONVIF WS-Discovery e varredura rápida TCP na rede local."""
    merged: dict[str, DiscoveredCamera] = {}
    for cam in discover_onvif_devices(timeout=onvif_timeout):
        merged[cam.ip] = cam
    for cam in scan_lan_cameras(timeout=scan_timeout):
        if cam.ip in merged:
            existing = merged[cam.ip]
            combined_ports = sorted(set(existing.open_ports + cam.open_ports))
            existing.open_ports = combined_ports
            if existing.suggested_brand == "Genérico RTSP":
                existing.suggested_brand = cam.suggested_brand
        else:
            merged[cam.ip] = cam
    return sorted(merged.values(), key=lambda c: tuple(int(p) for p in c.ip.split(".") if p.isdigit()) or (0,))


def _build_wsse_security_header(username: str, password: str, digest: bool = True) -> str:
    if not username:
        return ""
    nonce_bytes = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce_b64 = base64.b64encode(nonce_bytes).decode("ascii")
    if digest:
        digest_raw = hashlib.sha1(nonce_bytes + created.encode("utf-8") + password.encode("utf-8")).digest()
        pwd_val = base64.b64encode(digest_raw).decode("ascii")
        pwd_type = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
    else:
        pwd_val = password
        pwd_type = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText"
    return (
        '<s:Header><wsse:Security s:mustUnderstand="1" '
        'xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" '
        'xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
        "<wsse:UsernameToken>"
        f"<wsse:Username>{username}</wsse:Username>"
        f'<wsse:Password Type="{pwd_type}">{pwd_val}</wsse:Password>'
        '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f"{nonce_b64}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken></wsse:Security></s:Header>"
    )


def _soap_post(endpoint: str, body_xml: str, username: str, password: str, timeout: float) -> bytes | None:
    for use_digest in (True, False):
        header_xml = _build_wsse_security_header(username, password, digest=use_digest)
        envelope = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
            'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
            'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" '
            'xmlns:tt="http://www.onvif.org/ver10/schema">'
            f"{header_xml}<s:Body>{body_xml}</s:Body></s:Envelope>"
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=envelope,
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (400, 401, 500) and use_digest and username:
                continue
            return None
        except Exception:
            return None
    return None


def parse_onvif_capabilities_media_url(xml_bytes: bytes) -> str:
    """Extrai a URL do serviço de mídia ONVIF da resposta de GetCapabilities."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    in_media = False
    for elem in root.iter():
        name = _local_name(elem.tag)
        if name == "Media":
            in_media = True
            for child in elem:
                if _local_name(child.tag) == "XAddr" and child.text:
                    return child.text.strip()
        elif in_media and name == "XAddr" and elem.text:
            return elem.text.strip()
    return ""


def parse_onvif_profiles(xml_bytes: bytes) -> list[str]:
    """Extrai os tokens de perfil (Main/Sub stream) da resposta de GetProfiles."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    tokens: list[str] = []
    for elem in root.iter():
        if _local_name(elem.tag) == "Profiles":
            token = elem.attrib.get("token")
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def parse_onvif_stream_uri(xml_bytes: bytes) -> str:
    """Extrai a URI RTSP da resposta de GetStreamUri."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    for elem in root.iter():
        if _local_name(elem.tag) == "Uri" and elem.text:
            return elem.text.strip()
    return ""


def query_onvif_stream_uris(
    ip_or_xaddr: str,
    onvif_port: int = 8899,
    username: str = "admin",
    password: str = "",
    timeout: float = 3.5,
) -> list[str]:
    """Consulta GetCapabilities -> GetProfiles -> GetStreamUri via ONVIF SOAP."""
    target = ip_or_xaddr.strip()
    if not target:
        return []

    candidates: list[str] = []
    if target.startswith(("http://", "https://")):
        candidates.append(target)
        host = urlsplit(target).hostname or ""
    else:
        host = target

    if host:
        ports = [onvif_port, 8899, 80, 8000, 8080, 2020]
        seen_ports: set[int] = set()
        for p in ports:
            if p and p not in seen_ports:
                seen_ports.add(p)
                endpoint = f"http://{host}:{p}/onvif/device_service"
                if endpoint not in candidates:
                    candidates.append(endpoint)

    cap_body = "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>"
    profiles_body = "<trt:GetProfiles/>"

    uris: list[str] = []
    for device_url in candidates:
        cap_xml = _soap_post(device_url, cap_body, username, password, timeout)
        media_url = parse_onvif_capabilities_media_url(cap_xml) if cap_xml else ""
        if not media_url:
            media_url = device_url.replace("/device_service", "/media_service")

        prof_xml = _soap_post(media_url, profiles_body, username, password, timeout)
        if not prof_xml and media_url != device_url:
            prof_xml = _soap_post(device_url, profiles_body, username, password, timeout)
            if prof_xml:
                media_url = device_url
        if not prof_xml:
            continue

        tokens = parse_onvif_profiles(prof_xml)
        for token in tokens:
            stream_body = (
                "<trt:GetStreamUri>"
                "<trt:StreamSetup>"
                "<tt:Stream>RTP-Unicast</tt:Stream>"
                "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>"
                "</trt:StreamSetup>"
                f"<trt:ProfileToken>{token}</trt:ProfileToken>"
                "</trt:GetStreamUri>"
            )
            uri_xml = _soap_post(media_url, stream_body, username, password, timeout)
            if uri_xml:
                uri = parse_onvif_stream_uri(uri_xml)
                if uri:
                    final_uri = _inject_rtsp_credentials(uri, username, password)
                    if final_uri not in uris:
                        uris.append(final_uri)
        if uris:
            break
    return uris


def detect_usb_webcams(max_index: int = 4) -> list[int]:
    """Verifica índices locais de webcam USB disponíveis via OpenCV."""
    found: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        try:
            if cap.isOpened():
                found.append(idx)
        finally:
            cap.release()
    return found


def open_video_capture(source: str | int, timeout_ms: int = 5000) -> cv2.VideoCapture | XiongmaiVideoCapture:
    """Abre captura via Webcam USB, Xiongmai NetIP (`dvrip://`) ou RTSP/FFmpeg."""
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    if source.strip().lower().startswith("dvrip://"):
        return XiongmaiVideoCapture(source, timeout_sec=max(4.0, timeout_ms / 1000.0))

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;tcp|stimeout;{timeout_ms * 1000}"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def _test_dvrip_source(dvrip_url: str, timeout_sec: float) -> ConnectionTestResult:
    ip, port, user, pwd, ch, st = parse_dvrip_url(dvrip_url)
    res = probe_xiongmai_netip(
        ip=ip,
        port=port,
        username=user,
        password=pwd,
        channel=ch,
        stream_type=st,
        timeout=max(5.0, timeout_sec),
    )
    return ConnectionTestResult(
        ok=res.ok,
        message=res.message,
        width=res.width,
        height=res.height,
        fps=res.fps,
        resolved_source=mask_rtsp_url(dvrip_url),
    )


def test_camera_connection(source: str | int, timeout_sec: float = 5.0) -> ConnectionTestResult:
    """Testa conectividade TCP (RTSP ou Xiongmai NetIP) e captura real de quadro."""
    if isinstance(source, str):
        clean_src = source.strip()
        if not clean_src:
            return ConnectionTestResult(ok=False, message="Informe o IP da câmera ou a URL RTSP.")
        parsed = urlsplit(clean_src)
        scheme = parsed.scheme.lower()
        if scheme == "dvrip":
            return _test_dvrip_source(clean_src, timeout_sec)
        if scheme in ("rtsp", "http", "https"):
            host = parsed.hostname
            port = parsed.port or (554 if scheme == "rtsp" else 80)
            if not host:
                return ConnectionTestResult(
                    ok=False,
                    message="URL inválida: endereço IP/host não encontrado.",
                    resolved_source=mask_rtsp_url(clean_src),
                )
            if not _check_tcp_port(host, port, timeout=min(timeout_sec, 2.0)):
                fallback = _rtsp_to_dvrip_fallback_if_available(clean_src, timeout=1.5)
                if fallback:
                    return _test_dvrip_source(fallback, timeout_sec)
                return ConnectionTestResult(
                    ok=False,
                    message=(
                        f"Não foi possível alcançar {host}:{port}. "
                        "Verifique se a câmera está ligada no mesmo Wi-Fi/roteador e se a porta está correta."
                    ),
                    resolved_source=mask_rtsp_url(clean_src),
                )
        display_src = mask_rtsp_url(clean_src)
    else:
        display_src = f"Webcam USB #{source}"

    cap = open_video_capture(source, timeout_ms=int(timeout_sec * 1000))
    try:
        if not cap.isOpened():
            return ConnectionTestResult(
                ok=False,
                message=(
                    f"Conexão TCP aceita, mas o fluxo de vídeo não abriu ({display_src}). "
                    "Confira usuário/senha, marca selecionada ou tente Buscar URL via ONVIF."
                ),
                resolved_source=display_src,
            )

        frame = None
        for _ in range(5):
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
                break

        if frame is None:
            return ConnectionTestResult(
                ok=False,
                message="Câmera abriu o canal, mas não enviou imagem. Verifique senha, canal ou stream (Main/Sub).",
                resolved_source=display_src,
            )

        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or fps > 120:
            fps = 15.0
        return ConnectionTestResult(
            ok=True,
            message=f"Conectado com sucesso! Resolução {w}x{h} (~{fps:.0f} FPS).",
            width=w,
            height=h,
            fps=fps,
            resolved_source=display_src,
        )
    finally:
        cap.release()
