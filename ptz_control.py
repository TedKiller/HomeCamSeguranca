"""Abstração unificada de controle PTZ, pontos fixos (presets) e retorno pós-rastreamento."""

from __future__ import annotations

from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

from camera_discovery import (
    _check_tcp_port,
    _local_name,
    _soap_post,
    parse_onvif_capabilities_media_url,
    parse_onvif_profiles,
)
from config_manager import CameraConfig
from xiongmai_netip import (
    XM_DEFAULT_PORT,
    DetectTrackSettings,
    PTZPresetItem,
    PTZStatusSnapshot,
    XiongmaiPTZClient,
    parse_dvrip_url,
)

_ONVIF_VELOCITIES: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "left_up": (-0.7, 0.7, 0.0),
    "right_up": (0.7, 0.7, 0.0),
    "left_down": (-0.7, -0.7, 0.0),
    "right_down": (0.7, -0.7, 0.0),
    "zoom_in": (0.0, 0.0, 1.0),
    "zoom_out": (0.0, 0.0, -1.0),
}


def parse_onvif_ptz_url(xml_bytes: bytes) -> str:
    """Extrai a URL do serviço PTZ ONVIF da resposta de GetCapabilities."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    in_ptz = False
    for elem in root.iter():
        name = _local_name(elem.tag)
        if name == "PTZ":
            in_ptz = True
            for child in elem:
                if _local_name(child.tag) == "XAddr" and child.text:
                    return child.text.strip()
        elif in_ptz and name == "XAddr" and elem.text:
            return elem.text.strip()
    return ""


def parse_onvif_presets(xml_bytes: bytes) -> list[PTZPresetItem]:
    """Extrai presets da resposta SOAP GetPresets ONVIF."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    presets: list[PTZPresetItem] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "Preset":
            continue
        token = (elem.attrib.get("token") or "").strip()
        name = ""
        for child in elem:
            if _local_name(child.tag) == "Name" and child.text:
                name = child.text.strip()
                break
        if token.isdigit():
            pid = int(token)
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if not digits:
                continue
            pid = int(digits)
        presets.append(PTZPresetItem(id=pid, name=name or f"Ponto {pid}"))
    return sorted(presets, key=lambda p: p.id)


class OnvifPTZClient:
    """Cliente PTZ padrão via ONVIF SOAP (ContinuousMove, Stop, Presets)."""

    supports_firmware_tracking = False
    backend_label = "ONVIF PTZ"

    def __init__(
        self,
        ip: str,
        onvif_port: int = 80,
        username: str = "admin",
        password: str = "",
        timeout: float = 3.5,
    ) -> None:
        self.ip = ip.strip()
        self.onvif_port = int(onvif_port or 80)
        self.username = username
        self.password = password
        self.timeout = float(timeout)
        self._ptz_url = ""
        self._profile_token = ""

    def _ensure_endpoints(self) -> tuple[str, str]:
        if self._ptz_url and self._profile_token:
            return self._ptz_url, self._profile_token

        device_url = f"http://{self.ip}:{self.onvif_port}/onvif/device_service"
        cap_body = '<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>'
        cap_resp = _soap_post(device_url, cap_body, self.username, self.password, self.timeout)
        media_url = ""
        if cap_resp:
            self._ptz_url = parse_onvif_ptz_url(cap_resp)
            media_url = parse_onvif_capabilities_media_url(cap_resp)

        if not self._ptz_url:
            self._ptz_url = f"http://{self.ip}:{self.onvif_port}/onvif/ptz_service"
        if not media_url:
            media_url = f"http://{self.ip}:{self.onvif_port}/onvif/media_service"

        prof_resp = _soap_post(media_url, "<trt:GetProfiles/>", self.username, self.password, self.timeout)
        if prof_resp:
            tokens = parse_onvif_profiles(prof_resp)
            if tokens:
                self._profile_token = tokens[0]
        if not self._profile_token:
            self._profile_token = "Profile_1"
        return self._ptz_url, self._profile_token

    def move_start(self, direction: str, speed: int = 5) -> bool:
        vec = _ONVIF_VELOCITIES.get(direction.lower())
        if not vec:
            return False
        ptz_url, token = self._ensure_endpoints()
        scale = round(max(1, min(8, int(speed))) / 8.0, 2)
        px, py, zz = vec[0] * scale, vec[1] * scale, vec[2] * scale
        vel_xml = ""
        if px != 0.0 or py != 0.0:
            vel_xml += f'<tt:PanTilt x="{px:.2f}" y="{py:.2f}"/>'
        if zz != 0.0:
            vel_xml += f'<tt:Zoom x="{zz:.2f}"/>'
        body = (
            "<tptz:ContinuousMove>"
            f"<tptz:ProfileToken>{token}</tptz:ProfileToken>"
            f"<tptz:Velocity>{vel_xml}</tptz:Velocity>"
            "</tptz:ContinuousMove>"
        )
        return _soap_post(ptz_url, body, self.username, self.password, self.timeout) is not None

    def move_stop(self, direction: str = "") -> bool:
        _ = direction
        ptz_url, token = self._ensure_endpoints()
        body = (
            "<tptz:Stop>"
            f"<tptz:ProfileToken>{token}</tptz:ProfileToken>"
            "<tptz:PanTilt>true</tptz:PanTilt>"
            "<tptz:Zoom>true</tptz:Zoom>"
            "</tptz:Stop>"
        )
        return _soap_post(ptz_url, body, self.username, self.password, self.timeout) is not None

    def set_preset(self, preset_id: int) -> bool:
        ptz_url, token = self._ensure_endpoints()
        pid = int(preset_id)
        body = (
            "<tptz:SetPreset>"
            f"<tptz:ProfileToken>{token}</tptz:ProfileToken>"
            f"<tptz:PresetName>Ponto{pid}</tptz:PresetName>"
            f"<tptz:PresetToken>{pid}</tptz:PresetToken>"
            "</tptz:SetPreset>"
        )
        return _soap_post(ptz_url, body, self.username, self.password, self.timeout) is not None

    def goto_preset(self, preset_id: int) -> bool:
        ptz_url, token = self._ensure_endpoints()
        body = (
            "<tptz:GotoPreset>"
            f"<tptz:ProfileToken>{token}</tptz:ProfileToken>"
            f"<tptz:PresetToken>{int(preset_id)}</tptz:PresetToken>"
            "</tptz:GotoPreset>"
        )
        return _soap_post(ptz_url, body, self.username, self.password, self.timeout) is not None

    def clear_preset(self, preset_id: int) -> bool:
        ptz_url, token = self._ensure_endpoints()
        body = (
            "<tptz:RemovePreset>"
            f"<tptz:ProfileToken>{token}</tptz:ProfileToken>"
            f"<tptz:PresetToken>{int(preset_id)}</tptz:PresetToken>"
            "</tptz:RemovePreset>"
        )
        return _soap_post(ptz_url, body, self.username, self.password, self.timeout) is not None

    def get_presets(self) -> list[PTZPresetItem]:
        ptz_url, token = self._ensure_endpoints()
        body = f"<tptz:GetPresets><tptz:ProfileToken>{token}</tptz:ProfileToken></tptz:GetPresets>"
        resp = _soap_post(ptz_url, body, self.username, self.password, self.timeout)
        return parse_onvif_presets(resp) if resp else []

    def get_detect_track(self) -> DetectTrackSettings:
        return DetectTrackSettings(supported=False, watch_point_supported=False)

    def set_detect_track(self, enable: bool, return_time: int, sensitivity: int) -> bool:
        _ = (enable, return_time, sensitivity)
        return False

    def get_status_snapshot(self) -> PTZStatusSnapshot:
        return PTZStatusSnapshot(
            supported_ptz=True,
            presets=self.get_presets(),
            detect_track=self.get_detect_track(),
        )

    def close(self) -> None:
        return


def create_ptz_backend(
    cam: CameraConfig,
    resolved_source: str | int,
) -> XiongmaiPTZClient | OnvifPTZClient | None:
    """Cria o cliente PTZ adequado (Xiongmai NetIP 34567 ou ONVIF) para a câmera."""
    if cam.brand == "Webcam USB" or isinstance(resolved_source, int):
        return None

    src_str = str(resolved_source or "").strip()
    if src_str.lower().startswith("dvrip://"):
        host, port, user, pwd, ch, _st = parse_dvrip_url(src_str)
        if not host:
            return None
        return XiongmaiPTZClient(host, port=port, username=user, password=pwd, channel=ch)

    host = cam.ip.strip()
    if not host and src_str.lower().startswith("rtsp://"):
        host = (urlsplit(src_str).hostname or "").strip()
    if not host:
        return None

    if "xiongmai" in cam.brand.lower() or _check_tcp_port(host, XM_DEFAULT_PORT, timeout=0.8):
        if _check_tcp_port(host, XM_DEFAULT_PORT, timeout=0.8):
            return XiongmaiPTZClient(
                host,
                port=XM_DEFAULT_PORT,
                username=cam.username or "admin",
                password=cam.password,
                channel=cam.channel,
            )

    onvif_port = int(cam.onvif_port or 80)
    if onvif_port == XM_DEFAULT_PORT:
        onvif_port = 8899
    return OnvifPTZClient(
        host,
        onvif_port=onvif_port,
        username=cam.username or "admin",
        password=cam.password,
    )

