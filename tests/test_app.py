"""Testes automatizados do HomeCam Segurança."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_discovery import (
    _guess_brand_from_info,
    build_rtsp_url,
    mask_rtsp_url,
    parse_onvif_capabilities_media_url,
    parse_onvif_probe_match,
    parse_onvif_profiles,
    parse_onvif_stream_uri,
    query_onvif_stream_uris,
    resolve_best_connection_source,
    resolve_camera_source,
    scan_lan_cameras,
    test_camera_connection,
)
from config_manager import (
    AppConfig,
    CameraConfig,
    ConfigManager,
    find_virtual_drive_subdir,
    normalize_drive_folder_dir,
    protect_secret,
    resolve_writable_recordings_dir,
    sanitize_filename_part,
    unprotect_secret,
)
from gdrive_uploader import GoogleDriveUploader, validate_oauth_client_json
from ptz_control import (
    OnvifPTZClient,
    create_ptz_backend,
    parse_onvif_presets,
    parse_onvif_ptz_url,
)
from recorder import MotionDetector, VideoRecorder, _segment_muxer_args, is_valid_mp4_file, save_snapshot
from xiongmai_netip import (
    XM_DEFAULT_PORT,
    XM_WATCH_PRESET_ID,
    DetectTrackSettings,
    PTZPresetItem,
    PTZStatusSnapshot,
    XiongmaiPTZClient,
    build_dvrip_url,
    extract_xm_video_nalus,
    find_xiongmai_by_serial,
    pack_xm_packet,
    parse_dvrip_url,
    probe_xiongmai_netip,
    recv_xm_packet,
    xm_sofia_hash,
)


def _create_sample_mp4(path: Path, frames: int = 12, width: int = 160, height: int = 120) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"Falha ao criar vídeo de teste em {path}"
    try:
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (20, 40, 60)
            cv2.rectangle(frame, (10 + i * 5, 30), (40 + i * 5, 70), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


class _FakeCredentials:
    valid = True
    expired = False
    refresh_token = None


class _FakeReq:
    def __init__(self, result: object) -> None:
        self._result = result

    def execute(self) -> object:
        return self._result


class _FakeUploadReq:
    def __init__(self, response: dict | None, fail_error: Exception | None = None) -> None:
        self._response = response
        self._fail_error = fail_error
        self._chunk = 0

    def next_chunk(self) -> tuple[object, dict | None]:
        if self._fail_error is not None:
            raise self._fail_error
        self._chunk += 1
        if self._chunk == 1:
            return None, None
        return None, self._response


class _FakeFilesResource:
    def __init__(self, fail_upload: bool = False) -> None:
        self.folders: dict[tuple[str, str], str] = {}
        self.uploaded: list[dict] = []
        self.fail_upload = fail_upload
        self._seq = 1

    def list(self, q: str = "", **_kwargs: object) -> _FakeReq:
        # Extrai nome entre name='...' e parent '...' in parents
        name = ""
        parent = "root"
        if "name='" in q:
            name = q.split("name='", 1)[1].split("'", 1)[0]
        if "' in parents" in q:
            left = q.split("' in parents", 1)[0]
            parent = left.rsplit("'", 1)[-1]
        fid = self.folders.get((parent, name))
        return _FakeReq({"files": [{"id": fid, "name": name}] if fid else []})

    def create(self, body: dict, media_body: object = None, **_kwargs: object) -> object:
        if body.get("mimeType") == "application/vnd.google-apps.folder":
            parent = (body.get("parents") or ["root"])[0]
            name = body["name"]
            fid = f"folder_{self._seq}"
            self._seq += 1
            self.folders[(parent, name)] = fid
            return _FakeReq({"id": fid})

        if self.fail_upload:
            return _FakeUploadReq(None, fail_error=RuntimeError("Falha simulada de rede"))

        fid = f"file_{self._seq}"
        self._seq += 1
        self.uploaded.append({"id": fid, "body": body, "media_body": media_body})
        return _FakeUploadReq({"id": fid, "name": body.get("name")})


class _FakeAboutResource:
    def get(self, **_kwargs: object) -> _FakeReq:
        return _FakeReq({"user": {"emailAddress": "morador@gmail.com"}})


class _FakeDriveService:
    def __init__(self, fail_upload: bool = False) -> None:
        self.files_res = _FakeFilesResource(fail_upload=fail_upload)

    def files(self) -> _FakeFilesResource:
        return self.files_res

    def about(self) -> _FakeAboutResource:
        return _FakeAboutResource()


class ConfigAndSecurityTests(unittest.TestCase):
    def test_dpapi_secret_roundtrip_and_disk_protection(self) -> None:
        secret = "S3nh@Segura#ç!"
        encrypted = protect_secret(secret)
        if sys.platform == "win32":
            self.assertTrue(encrypted.startswith("dpapi:"))
            self.assertNotIn(secret, encrypted)
        self.assertEqual(unprotect_secret(encrypted), secret)

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            mgr = ConfigManager(cfg_path)
            cfg = AppConfig(
                cameras=[
                    CameraConfig(
                        id="c1",
                        name="Portão",
                        ip="192.168.1.50",
                        password=secret,
                        serial_number="54550d137d64efea",
                    )
                ],
                selected_camera_id="c1",
            )
            mgr.save(cfg)

            raw_disk = cfg_path.read_text(encoding="utf-8")
            if sys.platform == "win32":
                self.assertNotIn(secret, raw_disk)

            loaded = mgr.load()
            self.assertEqual(loaded.cameras[0].password, secret)
            self.assertEqual(loaded.cameras[0].serial_number, "54550d137d64efea")
            self.assertEqual(loaded.selected_camera_id, "c1")

    def test_sanitize_filename_part(self) -> None:
        self.assertEqual(sanitize_filename_part('Sala<>:"/\\|?* 1.'), "Sala_ 1")
        self.assertEqual(sanitize_filename_part("   ...   ", "Fallback"), "Fallback")


class CameraDiscoveryTests(unittest.TestCase):
    def test_brand_rtsp_presets_and_masking(self) -> None:
        xm = build_rtsp_url("Xiongmai / General / ICSee / XMEye", "192.168.1.20", 554, "admin", "p@ss", 1, "main")
        self.assertEqual(xm, "rtsp://192.168.1.20:554/user=admin&password=p%40ss&channel=1&stream=0.sdp")
        self.assertEqual(
            mask_rtsp_url(xm),
            "rtsp://192.168.1.20:554/user=admin&password=***&channel=1&stream=0.sdp",
        )

        intel = build_rtsp_url("Intelbras / Dahua", "192.168.1.30", 554, "admin", "abc123", 1, "sub")
        self.assertEqual(intel, "rtsp://admin:abc123@192.168.1.30:554/cam/realmonitor?channel=1&subtype=1")
        self.assertEqual(
            mask_rtsp_url(intel),
            "rtsp://admin:***@192.168.1.30:554/cam/realmonitor?channel=1&subtype=1",
        )

        hik = build_rtsp_url("Hikvision", "192.168.1.40", 554, "admin", "123", 2, "main")
        self.assertEqual(hik, "rtsp://admin:123@192.168.1.40:554/Streaming/Channels/201")

        tapo = build_rtsp_url("TP-Link Tapo", "192.168.1.60", 554, "camuser", "pw", 1, "sub")
        self.assertEqual(tapo, "rtsp://camuser:pw@192.168.1.60:554/stream2")

        usb_cam = CameraConfig(brand="Webcam USB", usb_index=2)
        self.assertEqual(resolve_camera_source(usb_cam), 2)

    def test_onvif_xml_parsers_and_local_soap_server(self) -> None:
        probe_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
                    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
          <s:Body>
            <d:ProbeMatches>
              <d:ProbeMatch>
                <d:Scopes>onvif://www.onvif.org/name/Camera_Garagem onvif://www.onvif.org/hardware/Xiongmai_IPC</d:Scopes>
                <d:XAddrs>http://192.168.1.77:8899/onvif/device_service</d:XAddrs>
              </d:ProbeMatch>
            </d:ProbeMatches>
          </s:Body>
        </s:Envelope>"""
        discovered = parse_onvif_probe_match(probe_xml, "192.168.1.77")
        self.assertIsNotNone(discovered)
        assert discovered is not None
        self.assertEqual(discovered.ip, "192.168.1.77")
        self.assertEqual(discovered.onvif_port, 8899)
        self.assertEqual(discovered.name, "Camera Garagem")
        self.assertEqual(discovered.suggested_brand, "Xiongmai / General / ICSee / XMEye")

        port_holder: dict[str, int] = {}

        class _MockOnvifHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8", errors="ignore")
                p = port_holder["port"]
                if "GetCapabilities" in body:
                    resp = (
                        f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
                        f'<s:Body><tds:GetCapabilitiesResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
                        f'xmlns:tt="http://www.onvif.org/ver10/schema">'
                        f'<tds:Capabilities><tt:Media><tt:XAddr>http://127.0.0.1:{p}/onvif/media_service</tt:XAddr>'
                        f'</tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse></s:Body></s:Envelope>'
                    )
                elif "GetProfiles" in body:
                    resp = (
                        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
                        '<s:Body><trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">'
                        '<trt:Profiles token="main_000"/><trt:Profiles token="sub_001"/>'
                        '</trt:GetProfilesResponse></s:Body></s:Envelope>'
                    )
                else:
                    uri = (
                        "rtsp://127.0.0.1:554/stream0"
                        if "main_000" in body
                        else "rtsp://127.0.0.1:554/stream1"
                    )
                    resp = (
                        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
                        '<s:Body><trt:GetStreamUriResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
                        'xmlns:tt="http://www.onvif.org/ver10/schema">'
                        f'<trt:MediaUri><tt:Uri>{uri}</tt:Uri></trt:MediaUri>'
                        '</trt:GetStreamUriResponse></s:Body></s:Envelope>'
                    )
                raw = resp.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/soap+xml; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = HTTPServer(("127.0.0.1", 0), _MockOnvifHandler)
        port_holder["port"] = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            uris = query_onvif_stream_uris(
                "127.0.0.1",
                onvif_port=port_holder["port"],
                username="admin",
                password="secret",
                timeout=2.0,
            )
            self.assertEqual(
                uris,
                [
                    "rtsp://admin:secret@127.0.0.1:554/stream0",
                    "rtsp://admin:secret@127.0.0.1:554/stream1",
                ],
            )
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(
            parse_onvif_capabilities_media_url(
                b'<X><Media><XAddr>http://10.0.0.2/media</XAddr></Media></X>'
            ),
            "http://10.0.0.2/media",
        )
        self.assertEqual(
            parse_onvif_profiles(b'<X><Profiles token="p1"/><Profiles token="p2"/></X>'),
            ["p1", "p2"],
        )
        self.assertEqual(
            parse_onvif_stream_uri(b"<X><Uri>rtsp://cam/live</Uri></X>"),
            "rtsp://cam/live",
        )

    def test_connection_probe_closed_port_and_valid_video(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]
        sock.close()

        res_fail = test_camera_connection(f"rtsp://127.0.0.1:{unused_port}/live", timeout_sec=1.0)
        self.assertFalse(res_fail.ok)
        self.assertIn("127.0.0.1", res_fail.message)

        with tempfile.TemporaryDirectory() as tmp:
            sample = _create_sample_mp4(Path(tmp) / "probe.mp4", frames=6, width=160, height=120)
            res_ok = test_camera_connection(str(sample), timeout_sec=3.0)
            self.assertTrue(res_ok.ok)
            self.assertEqual(res_ok.width, 160)
            self.assertEqual(res_ok.height, 120)

    def test_netip_preset_fallback_and_lan_scan_suggestion(self) -> None:
        dvrip = build_rtsp_url(
            "Xiongmai NetIP / iCSee (Porta 34567)",
            "192.168.1.2",
            34567,
            "admin",
            "QAZ@qaz2026",
            1,
            "main",
        )
        self.assertEqual(
            dvrip,
            "dvrip://admin:QAZ%40qaz2026@192.168.1.2:34567?channel=1&stream=main",
        )
        self.assertEqual(
            mask_rtsp_url(dvrip),
            "dvrip://admin:***@192.168.1.2:34567?channel=1&stream=main",
        )

        legacy_cam = CameraConfig(
            brand="Xiongmai / General / ICSee / XMEye",
            ip="192.168.1.2",
            rtsp_port=554,
            username="admin",
            password="QAZ@qaz2026",
        )
        with mock.patch(
            "camera_discovery._check_tcp_port",
            side_effect=lambda _ip, port, timeout=1.5: port == XM_DEFAULT_PORT,
        ):
            best = resolve_best_connection_source(legacy_cam)
            self.assertEqual(
                best,
                "dvrip://admin:QAZ%40qaz2026@192.168.1.2:34567?channel=1&stream=main",
            )

            found = scan_lan_cameras(
                ports=(554, 34567),
                timeout=0.05,
                max_workers=4,
                subnet_cidr="192.0.2.0/30",
            )
            self.assertEqual(len(found), 2)
            self.assertEqual(found[0].rtsp_port, XM_DEFAULT_PORT)
            self.assertEqual(found[0].onvif_port, XM_DEFAULT_PORT)
            self.assertEqual(found[0].suggested_brand, "Xiongmai NetIP / iCSee (Porta 34567)")
            self.assertEqual(_guess_brand_from_info("", [34567]), "Xiongmai NetIP / iCSee (Porta 34567)")


class XiongmaiNetIPTests(unittest.TestCase):
    def test_sofia_hash_dvrip_url_and_packet_roundtrip(self) -> None:
        self.assertEqual(len(xm_sofia_hash("QAZ@qaz2026")), 8)
        self.assertEqual(xm_sofia_hash("QAZ@qaz2026"), xm_sofia_hash("QAZ@qaz2026"))
        self.assertNotEqual(xm_sofia_hash("QAZ@qaz2026"), xm_sofia_hash("outra"))

        url = build_dvrip_url("192.168.1.2", 34567, "admin", "QAZ@qaz2026", channel=2, stream_type="sub")
        self.assertEqual(parse_dvrip_url(url), ("192.168.1.2", 34567, "admin", "QAZ@qaz2026", 2, "sub"))

        s1, s2 = socket.socketpair()
        try:
            s1.sendall(pack_xm_packet(0x1A2B, 7, 1000, {"UserName": "admin", "Ret": 100}))
            sid, seq, msg_id, body = recv_xm_packet(s2)
            self.assertEqual((sid, seq, msg_id), (0x1A2B, 7, 1000))
            self.assertEqual(json.loads(body.rstrip(b"\x00\r\n ").decode("utf-8"))["UserName"], "admin")
        finally:
            s1.close()
            s2.close()

    def test_extract_xm_video_nalus_strips_headers_and_audio(self) -> None:
        iframe_payload = b"\x00\x00\x00\x01\x40\x01"
        pframe_payload = b"\x00\x00\x00\x01\x02\x01"
        audio_payload = b"\x55\xaa\x55\xaa"

        iframe_hdr = b"\x00\x00\x01\xfc" + bytes([3, 12, 120, 68]) + b"\x00\x00\x00\x00" + struct.pack("<I", len(iframe_payload))
        pframe_hdr = b"\x00\x00\x01\xfd" + struct.pack("<I", len(pframe_payload))
        audio_hdr = b"\x00\x00\x01\xfa\x0e\x02" + struct.pack("<H", len(audio_payload))

        buf = bytearray(b"junk" + iframe_hdr + iframe_payload + audio_hdr + audio_payload + pframe_hdr + pframe_payload)
        codec, nalus = extract_xm_video_nalus(buf)
        self.assertEqual(codec, "hevc")
        self.assertEqual(nalus, iframe_payload + pframe_payload)
        self.assertEqual(bytes(buf), b"")

    def test_mock_netip_serial_lookup_and_auth_error(self) -> None:
        expected_hash = xm_sofia_hash("QAZ@qaz2026")
        stop_evt = threading.Event()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        srv.settimeout(0.2)
        mock_port = srv.getsockname()[1]

        def handle_client(conn: socket.socket) -> None:
            with conn:
                while not stop_evt.is_set():
                    try:
                        _sid, seq, mid, body = recv_xm_packet(conn)
                    except (EOFError, OSError):
                        break
                    req = json.loads(body.rstrip(b"\x00\r\n ").decode("utf-8"))
                    if mid == 1000:
                        if req.get("PassWord") == expected_hash:
                            resp = {"Ret": 100, "SessionID": "0x00000042"}
                        else:
                            resp = {"Ret": 105, "SessionID": "0x0"}
                    elif mid == 1020:
                        resp = {
                            "Ret": 100,
                            "SystemInfo": {
                                "DeviceModel": "X2C-WQ",
                                "SerialNo": "54550d137d64efea",
                            },
                        }
                    elif mid == 1042:
                        resp = {
                            "Ret": 100,
                            "Simplify.Encode": [{"MainFormat": {"Video": {"FPS": 12}}}],
                        }
                    else:
                        resp = {"Ret": 100}
                    conn.sendall(pack_xm_packet(0x42, seq, mid + 1, resp))

        def accept_loop() -> None:
            while not stop_evt.is_set():
                try:
                    c, _addr = srv.accept()
                    threading.Thread(target=handle_client, args=(c,), daemon=True).start()
                except socket.timeout:
                    continue
                except OSError:
                    break

        t = threading.Thread(target=accept_loop, daemon=True)
        t.start()
        try:
            class _SingleHostNet:
                def hosts(self) -> list[ipaddress.IPv4Address]:
                    return [ipaddress.IPv4Address("127.0.0.1")]

            with mock.patch("xiongmai_netip.ipaddress.IPv4Network", return_value=_SingleHostNet()):
                found = find_xiongmai_by_serial(
                    "54550d137d64efea",
                    local_ip="127.0.0.2",
                    username="admin",
                    password="QAZ@qaz2026",
                    port=mock_port,
                    timeout=0.5,
                )
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found[0], "127.0.0.1")
            self.assertEqual(found[1]["SerialNo"], "54550d137d64efea")

            bad_probe = probe_xiongmai_netip(
                "127.0.0.1",
                port=mock_port,
                username="admin",
                password="wrong",
                timeout=1.5,
            )
            self.assertFalse(bad_probe.ok)
            self.assertIn("Senha incorreta", bad_probe.message)
        finally:
            stop_evt.set()
            srv.close()
            t.join(timeout=2.0)


class RecorderTests(unittest.TestCase):
    def test_motion_detector_and_motion_recording(self) -> None:
        detector = MotionDetector(sensitivity=0.015, warmup_frames=4)
        static = np.full((120, 160, 3), 40, dtype=np.uint8)
        for _ in range(6):
            moved, _ = detector.update(static)
            self.assertFalse(moved)

        moving = static.copy()
        cv2.rectangle(moving, (20, 20), (110, 95), (255, 255, 255), -1)
        moved, ratio = detector.update(moving)
        self.assertTrue(moved)
        self.assertGreater(ratio, 0.015)

        emitted: list[tuple[Path, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "gravacoes"
            recorder = VideoRecorder(
                recordings_root=rec_root,
                on_segment_ready=lambda p, cam: emitted.append((p, cam)),
            )
            recorder.start(
                source=0,
                camera_name="Garagem",
                segment_minutes=1,
                motion_only=True,
                motion_sensitivity=0.015,
                motion_post_seconds=5,
                fps=10.0,
            )
            for _ in range(5):
                recorder.feed_frame(static)
            for _ in range(6):
                recorder.feed_frame(moving)
            recorder.stop()

            self.assertEqual(len(emitted), 1)
            clip_path, cam_name = emitted[0]
            self.assertEqual(cam_name, "Garagem")
            self.assertTrue(clip_path.exists())
            self.assertGreater(clip_path.stat().st_size, 0)

            snap_path = save_snapshot(moving, rec_root, "Garagem")
            self.assertTrue(snap_path.exists())
            self.assertGreater(snap_path.stat().st_size, 0)

    def test_ffmpeg_segmented_recording_on_local_video(self) -> None:
        emitted: list[tuple[Path, str]] = []
        done = threading.Event()

        def on_ready(p: Path, cam: str) -> None:
            emitted.append((p, cam))
            done.set()

        with tempfile.TemporaryDirectory() as tmp:
            src_video = _create_sample_mp4(Path(tmp) / "source.mp4", frames=20, width=160, height=120)
            rec_root = Path(tmp) / "gravacoes"
            recorder = VideoRecorder(recordings_root=rec_root, on_segment_ready=on_ready)
            recorder.start(source=str(src_video), camera_name="Portão Frente", segment_minutes=1, motion_only=False)
            self.assertTrue(done.wait(timeout=15.0), "FFmpeg não concluiu a gravação segmentada a tempo.")
            recorder.stop()

            self.assertGreaterEqual(len(emitted), 1)
            out_file, cam_name = emitted[0]
            self.assertEqual(cam_name, "Portão Frente")
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 0)


class GoogleDriveUploaderTests(unittest.TestCase):
    def test_persistent_queue_upload_and_delete_after_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pending_json = tmp_path / "pending_uploads.json"
            creds_json = tmp_path / "credentials.json"
            creds_json.write_text("{}", encoding="utf-8")

            video_dir = tmp_path / "gravacoes" / "Entrada" / "2026-11-12"
            video_dir.mkdir(parents=True)
            video_file = _create_sample_mp4(video_dir / "Entrada_01.mp4", frames=5)

            fake_service = _FakeDriveService(fail_upload=False)
            uploader = GoogleDriveUploader(
                pending_path=pending_json,
                credentials_path=creds_json,
                token_path=tmp_path / "token.json",
                root_folder_name="MinhaCasa",
                delete_after_upload=True,
                service_factory=lambda _c: fake_service,
            )
            uploader._creds = _FakeCredentials()
            self.assertTrue(uploader.enqueue_file(video_file, camera_name="Entrada"))
            self.assertFalse(uploader.enqueue_file(video_file, camera_name="Entrada"))
            self.assertEqual(uploader.get_pending_count(), 1)

            # Verifica persistência recriando o uploader
            reloaded = GoogleDriveUploader(
                pending_path=pending_json,
                credentials_path=creds_json,
                token_path=tmp_path / "token.json",
                root_folder_name="MinhaCasa",
                delete_after_upload=True,
                service_factory=lambda _c: fake_service,
            )
            reloaded._creds = _FakeCredentials()
            self.assertEqual(reloaded.get_pending_count(), 1)
            self.assertTrue(reloaded.process_next_pending())
            self.assertEqual(reloaded.get_pending_count(), 0)
            self.assertFalse(video_file.exists(), "Arquivo local deveria ter sido apagado após upload confirmado.")

            self.assertEqual(len(fake_service.files_res.uploaded), 1)
            uploaded_body = fake_service.files_res.uploaded[0]["body"]
            self.assertEqual(uploaded_body["name"], "Entrada_01.mp4")
            self.assertIn(("root", "MinhaCasa"), fake_service.files_res.folders)

    def test_upload_failure_keeps_local_file_and_records_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pending_json = tmp_path / "pending_uploads.json"
            video_file = _create_sample_mp4(tmp_path / "Entrada_err.mp4", frames=4)

            failing_service = _FakeDriveService(fail_upload=True)
            uploader = GoogleDriveUploader(
                pending_path=pending_json,
                credentials_path=tmp_path / "credentials.json",
                token_path=tmp_path / "token.json",
                delete_after_upload=True,
                service_factory=lambda _c: failing_service,
            )
            uploader._creds = _FakeCredentials()
            self.assertTrue(uploader.enqueue_file(video_file, camera_name="Quintal", date_folder="2026-11-12"))

            self.assertFalse(uploader.process_next_pending())
            self.assertTrue(video_file.exists(), "Arquivo local não pode ser apagado quando upload falha.")
            jobs = uploader.get_pending_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].attempts, 1)
            self.assertIn("Falha simulada", jobs[0].last_error)

    def test_validate_oauth_client_json_and_folder_sync_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            valid_json = tmp_path / "client_secret_ok.json"
            valid_json.write_text(
                json.dumps({"installed": {"client_id": "abc.apps.googleusercontent.com", "client_secret": "sec"}}),
                encoding="utf-8",
            )
            ok_v, _msg_v = validate_oauth_client_json(valid_json)
            self.assertTrue(ok_v)

            sa_json = tmp_path / "sa.json"
            sa_json.write_text(json.dumps({"type": "service_account", "client_id": "123"}), encoding="utf-8")
            ok_sa, msg_sa = validate_oauth_client_json(sa_json)
            self.assertFalse(ok_sa)
            self.assertIn("Conta de serviço", msg_sa)

            bad_json = tmp_path / "bad.json"
            bad_json.write_text("{invalid", encoding="utf-8")
            ok_bad, _ = validate_oauth_client_json(bad_json)
            self.assertFalse(ok_bad)

            drive_mount = tmp_path / "MeuDrive"
            drive_mount.mkdir()
            src_video = _create_sample_mp4(tmp_path / "Cam1_01.mp4", frames=4)

            uploader = GoogleDriveUploader(
                pending_path=tmp_path / "pending.json",
                root_folder_name="HomeCamSeguranca",
                delete_after_upload=True,
                sync_mode="folder",
                local_drive_dir=drive_mount,
            )
            self.assertTrue(uploader.is_ready())
            self.assertTrue(uploader.enqueue_file(src_video, camera_name="Frente", date_folder="2026-11-12"))
            self.assertTrue(uploader.process_next_pending())

            dest = drive_mount / "HomeCamSeguranca" / "Frente" / "2026-11-12" / "Cam1_01.mp4"
            self.assertTrue(dest.exists())
            self.assertGreater(dest.stat().st_size, 0)
            self.assertFalse(src_video.exists())

            missing_uploader = GoogleDriveUploader(
                pending_path=tmp_path / "pending2.json",
                sync_mode="folder",
                local_drive_dir=tmp_path / "NaoExiste",
            )
            self.assertFalse(missing_uploader.is_ready())
            src2 = _create_sample_mp4(tmp_path / "Cam1_02.mp4", frames=4)
            self.assertTrue(missing_uploader.enqueue_file(src2, camera_name="Frente", date_folder="2026-11-12"))
            self.assertFalse(missing_uploader.process_next_pending())
            self.assertTrue(src2.exists())

            cfg_mgr = ConfigManager(tmp_path / "cfg.json")
            cfg = cfg_mgr.load()
            cfg.drive.sync_mode = "folder"
            cfg.drive.local_drive_dir = str(drive_mount)
            cfg_mgr.save(cfg)
            reloaded_cfg = cfg_mgr.load()
            self.assertEqual(reloaded_cfg.drive.sync_mode, "folder")
            self.assertEqual(reloaded_cfg.drive.local_drive_dir, str(drive_mount))


class PTZAndFixedPointTests(unittest.TestCase):
    def test_xiongmai_ptz_client_commands_presets_detect_track_and_reconnect(self) -> None:
        recorded_cmds: list[tuple[int, dict]] = []
        detect_state = {"Enable": 1, "ReturnTime": 300, "Sensitivity": 2}
        conn_counter = 0
        stop_event = threading.Event()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        srv.settimeout(0.25)
        port = srv.getsockname()[1]

        def handle_client(conn: socket.socket, conn_idx: int) -> None:
            sess_hex = f"0x{0x100 + conn_idx:08x}"
            sess_num = 0x100 + conn_idx
            drop_after_next = conn_idx == 1
            op_count = 0
            with conn:
                conn.settimeout(2.0)
                while not stop_event.is_set():
                    try:
                        _sid, seq, mid, body = recv_xm_packet(conn)
                    except (EOFError, OSError):
                        break
                    req = json.loads(body.rstrip(b"\x00\r\n ").decode("utf-8"))
                    recorded_cmds.append((mid, req))

                    if mid == 1000:
                        resp = {"Ret": 100, "SessionID": sess_hex, "AliveInterval": 20}
                    elif mid == 1400:
                        op_count += 1
                        resp = {"Ret": 100, "SessionID": sess_hex}
                        if drop_after_next and op_count >= 5:
                            conn.sendall(pack_xm_packet(sess_num, seq, mid + 1, resp))
                            break
                    elif mid == 1042 and req.get("Name") == "Uart.PTZPreset":
                        resp = {
                            "Ret": 100,
                            "Name": "Uart.PTZPreset",
                            "SessionID": sess_hex,
                            "Uart.PTZPreset": [[{"Id": 100, "PresetName": "Posição100"}, {"Id": 99}]],
                        }
                    elif mid == 1360:
                        resp = {
                            "Ret": 100,
                            "Name": "SystemFunction",
                            "SessionID": sess_hex,
                            "SystemFunction": {
                                "OtherFunction": {
                                    "SupportPTZDirectionControl": True,
                                    "SupportDetectTrack": True,
                                    "SupportSetDetectTrackWatchPoint": True,
                                }
                            },
                        }
                    elif mid == 1042 and req.get("Name") == "Detect.DetectTrack":
                        resp = {
                            "Ret": 100,
                            "Name": "Detect.DetectTrack",
                            "SessionID": sess_hex,
                            "Detect.DetectTrack": dict(detect_state),
                        }
                    elif mid == 1040 and req.get("Name") == "Detect.DetectTrack":
                        detect_state.update(req["Detect.DetectTrack"])
                        resp = {"Ret": 100, "Name": "Detect.DetectTrack", "SessionID": sess_hex}
                    else:
                        resp = {"Ret": 100, "SessionID": sess_hex}

                    conn.sendall(pack_xm_packet(sess_num, seq, mid + 1, resp))

        def accept_loop() -> None:
            nonlocal conn_counter
            while not stop_event.is_set():
                try:
                    c, _ = srv.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                conn_counter += 1
                threading.Thread(target=handle_client, args=(c, conn_counter), daemon=True).start()

        accept_thread = threading.Thread(target=accept_loop, daemon=True)
        accept_thread.start()
        try:
            client = XiongmaiPTZClient("127.0.0.1", port=port, username="admin", password="123", timeout=2.0)
            self.assertTrue(client.move_start("left", speed=6))
            self.assertTrue(client.move_stop())
            self.assertTrue(client.set_preset(XM_WATCH_PRESET_ID))
            self.assertTrue(client.goto_preset(XM_WATCH_PRESET_ID))
            self.assertTrue(client.clear_preset(1))

            # O mock encerrou a 1a conexão após o 5o OPPTZControl -> deve reconectar e tentar novamente
            presets = client.get_presets()
            self.assertEqual(presets, [PTZPresetItem(id=99, name=""), PTZPresetItem(id=100, name="Posição100")])

            snap = client.get_status_snapshot()
            self.assertIsInstance(snap, PTZStatusSnapshot)
            self.assertTrue(snap.supported_ptz)
            self.assertEqual(
                snap.detect_track,
                DetectTrackSettings(
                    supported=True,
                    watch_point_supported=True,
                    enable=True,
                    return_time=300,
                    sensitivity=2,
                ),
            )

            self.assertTrue(client.set_detect_track(True, 20, 1))
            updated = client.get_detect_track()
            self.assertTrue(updated.enable)
            self.assertEqual(updated.return_time, 20)
            self.assertEqual(updated.sensitivity, 1)
            client.close()

            ptz_cmds = [req["OPPTZControl"] for mid, req in recorded_cmds if mid == 1400]
            self.assertEqual(ptz_cmds[0]["Command"], "DirectionLeft")
            self.assertEqual(ptz_cmds[0]["Parameter"]["Preset"], 65535)
            self.assertEqual(ptz_cmds[0]["Parameter"]["Step"], 6)
            self.assertEqual(ptz_cmds[0]["Parameter"]["MenuOpts"], "Enter")
            self.assertEqual(ptz_cmds[0]["Parameter"]["Pattern"], "SetBegin")

            self.assertEqual(ptz_cmds[1]["Command"], "DirectionLeft")
            self.assertEqual(ptz_cmds[1]["Parameter"]["Preset"], -1)
            self.assertEqual(ptz_cmds[1]["Parameter"]["Step"], 0)
            self.assertEqual(ptz_cmds[1]["Parameter"]["MenuOpts"], "Exit")
            self.assertEqual(ptz_cmds[1]["Parameter"]["Pattern"], "SetEnd")

            self.assertEqual(ptz_cmds[2]["Command"], "SetPreset")
            self.assertEqual(ptz_cmds[2]["Parameter"]["Preset"], 100)
            self.assertEqual(ptz_cmds[3]["Command"], "GotoPreset")
            self.assertEqual(ptz_cmds[3]["Parameter"]["Preset"], 100)
            self.assertEqual(ptz_cmds[4]["Command"], "ClearPreset")
            self.assertEqual(ptz_cmds[4]["Parameter"]["Preset"], 1)
            self.assertGreaterEqual(conn_counter, 2)
        finally:
            stop_event.set()
            srv.close()
            accept_thread.join(timeout=1.5)

    def test_onvif_ptz_client_soap_and_routing(self) -> None:
        soap_actions: list[str] = []
        server_port_box = [0]

        class OnvifMockHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length).decode("utf-8", errors="replace")
                p = server_port_box[0]
                if "GetCapabilities" in body:
                    soap_actions.append("GetCapabilities")
                    xml = (
                        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                        'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
                        'xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>'
                        '<tds:GetCapabilitiesResponse><tds:Capabilities>'
                        f'<tt:Media><tt:XAddr>http://127.0.0.1:{p}/onvif/media_service</tt:XAddr></tt:Media>'
                        f'<tt:PTZ><tt:XAddr>http://127.0.0.1:{p}/onvif/ptz_service</tt:XAddr></tt:PTZ>'
                        '</tds:Capabilities></tds:GetCapabilitiesResponse>'
                        '</s:Body></s:Envelope>'
                    )
                elif "GetProfiles" in body:
                    soap_actions.append("GetProfiles")
                    xml = (
                        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                        'xmlns:trt="http://www.onvif.org/ver10/media/wsdl"><s:Body>'
                        '<trt:GetProfilesResponse><trt:Profiles token="MainProfile"/></trt:GetProfilesResponse>'
                        '</s:Body></s:Envelope>'
                    )
                elif "ContinuousMove" in body:
                    soap_actions.append("ContinuousMove")
                    xml = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>'
                elif "<tptz:Stop>" in body:
                    soap_actions.append("Stop")
                    xml = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>'
                elif "SetPreset" in body:
                    soap_actions.append("SetPreset")
                    xml = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>'
                elif "GotoPreset" in body:
                    soap_actions.append("GotoPreset")
                    xml = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>'
                elif "GetPresets" in body:
                    soap_actions.append("GetPresets")
                    xml = (
                        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                        'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" '
                        'xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>'
                        '<tptz:GetPresetsResponse>'
                        '<tptz:Preset token="100"><tt:Name>Vigia</tt:Name></tptz:Preset>'
                        '<tptz:Preset token="2"><tt:Name>Portao</tt:Name></tptz:Preset>'
                        '</tptz:GetPresetsResponse></s:Body></s:Envelope>'
                    )
                else:
                    soap_actions.append("Other")
                    xml = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body/></s:Envelope>'

                raw = xml.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/soap+xml; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args: object) -> None:
                return

        srv = HTTPServer(("127.0.0.1", 0), OnvifMockHandler)
        server_port_box[0] = srv.server_port
        srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        srv_thread.start()
        try:
            onvif_cam = CameraConfig(
                name="Entrada",
                brand="Hikvision",
                ip="127.0.0.1",
                onvif_port=srv.server_port,
                username="admin",
                password="123",
            )
            with mock.patch("ptz_control._check_tcp_port", return_value=False):
                backend = create_ptz_backend(onvif_cam, f"rtsp://admin:123@127.0.0.1:554/Streaming/Channels/101")
            self.assertIsInstance(backend, OnvifPTZClient)
            assert isinstance(backend, OnvifPTZClient)
            self.assertFalse(backend.supports_firmware_tracking)
            self.assertTrue(backend.move_start("right_up", speed=4))
            self.assertTrue(backend.move_stop())
            self.assertTrue(backend.set_preset(100))
            self.assertTrue(backend.goto_preset(100))
            presets = backend.get_presets()
            self.assertEqual(presets, [PTZPresetItem(id=2, name="Portao"), PTZPresetItem(id=100, name="Vigia")])
            self.assertEqual(
                soap_actions,
                ["GetCapabilities", "GetProfiles", "ContinuousMove", "Stop", "SetPreset", "GotoPreset", "GetPresets"],
            )

            cap_xml = (
                f'<Caps><PTZ><XAddr>http://127.0.0.1:{srv.server_port}/onvif/ptz</XAddr></PTZ></Caps>'.encode("utf-8")
            )
            self.assertEqual(parse_onvif_ptz_url(cap_xml), f"http://127.0.0.1:{srv.server_port}/onvif/ptz")
            self.assertEqual(parse_onvif_presets(b"<broken"), [])

            usb_cam = CameraConfig(name="USB", brand="Webcam USB", usb_index=0)
            self.assertIsNone(create_ptz_backend(usb_cam, 0))
            xm_cam = CameraConfig(name="XM", brand="Xiongmai NetIP / iCSee (Porta 34567)", ip="192.168.1.2")
            xm_backend = create_ptz_backend(xm_cam, "dvrip://admin:pw@192.168.1.2:34567?channel=1&stream=main")
            self.assertIsInstance(xm_backend, XiongmaiPTZClient)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_config_manager_ptz_fields_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ConfigManager(Path(tmp) / "cfg.json")
            cfg = mgr.load()
            cfg.cameras = [
                CameraConfig(
                    name="Varanda",
                    ip="192.168.1.50",
                    ptz_speed=7,
                    ptz_home_preset=100,
                    ptz_auto_return_enabled=True,
                    ptz_auto_return_seconds=45,
                )
            ]
            cfg.selected_camera_id = cfg.cameras[0].id
            mgr.save(cfg)

            loaded = mgr.load()
            cam = mgr.get_selected_camera(loaded)
            self.assertEqual(cam.ptz_speed, 7)
            self.assertEqual(cam.ptz_home_preset, 100)
            self.assertTrue(cam.ptz_auto_return_enabled)
            self.assertEqual(cam.ptz_auto_return_seconds, 45)

    def test_virtual_drive_root_resolution_and_unwritable_recorder_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            virt_drive = root / "FakeDriveRoot"
            meu_drive = virt_drive / "Meu Drive"
            meu_drive.mkdir(parents=True)
            fallback_dir = root / "gravacoes_fallback"

            self.assertEqual(find_virtual_drive_subdir(virt_drive), meu_drive)

            def fake_writable(candidate: Path | str) -> bool:
                return Path(candidate).resolve() != virt_drive.resolve()

            with mock.patch("config_manager.is_directory_writable", side_effect=fake_writable):
                norm_drive, drive_changed = normalize_drive_folder_dir(virt_drive)
                self.assertTrue(drive_changed)
                self.assertEqual(Path(norm_drive), meu_drive)

                rec_oauth, ch1 = resolve_writable_recordings_dir(
                    virt_drive, prefer_drive_subdir=False, fallback=fallback_dir
                )
                self.assertTrue(ch1)
                self.assertEqual(rec_oauth, fallback_dir)

                rec_local, ch2 = resolve_writable_recordings_dir(
                    virt_drive, prefer_drive_subdir=True, fallback=fallback_dir
                )
                self.assertTrue(ch2)
                self.assertEqual(rec_local, meu_drive / "HomeCamGravacoes")

                cfg_path = root / "cfg.json"
                cfg_path.write_text(
                    json.dumps(
                        {
                            "recording": {"recordings_dir": str(virt_drive)},
                            "drive": {"enabled": True, "sync_mode": "oauth", "local_drive_dir": str(virt_drive)},
                        }
                    ),
                    encoding="utf-8",
                )
                loaded = ConfigManager(cfg_path).load()
                self.assertEqual(Path(loaded.drive.local_drive_dir), meu_drive)
                self.assertNotEqual(Path(loaded.recording.recordings_dir), virt_drive)

            rec = VideoRecorder(root / "blocked")
            with mock.patch("recorder.build_recording_dir", side_effect=PermissionError("Somente leitura")):
                with self.assertRaises(PermissionError):
                    rec.start("dvrip://admin@127.0.0.1:34567?channel=1&stream=main", "Frente")
            self.assertFalse(rec.is_recording)

    def test_save_snapshot_supports_unicode_camera_and_folder_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uni_root = Path(tmp) / "segurança_ção"
            img = np.zeros((120, 160, 3), dtype=np.uint8)
            img[:, :] = (40, 120, 200)
            out_path = save_snapshot(img, uni_root, "Câmera Principal")
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
            decoded = cv2.imdecode(np.frombuffer(out_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            assert decoded is not None
            self.assertEqual(decoded.shape[:2], (120, 160))

    def test_is_valid_mp4_file_and_fragmented_muxer_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_mp4 = _create_sample_mp4(root / "valid.mp4", frames=6, width=160, height=120)
            self.assertTrue(is_valid_mp4_file(valid_mp4))

            # Simula MP4 interrompido (ftyp + mdat, sem átomo moov)
            broken_no_moov = root / "broken.mp4"
            ftyp = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
            mdat = b"\x00\x00\x00\x40mdat" + (b"\xaa" * (0x40 - 8))
            broken_no_moov.write_bytes(ftyp + mdat)
            self.assertFalse(is_valid_mp4_file(broken_no_moov))

            # Simula MP4 fragmentado vazio (ftyp + moov inicial, sem moof/mdat)
            empty_frag = root / "empty_frag.mp4"
            moov_empty = b"\x00\x00\x00\x20moov" + (b"\x00" * (0x20 - 8))
            empty_frag.write_bytes(ftyp + moov_empty)
            self.assertFalse(is_valid_mp4_file(empty_frag))

            args = _segment_muxer_args(120, str(root / "out_%Y-%m-%d.mp4"))
            joined = " ".join(args)
            self.assertIn("movflags=+frag_keyframe+empty_moov+default_base_moof", joined)
            self.assertIn("-segment_format mp4", joined)

    def test_recorder_start_does_not_blacklist_leftovers_and_skips_corrupt_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "gravacoes"
            day_dir = rec_root / "Frente" / "2026-11-12"
            day_dir.mkdir(parents=True)

            valid_leftover = _create_sample_mp4(day_dir / "Frente_2026-11-12_23-00-00.mp4", frames=4)
            corrupt_leftover = day_dir / "Frente_2026-11-12_23-18-08.mp4"
            corrupt_leftover.write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
                + b"\x00\x00\x00\x40mdat"
                + (b"\x11" * (0x40 - 8))
            )

            emitted: list[Path] = []
            logs: list[str] = []
            rec = VideoRecorder(
                recordings_root=rec_root,
                on_segment_ready=lambda p, _c: emitted.append(p),
                on_log=logs.append,
            )

            def fake_loop() -> None:
                rec._stop_event.wait(0.1)

            with mock.patch.object(rec, "_run_dvrip_ffmpeg_loop", side_effect=fake_loop):
                rec.start(
                    source="dvrip://admin@127.0.0.1:34567?channel=1&stream=main",
                    camera_name="Frente",
                    segment_minutes=2,
                    motion_only=False,
                )
                fname, size_b, _elapsed, seg_s = rec.get_active_recording_stats()
                self.assertTrue(fname.endswith(".mp4"))
                self.assertGreater(size_b, 0)
                self.assertEqual(seg_s, 120)
                rec.stop()

            self.assertEqual([p.resolve() for p in emitted], [valid_leftover.resolve()])
            self.assertEqual(rec.get_active_recording_stats()[0], "")

    def test_uploader_history_and_auto_connect_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_mgr = ConfigManager(tmp_path / "cfg.json")
            cfg = cfg_mgr.load()
            self.assertFalse(cfg.recording.auto_connect_on_startup)
            cfg.recording.auto_connect_on_startup = True
            cfg_mgr.save(cfg)
            self.assertTrue(cfg_mgr.load().recording.auto_connect_on_startup)

            src_video = _create_sample_mp4(tmp_path / "Cam1_01.mp4", frames=4)
            fake_service = _FakeDriveService(fail_upload=False)
            pending_json = tmp_path / "pending.json"
            uploader1 = GoogleDriveUploader(
                pending_path=pending_json,
                credentials_path=tmp_path / "creds.json",
                token_path=tmp_path / "token.json",
                delete_after_upload=False,
                service_factory=lambda _c: fake_service,
            )
            uploader1._creds = _FakeCredentials()
            self.assertTrue(uploader1.enqueue_file(src_video, camera_name="Sala", date_folder="2026-11-12"))
            self.assertTrue(uploader1.process_next_pending())
            self.assertTrue(src_video.exists())
            self.assertTrue(uploader1.was_uploaded(src_video))

            uploader2 = GoogleDriveUploader(
                pending_path=pending_json,
                credentials_path=tmp_path / "creds.json",
                token_path=tmp_path / "token.json",
                delete_after_upload=False,
                service_factory=lambda _c: fake_service,
            )
            uploader2._creds = _FakeCredentials()
            self.assertFalse(uploader2.enqueue_file(src_video, camera_name="Sala", date_folder="2026-11-12"))
            self.assertEqual(uploader2.get_pending_count(), 0)

    def test_homecam_app_orphan_recovery_and_quarantine(self) -> None:
        import tkinter as tk
        from app_gui import HomeCamApp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rec_root = tmp_path / "gravacoes"
            day_dir = rec_root / "Garagem" / "2026-11-12"
            day_dir.mkdir(parents=True)

            valid_old = _create_sample_mp4(day_dir / "Garagem_2026-11-12_22-00-00.mp4", frames=4)
            broken_old = day_dir / "Garagem_2026-11-12_23-18-08.mp4"
            broken_old.write_bytes(
                b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
                + b"\x00\x00\x00\x40mdat"
                + (b"\x22" * (0x40 - 8))
            )
            old_ts = time.time() - 120.0
            os.utime(valid_old, (old_ts, old_ts))
            os.utime(broken_old, (old_ts, old_ts))

            cfg_mgr = ConfigManager(tmp_path / "config.json")
            cfg = cfg_mgr.load()
            cfg.recording.recordings_dir = str(rec_root)
            cfg.drive.enabled = True
            cfg.drive.auto_upload = True
            cfg.drive.delete_after_upload = True
            cfg.drive.sync_mode = "folder"
            drive_dir = tmp_path / "MeuDrive"
            drive_dir.mkdir()
            cfg.drive.local_drive_dir = str(drive_dir)
            cfg_mgr.save(cfg)

            root = tk.Tk()
            root.withdraw()
            with mock.patch("app_gui.DEFAULT_PENDING_PATH", tmp_path / "pending.json"):
                app = HomeCamApp(
                    root,
                    config_manager=cfg_mgr,
                    allow_auto_connect=False,
                    enable_startup_tasks=False,
                )
            try:
                app._recover_orphaned_recordings()
                expected_dest = (
                    drive_dir / "HomeCamSeguranca" / "Garagem" / "2026-11-12" / "Garagem_2026-11-12_22-00-00.mp4"
                )
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and not (expected_dest.exists() and not valid_old.exists()):
                    time.sleep(0.05)
                self.assertTrue(expected_dest.exists(), "Arquivo recuperado deveria ter sido enviado ao Drive.")
                self.assertFalse(valid_old.exists(), "Arquivo local deveria ter sido apagado após upload.")
                self.assertEqual(app.uploader.get_pending_count(), 0)
                self.assertFalse(broken_old.exists())
                self.assertTrue((rec_root / "_corrompidos" / broken_old.name).exists())
            finally:
                app.on_close(wait_for_uploads=False)


class AIAnalyticsTests(unittest.TestCase):
    def test_normalize_brazilian_plate(self) -> None:
        from ai_analytics import normalize_brazilian_plate

        self.assertEqual(normalize_brazilian_plate("abc1d23"), "ABC1D23")
        self.assertEqual(normalize_brazilian_plate("abc-1234"), "ABC-1234")
        self.assertEqual(normalize_brazilian_plate("HSX49S6"), "HSX-4956")
        self.assertEqual(normalize_brazilian_plate("4BC1D23"), "ABC1D23")
        self.assertEqual(normalize_brazilian_plate("BR ABC1D23"), "ABC1D23")
        self.assertIsNone(normalize_brazilian_plate("12345"))
        self.assertIsNone(normalize_brazilian_plate(""))

    def test_estimate_vehicle_color(self) -> None:
        from ai_analytics import estimate_vehicle_color

        # Vermelho com faixa superior escura (simulando para-brisa)
        red_crop = np.zeros((100, 120, 3), dtype=np.uint8)
        red_crop[35:, :] = (20, 20, 220)  # BGR vermelho
        self.assertEqual(estimate_vehicle_color(red_crop), "Vermelho")

        blue_crop = np.zeros((100, 120, 3), dtype=np.uint8)
        blue_crop[:, :] = (210, 60, 20)  # BGR azul
        self.assertEqual(estimate_vehicle_color(blue_crop), "Azul")

        white_crop = np.full((100, 120, 3), 232, dtype=np.uint8)
        white_crop[40:75, 25:95] = (228, 228, 232)
        self.assertEqual(estimate_vehicle_color(white_crop), "Branco")

        # Veículo preto sob sombra fria/azulada de câmera CCTV não pode virar azul
        black_crop = np.full((100, 120, 3), (38, 30, 26), dtype=np.uint8)
        self.assertEqual(estimate_vehicle_color(black_crop), "Preto")

        silver_crop = np.full((100, 120, 3), (172, 168, 165), dtype=np.uint8)
        self.assertEqual(estimate_vehicle_color(silver_crop), "Prata")

        moto_crop = np.zeros((120, 90, 3), dtype=np.uint8)
        moto_crop[38:84, 20:70] = (30, 175, 35)  # Tanque/carenagem verde
        self.assertEqual(estimate_vehicle_color(moto_crop, vehicle_type="Moto"), "Verde")

        ir_crop = np.full((100, 120, 3), 120, dtype=np.uint8)
        self.assertEqual(estimate_vehicle_color(ir_crop), "P&B (Noturno IR)")

    def test_estimate_clothing_colors(self) -> None:
        from ai_analytics import estimate_clothing_colors

        # Corpo inteiro: pele na cabeça/pescoço + camisa azul + calça preta
        person = np.full((200, 90, 3), 130, dtype=np.uint8)
        person[0:38, 20:70] = (140, 170, 210)  # Pele/pescoço (BGR)
        person[38:104, 15:75] = (205, 75, 20)  # Camisa azul
        person[104:180, 18:72] = (22, 22, 26)  # Calça preta
        upper, lower, summary = estimate_clothing_colors(person)
        self.assertEqual(upper, "Azul")
        self.assertEqual(lower, "Preto")
        self.assertEqual(summary, "Sup: Azul | Inf: Preto")

        # Enquadramento curto (busto): apenas camisa vermelha visível
        bust = np.zeros((95, 85, 3), dtype=np.uint8)
        bust[18:55, 16:69] = (22, 22, 215)  # Vermelho
        upper_b, lower_b, summary_b = estimate_clothing_colors(bust)
        self.assertEqual(upper_b, "Vermelho")
        self.assertEqual(lower_b, "Não visível")
        self.assertIn("Sup: Vermelho", summary_b)

    def test_class_aware_iou_tracker(self) -> None:
        from ai_analytics import ClassAwareIoUTracker

        tracker = ClassAwareIoUTracker(iou_threshold=0.3, max_age_sec=1.0)
        res1 = tracker.update([("Carro", (10, 10, 110, 110), 0.9), ("Pessoa", (12, 12, 108, 108), 0.85)], now=10.0)
        id_car_1, id_person_1 = res1[0][0], res1[1][0]
        self.assertNotEqual(id_car_1, id_person_1)

        res2 = tracker.update([("Carro", (15, 14, 115, 114), 0.92), ("Pessoa", (14, 14, 109, 109), 0.88)], now=10.4)
        self.assertEqual(res2[0][0], id_car_1)
        self.assertEqual(res2[1][0], id_person_1)

        # Movimento rápido constante mantendo predição linear do mesmo veículo
        res2b = tracker.update([("Carro", (45, 42, 145, 142), 0.90)], now=10.8)
        self.assertEqual(res2b[0][0], id_car_1)

        res3 = tracker.update([("Carro", (45, 42, 145, 142), 0.9)], now=12.0)
        self.assertNotEqual(res3[0][0], id_car_1)

    def test_ai_pipeline_end_to_end_and_csv(self) -> None:
        from ai_analytics import CCTVAnalyticsPipeline

        class _FakeDetector:
            def detect(self, _frame, **_kw):
                return [
                    ("Pessoa", (10, 10, 90, 210), 0.91),
                    ("Animal: Cachorro", (100, 140, 170, 210), 0.88),
                    ("Carro", (200, 80, 420, 240), 0.95),
                    ("Moto", (450, 120, 560, 240), 0.89),
                    ("Bicicleta", (570, 130, 620, 230), 0.86),
                ]

        class _FakeGender:
            def predict(self, _crop):
                return ("Homem", 0.93)

        class _FakeALPR:
            def __init__(self) -> None:
                self.calls = 0

            def read_plate(self, crop):
                self.calls += 1
                if crop.shape[1] > 180:
                    return ("ABC1D23", 0.92)
                return ("XYZ-9876", 0.87)

        class _FakeVehicleClf:
            def __init__(self) -> None:
                self.car_calls = 0

            def predict_vehicle_model(self, _crop, vtype):
                if vtype == "Moto":
                    return ("Honda CG / Biz (Moto Street)", 0.72)
                self.car_calls += 1
                if self.car_calls == 1:
                    return ("Hatch Subcompacto — Kwid / Mobi / Uno", 0.35)
                return ("Sedan Médio — Corolla / Civic / Virtus", 0.85)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "veiculos_test.csv"
            people_csv_path = tmp_path / "pessoas_test.csv"
            pipe = CCTVAnalyticsPipeline(
                detector=_FakeDetector(),
                alpr=_FakeALPR(),
                gender_clf=_FakeGender(),
                vehicle_clf=_FakeVehicleClf(),
                csv_path=csv_path,
                people_csv_path=people_csv_path,
                save_csv=True,
                secondary_cooldown_sec=0.0,
            )
            frame = np.zeros((320, 640, 3), dtype=np.uint8)
            frame[45:110, 20:80] = (205, 80, 25)    # Camisa azul
            frame[115:185, 25:75] = (24, 24, 24)    # Calça preta
            frame[80:240, 200:420] = (15, 15, 210)  # Carro vermelho
            frame[120:240, 450:560] = (200, 50, 15) # Moto azul

            items, dirty = pipe.process_frame(frame, camera_name="Portão")
            self.assertTrue(dirty)
            self.assertEqual(len(items), 5)

            summary = pipe.get_summary()
            self.assertEqual(summary.people_total, 1)
            self.assertEqual(summary.men_count, 1)
            self.assertEqual(summary.animals_count, 1)
            self.assertEqual(summary.cars_count, 1)
            self.assertEqual(summary.motos_count, 1)
            self.assertEqual(summary.bicycles_count, 1)

            # Quadros seguintes consolidam votação temporal do modelo veicular
            pipe.process_frame(frame, camera_name="Portão")
            pipe.process_frame(frame, camera_name="Portão")

            records = pipe.get_vehicle_records()
            self.assertEqual(len(records), 2)
            by_type = {r.vehicle_type: r for r in records}
            self.assertEqual(by_type["Carro"].plate, "ABC1D23")
            self.assertEqual(by_type["Carro"].color, "Vermelho")
            self.assertEqual(by_type["Carro"].approx_model, "Sedan Médio — Corolla / Civic / Virtus")
            self.assertEqual(by_type["Moto"].plate, "XYZ-9876")
            self.assertEqual(by_type["Moto"].color, "Azul")

            person_records = pipe.get_person_records()
            self.assertEqual(len(person_records), 1)
            self.assertEqual(person_records[0].person_type, "Homem")
            self.assertEqual(person_records[0].upper_color, "Azul")
            self.assertEqual(person_records[0].lower_color, "Preto")
            self.assertEqual(person_records[0].clothing_colors, "Sup: Azul | Inf: Preto")

            pipe._write_all_csv_snapshots()
            self.assertTrue(csv_path.exists())
            self.assertTrue(people_csv_path.exists())
            veh_content = csv_path.read_text(encoding="utf-8-sig")
            ppl_content = people_csv_path.read_text(encoding="utf-8-sig")
            self.assertIn("Timestamp;Tipo de Veículo;Placa;Cor;Modelo Aproximado;Câmera", veh_content)
            self.assertIn("ABC1D23", veh_content)
            self.assertIn("Timestamp;Tipo (Homem/Mulher);Cor das Roupas;Cor Superior;Cor Inferior;Câmera", ppl_content)
            self.assertIn("Homem;Sup: Azul | Inf: Preto", ppl_content)

            # Verifica recarregamento de histórico CSV usando delimitador `;`
            reloaded = CCTVAnalyticsPipeline(csv_path=csv_path, people_csv_path=people_csv_path, save_csv=False)
            self.assertEqual(len(reloaded.get_vehicle_records()), 2)
            self.assertEqual(len(reloaded.get_person_records()), 1)

    def test_ai_config_persistence_and_gui_tab(self) -> None:
        import tkinter as tk
        from ai_analytics import AISummary, PersonRecord, VehicleRecord
        from app_gui import HomeCamApp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_mgr = ConfigManager(tmp_path / "config.json")
            cfg = AppConfig()
            cfg.ai.enabled = True
            cfg.ai.detect_animals = False
            cfg_mgr.save(cfg)

            reloaded = cfg_mgr.load()
            self.assertTrue(reloaded.ai.enabled)
            self.assertFalse(reloaded.ai.detect_animals)

            reloaded.ai.enabled = False
            cfg_mgr.save(reloaded)

            root = tk.Tk()
            root.withdraw()
            with mock.patch("app_gui.DEFAULT_PENDING_PATH", tmp_path / "pending.json"):
                app = HomeCamApp(
                    root,
                    config_manager=cfg_mgr,
                    allow_auto_connect=False,
                    enable_startup_tasks=False,
                )
            try:
                self.assertEqual(
                    tuple(app.ai_vehicle_tree["columns"]),
                    ("timestamp", "tipo", "placa", "cor", "modelo"),
                )
                self.assertEqual(
                    tuple(app.ai_people_tree["columns"]),
                    ("timestamp", "tipo", "roupas"),
                )
                sample_summary = AISummary(
                    people_total=2,
                    men_count=1,
                    women_count=1,
                    animals_count=1,
                    cars_count=1,
                    motos_count=0,
                    bicycles_count=1,
                    last_plate="BRA2E19",
                    backend_name="Teste",
                )
                sample_row = VehicleRecord(
                    track_id=7,
                    timestamp="2026-11-13 09:00:00",
                    vehicle_type="Carro",
                    plate="BRA2E19",
                    color="Branco",
                    approx_model="Toyota Corolla / Sedan Médio",
                )
                sample_person = PersonRecord(
                    track_id=3,
                    timestamp="2026-11-13 09:00:02",
                    person_type="Mulher",
                    clothing_colors="Sup: Branco | Inf: Azul",
                    upper_color="Branco",
                    lower_color="Azul",
                )
                app._refresh_ai_table([sample_row], [sample_person])
                app._update_ai_banner(sample_summary)
                children = app.ai_vehicle_tree.get_children()
                self.assertEqual(len(children), 1)
                self.assertEqual(
                    tuple(app.ai_vehicle_tree.item(children[0], "values")),
                    ("2026-11-13 09:00:00", "Carro", "BRA2E19", "Branco", "Toyota Corolla / Sedan Médio"),
                )
                p_children = app.ai_people_tree.get_children()
                self.assertEqual(len(p_children), 1)
                self.assertEqual(
                    tuple(app.ai_people_tree.item(p_children[0], "values")),
                    ("2026-11-13 09:00:02", "Mulher", "Sup: Branco | Inf: Azul"),
                )
            finally:
                app.on_close(wait_for_uploads=False)

    def test_onnx_models_smoke_inference(self) -> None:
        from ai_analytics import CCTVAnalyticsPipeline, DEFAULT_MODELS_DIR

        if not (DEFAULT_MODELS_DIR / "yolov10s" / "onnx" / "model.onnx").exists():
            self.skipTest("Modelos ONNX não baixados localmente.")

        pipe = CCTVAnalyticsPipeline(save_csv=False)
        ok = pipe.load_default_models(DEFAULT_MODELS_DIR)
        self.assertTrue(ok)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        items, dirty = pipe.process_frame(blank, camera_name="SmokeTest")
        self.assertEqual(items, [])
        self.assertFalse(dirty)


class DVRPlaybackAndTimelineTests(unittest.TestCase):
    @staticmethod
    def _write_numbered_mp4(path: Path, frame_count: int = 20, fps: float = 10.0, base_hue: int = 0) -> None:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
        assert writer.isOpened()
        for idx in range(frame_count):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :] = ((base_hue + idx * 10) % 250, 80, 180)
            cv2.rectangle(frame, (10 + idx * 4, 20), (30 + idx * 4, 50), (255, 255, 255), -1)
            # Marca o índice do quadro em pixel discreto B=[idx] para validar step frame exato
            frame[0:8, 0:8, 0] = idx % 255
            writer.write(frame)
        writer.release()

    def test_gdrive_remote_listing_and_download(self) -> None:
        from gdrive_uploader import GoogleDriveUploader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drive_dir = root / "MeuDrive"
            rec_day = drive_dir / "HomeCamSeguranca" / "Portão" / "2026-11-13"
            rec_day.mkdir(parents=True, exist_ok=True)
            (rec_day / "Portão_2026-11-13_08-00-00.mp4").write_bytes(b"mp4bytes")

            folder_up = GoogleDriveUploader(
                pending_path=root / "pending_f.json",
                sync_mode="folder",
                local_drive_dir=str(drive_dir),
            )
            self.assertEqual(folder_up.list_remote_dates("Portão"), ["2026-11-13"])
            folder_files = folder_up.list_remote_recordings("Portão", "2026-11-13")
            self.assertEqual(len(folder_files), 1)
            self.assertEqual(folder_files[0]["name"], "Portão_2026-11-13_08-00-00.mp4")

            class _FakeExec:
                def __init__(self, data: object) -> None:
                    self._data = data

                def execute(self) -> object:
                    return self._data

            class _FakeFilesResource:
                def list(self, **kwargs):
                    q = kwargs.get("q", "")
                    if "name='HomeCamSeguranca'" in q:
                        return _FakeExec({"files": [{"id": "root_1", "name": "HomeCamSeguranca"}]})
                    if "name='Portão'" in q:
                        return _FakeExec({"files": [{"id": "cam_1", "name": "Portão"}]})
                    if "'cam_1' in parents" in q and "folder" in q:
                        return _FakeExec({"files": [{"id": "date_1", "name": "2026-11-13"}]})
                    if "name='2026-11-13'" in q:
                        return _FakeExec({"files": [{"id": "date_1", "name": "2026-11-13"}]})
                    if "'date_1' in parents" in q:
                        return _FakeExec(
                            {
                                "files": [
                                    {"id": "vid_99", "name": "Portão_2026-11-13_10-15-00.mp4", "size": "123456"},
                                ]
                            }
                        )
                    return _FakeExec({"files": []})

                def get_media(self, fileId: str):
                    assert fileId == "vid_99"
                    return {"id": fileId}

            class _FakeService:
                def files(self):
                    return _FakeFilesResource()

            class _FakeMediaDl:
                def __init__(self, fh, _req, chunksize=None) -> None:
                    self._fh = fh

                def next_chunk(self):
                    self._fh.write(b"CLOUD_MP4_PAYLOAD")
                    return (None, True)

            creds = mock.MagicMock(valid=True, expired=False)
            tok = root / "token.json"
            tok.write_text("{}", encoding="utf-8")
            oauth_up = GoogleDriveUploader(
                pending_path=root / "pending_o.json",
                token_path=tok,
                sync_mode="oauth",
                service_factory=lambda _c: _FakeService(),
            )
            with mock.patch.object(oauth_up, "load_saved_credentials", return_value=creds):
                self.assertEqual(oauth_up.list_remote_dates("Portão"), ["2026-11-13"])
                recs = oauth_up.list_remote_recordings("Portão", "2026-11-13")
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0]["id"], "vid_99")
                with mock.patch("gdrive_uploader.MediaIoBaseDownload", _FakeMediaDl):
                    dl_dest = root / "cache" / "video.mp4"
                    out = oauth_up.download_remote_file("vid_99", dl_dest)
                    self.assertEqual(out.read_bytes(), b"CLOUD_MP4_PAYLOAD")

    def test_timeline_index_and_event_store(self) -> None:
        from datetime import datetime
        from dvr_playback import DVREventStore, DVRTimelineIndex

        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "gravacoes"
            day_dir = rec_root / "Garagem" / "2026-11-13"
            seg1_path = day_dir / "Garagem_2026-11-13_09-00-00.mp4"
            seg2_path = day_dir / "Garagem_mov_2026-11-13_09-05-00.mp4"
            self._write_numbered_mp4(seg1_path, frame_count=20, fps=10.0)
            self._write_numbered_mp4(seg2_path, frame_count=20, fps=10.0)

            # Arquivos em _playback_cache e _corrompidos não devem poluir a data principal
            corrupt_dir = rec_root / "_corrompidos"
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            (corrupt_dir / "Garagem_2026-11-13_01-00-00.mp4").write_bytes(b"bad")

            store = DVREventStore(rec_root / "_timeline_events")
            dt1 = datetime(2026, 11, 13, 9, 0, 1)
            self.assertTrue(store.record_event("Garagem", "motion", "Movimento portão", when=dt1, min_interval_sec=5.0))
            self.assertFalse(store.record_event("Garagem", "motion", "Ruído próximo", when=dt1, min_interval_sec=5.0))
            self.assertTrue(
                store.record_event("Garagem", "ai_plate", "Placa ABC1D23", when=dt1, dedupe_key="ABC1D23")
            )
            self.assertTrue(
                store.record_event("Garagem", "ai_plate", "Placa XYZ9K88", when=dt1, dedupe_key="XYZ9K88")
            )

            csv_path = rec_root / "veiculos_detectados.csv"
            csv_path.write_text(
                "Timestamp;Tipo de Veículo;Placa;Cor;Modelo Aproximado;Câmera\n"
                "2026-11-13 09:05:01;Carro;BRA2E19;Prata;Sedan Médio;Garagem\n",
                encoding="utf-8-sig",
            )

            dates = DVRTimelineIndex.discover_dates(rec_root, "Garagem")
            self.assertEqual(dates, ["2026-11-13"])

            segs, evs = DVRTimelineIndex.build_day_timeline(
                rec_root,
                "Garagem",
                "2026-11-13",
                event_store=store,
                csv_path=csv_path,
            )
            self.assertEqual(len(segs), 2)
            self.assertAlmostEqual(segs[0].duration_sec, 2.0, places=1)
            self.assertTrue(segs[1].is_motion_clip)
            ev_types = [e.event_type for e in evs]
            self.assertIn("motion", ev_types)
            self.assertEqual(ev_types.count("ai_plate"), 3)  # 2 do JSONL + 1 do CSV

    def test_digital_zoom_viewport_and_pan(self) -> None:
        from dvr_playback import DigitalZoomViewport

        vp = DigitalZoomViewport()
        frame = np.full((360, 640, 3), 60, dtype=np.uint8)
        roi0 = vp.get_source_roi(640, 360)
        self.assertEqual(roi0, (0, 0, 640, 360))

        vp.set_zoom_at_point(4.0, 320, 180, 640, 360, 640, 360)
        x1, y1, x2, y2 = vp.get_source_roi(640, 360)
        self.assertEqual(x2 - x1, 160)
        self.assertEqual(y2 - y1, 90)
        self.assertAlmostEqual((x1 + x2) / 2.0, 320.0, delta=2.0)

        # Arrasto extremo nunca deve ultrapassar as bordas do quadro
        vp.pan_by_pixels(5000, 5000, 640, 360, 640, 360)
        px1, py1, px2, py2 = vp.get_source_roi(640, 360)
        self.assertGreaterEqual(px1, 0)
        self.assertGreaterEqual(py1, 0)
        self.assertLessEqual(px2, 640)
        self.assertLessEqual(py2, 360)

        rendered = vp.render(frame, 800, 450, draw_minimap=True)
        self.assertEqual(rendered.shape[:2], (450, 800))

    def test_dvr_playback_engine_playback_rewind_step_and_scrub(self) -> None:
        from dvr_playback import DVRPlaybackEngine, DVRTimelineIndex

        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp)
            day_dir = rec_root / "Portão" / "2026-11-13"
            seg1 = day_dir / "Portão_2026-11-13_10-00-00.mp4"
            seg2 = day_dir / "Portão_2026-11-13_10-00-02.mp4"
            self._write_numbered_mp4(seg1, frame_count=20, fps=10.0, base_hue=10)
            self._write_numbered_mp4(seg2, frame_count=20, fps=10.0, base_hue=120)

            segs, evs = DVRTimelineIndex.build_day_timeline(rec_root, "Portão", "2026-11-13")
            self.assertEqual(len(segs), 2)

            received: list[object] = []
            ev_signal = threading.Event()

            def on_frame(st) -> None:
                received.append(st)
                ev_signal.set()

            player = DVRPlaybackEngine(recordings_root=rec_root, on_frame=on_frame)
            try:
                ev_signal.clear()
                player.load_timeline(segs, evs, auto_seek_sec=segs[0].start_sec)
                self.assertTrue(ev_signal.wait(timeout=3.0))
                self.assertEqual(received[-1].filename, seg1.name)
                self.assertEqual(received[-1].frame_index, 0)

                # Avança +1 quadro exato duas vezes
                for expected_idx in (1, 2):
                    ev_signal.clear()
                    player.step_frame(1)
                    self.assertTrue(ev_signal.wait(timeout=2.0))
                    self.assertEqual(received[-1].frame_index, expected_idx)

                # Retrocede -1 quadro exato via cache GOP
                ev_signal.clear()
                player.step_frame(-1)
                self.assertTrue(ev_signal.wait(timeout=2.0))
                self.assertEqual(received[-1].frame_index, 1)

                # Scrub direto para o segundo segmento e passo -1 quadro cruzando fronteira para o segmento 1!
                ev_signal.clear()
                player.seek_to_day_second(segs[1].start_sec)
                self.assertTrue(ev_signal.wait(timeout=2.0))
                self.assertEqual(received[-1].filename, seg2.name)
                self.assertEqual(received[-1].frame_index, 0)

                ev_signal.clear()
                player.step_frame(-1)
                self.assertTrue(ev_signal.wait(timeout=2.0))
                self.assertEqual(received[-1].filename, seg1.name)
                self.assertEqual(received[-1].frame_index, 19)

                # Reprodução acelerada (+4x) atravessa segmentos automaticamente
                player.set_speed(4.0)
                player.play()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and not any(st.filename == seg2.name for st in received):
                    time.sleep(0.05)
                player.pause()
                self.assertTrue(any(st.filename == seg2.name for st in received))

                # Retrocesso rápido (-4x) retrocede índices sem travar
                start_rev_idx = received[-1].frame_index
                start_rev_count = len(received)
                player.set_speed(-4.0)
                player.play()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and len(received) < start_rev_count + 4:
                    time.sleep(0.05)
                player.pause()
                self.assertGreaterEqual(len(received), start_rev_count + 2)
                self.assertLess(
                    received[-1].day_sec,
                    received[start_rev_count - 1].day_sec + 0.01 if start_rev_idx >= 0 else 86400.0,
                )
            finally:
                player.stop()

    def test_homecam_app_dvr_playback_tab(self) -> None:
        import tkinter as tk
        from app_gui import HomeCamApp
        from config_manager import AppConfig, CameraConfig, ConfigManager

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rec_root = tmp_path / "gravacoes"
            mp4 = rec_root / "Entrada" / "2026-11-13" / "Entrada_2026-11-13_08-30-00.mp4"
            self._write_numbered_mp4(mp4, frame_count=15, fps=10.0)

            cfg_mgr = ConfigManager(tmp_path / "config.json")
            cfg = AppConfig(cameras=[CameraConfig(name="Entrada")])
            cfg.recording.recordings_dir = str(rec_root)
            cfg_mgr.save(cfg)

            root = tk.Tk()
            root.withdraw()
            with mock.patch("app_gui.DEFAULT_PENDING_PATH", tmp_path / "pending.json"):
                app = HomeCamApp(
                    root,
                    config_manager=cfg_mgr,
                    allow_auto_connect=False,
                    enable_startup_tasks=False,
                )
            try:
                app._open_playback_for_active_camera()
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and (not app._pb_segments or app._pb_latest_frame is None):
                    root.update()
                    time.sleep(0.05)
                self.assertEqual(len(app._pb_segments), 1)
                self.assertIsNotNone(app._pb_latest_frame)
                self.assertGreaterEqual(len(app.pb_events_tree.get_children()), 1)

                # Zoom interativo na aba de playback sem exceções
                app._pb_adjust_zoom(2.0)
                self.assertGreater(app._pb_viewport.zoom, 1.5)
                app._pb_reset_zoom()
                self.assertEqual(app._pb_viewport.zoom, 1.0)
                self.assertTrue(len(app.pb_timeline_canvas.find_withtag("playhead")) >= 1)
            finally:
                app.on_close(wait_for_uploads=False)


class HeadlessAndWebPortalTests(unittest.TestCase):
    def test_config_env_overrides_and_web_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_mgr = ConfigManager(root / "config.json")
            cfg = AppConfig()
            cfg.web.enabled = True
            cfg.web.port = 9123
            cfg.web.auth_token = "local-secret"
            cfg_mgr.save(cfg)

            override_dir = root / "nvr_data"
            with mock.patch.dict(
                os.environ,
                {
                    "HOMECAM_RECORDINGS_DIR": str(override_dir),
                    "HOMECAM_CAM_PASSWORD": "senha-env-24h",
                    "HOMECAM_RTSP_URL": "rtsp://admin:senha-env-24h@10.8.0.5:554/stream1",
                    "HOMECAM_WEB_PORT": "9876",
                    "HOMECAM_WEB_TOKEN": "token-cloud",
                },
                clear=False,
            ):
                loaded = cfg_mgr.load()

            self.assertTrue(loaded.web.enabled)
            self.assertEqual(loaded.web.port, 9876)
            self.assertEqual(loaded.web.auth_token, "token-cloud")
            self.assertEqual(Path(loaded.recording.recordings_dir), override_dir.resolve())
            self.assertEqual(loaded.ai.csv_log_path, str(override_dir.resolve() / "veiculos_detectados.csv"))
            self.assertEqual(loaded.ai.people_csv_log_path, str(override_dir.resolve() / "pessoas_detectadas.csv"))
            cam = cfg_mgr.get_selected_camera(loaded)
            self.assertEqual(cam.password, "senha-env-24h")
            self.assertEqual(cam.source_url_override, "rtsp://admin:senha-env-24h@10.8.0.5:554/stream1")

    def test_windows_power_guard_acquire_fallback_and_release(self) -> None:
        from headless_nvr import (
            ES_AWAYMODE_REQUIRED,
            ES_CONTINUOUS,
            ES_SYSTEM_REQUIRED,
            WindowsPowerGuard,
        )

        calls: list[int] = []

        def fake_set_state(flags: int) -> int:
            calls.append(flags)
            if flags & ES_AWAYMODE_REQUIRED:
                return 0
            return 1

        guard = WindowsPowerGuard(keepalive_sec=60.0, set_state_fn=fake_set_state)
        self.assertTrue(guard.is_supported)
        self.assertTrue(guard.acquire())
        self.assertTrue(guard.is_active)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        self.assertEqual(calls[1], ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

        guard.release()
        self.assertFalse(guard.is_active)
        self.assertEqual(calls[-1], ES_CONTINUOUS)

    def test_main_headless_smoke_check(self) -> None:
        import main as entry_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_mgr = ConfigManager(root / "config.json")
            cfg = AppConfig()
            cfg.recording.recordings_dir = str(root / "gravacoes")
            cfg_mgr.save(cfg)

            rc = entry_main.main(["--headless", "--check", "--config", str(root / "config.json")])
            self.assertEqual(rc, 0)

    def test_headless_nvr_and_web_portal_end_to_end(self) -> None:
        import urllib.error
        import urllib.request
        from ai_analytics import CCTVAnalyticsPipeline
        from headless_nvr import HeadlessNVRService, WindowsPowerGuard
        from web_portal import NVRWebServer

        class _FakeDet:
            backend_name = "Test-24h"
            provider = "CPU-Test"

            def detect(self, _frame: np.ndarray, **_kwargs: object) -> list[tuple[str, tuple[int, int, int, int], float]]:
                return [
                    ("Pessoa", (15, 25, 115, 220), 0.94),
                    ("Carro", (180, 70, 440, 240), 0.95),
                ]

        class _FakeALPR:
            def read_plate(self, _crop: np.ndarray, camera_name: str = "Cam") -> tuple[str, float]:
                _ = camera_name
                return ("BRA2E19", 0.92)

        class _FakeGender:
            def predict(self, _crop: np.ndarray, *, frame_ir: bool = False) -> tuple[str, float]:
                _ = frame_ir
                return ("Mulher", 0.90)

        class _FakeVehicleClf:
            def predict_vehicle_model(self, _crop: np.ndarray, vehicle_type: str) -> tuple[str, float]:
                _ = vehicle_type
                return ("SUV / Crossover — Compass / Creta / Tracker", 0.87)

        class _FakePTZ:
            backend_label = "FakePTZ"

            def __init__(self) -> None:
                self.moves: list[tuple[str, int]] = []
                self.stops: list[str] = []
                self.home_calls = 0

            def move_start(self, direction: str, speed: int = 5) -> bool:
                self.moves.append((direction, speed))
                return True

            def move_stop(self, direction: str = "") -> bool:
                self.stops.append(direction)
                return True

            def goto_preset(self, _preset_id: int) -> bool:
                self.home_calls += 1
                return True

            def close(self) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_root = root / "gravacoes"
            mp4_path = rec_root / "Garagem" / "2026-11-13" / "Garagem_2026-11-13_14-00-00.mp4"
            DVRPlaybackAndTimelineTests._write_numbered_mp4(mp4_path, frame_count=12, fps=10.0)

            cfg_mgr = ConfigManager(root / "config.json")
            cfg = AppConfig(cameras=[CameraConfig(name="Garagem", ptz_speed=6)])
            cfg.recording.recordings_dir = str(rec_root)
            cfg.ai.enabled = True
            cfg.ai.csv_log_path = str(rec_root / "veiculos_detectados.csv")
            cfg.ai.people_csv_log_path = str(rec_root / "pessoas_detectadas.csv")
            cfg_mgr.save(cfg)

            fake_pipe = CCTVAnalyticsPipeline(
                detector=_FakeDet(),
                alpr=_FakeALPR(),
                gender_clf=_FakeGender(),
                vehicle_clf=_FakeVehicleClf(),
                csv_path=cfg.ai.csv_log_path,
                people_csv_path=cfg.ai.people_csv_log_path,
                save_csv=True,
                secondary_cooldown_sec=0.0,
            )

            guard = WindowsPowerGuard(set_state_fn=lambda _flags: 1)
            nvr = HeadlessNVRService(
                config_manager=cfg_mgr,
                recordings_root=rec_root,
                pending_path=root / "pending.json",
                auto_connect=False,
                auto_record=False,
                enable_ai=True,
                load_ai_async=False,
                power_guard=guard,
                ai_pipeline=fake_pipe,
            )
            fake_ptz = _FakePTZ()
            nvr._ptz_client = fake_ptz
            nvr._ptz_backend_label = fake_ptz.backend_label

            server = NVRWebServer(nvr, host="127.0.0.1", port=0, auth_token="segredo-24h")
            nvr.start()
            port = server.start()
            base = f"http://127.0.0.1:{port}"
            try:
                # Health check sem token funciona para Docker HEALTHCHECK
                with urllib.request.urlopen(f"{base}/api/health", timeout=3.0) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertTrue(json.loads(resp.read().decode("utf-8"))["ok"])

                # Endpoints protegidos exigem token
                with self.assertRaises(urllib.error.HTTPError) as unauth_ctx:
                    urllib.request.urlopen(f"{base}/api/status", timeout=3.0)
                self.assertEqual(unauth_ctx.exception.code, 401)
                unauth_ctx.exception.close()

                # HTML Portal renderizado
                with urllib.request.urlopen(f"{base}/", timeout=3.0) as resp:
                    html = resp.read().decode("utf-8")
                    self.assertIn("HomeCam Segurança 24/7", html)

                # Processa frames ao vivo diretamente (substituindo câmera física)
                frame = np.zeros((280, 480, 3), dtype=np.uint8)
                frame[40:120, 30:100] = (200, 70, 20)
                frame[125:205, 35:95] = (30, 30, 30)
                frame[80:230, 190:430] = (20, 20, 210)
                nvr.process_live_frame(frame, camera_name="Garagem")

                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline and not nvr.ai_pipeline.get_summary().last_plate:
                    time.sleep(0.05)

                def _get_json(path: str) -> dict:
                    req = urllib.request.Request(
                        f"{base}{path}",
                        headers={"Authorization": "Bearer segredo-24h"},
                    )
                    with urllib.request.urlopen(req, timeout=3.0) as r:
                        return json.loads(r.read().decode("utf-8"))

                status_data = _get_json("/api/status")
                self.assertTrue(status_data["service_running"])
                self.assertTrue(status_data["power_guard_active"])
                self.assertEqual(status_data["camera"]["name"], "Garagem")
                self.assertEqual(status_data["ai"]["summary"]["last_plate"], "BRA2E19")

                veh_data = _get_json("/api/vehicles")
                self.assertEqual(len(veh_data["records"]), 1)
                self.assertEqual(veh_data["records"][0]["plate"], "BRA2E19")
                self.assertEqual(veh_data["records"][0]["color"], "Vermelho")

                ppl_data = _get_json("/api/people")
                self.assertEqual(len(ppl_data["records"]), 1)
                self.assertEqual(ppl_data["records"][0]["person_type"], "Mulher")

                dates_data = _get_json("/api/timeline/dates")
                self.assertIn("2026-11-13", dates_data["dates"])

                day_data = _get_json("/api/timeline/day?date=2026-11-13")
                self.assertEqual(len(day_data["segments"]), 1)
                self.assertEqual(day_data["segments"][0]["filename"], mp4_path.name)

                # Streaming MP4 com HTTP Range (206 Partial Content) para seeking no navegador
                vid_req = urllib.request.Request(
                    f"{base}/api/video?date=2026-11-13&file={mp4_path.name}&token=segredo-24h",
                    headers={"Range": "bytes=0-63"},
                )
                with urllib.request.urlopen(vid_req, timeout=3.0) as vresp:
                    self.assertEqual(vresp.status, 206)
                    self.assertTrue(vresp.headers.get("Content-Range", "").startswith("bytes 0-63/"))
                    chunk = vresp.read()
                    self.assertEqual(chunk, mp4_path.read_bytes()[:64])

                # Snapshot JPEG ao vivo e CSV remoto
                with urllib.request.urlopen(f"{base}/api/snapshot.jpg?token=segredo-24h", timeout=3.0) as sresp:
                    jpg = sresp.read()
                    self.assertTrue(jpg.startswith(b"\xff\xd8"))

                with urllib.request.urlopen(f"{base}/api/export/vehicles.csv?token=segredo-24h", timeout=3.0) as cresp:
                    csv_txt = cresp.read().decode("utf-8-sig")
                    self.assertIn("BRA2E19", csv_txt)

                # PTZ remoto via POST
                ptz_req = urllib.request.Request(
                    f"{base}/api/ptz/move",
                    method="POST",
                    data=json.dumps({"direction": "left", "duration": 0.1}).encode("utf-8"),
                    headers={
                        "Authorization": "Bearer segredo-24h",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(ptz_req, timeout=3.0) as presp:
                    self.assertTrue(json.loads(presp.read().decode("utf-8"))["ok"])
                self.assertEqual(fake_ptz.moves[0][0], "left")
                self.assertEqual(fake_ptz.stops[0], "left")
            finally:
                server.stop()
                nvr.stop()


if __name__ == "__main__":
    unittest.main()

