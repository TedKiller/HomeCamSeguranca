"""Servidor HTTP/REST embutido e Portal Web responsivo para acesso 24/7 (PC/Celular)."""

from __future__ import annotations

import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit

from headless_nvr import HeadlessNVRService

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_PORTAL_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>HomeCam Segurança 24/7 — Portal Remoto &amp; DVR</title>
<style>
  :root {
    --bg: #0b1120;
    --panel: #111827;
    --panel-alt: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --blue: #2563eb;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --cyan: #0891b2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    line-height: 1.4;
  }
  header {
    background: #161e2e;
    border-bottom: 1px solid var(--border);
    padding: 12px 18px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: space-between;
  }
  .brand { font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .badge {
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: var(--panel-alt);
    border: 1px solid var(--border);
  }
  .badge.ok { background: #14532d; border-color: #22c55e; color: #dcfce7; }
  .badge.rec { background: #7f1d1d; border-color: #ef4444; color: #fee2e2; }
  .badge.warn { background: #78350f; border-color: #f59e0b; color: #fef3c7; }
  main { padding: 16px; max-width: 1480px; margin: 0 auto; display: grid; gap: 16px; }
  .grid-2 { display: grid; grid-template-columns: 1.55fr 1fr; gap: 16px; }
  .grid-tables { display: grid; grid-template-columns: 1.25fr 1fr; gap: 16px; }
  @media (max-width: 1024px) { .grid-2, .grid-tables { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .card-head {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .card-title { font-size: 0.98rem; font-weight: 700; margin: 0; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  button, select, input {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.84rem;
    cursor: pointer;
  }
  button.primary { background: var(--blue); border-color: #3b82f6; }
  button.green { background: var(--green); border-color: #22c55e; }
  button:hover { filter: brightness(1.12); }
  .viewport {
    position: relative;
    background: #020617;
    border-radius: 8px;
    overflow: hidden;
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
  }
  .viewport img, .viewport video {
    width: 100%;
    height: 100%;
    object-fit: contain;
    transform-origin: center center;
  }
  .ptz-pad {
    display: grid;
    grid-template-columns: repeat(3, 52px);
    gap: 6px;
    justify-content: center;
    margin-top: 4px;
  }
  .ptz-pad button { padding: 10px 0; font-weight: 700; }
  canvas.timeline {
    width: 100%;
    height: 62px;
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: pointer;
  }
  .table-wrap { max-height: 290px; overflow: auto; border: 1px solid var(--border); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { padding: 7px 9px; border-bottom: 1px solid #1e293b; text-align: left; }
  th { position: sticky; top: 0; background: #1e293b; color: #cbd5e1; z-index: 1; }
  tr:hover td { background: #172554; }
  .log-box {
    max-height: 150px;
    overflow: auto;
    background: #020617;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 0.78rem;
    color: #cbd5e1;
    white-space: pre-wrap;
  }
  .muted { color: var(--muted); font-size: 0.8rem; }
  .token-bar {
    display: none;
    background: #451a03;
    border-bottom: 1px solid #d97706;
    padding: 8px 16px;
    gap: 8px;
    align-items: center;
  }
</style>
</head>
<body>
<div id="tokenBar" class="token-bar">
  <span>Acesso protegido por token:</span>
  <input id="tokenInput" type="password" placeholder="Digite HOMECAM_WEB_TOKEN" />
  <button class="primary" onclick="saveToken()">Entrar</button>
</div>
<header>
  <div class="brand">
    <span>📹 HomeCam Segurança 24/7</span>
    <span id="camNameBadge" class="badge">Carregando...</span>
  </div>
  <div class="badges">
    <span id="streamBadge" class="badge">Fluxo: --</span>
    <span id="recBadge" class="badge">Gravação: --</span>
    <span id="powerBadge" class="badge">Energia: --</span>
    <span id="driveBadge" class="badge">Drive: --</span>
    <button onclick="triggerSync()">☁ Sincronizar Drive</button>
  </div>
</header>
<main>
  <section class="grid-2">
    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Ao Vivo &amp; Detecções em Tempo Real</h2>
        <div class="toolbar">
          <label class="muted"><input type="checkbox" id="chkOverlay" checked onchange="reloadLiveStream()" /> Overlays IA</label>
          <span id="aiCounters" class="muted">IA inicializando...</span>
        </div>
      </div>
      <div class="viewport">
        <img id="liveImg" alt="Fluxo ao vivo HomeCam" />
      </div>
    </div>
    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Controle PTZ &amp; Telemetria 24h</h2>
        <span id="ptzBackend" class="muted">PTZ: --</span>
      </div>
      <div class="ptz-pad">
        <button onclick="ptzMove('left_up')">↖</button>
        <button onclick="ptzMove('up')">▲</button>
        <button onclick="ptzMove('right_up')">↗</button>
        <button onclick="ptzMove('left')">◀</button>
        <button class="primary" onclick="ptzHome()">🏠</button>
        <button onclick="ptzMove('right')">▶</button>
        <button onclick="ptzMove('left_down')">↙</button>
        <button onclick="ptzMove('down')">▼</button>
        <button onclick="ptzMove('right_down')">↘</button>
      </div>
      <div class="toolbar" style="justify-content:center;">
        <button onclick="ptzMove('zoom_in')">🔍 Zoom +</button>
        <button onclick="ptzMove('zoom_out')">🔎 Zoom -</button>
      </div>
      <div class="card-head" style="margin-top:6px;">
        <span class="muted">Logs recentes do servidor</span>
      </div>
      <div id="logsBox" class="log-box"></div>
    </div>
  </section>

  <section class="card">
    <div class="card-head">
      <h2 class="card-title">Linha do Tempo DVR 24h &amp; Reprodução Remota</h2>
      <div class="toolbar">
        <select id="dateSelect" onchange="loadTimeline()"></select>
        <button class="primary" onclick="refreshDates(true)">↻ Atualizar Dias</button>
        <button onclick="stepFrame(-1)">⏮ -1 Quadro</button>
        <button onclick="stepFrame(1)">⏭ +1 Quadro</button>
        <button onclick="setPlaybackSpeed(0.5)">0.5x</button>
        <button onclick="setPlaybackSpeed(1)">1x</button>
        <button onclick="setPlaybackSpeed(2)">2x</button>
        <button onclick="setPlaybackSpeed(4)">4x</button>
        <button onclick="setPlaybackSpeed(8)">8x</button>
        <button onclick="adjustZoom(1.35)">Zoom +</button>
        <button onclick="adjustZoom(1/1.35)">Zoom -</button>
        <button onclick="resetZoom()">Resetar Zoom</button>
      </div>
    </div>
    <canvas id="timelineCanvas" class="timeline" width="1200" height="62"></canvas>
    <div class="grid-2">
      <div id="dvrViewport" class="viewport">
        <video id="dvrVideo" controls playsinline preload="metadata"></video>
      </div>
      <div>
        <div class="muted" id="dvrStatus" style="margin-bottom:6px;">Selecione uma gravação na barra 24h ou na lista abaixo.</div>
        <div class="table-wrap" style="max-height:240px;">
          <table>
            <thead><tr><th>Hora</th><th>Tipo / Origem</th><th>Detalhes</th></tr></thead>
            <tbody id="timelineEventsBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <section class="grid-tables">
    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Veículos Detectados (ALPR, Cor e Modelo)</h2>
        <button class="green" onclick="downloadCsv('vehicles')">⬇ Exportar CSV</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Horário</th><th>Tipo</th><th>Placa</th><th>Cor</th><th>Modelo Aprox.</th></tr></thead>
          <tbody id="vehiclesBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-head">
        <h2 class="card-title">Pessoas &amp; Vestuário</h2>
        <button class="green" onclick="downloadCsv('people')">⬇ Exportar CSV</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Horário</th><th>Gênero</th><th>Cor das Roupas</th></tr></thead>
          <tbody id="peopleBody"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>
<script>
const URL_PARAMS = new URLSearchParams(window.location.search);
let AUTH_TOKEN = URL_PARAMS.get("token") || sessionStorage.getItem("homecam_token") || "";
if (URL_PARAMS.get("token")) sessionStorage.setItem("homecam_token", AUTH_TOKEN);

let CURRENT_TIMELINE = { segments: [], events: [] };
let CURRENT_SEG_FPS = 15;
let ZOOM = 1, PAN_X = 0, PAN_Y = 0, DRAGGING = false, DRAG_X = 0, DRAG_Y = 0;

function withToken(path) {
  if (!AUTH_TOKEN) return path;
  return path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(AUTH_TOKEN);
}
async function apiFetch(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (AUTH_TOKEN) headers["Authorization"] = "Bearer " + AUTH_TOKEN;
  const resp = await fetch(withToken(path), Object.assign({}, opts, { headers }));
  if (resp.status === 401) {
    document.getElementById("tokenBar").style.display = "flex";
    throw new Error("401");
  }
  return resp;
}
function saveToken() {
  AUTH_TOKEN = document.getElementById("tokenInput").value.trim();
  sessionStorage.setItem("homecam_token", AUTH_TOKEN);
  document.getElementById("tokenBar").style.display = "none";
  reloadLiveStream();
  refreshStatus();
  refreshTables();
  refreshDates(true);
}
function reloadLiveStream() {
  const ov = document.getElementById("chkOverlay").checked ? "1" : "0";
  document.getElementById("liveImg").src = withToken(`/api/stream/live.mjpg?overlay=${ov}&_t=${Date.now()}`);
}
function esc(text) {
  return String(text ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
}
function fmtBytes(b) {
  if (!b) return "0 KB";
  const mb = b / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;
}
function fmtSec(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

async function refreshStatus() {
  try {
    const data = await (await apiFetch("/api/status")).json();
    document.getElementById("camNameBadge").textContent = data.camera.name;
    const sb = document.getElementById("streamBadge");
    sb.textContent = data.camera.connected ? `Ao Vivo (${data.camera.fps} FPS)` : `Estado: ${data.camera.state}`;
    sb.className = "badge " + (data.camera.connected ? "ok" : "warn");
    const rb = document.getElementById("recBadge");
    rb.textContent = data.recording.active ? `Gravando ${data.recording.current_file} (${fmtBytes(data.recording.size_bytes)})` : "Gravação parada";
    rb.className = "badge " + (data.recording.active ? "rec" : "");
    const pb = document.getElementById("powerBadge");
    pb.textContent = data.power_guard_active ? "24/7 PC Ativo (Monitor livre)" : `SO: ${data.platform}`;
    pb.className = "badge " + (data.power_guard_active ? "ok" : "");
    const db = document.getElementById("driveBadge");
    db.textContent = data.drive.enabled ? `Drive: ${data.drive.pending_count} na fila` : "Drive: desligado";
    const s = data.ai.summary || {};
    document.getElementById("aiCounters").textContent =
      `Pessoas: ${s.people_total || 0} (H:${s.men_count || 0} M:${s.women_count || 0}) | Carros: ${s.cars_count || 0} | Motos: ${s.motos_count || 0}` +
      (s.last_plate ? ` | Placa: ${s.last_plate}` : "");
    document.getElementById("ptzBackend").textContent = data.ptz.available ? `PTZ: ${data.ptz.backend || "Ativo"}` : "PTZ: Indisponível";
    document.getElementById("logsBox").textContent = (data.logs || []).join("\\n");
  } catch (e) {}
}

async function refreshTables() {
  try {
    const [vr, pr] = await Promise.all([
      apiFetch("/api/vehicles?limit=150").then(r => r.json()),
      apiFetch("/api/people?limit=150").then(r => r.json())
    ]);
    document.getElementById("vehiclesBody").innerHTML = (vr.records || []).map(r =>
      `<tr><td>${esc(r.timestamp)}</td><td>${esc(r.vehicle_type)}</td><td><b>${esc(r.plate)}</b></td><td>${esc(r.color)}</td><td>${esc(r.approx_model)}</td></tr>`
    ).join("");
    document.getElementById("peopleBody").innerHTML = (pr.records || []).map(r =>
      `<tr><td>${esc(r.timestamp)}</td><td><b>${esc(r.person_type)}</b></td><td>${esc(r.clothing_colors)}</td></tr>`
    ).join("");
  } catch (e) {}
}

async function refreshDates(loadFirst = false) {
  try {
    const data = await (await apiFetch("/api/timeline/dates")).json();
    const sel = document.getElementById("dateSelect");
    const prev = sel.value;
    sel.innerHTML = (data.dates || []).map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
    if (prev && (data.dates || []).includes(prev)) sel.value = prev;
    if (loadFirst && sel.value) await loadTimeline();
  } catch (e) {}
}

function drawTimeline() {
  const cv = document.getElementById("timelineCanvas");
  const ctx = cv.getContext("2d");
  const w = cv.width, h = cv.height;
  ctx.fillStyle = "#090e1a";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#1e293b";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  for (let hr = 0; hr <= 24; hr += 2) {
    const x = (hr / 24) * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    if (hr < 24) ctx.fillText(String(hr).padStart(2, "0") + "h", x + 3, 12);
  }
  for (const seg of CURRENT_TIMELINE.segments) {
    const x1 = (seg.start_sec / 86400) * w;
    const sw = Math.max(3, ((seg.end_sec - seg.start_sec) / 86400) * w);
    ctx.fillStyle = seg.is_motion_clip ? "#d97706" : (seg.source_kind === "drive_cloud" ? "#0284c7" : "#16a34a");
    ctx.fillRect(x1, 20, sw, 22);
  }
  const evColors = { motion: "#f59e0b", ai_person: "#3b82f6", ai_vehicle: "#06b6d4", ai_plate: "#10b981", ai_animal: "#a855f7" };
  for (const ev of CURRENT_TIMELINE.events) {
    const x = (ev.sec_of_day / 86400) * w;
    ctx.fillStyle = evColors[ev.event_type] || "#e2e8f0";
    ctx.fillRect(x, 45, 3, 14);
  }
}

function playSegment(seg, seekOffsetSec = 0) {
  if (!seg) return;
  const v = document.getElementById("dvrVideo");
  CURRENT_SEG_FPS = Math.max(1, seg.fps || 15);
  const src = withToken(`/api/video?date=${encodeURIComponent(seg.date_str)}&file=${encodeURIComponent(seg.filename)}&drive_id=${encodeURIComponent(seg.drive_file_id || "")}`);
  document.getElementById("dvrStatus").textContent = `Reproduzindo ${seg.filename} (${fmtSec(seg.start_sec)} — ${seg.source_kind})`;
  if (v.dataset.seg !== seg.filename) {
    v.dataset.seg = seg.filename;
    v.src = src;
    v.onloadedmetadata = () => {
      if (seekOffsetSec > 0 && seekOffsetSec < (v.duration || 1e9)) v.currentTime = seekOffsetSec;
      v.play().catch(() => {});
    };
  } else {
    if (seekOffsetSec >= 0) v.currentTime = seekOffsetSec;
    v.play().catch(() => {});
  }
}

async function loadTimeline() {
  const dateStr = document.getElementById("dateSelect").value;
  if (!dateStr) return;
  try {
    const data = await (await apiFetch(`/api/timeline/day?date=${encodeURIComponent(dateStr)}`)).json();
    CURRENT_TIMELINE = data;
    drawTimeline();
    const rows = [];
    for (const seg of data.segments || []) {
      rows.push(`<tr style="cursor:pointer" onclick="jumpToSec(${seg.start_sec})"><td>${fmtSec(seg.start_sec)}</td><td>Vídeo (${esc(seg.source_kind)})</td><td>${esc(seg.filename)} (${Math.round(seg.duration_sec)}s)</td></tr>`);
    }
    for (const ev of data.events || []) {
      rows.push(`<tr style="cursor:pointer" onclick="jumpToSec(${ev.sec_of_day})"><td>${fmtSec(ev.sec_of_day)}</td><td>${esc(ev.event_label)}</td><td>${esc(ev.label)}</td></tr>`);
    }
    document.getElementById("timelineEventsBody").innerHTML = rows.join("");
    if (data.segments && data.segments.length && !document.getElementById("dvrVideo").src) {
      playSegment(data.segments[0], 0);
    }
  } catch (e) {}
}

function jumpToSec(daySec) {
  const segs = CURRENT_TIMELINE.segments || [];
  let chosen = segs.find(s => daySec >= s.start_sec && daySec <= s.end_sec);
  if (!chosen && segs.length) {
    chosen = segs.reduce((a, b) => Math.abs(b.start_sec - daySec) < Math.abs(a.start_sec - daySec) ? b : a);
  }
  if (chosen) playSegment(chosen, Math.max(0, daySec - chosen.start_sec));
}

document.getElementById("timelineCanvas").addEventListener("click", ev => {
  const rect = ev.target.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
  jumpToSec(ratio * 86400);
});

function setPlaybackSpeed(rate) { document.getElementById("dvrVideo").playbackRate = rate; }
function stepFrame(delta) {
  const v = document.getElementById("dvrVideo");
  v.pause();
  v.currentTime = Math.max(0, v.currentTime + delta * (1.0 / CURRENT_SEG_FPS));
}
function applyZoomTransform() {
  document.getElementById("dvrVideo").style.transform = `translate(${PAN_X}px, ${PAN_Y}px) scale(${ZOOM})`;
}
function adjustZoom(factor) { ZOOM = Math.max(1, Math.min(6, ZOOM * factor)); if (ZOOM <= 1.01) { ZOOM = 1; PAN_X = 0; PAN_Y = 0; } applyZoomTransform(); }
function resetZoom() { ZOOM = 1; PAN_X = 0; PAN_Y = 0; applyZoomTransform(); }
const vp = document.getElementById("dvrViewport");
vp.addEventListener("mousedown", e => { if (ZOOM > 1) { DRAGGING = true; DRAG_X = e.clientX - PAN_X; DRAG_Y = e.clientY - PAN_Y; } });
window.addEventListener("mousemove", e => { if (DRAGGING) { PAN_X = e.clientX - DRAG_X; PAN_Y = e.clientY - DRAG_Y; applyZoomTransform(); } });
window.addEventListener("mouseup", () => { DRAGGING = false; });

async function ptzMove(direction) {
  try { await apiFetch("/api/ptz/move", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ direction, duration: 0.35 }) }); } catch (e) {}
}
async function ptzHome() {
  try { await apiFetch("/api/ptz/home", { method: "POST" }); } catch (e) {}
}
async function triggerSync() {
  try { await apiFetch("/api/drive/sync", { method: "POST" }); refreshStatus(); } catch (e) {}
}
function downloadCsv(kind) { window.open(withToken(`/api/export/${kind}.csv`), "_blank"); }

reloadLiveStream();
refreshStatus();
refreshTables();
refreshDates(true);
setInterval(refreshStatus, 2500);
setInterval(refreshTables, 2500);
</script>
</body>
</html>
"""


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _build_csv_bytes(nvr: HeadlessNVRService, kind: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{kind}.csv"
        if kind == "vehicles":
            nvr.ai_pipeline.export_csv(out_path)
        else:
            nvr.ai_pipeline.export_people_csv(out_path)
        return out_path.read_bytes()


class NVRWebServer:
    """Servidor HTTP/REST multi-thread desacoplado para o Portal Web e sincronização remota."""

    def __init__(
        self,
        nvr: HeadlessNVRService,
        host: str = "0.0.0.0",
        port: int = 8765,
        auth_token: str = "",
    ) -> None:
        self.nvr = nvr
        self.host = (host or "0.0.0.0").strip()
        self.port = int(port)
        self.auth_token = (auth_token or "").strip()

        self._stop_event = threading.Event()
        self._httpd: _ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def actual_port(self) -> int:
        if self._httpd is not None:
            return int(self._httpd.server_address[1])
        return self.port

    def _is_authorized(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> bool:
        if not self.auth_token:
            return True
        auth_hdr = handler.headers.get("Authorization") or ""
        candidate = ""
        if auth_hdr.lower().startswith("bearer "):
            candidate = auth_hdr[7:].strip()
        elif "token" in query and query["token"]:
            candidate = query["token"][0].strip()
        return bool(candidate) and hmac.compare_digest(candidate, self.auth_token)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        portal = self
        nvr = self.nvr
        stop_ev = self._stop_event

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def _send_json(self, status: int, payload: object) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, status: int, content_type: str, body: bytes, filename: str = "") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if filename:
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(body)

            def _stream_file_with_range(self, file_path: Path) -> None:
                file_size = file_path.stat().st_size
                range_hdr = (self.headers.get("Range") or "").strip()
                start = 0
                end = max(0, file_size - 1)
                status = HTTPStatus.OK

                if range_hdr and file_size > 0:
                    m = _RANGE_RE.match(range_hdr)
                    if not m:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    s_raw, e_raw = m.group(1), m.group(2)
                    if s_raw and e_raw:
                        start, end = int(s_raw), int(e_raw)
                    elif s_raw:
                        start = int(s_raw)
                    elif e_raw:
                        suffix = int(e_raw)
                        start = max(0, file_size - suffix)
                    if start >= file_size or start > end:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    end = min(end, file_size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT

                length = max(0, (end - start + 1) if file_size > 0 else 0)
                self.send_response(status)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.end_headers()

                with file_path.open("rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0 and not stop_ev.is_set():
                        chunk = fh.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

            def _stream_mjpeg(self, with_overlays: bool) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                last_seq = -1
                try:
                    while not stop_ev.is_set():
                        seq, jpg = nvr.wait_next_jpeg(last_seq, timeout=0.6, with_overlays=with_overlays)
                        if jpg is None or seq == last_seq:
                            continue
                        last_seq = seq
                        part_hdr = (
                            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                        ).encode("ascii")
                        self.wfile.write(part_hdr + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                route = parsed.path or "/"
                query = parse_qs(parsed.query)

                if route in ("/", "/index.html"):
                    self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", _PORTAL_HTML.encode("utf-8"))
                    return

                if route == "/api/health":
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return

                if not portal._is_authorized(self, query):
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Token de acesso inválido ou ausente."})
                    return

                if route == "/api/status":
                    self._send_json(HTTPStatus.OK, nvr.get_status_payload())
                elif route == "/api/vehicles":
                    limit = int((query.get("limit") or ["200"])[0])
                    self._send_json(HTTPStatus.OK, {"records": nvr.get_vehicles_payload(limit=limit)})
                elif route == "/api/people":
                    limit = int((query.get("limit") or ["200"])[0])
                    self._send_json(HTTPStatus.OK, {"records": nvr.get_people_payload(limit=limit)})
                elif route == "/api/timeline/dates":
                    cam = (query.get("camera") or [""])[0] or None
                    cloud = (query.get("cloud") or ["1"])[0] != "0"
                    self._send_json(HTTPStatus.OK, {"dates": nvr.get_timeline_dates(camera_name=cam, include_cloud=cloud)})
                elif route == "/api/timeline/day":
                    date_str = (query.get("date") or [""])[0].strip()
                    cam = (query.get("camera") or [""])[0] or None
                    cloud = (query.get("cloud") or ["1"])[0] != "0"
                    try:
                        payload = nvr.get_day_timeline(date_str, camera_name=cam, include_cloud=cloud)
                        self._send_json(HTTPStatus.OK, payload)
                    except ValueError as err:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                elif route == "/api/video":
                    date_str = (query.get("date") or [""])[0].strip()
                    filename = (query.get("file") or [""])[0].strip()
                    cam = (query.get("camera") or [""])[0] or None
                    drive_id = (query.get("drive_id") or [""])[0].strip()
                    try:
                        path = nvr.resolve_video_path(date_str, filename, camera_name=cam, drive_file_id=drive_id)
                        self._stream_file_with_range(path)
                    except ValueError as err:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                    except FileNotFoundError as err:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": str(err)})
                    except Exception as err:
                        self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(err)})
                elif route == "/api/stream/live.mjpg":
                    ov = (query.get("overlay") or ["1"])[0] != "0"
                    self._stream_mjpeg(with_overlays=ov)
                elif route == "/api/snapshot.jpg":
                    ov = (query.get("overlay") or ["1"])[0] != "0"
                    jpg = nvr.get_latest_jpeg(with_overlays=ov)
                    if jpg is None:
                        self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Sem quadro ao vivo no momento."})
                    else:
                        self._send_bytes(HTTPStatus.OK, "image/jpeg", jpg)
                elif route in ("/api/export/vehicles.csv", "/api/export/people.csv"):
                    kind = "vehicles" if "vehicles" in route else "people"
                    csv_bytes = _build_csv_bytes(nvr, kind)
                    self._send_bytes(
                        HTTPStatus.OK,
                        "text/csv; charset=utf-8",
                        csv_bytes,
                        filename=f"{kind}_homecam.csv",
                    )
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint não encontrado."})

            def do_POST(self) -> None:
                parsed = urlsplit(self.path)
                route = parsed.path or "/"
                query = parse_qs(parsed.query)

                if not portal._is_authorized(self, query):
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Token de acesso inválido ou ausente."})
                    return

                content_len = min(65536, max(0, int(self.headers.get("Content-Length") or 0)))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    data = json.loads(raw_body.decode("utf-8") or "{}")
                except Exception:
                    data = {}

                if route == "/api/ptz/move":
                    ok, msg = nvr.ptz_move(
                        str(data.get("direction") or ""),
                        duration_sec=float(data.get("duration") or 0.35),
                        speed=int(data["speed"]) if data.get("speed") else None,
                    )
                    self._send_json(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST, {"ok": ok, "message": msg})
                elif route == "/api/ptz/home":
                    ok, msg = nvr.ptz_home()
                    self._send_json(HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST, {"ok": ok, "message": msg})
                elif route == "/api/drive/sync":
                    pending = nvr.trigger_drive_sync()
                    self._send_json(HTTPStatus.OK, {"ok": True, "pending_count": pending})
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint POST não encontrado."})

        return _Handler

    def start(self) -> int:
        """Inicia o servidor HTTP em thread dedicada e retorna a porta escutada."""
        if self._httpd is not None:
            return self.actual_port
        self._stop_event.clear()
        self._httpd = _ReusableThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="HomeCamWebServer",
            daemon=True,
        )
        self._thread.start()
        return self.actual_port

    def stop(self) -> None:
        """Encerra o servidor HTTP e aguarda finalização limpa."""
        self._stop_event.set()
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

