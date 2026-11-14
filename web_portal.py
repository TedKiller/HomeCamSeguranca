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
import time
from urllib.parse import parse_qs, urlsplit

from frame_interpolator import LiveFrameInterpolator
from headless_nvr import HeadlessNVRService

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_PORTAL_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>HomeCam Segurança 24/7 — Monitoramento</title>
<style>
  :root {
    --bg:#0b0c0e; --panel:#131418; --panel2:#181a1f; --panel3:#1e2026;
    --txt:#f2f3f5; --mute:#8b8f98; --dim:#5a5e66;
    --orange:#ff7a1a; --orange-glow:rgba(255,122,26,.35);
    --green:#3ddc97; --red:#ff4d5e; --r:14px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html, body { height:100%; }
  body { background:var(--bg); color:var(--txt); font:400 13px/1.45 "Inter","Segoe UI",system-ui,sans-serif; overflow:hidden; }
  button, select, input { font:inherit; color:inherit; }
  .app { display:grid; grid-template-columns:64px minmax(0,1fr) 390px; height:100vh; gap:14px; padding:14px; }

  /* Dock */
  .dock { background:var(--panel); border-radius:var(--r); display:flex; flex-direction:column; align-items:center; padding:16px 0; gap:8px; }
  .logo { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,var(--orange),#ff4d00); display:grid; place-items:center; font-weight:700; margin-bottom:14px; box-shadow:0 0 18px var(--orange-glow); color:#111; }
  .dock button { width:40px; height:40px; border:0; border-radius:10px; display:grid; place-items:center; color:var(--mute); background:transparent; font-size:17px; cursor:pointer; transition:.2s; }
  .dock button:hover { background:var(--panel3); color:var(--txt); }
  .dock button.on { background:rgba(255,122,26,.12); color:var(--orange); }
  .dock .sp { flex:1; }

  /* Main */
  main { display:flex; flex-direction:column; gap:14px; min-width:0; min-height:0; }
  header { display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }
  header h1 { font-size:18px; font-weight:600; }
  header h1 span { color:var(--mute); font-weight:400; font-size:12.5px; margin-left:10px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; }
  .chip { background:var(--panel); padding:7px 12px; border-radius:999px; color:var(--mute); font-size:12px; display:flex; align-items:center; gap:7px; border:0; }
  button.chip { cursor:pointer; } button.chip:hover { color:var(--txt); }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--dim); }
  .dot.ok { background:var(--green); box-shadow:0 0 8px var(--green); }
  .dot.red { background:var(--red); box-shadow:0 0 8px var(--red); }
  .dot.warn { background:var(--orange); box-shadow:0 0 8px var(--orange); }

  .player { position:relative; flex:1; min-height:240px; border-radius:var(--r); overflow:hidden; background:radial-gradient(ellipse at 30% 40%,#1d2026 0%,#111215 60%,#0b0c0e 100%); user-select:none; }
  .player img, .player video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; transform-origin:center center; }
  .overlay-layer { position:absolute; inset:0; pointer-events:none; transform-origin:center center; z-index:2; }
  #overlayBox { position:absolute; overflow:visible; }
  .bbox { position:absolute; border:1.5px solid var(--orange); border-radius:6px; box-shadow:0 0 14px var(--orange-glow); transition:left .12s linear,top .12s linear,width .12s linear,height .12s linear; }
  .bbox.p { border-color:var(--green); box-shadow:0 0 14px rgba(61,220,151,.32); }
  .bbox.a { border-color:#4da3ff; box-shadow:0 0 14px rgba(77,163,255,.32); }
  .bbox i { position:absolute; top:-22px; left:-1px; font-style:normal; font-size:10.5px; background:var(--orange); color:#111; padding:2px 7px; border-radius:5px 5px 5px 0; font-weight:600; white-space:nowrap; }
  .bbox.p i { background:var(--green); }
  .bbox.a i { background:#4da3ff; }
  .player.zoomed { cursor:grab; } .player.dragging { cursor:grabbing; }
  .ov-tl, .ov-tr { position:absolute; top:14px; display:flex; gap:8px; z-index:3; flex-wrap:wrap; }
  .ov-tl { left:14px; } .ov-tr { right:14px; }
  .tag { background:rgba(10,10,12,.6); backdrop-filter:blur(8px); padding:6px 10px; border-radius:8px; font-size:11.5px; display:flex; align-items:center; gap:6px; border:0; }
  label.tag, button.tag { cursor:pointer; }
  .live { color:var(--red); font-weight:600; } .pb { color:var(--orange); font-weight:600; }
  .ctrl { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); display:flex; gap:4px; padding:6px; border-radius:12px; background:rgba(12,12,14,.65); backdrop-filter:blur(12px); z-index:3; }
  .ctrl button { min-width:34px; height:34px; border:0; border-radius:8px; background:transparent; font-size:14px; cursor:pointer; padding:0 8px; }
  .ctrl button:hover { background:rgba(255,255,255,.08); }
  .ctrl button.pr { background:var(--orange); color:#111; box-shadow:0 0 14px var(--orange-glow); }
  .ctrl button.act { color:var(--orange); }
  .ctrl .sep { width:1px; background:rgba(255,255,255,.1); margin:6px 4px; }
  .ctrl .spd { font-size:12px; color:var(--mute); }
  .zoom { position:absolute; right:14px; bottom:16px; display:flex; flex-direction:column; gap:4px; padding:5px; border-radius:10px; background:rgba(12,12,14,.65); backdrop-filter:blur(12px); z-index:3; }
  .zoom button { width:30px; height:30px; border:0; border-radius:7px; background:transparent; cursor:pointer; }
  .zoom button:hover { background:rgba(255,255,255,.08); }
  .zoom span { font-size:10px; color:var(--mute); text-align:center; }
  .mini { position:absolute; left:14px; bottom:16px; width:120px; height:68px; border-radius:8px; background:rgba(12,12,14,.65); backdrop-filter:blur(8px); overflow:hidden; display:none; z-index:3; }
  .mini b { position:absolute; border:1.5px solid var(--orange); border-radius:3px; }
  .status-line { position:absolute; left:14px; top:50px; font-size:11.5px; color:var(--mute); z-index:3; max-width:60%; }

  /* Timeline */
  .tl { background:var(--panel); border-radius:var(--r); padding:14px 18px; }
  .tl-h { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:10px; flex-wrap:wrap; }
  .tl-h b { font-weight:600; }
  .tl-h select { background:var(--panel2); border:0; border-radius:8px; padding:5px 10px; margin-left:10px; cursor:pointer; }
  .tl-h .ref { background:var(--panel2); border:0; border-radius:8px; padding:5px 9px; cursor:pointer; color:var(--mute); }
  .legend { display:flex; gap:14px; color:var(--mute); font-size:11.5px; flex-wrap:wrap; }
  .legend span::before { content:""; display:inline-block; width:10px; height:4px; border-radius:2px; margin-right:6px; vertical-align:middle; background:var(--c); }
  canvas.timeline { width:100%; height:58px; display:block; cursor:pointer; border-radius:8px; }

  /* Side */
  aside { display:flex; flex-direction:column; gap:14px; min-height:0; }
  .card { background:var(--panel); border-radius:var(--r); padding:16px; }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .stat { background:var(--panel2); border-radius:10px; padding:12px; }
  .stat small { color:var(--mute); font-size:11px; }
  .stat div { font-size:22px; font-weight:600; margin-top:2px; }
  .stat em { font-style:normal; font-size:11px; color:var(--mute); }
  .stat.al div { color:var(--red); }
  .feed { flex:1; display:flex; flex-direction:column; min-height:0; }
  .feed-h { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:8px; }
  .feed-h b { font-weight:600; }
  .tabs { display:flex; background:var(--panel2); border-radius:8px; padding:3px; }
  .tabs button { padding:5px 10px; border:0; border-radius:6px; font-size:11.5px; color:var(--mute); background:transparent; cursor:pointer; }
  .tabs button.on { background:var(--panel3); color:var(--txt); }
  .feed-actions { display:flex; gap:6px; margin-top:10px; }
  .feed-actions button { flex:1; background:var(--panel2); border:0; border-radius:8px; padding:7px; color:var(--mute); cursor:pointer; font-size:11.5px; }
  .feed-actions button:hover { color:var(--orange); }
  table { width:100%; border-collapse:collapse; }
  th { color:var(--dim); font-weight:500; font-size:10.5px; text-transform:uppercase; letter-spacing:.6px; text-align:left; padding:0 6px 8px; position:sticky; top:0; background:var(--panel); z-index:1; }
  td { padding:9px 6px; font-weight:300; vertical-align:middle; }
  tbody tr:nth-child(odd) td { background:rgba(255,255,255,.018); }
  tbody tr td:first-child { border-radius:8px 0 0 8px; }
  tbody tr td:last-child { border-radius:0 8px 8px 0; }
  tbody tr:hover td { background:rgba(255,122,26,.06); }
  tbody tr.click { cursor:pointer; }
  .t { color:var(--mute); font-variant-numeric:tabular-nums; font-size:12px; white-space:nowrap; }
  .ic { width:26px; height:26px; border-radius:7px; display:inline-grid; place-items:center; background:var(--panel3); font-size:13px; }
  .plate { font-family:"JetBrains Mono",Consolas,monospace; font-size:11.5px; font-weight:500; background:#f2f3f5; color:#111; padding:2px 6px; border-radius:4px; letter-spacing:.5px; white-space:nowrap; }
  .sw { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:-1px; border:1px solid rgba(255,255,255,.25); }
  .pill { font-size:10.5px; padding:2px 7px; border-radius:999px; background:var(--panel3); color:var(--mute); white-space:nowrap; }
  .pill.o { background:rgba(255,122,26,.14); color:var(--orange); }
  .empty { color:var(--dim); text-align:center; padding:24px 0; }
  .scroll { overflow:auto; flex:1; min-height:0; }
  .scroll::-webkit-scrollbar { width:4px; } .scroll::-webkit-scrollbar-thumb { background:var(--panel3); border-radius:4px; }

  /* PTZ / Logs drawer */
  .drawer { display:none; }
  .drawer.open { display:block; }
  .ptz-pad { display:grid; grid-template-columns:repeat(3,44px); gap:6px; justify-content:center; margin:6px 0 10px; }
  .ptz-pad button, .ptz-z button { height:38px; border:0; border-radius:9px; background:var(--panel2); cursor:pointer; }
  .ptz-pad button:hover, .ptz-z button:hover { background:var(--panel3); color:var(--orange); }
  .ptz-pad button.home { background:rgba(255,122,26,.14); color:var(--orange); }
  .ptz-z { display:flex; gap:6px; justify-content:center; } .ptz-z button { padding:0 14px; }
  .log-box { max-height:160px; overflow:auto; background:var(--panel2); border-radius:8px; padding:8px 10px; font-family:Consolas,monospace; font-size:11px; color:var(--mute); white-space:pre-wrap; margin-top:10px; }
  .card-h { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
  .card-h b { font-weight:600; } .muted { color:var(--mute); font-size:11.5px; }

  .token-bar { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7); backdrop-filter:blur(6px); z-index:50; align-items:center; justify-content:center; }
  .token-box { background:var(--panel); border-radius:var(--r); padding:22px; display:flex; flex-direction:column; gap:10px; width:320px; }
  .token-box input { background:var(--panel2); border:0; border-radius:8px; padding:9px 11px; }
  .token-box button { background:var(--orange); color:#111; border:0; border-radius:8px; padding:9px; font-weight:600; cursor:pointer; }

  @media (max-width:1100px) {
    body { overflow:auto; }
    .app { grid-template-columns:1fr; height:auto; }
    .dock { flex-direction:row; padding:8px 12px; justify-content:flex-start; }
    .logo { margin:0 10px 0 0; } .dock .sp { flex:1; }
    .player { min-height:56vw; }
    .feed { max-height:70vh; }
  }
</style>
</head>
<body>
<div id="tokenBar" class="token-bar">
  <div class="token-box">
    <b>Acesso protegido</b>
    <span class="muted">Digite o token HOMECAM_WEB_TOKEN</span>
    <input id="tokenInput" type="password" placeholder="Token" />
    <button onclick="saveToken()">Entrar</button>
  </div>
</div>
<div class="app">
  <nav class="dock">
    <div class="logo">H</div>
    <button id="navLive" class="on" title="Ao vivo" onclick="goLive()">◉</button>
    <button id="navRec" title="Gravações / DVR" onclick="goPlayback()">▤</button>
    <button id="navAll" title="Eventos IA" onclick="setFeed('all')">◈</button>
    <button id="navVeh" title="Veículos" onclick="setFeed('vehicles')">⛟</button>
    <button id="navPpl" title="Pessoas" onclick="setFeed('people')">☺</button>
    <div class="sp"></div>
    <button id="navPtz" title="PTZ &amp; Logs" onclick="toggleDrawer()">⚙</button>
  </nav>

  <main>
    <header>
      <h1 id="camNameBadge">Carregando… <span id="camMeta"></span></h1>
      <div class="chips">
        <div class="chip"><span id="streamDot" class="dot"></span><span id="streamBadge">Fluxo: --</span></div>
        <div class="chip"><span id="recDot" class="dot"></span><span id="recBadge">Gravação: --</span></div>
        <div class="chip"><span id="powerDot" class="dot"></span><span id="powerBadge">Energia: --</span></div>
        <button class="chip" onclick="triggerSync()" title="Sincronizar Google Drive"><span id="driveDot" class="dot"></span><span id="driveBadge">Drive: --</span></button>
      </div>
    </header>

    <section id="dvrViewport" class="player">
      <img id="liveImg" alt="Fluxo ao vivo HomeCam" />
      <div id="overlayLayer" class="overlay-layer"><div id="overlayBox"></div></div>
      <video id="dvrVideo" playsinline preload="metadata" style="display:none"></video>

      <div class="ov-tl">
        <div class="tag" id="modeTag"><span class="live">● AO VIVO</span></div>
        <div class="tag" id="clockTag">--</div>
      </div>
      <div class="ov-tr">
        <div class="tag" id="aiCounters">◈ IA inicializando…</div>
        <label class="tag"><input type="checkbox" id="chkEnhance" onchange="reloadLiveStream()" style="accent-color:#ff7a1a" /> ✨ Nitidez</label>
        <label class="tag"><input type="checkbox" id="chkFps30" onchange="reloadLiveStream()" style="accent-color:#ff7a1a" /> ⚡ 30 FPS</label>
        <label class="tag"><input type="checkbox" id="chkOverlay" checked onchange="reloadLiveStream()" style="accent-color:#ff7a1a" /> Overlays IA</label>
        <button class="tag" onclick="toggleFullscreen()" title="Tela cheia">⤢</button>
      </div>
      <div class="status-line" id="dvrStatus"></div>

      <div class="mini" id="miniMap"><b id="miniBox"></b></div>

      <div class="ctrl">
        <button id="btnRew" title="Retroceder (2x/4x/8x/16x)" onclick="rewind()">«</button>
        <button title="Quadro anterior" onclick="stepFrame(-1)">⏮</button>
        <button id="btnPlay" class="pr" title="Play / Pause" onclick="togglePlay()">❚❚</button>
        <button title="Próximo quadro" onclick="stepFrame(1)">⏭</button>
        <button id="btnFf" title="Avançar (2x/4x/8x/16x)" onclick="fastForward()">»</button>
        <div class="sep"></div>
        <button class="spd" id="speedLbl" title="Voltar para 1x" onclick="setPlaybackSpeed(1)">1×</button>
        <button title="Snapshot" onclick="openSnapshot()">◰</button>
        <button title="Exportar clipe 30 FPS" onclick="exportClip30()" style="width:auto;padding:0 9px;font-size:11.5px">⤓ 30 FPS</button>
        <button id="btnLive" title="Voltar ao vivo" onclick="goLive()">● Live</button>
      </div>

      <div class="zoom">
        <button title="Zoom +" onclick="adjustZoom(1.35)">＋</button>
        <span id="zoomLbl">1.0×</span>
        <button title="Zoom -" onclick="adjustZoom(1/1.35)">－</button>
        <button title="Resetar zoom" onclick="resetZoom()">⟲</button>
      </div>
    </section>

    <section class="tl">
      <div class="tl-h">
        <div><b>Linha do tempo · 24h</b><select id="dateSelect" onchange="loadTimeline()"></select> <button class="ref" title="Atualizar dias" onclick="refreshDates(true)">↻</button></div>
        <div class="legend">
          <span style="--c:rgba(255,255,255,.28)">Gravação contínua</span>
          <span style="--c:rgba(255,122,26,.45)">Movimento</span>
          <span style="--c:#ff7a1a">IA · Pessoa/Veículo</span>
          <span style="--c:#4da3ff">Nuvem</span>
        </div>
      </div>
      <canvas id="timelineCanvas" class="timeline"></canvas>
    </section>
  </main>

  <aside>
    <div class="card stats">
      <div class="stat"><small>Pessoas</small><div id="stPeople">0</div><em id="stPeopleSub">H 0 · M 0</em></div>
      <div class="stat"><small>Veículos</small><div id="stVeh">0</div><em id="stVehSub">0 carros · 0 motos</em></div>
      <div class="stat al"><small>Eventos</small><div id="stEv">0</div><em>no dia</em></div>
    </div>

    <div id="ptzDrawer" class="card drawer">
      <div class="card-h"><b>Controle PTZ</b><span id="ptzBackend" class="muted">PTZ: --</span></div>
      <div class="ptz-pad">
        <button onclick="ptzMove('left_up')">↖</button><button onclick="ptzMove('up')">▲</button><button onclick="ptzMove('right_up')">↗</button>
        <button onclick="ptzMove('left')">◀</button><button class="home" onclick="ptzHome()">⌂</button><button onclick="ptzMove('right')">▶</button>
        <button onclick="ptzMove('left_down')">↙</button><button onclick="ptzMove('down')">▼</button><button onclick="ptzMove('right_down')">↘</button>
      </div>
      <div class="ptz-z"><button onclick="ptzMove('zoom_in')">Zoom ＋</button><button onclick="ptzMove('zoom_out')">Zoom －</button></div>
      <div id="logsBox" class="log-box"></div>
    </div>

    <div class="card feed">
      <div class="feed-h">
        <b id="feedTitle">Eventos em tempo real</b>
        <div class="tabs">
          <button id="tabAll" class="on" onclick="setFeed('all')">Todos</button>
          <button id="tabVeh" onclick="setFeed('vehicles')">Veículos</button>
          <button id="tabPpl" onclick="setFeed('people')">Pessoas</button>
          <button id="tabDvr" onclick="setFeed('dvr')">DVR</button>
        </div>
      </div>
      <div class="scroll">
        <table>
          <thead id="feedHead"></thead>
          <tbody id="feedBody"></tbody>
        </table>
      </div>
      <div class="feed-actions">
        <button onclick="downloadCsv('vehicles')">⤓ CSV Veículos</button>
        <button onclick="downloadCsv('people')">⤓ CSV Pessoas</button>
      </div>
    </div>
  </aside>
</div>
<script>
const URL_PARAMS = new URLSearchParams(window.location.search);
let AUTH_TOKEN = URL_PARAMS.get("token") || sessionStorage.getItem("homecam_token") || "";
if (URL_PARAMS.get("token")) sessionStorage.setItem("homecam_token", AUTH_TOKEN);

let CURRENT_TIMELINE = { segments: [], events: [] };
let CURRENT_SEG = null;
let CURRENT_SEG_FPS = 15;
let MODE = "live";
let FEED = "all";
let VEHICLES = [], PEOPLE = [];
let SPEED = 1, REVERSE_TIMER = null;
let ZOOM = 1, PAN_X = 0, PAN_Y = 0, DRAGGING = false, DRAG_X = 0, DRAG_Y = 0;

const COLOR_MAP = {
  preto:"#111", preta:"#111", branco:"#f5f5f5", branca:"#f5f5f5", prata:"#c0c4ca", cinza:"#7a7d83",
  vermelho:"#d32f2f", vermelha:"#d32f2f", azul:"#2f6fe0", verde:"#3a9d54", amarelo:"#f2c230", amarela:"#f2c230",
  laranja:"#ff7a1a", marrom:"#7b4a2a", bege:"#d8c3a0", rosa:"#e91e63", roxo:"#7e3fbf", roxa:"#7e3fbf",
  vinho:"#7a1530", dourado:"#c9a227", dourada:"#c9a227", jeans:"#2b4a8a", bordo:"#7a1530"
};

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
  reloadLiveStream(); refreshStatus(); refreshTables(); refreshDates(true);
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
function timeOnly(ts) {
  const m = String(ts || "").match(/(\\d{2}:\\d{2}:\\d{2})/);
  return m ? m[1] : String(ts || "");
}
function swatches(text) {
  const raw = String(text || "");
  let out = esc(raw);
  const words = raw.toLowerCase().normalize("NFD").replace(/[\\u0300-\\u036f]/g, "").match(/[a-z]+/g) || [];
  const found = [];
  for (const w of words) if (COLOR_MAP[w] && !found.includes(COLOR_MAP[w])) found.push(COLOR_MAP[w]);
  return found.slice(0, 3).map(c => `<span class="sw" style="background:${c}"></span>`).join("") + out;
}
function setDot(id, cls) { document.getElementById(id).className = "dot " + (cls || ""); }

/* ---------------- Modo Ao vivo / Playback ---------------- */
function activeMedia() { return MODE === "live" ? document.getElementById("liveImg") : document.getElementById("dvrVideo"); }
function setMode(mode) {
  MODE = mode;
  const img = document.getElementById("liveImg"), v = document.getElementById("dvrVideo");
  img.style.display = mode === "live" ? "" : "none";
  document.getElementById("overlayLayer").style.display = mode === "live" ? "" : "none";
  v.style.display = mode === "live" ? "none" : "";
  document.getElementById("navLive").classList.toggle("on", mode === "live");
  document.getElementById("navRec").classList.toggle("on", mode !== "live");
  document.getElementById("btnLive").classList.toggle("act", mode === "live");
  document.getElementById("modeTag").innerHTML = mode === "live" ? '<span class="live">● AO VIVO</span>' : '<span class="pb">▶ PLAYBACK</span>';
  resetZoom();
  updatePlayBtn();
}
function goLive() {
  stopReverse();
  const v = document.getElementById("dvrVideo"); v.pause();
  setMode("live");
  document.getElementById("dvrStatus").textContent = "";
  reloadLiveStream();
}
function goPlayback() {
  setFeed("dvr");
  if (CURRENT_SEG) { setMode("playback"); return; }
  const segs = CURRENT_TIMELINE.segments || [];
  if (segs.length) playSegment(segs[segs.length - 1], 0);
  else document.getElementById("dvrStatus").textContent = "Nenhuma gravação neste dia.";
}
function reloadLiveStream() {
  const show = document.getElementById("chkOverlay").checked;
  const enh = document.getElementById("chkEnhance").checked ? "1" : "0";
  const f30 = document.getElementById("chkFps30").checked ? "1" : "0";
  if (!show) document.getElementById("overlayBox").innerHTML = "";
  document.getElementById("liveImg").src = withToken(`/api/stream/live.mjpg?overlay=0&enhance=${enh}&fps30=${f30}&_t=${Date.now()}`);
}

/* ---------------- Status ---------------- */
async function refreshStatus() {
  try {
    const data = await (await apiFetch("/api/status")).json();
    document.getElementById("camNameBadge").innerHTML = `${esc(data.camera.name)} <span id="camMeta">${data.camera.fps ? esc(data.camera.fps) + " fps" : ""}</span>`;
    document.getElementById("streamBadge").textContent = data.camera.connected ? `Ao vivo · ${data.camera.fps} FPS` : `Estado: ${data.camera.state}`;
    setDot("streamDot", data.camera.connected ? "ok" : "warn");
    document.getElementById("recBadge").textContent = data.recording.active ? `Gravando · ${fmtBytes(data.recording.size_bytes)}` : "Gravação parada";
    document.getElementById("recBadge").title = data.recording.current_file || "";
    setDot("recDot", data.recording.active ? "red" : "");
    document.getElementById("powerBadge").textContent = data.power_guard_active ? "24/7 ativo" : `SO: ${data.platform}`;
    setDot("powerDot", data.power_guard_active ? "ok" : "");
    document.getElementById("driveBadge").textContent = data.drive.enabled ? `Drive · ${data.drive.pending_count} na fila` : "Drive: desligado";
    setDot("driveDot", data.drive.enabled ? (data.drive.pending_count ? "warn" : "ok") : "");
    const s = data.ai.summary || {};
    const objs = (s.people_total || 0) + (s.cars_count || 0) + (s.motos_count || 0);
    document.getElementById("aiCounters").textContent = `◈ ${objs} objetos` + (s.last_plate ? ` · ${s.last_plate}` : "");
    document.getElementById("stPeople").textContent = s.people_total || 0;
    document.getElementById("stPeopleSub").textContent = `H ${s.men_count || 0} · M ${s.women_count || 0}`;
    document.getElementById("stVeh").textContent = (s.cars_count || 0) + (s.motos_count || 0);
    document.getElementById("stVehSub").textContent = `${s.cars_count || 0} carros · ${s.motos_count || 0} motos`;
    document.getElementById("ptzBackend").textContent = data.ptz.available ? `PTZ: ${data.ptz.backend || "Ativo"}` : "PTZ: indisponível";
    document.getElementById("logsBox").textContent = (data.logs || []).join(String.fromCharCode(10));
  } catch (e) {}
}

/* ---------------- Tabelas / Feed IA ---------------- */
async function refreshTables() {
  try {
    const [vr, pr] = await Promise.all([
      apiFetch("/api/vehicles?limit=150").then(r => r.json()),
      apiFetch("/api/people?limit=150").then(r => r.json())
    ]);
    VEHICLES = vr.records || [];
    PEOPLE = pr.records || [];
    renderFeed();
  } catch (e) {}
}
function setFeed(kind) {
  FEED = kind;
  const map = { all:"tabAll", vehicles:"tabVeh", people:"tabPpl", dvr:"tabDvr" };
  for (const k in map) document.getElementById(map[k]).classList.toggle("on", k === kind);
  document.getElementById("navAll").classList.toggle("on", kind === "all");
  document.getElementById("navVeh").classList.toggle("on", kind === "vehicles");
  document.getElementById("navPpl").classList.toggle("on", kind === "people");
  const titles = { all:"Eventos em tempo real", vehicles:"Veículos detectados", people:"Pessoas & vestuário", dvr:"Gravações do dia" };
  document.getElementById("feedTitle").textContent = titles[kind];
  renderFeed();
}
function vehRow(r) {
  const moto = /moto/i.test(r.vehicle_type || "");
  return `<tr><td class="t">${esc(timeOnly(r.timestamp))}</td><td><span class="ic">${moto ? "🏍" : "⛟"}</span></td>` +
    `<td>${r.plate ? `<span class="plate">${esc(r.plate)}</span>` : '<span class="pill">sem placa</span>'}</td>` +
    `<td>${swatches(r.color)}${r.approx_model ? " · " + esc(r.approx_model) : ""} ${moto ? '<span class="pill">Moto</span>' : ""}</td></tr>`;
}
function pplRow(r) {
  const fem = /mulher|fem/i.test(r.person_type || "");
  return `<tr><td class="t">${esc(timeOnly(r.timestamp))}</td><td><span class="ic">${fem ? "♀" : "♂"}</span></td>` +
    `<td>${esc(r.person_type)}</td><td>${swatches(r.clothing_colors)}</td></tr>`;
}
function renderFeed() {
  const head = document.getElementById("feedHead"), body = document.getElementById("feedBody");
  let rows = [];
  if (FEED === "dvr") {
    head.innerHTML = "<tr><th>Hora</th><th></th><th>Origem</th><th>Detalhes</th></tr>";
    const items = [];
    for (const seg of CURRENT_TIMELINE.segments || []) items.push({ sec: seg.start_sec, html:
      `<tr class="click" onclick="jumpToSec(${seg.start_sec})"><td class="t">${fmtSec(seg.start_sec)}</td><td><span class="ic">▶</span></td><td>${seg.is_motion_clip ? '<span class="pill o">Movimento</span>' : '<span class="pill">' + esc(seg.source_kind) + "</span>"}</td><td>${esc(seg.filename)} · ${Math.round(seg.duration_sec)}s</td></tr>` });
    for (const ev of CURRENT_TIMELINE.events || []) items.push({ sec: ev.sec_of_day, html:
      `<tr class="click" onclick="jumpToSec(${ev.sec_of_day})"><td class="t">${fmtSec(ev.sec_of_day)}</td><td><span class="ic">◈</span></td><td><span class="pill o">${esc(ev.event_label)}</span></td><td>${esc(ev.label)}</td></tr>` });
    items.sort((a, b) => b.sec - a.sec);
    rows = items.map(i => i.html);
  } else {
    head.innerHTML = "<tr><th>Hora</th><th></th><th>Detecção</th><th>Atributos</th></tr>";
    if (FEED === "vehicles") rows = VEHICLES.map(vehRow);
    else if (FEED === "people") rows = PEOPLE.map(pplRow);
    else {
      const mixed = VEHICLES.map(r => ({ ts: String(r.timestamp || ""), html: vehRow(r) }))
        .concat(PEOPLE.map(r => ({ ts: String(r.timestamp || ""), html: pplRow(r) })));
      mixed.sort((a, b) => b.ts.localeCompare(a.ts));
      rows = mixed.slice(0, 200).map(m => m.html);
    }
  }
  body.innerHTML = rows.length ? rows.join("") : '<tr><td colspan="4" class="empty">Nenhum registro ainda</td></tr>';
}

/* ---------------- Timeline 24h ---------------- */
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
function playheadSec() {
  if (MODE !== "live" && CURRENT_SEG) return CURRENT_SEG.start_sec + (document.getElementById("dvrVideo").currentTime || 0);
  const d = new Date();
  return d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds();
}
function drawTimeline() {
  const cv = document.getElementById("timelineCanvas");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#181a1f";
  roundRect(ctx, 0, 0, w, h - 16, 8); ctx.fill();
  const trackH = h - 16;

  ctx.fillStyle = "#5a5e66"; ctx.font = "10.5px Segoe UI, sans-serif"; ctx.textBaseline = "top";
  for (let hr = 0; hr <= 24; hr += 3) {
    const x = (hr / 24) * w;
    ctx.fillStyle = "rgba(255,255,255,.04)"; ctx.fillRect(x, 0, 1, trackH);
    ctx.fillStyle = "#5a5e66";
    const lbl = String(hr).padStart(2, "0") + "h";
    const tx = hr === 24 ? w - ctx.measureText(lbl).width : (hr === 0 ? 0 : x - ctx.measureText(lbl).width / 2);
    ctx.fillText(lbl, tx, trackH + 4);
  }
  for (const seg of CURRENT_TIMELINE.segments || []) {
    const x1 = (seg.start_sec / 86400) * w;
    const sw = Math.max(2, ((seg.end_sec - seg.start_sec) / 86400) * w);
    if (seg.is_motion_clip) { ctx.fillStyle = "rgba(255,122,26,.42)"; roundRect(ctx, x1, 9, sw, trackH - 18, 4); }
    else { ctx.fillStyle = seg.source_kind === "drive_cloud" ? "rgba(77,163,255,.55)" : "rgba(255,255,255,.14)"; roundRect(ctx, x1, trackH / 2 - 4, sw, 8, 4); }
    ctx.fill();
  }
  ctx.shadowColor = "rgba(255,122,26,.6)"; ctx.shadowBlur = 8;
  for (const ev of CURRENT_TIMELINE.events || []) {
    const x = (ev.sec_of_day / 86400) * w;
    ctx.fillStyle = ev.event_type === "motion" ? "rgba(255,122,26,.55)" : "#ff7a1a";
    roundRect(ctx, x - 1.5, 7, 3, trackH - 14, 2); ctx.fill();
  }
  ctx.shadowBlur = 0;
  const ph = (playheadSec() / 86400) * w;
  ctx.fillStyle = "#f2f3f5"; ctx.fillRect(ph - 1, 0, 2, trackH);
  const lbl = fmtSec(playheadSec());
  ctx.font = "10px Segoe UI, sans-serif";
  const lw = ctx.measureText(lbl).width + 10;
  const lx = Math.min(w - lw, ph + 5);
  ctx.fillStyle = "#1e2026"; roundRect(ctx, lx, 3, lw, 15, 4); ctx.fill();
  ctx.fillStyle = "#f2f3f5"; ctx.fillText(lbl, lx + 5, 5);
  document.getElementById("stEv").textContent = (CURRENT_TIMELINE.events || []).length;
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function playSegment(seg, seekOffsetSec = 0) {
  if (!seg) return;
  stopReverse();
  const v = document.getElementById("dvrVideo");
  CURRENT_SEG = seg;
  CURRENT_SEG_FPS = Math.max(1, seg.fps || 15);
  setMode("playback");
  const src = withToken(`/api/video?date=${encodeURIComponent(seg.date_str)}&file=${encodeURIComponent(seg.filename)}&drive_id=${encodeURIComponent(seg.drive_file_id || "")}`);
  document.getElementById("dvrStatus").textContent = `${seg.filename} · ${fmtSec(seg.start_sec)} · ${seg.source_kind}`;
  if (v.dataset.seg !== seg.filename) {
    v.dataset.seg = seg.filename;
    v.src = src;
    v.onloadedmetadata = () => {
      if (seekOffsetSec > 0 && seekOffsetSec < (v.duration || 1e9)) v.currentTime = seekOffsetSec;
      v.playbackRate = SPEED;
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
    if (FEED === "dvr") renderFeed();
  } catch (e) {}
}
function jumpToSec(daySec) {
  const segs = CURRENT_TIMELINE.segments || [];
  let chosen = segs.find(s => daySec >= s.start_sec && daySec <= s.end_sec);
  if (!chosen && segs.length) chosen = segs.reduce((a, b) => Math.abs(b.start_sec - daySec) < Math.abs(a.start_sec - daySec) ? b : a);
  if (chosen) playSegment(chosen, Math.max(0, daySec - chosen.start_sec));
}
document.getElementById("timelineCanvas").addEventListener("click", ev => {
  const rect = ev.target.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
  jumpToSec(ratio * 86400);
});

/* ---------------- Controles de reprodução ---------------- */
const SPEEDS = [2, 4, 8, 16];
function nextSpeed(cur) { const i = SPEEDS.indexOf(Math.abs(cur)); return SPEEDS[(i + 1) % SPEEDS.length]; }
function updateSpeedLbl() {
  const lbl = document.getElementById("speedLbl");
  lbl.textContent = (REVERSE_TIMER ? "−" : "") + SPEED + "×";
  document.getElementById("btnRew").classList.toggle("act", !!REVERSE_TIMER);
  document.getElementById("btnFf").classList.toggle("act", !REVERSE_TIMER && SPEED > 1);
}
function updatePlayBtn() {
  const v = document.getElementById("dvrVideo");
  const playing = MODE === "live" ? true : (!v.paused || !!REVERSE_TIMER);
  document.getElementById("btnPlay").textContent = playing ? "❚❚" : "▶";
}
function ensurePlayback() {
  if (MODE === "live") { goPlayback(); return !!CURRENT_SEG; }
  return true;
}
function setPlaybackSpeed(rate) {
  stopReverse();
  SPEED = rate;
  const v = document.getElementById("dvrVideo");
  v.playbackRate = rate;
  updateSpeedLbl();
}
function fastForward() {
  if (!ensurePlayback()) return;
  const next = REVERSE_TIMER ? 2 : (SPEED < 2 ? 2 : nextSpeed(SPEED));
  setPlaybackSpeed(next);
  document.getElementById("dvrVideo").play().catch(() => {});
  updatePlayBtn();
}
function rewind() {
  if (!ensurePlayback()) return;
  const v = document.getElementById("dvrVideo");
  const next = REVERSE_TIMER ? nextSpeed(SPEED) : 2;
  stopReverse();
  SPEED = next;
  v.pause();
  REVERSE_TIMER = setInterval(() => {
    const t = v.currentTime - SPEED * 0.1;
    if (t <= 0) { v.currentTime = 0; stopReverse(); updatePlayBtn(); return; }
    v.currentTime = t;
  }, 100);
  updateSpeedLbl(); updatePlayBtn();
}
function stopReverse() {
  if (REVERSE_TIMER) { clearInterval(REVERSE_TIMER); REVERSE_TIMER = null; }
  updateSpeedLbl();
}
function togglePlay() {
  const v = document.getElementById("dvrVideo");
  if (MODE === "live") {
    if (!ensurePlayback()) return;
    return;
  }
  if (REVERSE_TIMER) { stopReverse(); v.pause(); }
  else if (v.paused) { v.playbackRate = SPEED; v.play().catch(() => {}); }
  else v.pause();
  updatePlayBtn();
}
function stepFrame(delta) {
  if (!ensurePlayback()) return;
  stopReverse();
  const v = document.getElementById("dvrVideo");
  v.pause();
  v.currentTime = Math.max(0, v.currentTime + delta * (1.0 / CURRENT_SEG_FPS));
  updatePlayBtn();
}
const _v = document.getElementById("dvrVideo");
_v.addEventListener("play", updatePlayBtn);
_v.addEventListener("pause", updatePlayBtn);
_v.addEventListener("ended", () => {
  const segs = CURRENT_TIMELINE.segments || [];
  const i = segs.indexOf(CURRENT_SEG);
  if (i >= 0 && i + 1 < segs.length) playSegment(segs[i + 1], 0);
});
function openSnapshot() {
  if (MODE === "live") {
    const enh = document.getElementById("chkEnhance").checked ? "1" : "0";
    window.open(withToken(`/api/snapshot.jpg?enhance=${enh}&_t=${Date.now()}`), "_blank");
    return;
  }
  const v = document.getElementById("dvrVideo");
  if (!v.videoWidth) return;
  const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  const a = document.createElement("a"); a.href = c.toDataURL("image/jpeg", 0.92);
  a.download = `homecam_${fmtSec(playheadSec()).replace(/:/g, "-")}.jpg`; a.click();
}
function exportClip30() {
  const seg = CURRENT_SEG || ((CURRENT_TIMELINE.segments || [])[0] || null);
  if (!seg) { document.getElementById("dvrStatus").textContent = "Selecione um clipe na linha do tempo para exportar em 30 FPS."; return; }
  const enh = document.getElementById("chkEnhance").checked ? "1" : "0";
  document.getElementById("dvrStatus").textContent = `Renderizando ${seg.filename} em 30 FPS constantes…`;
  window.open(withToken(`/api/video/export?date=${encodeURIComponent(seg.date_str)}&file=${encodeURIComponent(seg.filename)}&drive_id=${encodeURIComponent(seg.drive_file_id || "")}&fps=30&enhance=${enh}`), "_blank");
}
function toggleFullscreen() {
  const el = document.getElementById("dvrViewport");
  if (document.fullscreenElement) document.exitFullscreen(); else el.requestFullscreen().catch(() => {});
}

/* ---------------- Zoom digital & Pan ---------------- */
function clampPan() {
  const vp = document.getElementById("dvrViewport");
  const mx = (vp.clientWidth * (ZOOM - 1)) / 2, my = (vp.clientHeight * (ZOOM - 1)) / 2;
  PAN_X = Math.max(-mx, Math.min(mx, PAN_X)); PAN_Y = Math.max(-my, Math.min(my, PAN_Y));
}
function applyZoomTransform() {
  clampPan();
  const t = `translate(${PAN_X}px, ${PAN_Y}px) scale(${ZOOM})`;
  document.getElementById("dvrVideo").style.transform = t;
  document.getElementById("liveImg").style.transform = t;
  document.getElementById("overlayLayer").style.transform = t;
  document.getElementById("zoomLbl").textContent = ZOOM.toFixed(1) + "×";
  const vp = document.getElementById("dvrViewport");
  vp.classList.toggle("zoomed", ZOOM > 1);
  const mini = document.getElementById("miniMap"), box = document.getElementById("miniBox");
  mini.style.display = ZOOM > 1 ? "block" : "none";
  if (ZOOM > 1) {
    const bw = 100 / ZOOM, bh = 100 / ZOOM;
    const cx = 50 - (PAN_X / (vp.clientWidth * ZOOM)) * 100, cy = 50 - (PAN_Y / (vp.clientHeight * ZOOM)) * 100;
    box.style.width = bw + "%"; box.style.height = bh + "%";
    box.style.left = (cx - bw / 2) + "%"; box.style.top = (cy - bh / 2) + "%";
  }
}
function adjustZoom(factor) { ZOOM = Math.max(1, Math.min(8, ZOOM * factor)); if (ZOOM <= 1.01) { ZOOM = 1; PAN_X = 0; PAN_Y = 0; } applyZoomTransform(); }
function resetZoom() { ZOOM = 1; PAN_X = 0; PAN_Y = 0; applyZoomTransform(); }
const vp = document.getElementById("dvrViewport");
vp.addEventListener("wheel", e => { e.preventDefault(); adjustZoom(e.deltaY < 0 ? 1.15 : 1 / 1.15); }, { passive: false });
vp.addEventListener("mousedown", e => { if (ZOOM > 1 && !e.target.closest(".ctrl,.zoom,.ov-tr")) { DRAGGING = true; vp.classList.add("dragging"); DRAG_X = e.clientX - PAN_X; DRAG_Y = e.clientY - PAN_Y; } });
window.addEventListener("mousemove", e => { if (DRAGGING) { PAN_X = e.clientX - DRAG_X; PAN_Y = e.clientY - DRAG_Y; applyZoomTransform(); } });
window.addEventListener("mouseup", () => { DRAGGING = false; vp.classList.remove("dragging"); });
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowRight") stepFrame(1);
  else if (e.key === "ArrowLeft") stepFrame(-1);
  else if (e.key === "+" || e.key === "=") adjustZoom(1.35);
  else if (e.key === "-") adjustZoom(1 / 1.35);
  else if (e.key === "0") resetZoom();
  else if (e.key === "l" || e.key === "L") goLive();
});

/* ---------------- PTZ / Drive ---------------- */
function toggleDrawer() {
  const d = document.getElementById("ptzDrawer");
  d.classList.toggle("open");
  document.getElementById("navPtz").classList.toggle("on", d.classList.contains("open"));
}
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

function tickClock() {
  const tag = document.getElementById("clockTag");
  if (MODE === "live") tag.textContent = new Date().toLocaleString("pt-BR");
  else tag.textContent = `${document.getElementById("dateSelect").value} · ${fmtSec(playheadSec())}`;
  drawTimeline();
}

let DET_BUSY = false;
function layoutOverlayBox(fw, fh) {
  const vp = document.getElementById("dvrViewport"), box = document.getElementById("overlayBox"), img = document.getElementById("liveImg");
  const iw = fw || img.naturalWidth || 1280, ih = fh || img.naturalHeight || 720;
  const vw = vp.clientWidth, vh = vp.clientHeight;
  if (!vw || !vh || !iw || !ih) return;
  const scale = Math.min(vw / iw, vh / ih);
  const rw = iw * scale, rh = ih * scale;
  box.style.left = ((vw - rw) / 2) + "px";
  box.style.top = ((vh - rh) / 2) + "px";
  box.style.width = rw + "px";
  box.style.height = rh + "px";
}
async function refreshDetections() {
  if (MODE !== "live" || !document.getElementById("chkOverlay").checked) {
    document.getElementById("overlayBox").innerHTML = "";
    return;
  }
  if (DET_BUSY) return;
  DET_BUSY = true;
  try {
    const data = await (await apiFetch("/api/detections")).json();
    layoutOverlayBox(data.frame_width, data.frame_height);
    const dets = data.detections || [];
    document.getElementById("overlayBox").innerHTML = dets.map(d => {
      const cls = d.category === "Pessoa" ? "p" : ((d.category === "Animal" || d.category === "Bicicleta") ? "a" : "");
      return `<div class="bbox ${cls}" style="left:${(d.x * 100).toFixed(2)}%;top:${(d.y * 100).toFixed(2)}%;width:${(d.w * 100).toFixed(2)}%;height:${(d.h * 100).toFixed(2)}%"><i>${esc(d.label)}</i></div>`;
    }).join("");
  } catch (e) {
  } finally { DET_BUSY = false; }
}

setMode("live");
setFeed("all");
reloadLiveStream();
refreshStatus();
refreshDetections();
refreshTables();
refreshDates(true);
setInterval(refreshStatus, 2500);
setInterval(refreshDetections, 350);
setInterval(refreshTables, 2500);
setInterval(tickClock, 500);
setInterval(() => { if (document.getElementById("dateSelect").value) loadTimeline(); }, 30000);
window.addEventListener("resize", () => { drawTimeline(); layoutOverlayBox(); });
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

            def _stream_file_with_range(self, file_path: Path, download_name: str = "") -> None:
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
                if download_name:
                    self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
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

            def _write_mjpeg_part(self, jpg: bytes) -> None:
                part_hdr = (
                    f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(part_hdr + jpg + b"\r\n")
                self.wfile.flush()

            def _stream_mjpeg(
                self,
                with_overlays: bool,
                enhance: bool = False,
                fps30: bool = False,
            ) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                last_seq = -1
                prev_frame = None
                interp = LiveFrameInterpolator(max_flow_width=360) if fps30 else None
                try:
                    while not stop_ev.is_set():
                        if interp is None:
                            seq, jpg = nvr.wait_next_jpeg(
                                last_seq, timeout=0.6, with_overlays=with_overlays, enhance=enhance
                            )
                            if jpg is None or seq == last_seq:
                                continue
                            last_seq = seq
                            self._write_mjpeg_part(jpg)
                            continue

                        seq, frame = nvr.wait_next_frame(last_seq, timeout=0.6)
                        if frame is None or seq == last_seq:
                            continue
                        last_seq = seq
                        cur = frame.copy()
                        if prev_frame is not None and prev_frame.shape == cur.shape:
                            subframes = 2 if nvr._current_fps < 16.0 else 1
                            mids = interp.interpolate_intermediate_frames(prev_frame, cur, subframes=subframes)
                            step_sleep = max(0.015, min(0.033, (1.0 / max(10.0, nvr._current_fps)) / (len(mids) + 1)))
                            for mid in mids:
                                mid_jpg = nvr.encode_frame_jpeg(mid, with_overlays=with_overlays, enhance=enhance)
                                if mid_jpg is not None:
                                    self._write_mjpeg_part(mid_jpg)
                                    time.sleep(step_sleep)
                        prev_frame = cur
                        jpg = nvr.encode_frame_jpeg(cur, with_overlays=with_overlays, enhance=enhance)
                        if jpg is not None:
                            self._write_mjpeg_part(jpg)
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
                elif route == "/api/detections":
                    self._send_json(HTTPStatus.OK, nvr.get_detections_payload())
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
                elif route == "/api/video/export":
                    date_str = (query.get("date") or [""])[0].strip()
                    filename = (query.get("file") or [""])[0].strip()
                    cam = (query.get("camera") or [""])[0] or None
                    drive_id = (query.get("drive_id") or [""])[0].strip()
                    fps = int((query.get("fps") or ["30"])[0] or 30)
                    enh = (query.get("enhance") or ["0"])[0] == "1"
                    try:
                        out_path = nvr.export_interpolated_video(
                            date_str,
                            filename,
                            camera_name=cam,
                            drive_file_id=drive_id,
                            target_fps=fps,
                            enhance=enh,
                        )
                        self._stream_file_with_range(out_path, download_name=out_path.name)
                    except ValueError as err:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                    except FileNotFoundError as err:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": str(err)})
                    except Exception as err:
                        self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(err)})
                elif route == "/api/stream/live.mjpg":
                    ov = (query.get("overlay") or ["1"])[0] != "0"
                    enh = (query.get("enhance") or ["0"])[0] == "1"
                    f30 = (query.get("fps30") or ["0"])[0] == "1"
                    self._stream_mjpeg(with_overlays=ov, enhance=enh, fps30=f30)
                elif route == "/api/snapshot.jpg":
                    ov = (query.get("overlay") or ["1"])[0] != "0"
                    enh = (query.get("enhance") or ["0"])[0] == "1"
                    jpg = nvr.get_latest_jpeg(with_overlays=ov, enhance=enh)
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

