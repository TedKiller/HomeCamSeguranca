"""Interface gráfica principal (Tkinter) do HomeCam Segurança."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageTk

from ai_analytics import (
    AISummary,
    CCTVAnalyticsPipeline,
    PersonRecord,
    VehicleRecord,
    draw_ai_overlays,
)
from camera_discovery import (
    BRAND_NAMES,
    DiscoveredCamera,
    detect_usb_webcams,
    discover_all_cameras,
    get_brand_defaults,
    get_local_ipv4,
    mask_rtsp_url,
    open_video_capture,
    query_onvif_stream_uris,
    resolve_best_connection_source,
    resolve_camera_source,
    test_camera_connection,
)
from config_manager import (
    DEFAULT_PENDING_PATH,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_SECRETS_DIR,
    AppConfig,
    CameraConfig,
    ConfigManager,
    is_directory_writable,
    normalize_drive_folder_dir,
    resolve_writable_recordings_dir,
    sanitize_filename_part,
)
from dvr_playback import (
    EVENT_TYPE_LABELS,
    SUPPORTED_SPEEDS,
    TIMELINE_COLORS,
    TIMELINE_WINDOW_PRESETS_SEC,
    DVREventStore,
    DVRPlaybackEngine,
    DVRTimelineIndex,
    DigitalZoomViewport,
    PlaybackFrameState,
    TimelineEvent,
    TimelineSegment,
    format_day_seconds,
)
from gdrive_uploader import (
    DRIVE_SETUP_GUIDE_PT,
    GoogleDriveUploader,
    detect_google_drive_desktop_dirs,
    validate_oauth_client_json,
)
from ptz_control import create_ptz_backend
from headless_nvr import record_ai_summary_events, recover_orphaned_recordings
from recorder import VideoRecorder, is_valid_mp4_file, save_snapshot
from xiongmai_netip import (
    XM_DEFAULT_PORT,
    XM_WATCH_PRESET_ID,
    DetectTrackSettings,
    PTZPresetItem,
    PTZStatusSnapshot,
    find_xiongmai_by_serial,
)

_SENSITIVITY_MAP = {
    "Alta (qualquer movimento)": 0.008,
    "Média (recomendado)": 0.015,
    "Baixa (movimentos grandes)": 0.032,
}


class HomeCamApp:
    """Aplicativo desktop para conexão fácil a câmeras de segurança e backup no Google Drive."""

    def __init__(
        self,
        root: tk.Tk,
        config_manager: ConfigManager | None = None,
        allow_auto_connect: bool = True,
        enable_startup_tasks: bool = True,
    ) -> None:
        self.root = root
        self.root.title("HomeCam Segurança — Câmeras IP & Google Drive")
        self.root.geometry("1160x760")
        self.root.minsize(920, 620)
        self.root.configure(bg="#181c24")

        self.config_manager = config_manager or ConfigManager()
        self.config: AppConfig = self.config_manager.load()

        self._ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_photo: ImageTk.PhotoImage | None = None
        self._frame_render_scheduled = False

        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._is_connected = False
        self._current_fps = 15.0
        self._record_request_id = 0
        self._closing = False

        self._ptz_backend: object | None = None
        self._ptz_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._ptz_worker_stop = threading.Event()
        self._ptz_session_token = 0
        self._ptz_controls_enabled = False
        self._ptz_fw_supported = False
        self._ptz_press_started_at = 0.0
        self._ptz_holding = False
        self._ptz_active_dir = ""
        self._ptz_stop_after_id: str | None = None
        self._ptz_needs_home_return = False
        self._ptz_return_deadline = 0.0
        self._ptz_presets: list[PTZPresetItem] = []
        self._ptz_buttons: list[tk.Button] = []
        self._ptz_fw_widgets: list[tk.Widget] = []
        self._ptz_worker_thread = threading.Thread(target=self._ptz_worker_loop, daemon=True)
        self._ptz_worker_thread.start()

        self._discovered: list[DiscoveredCamera] = []
        self._editing_cam_id: str = self.config_manager.get_selected_camera(self.config).id
        self._show_password_var = tk.BooleanVar(value=False)

        self.uploader = GoogleDriveUploader(
            pending_path=DEFAULT_PENDING_PATH,
            credentials_path=self.config.drive.credentials_path,
            token_path=self.config.drive.token_path,
            root_folder_name=self.config.drive.root_folder_name,
            delete_after_upload=self.config.drive.delete_after_upload,
            on_log=lambda msg: self.post_log(f"[Drive] {msg}"),
            on_queue_change=lambda count: self._ui_queue.put(("queue_count", count)),
            sync_mode=self.config.drive.sync_mode,
            local_drive_dir=self.config.drive.local_drive_dir,
        )
        self.recorder = VideoRecorder(
            recordings_root=self.config.recording.recordings_dir,
            on_segment_ready=self._on_recording_segment_ready,
            on_log=lambda msg: self.post_log(f"[Gravação] {msg}"),
        )
        self._ai_vehicle_snapshot_key: tuple[tuple[str, str, str, str, str], ...] = ()
        self._ai_people_snapshot_key: tuple[tuple[str, str, str], ...] = ()
        self._ai_preview_photo: ImageTk.PhotoImage | None = None
        self.ai_pipeline = CCTVAnalyticsPipeline(
            csv_path=self.config.ai.csv_log_path,
            people_csv_path=self.config.ai.people_csv_log_path,
            save_csv=self.config.ai.save_csv_log,
            analysis_fps=self.config.ai.analysis_fps,
            on_update=lambda s, r: self._ui_queue.put(("ai_update", (s, r, self.ai_pipeline.get_person_records()))),
            on_log=lambda msg: self.post_log(msg),
        )
        self.ai_pipeline.configure(
            analysis_fps=self.config.ai.analysis_fps,
            detect_people_gender=self.config.ai.detect_people_gender,
            detect_animals=self.config.ai.detect_animals,
            detect_vehicles_alpr=self.config.ai.detect_vehicles_alpr,
            save_csv=self.config.ai.save_csv_log,
            csv_path=self.config.ai.csv_log_path,
            people_csv_path=self.config.ai.people_csv_log_path,
        )

        self.dvr_events = DVREventStore(Path(self.config.recording.recordings_dir) / "_timeline_events")
        self._pb_segments: list[TimelineSegment] = []
        self._pb_events: list[TimelineEvent] = []
        self._pb_viewport = DigitalZoomViewport()
        self._pb_latest_frame: np.ndarray | None = None
        self._pb_latest_state: PlaybackFrameState | None = None
        self._pb_latest_photo: ImageTk.PhotoImage | None = None
        self._pb_playhead_sec: float = 0.0
        self._pb_win_start: float = 0.0
        self._pb_win_span: float = 86400.0
        self._pb_active_scale: str = "24h"
        self._pb_scrubbing: bool = False
        self._pb_last_drag_seek: float = 0.0
        self._pb_pan_anchor: tuple[int, int] | None = None
        self._pb_loaded_once: bool = False
        self._pb_loading_index: bool = False
        self._pb_render_scheduled: bool = False
        self.dvr_player = DVRPlaybackEngine(
            recordings_root=self.config.recording.recordings_dir,
            uploader=self.uploader,
            on_frame=lambda st: self._ui_queue.put(("pb_frame", st)),
            on_state=lambda msg: self._ui_queue.put(("pb_status", msg)),
        )

        self._configure_styles()
        self._build_layout()
        selected_cam = self.config_manager.get_selected_camera(self.config)
        self._load_camera_into_form(selected_cam)
        self._load_ptz_prefs_from_camera(selected_cam)
        self._refresh_camera_lists()
        self._refresh_drive_status_ui()
        self._refresh_queue_tree()
        self._refresh_ai_table(self.ai_pipeline.get_vehicle_records())
        self._refresh_people_table(self.ai_pipeline.get_person_records())
        self._update_ai_banner(self.ai_pipeline.get_summary())
        self._update_ai_button_state()
        self._sync_playback_camera_choices(selected_cam.name)

        self.uploader.start_worker()
        if enable_startup_tasks:
            if self.config.drive.sync_mode == "oauth" and self.uploader.has_saved_token():
                threading.Thread(
                    target=lambda: (
                        self.uploader.get_account_email(),
                        self._ui_queue.put(("drive_status_refresh", None)),
                    ),
                    daemon=True,
                ).start()
            threading.Thread(target=self._recover_orphaned_recordings, daemon=True).start()
            if self.config.ai.enabled:
                threading.Thread(target=self._init_ai_models_worker, daemon=True).start()
        self.post_log("Sistema iniciado. Configure ou selecione sua câmera para começar.")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_after_id: str | None = self.root.after(40, self._poll_ui_queue)
        self._ptz_idle_after_id: str | None = self.root.after(1000, self._ptz_idle_tick)
        self._update_live_recording_status()

        if allow_auto_connect and self.config.recording.auto_connect_on_startup:
            initial_src = resolve_camera_source(selected_cam)
            if not (isinstance(initial_src, str) and not initial_src.strip()):
                self.root.after(250, self.connect_active_camera)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#181c24")
        style.configure("Card.TFrame", background="#222834")
        style.configure("TLabel", background="#181c24", foreground="#eef2f7", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#222834", foreground="#eef2f7", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#222834", foreground="#9aa6b8", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#222834", foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background="#222834", foreground="#eef2f7", font=("Segoe UI", 10))
        style.configure("TNotebook", background="#181c24", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe", background="#222834", foreground="#eef2f7")
        style.configure("TLabelframe.Label", background="#222834", foreground="#93c5fd", font=("Segoe UI", 10, "bold"))
        style.configure(
            "Treeview",
            background="#11161f",
            foreground="#eef2f7",
            fieldbackground="#11161f",
            rowheight=24,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#2d3748",
            foreground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#2563eb")], foreground=[("selected", "#ffffff")])

    def post_log(self, message: str) -> None:
        """Agenda uma mensagem no painel de eventos de forma thread-safe."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._ui_queue.put(("log", f"[{ts}] {message}"))

    def _on_recording_segment_ready(self, file_path: Path, camera_name: str) -> None:
        if self.config.drive.enabled and self.config.drive.auto_upload:
            added = self.uploader.enqueue_file(file_path, camera_name=camera_name)
            if added:
                self.post_log(f"[Drive] Na fila para backup: {file_path.name}")
                self._ui_queue.put(("refresh_queue", None))

    def _build_layout(self) -> None:
        top = tk.Frame(self.root, bg="#222834", padx=12, pady=10)
        top.pack(fill=tk.X)

        tk.Label(
            top,
            text="Câmera Ativa:",
            bg="#222834",
            fg="#eef2f7",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.active_cam_var = tk.StringVar()
        self.active_cam_combo = ttk.Combobox(
            top,
            textvariable=self.active_cam_var,
            state="readonly",
            width=28,
        )
        self.active_cam_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.active_cam_combo.bind("<<ComboboxSelected>>", self._on_top_camera_selected)

        self.btn_connect = tk.Button(
            top,
            text="▶ Conectar",
            command=self.connect_active_camera,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_connect.pack(side=tk.LEFT, padx=4)

        self.btn_disconnect = tk.Button(
            top,
            text="■ Desconectar",
            command=self.disconnect_camera,
            bg="#374151",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
            state=tk.DISABLED,
        )
        self.btn_disconnect.pack(side=tk.LEFT, padx=4)

        self.btn_record = tk.Button(
            top,
            text="● Iniciar Gravação",
            command=self.toggle_recording,
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_record.pack(side=tk.LEFT, padx=(12, 4))

        self.btn_snapshot = tk.Button(
            top,
            text="📷 Tirar Foto",
            command=self.take_snapshot,
            bg="#0f766e",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=4)

        self.btn_ai_toggle = tk.Button(
            top,
            text="🤖 IA: Desligada",
            command=self.toggle_ai_pipeline,
            bg="#374151",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_ai_toggle.pack(side=tk.LEFT, padx=4)

        self.status_badge = tk.Label(
            top,
            text="Desconectado",
            bg="#374151",
            fg="#f9fafb",
            padx=10,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_badge.pack(side=tk.RIGHT, padx=(6, 0))

        self.drive_badge = tk.Label(
            top,
            text="Drive: Desativado",
            bg="#374151",
            fg="#e5e7eb",
            padx=10,
            pady=4,
            font=("Segoe UI", 9),
        )
        self.drive_badge.pack(side=tk.RIGHT, padx=4)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.tab_live = ttk.Frame(self.notebook)
        self.tab_playback = ttk.Frame(self.notebook)
        self.tab_cameras = ttk.Frame(self.notebook)
        self.tab_drive = ttk.Frame(self.notebook)
        self.tab_ai = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_live, text="  Ao Vivo & Gravação  ")
        self.notebook.add(self.tab_playback, text="  Reprodução DVR (Timeline)  ")
        self.notebook.add(self.tab_cameras, text="  Câmeras (Conexão Fácil)  ")
        self.notebook.add(self.tab_drive, text="  Google Drive (Backup)  ")
        self.notebook.add(self.tab_ai, text="  IA & Veículos (ALPR)  ")

        self._build_live_tab()
        self._build_playback_tab()
        self._build_cameras_tab()
        self._build_drive_tab()
        self._build_ai_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)
        self.root.bind("<space>", self._on_playback_key_space)
        self.root.bind("<Left>", self._on_playback_key_left)
        self.root.bind("<Right>", self._on_playback_key_right)

    # ------------------------------------------------------------------ #
    # Aba 1: Ao Vivo & Gravação                                          #
    # ------------------------------------------------------------------ #

    def _build_live_tab(self) -> None:
        container = tk.Frame(self.tab_live, bg="#181c24")
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        main_pane = tk.PanedWindow(container, orient=tk.VERTICAL, bg="#181c24", sashwidth=6)
        main_pane.pack(fill=tk.BOTH, expand=True)

        top_area = tk.Frame(main_pane, bg="#181c24")

        video_holder = tk.Frame(top_area, bg="#0d1117", highlightbackground="#2d3748", highlightthickness=1)
        self.video_label = tk.Label(
            video_holder,
            text=(
                "Nenhuma câmera conectada.\n\n"
                "Dica rápida: abra a aba 'Câmeras (Conexão Fácil)', clique em\n"
                "'Buscar Câmeras na Rede' ou escolha sua marca e clique em 'Conectar'."
            ),
            bg="#0d1117",
            fg="#9aa6b8",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        video_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_ptz_sidebar(top_area)
        main_pane.add(top_area, minsize=380, stretch="always")

        bottom_frame = tk.Frame(main_pane, bg="#222834", padx=10, pady=6)
        main_pane.add(bottom_frame, minsize=140, stretch="never")

        controls = tk.Frame(bottom_frame, bg="#222834")
        controls.pack(fill=tk.X, pady=(0, 8))

        tk.Label(controls, text="Duração dos clipes:", bg="#222834", fg="#eef2f7").pack(side=tk.LEFT)
        self.segment_min_var = tk.StringVar(value=str(self.config.recording.segment_minutes))
        seg_combo = ttk.Combobox(
            controls,
            textvariable=self.segment_min_var,
            values=("1", "2", "5", "10", "15"),
            width=4,
            state="readonly",
        )
        seg_combo.pack(side=tk.LEFT, padx=(4, 4))
        seg_combo.bind("<<ComboboxSelected>>", lambda _e: self._save_recording_prefs())
        tk.Label(controls, text="min", bg="#222834", fg="#9aa6b8").pack(side=tk.LEFT, padx=(0, 14))

        self.motion_only_var = tk.BooleanVar(value=self.config.recording.motion_only)
        ttk.Checkbutton(
            controls,
            text="Gravar apenas com movimento",
            variable=self.motion_only_var,
            command=self._save_recording_prefs,
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(controls, text="Sensibilidade:", bg="#222834", fg="#eef2f7").pack(side=tk.LEFT)
        self.sensitivity_var = tk.StringVar(
            value=self._sensitivity_label_from_float(self.config.recording.motion_sensitivity)
        )
        sens_combo = ttk.Combobox(
            controls,
            textvariable=self.sensitivity_var,
            values=tuple(_SENSITIVITY_MAP.keys()),
            width=22,
            state="readonly",
        )
        sens_combo.pack(side=tk.LEFT, padx=(4, 14))
        sens_combo.bind("<<ComboboxSelected>>", lambda _e: self._save_recording_prefs())

        self.auto_record_var = tk.BooleanVar(value=self.config.recording.auto_start_recording)
        ttk.Checkbutton(
            controls,
            text="Auto-gravar ao conectar",
            variable=self.auto_record_var,
            command=self._save_recording_prefs,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.auto_connect_var = tk.BooleanVar(value=self.config.recording.auto_connect_on_startup)
        ttk.Checkbutton(
            controls,
            text="Auto-conectar ao abrir",
            variable=self.auto_connect_var,
            command=self._save_recording_prefs,
        ).pack(side=tk.LEFT, padx=(0, 6))

        path_row = tk.Frame(bottom_frame, bg="#222834")
        path_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(path_row, text="Pasta local:", bg="#222834", fg="#eef2f7").pack(side=tk.LEFT)
        self.recordings_dir_var = tk.StringVar(value=self.config.recording.recordings_dir)
        ttk.Entry(path_row, textvariable=self.recordings_dir_var, width=56).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Alterar...", command=self._choose_recordings_dir).pack(side=tk.LEFT, padx=3)
        ttk.Button(path_row, text="Abrir Pasta", command=self._open_recordings_dir).pack(side=tk.LEFT, padx=3)
        tk.Button(
            path_row,
            text="🎞 Timeline DVR",
            command=self._open_playback_for_active_camera,
            bg="#1d4ed8",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=9,
            pady=3,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(6, 4))

        self.motion_indicator = tk.Label(
            path_row,
            text="Movimento: aguardando",
            bg="#2d3748",
            fg="#cbd5e1",
            padx=8,
            pady=3,
            font=("Segoe UI", 9),
        )
        self.motion_indicator.pack(side=tk.RIGHT, padx=(8, 0))

        self.ai_live_summary_var = tk.StringVar(
            value="IA: Desligada • Clique em '🤖 IA' ou na aba 'IA & Veículos (ALPR)'"
        )
        tk.Label(
            bottom_frame,
            textvariable=self.ai_live_summary_var,
            bg="#1a202c",
            fg="#a7f3d0",
            anchor=tk.W,
            padx=8,
            pady=4,
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, pady=(0, 4))

        self.rec_live_status_var = tk.StringVar(value="Gravação automática: parada • Drive: aguardando")
        tk.Label(
            bottom_frame,
            textvariable=self.rec_live_status_var,
            bg="#1a202c",
            fg="#93c5fd",
            anchor=tk.W,
            padx=8,
            pady=4,
            font=("Segoe UI", 9),
        ).pack(fill=tk.X, pady=(0, 6))

        log_frame = tk.Frame(bottom_frame, bg="#222834")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(
            log_frame,
            height=5,
            bg="#11161f",
            fg="#cbd5e1",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------ #
    # Aba 2 (DVR): Reprodução de Gravações, Timeline 24h & Zoom/Pan      #
    # ------------------------------------------------------------------ #

    def _build_playback_tab(self) -> None:
        outer = tk.Frame(self.tab_playback, bg="#181c24")
        outer.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Barra superior de filtros (câmera, data, fonte Drive e filtros de evento)
        top_bar = tk.Frame(outer, bg="#222834", padx=10, pady=7)
        top_bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(top_bar, text="Câmera:", bg="#222834", fg="#eef2f7", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.pb_cam_var = tk.StringVar()
        self.pb_cam_combo = ttk.Combobox(top_bar, textvariable=self.pb_cam_var, state="readonly", width=22)
        self.pb_cam_combo.pack(side=tk.LEFT, padx=(4, 10))
        self.pb_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_playback_timeline_clicked())

        tk.Label(top_bar, text="Data:", bg="#222834", fg="#eef2f7", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.pb_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.pb_date_combo = ttk.Combobox(top_bar, textvariable=self.pb_date_var, width=12)
        self.pb_date_combo.pack(side=tk.LEFT, padx=(4, 4))
        self.pb_date_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_playback_timeline_clicked(reload_dates=False))
        self.pb_date_combo.bind("<Return>", lambda _e: self._refresh_playback_timeline_clicked(reload_dates=False))
        ttk.Button(top_bar, text="Hoje", command=self._pb_select_today).pack(side=tk.LEFT, padx=(0, 8))

        self.pb_include_cloud_var = tk.BooleanVar(value=bool(self.config.drive.enabled))
        ttk.Checkbutton(
            top_bar,
            text="☁ Incluir Google Drive",
            variable=self.pb_include_cloud_var,
            command=self._refresh_playback_timeline_clicked,
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.btn_pb_refresh = ttk.Button(
            top_bar,
            text="🔄 Atualizar",
            command=self._refresh_playback_timeline_clicked,
        )
        self.btn_pb_refresh.pack(side=tk.LEFT, padx=(0, 12))

        self.pb_filter_motion_var = tk.BooleanVar(value=True)
        self.pb_filter_person_var = tk.BooleanVar(value=True)
        self.pb_filter_animal_var = tk.BooleanVar(value=True)
        self.pb_filter_vehicle_var = tk.BooleanVar(value=True)
        for lbl, var in (
            ("Movimento", self.pb_filter_motion_var),
            ("IA Pessoas", self.pb_filter_person_var),
            ("IA Animais", self.pb_filter_animal_var),
            ("IA Veículos/Placas", self.pb_filter_vehicle_var),
        ):
            ttk.Checkbutton(
                top_bar,
                text=lbl,
                variable=var,
                command=self._on_pb_event_filter_changed,
            ).pack(side=tk.LEFT, padx=4)

        # Área central: player com Zoom/Pan à esquerda + lista de eventos/clipes à direita
        mid_pane = tk.PanedWindow(outer, orient=tk.HORIZONTAL, bg="#181c24", sashwidth=6)
        mid_pane.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        player_box = tk.Frame(mid_pane, bg="#0d1117", highlightbackground="#2d3748", highlightthickness=1)
        self.pb_video_label = tk.Label(
            player_box,
            text=(
                "Nenhuma gravação carregada.\n\n"
                "Clique em '🔄 Atualizar' ou selecione uma data gravada na lista acima.\n"
                "Dica: Roda do mouse amplia (Zoom até 10x) • Arraste para mover (Pan)."
            ),
            bg="#090d14",
            fg="#9aa6b8",
            font=("Segoe UI", 11),
            justify=tk.CENTER,
            cursor="crosshair",
        )
        self.pb_video_label.pack(fill=tk.BOTH, expand=True)
        self.pb_video_label.bind("<MouseWheel>", self._pb_on_video_wheel)
        self.pb_video_label.bind("<Button-4>", lambda e: self._pb_zoom_step(1.25, e.x, e.y))
        self.pb_video_label.bind("<Button-5>", lambda e: self._pb_zoom_step(0.80, e.x, e.y))
        self.pb_video_label.bind("<ButtonPress-1>", self._pb_on_pan_start)
        self.pb_video_label.bind("<B1-Motion>", self._pb_on_pan_drag)
        self.pb_video_label.bind("<ButtonRelease-1>", self._pb_on_pan_end)
        self.pb_video_label.bind("<Double-Button-1>", lambda _e: self._pb_reset_zoom())
        self.pb_video_label.bind("<ButtonPress-3>", lambda _e: self._pb_reset_zoom())
        self.pb_video_label.bind("<Configure>", lambda _e: self._pb_render_viewport())

        hud_row = tk.Frame(player_box, bg="#141923", padx=8, pady=4)
        hud_row.pack(fill=tk.X, side=tk.BOTTOM)
        self.pb_hud_var = tk.StringVar(value="--:--:-- • PAUSADO • 1x • Zoom 1.0x")
        tk.Label(
            hud_row,
            textvariable=self.pb_hud_var,
            bg="#141923",
            fg="#67e8f9",
            font=("Consolas", 9, "bold"),
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.pb_zoom_var = tk.StringVar(value="1.0x")
        ttk.Button(hud_row, text="Reset 1x", command=self._pb_reset_zoom, width=8).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(hud_row, text="🔍 +", command=lambda: self._pb_adjust_zoom(1.35), width=5).pack(side=tk.RIGHT, padx=2)
        tk.Label(
            hud_row,
            textvariable=self.pb_zoom_var,
            bg="#1e293b",
            fg="#fef08a",
            padx=7,
            pady=2,
            font=("Consolas", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=2)
        ttk.Button(hud_row, text="🔍 -", command=lambda: self._pb_adjust_zoom(0.74), width=5).pack(side=tk.RIGHT, padx=2)

        mid_pane.add(player_box, minsize=520, stretch="always")

        side_panel = tk.Frame(mid_pane, bg="#222834", padx=8, pady=8, width=305)
        tk.Label(
            side_panel,
            text="Eventos & Clipes do Dia",
            bg="#222834",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        ev_wrap = tk.Frame(side_panel, bg="#222834")
        ev_wrap.pack(fill=tk.BOTH, expand=True)
        ev_cols = ("hora", "tipo", "detalhe")
        self.pb_events_tree = ttk.Treeview(ev_wrap, columns=ev_cols, show="headings", height=12)
        self.pb_events_tree.heading("hora", text="Hora")
        self.pb_events_tree.heading("tipo", text="Tipo")
        self.pb_events_tree.heading("detalhe", text="Descrição / Arquivo")
        self.pb_events_tree.column("hora", width=66, anchor=tk.CENTER)
        self.pb_events_tree.column("tipo", width=86, anchor=tk.CENTER)
        self.pb_events_tree.column("detalhe", width=155, anchor=tk.W)
        ev_scroll = ttk.Scrollbar(ev_wrap, orient=tk.VERTICAL, command=self.pb_events_tree.yview)
        self.pb_events_tree.configure(yscrollcommand=ev_scroll.set)
        self.pb_events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ev_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pb_events_tree.bind("<Double-1>", lambda _e: self._pb_on_event_activate())

        self._pb_tree_sec_map: dict[str, float] = {}
        side_btns = tk.Frame(side_panel, bg="#222834")
        side_btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(side_btns, text="▶ Reproduzir Selecionado", command=self._pb_on_event_activate).pack(
            fill=tk.X
        )

        mid_pane.add(side_panel, minsize=250, stretch="never")

        # Barra de transporte profissional DVR (Play/Pause, ±1x..±16x, ±1 Frame, Eventos, Foto)
        transport = tk.Frame(outer, bg="#222834", padx=10, pady=6)
        transport.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(transport, text="|◀ Evento", command=lambda: self._pb_jump_event(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(transport, text="⏪ Ré / Lento", command=lambda: self._pb_step_speed(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(transport, text="◀| -1 Quadro", command=lambda: self._pb_step_frame(-1)).pack(side=tk.LEFT, padx=2)

        self.btn_pb_play = tk.Button(
            transport,
            text="▶ Reproduzir",
            command=self._pb_toggle_play,
            bg="#059669",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_pb_play.pack(side=tk.LEFT, padx=5)

        ttk.Button(transport, text="+1 Quadro |▶", command=lambda: self._pb_step_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(transport, text="Avançar ⏩", command=lambda: self._pb_step_speed(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(transport, text="Evento ▶|", command=lambda: self._pb_jump_event(1)).pack(side=tk.LEFT, padx=2)

        tk.Label(transport, text="Velocidade:", bg="#222834", fg="#eef2f7", font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=(12, 4)
        )
        self.pb_speed_var = tk.StringVar(value="+1x")
        speed_values = tuple(self._pb_format_speed(s) for s in SUPPORTED_SPEEDS)
        self.pb_speed_combo = ttk.Combobox(
            transport,
            textvariable=self.pb_speed_var,
            values=speed_values,
            width=7,
            state="readonly",
        )
        self.pb_speed_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.pb_speed_combo.bind("<<ComboboxSelected>>", self._pb_speed_combo_changed)

        ttk.Button(transport, text="📷 Foto do Quadro", command=self._pb_snapshot_frame).pack(side=tk.LEFT, padx=3)
        ttk.Button(transport, text="📂 Abrir Clipe", command=self._pb_open_current_clip).pack(side=tk.LEFT, padx=3)

        self.pb_status_var = tk.StringVar(value="Selecione uma data ou clique em '🔄 Atualizar'")
        tk.Label(
            transport,
            textvariable=self.pb_status_var,
            bg="#222834",
            fg="#cbd5e1",
            font=("Segoe UI", 9),
            anchor=tk.E,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Painel da Linha do Tempo Contínua 24h
        tl_frame = tk.Frame(outer, bg="#141821", padx=8, pady=6, highlightbackground="#2d3748", highlightthickness=1)
        tl_frame.pack(fill=tk.X)

        tl_head = tk.Frame(tl_frame, bg="#141821")
        tl_head.pack(fill=tk.X, pady=(0, 4))

        tk.Label(tl_head, text="Escala Timeline:", bg="#141821", fg="#cbd5e1", font=("Segoe UI", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.pb_win_buttons: dict[str, tk.Button] = {}
        for scale_key in ("24h", "6h", "2h", "1h", "15m"):
            btn = tk.Button(
                tl_head,
                text=scale_key,
                command=lambda k=scale_key: self._pb_set_time_scale(k),
                bg="#2563eb" if scale_key == "24h" else "#2d3748",
                fg="#ffffff",
                relief=tk.FLAT,
                padx=8,
                pady=1,
                font=("Segoe UI", 8, "bold"),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.pb_win_buttons[scale_key] = btn

        ttk.Button(tl_head, text="◀", width=3, command=lambda: self._pb_scroll_window(-0.45)).pack(
            side=tk.LEFT, padx=(10, 2)
        )
        ttk.Button(tl_head, text="Agulha", width=7, command=self._pb_center_window_on_playhead).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(tl_head, text="▶", width=3, command=lambda: self._pb_scroll_window(0.45)).pack(
            side=tk.LEFT, padx=(2, 10)
        )

        # Legenda codificada por cores DVR
        for title, hex_color in (
            ("Contínua", TIMELINE_COLORS["continuous"]),
            ("Drive", TIMELINE_COLORS["drive_cloud"]),
            ("Mov.", TIMELINE_COLORS["motion"]),
            ("Pessoa", TIMELINE_COLORS["ai_person"]),
            ("Animal", TIMELINE_COLORS["ai_animal"]),
            ("Veíc./Placa", TIMELINE_COLORS["ai_vehicle"]),
        ):
            tk.Label(tl_head, text="■", bg="#141821", fg=hex_color, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            tk.Label(tl_head, text=title, bg="#141821", fg="#94a3b8", font=("Segoe UI", 8)).pack(
                side=tk.LEFT, padx=(1, 6)
            )

        self.pb_hover_var = tk.StringVar(value="Passe o mouse ou arraste na linha do tempo 24h")
        tk.Label(
            tl_head,
            textvariable=self.pb_hover_var,
            bg="#141821",
            fg="#fde047",
            font=("Consolas", 9),
            anchor=tk.E,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.pb_timeline_canvas = tk.Canvas(
            tl_frame,
            height=88,
            bg="#0b0f17",
            highlightthickness=0,
            cursor="sb_h_double_arrow",
        )
        self.pb_timeline_canvas.pack(fill=tk.X, expand=True)
        self.pb_timeline_canvas.bind("<Configure>", lambda _e: self._draw_playback_timeline())
        self.pb_timeline_canvas.bind("<ButtonPress-1>", self._pb_on_timeline_press)
        self.pb_timeline_canvas.bind("<B1-Motion>", self._pb_on_timeline_drag)
        self.pb_timeline_canvas.bind("<ButtonRelease-1>", self._pb_on_timeline_release)
        self.pb_timeline_canvas.bind("<Motion>", self._pb_on_timeline_hover)
        self.pb_timeline_canvas.bind("<MouseWheel>", self._pb_on_timeline_wheel)
        self.pb_timeline_canvas.bind("<Button-4>", lambda e: self._pb_timeline_wheel_step(1, e.x))
        self.pb_timeline_canvas.bind("<Button-5>", lambda e: self._pb_timeline_wheel_step(-1, e.x))

    @staticmethod
    def _pb_format_speed(speed: float) -> str:
        val = int(round(speed))
        return f"{val:+d}x"

    def _sync_playback_camera_choices(self, preferred_name: str | None = None) -> None:
        if not hasattr(self, "pb_cam_combo"):
            return
        names = [c.name for c in self.config.cameras] or ["Câmera Principal"]
        self.pb_cam_combo.configure(values=names)
        cur = self.pb_cam_var.get().strip()
        if preferred_name and preferred_name in names:
            self.pb_cam_var.set(preferred_name)
        elif cur not in names:
            self.pb_cam_var.set(names[0])

    def _on_notebook_tab_changed(self, _event: object = None) -> None:
        if self._closing:
            return
        try:
            current = self.notebook.select()
        except tk.TclError:
            return
        if current == str(self.tab_playback) and not self._pb_loaded_once:
            self._pb_loaded_once = True
            self._refresh_playback_timeline_clicked()

    def _open_playback_for_active_camera(self) -> None:
        selected = self.config_manager.get_selected_camera(self.config)
        self._sync_playback_camera_choices(selected.name)
        self._pb_loaded_once = True
        self.notebook.select(self.tab_playback)
        self._refresh_playback_timeline_clicked()

    def _pb_select_today(self) -> None:
        self.pb_date_var.set(datetime.now().strftime("%Y-%m-%d"))
        self._refresh_playback_timeline_clicked(reload_dates=False)

    def _refresh_playback_timeline_clicked(self, reload_dates: bool = True) -> None:
        if self._pb_loading_index:
            return
        cam_name = self.pb_cam_var.get().strip() or self.config_manager.get_selected_camera(self.config).name
        req_date = self.pb_date_var.get().strip() or datetime.now().strftime("%Y-%m-%d")
        include_cloud = bool(self.pb_include_cloud_var.get())
        rec_dir = self.config.recording.recordings_dir
        csv_path = self.config.ai.csv_log_path if self.config.ai.save_csv_log else None
        seg_sec = float(max(1, self.config.recording.segment_minutes) * 60)

        self._pb_loading_index = True
        self.btn_pb_refresh.configure(state=tk.DISABLED)
        self.pb_status_var.set("Indexando gravações e eventos...")

        def worker() -> None:
            try:
                dates = (
                    DVRTimelineIndex.discover_dates(
                        rec_dir,
                        cam_name,
                        uploader=self.uploader,
                        include_cloud=include_cloud,
                    )
                    if reload_dates
                    else []
                )
                chosen_date = req_date
                if reload_dates and dates and chosen_date not in dates:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    chosen_date = today_str if (today_str == chosen_date and not dates) else dates[0]

                segments, events = DVRTimelineIndex.build_day_timeline(
                    rec_dir,
                    cam_name,
                    chosen_date,
                    event_store=self.dvr_events,
                    csv_path=csv_path,
                    uploader=self.uploader,
                    include_cloud=include_cloud,
                    fallback_segment_sec=seg_sec,
                )
                self._ui_queue.put(("pb_timeline_loaded", (cam_name, chosen_date, dates, segments, events)))
            except Exception as err:
                self._ui_queue.put(("pb_timeline_error", str(err)))

        threading.Thread(target=worker, daemon=True).start()

    def _pb_apply_loaded_timeline(
        self,
        cam_name: str,
        chosen_date: str,
        dates: list[str],
        segments: list[TimelineSegment],
        events: list[TimelineEvent],
    ) -> None:
        self._pb_loading_index = False
        self.btn_pb_refresh.configure(state=tk.NORMAL)
        if dates:
            today = datetime.now().strftime("%Y-%m-%d")
            combo_vals = list(dates)
            if today not in combo_vals:
                combo_vals.insert(0, today)
            self.pb_date_combo.configure(values=combo_vals)
        self.pb_date_var.set(chosen_date)

        self._pb_segments = segments
        self._pb_events = events
        self._pb_viewport.reset()
        self.pb_zoom_var.set("1.0x")

        if segments:
            self._pb_playhead_sec = segments[0].start_sec
            self.pb_status_var.set(
                f"{len(segments)} bloco(s) e {len(events)} evento(s) em {chosen_date} ({cam_name})."
            )
        else:
            self._pb_playhead_sec = 0.0
            self.pb_status_var.set(f"Nenhuma gravação encontrada em {chosen_date} para '{cam_name}'.")

        self._pb_refresh_events_tree()
        self._draw_playback_timeline()
        self.dvr_player.load_timeline(segments, events, auto_seek_sec=self._pb_playhead_sec if segments else None)

    def _pb_allowed_event_types(self) -> set[str]:
        allowed: set[str] = set()
        if self.pb_filter_motion_var.get():
            allowed.add("motion")
        if self.pb_filter_person_var.get():
            allowed.add("ai_person")
        if self.pb_filter_animal_var.get():
            allowed.add("ai_animal")
        if self.pb_filter_vehicle_var.get():
            allowed.update(("ai_vehicle", "ai_plate"))
        return allowed

    def _on_pb_event_filter_changed(self) -> None:
        self._pb_refresh_events_tree()
        self._draw_playback_timeline()

    def _pb_refresh_events_tree(self) -> None:
        if not hasattr(self, "pb_events_tree"):
            return
        for row_id in self.pb_events_tree.get_children():
            self.pb_events_tree.delete(row_id)
        self._pb_tree_sec_map.clear()

        allowed = self._pb_allowed_event_types()
        rows: list[tuple[float, str, str]] = []
        for seg in self._pb_segments:
            tag = "[Drive]" if seg.source_kind == "drive_cloud" else "[Local]"
            if seg.source_kind == "drive_folder":
                tag = "[Drive Pasta]"
            dur_m, dur_s = divmod(int(round(seg.duration_sec)), 60)
            rows.append((seg.start_sec, f"Clipe {tag}", f"{seg.filename} ({dur_m:02d}:{dur_s:02d})"))

        for ev in self._pb_events:
            if ev.event_type in allowed:
                rows.append((ev.sec_of_day, EVENT_TYPE_LABELS.get(ev.event_type, ev.event_type), ev.label))

        rows.sort(key=lambda item: item[0])
        for sec, kind_lbl, detail in rows[:600]:
            iid = self.pb_events_tree.insert("", tk.END, values=(format_day_seconds(sec), kind_lbl, detail))
            self._pb_tree_sec_map[iid] = sec

    def _pb_on_event_activate(self) -> None:
        sel = self.pb_events_tree.selection()
        if not sel:
            return
        sec = self._pb_tree_sec_map.get(sel[0])
        if sec is None:
            return
        self._pb_playhead_sec = sec
        self._pb_center_window_on_playhead()
        self.dvr_player.seek_to_day_second(sec)
        self.dvr_player.play()

    def _pb_toggle_play(self) -> None:
        if not self._pb_segments:
            self._refresh_playback_timeline_clicked()
            return
        self.dvr_player.toggle_play_pause()

    def _pb_step_speed(self, direction: int) -> None:
        new_spd = self.dvr_player.step_speed(direction)
        self.pb_speed_var.set(self._pb_format_speed(new_spd))
        if not self.dvr_player.is_playing and self._pb_segments:
            self.dvr_player.play()

    def _pb_speed_combo_changed(self, _event: object = None) -> None:
        raw = self.pb_speed_var.get().strip().rstrip("x")
        try:
            spd = float(raw)
        except ValueError:
            spd = 1.0
        self.dvr_player.set_speed(spd)

    def _pb_step_frame(self, delta: int) -> None:
        if not self._pb_segments:
            return
        self.dvr_player.step_frame(delta)

    def _pb_jump_event(self, direction: int) -> None:
        self.dvr_player.jump_to_adjacent_event(direction, allowed_types=self._pb_allowed_event_types())

    def _pb_snapshot_frame(self) -> None:
        frame = self.dvr_player.get_current_frame()
        if frame is None:
            messagebox.showinfo("Foto do Quadro", "Posicione a linha do tempo sobre uma gravação primeiro.")
            return
        cam_name = self.pb_cam_var.get().strip() or "Playback"
        target_frame = frame
        if self._pb_viewport.zoom > 1.02:
            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = self._pb_viewport.get_source_roi(fw, fh)
            target_frame = frame[y1:y2, x1:x2]
        out_path = save_snapshot(target_frame, self.config.recording.recordings_dir, f"{cam_name}_DVR")
        self.pb_status_var.set(f"Foto salva: {out_path.name}")
        self.post_log(f"[DVR] Snapshot exportado de reprodução: {out_path}")

    def _pb_open_current_clip(self) -> None:
        st = self._pb_latest_state
        target: Path | None = None
        if st is not None:
            for seg in self._pb_segments:
                if seg.filename == st.filename and seg.local_path and seg.local_path.exists():
                    target = seg.local_path
                    break
        if target is None:
            target = Path(self.config.recording.recordings_dir)
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            webbrowser.open(target.as_uri())

    # ---- Zoom Digital & Panorâmica ---- #

    def _pb_on_video_wheel(self, event: object) -> None:
        delta = getattr(event, "delta", 0)
        factor = 1.25 if delta > 0 else 0.80
        self._pb_zoom_step(factor, getattr(event, "x", 0), getattr(event, "y", 0))

    def _pb_zoom_step(self, factor: float, cursor_x: float, cursor_y: float) -> None:
        frame = self._pb_latest_frame
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        dw = max(1, self.pb_video_label.winfo_width())
        dh = max(1, self.pb_video_label.winfo_height())
        self._pb_viewport.set_zoom_at_point(self._pb_viewport.zoom * factor, cursor_x, cursor_y, dw, dh, fw, fh)
        self.pb_zoom_var.set(f"{self._pb_viewport.zoom:.1f}x")
        self._pb_render_viewport()

    def _pb_adjust_zoom(self, factor: float) -> None:
        dw = max(1, self.pb_video_label.winfo_width())
        dh = max(1, self.pb_video_label.winfo_height())
        self._pb_zoom_step(factor, dw * 0.5, dh * 0.5)

    def _pb_reset_zoom(self) -> None:
        self._pb_viewport.reset()
        self.pb_zoom_var.set("1.0x")
        self._pb_render_viewport()

    def _pb_on_pan_start(self, event: object) -> None:
        if self._pb_viewport.zoom <= 1.01 or self._pb_latest_frame is None:
            self._pb_pan_anchor = None
            return
        self._pb_pan_anchor = (int(getattr(event, "x", 0)), int(getattr(event, "y", 0)))

    def _pb_on_pan_drag(self, event: object) -> None:
        if self._pb_pan_anchor is None or self._pb_latest_frame is None:
            return
        nx = int(getattr(event, "x", 0))
        ny = int(getattr(event, "y", 0))
        dx = nx - self._pb_pan_anchor[0]
        dy = ny - self._pb_pan_anchor[1]
        self._pb_pan_anchor = (nx, ny)
        fh, fw = self._pb_latest_frame.shape[:2]
        dw = max(1, self.pb_video_label.winfo_width())
        dh = max(1, self.pb_video_label.winfo_height())
        self._pb_viewport.pan_by_pixels(dx, dy, dw, dh, fw, fh)
        self._pb_render_viewport()

    def _pb_on_pan_end(self, _event: object = None) -> None:
        self._pb_pan_anchor = None

    def _pb_render_viewport(self) -> None:
        if not hasattr(self, "pb_video_label"):
            return
        frame = self._pb_latest_frame
        if frame is None:
            return
        tw = max(320, self.pb_video_label.winfo_width() - 6)
        th = max(200, self.pb_video_label.winfo_height() - 6)
        rendered = self._pb_viewport.render(frame, tw, th, draw_minimap=True)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._pb_latest_photo = photo
        self.pb_video_label.configure(
            image=photo,
            text="",
            cursor="fleur" if self._pb_viewport.zoom > 1.02 else "crosshair",
        )
        self._pb_refresh_hud_label()

    def _pb_refresh_hud_label(self) -> None:
        st = self._pb_latest_state
        if st is None or st.frame_bgr is None:
            self.pb_hud_var.set(f"Sem gravação carregada • Zoom {self._pb_viewport.zoom:.1f}x")
            self.btn_pb_play.configure(text="▶ Reproduzir", bg="#059669")
            return
        play_lbl = "REPRODUZINDO" if st.playing else "PAUSADO"
        spd_lbl = self._pb_format_speed(st.speed)
        src_tag = "Drive" if st.source_kind == "drive_cloud" else "Local"
        self.pb_hud_var.set(
            f"{st.date_str} {format_day_seconds(st.day_sec)} • {play_lbl} ({spd_lbl}) • "
            f"Quadro {st.frame_index + 1}/{st.total_frames} • {src_tag} • Zoom {self._pb_viewport.zoom:.1f}x"
        )
        self.pb_speed_var.set(spd_lbl)
        self.btn_pb_play.configure(
            text="⏸ Pausar" if st.playing else "▶ Reproduzir",
            bg="#d97706" if st.playing else "#059669",
        )

    # ---- Renderização e Scrubbing da Linha do Tempo 24h ---- #

    def _pb_set_time_scale(self, scale_key: str, anchor_sec: float | None = None) -> None:
        span = TIMELINE_WINDOW_PRESETS_SEC.get(scale_key, 86400.0)
        pivot = self._pb_playhead_sec if anchor_sec is None else anchor_sec
        self._pb_active_scale = scale_key
        self._pb_win_span = span
        self._pb_win_start = max(0.0, min(86400.0 - span, pivot - span * 0.5))
        for k, btn in self.pb_win_buttons.items():
            btn.configure(bg="#2563eb" if k == scale_key else "#2d3748")
        self._draw_playback_timeline()

    def _pb_scroll_window(self, fraction: float) -> None:
        if self._pb_win_span >= 86400.0:
            return
        self._pb_win_start = max(
            0.0,
            min(86400.0 - self._pb_win_span, self._pb_win_start + self._pb_win_span * fraction),
        )
        self._draw_playback_timeline()

    def _pb_center_window_on_playhead(self) -> None:
        self._pb_win_start = max(
            0.0,
            min(86400.0 - self._pb_win_span, self._pb_playhead_sec - self._pb_win_span * 0.5),
        )
        self._draw_playback_timeline()

    def _pb_sec_to_canvas_x(self, sec: float, width: int) -> float:
        return ((sec - self._pb_win_start) / max(1.0, self._pb_win_span)) * float(max(1, width))

    def _pb_canvas_x_to_sec(self, x: float, width: int) -> float:
        rel = max(0.0, min(1.0, float(x) / float(max(1, width))))
        return max(0.0, min(86399.0, self._pb_win_start + rel * self._pb_win_span))

    def _pb_segment_is_ready_at(self, sec: float) -> bool:
        for seg in self._pb_segments:
            if seg.start_sec <= sec <= seg.end_sec:
                return seg.local_path is not None and seg.local_path.exists()
        return False

    def _pb_timeline_wheel_step(self, direction: int, cursor_x: float) -> None:
        order = ["24h", "6h", "2h", "1h", "15m"]
        idx = order.index(self._pb_active_scale) if self._pb_active_scale in order else 0
        next_idx = max(0, min(len(order) - 1, idx + (1 if direction > 0 else -1)))
        if next_idx == idx:
            return
        w = max(1, self.pb_timeline_canvas.winfo_width())
        anchor = self._pb_canvas_x_to_sec(cursor_x, w)
        self._pb_set_time_scale(order[next_idx], anchor_sec=anchor)

    def _pb_on_timeline_wheel(self, event: object) -> None:
        delta = getattr(event, "delta", 0)
        self._pb_timeline_wheel_step(1 if delta > 0 else -1, getattr(event, "x", 0))

    def _pb_on_timeline_press(self, event: object) -> None:
        self._pb_scrubbing = True
        w = max(1, self.pb_timeline_canvas.winfo_width())
        sec = self._pb_canvas_x_to_sec(getattr(event, "x", 0), w)
        self._pb_playhead_sec = sec
        self._pb_last_drag_seek = time.monotonic()
        self._draw_playback_playhead()
        if self._pb_segment_is_ready_at(sec):
            self.dvr_player.seek_to_day_second(sec)

    def _pb_on_timeline_drag(self, event: object) -> None:
        if not self._pb_scrubbing:
            return
        w = max(1, self.pb_timeline_canvas.winfo_width())
        sec = self._pb_canvas_x_to_sec(getattr(event, "x", 0), w)
        self._pb_playhead_sec = sec
        self._draw_playback_playhead()
        self.pb_hover_var.set(f"Agulha: {format_day_seconds(sec)}")
        now = time.monotonic()
        if (now - self._pb_last_drag_seek) >= 0.09 and self._pb_segment_is_ready_at(sec):
            self._pb_last_drag_seek = now
            self.dvr_player.seek_to_day_second(sec)

    def _pb_on_timeline_release(self, event: object) -> None:
        if not self._pb_scrubbing:
            return
        self._pb_scrubbing = False
        w = max(1, self.pb_timeline_canvas.winfo_width())
        sec = self._pb_canvas_x_to_sec(getattr(event, "x", 0), w)
        self._pb_playhead_sec = sec
        self._draw_playback_playhead()
        if self._pb_segments:
            self.dvr_player.seek_to_day_second(sec)

    def _pb_on_timeline_hover(self, event: object) -> None:
        w = max(1, self.pb_timeline_canvas.winfo_width())
        sec = self._pb_canvas_x_to_sec(getattr(event, "x", 0), w)
        notes: list[str] = [f"Horário: {format_day_seconds(sec)}"]
        for seg in self._pb_segments:
            if seg.start_sec <= sec <= seg.end_sec:
                tag = "Drive Nuvem" if seg.source_kind == "drive_cloud" else "Gravado"
                notes.append(f"[{tag}] {seg.filename}")
                break
        allowed = self._pb_allowed_event_types()
        for ev in self._pb_events:
            if ev.event_type in allowed and (ev.sec_of_day - 8.0) <= sec <= (ev.end_sec + 8.0):
                notes.append(ev.label)
                break
        self.pb_hover_var.set("  •  ".join(notes))

    def _draw_playback_timeline(self) -> None:
        if not hasattr(self, "pb_timeline_canvas"):
            return
        cv = self.pb_timeline_canvas
        cv.delete("all")
        w = max(300, cv.winfo_width())
        h = max(84, cv.winfo_height())

        # Fundo das faixas: Régua (0..22), Gravações (24..54), Eventos (57..80)
        cv.create_rectangle(0, 0, w, 22, fill="#161c28", outline="")
        cv.create_rectangle(0, 24, w, 54, fill="#0f141e", outline="#1e293b")
        cv.create_rectangle(0, 56, w, h - 6, fill="#0d121b", outline="#1e293b")

        span = self._pb_win_span
        if span >= 43200:
            major_step, minor_step = 3600, 1800
        elif span >= 14400:
            major_step, minor_step = 1800, 600
        elif span >= 3600:
            major_step, minor_step = 600, 120
        else:
            major_step, minor_step = 120, 30

        win_end = self._pb_win_start + span
        first_minor = int(self._pb_win_start // minor_step) * minor_step
        t = first_minor
        while t <= win_end:
            if t >= self._pb_win_start:
                x = self._pb_sec_to_canvas_x(t, w)
                is_major = (t % major_step) == 0
                tick_y = 6 if is_major else 15
                cv.create_line(x, tick_y, x, 22, fill="#64748b" if is_major else "#334155", width=1)
                if is_major:
                    cv.create_line(x, 24, x, h - 6, fill="#172030", dash=(2, 4))
                    cv.create_text(
                        max(18, min(w - 18, x + 2)),
                        10,
                        text=format_day_seconds(t)[:5],
                        fill="#cbd5e1",
                        font=("Consolas", 8),
                    )
            t += minor_step

        # Pista principal de blocos gravados
        for seg in self._pb_segments:
            if seg.end_sec < self._pb_win_start or seg.start_sec > win_end:
                continue
            x1 = max(0.0, self._pb_sec_to_canvas_x(seg.start_sec, w))
            x2 = min(float(w), self._pb_sec_to_canvas_x(seg.end_sec, w))
            if x2 - x1 < 2.0:
                x2 = x1 + 2.0
            if seg.is_motion_clip:
                fill = TIMELINE_COLORS["motion"]
            elif seg.source_kind == "drive_cloud":
                fill = TIMELINE_COLORS["drive_cloud"]
            else:
                fill = TIMELINE_COLORS["continuous"]
            cv.create_rectangle(x1, 26, x2, 52, fill=fill, outline="#0b0f17", width=1)

        # Pista inferior de eventos coloridos (Movimento e alertas de IA)
        allowed = self._pb_allowed_event_types()
        for ev in self._pb_events:
            if ev.event_type not in allowed:
                continue
            if ev.end_sec < self._pb_win_start or ev.sec_of_day > win_end:
                continue
            ex1 = max(0.0, self._pb_sec_to_canvas_x(ev.sec_of_day, w))
            ex2 = min(float(w), self._pb_sec_to_canvas_x(ev.end_sec, w))
            if ex2 - ex1 < 3.0:
                ex2 = ex1 + 3.0
            cv.create_rectangle(ex1, 58, ex2, h - 8, fill=ev.color, outline="")

        self._draw_playback_playhead()

    def _draw_playback_playhead(self) -> None:
        if not hasattr(self, "pb_timeline_canvas"):
            return
        cv = self.pb_timeline_canvas
        cv.delete("playhead")
        w = max(300, cv.winfo_width())
        h = max(84, cv.winfo_height())
        if not (self._pb_win_start <= self._pb_playhead_sec <= self._pb_win_start + self._pb_win_span):
            return
        px = self._pb_sec_to_canvas_x(self._pb_playhead_sec, w)
        cv.create_line(px, 0, px, h, fill="#ff3b30", width=2, tags="playhead")
        cv.create_polygon(px - 6, 0, px + 6, 0, px, 9, fill="#ff3b30", outline="", tags="playhead")
        ts_txt = format_day_seconds(self._pb_playhead_sec)
        bx = max(32.0, min(float(w - 32), px))
        cv.create_rectangle(bx - 29, 10, bx + 29, 23, fill="#b91c1c", outline="#fecaca", tags="playhead")
        cv.create_text(bx, 16, text=ts_txt, fill="#ffffff", font=("Consolas", 8, "bold"), tags="playhead")

    def _on_playback_key_space(self, _event: object = None) -> None:
        if not self._pb_keyboard_shortcuts_enabled():
            return
        self._pb_toggle_play()

    def _on_playback_key_left(self, _event: object = None) -> None:
        if not self._pb_keyboard_shortcuts_enabled():
            return
        self._pb_step_frame(-1)

    def _on_playback_key_right(self, _event: object = None) -> None:
        if not self._pb_keyboard_shortcuts_enabled():
            return
        self._pb_step_frame(1)

    def _pb_keyboard_shortcuts_enabled(self) -> bool:
        if self._closing:
            return False
        try:
            if self.notebook.select() != str(self.tab_playback):
                return False
            focused = self.root.focus_get()
            if focused is not None and focused.winfo_class() in ("Entry", "TEntry", "TCombobox", "Text"):
                return False
        except tk.TclError:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Aba 2: Câmeras (Conexão Fácil)                                     #
    # ------------------------------------------------------------------ #

    def _build_cameras_tab(self) -> None:
        wrapper = tk.Frame(self.tab_cameras, bg="#181c24")
        wrapper.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = tk.Frame(wrapper, bg="#222834", padx=10, pady=10, width=410)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))

        tk.Label(left, text="Câmeras Salvas", bg="#222834", fg="#ffffff", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 4)
        )
        self.saved_cams_listbox = tk.Listbox(
            left,
            height=5,
            bg="#11161f",
            fg="#f1f5f9",
            selectbackground="#2563eb",
            relief=tk.FLAT,
            font=("Segoe UI", 10),
        )
        self.saved_cams_listbox.pack(fill=tk.X, pady=(0, 6))
        self.saved_cams_listbox.bind("<<ListboxSelect>>", self._on_saved_listbox_select)

        cam_btn_row = tk.Frame(left, bg="#222834")
        cam_btn_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(cam_btn_row, text="+ Nova Câmera", command=self._new_camera_profile).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(cam_btn_row, text="Excluir Selecionada", command=self._delete_selected_camera).pack(side=tk.LEFT)

        tk.Label(
            left,
            text="Descoberta Automática na Rede",
            bg="#222834",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=tk.W, pady=(4, 4))

        disc_btn_row = tk.Frame(left, bg="#222834")
        disc_btn_row.pack(fill=tk.X, pady=(0, 6))
        self.btn_scan_net = tk.Button(
            disc_btn_row,
            text="🔍 Buscar Câmeras na Rede",
            command=self._start_network_discovery,
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_scan_net.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(disc_btn_row, text="Detectar USB", command=self._start_usb_discovery).pack(side=tk.LEFT)

        self.discovery_status_var = tk.StringVar(value="Clique em buscar para localizar câmeras Wi-Fi/cabo na casa.")
        tk.Label(
            left,
            textvariable=self.discovery_status_var,
            bg="#222834",
            fg="#9aa6b8",
            wraplength=380,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(0, 6))

        self.discovered_listbox = tk.Listbox(
            left,
            height=9,
            bg="#11161f",
            fg="#f1f5f9",
            selectbackground="#2563eb",
            relief=tk.FLAT,
            font=("Consolas", 9),
        )
        self.discovered_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.discovered_listbox.bind("<Double-Button-1>", lambda _e: self._apply_selected_discovered())

        ttk.Button(
            left,
            text="Usar Câmera Encontrada →",
            command=self._apply_selected_discovered,
        ).pack(anchor=tk.E)

        # Formulário da direita
        right = tk.Frame(wrapper, bg="#222834", padx=14, pady=12)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right,
            text="Configuração Rápida da Câmera",
            bg="#222834",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 12))

        self.form_name_var = tk.StringVar()
        self.form_brand_var = tk.StringVar()
        self.form_ip_var = tk.StringVar()
        self.form_rtsp_port_var = tk.StringVar()
        self.form_onvif_port_var = tk.StringVar()
        self.form_user_var = tk.StringVar()
        self.form_pass_var = tk.StringVar()
        self.form_channel_var = tk.StringVar()
        self.form_stream_var = tk.StringVar()
        self.form_usb_var = tk.StringVar()
        self.form_serial_var = tk.StringVar()
        self.form_custom_url_var = tk.StringVar()
        self.form_preview_url_var = tk.StringVar()

        fields = [
            ("Nome da Câmera:", self.form_name_var),
        ]
        row_idx = 1
        for label_text, var in fields:
            tk.Label(right, text=label_text, bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
            ttk.Entry(right, textvariable=var, width=36).grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
            row_idx += 1

        tk.Label(right, text="Marca / Tipo:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        self.brand_combo = ttk.Combobox(
            right,
            textvariable=self.form_brand_var,
            values=BRAND_NAMES,
            state="readonly",
            width=34,
        )
        self.brand_combo.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
        self.brand_combo.bind("<<ComboboxSelected>>", self._on_brand_changed)
        row_idx += 1

        tk.Label(right, text="Endereço IP:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        ttk.Entry(right, textvariable=self.form_ip_var, width=20).grid(row=row_idx, column=1, sticky=tk.EW, pady=5)
        tk.Label(right, text="Porta RTSP:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=2, sticky=tk.E, padx=(10, 4))
        ttk.Entry(right, textvariable=self.form_rtsp_port_var, width=8).grid(row=row_idx, column=3, sticky=tk.W, pady=5)
        row_idx += 1

        tk.Label(right, text="Usuário:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        ttk.Entry(right, textvariable=self.form_user_var, width=20).grid(row=row_idx, column=1, sticky=tk.EW, pady=5)
        tk.Label(right, text="Porta ONVIF:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=2, sticky=tk.E, padx=(10, 4))
        ttk.Entry(right, textvariable=self.form_onvif_port_var, width=8).grid(row=row_idx, column=3, sticky=tk.W, pady=5)
        row_idx += 1

        tk.Label(right, text="Senha da Câmera:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        self.pass_entry = ttk.Entry(right, textvariable=self.form_pass_var, show="*", width=20)
        self.pass_entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=5)
        ttk.Checkbutton(
            right,
            text="Mostrar senha",
            variable=self._show_password_var,
            command=lambda: self.pass_entry.configure(show="" if self._show_password_var.get() else "*"),
        ).grid(row=row_idx, column=2, columnspan=2, sticky=tk.W, padx=(10, 0))
        row_idx += 1

        tk.Label(right, text="Canal / Stream:", bg="#222834", fg="#eef2f7").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        ch_frame = tk.Frame(right, bg="#222834")
        ch_frame.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
        ttk.Entry(ch_frame, textvariable=self.form_channel_var, width=5).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Combobox(
            ch_frame,
            textvariable=self.form_stream_var,
            values=("main", "sub"),
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(ch_frame, text="Índice USB (se Webcam):", bg="#222834", fg="#9aa6b8").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(ch_frame, textvariable=self.form_usb_var, width=5).pack(side=tk.LEFT)
        row_idx += 1

        tk.Label(right, text="N.º de Série (iCSee / XM):", bg="#222834", fg="#eef2f7").grid(
            row=row_idx, column=0, sticky=tk.W, pady=5
        )
        serial_frame = tk.Frame(right, bg="#222834")
        serial_frame.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
        ttk.Entry(serial_frame, textvariable=self.form_serial_var, width=22).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.btn_serial_lookup = ttk.Button(
            serial_frame,
            text="Buscar IP por Serial",
            command=self._lookup_by_serial_clicked,
        )
        self.btn_serial_lookup.pack(side=tk.LEFT, padx=(6, 0))
        row_idx += 1

        tk.Label(
            right,
            text="URL RTSP manual / ONVIF:",
            bg="#222834",
            fg="#eef2f7",
        ).grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        ttk.Entry(right, textvariable=self.form_custom_url_var, width=42).grid(
            row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5
        )
        row_idx += 1

        tk.Label(right, text="Prévia da conexão:", bg="#222834", fg="#9aa6b8").grid(
            row=row_idx, column=0, sticky=tk.W, pady=(6, 2)
        )
        tk.Label(
            right,
            textvariable=self.form_preview_url_var,
            bg="#11161f",
            fg="#93c5fd",
            padx=8,
            pady=6,
            anchor=tk.W,
            wraplength=460,
            justify=tk.LEFT,
            font=("Consolas", 9),
        ).grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=(6, 2))
        row_idx += 1

        for var in (
            self.form_brand_var,
            self.form_ip_var,
            self.form_rtsp_port_var,
            self.form_user_var,
            self.form_pass_var,
            self.form_channel_var,
            self.form_stream_var,
            self.form_usb_var,
            self.form_custom_url_var,
        ):
            var.trace_add("write", lambda *_a: self._update_rtsp_preview_label())

        action_bar = tk.Frame(right, bg="#222834")
        action_bar.grid(row=row_idx, column=0, columnspan=4, sticky=tk.EW, pady=(16, 6))

        ttk.Button(
            action_bar,
            text="Descobrir URL via ONVIF",
            command=self._query_onvif_clicked,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_test_conn = ttk.Button(
            action_bar,
            text="Testar Conexão",
            command=self._test_camera_clicked,
        )
        self.btn_test_conn.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            action_bar,
            text="Salvar Câmera",
            command=lambda: self._save_form_camera(connect_after=False),
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            action_bar,
            text="Salvar e Conectar Agora",
            command=lambda: self._save_form_camera(connect_after=True),
            bg="#16a34a",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT)
        row_idx += 1

        self.camera_test_msg_var = tk.StringVar(
            value="Dica: se não souber o caminho RTSP exato da sua câmera, use 'Descobrir URL via ONVIF'."
        )
        tk.Label(
            right,
            textvariable=self.camera_test_msg_var,
            bg="#222834",
            fg="#cbd5e1",
            wraplength=540,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).grid(row=row_idx, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        right.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ #
    # Aba 3: Google Drive (Backup)                                       #
    # ------------------------------------------------------------------ #

    def _build_drive_tab(self) -> None:
        wrapper = tk.Frame(self.tab_drive, bg="#181c24")
        wrapper.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        card = tk.Frame(wrapper, bg="#222834", padx=14, pady=12)
        card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            card,
            text="Backup Automático no Google Drive",
            bg="#222834",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        self.drive_account_var = tk.StringVar()
        tk.Label(card, textvariable=self.drive_account_var, bg="#222834", fg="#93c5fd", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 8)
        )

        tk.Label(card, text="Modo de envio:", bg="#222834", fg="#eef2f7").grid(row=2, column=0, sticky=tk.W, pady=4)
        mode_row = tk.Frame(card, bg="#222834")
        mode_row.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=6, pady=4)
        self.drive_sync_mode_var = tk.StringVar(
            value=self.config.drive.sync_mode if self.config.drive.sync_mode in ("oauth", "folder") else "oauth"
        )
        tk.Radiobutton(
            mode_row,
            text="API Direta Google Drive (credentials.json)",
            variable=self.drive_sync_mode_var,
            value="oauth",
            command=self._on_drive_mode_changed,
            bg="#222834",
            fg="#eef2f7",
            selectcolor="#11161f",
            activebackground="#222834",
            activeforeground="#ffffff",
        ).pack(side=tk.LEFT, padx=(0, 14))
        tk.Radiobutton(
            mode_row,
            text="Pasta Google Drive para Computador (Sem JSON)",
            variable=self.drive_sync_mode_var,
            value="folder",
            command=self._on_drive_mode_changed,
            bg="#222834",
            fg="#93c5fd",
            selectcolor="#11161f",
            activebackground="#222834",
            activeforeground="#ffffff",
        ).pack(side=tk.LEFT)

        tk.Label(card, text="Pasta do Drive Desktop:", bg="#222834", fg="#eef2f7").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.drive_folder_var = tk.StringVar(value=self.config.drive.local_drive_dir)
        ttk.Entry(card, textvariable=self.drive_folder_var, width=54).grid(row=3, column=1, sticky=tk.EW, padx=6, pady=4)
        folder_btns = tk.Frame(card, bg="#222834")
        folder_btns.grid(row=3, column=2, sticky=tk.W, pady=4)
        ttk.Button(folder_btns, text="Detectar Meu Drive", command=self._detect_drive_folder_clicked).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(folder_btns, text="Escolher Pasta...", command=self._choose_drive_folder_clicked).pack(side=tk.LEFT)

        tk.Label(card, text="Arquivo credentials.json:", bg="#222834", fg="#eef2f7").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.drive_creds_var = tk.StringVar(value=self.config.drive.credentials_path)
        ttk.Entry(card, textvariable=self.drive_creds_var, width=54).grid(row=4, column=1, sticky=tk.EW, padx=6, pady=4)
        ttk.Button(card, text="Selecionar JSON...", command=self._choose_credentials_json).grid(row=4, column=2, sticky=tk.W, pady=4)

        tk.Label(card, text="Pasta raiz no Drive:", bg="#222834", fg="#eef2f7").grid(row=5, column=0, sticky=tk.W, pady=4)
        self.drive_root_var = tk.StringVar(value=self.config.drive.root_folder_name)
        ttk.Entry(card, textvariable=self.drive_root_var, width=32).grid(row=5, column=1, sticky=tk.W, padx=6, pady=4)
        tk.Label(
            card,
            text="Organização: <Pasta>/<Câmera>/<YYYY-MM-DD>/video.mp4",
            bg="#222834",
            fg="#9aa6b8",
            font=("Segoe UI", 9),
        ).grid(row=5, column=2, sticky=tk.W)

        opts_row = tk.Frame(card, bg="#222834")
        opts_row.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=8)

        self.drive_enabled_var = tk.BooleanVar(value=self.config.drive.enabled)
        ttk.Checkbutton(
            opts_row,
            text="Ativar envio automático para o Google Drive",
            variable=self.drive_enabled_var,
            command=self._save_drive_settings,
        ).pack(side=tk.LEFT, padx=(0, 14))

        self.drive_delete_local_var = tk.BooleanVar(value=self.config.drive.delete_after_upload)
        ttk.Checkbutton(
            opts_row,
            text="Apagar vídeo local após confirmar envio",
            variable=self.drive_delete_local_var,
            command=self._save_drive_settings,
        ).pack(side=tk.LEFT)

        btn_row = tk.Frame(card, bg="#222834")
        btn_row.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))

        self.btn_drive_auth = tk.Button(
            btn_row,
            text="Conectar Conta Google Drive",
            command=self._connect_google_drive,
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        self.btn_drive_auth.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_row, text="Desconectar Conta", command=self._disconnect_google_drive).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Salvar Preferências", command=self._save_drive_settings).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Assistente / Como configurar?", command=self._open_drive_setup_wizard).pack(side=tk.RIGHT)

        card.columnconfigure(1, weight=1)

        # Painel Fila de Uploads
        queue_card = tk.Frame(wrapper, bg="#222834", padx=14, pady=10)
        queue_card.pack(fill=tk.BOTH, expand=True)

        q_head = tk.Frame(queue_card, bg="#222834")
        q_head.pack(fill=tk.X, pady=(0, 6))
        self.queue_title_var = tk.StringVar(value="Fila de Envios Pendentes (0)")
        tk.Label(q_head, textvariable=self.queue_title_var, bg="#222834", fg="#ffffff", font=("Segoe UI", 11, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Button(q_head, text="Enviar Pendentes Agora", command=self._sync_pending_now).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(q_head, text="Adicionar Gravações Locais à Fila", command=self._enqueue_local_archive).pack(side=tk.RIGHT)

        cols = ("arquivo", "camera", "data", "tentativas", "erro")
        self.queue_tree = ttk.Treeview(queue_card, columns=cols, show="headings", height=8)
        self.queue_tree.heading("arquivo", text="Arquivo")
        self.queue_tree.heading("camera", text="Câmera")
        self.queue_tree.heading("data", text="Pasta / Data")
        self.queue_tree.heading("tentativas", text="Tentativas")
        self.queue_tree.heading("erro", text="Último Aviso / Erro")
        self.queue_tree.column("arquivo", width=260)
        self.queue_tree.column("camera", width=140)
        self.queue_tree.column("data", width=110)
        self.queue_tree.column("tentativas", width=85, anchor=tk.CENTER)
        self.queue_tree.column("erro", width=320)
        self.queue_tree.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ #
    # Aba 4: IA em Tempo Real & Reconhecimento Veicular (ALPR)           #
    # ------------------------------------------------------------------ #

    def _build_ai_tab(self) -> None:
        outer = ttk.Frame(self.tab_ai, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        cfg_card = ttk.LabelFrame(
            outer,
            text="Motor de IA em Tempo Real (YOLOv10 + FairFace + Vestuário + Fast-ALPR + MobileCLIP-S2)",
            padding=8,
        )
        cfg_card.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(cfg_card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=(0, 6))
        self.ai_status_var = tk.StringVar(value="Modelos aguardando inicialização")
        ttk.Label(row1, text="Status:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(row1, textvariable=self.ai_status_var, style="Title.TLabel").pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(row1, text="🖼 Testar IA em Imagem...", command=self._test_ai_image_clicked).pack(
            side=tk.RIGHT, padx=(6, 0)
        )
        self.btn_init_ai = ttk.Button(
            row1,
            text="Baixar / Inicializar Modelos IA",
            command=self._init_ai_models_clicked,
        )
        self.btn_init_ai.pack(side=tk.RIGHT)

        row2 = ttk.Frame(cfg_card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=2)
        self.ai_enabled_var = tk.BooleanVar(value=self.config.ai.enabled)
        self.ai_overlay_var = tk.BooleanVar(value=self.config.ai.show_overlays)
        self.ai_gender_var = tk.BooleanVar(value=self.config.ai.detect_people_gender)
        self.ai_animals_var = tk.BooleanVar(value=self.config.ai.detect_animals)
        self.ai_vehicles_var = tk.BooleanVar(value=self.config.ai.detect_vehicles_alpr)
        self.ai_csv_var = tk.BooleanVar(value=self.config.ai.save_csv_log)

        for label, var in (
            ("Ativar IA em tempo real", self.ai_enabled_var),
            ("Exibir caixas no vídeo", self.ai_overlay_var),
            ("Pessoas (Homem/Mulher + Roupas)", self.ai_gender_var),
            ("Animais", self.ai_animals_var),
            ("Veículos + Placa (ALPR)/Cor/Modelo", self.ai_vehicles_var),
            ("Salvar CSV automático", self.ai_csv_var),
        ):
            ttk.Checkbutton(
                row2,
                text=label,
                variable=var,
                command=self._on_ai_prefs_changed,
            ).pack(side=tk.LEFT, padx=(0, 10))

        self.ai_tab_counters_var = tk.StringVar(
            value="Pessoas: 0 (Homem: 0 | Mulher: 0)  •  Animais: 0  •  Carros: 0  •  Motos: 0  •  Bicicletas: 0"
        )
        tk.Label(
            cfg_card,
            textvariable=self.ai_tab_counters_var,
            bg="#141821",
            fg="#a7f3d0",
            anchor=tk.W,
            padx=10,
            pady=5,
            font=("Consolas", 10, "bold"),
        ).pack(fill=tk.X, pady=(6, 0))

        tables_pane = tk.PanedWindow(outer, orient=tk.VERTICAL, bg="#181c24", sashwidth=6)
        tables_pane.pack(fill=tk.BOTH, expand=True)

        # --- Tabela 1: Veículos (Carros e Motos + ALPR) ---
        veh_card = ttk.LabelFrame(tables_pane, text="Log Consolidado de Veículos (Tempo Real)", padding=8)
        veh_wrap = ttk.Frame(veh_card, style="Card.TFrame")
        veh_wrap.pack(fill=tk.BOTH, expand=True)
        veh_cols = ("timestamp", "tipo", "placa", "cor", "modelo")
        self.ai_vehicle_tree = ttk.Treeview(veh_wrap, columns=veh_cols, show="headings", height=7)
        self.ai_vehicle_tree.heading("timestamp", text="Timestamp (Horário)")
        self.ai_vehicle_tree.heading("tipo", text="Tipo de Veículo")
        self.ai_vehicle_tree.heading("placa", text="Placa")
        self.ai_vehicle_tree.heading("cor", text="Cor")
        self.ai_vehicle_tree.heading("modelo", text="Modelo Aproximado")

        self.ai_vehicle_tree.column("timestamp", width=155, anchor=tk.CENTER)
        self.ai_vehicle_tree.column("tipo", width=115, anchor=tk.CENTER)
        self.ai_vehicle_tree.column("placa", width=110, anchor=tk.CENTER)
        self.ai_vehicle_tree.column("cor", width=125, anchor=tk.CENTER)
        self.ai_vehicle_tree.column("modelo", width=395, anchor=tk.W)

        veh_scroll = ttk.Scrollbar(veh_wrap, orient=tk.VERTICAL, command=self.ai_vehicle_tree.yview)
        self.ai_vehicle_tree.configure(yscrollcommand=veh_scroll.set)
        self.ai_vehicle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        veh_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        veh_actions = ttk.Frame(veh_card, style="Card.TFrame")
        veh_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(veh_actions, text="📤 Exportar Veículos CSV...", command=self._export_ai_csv_clicked).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(veh_actions, text="📄 Abrir CSV Veículos", command=self._open_ai_csv_clicked).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(veh_actions, text="🗑 Limpar Veículos", command=self._clear_ai_table_clicked).pack(
            side=tk.LEFT, padx=4
        )
        self.ai_csv_path_var = tk.StringVar(value=f"CSV automático: {Path(self.config.ai.csv_log_path).name}")
        ttk.Label(veh_actions, textvariable=self.ai_csv_path_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        tables_pane.add(veh_card, minsize=185, stretch="always")

        # --- Tabela 2: Pessoas e Vestuário (mesmo padrão estrutural) ---
        ppl_card = ttk.LabelFrame(
            tables_pane,
            text="Log Consolidado de Pessoas e Vestuário (Tempo Real)",
            padding=8,
        )
        ppl_wrap = ttk.Frame(ppl_card, style="Card.TFrame")
        ppl_wrap.pack(fill=tk.BOTH, expand=True)
        ppl_cols = ("timestamp", "tipo", "roupas")
        self.ai_people_tree = ttk.Treeview(ppl_wrap, columns=ppl_cols, show="headings", height=7)
        self.ai_people_tree.heading("timestamp", text="Timestamp (Horário)")
        self.ai_people_tree.heading("tipo", text="Tipo (Homem/Mulher)")
        self.ai_people_tree.heading("roupas", text="Cor das Roupas")

        self.ai_people_tree.column("timestamp", width=165, anchor=tk.CENTER)
        self.ai_people_tree.column("tipo", width=170, anchor=tk.CENTER)
        self.ai_people_tree.column("roupas", width=565, anchor=tk.W)

        ppl_scroll = ttk.Scrollbar(ppl_wrap, orient=tk.VERTICAL, command=self.ai_people_tree.yview)
        self.ai_people_tree.configure(yscrollcommand=ppl_scroll.set)
        self.ai_people_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ppl_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ppl_actions = ttk.Frame(ppl_card, style="Card.TFrame")
        ppl_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(ppl_actions, text="📤 Exportar Pessoas CSV...", command=self._export_people_csv_clicked).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(ppl_actions, text="📄 Abrir CSV Pessoas", command=self._open_people_csv_clicked).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(ppl_actions, text="🗑 Limpar Pessoas", command=self._clear_people_table_clicked).pack(
            side=tk.LEFT, padx=4
        )
        self.ai_people_csv_path_var = tk.StringVar(
            value=f"CSV automático: {Path(self.config.ai.people_csv_log_path).name}"
        )
        ttk.Label(ppl_actions, textvariable=self.ai_people_csv_path_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        tables_pane.add(ppl_card, minsize=185, stretch="always")

    def _save_ai_prefs(self) -> None:
        self.config.ai.enabled = bool(self.ai_enabled_var.get())
        self.config.ai.show_overlays = bool(self.ai_overlay_var.get())
        self.config.ai.detect_people_gender = bool(self.ai_gender_var.get())
        self.config.ai.detect_animals = bool(self.ai_animals_var.get())
        self.config.ai.detect_vehicles_alpr = bool(self.ai_vehicles_var.get())
        self.config.ai.save_csv_log = bool(self.ai_csv_var.get())
        self.config_manager.save(self.config)

        self.ai_pipeline.configure(
            analysis_fps=self.config.ai.analysis_fps,
            detect_people_gender=self.config.ai.detect_people_gender,
            detect_animals=self.config.ai.detect_animals,
            detect_vehicles_alpr=self.config.ai.detect_vehicles_alpr,
            save_csv=self.config.ai.save_csv_log,
            csv_path=self.config.ai.csv_log_path,
            people_csv_path=self.config.ai.people_csv_log_path,
        )

    def _on_ai_prefs_changed(self) -> None:
        self._save_ai_prefs()
        if self.config.ai.enabled:
            if self.ai_pipeline.is_ready:
                self.ai_pipeline.start()
                self.ai_status_var.set(f"Ativa ({self.ai_pipeline.get_summary().backend_name})")
                self.post_log("[IA] Análise em tempo real ativada.")
            else:
                self._init_ai_models_clicked()
        else:
            self.ai_pipeline.stop()
            if self.ai_pipeline.is_ready:
                self.ai_status_var.set("Modelos prontos (pausada)")
            self.post_log("[IA] Análise em tempo real pausada.")
        self._update_ai_button_state()
        self._update_ai_banner(self.ai_pipeline.get_summary())

    def toggle_ai_pipeline(self) -> None:
        """Alterna rapidamente a análise de IA pelo botão da barra superior."""
        self.ai_enabled_var.set(not bool(self.ai_enabled_var.get()))
        self._on_ai_prefs_changed()

    def _update_ai_button_state(self) -> None:
        if self.config.ai.enabled and self.ai_pipeline.is_loading:
            self.btn_ai_toggle.configure(text="🤖 IA: Carregando...", bg="#d97706")
        elif self.config.ai.enabled and self.ai_pipeline.is_ready:
            self.btn_ai_toggle.configure(text="🤖 IA: Ativa", bg="#059669")
        elif self.config.ai.enabled:
            self.btn_ai_toggle.configure(text="🤖 IA: Iniciando", bg="#2563eb")
        else:
            self.btn_ai_toggle.configure(text="🤖 IA: Desligada", bg="#374151")

    def _init_ai_models_clicked(self) -> None:
        if self.ai_pipeline.is_ready:
            self.ai_status_var.set(f"Pronta ({self.ai_pipeline.get_summary().backend_name})")
            if self.config.ai.enabled:
                self.ai_pipeline.start()
            self._update_ai_button_state()
            return
        if self.ai_pipeline.is_loading:
            return
        self.btn_init_ai.configure(state=tk.DISABLED)
        self.ai_status_var.set("Carregando modelos ONNX (YOLOv10, FairFace, ALPR, MobileCLIP)...")
        self._update_ai_button_state()
        threading.Thread(target=self._init_ai_models_worker, daemon=True).start()

    def _init_ai_models_worker(self) -> None:
        ok = self.ai_pipeline.load_default_models()
        backend = self.ai_pipeline.get_summary().backend_name
        msg = f"Ativa ({backend})" if ok else "Falha ao carregar modelos (veja o Log)"
        self._ui_queue.put(("ai_ready", (ok, msg)))

    def _update_ai_banner(self, summary: AISummary) -> None:
        counters = (
            f"Pessoas: {summary.people_total} (Homem: {summary.men_count} | Mulher: {summary.women_count})  •  "
            f"Animais: {summary.animals_count}  •  "
            f"Carros: {summary.cars_count}  •  "
            f"Motos: {summary.motos_count}  •  "
            f"Bicicletas: {summary.bicycles_count}"
        )
        self.ai_tab_counters_var.set(counters)

        if not self.config.ai.enabled:
            self.ai_live_summary_var.set(
                "IA: Desligada • Clique em '🤖 IA' para ativar pessoas (gênero), animais e placas ALPR"
            )
            return
        if self.ai_pipeline.is_loading or not self.ai_pipeline.is_ready:
            self.ai_live_summary_var.set("IA: Inicializando motores ONNX DirectML (RX 580)...")
            return

        plate_part = f" | Última placa: {summary.last_plate}" if summary.last_plate else ""
        self.ai_live_summary_var.set(f"IA ({summary.backend_name}): {counters}{plate_part}")

    def _refresh_ai_table(
        self,
        records: list[VehicleRecord],
        person_records: list[PersonRecord] | None = None,
    ) -> None:
        snapshot = tuple(
            (r.timestamp, r.vehicle_type, r.plate, r.color, r.approx_model) for r in records[:200]
        )
        if snapshot != self._ai_vehicle_snapshot_key:
            self._ai_vehicle_snapshot_key = snapshot
            for row_id in self.ai_vehicle_tree.get_children():
                self.ai_vehicle_tree.delete(row_id)
            for row in snapshot:
                self.ai_vehicle_tree.insert("", tk.END, values=row)
        if person_records is not None:
            self._refresh_people_table(person_records)

    def _refresh_people_table(self, records: list[PersonRecord]) -> None:
        snapshot = tuple(
            (r.timestamp, r.person_type, r.clothing_colors) for r in records[:200]
        )
        if snapshot == self._ai_people_snapshot_key:
            return
        self._ai_people_snapshot_key = snapshot
        for row_id in self.ai_people_tree.get_children():
            self.ai_people_tree.delete(row_id)
        for row in snapshot:
            self.ai_people_tree.insert("", tk.END, values=row)

    def _clear_ai_table_clicked(self) -> None:
        if not messagebox.askyesno("Limpar Veículos", "Limpar todos os registros veiculares da tabela e do CSV atual?"):
            return
        self._ai_vehicle_snapshot_key = ()
        self.ai_pipeline.clear_vehicle_records()

    def _clear_people_table_clicked(self) -> None:
        if not messagebox.askyesno("Limpar Pessoas", "Limpar todos os registros de pessoas da tabela e do CSV atual?"):
            return
        self._ai_people_snapshot_key = ()
        self.ai_pipeline.clear_person_records()

    def _export_csv_dialog(self, *, title: str, prefix: str, export_fn: Callable[[str | Path], Path]) -> None:
        target = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".csv",
            initialfile=f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
            filetypes=[("Planilha CSV", "*.csv")],
        )
        if not target:
            return
        out = export_fn(target)
        messagebox.showinfo("Exportar CSV", f"Tabela exportada com sucesso em:\n{out}")

    def _open_csv_file(self, csv_path: Path, export_fn: Callable[[str | Path], Path]) -> None:
        if not csv_path.exists():
            export_fn(csv_path)
        try:
            if os.name == "nt":
                os.startfile(str(csv_path))  # type: ignore[attr-defined]
            else:
                webbrowser.open(csv_path.as_uri())
        except OSError as err:
            messagebox.showerror("Abrir CSV", f"Não foi possível abrir o arquivo:\n{err}")

    def _export_ai_csv_clicked(self) -> None:
        self._export_csv_dialog(
            title="Exportar Tabela de Veículos (CSV)",
            prefix="veiculos",
            export_fn=self.ai_pipeline.export_csv,
        )

    def _open_ai_csv_clicked(self) -> None:
        self._open_csv_file(Path(self.config.ai.csv_log_path), self.ai_pipeline.export_csv)

    def _export_people_csv_clicked(self) -> None:
        self._export_csv_dialog(
            title="Exportar Tabela de Pessoas (CSV)",
            prefix="pessoas",
            export_fn=self.ai_pipeline.export_people_csv,
        )

    def _open_people_csv_clicked(self) -> None:
        self._open_csv_file(Path(self.config.ai.people_csv_log_path), self.ai_pipeline.export_people_csv)

    def _test_ai_image_clicked(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Selecionar foto para testar detecção + ALPR + vestuário",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if not chosen:
            return
        self.ai_status_var.set("Analisando imagem selecionada...")

        def worker() -> None:
            if not self.ai_pipeline.is_ready:
                ok = self.ai_pipeline.load_default_models()
                if not ok:
                    self._ui_queue.put(("ai_ready", (False, "Falha ao carregar modelos")))
                    return
            try:
                raw = np.fromfile(chosen, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            except OSError:
                img = None
            if img is None or img.size == 0:
                self.post_log(f"[IA] Não foi possível decodificar a imagem: {chosen}")
                self._ui_queue.put(("ai_ready", (self.ai_pipeline.is_ready, "Falha ao ler imagem")))
                return
            items, _dirty = self.ai_pipeline.process_frame(img, camera_name=f"Foto: {Path(chosen).name}")
            annotated = draw_ai_overlays(img, items)
            self._ui_queue.put(
                (
                    "ai_test_result",
                    (
                        annotated,
                        len(items),
                        Path(chosen).name,
                        self.ai_pipeline.get_summary(),
                        self.ai_pipeline.get_vehicle_records(),
                        self.ai_pipeline.get_person_records(),
                    ),
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    def _show_ai_test_preview(self, annotated_bgr: np.ndarray, count: int, filename: str) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"Resultado IA — {filename} ({count} objeto(s))")
        win.configure(bg="#181c24")

        h, w = annotated_bgr.shape[:2]
        scale = min(960.0 / float(w), 640.0 / float(h), 1.0)
        disp = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        if scale < 1.0:
            disp = cv2.resize(disp, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(image=Image.fromarray(disp))
        self._ai_preview_photo = photo
        tk.Label(win, image=photo, bg="#181c24").pack(padx=10, pady=10)
        tk.Label(
            win,
            text=f"{count} detecção(ões) em '{filename}'. Veja veículos consolidados na tabela abaixo.",
            bg="#181c24",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
        ).pack(pady=(0, 10))

    # ------------------------------------------------------------------ #
    # Lógica de Câmeras, Descoberta e Preview                            #
    # ------------------------------------------------------------------ #

    def _sensitivity_label_from_float(self, value: float) -> str:
        closest_label = "Média (recomendado)"
        best_diff = 999.0
        for label, threshold in _SENSITIVITY_MAP.items():
            diff = abs(threshold - value)
            if diff < best_diff:
                best_diff = diff
                closest_label = label
        return closest_label

    def _refresh_camera_lists(self) -> None:
        names = [f"{cam.name} ({cam.ip or 'USB ' + str(cam.usb_index)})" for cam in self.config.cameras]
        self.active_cam_combo.configure(values=names)

        selected_cam = self.config_manager.get_selected_camera(self.config)
        for idx, cam in enumerate(self.config.cameras):
            if cam.id == selected_cam.id:
                self.active_cam_combo.current(idx)
                break

        self.saved_cams_listbox.delete(0, tk.END)
        for idx, cam in enumerate(self.config.cameras):
            target = cam.ip or f"USB #{cam.usb_index}"
            self.saved_cams_listbox.insert(tk.END, f"{cam.name} — {cam.brand.split('/')[0].strip()} ({target})")
            if cam.id == self._editing_cam_id:
                self.saved_cams_listbox.selection_clear(0, tk.END)
                self.saved_cams_listbox.selection_set(idx)
        self._sync_playback_camera_choices()

    def _load_camera_into_form(self, cam: CameraConfig) -> None:
        self._editing_cam_id = cam.id
        self.form_name_var.set(cam.name)
        self.form_brand_var.set(cam.brand if cam.brand in BRAND_NAMES else BRAND_NAMES[0])
        self.form_ip_var.set(cam.ip)
        self.form_rtsp_port_var.set(str(cam.rtsp_port))
        self.form_onvif_port_var.set(str(cam.onvif_port))
        self.form_user_var.set(cam.username)
        self.form_pass_var.set(cam.password)
        self.form_channel_var.set(str(cam.channel))
        self.form_stream_var.set(cam.stream_type if cam.stream_type in ("main", "sub") else "main")
        self.form_usb_var.set(str(cam.usb_index))
        self.form_serial_var.set(cam.serial_number)
        self.form_custom_url_var.set(cam.source_url_override or cam.custom_rtsp_url)
        self._update_rtsp_preview_label()

    def _build_camera_from_form(self) -> CameraConfig:
        try:
            rtsp_port = int(self.form_rtsp_port_var.get().strip() or 554)
        except ValueError:
            rtsp_port = 554
        try:
            onvif_port = int(self.form_onvif_port_var.get().strip() or 8899)
        except ValueError:
            onvif_port = 8899
        try:
            channel = max(1, int(self.form_channel_var.get().strip() or 1))
        except ValueError:
            channel = 1
        try:
            usb_index = max(0, int(self.form_usb_var.get().strip() or 0))
        except ValueError:
            usb_index = 0

        manual_url = self.form_custom_url_var.get().strip()
        brand = self.form_brand_var.get() or BRAND_NAMES[0]
        existing = next((c for c in self.config.cameras if c.id == self._editing_cam_id), None)
        return CameraConfig(
            id=self._editing_cam_id,
            name=self.form_name_var.get().strip() or "Câmera",
            brand=brand,
            ip=self.form_ip_var.get().strip(),
            rtsp_port=rtsp_port,
            onvif_port=onvif_port,
            username=self.form_user_var.get().strip(),
            password=self.form_pass_var.get(),
            channel=channel,
            stream_type=self.form_stream_var.get() or "main",
            custom_rtsp_url=manual_url if brand == "Genérico RTSP" else "",
            usb_index=usb_index,
            source_url_override=manual_url if brand != "Genérico RTSP" else "",
            serial_number=self.form_serial_var.get().strip(),
            ptz_speed=existing.ptz_speed if existing else 5,
            ptz_home_preset=existing.ptz_home_preset if existing else XM_WATCH_PRESET_ID,
            ptz_auto_return_enabled=existing.ptz_auto_return_enabled if existing else False,
            ptz_auto_return_seconds=existing.ptz_auto_return_seconds if existing else 30,
        )

    def _update_rtsp_preview_label(self) -> None:
        cam = self._build_camera_from_form()
        src = resolve_camera_source(cam)
        if isinstance(src, int):
            self.form_preview_url_var.set(f"Webcam USB local (dispositivo #{src})")
        elif not src:
            self.form_preview_url_var.set("Informe o IP da câmera ou busque na rede.")
        else:
            self.form_preview_url_var.set(mask_rtsp_url(src))

    def _on_brand_changed(self, _event: object = None) -> None:
        brand = self.form_brand_var.get()
        rtsp_p, onvif_p = get_brand_defaults(brand)
        self.form_rtsp_port_var.set(str(rtsp_p))
        self.form_onvif_port_var.set(str(onvif_p))
        if brand != "Genérico RTSP":
            self.form_custom_url_var.set("")
        self._update_rtsp_preview_label()

    def _on_top_camera_selected(self, _event: object = None) -> None:
        idx = self.active_cam_combo.current()
        if 0 <= idx < len(self.config.cameras):
            cam = self.config.cameras[idx]
            self.config.selected_camera_id = cam.id
            self.config_manager.save(self.config)
            self._load_camera_into_form(cam)
            self._load_ptz_prefs_from_camera(cam)
            self._refresh_camera_lists()

    def _on_saved_listbox_select(self, _event: object = None) -> None:
        sel = self.saved_cams_listbox.curselection()
        if not sel:
            return
        cam = self.config.cameras[sel[0]]
        self._load_camera_into_form(cam)

    def _new_camera_profile(self) -> None:
        fresh = CameraConfig(name=f"Câmera {len(self.config.cameras) + 1}")
        self._load_camera_into_form(fresh)
        self.camera_test_msg_var.set("Preencha os dados da nova câmera e clique em 'Salvar Câmera'.")

    def _delete_selected_camera(self) -> None:
        if len(self.config.cameras) <= 1:
            messagebox.showinfo("HomeCam Segurança", "Mantenha ao menos um perfil de câmera cadastrado.")
            return
        self.config_manager.delete_camera(self.config, self._editing_cam_id)
        self.config_manager.save(self.config)
        selected = self.config_manager.get_selected_camera(self.config)
        self._load_camera_into_form(selected)
        self._refresh_camera_lists()

    def _save_form_camera(self, connect_after: bool = False) -> None:
        cam = self._build_camera_from_form()
        self.config_manager.upsert_camera(self.config, cam)
        self.config_manager.save(self.config)
        self._refresh_camera_lists()
        self.post_log(f"Perfil salvo: {cam.name}")
        if connect_after:
            self.notebook.select(self.tab_live)
            self.connect_active_camera()

    def _start_network_discovery(self) -> None:
        self.btn_scan_net.configure(state=tk.DISABLED)
        self.discovery_status_var.set("Procurando câmeras na rede local (ONVIF + portas RTSP)...")

        def worker() -> None:
            cams = discover_all_cameras()
            self._ui_queue.put(("discovery_done", cams))

        threading.Thread(target=worker, daemon=True).start()

    def _start_usb_discovery(self) -> None:
        self.discovery_status_var.set("Verificando webcams USB conectadas ao computador...")

        def worker() -> None:
            indices = detect_usb_webcams()
            self._ui_queue.put(("usb_discovery_done", indices))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_selected_discovered(self) -> None:
        sel = self.discovered_listbox.curselection()
        if not sel:
            messagebox.showinfo("Descoberta", "Selecione uma câmera encontrada na lista.")
            return
        item = self._discovered[sel[0]]
        self.form_ip_var.set(item.ip)
        self.form_brand_var.set(item.suggested_brand)
        self.form_rtsp_port_var.set(str(item.rtsp_port or 554))
        self.form_onvif_port_var.set(str(item.onvif_port or 8899))
        if item.name:
            self.form_name_var.set(item.name)
        self.form_custom_url_var.set("")
        self.camera_test_msg_var.set(
            f"IP {item.ip} preenchido! Informe usuário/senha da câmera e clique em 'Testar Conexão' ou 'Descobrir URL via ONVIF'."
        )

    def _query_onvif_clicked(self) -> None:
        cam = self._build_camera_from_form()
        if not cam.ip:
            messagebox.showwarning("ONVIF", "Informe o IP da câmera antes de consultar via ONVIF.")
            return
        self.camera_test_msg_var.set("Consultando URL de vídeo via ONVIF SOAP...")

        def worker() -> None:
            uris = query_onvif_stream_uris(
                cam.ip,
                onvif_port=cam.onvif_port,
                username=cam.username,
                password=cam.password,
            )
            self._ui_queue.put(("onvif_uris", uris))

        threading.Thread(target=worker, daemon=True).start()

    def _lookup_by_serial_clicked(self) -> None:
        cam = self._build_camera_from_form()
        serial = cam.serial_number.strip()
        if not serial:
            messagebox.showwarning(
                "Buscar por Serial",
                "Informe o N.º de Série completo da câmera (ex.: 54550d137d64efea).",
            )
            return
        self.btn_serial_lookup.configure(state=tk.DISABLED)
        self.camera_test_msg_var.set("Procurando câmera na rede local pelo N.º de Série (porta 34567)...")

        def worker() -> None:
            local_ip = get_local_ipv4()
            found = find_xiongmai_by_serial(
                serial,
                local_ip=local_ip,
                username=cam.username or "admin",
                password=cam.password,
                port=XM_DEFAULT_PORT,
            )
            self._ui_queue.put(("serial_lookup_done", (serial, found)))

        threading.Thread(target=worker, daemon=True).start()

    def _test_camera_clicked(self) -> None:
        cam = self._build_camera_from_form()
        self.btn_test_conn.configure(state=tk.DISABLED)
        self.camera_test_msg_var.set("Testando conexão e recebimento de vídeo...")

        def worker() -> None:
            src = resolve_best_connection_source(cam)
            result = test_camera_connection(src, timeout_sec=5.0)
            self._ui_queue.put(("test_result", result))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Streaming ao vivo, Gravação e Snapshot                             #
    # ------------------------------------------------------------------ #

    def connect_active_camera(self, start_auto_record: bool = True) -> None:
        """Conecta ao fluxo da câmera ativa e inicia exibição em tempo real."""
        self.disconnect_camera()
        cam = self.config_manager.get_selected_camera(self.config)
        source = resolve_camera_source(cam)
        if isinstance(source, str) and not source.strip():
            messagebox.showwarning(
                "Câmera não configurada",
                "Informe o IP ou URL RTSP da câmera na aba 'Câmeras (Conexão Fácil)'.",
            )
            self.notebook.select(self.tab_cameras)
            return

        self._stream_stop.clear()
        self._is_connected = True
        self._ptz_session_token += 1
        ptz_token = self._ptz_session_token
        self.ptz_backend_var.set("Conectando controle PTZ...")
        self.btn_connect.configure(state=tk.DISABLED)
        self.btn_disconnect.configure(state=tk.NORMAL)
        self._set_status_badge("Conectando...", "#d97706")
        display_target = f"Webcam USB #{source}" if isinstance(source, int) else mask_rtsp_url(source)
        self.post_log(f"Conectando à câmera '{cam.name}' ({display_target})...")

        self._stream_thread = threading.Thread(
            target=self._capture_loop,
            args=(cam, ptz_token),
            daemon=True,
        )
        self._stream_thread.start()

        if self.config.ai.enabled and self.ai_pipeline.is_ready:
            self.ai_pipeline.start()

        if start_auto_record and self.config.recording.auto_start_recording and not self.recorder.is_recording:
            self.start_recording()

    def disconnect_camera(self) -> None:
        """Encerra visualização ao vivo e gravação ativa."""
        self.ai_pipeline.stop()
        self._record_request_id += 1
        prev_token = self._ptz_session_token
        self._ptz_session_token += 1
        self._cancel_pending_ptz_stop()
        if self._ptz_holding and self._ptz_active_dir:
            self._ptz_queue.put(("move_stop", (prev_token, self._ptz_active_dir)))
        self._ptz_holding = False
        self._ptz_active_dir = ""
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("clear_backend", None))
        self._set_ptz_controls_enabled(False, fw_supported=False)
        self.ptz_backend_var.set("Conecte a câmera para usar PTZ")

        self.btn_record.configure(state=tk.NORMAL)
        if self.recorder.is_recording:
            self.stop_recording()
        self._stream_stop.set()
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        self._stream_thread = None
        self._is_connected = False
        self.btn_connect.configure(state=tk.NORMAL)
        self.btn_disconnect.configure(state=tk.DISABLED)
        self._set_status_badge("Desconectado", "#374151")
        self.motion_indicator.configure(text="Movimento: aguardando", bg="#2d3748", fg="#cbd5e1")
        self.video_label.configure(
            image="",
            text="Câmera desconectada.\nClique em '▶ Conectar' quando desejar visualizar novamente.",
        )
        self._latest_photo = None
        self._update_live_recording_status()

    def _capture_loop(self, cam: CameraConfig, ptz_token: int = 0) -> None:
        cam_name = cam.name
        fallback_logged = False
        ptz_init_source: object = None
        while not self._stream_stop.is_set():
            source = resolve_best_connection_source(cam)
            if (
                not fallback_logged
                and isinstance(source, str)
                and source.lower().startswith("dvrip://")
                and cam.brand != "Xiongmai NetIP / iCSee (Porta 34567)"
            ):
                fallback_logged = True
                self.post_log(
                    f"RTSP indisponível; usando protocolo nativo Xiongmai NetIP ({mask_rtsp_url(source)})."
                )
            cap = open_video_capture(source, timeout_ms=5000)
            if not cap.isOpened():
                cap.release()
                self._ui_queue.put(("status_badge", ("Reconectando...", "#d97706")))
                self.post_log(f"Falha ao abrir fluxo de '{cam_name}'. Nova tentativa em 4s...")
                self._stream_stop.wait(4.0)
                continue

            if ptz_init_source != source:
                ptz_init_source = source
                self._ptz_queue.put(("init_backend", (cam, source, ptz_token)))

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
            if fps <= 0 or fps > 120:
                fps = 15.0
            self._current_fps = fps
            self._ui_queue.put(("status_badge_live", None))
            self.post_log(f"Visualização ao vivo ativa: {cam_name}")

            try:
                while not self._stream_stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.post_log("Sinal de vídeo perdido. Tentando reconectar...")
                        self._ui_queue.put(("status_badge", ("Reconectando...", "#d97706")))
                        break

                    if self.recorder.is_recording:
                        motion = self.recorder.feed_frame(frame)
                        self._ui_queue.put(("motion_state", motion))
                        if motion:
                            self.dvr_events.record_event(
                                cam_name,
                                "motion",
                                "Movimento detectado",
                                min_interval_sec=4.0,
                            )

                    if self.config.ai.enabled and self.ai_pipeline.is_ready:
                        self.ai_pipeline.submit_frame(frame, camera_name=cam_name)

                    with self._frame_lock:
                        self._latest_frame = frame
                        should_schedule = not self._frame_render_scheduled
                        self._frame_render_scheduled = True
                    if should_schedule:
                        self._ui_queue.put(("render_frame", None))
            finally:
                cap.release()

            if not self._stream_stop.is_set():
                self._stream_stop.wait(3.0)

    def _render_latest_frame(self) -> None:
        with self._frame_lock:
            frame = self._latest_frame
            self._frame_render_scheduled = False
        if frame is None or not self._is_connected:
            return

        target_w = max(320, self.video_label.winfo_width() - 8)
        target_h = max(240, self.video_label.winfo_height() - 8)
        h, w = frame.shape[:2]
        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        display_frame = frame
        if self.config.ai.enabled and self.config.ai.show_overlays and self.ai_pipeline.is_ready:
            dets = self.ai_pipeline.get_overlay_detections()
            if dets:
                display_frame = draw_ai_overlays(frame, dets)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        if new_w != w or new_h != h:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=pil_img)
        self._latest_photo = photo
        self.video_label.configure(image=photo, text="")

    def toggle_recording(self) -> None:
        if self.recorder.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        self._save_recording_prefs()
        cam = self.config_manager.get_selected_camera(self.config)
        source = resolve_camera_source(cam)
        if isinstance(source, str) and not source.strip():
            messagebox.showwarning("Gravação", "Configure uma câmera válida antes de gravar.")
            return
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            self._record_request_id += 1
            req_id = self._record_request_id
            self.btn_record.configure(state=tk.DISABLED)

            def worker() -> None:
                best = resolve_best_connection_source(cam)
                self._ui_queue.put(("start_recording_resolved", (req_id, cam, best)))

            threading.Thread(target=worker, daemon=True).start()
            return
        self._apply_start_recording(cam, source)

    def _apply_start_recording(self, cam: CameraConfig, source: str | int) -> None:
        if (isinstance(source, int) or self.config.recording.motion_only) and not self._is_connected:
            self.connect_active_camera(start_auto_record=False)

        use_oauth = self.config.drive.enabled and self.config.drive.sync_mode == "oauth"
        resolved_dir, changed_dir = resolve_writable_recordings_dir(
            self.config.recording.recordings_dir,
            prefer_drive_subdir=not use_oauth,
            fallback=DEFAULT_RECORDINGS_DIR,
        )
        if changed_dir:
            self.config.recording.recordings_dir = str(resolved_dir)
            self.recordings_dir_var.set(str(resolved_dir))
            self.config_manager.save(self.config)
            self.post_log(
                f"[Gravação] Pasta selecionada sem permissão de escrita; ajustada para {resolved_dir}."
            )

        self.recorder.recordings_root = resolved_dir
        try:
            self.recorder.start(
                source=source,
                camera_name=cam.name,
                segment_minutes=self.config.recording.segment_minutes,
                motion_only=self.config.recording.motion_only,
                motion_sensitivity=self.config.recording.motion_sensitivity,
                motion_post_seconds=self.config.recording.motion_post_seconds,
                fps=self._current_fps,
            )
        except Exception as err:
            self.btn_record.configure(text="● Iniciar Gravação", bg="#dc2626", state=tk.NORMAL)
            if self._is_connected:
                self._set_status_badge("Ao Vivo", "#16a34a")
            else:
                self._set_status_badge("Desconectado", "#374151")
            self.post_log(f"[Gravação] Falha ao iniciar gravação: {err}")
            messagebox.showerror("Erro na Gravação", f"Não foi possível iniciar a gravação:\n\n{err}")
            return

        self.btn_record.configure(text="⏹ Parar Gravação", bg="#7f1d1d", state=tk.NORMAL)
        self._set_status_badge("Gravando", "#dc2626")
        self._update_live_recording_status()

    def stop_recording(self) -> None:
        self._record_request_id += 1
        self.recorder.stop()
        self.btn_record.configure(text="● Iniciar Gravação", bg="#dc2626", state=tk.NORMAL)
        if self._is_connected:
            self._set_status_badge("Ao Vivo", "#16a34a")
        else:
            self._set_status_badge("Desconectado", "#374151")
        self.post_log("Gravação interrompida e arquivo finalizado.")
        self._update_live_recording_status()

    def take_snapshot(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            messagebox.showinfo("Foto", "Conecte à câmera primeiro para capturar uma foto.")
            return
        cam = self.config_manager.get_selected_camera(self.config)
        out_path = save_snapshot(frame, self.config.recording.recordings_dir, cam.name)
        self.post_log(f"Foto salva em {out_path}")
        if self.config.drive.enabled and self.config.drive.auto_upload:
            if self.uploader.enqueue_file(out_path, camera_name=cam.name):
                self._refresh_queue_tree()

    def _save_recording_prefs(self) -> None:
        try:
            seg_min = max(1, int(self.segment_min_var.get()))
        except ValueError:
            seg_min = 5
        self.config.recording.segment_minutes = seg_min
        self.config.recording.motion_only = bool(self.motion_only_var.get())
        self.config.recording.motion_sensitivity = _SENSITIVITY_MAP.get(self.sensitivity_var.get(), 0.015)
        self.config.recording.auto_start_recording = bool(self.auto_record_var.get())
        self.config.recording.auto_connect_on_startup = bool(self.auto_connect_var.get())
        self.config.recording.recordings_dir = self.recordings_dir_var.get().strip() or self.config.recording.recordings_dir
        self.dvr_events.events_dir = Path(self.config.recording.recordings_dir) / "_timeline_events"
        self.dvr_player.set_recordings_root(self.config.recording.recordings_dir)
        self.config_manager.save(self.config)

    def _update_live_recording_status(self) -> None:
        """Atualiza o resumo instantâneo da gravação ativa e fila de envio ao Drive."""
        if not hasattr(self, "rec_live_status_var"):
            return
        pending = self.uploader.get_pending_count()
        if not self.config.drive.enabled:
            drive_txt = "Drive: desativado"
        elif pending > 0:
            drive_txt = f"Drive: enviando / fila ({pending})"
        elif self.uploader.is_ready():
            drive_txt = "Drive: pronto (fila: 0)"
        else:
            drive_txt = "Drive: aguardando login"

        if not self.recorder.is_recording:
            self.rec_live_status_var.set(f"Gravação automática: parada • {drive_txt}")
            return

        fname, size_bytes, elapsed_sec, seg_secs = self.recorder.get_active_recording_stats()
        if size_bytes >= 1024 * 1024:
            size_txt = f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes > 0:
            size_txt = f"{max(1, size_bytes // 1024)} KB"
        else:
            size_txt = "aguardando 1º keyframe..."

        if self.config.recording.motion_only:
            self.rec_live_status_var.set(
                f"● Gravando por movimento ({fname} | {size_txt}) • {drive_txt}"
            )
            return

        rem = max(0, int(seg_secs - elapsed_sec))
        mm, ss = divmod(rem, 60)
        self.rec_live_status_var.set(
            f"● Gravando: {fname} ({size_txt} | próx. envio Drive em {mm:02d}:{ss:02d}) • {drive_txt}"
        )

    def _recover_orphaned_recordings(self) -> None:
        """Recupera gravações válidas pendentes de sessões anteriores e isola MP4s sem índice."""
        if self._closing:
            return
        recovered, quarantined = recover_orphaned_recordings(
            Path(self.config.recording.recordings_dir),
            self.uploader,
            drive_enabled=bool(self.config.drive.enabled),
            auto_upload=bool(self.config.drive.auto_upload),
            delete_after_upload=bool(self.config.drive.delete_after_upload),
            should_stop=lambda: self._closing,
            on_log=self.post_log,
        )
        if recovered > 0 or quarantined > 0:
            self._ui_queue.put(("refresh_queue", None))

    def _choose_recordings_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.recordings_dir_var.get())
        if chosen:
            if not is_directory_writable(chosen):
                use_oauth = self.config.drive.enabled and self.config.drive.sync_mode == "oauth"
                resolved, _ = resolve_writable_recordings_dir(
                    chosen,
                    prefer_drive_subdir=not use_oauth,
                    fallback=DEFAULT_RECORDINGS_DIR,
                )
                messagebox.showwarning(
                    "Pasta sem permissão de escrita",
                    f"Não é possível criar subpastas diretamente em:\n{chosen}\n\n"
                    f"Observação: na unidade do Google Drive (ex.: H:\\), apenas a pasta 'Meu Drive' é gravável.\n\n"
                    f"Usaremos a pasta válida:\n{resolved}",
                )
                chosen = str(resolved)
            self.recordings_dir_var.set(chosen)
            self._save_recording_prefs()

    def _open_recordings_dir(self) -> None:
        raw_path = self.recordings_dir_var.get().strip() or self.config.recording.recordings_dir
        path, changed = resolve_writable_recordings_dir(raw_path, fallback=DEFAULT_RECORDINGS_DIR)
        if changed:
            self.recordings_dir_var.set(str(path))
            self._save_recording_prefs()
        os.startfile(str(path))

    # ------------------------------------------------------------------ #
    # Google Drive Actions                                               #
    # ------------------------------------------------------------------ #

    def _choose_credentials_json(self) -> bool:
        chosen = filedialog.askopenfilename(
            title="Selecione o JSON OAuth baixado (credentials.json ou client_secret_*.json)",
            filetypes=[("JSON OAuth", "*.json"), ("Todos os arquivos", "*.*")],
        )
        if not chosen:
            return False
        src = Path(chosen)
        valid_json, err_msg = validate_oauth_client_json(src)
        if not valid_json:
            messagebox.showerror("Arquivo OAuth inválido", err_msg)
            return False

        dest_dir = DEFAULT_SECRETS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "credentials.json"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        self.drive_creds_var.set(str(dest))
        self.drive_sync_mode_var.set("oauth")
        self._save_drive_settings()
        self.post_log(f"Arquivo credentials.json válido salvo em {dest}")
        return True

    def _choose_drive_folder_clicked(self) -> None:
        chosen = filedialog.askdirectory(
            title="Selecione a pasta do Google Drive (ex.: H:\\Meu Drive)",
            initialdir=self.drive_folder_var.get().strip() or None,
        )
        if not chosen:
            return
        norm_dir, changed = normalize_drive_folder_dir(chosen)
        if changed:
            messagebox.showinfo(
                "Pasta do Google Drive",
                f"A raiz '{chosen}' não permite gravação direta; ajustada automaticamente para:\n{norm_dir}",
            )
            chosen = norm_dir
        self.drive_folder_var.set(chosen)
        self.drive_sync_mode_var.set("folder")
        self._save_drive_settings()
        self.post_log(f"Pasta do Google Drive selecionada: {chosen}")

    def _detect_drive_folder_clicked(self) -> None:
        dirs = detect_google_drive_desktop_dirs()
        if dirs:
            chosen = dirs[0]
            self.drive_folder_var.set(str(chosen))
            self.drive_sync_mode_var.set("folder")
            self.drive_enabled_var.set(True)
            self._save_drive_settings()
            self.uploader.trigger_sync()
            self.post_log(f"Google Drive para computador detectado em {chosen}")
            messagebox.showinfo(
                "Google Drive",
                f"Pasta detectada e ativada!\n{chosen}\n\n"
                f"Os vídeos serão salvos automaticamente em:\n{chosen / self.config.drive.root_folder_name}",
            )
            return

        if messagebox.askyesno(
            "Google Drive para computador não encontrado",
            "Não encontramos uma pasta 'Meu Drive' instalada neste PC.\n\n"
            "Deseja abrir o site oficial para baixar o Google Drive para computador?\n"
            "(Clique em 'Não' para selecionar manualmente outra pasta sincronizada.)",
        ):
            webbrowser.open("https://www.google.com/intl/pt-BR/drive/download/")
        else:
            self._choose_drive_folder_clicked()

    def _on_drive_mode_changed(self) -> None:
        self._save_drive_settings()

    def _save_drive_settings(self) -> None:
        mode = self.drive_sync_mode_var.get()
        self.config.drive.sync_mode = "folder" if mode == "folder" else "oauth"
        self.config.drive.local_drive_dir = self.drive_folder_var.get().strip()
        self.config.drive.credentials_path = self.drive_creds_var.get().strip() or self.config.drive.credentials_path
        self.config.drive.root_folder_name = (
            sanitize_filename_part(self.drive_root_var.get(), "HomeCamSeguranca")
        )
        self.config.drive.enabled = bool(self.drive_enabled_var.get())
        self.config.drive.delete_after_upload = bool(self.drive_delete_local_var.get())
        self.config_manager.save(self.config)

        self.uploader.configure(
            credentials_path=self.config.drive.credentials_path,
            token_path=self.config.drive.token_path,
            root_folder_name=self.config.drive.root_folder_name,
            delete_after_upload=self.config.drive.delete_after_upload,
            sync_mode=self.config.drive.sync_mode,
            local_drive_dir=self.config.drive.local_drive_dir,
        )
        self._refresh_drive_status_ui()

    def _connect_google_drive(self) -> None:
        self._save_drive_settings()
        if self.config.drive.sync_mode == "folder":
            if not self.uploader.is_folder_ready():
                dirs = detect_google_drive_desktop_dirs()
                if dirs:
                    self.drive_folder_var.set(str(dirs[0]))
                    self._save_drive_settings()
                else:
                    self._detect_drive_folder_clicked()
                    return
            self.drive_enabled_var.set(True)
            self._save_drive_settings()
            self.uploader.trigger_sync()
            messagebox.showinfo(
                "Google Drive",
                f"Modo pasta sincronizada ativo!\nDestino: {self.config.drive.local_drive_dir}",
            )
            return

        if not self.uploader.has_credentials_file():
            self._open_drive_setup_wizard()
            return

        self.btn_drive_auth.configure(state=tk.DISABLED)

        def worker() -> None:
            ok, msg = self.uploader.authenticate_interactive()
            self._ui_queue.put(("drive_auth_done", (ok, msg)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_drive_setup_wizard(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Configurar Google Drive — HomeCam Segurança")
        win.configure(bg="#181c24")
        win.geometry("680x470")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        frame = tk.Frame(win, bg="#222834", padx=18, pady=16)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        tk.Label(
            frame,
            text="Por que apareceu 'credentials.json não encontrado'?",
            bg="#222834",
            fg="#ffffff",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))

        tk.Label(
            frame,
            text=(
                "Por segurança do Google, cada usuário gera sua própria chave OAuth gratuita no Google Cloud "
                "(ou usa a pasta do aplicativo Google Drive para computador sem precisar de chave nenhuma)."
            ),
            bg="#222834",
            fg="#cbd5e1",
            wraplength=610,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(0, 10))

        steps_box = tk.Text(
            frame,
            height=11,
            bg="#11161f",
            fg="#e2e8f0",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )
        steps_box.insert("1.0", DRIVE_SETUP_GUIDE_PT)
        steps_box.configure(state=tk.DISABLED)
        steps_box.pack(fill=tk.X, pady=(0, 12))

        row1 = tk.Frame(frame, bg="#222834")
        row1.pack(fill=tk.X, pady=(0, 8))
        tk.Button(
            row1,
            text="1. Abrir Google Cloud Console no Navegador",
            command=lambda: webbrowser.open("https://console.cloud.google.com/apis/credentials"),
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))

        def pick_and_connect() -> None:
            if self._choose_credentials_json():
                win.destroy()
                self._connect_google_drive()

        tk.Button(
            row1,
            text="2. Já baixei o JSON — Selecionar e Conectar",
            command=pick_and_connect,
            bg="#16a34a",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)

        row2 = tk.Frame(frame, bg="#222834")
        row2.pack(fill=tk.X, pady=(4, 0))

        def switch_to_folder_mode() -> None:
            self.drive_sync_mode_var.set("folder")
            self._on_drive_mode_changed()
            dirs = detect_google_drive_desktop_dirs()
            win.destroy()
            if dirs:
                self._detect_drive_folder_clicked()
            else:
                webbrowser.open("https://www.google.com/intl/pt-BR/drive/download/")
                messagebox.showinfo(
                    "Google Drive para Computador",
                    "Instale o Google Drive para computador no navegador que acabou de abrir, faça login com "
                    "sua conta Google e depois clique em 'Detectar Meu Drive' nesta aba.",
                )

        ttk.Button(
            row2,
            text="Alternativa sem JSON: Usar Google Drive para Computador",
            command=switch_to_folder_mode,
        ).pack(side=tk.LEFT)
        ttk.Button(row2, text="Fechar", command=win.destroy).pack(side=tk.RIGHT)

    def _disconnect_google_drive(self) -> None:
        if self.config.drive.sync_mode == "folder":
            self.drive_enabled_var.set(False)
            self._save_drive_settings()
        else:
            self.uploader.disconnect()
            self._refresh_drive_status_ui()

    def _sync_pending_now(self) -> None:
        if not self.uploader.is_ready():
            messagebox.showwarning(
                "Google Drive",
                "Conecte sua conta (ou selecione a pasta do Google Drive) primeiro antes de enviar a fila.",
            )
            return
        self.uploader.trigger_sync()
        self.post_log("Sincronização manual com o Google Drive solicitada.")

    def _enqueue_local_archive(self) -> None:
        added = self.uploader.enqueue_existing_folder(self.config.recording.recordings_dir)
        self._refresh_queue_tree()
        self.post_log(f"{added} arquivo(s) local(is) adicionado(s) à fila do Google Drive.")

    def _show_drive_help(self) -> None:
        self._open_drive_setup_wizard()

    def _refresh_drive_status_ui(self) -> None:
        mode = self.config.drive.sync_mode
        pending = self.uploader.get_pending_count()

        if mode == "folder":
            self.btn_drive_auth.configure(text="Ativar Pasta do Drive")
            folder_ready = self.uploader.is_folder_ready()
            if folder_ready and self.config.drive.enabled:
                self.drive_account_var.set(
                    f"Status: Pasta do Google Drive pronta ({self.config.drive.local_drive_dir})"
                )
                self.drive_badge.configure(text=f"Drive Pasta: Ativo (fila: {pending})", bg="#15803d", fg="#ffffff")
            elif folder_ready:
                self.drive_account_var.set(
                    f"Status: Pasta selecionada ({self.config.drive.local_drive_dir}) — marque 'Ativar envio' ou clique em 'Ativar Pasta do Drive'."
                )
                self.drive_badge.configure(text=f"Drive: Desativado ({pending})", bg="#374151", fg="#e5e7eb")
            else:
                self.drive_account_var.set(
                    "Status: Selecione a pasta do Google Drive para computador (ou clique em 'Detectar Meu Drive')."
                )
                self.drive_badge.configure(
                    text=f"Drive: Pasta ausente ({pending})" if self.config.drive.enabled else f"Drive: Desativado ({pending})",
                    bg="#b45309" if self.config.drive.enabled else "#374151",
                    fg="#ffffff" if self.config.drive.enabled else "#e5e7eb",
                )
            self.queue_title_var.set(f"Fila de Envios Pendentes ({pending})")
            return

        self.btn_drive_auth.configure(text="Conectar Conta Google Drive")
        has_creds = self.uploader.has_credentials_file()
        authed = self.uploader.has_saved_token()

        if authed:
            email = self.uploader.cached_account_email or "Sessão Google salva"
            self.drive_account_var.set(f"Status: Conectado ({email})")
        elif has_creds:
            self.drive_account_var.set("Status: credentials.json encontrado — clique em 'Conectar Conta Google Drive'.")
        else:
            self.drive_account_var.set(
                "Status: Aguardando credentials.json (clique em 'Conectar Conta Google Drive' para abrir o assistente)."
            )

        if self.config.drive.enabled and authed:
            self.drive_badge.configure(text=f"Drive: Ativo (fila: {pending})", bg="#15803d", fg="#ffffff")
        elif self.config.drive.enabled:
            self.drive_badge.configure(text=f"Drive: Aguardando login ({pending})", bg="#b45309", fg="#ffffff")
        else:
            self.drive_badge.configure(text=f"Drive: Desativado ({pending})", bg="#374151", fg="#e5e7eb")

        self.queue_title_var.set(f"Fila de Envios Pendentes ({pending})")

    def _refresh_queue_tree(self) -> None:
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        for job in self.uploader.get_pending_jobs():
            self.queue_tree.insert(
                "",
                tk.END,
                values=(
                    Path(job.local_path).name,
                    job.camera_name,
                    job.date_folder,
                    job.attempts,
                    job.last_error[:90],
                ),
            )
        self._refresh_drive_status_ui()

    # ------------------------------------------------------------------ #
    # Controle PTZ, Presets e Rastreamento Fixo                          #
    # ------------------------------------------------------------------ #

    def _build_ptz_sidebar(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent,
            bg="#222834",
            width=308,
            padx=10,
            pady=8,
            highlightbackground="#2d3748",
            highlightthickness=1,
        )
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        panel.pack_propagate(False)

        self.ptz_backend_var = tk.StringVar(value="Conecte a câmera para usar PTZ")
        self.ptz_speed_var = tk.StringVar(value="5")
        self.ptz_preset_var = tk.StringVar(value=f"{XM_WATCH_PRESET_ID} - Ponto Fixo")
        self.ptz_home_preset_var = tk.StringVar(value=str(XM_WATCH_PRESET_ID))
        self.ptz_fw_track_var = tk.BooleanVar(value=False)
        self.ptz_fw_return_var = tk.StringVar(value="30")
        self.ptz_fw_sens_var = tk.StringVar(value="Média")
        self.ptz_app_return_var = tk.BooleanVar(value=False)
        self.ptz_app_return_sec_var = tk.StringVar(value="30")
        self.ptz_feedback_var = tk.StringVar(value="Segure uma seta para mover ou defina o Ponto Fixo.")

        ttk.Label(panel, text="Controle PTZ & Ponto Fixo", style="Title.TLabel").pack(anchor=tk.W)
        tk.Label(
            panel,
            textvariable=self.ptz_backend_var,
            bg="#222834",
            fg="#93c5fd",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        ctrl_row = tk.Frame(panel, bg="#222834")
        ctrl_row.pack(fill=tk.X, pady=(0, 6))

        dpad_frame = tk.Frame(ctrl_row, bg="#222834")
        dpad_frame.pack(side=tk.LEFT)

        dpad_spec = [
            ("↖", "left_up", 0, 0),
            ("↑", "up", 0, 1),
            ("↗", "right_up", 0, 2),
            ("←", "left", 1, 0),
            ("⏹", "stop", 1, 1),
            ("→", "right", 1, 2),
            ("↙", "left_down", 2, 0),
            ("↓", "down", 2, 1),
            ("↘", "right_down", 2, 2),
        ]
        for label, direction, r, c in dpad_spec:
            if direction == "stop":
                btn = tk.Button(
                    dpad_frame,
                    text=label,
                    width=3,
                    bg="#991b1b",
                    fg="#ffffff",
                    activebackground="#dc2626",
                    activeforeground="#ffffff",
                    relief=tk.FLAT,
                    font=("Segoe UI", 10, "bold"),
                    command=self._ptz_force_stop,
                )
            else:
                btn = tk.Button(
                    dpad_frame,
                    text=label,
                    width=3,
                    bg="#334155",
                    fg="#f8fafc",
                    activebackground="#3b82f6",
                    activeforeground="#ffffff",
                    relief=tk.FLAT,
                    font=("Segoe UI", 10, "bold"),
                )
                btn.bind("<ButtonPress-1>", lambda _e, d=direction: self._on_ptz_press(d))
                btn.bind("<ButtonRelease-1>", lambda _e, d=direction: self._on_ptz_release(d))
            btn.grid(row=r, column=c, padx=2, pady=2)
            self._ptz_buttons.append(btn)

        side_ctrl = tk.Frame(ctrl_row, bg="#222834")
        side_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        for z_label, z_dir in (("🔍 Zoom +", "zoom_in"), ("🔎 Zoom -", "zoom_out")):
            z_btn = tk.Button(
                side_ctrl,
                text=z_label,
                bg="#334155",
                fg="#f8fafc",
                activebackground="#3b82f6",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                font=("Segoe UI", 9, "bold"),
                pady=3,
            )
            z_btn.bind("<ButtonPress-1>", lambda _e, d=z_dir: self._on_ptz_press(d))
            z_btn.bind("<ButtonRelease-1>", lambda _e, d=z_dir: self._on_ptz_release(d))
            z_btn.pack(fill=tk.X, pady=(0, 4))
            self._ptz_buttons.append(z_btn)

        spd_row = tk.Frame(side_ctrl, bg="#222834")
        spd_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(spd_row, text="Velocidade:", bg="#222834", fg="#cbd5e1", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        spd_spin = tk.Spinbox(
            spd_row,
            from_=1,
            to=8,
            width=4,
            textvariable=self.ptz_speed_var,
            command=self._save_ptz_prefs_from_ui,
        )
        spd_spin.pack(side=tk.RIGHT)
        spd_spin.bind("<FocusOut>", lambda _e: self._save_ptz_prefs_from_ui())

        home_box = ttk.LabelFrame(panel, text="Ponto Fixo & Presets", padding=4)
        home_box.pack(fill=tk.X, pady=(0, 4))

        home_btn_row = tk.Frame(home_box, bg="#222834")
        home_btn_row.pack(fill=tk.X, pady=(0, 3))

        btn_save_home = tk.Button(
            home_btn_row,
            text="📌 Definir Ponto Fixo",
            bg="#0f766e",
            fg="#ffffff",
            activebackground="#0d9488",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            pady=3,
            command=self._save_home_preset_clicked,
        )
        btn_save_home.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._ptz_buttons.append(btn_save_home)

        btn_go_home = tk.Button(
            home_btn_row,
            text="🏠 Ir p/ Ponto Fixo",
            bg="#1d4ed8",
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            pady=3,
            command=self._goto_home_preset_clicked,
        )
        btn_go_home.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        self._ptz_buttons.append(btn_go_home)

        fix_id_row = tk.Frame(home_box, bg="#222834")
        fix_id_row.pack(fill=tk.X, pady=(0, 3))
        tk.Label(
            fix_id_row,
            text="Nº Ponto Fixo (Vigia = 100):",
            bg="#222834",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)
        home_spin = tk.Spinbox(
            fix_id_row,
            from_=1,
            to=255,
            width=5,
            textvariable=self.ptz_home_preset_var,
            command=self._save_ptz_prefs_from_ui,
        )
        home_spin.pack(side=tk.RIGHT)
        home_spin.bind("<FocusOut>", lambda _e: self._save_ptz_prefs_from_ui())

        preset_row = tk.Frame(home_box, bg="#222834")
        preset_row.pack(fill=tk.X, pady=(0, 3))
        tk.Label(preset_row, text="Preset:", bg="#222834", fg="#cbd5e1", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.ptz_preset_combo = ttk.Combobox(preset_row, textvariable=self.ptz_preset_var, width=20)
        self.ptz_preset_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        self._update_preset_combo([])

        preset_act_row = tk.Frame(home_box, bg="#222834")
        preset_act_row.pack(fill=tk.X)
        for text, cmd, bg in (
            ("Ir", self._goto_selected_preset_clicked, "#334155"),
            ("Salvar", self._save_selected_preset_clicked, "#334155"),
            ("Apagar", self._clear_selected_preset_clicked, "#7f1d1d"),
            ("↻", self._refresh_ptz_status_clicked, "#334155"),
        ):
            pbtn = tk.Button(
                preset_act_row,
                text=text,
                bg=bg,
                fg="#f8fafc",
                activebackground="#475569",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
                pady=2,
                command=cmd,
            )
            pbtn.pack(side=tk.LEFT, fill=tk.X, expand=(text != "↻"), padx=1)
            self._ptz_buttons.append(pbtn)

        track_box = ttk.LabelFrame(panel, text="Retorno Fixo Pós-Rastreamento", padding=4)
        track_box.pack(fill=tk.X, pady=(0, 2))

        chk_fw = ttk.Checkbutton(
            track_box,
            text="Auto-rastrear na câmera (volta #100)",
            variable=self.ptz_fw_track_var,
        )
        chk_fw.pack(anchor=tk.W, pady=(0, 2))
        self._ptz_fw_widgets.append(chk_fw)

        fw_row = tk.Frame(track_box, bg="#222834")
        fw_row.pack(fill=tk.X, pady=(0, 3))
        tk.Label(fw_row, text="Voltar(s):", bg="#222834", fg="#cbd5e1", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        combo_fw_ret = ttk.Combobox(
            fw_row,
            textvariable=self.ptz_fw_return_var,
            values=("10", "15", "20", "30", "60", "120", "300"),
            width=4,
        )
        combo_fw_ret.pack(side=tk.LEFT, padx=(2, 4))
        self._ptz_fw_widgets.append(combo_fw_ret)

        self.ptz_fw_sens_combo = ttk.Combobox(
            fw_row,
            textvariable=self.ptz_fw_sens_var,
            values=("Baixa", "Média", "Alta"),
            state="readonly",
            width=6,
        )
        self.ptz_fw_sens_combo.pack(side=tk.LEFT, padx=(0, 4))

        btn_fw_apply = tk.Button(
            fw_row,
            text="Aplicar",
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2,
            command=self._apply_detect_track_clicked,
        )
        btn_fw_apply.pack(side=tk.RIGHT)
        self._ptz_fw_widgets.append(btn_fw_apply)

        app_ret_row = tk.Frame(track_box, bg="#222834")
        app_ret_row.pack(fill=tk.X)
        ttk.Checkbutton(
            app_ret_row,
            text="Voltar ao Fixo se ocioso (s):",
            variable=self.ptz_app_return_var,
            command=self._save_ptz_prefs_from_ui,
        ).pack(side=tk.LEFT)
        app_sec_spin = tk.Spinbox(
            app_ret_row,
            from_=5,
            to=3600,
            width=4,
            textvariable=self.ptz_app_return_sec_var,
            command=self._save_ptz_prefs_from_ui,
        )
        app_sec_spin.pack(side=tk.RIGHT)
        app_sec_spin.bind("<FocusOut>", lambda _e: self._save_ptz_prefs_from_ui())

        tk.Label(
            panel,
            textvariable=self.ptz_feedback_var,
            bg="#222834",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
            wraplength=284,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 0))

        self._set_ptz_controls_enabled(False, fw_supported=False)

    def _set_ptz_controls_enabled(self, enabled: bool, fw_supported: bool = False) -> None:
        self._ptz_controls_enabled = enabled
        self._ptz_fw_supported = enabled and fw_supported
        btn_state = tk.NORMAL if enabled else tk.DISABLED
        fw_state = tk.NORMAL if self._ptz_fw_supported else tk.DISABLED
        for btn in self._ptz_buttons:
            btn.configure(state=btn_state)
        for widget in self._ptz_fw_widgets:
            widget.configure(state=fw_state)
        self.ptz_fw_sens_combo.configure(state="readonly" if self._ptz_fw_supported else tk.DISABLED)

    def _get_ptz_speed(self) -> int:
        try:
            return max(1, min(8, int(self.ptz_speed_var.get().strip() or 5)))
        except ValueError:
            return 5

    def _get_home_preset_id(self) -> int:
        try:
            return max(1, min(255, int(self.ptz_home_preset_var.get().strip() or XM_WATCH_PRESET_ID)))
        except ValueError:
            return XM_WATCH_PRESET_ID

    def _get_app_return_seconds(self) -> int:
        try:
            return max(5, min(3600, int(self.ptz_app_return_sec_var.get().strip() or 30)))
        except ValueError:
            return 30

    @staticmethod
    def _parse_preset_id(raw_text: str) -> int | None:
        token = (raw_text or "").strip().split(" ", 1)[0].split("-", 1)[0].strip()
        if not token.isdigit():
            return None
        pid = int(token)
        return pid if 1 <= pid <= 255 else None

    def _load_ptz_prefs_from_camera(self, cam: CameraConfig) -> None:
        self.ptz_speed_var.set(str(max(1, min(8, int(cam.ptz_speed)))))
        self.ptz_home_preset_var.set(str(max(1, min(255, int(cam.ptz_home_preset)))))
        self.ptz_app_return_var.set(bool(cam.ptz_auto_return_enabled))
        self.ptz_app_return_sec_var.set(str(max(5, min(3600, int(cam.ptz_auto_return_seconds)))))

    def _save_ptz_prefs_from_ui(self) -> None:
        cam = self.config_manager.get_selected_camera(self.config)
        cam.ptz_speed = self._get_ptz_speed()
        cam.ptz_home_preset = self._get_home_preset_id()
        cam.ptz_auto_return_enabled = bool(self.ptz_app_return_var.get())
        cam.ptz_auto_return_seconds = self._get_app_return_seconds()
        self.config_manager.save(self.config)
        if not cam.ptz_auto_return_enabled:
            self._ptz_needs_home_return = False

    def _update_preset_combo(self, presets: list[PTZPresetItem]) -> None:
        self._ptz_presets = list(presets)
        by_id = {p.id: p for p in presets}
        labels: list[str] = []
        ordered_ids = sorted(by_id.keys(), key=lambda pid: (0 if pid == XM_WATCH_PRESET_ID else 1, pid))
        for pid in ordered_ids:
            name = by_id[pid].name
            if pid == XM_WATCH_PRESET_ID:
                tag = f"{name} (Vigia)" if name else "Ponto Fixo (Vigia)"
                labels.append(f"{pid} - {tag}")
            else:
                labels.append(f"{pid} - {name}" if name else f"{pid} - Ponto {pid}")
        for fallback_id in (XM_WATCH_PRESET_ID, 1, 2, 3, 4):
            if fallback_id not in by_id:
                suffix = "Ponto Fixo (Vigia)" if fallback_id == XM_WATCH_PRESET_ID else f"Ponto {fallback_id}"
                labels.append(f"{fallback_id} - {suffix} [novo]")
        self.ptz_preset_combo.configure(values=labels)
        current_id = self._parse_preset_id(self.ptz_preset_var.get())
        if current_id is None:
            current_id = self._get_home_preset_id()
        for item in labels:
            if self._parse_preset_id(item) == current_id:
                self.ptz_preset_var.set(item)
                break

    def _cancel_pending_ptz_stop(self) -> None:
        if self._ptz_stop_after_id is not None:
            try:
                self.root.after_cancel(self._ptz_stop_after_id)
            except tk.TclError:
                pass
            self._ptz_stop_after_id = None

    def _schedule_app_home_return(self) -> None:
        if bool(self.ptz_app_return_var.get()):
            self._ptz_needs_home_return = True
            self._ptz_return_deadline = time.monotonic() + float(self._get_app_return_seconds())

    def _on_ptz_press(self, direction: str) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        self._cancel_pending_ptz_stop()
        self._ptz_press_started_at = time.monotonic()
        self._ptz_holding = True
        self._ptz_active_dir = direction
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("move_start", (self._ptz_session_token, (direction, self._get_ptz_speed()))))
        self.ptz_feedback_var.set(f"Movendo ({direction})...")

    def _enqueue_ptz_stop(self, direction: str) -> None:
        self._ptz_stop_after_id = None
        if not self._is_connected:
            return
        self._ptz_queue.put(("move_stop", (self._ptz_session_token, direction)))
        self._ptz_active_dir = ""
        self._schedule_app_home_return()
        if self._ptz_needs_home_return:
            secs = self._get_app_return_seconds()
            self.ptz_feedback_var.set(f"Parado. Retorna ao Fixo em {secs}s se ocioso.")
        else:
            self.ptz_feedback_var.set("Movimento parado.")

    def _on_ptz_release(self, direction: str) -> None:
        if not self._is_connected or not self._ptz_controls_enabled or not self._ptz_holding:
            return
        self._ptz_holding = False
        elapsed_ms = (time.monotonic() - self._ptz_press_started_at) * 1000.0
        remaining_ms = max(0, int(180 - elapsed_ms))
        if remaining_ms > 0:
            self._ptz_stop_after_id = self.root.after(
                remaining_ms,
                lambda d=direction: self._enqueue_ptz_stop(d),
            )
        else:
            self._enqueue_ptz_stop(direction)

    def _ptz_force_stop(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        self._cancel_pending_ptz_stop()
        direction = self._ptz_active_dir
        self._ptz_holding = False
        self._ptz_active_dir = ""
        self._ptz_queue.put(("move_stop", (self._ptz_session_token, direction)))
        self.ptz_feedback_var.set("Comando de parada enviado.")

    def _save_home_preset_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        pid = self._get_home_preset_id()
        self._save_ptz_prefs_from_ui()
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("set_preset", (self._ptz_session_token, pid)))
        self.ptz_feedback_var.set(f"Salvando Ponto Fixo #{pid}...")

    def _goto_home_preset_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        pid = self._get_home_preset_id()
        self._save_ptz_prefs_from_ui()
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("goto_preset", (self._ptz_session_token, pid)))
        self.ptz_feedback_var.set(f"Retornando ao Ponto Fixo #{pid}...")

    def _goto_selected_preset_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        pid = self._parse_preset_id(self.ptz_preset_var.get())
        if pid is None:
            messagebox.showwarning("PTZ", "Informe um número de preset válido (1 a 255).")
            return
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("goto_preset", (self._ptz_session_token, pid)))
        self.ptz_feedback_var.set(f"Indo para preset #{pid}...")

    def _save_selected_preset_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        pid = self._parse_preset_id(self.ptz_preset_var.get())
        if pid is None:
            messagebox.showwarning("PTZ", "Informe um número de preset válido (1 a 255).")
            return
        self._ptz_needs_home_return = False
        self._ptz_queue.put(("set_preset", (self._ptz_session_token, pid)))
        self.ptz_feedback_var.set(f"Salvando posição no preset #{pid}...")

    def _clear_selected_preset_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        pid = self._parse_preset_id(self.ptz_preset_var.get())
        if pid is None:
            messagebox.showwarning("PTZ", "Informe um número de preset válido (1 a 255).")
            return
        self._ptz_queue.put(("clear_preset", (self._ptz_session_token, pid)))
        self.ptz_feedback_var.set(f"Removendo preset #{pid}...")

    def _refresh_ptz_status_clicked(self) -> None:
        if not self._is_connected or not self._ptz_controls_enabled:
            return
        self._ptz_queue.put(("refresh_status", (self._ptz_session_token, None)))
        self.ptz_feedback_var.set("Atualizando presets e rastreamento da câmera...")

    def _apply_detect_track_clicked(self) -> None:
        if not self._is_connected or not self._ptz_fw_supported:
            return
        enable = bool(self.ptz_fw_track_var.get())
        try:
            return_time = max(5, min(3600, int(self.ptz_fw_return_var.get().strip() or 30)))
        except ValueError:
            return_time = 30
            self.ptz_fw_return_var.set("30")
        sens_map = {"Baixa": 0, "Média": 1, "Alta": 2}
        sensitivity = sens_map.get(self.ptz_fw_sens_var.get(), 1)
        self._ptz_queue.put(("set_detect_track", (self._ptz_session_token, (enable, return_time, sensitivity))))
        self.ptz_feedback_var.set("Gravando configuração de auto-rastreamento na câmera...")

    def _ptz_idle_tick(self) -> None:
        try:
            if (
                self._is_connected
                and self._ptz_controls_enabled
                and not self._ptz_holding
                and self._ptz_needs_home_return
                and bool(self.ptz_app_return_var.get())
                and time.monotonic() >= self._ptz_return_deadline
            ):
                self._ptz_needs_home_return = False
                pid = self._get_home_preset_id()
                self._ptz_queue.put(("goto_preset", (self._ptz_session_token, pid)))
                self.ptz_feedback_var.set(f"Ocioso: retornando ao Ponto Fixo #{pid}...")
                self.post_log(f"[PTZ] Retorno automático ao Ponto Fixo #{pid} após inatividade.")
            self._update_live_recording_status()
        finally:
            if not self._ptz_worker_stop.is_set():
                self._ptz_idle_after_id = self.root.after(1000, self._ptz_idle_tick)

    def _ptz_worker_loop(self) -> None:
        worker_token = -1
        while not self._ptz_worker_stop.is_set():
            try:
                cmd, payload = self._ptz_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if cmd == "quit":
                break

            if cmd == "clear_backend":
                worker_token = -1
                if self._ptz_backend is not None:
                    try:
                        self._ptz_backend.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._ptz_backend = None
                continue

            if cmd == "init_backend":
                cam, source, token = payload  # type: ignore[misc]
                if self._ptz_backend is not None:
                    try:
                        self._ptz_backend.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._ptz_backend = None
                worker_token = int(token)
                try:
                    backend = create_ptz_backend(cam, source)
                    self._ptz_backend = backend
                    if backend is None:
                        self._ui_queue.put(("ptz_status_loaded", (worker_token, "", False, None)))
                        continue
                    snap = backend.get_status_snapshot()
                    self._ui_queue.put(
                        (
                            "ptz_status_loaded",
                            (worker_token, backend.backend_label, backend.supports_firmware_tracking, snap),
                        )
                    )
                except Exception as exc:
                    self._ui_queue.put(("ptz_status_loaded", (worker_token, "", False, None)))
                    self._ui_queue.put(("ptz_msg", f"PTZ indisponível: {exc}"))
                continue

            token, data = payload  # type: ignore[misc]
            if int(token) != worker_token or self._ptz_backend is None:
                continue

            backend = self._ptz_backend
            try:
                if cmd == "move_start":
                    direction, speed = data
                    ok = backend.move_start(direction, speed)  # type: ignore[attr-defined]
                    if not ok:
                        self._ui_queue.put(("ptz_msg", "Câmera recusou movimento PTZ."))
                elif cmd == "move_stop":
                    backend.move_stop(str(data or ""))  # type: ignore[attr-defined]
                elif cmd == "set_preset":
                    pid = int(data)
                    ok = backend.set_preset(pid)  # type: ignore[attr-defined]
                    presets = backend.get_presets()  # type: ignore[attr-defined]
                    self._ui_queue.put(("ptz_presets_updated", (worker_token, presets)))
                    msg = f"Ponto #{pid} salvo!" if ok else f"Não foi possível salvar ponto #{pid}."
                    self._ui_queue.put(("ptz_msg", msg))
                    self.post_log(f"[PTZ] {msg}")
                elif cmd == "goto_preset":
                    pid = int(data)
                    ok = backend.goto_preset(pid)  # type: ignore[attr-defined]
                    msg = f"Movido para o Ponto #{pid}." if ok else f"Ponto #{pid} ainda não foi gravado na câmera."
                    self._ui_queue.put(("ptz_msg", msg))
                elif cmd == "clear_preset":
                    pid = int(data)
                    ok = backend.clear_preset(pid)  # type: ignore[attr-defined]
                    presets = backend.get_presets()  # type: ignore[attr-defined]
                    self._ui_queue.put(("ptz_presets_updated", (worker_token, presets)))
                    msg = f"Preset #{pid} removido." if ok else f"Falha ao remover preset #{pid}."
                    self._ui_queue.put(("ptz_msg", msg))
                elif cmd == "refresh_status":
                    snap = backend.get_status_snapshot()  # type: ignore[attr-defined]
                    self._ui_queue.put(
                        (
                            "ptz_status_loaded",
                            (worker_token, backend.backend_label, backend.supports_firmware_tracking, snap),  # type: ignore[attr-defined]
                        )
                    )
                    self._ui_queue.put(("ptz_msg", "Status PTZ atualizado."))
                elif cmd == "set_detect_track":
                    enable, return_time, sensitivity = data
                    ok = backend.set_detect_track(enable, return_time, sensitivity)  # type: ignore[attr-defined]
                    dt = backend.get_detect_track()  # type: ignore[attr-defined]
                    self._ui_queue.put(("ptz_detect_track_updated", (worker_token, dt)))
                    state_txt = f"ATIVO (retorna em {dt.return_time}s)" if dt.enable else "DESATIVADO"
                    msg = (
                        f"Auto-rastreamento {state_txt}!"
                        if ok
                        else "Câmera não aceitou alteração de auto-rastreamento."
                    )
                    self._ui_queue.put(("ptz_msg", msg))
                    self.post_log(f"[PTZ] {msg}")
            except Exception as exc:
                self._ui_queue.put(("ptz_msg", f"Erro PTZ: {exc}"))

    # ------------------------------------------------------------------ #
    # Fila de Eventos da UI e Encerramento                               #
    # ------------------------------------------------------------------ #

    def _set_status_badge(self, text: str, bg: str) -> None:
        self.status_badge.configure(text=text, bg=bg)

    def _append_log_line(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_ui_queue(self) -> None:
        pb_dirty_frame = False
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self._append_log_line(str(payload))
                elif kind == "render_frame":
                    self._render_latest_frame()
                elif kind == "status_badge":
                    txt, color = payload  # type: ignore[misc]
                    self._set_status_badge(txt, color)
                elif kind == "status_badge_live":
                    if self.recorder.is_recording:
                        self._set_status_badge("Gravando", "#dc2626")
                    else:
                        self._set_status_badge("Ao Vivo", "#16a34a")
                elif kind == "motion_state":
                    if payload:
                        self.motion_indicator.configure(text="Movimento: DETECTADO!", bg="#b91c1c", fg="#ffffff")
                    else:
                        self.motion_indicator.configure(text="Movimento: parado", bg="#1e293b", fg="#94a3b8")
                elif kind == "discovery_done":
                    self.btn_scan_net.configure(state=tk.NORMAL)
                    self._discovered = list(payload)  # type: ignore[arg-type]
                    self.discovered_listbox.delete(0, tk.END)
                    for item in self._discovered:
                        self.discovered_listbox.insert(tk.END, item.label())
                    self.discovery_status_var.set(
                        f"{len(self._discovered)} câmera(s) encontrada(s). Selecione uma e clique em 'Usar Câmera Encontrada'."
                        if self._discovered
                        else "Nenhuma câmera encontrada automaticamente. Você ainda pode preencher o IP e testar manualmente."
                    )
                elif kind == "usb_discovery_done":
                    indices = list(payload)  # type: ignore[arg-type]
                    if indices:
                        self.form_brand_var.set("Webcam USB")
                        self.form_usb_var.set(str(indices[0]))
                        self.discovery_status_var.set(f"Webcam(s) USB detectada(s): {', '.join('#' + str(i) for i in indices)}.")
                    else:
                        self.discovery_status_var.set("Nenhuma webcam USB foi detectada neste computador.")
                elif kind == "onvif_uris":
                    uris = list(payload)  # type: ignore[arg-type]
                    if uris:
                        self.form_custom_url_var.set(uris[0])
                        self.camera_test_msg_var.set(
                            f"URL RTSP obtida via ONVIF ({len(uris)} fluxo(s) encontrado(s)): {mask_rtsp_url(uris[0])}"
                        )
                    else:
                        self.camera_test_msg_var.set(
                            "Não foi possível obter a URL via ONVIF. Confira IP, porta ONVIF (8899/80/2020), usuário e senha."
                        )
                elif kind == "serial_lookup_done":
                    self.btn_serial_lookup.configure(state=tk.NORMAL)
                    serial_query, found = payload  # type: ignore[misc]
                    if found is not None:
                        found_ip, info = found
                        model = str(info.get("DeviceModel") or "Xiongmai")
                        detected_serial = str(info.get("SerialNo") or serial_query)
                        self.form_ip_var.set(found_ip)
                        self.form_brand_var.set("Xiongmai NetIP / iCSee (Porta 34567)")
                        self.form_rtsp_port_var.set(str(XM_DEFAULT_PORT))
                        self.form_onvif_port_var.set(str(XM_DEFAULT_PORT))
                        self.form_serial_var.set(detected_serial)
                        self.form_custom_url_var.set("")
                        self._update_rtsp_preview_label()
                        self.camera_test_msg_var.set(
                            f"Câmera encontrada! IP {found_ip} (modelo {model}, serial {detected_serial}). "
                            "Clique em 'Testar Conexão' ou 'Salvar e Conectar Agora'."
                        )
                    else:
                        self.camera_test_msg_var.set(
                            "Nenhuma câmera com este N.º de série respondeu na porta 34567 da rede local. Confira usuário/senha e Wi-Fi."
                        )
                elif kind == "start_recording_resolved":
                    req_id, cam_cfg, best_src = payload  # type: ignore[misc]
                    self.btn_record.configure(state=tk.NORMAL)
                    if req_id == self._record_request_id and not self.recorder.is_recording:
                        self._apply_start_recording(cam_cfg, best_src)
                elif kind == "test_result":
                    self.btn_test_conn.configure(state=tk.NORMAL)
                    self.camera_test_msg_var.set(payload.message)  # type: ignore[attr-defined]
                    if payload.ok:  # type: ignore[attr-defined]
                        messagebox.showinfo("Teste de Câmera", payload.message)  # type: ignore[attr-defined]
                    else:
                        messagebox.showwarning("Teste de Câmera", payload.message)  # type: ignore[attr-defined]
                elif kind == "drive_auth_done":
                    self.btn_drive_auth.configure(state=tk.NORMAL)
                    ok, msg = payload  # type: ignore[misc]
                    if ok:
                        self.drive_enabled_var.set(True)
                        self._save_drive_settings()
                        messagebox.showinfo("Google Drive", msg)
                    else:
                        messagebox.showerror("Google Drive", msg)
                    self._refresh_drive_status_ui()
                elif kind == "drive_status_refresh":
                    self._refresh_drive_status_ui()
                elif kind == "ptz_status_loaded":
                    token, label, fw_flag, snap = payload  # type: ignore[misc]
                    if int(token) == self._ptz_session_token and self._is_connected:
                        if snap is None:
                            self._set_ptz_controls_enabled(False, fw_supported=False)
                            self.ptz_backend_var.set("Sem controle PTZ nesta câmera")
                            self.ptz_feedback_var.set("Câmera conectada apenas para vídeo.")
                        else:
                            assert isinstance(snap, PTZStatusSnapshot)
                            dt = snap.detect_track or DetectTrackSettings()
                            fw_ok = bool(fw_flag and dt.supported)
                            self._set_ptz_controls_enabled(bool(snap.supported_ptz), fw_supported=fw_ok)
                            self.ptz_backend_var.set(f"{label} — Pronto" if snap.supported_ptz else f"{label} (sem PTZ)")
                            self._update_preset_combo(snap.presets or [])
                            if fw_ok:
                                self.ptz_fw_track_var.set(bool(dt.enable))
                                self.ptz_fw_return_var.set(str(dt.return_time))
                                sens_label = {0: "Baixa", 1: "Média", 2: "Alta"}.get(dt.sensitivity, "Média")
                                self.ptz_fw_sens_var.set(sens_label)
                            n_presets = len(snap.presets or [])
                            self.ptz_feedback_var.set(
                                f"Pronto ({n_presets} preset(s)). Segure as setas ou use Ponto Fixo."
                            )
                elif kind == "ptz_presets_updated":
                    token, presets = payload  # type: ignore[misc]
                    if int(token) == self._ptz_session_token and self._is_connected:
                        self._update_preset_combo(list(presets))
                elif kind == "ptz_detect_track_updated":
                    token, dt = payload  # type: ignore[misc]
                    if int(token) == self._ptz_session_token and self._is_connected and isinstance(dt, DetectTrackSettings):
                        self.ptz_fw_track_var.set(bool(dt.enable))
                        self.ptz_fw_return_var.set(str(dt.return_time))
                        self.ptz_fw_sens_var.set({0: "Baixa", 1: "Média", 2: "Alta"}.get(dt.sensitivity, "Média"))
                elif kind == "ptz_msg":
                    self.ptz_feedback_var.set(str(payload))
                elif kind in ("queue_count", "refresh_queue"):
                    self._refresh_queue_tree()
                    self._update_live_recording_status()
                elif kind == "ai_ready":
                    ok, msg = payload  # type: ignore[misc]
                    self.btn_init_ai.configure(state=tk.NORMAL)
                    self.ai_status_var.set(str(msg))
                    if bool(ok) and self.config.ai.enabled:
                        self.ai_pipeline.start()
                    self._update_ai_button_state()
                    self._update_ai_banner(self.ai_pipeline.get_summary())
                elif kind == "ai_update":
                    if len(payload) == 3:  # type: ignore[arg-type]
                        summary, records, person_records = payload  # type: ignore[misc]
                    else:
                        summary, records = payload  # type: ignore[misc]
                        person_records = self.ai_pipeline.get_person_records()
                    assert isinstance(summary, AISummary)
                    self._update_ai_banner(summary)
                    self._refresh_ai_table(list(records), list(person_records))
                    cam_evt_name = self.config_manager.get_selected_camera(self.config).name
                    record_ai_summary_events(
                        self.dvr_events,
                        cam_evt_name,
                        summary,
                        list(person_records),
                    )
                elif kind == "pb_timeline_loaded":
                    cam_name, chosen_date, dates, segs, evs = payload  # type: ignore[misc]
                    self._pb_apply_loaded_timeline(
                        str(cam_name),
                        str(chosen_date),
                        list(dates),
                        list(segs),
                        list(evs),
                    )
                elif kind == "pb_timeline_error":
                    self._pb_loading_index = False
                    self.btn_pb_refresh.configure(state=tk.NORMAL)
                    self.pb_status_var.set(f"Erro ao carregar gravações: {payload}")
                elif kind == "pb_status":
                    self.pb_status_var.set(str(payload))
                elif kind == "pb_frame":
                    st = payload  # type: ignore[assignment]
                    assert isinstance(st, PlaybackFrameState)
                    self._pb_latest_state = st
                    if st.frame_bgr is None:
                        self._pb_latest_frame = None
                        self._pb_latest_photo = None
                        self.pb_video_label.configure(
                            image="",
                            text="Nenhuma gravação carregada para esta câmera/data.",
                        )
                    else:
                        self._pb_latest_frame = st.frame_bgr
                        if not self._pb_scrubbing:
                            self._pb_playhead_sec = st.day_sec
                            if self._pb_win_span < 86400.0 and (
                                st.day_sec < self._pb_win_start or st.day_sec > self._pb_win_start + self._pb_win_span
                            ):
                                self._pb_center_window_on_playhead()
                        pb_dirty_frame = True
                    self._pb_refresh_hud_label()
                elif kind == "ai_test_result":
                    if len(payload) == 6:  # type: ignore[arg-type]
                        annotated_bgr, count, filename, summary, records, person_records = payload  # type: ignore[misc]
                    else:
                        annotated_bgr, count, filename, summary, records = payload  # type: ignore[misc]
                        person_records = self.ai_pipeline.get_person_records()
                    self.btn_init_ai.configure(state=tk.NORMAL)
                    if self.ai_pipeline.is_ready:
                        self.ai_status_var.set(
                            f"Ativa ({self.ai_pipeline.get_summary().backend_name})"
                            if self.config.ai.enabled
                            else "Modelos prontos (pausada)"
                        )
                    self._update_ai_button_state()
                    self._update_ai_banner(summary)
                    self._refresh_ai_table(list(records), list(person_records))
                    self._show_ai_test_preview(annotated_bgr, int(count), str(filename))
        except queue.Empty:
            pass
        if pb_dirty_frame and not self._closing:
            self._pb_render_viewport()
            self._draw_playback_playhead()
        if not self._ptz_worker_stop.is_set():
            self._poll_after_id = self.root.after(40, self._poll_ui_queue)

    def _wait_pending_uploads_modal(self) -> None:
        """Exibe janela responsiva enquanto finaliza envios pendentes para o Google Drive."""
        if not (self.config.drive.enabled and self.config.drive.auto_upload and self.uploader.is_ready()):
            return
        pending = self.uploader.get_pending_count()
        if pending <= 0:
            return

        self.uploader.trigger_sync()
        cancel_wait = False
        modal = tk.Toplevel(self.root)
        modal.title("Google Drive")
        modal.configure(bg="#222834")
        modal.resizable(False, False)
        modal.transient(self.root)
        modal.grab_set()

        status_var = tk.StringVar(
            value=f"Finalizando envio de {pending} arquivo(s) para o Google Drive...\n"
            "Os vídeos restantes serão enviados agora com segurança."
        )
        tk.Label(
            modal,
            textvariable=status_var,
            bg="#222834",
            fg="#eef2f7",
            padx=18,
            pady=14,
            justify=tk.LEFT,
            font=("Segoe UI", 10),
        ).pack()

        def _cancel() -> None:
            nonlocal cancel_wait
            cancel_wait = True

        ttk.Button(
            modal,
            text="Fechar agora (enviar na próxima abertura)",
            command=_cancel,
        ).pack(pady=(0, 14))
        modal.protocol("WM_DELETE_WINDOW", _cancel)

        initial_attempts = sum(j.attempts for j in self.uploader.get_pending_jobs())
        deadline = time.monotonic() + 90.0
        try:
            while not cancel_wait and time.monotonic() < deadline:
                rem = self.uploader.get_pending_count()
                if rem <= 0 or not self.uploader.is_ready():
                    break
                cur_attempts = sum(j.attempts for j in self.uploader.get_pending_jobs())
                if cur_attempts - initial_attempts >= 2:
                    break
                status_var.set(
                    f"Enviando gravações para o Google Drive ({rem} restante(s))...\n"
                    "Aguarde alguns segundos ou clique abaixo para fechar agora."
                )
                try:
                    self.root.update()
                except tk.TclError:
                    break
                time.sleep(0.08)
        finally:
            try:
                modal.grab_release()
            except tk.TclError:
                pass
            try:
                modal.destroy()
            except tk.TclError:
                pass

    def on_close(self, wait_for_uploads: bool = True) -> None:
        """Encerra gravação, captura, worker PTZ e worker do Drive de forma limpa."""
        if self._closing:
            return
        self._closing = True
        self.disconnect_camera()
        self.ai_pipeline.stop()
        self.dvr_player.stop()
        if wait_for_uploads:
            self._wait_pending_uploads_modal()
        self._ptz_worker_stop.set()
        for aid in (self._poll_after_id, self._ptz_idle_after_id):
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except tk.TclError:
                    pass
        self._poll_after_id = None
        self._ptz_idle_after_id = None
        self._ptz_queue.put(("quit", None))
        if self._ptz_worker_thread.is_alive():
            self._ptz_worker_thread.join(timeout=1.5)
        self.uploader.stop_worker()
        self.root.destroy()
