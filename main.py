"""Ponto de entrada do aplicativo HomeCam Segurança (Desktop GUI e Servidor 24/7 Headless)."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
import threading

from config_manager import ConfigManager, DEFAULT_CONFIG_PATH


def run_gui(smoke_check: bool = False, config_path: str | Path | None = None) -> int:
    """Inicia a janela principal Desktop ou executa verificação rápida de inicialização."""
    import tkinter as tk
    from app_gui import HomeCamApp

    root = tk.Tk()
    if smoke_check:
        root.withdraw()
    cfg_mgr = ConfigManager(config_path) if config_path else None
    app = HomeCamApp(
        root,
        config_manager=cfg_mgr,
        allow_auto_connect=not smoke_check,
        enable_startup_tasks=not smoke_check,
    )
    if smoke_check:
        root.update_idletasks()
        app.on_close(wait_for_uploads=False)
        return 0
    root.mainloop()
    return 0


def _make_headless_logger(log_path: Path):
    log_lock = threading.Lock()

    def _emit(line: str) -> None:
        if sys.stdout is not None:
            try:
                print(line, flush=True)
            except Exception:
                pass
        with log_lock:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
                    rotated = log_path.with_suffix(".log.old")
                    rotated.unlink(missing_ok=True)
                    log_path.replace(rotated)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{line}\n")
            except OSError:
                pass

    return _emit


def run_headless(
    *,
    smoke_check: bool = False,
    config_path: str | Path | None = None,
    web_host: str | None = None,
    web_port: int | None = None,
    enable_web: bool = True,
    auto_record: bool = True,
) -> int:
    """Executa captura, gravação contínua, IA, eventos DVR, Drive e Portal Web sem interface gráfica."""
    from headless_nvr import HeadlessNVRService
    from web_portal import NVRWebServer

    cfg_mgr = ConfigManager(config_path or DEFAULT_CONFIG_PATH)
    cfg = cfg_mgr.load()
    log_file = Path(cfg.recording.recordings_dir) / f"headless_nvr_{datetime.now().strftime('%Y-%m')}.log"
    log_fn = (lambda _line: None) if smoke_check else _make_headless_logger(log_file)

    service = HeadlessNVRService(
        config_manager=cfg_mgr,
        auto_connect=not smoke_check,
        auto_record=auto_record and not smoke_check,
        enable_ai=False if smoke_check else None,
        on_log=log_fn,
    )

    server: NVRWebServer | None = None
    if enable_web and not smoke_check:
        bind_host = web_host or service.config.web.host or "0.0.0.0"
        bind_port = int(web_port if web_port is not None else (service.config.web.port or 8765))
        server = NVRWebServer(
            service,
            host=bind_host,
            port=bind_port,
            auth_token=service.config.web.auth_token,
        )

    stop_event = threading.Event()
    if not smoke_check:
        def _handle_signal(signum: int, _frame: object) -> None:
            service.post_log(f"[NVR 24/7] Sinal {signum} recebido; finalizando gravações...")
            stop_event.set()

        for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle_signal)
                except (ValueError, OSError):
                    pass

    service.start()
    if server is not None:
        actual_port = server.start()
        service.post_log(f"[Web] Portal remoto ativo em http://{server.host}:{actual_port}/")

    try:
        if smoke_check:
            return 0
        while not stop_event.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        service.stop(wait_uploads_sec=0.0 if smoke_check else 3.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HomeCam Segurança — Monitoramento, IA & NVR 24/7")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Inicializa e fecha imediatamente para validar ambiente.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa motor NVR 24/7 + Portal Web em background sem janela Tkinter.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("HOMECAM_CONFIG_PATH") or "",
        help="Caminho alternativo para o arquivo config.json.",
    )
    parser.add_argument(
        "--web-host",
        default="",
        help="Endereço de escuta do Portal Web (padrão: config.web.host ou 0.0.0.0).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="Porta HTTP do Portal Web remoto (padrão: 8765).",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Desativa o servidor HTTP do Portal Web no modo headless.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Não inicia gravação automática ao iniciar o modo headless.",
    )
    args = parser.parse_args(argv)
    cfg_path = args.config.strip() or None

    if args.headless:
        return run_headless(
            smoke_check=args.check,
            config_path=cfg_path,
            web_host=args.web_host.strip() or None,
            web_port=args.web_port,
            enable_web=not args.no_web,
            auto_record=not args.no_record,
        )
    return run_gui(smoke_check=args.check, config_path=cfg_path)


if __name__ == "__main__":
    raise SystemExit(main())
