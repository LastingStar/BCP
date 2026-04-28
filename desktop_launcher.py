#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desktop launcher for Streamlit UI using pywebview.

Core requirements:
1. Reuse existing launcher entrypoint logic.
2. Source mode starts: `python launcher.py ui`.
3. Frozen mode starts: current_exe --launcher-ui-child (internally calls launcher.main(["ui"])).
4. Poll target localhost URL readiness before opening pywebview.
5. Do not swallow subprocess logs.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import ctypes
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Sequence
from urllib.parse import urlparse

import launcher as project_launcher


APP_TITLE = "无人机风暴环境突防测控中心"
DEFAULT_UI_URL = project_launcher.UI_URL
DEFAULT_WIDTH = 1500
DEFAULT_HEIGHT = 920
DEFAULT_STARTUP_TIMEOUT_S = 90.0
LOCAL_URL_PATTERN = re.compile(r"Local URL:\s*(http://\S+)", re.IGNORECASE)


def is_frozen() -> bool:
    """Return True when running from a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Resolve runtime project root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def configure_logging(debug: bool) -> None:
    """Configure logger."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(message)s")


def show_error_dialog(message: str, title: str = "Desktop Launcher Error") -> None:
    """Show Windows message box fallback to stderr."""
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def parse_url_endpoint(url: str) -> tuple[str, int]:
    """Parse host and port from target URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise ValueError(f"Cannot parse host from URL: {url}")

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, int(port)


def is_port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    """Return True if host:port accepts TCP connections."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


class ProcessOutputPump:
    """Print child output in real time and keep recent lines for failures."""

    def __init__(self, process: subprocess.Popen[str], tail_lines: int = 300) -> None:
        self.process = process
        self.tail: deque[str] = deque(maxlen=tail_lines)
        self.detected_local_url: str | None = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def tail_text(self) -> str:
        return "\n".join(self.tail) if self.tail else "(no output captured)"

    def _read_loop(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return

        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            self.tail.append(line)
            print(raw_line, end="")

            if self.detected_local_url is None:
                match = LOCAL_URL_PATTERN.search(line)
                if match:
                    self.detected_local_url = match.group(1)


def build_ui_command() -> list[str]:
    """
    Build UI child command.

    - Source mode: reuse `python launcher.py ui`.
    - Frozen mode: reuse launcher logic through child switch.
    """
    if is_frozen():
        return [sys.executable, "--launcher-ui-child"]
    return [sys.executable, "launcher.py", "ui"]


def start_ui_process() -> tuple[subprocess.Popen[str], ProcessOutputPump]:
    """Start UI subprocess and attach output pump."""
    cmd = build_ui_command()
    logging.info("Starting UI command: %s", " ".join(cmd))

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    process = subprocess.Popen(
        cmd,
        cwd=str(project_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    pump = ProcessOutputPump(process)
    pump.start()
    return process, pump


def wait_for_ui_ready(
    process: subprocess.Popen[str],
    pump: ProcessOutputPump,
    host: str,
    port: int,
    timeout_s: float,
) -> None:
    """Wait until UI endpoint is ready or process exits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_port_open(host, port):
            return

        if process.poll() is not None:
            raise RuntimeError(
                f"UI subprocess exited early with code {process.returncode}.\n"
                f"Recent logs:\n{pump.tail_text()}"
            )

        time.sleep(0.25)

    raise TimeoutError(
        f"Timed out waiting for UI endpoint {host}:{port}.\n"
        f"Recent logs:\n{pump.tail_text()}"
    )


def stop_ui_process(process: subprocess.Popen[str] | None, pump: ProcessOutputPump | None) -> None:
    """Stop UI subprocess and its child tree."""
    if process is None:
        return

    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()

    if pump is not None:
        pump.join(timeout=2)


def launch_webview(url: str, title: str, width: int, height: int, debug: bool) -> None:
    """Open URL with pywebview."""
    try:
        import webview
    except Exception as exc:
        raise RuntimeError("pywebview is not installed. Run: pip install pywebview") from exc

    webview.create_window(
        title=title,
        url=url,
        width=width,
        height=height,
        min_size=(1000, 700),
    )
    webview.start(debug=debug)


def run_launcher_ui_child() -> int:
    """Frozen child mode: call launcher main with ui mode."""
    old_argv = sys.argv[:]
    try:
        return int(project_launcher.main(["ui"]))
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 0 if exc.code is None else 1
    except Exception:
        logging.exception("launcher.main(['ui']) crashed in child mode.")
        return 2
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Desktop launcher based on launcher.py ui.")
    parser.add_argument("--ui-url", default=DEFAULT_UI_URL, help="Target Streamlit URL for pywebview.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Desktop window width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Desktop window height.")
    parser.add_argument("--title", default=APP_TITLE, help="Desktop window title.")
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_S, help="Startup timeout in seconds.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--launcher-ui-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_desktop(args: argparse.Namespace) -> int:
    """Desktop orchestration: start child UI, wait ready, open webview."""
    os.chdir(project_root())
    host, port = parse_url_endpoint(args.ui_url)

    process: subprocess.Popen[str] | None = None
    pump: ProcessOutputPump | None = None
    try:
        process, pump = start_ui_process()
        wait_for_ui_ready(process, pump, host, port, args.startup_timeout)

        if pump.detected_local_url and pump.detected_local_url != args.ui_url:
            logging.info("Detected local URL from streamlit logs: %s", pump.detected_local_url)

        logging.info("Opening desktop window: %s", args.ui_url)
        launch_webview(args.ui_url, args.title, args.width, args.height, args.debug)
        return 0
    except Exception:
        logging.exception("Desktop launcher failed.")
        if pump is not None:
            print("\n[DesktopLauncher] Recent child logs:")
            print(pump.tail_text())
        show_error_dialog("Desktop launcher failed. Check terminal logs for details.")
        return 1
    finally:
        stop_ui_process(process, pump)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(args.debug)

    if args.launcher_ui_child:
        os.chdir(project_root())
        return run_launcher_ui_child()

    return run_desktop(args)


if __name__ == "__main__":
    raise SystemExit(main())
