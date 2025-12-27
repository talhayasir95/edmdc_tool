# main.py
from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional

import requests


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _abs(p: str | Path) -> str:
    return str(Path(p).resolve())


def _wait_for_gui(gui_url: str, timeout_s: float = 20.0) -> bool:
    """
    Wartet, bis die GUI-API erreichbar ist.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{gui_url}/api/reload-flag", timeout=0.6)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _popen_gui(cmd: list[str], env: dict) -> subprocess.Popen:
    """
    Startet GUI in eigener Prozessgruppe (damit wir sauber killen können).
    """
    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            env=env,
            cwd=_abs(_project_root()),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    # posix
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=_abs(_project_root()),
        preexec_fn=os.setsid,
    )


def _popen_term(cmd: list[str], env: dict) -> subprocess.Popen:
    """
    Startet Terminal-Handler in eigener Prozessgruppe.
    """
    if os.name == "nt":
        return subprocess.Popen(
            cmd,
            env=env,
            cwd=_abs(_project_root()),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=_abs(_project_root()),
        preexec_fn=os.setsid,
    )


def _terminate_process(p: Optional[subprocess.Popen]) -> None:
    if p is None:
        return
    if p.poll() is not None:
        return

    try:
        if os.name == "nt":
            # versucht "soft" zu beenden
            p.terminate()
        else:
            # kill ganze Prozessgruppe
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def main() -> int:
    # ----------------------------
    # 0) Workdir fixen (wichtig für relative Pfade in app.py)
    # ----------------------------
    os.chdir(_project_root())

    # ----------------------------
    # 1) Gemeinsame Config-Datei
    #    app.py nutzt OBS_CFG_PATH (default: config.json)
    # ----------------------------
    cfg_path = os.environ.get("OBS_CFG_PATH", "config.json")
    cfg_path = _abs(cfg_path)

    # ----------------------------
    # 2) GUI Setup
    # ----------------------------
    port = int(os.environ.get("DASH_PORT", "8050"))
    gui_url = f"http://127.0.0.1:{port}"
    notify_url = f"{gui_url}/api/reload"

    # ----------------------------
    # 3) nx für CLI (terminal_handler.py nimmt argv[2], default=2)
    # ----------------------------
    nx = int(os.environ.get("OBS_NX", "2"))

    # ----------------------------
    # 4) ENV vorbereiten
    # ----------------------------
    env = os.environ.copy()
    env["OBS_NOTIFY_URL"] = notify_url
    env["OBS_CFG_PATH"] = cfg_path
    env["EDMDC_GUI_URL"] = gui_url

    # ----------------------------
    # 5) Commands
    # ----------------------------
    py = sys.executable

    gui_cmd = [py, "-u", "app.py"]  # Dash GUI
    term_cmd = [py, "-u", "terminal_handler.py", cfg_path, str(nx)]  # CLI

    gui_proc: Optional[subprocess.Popen] = None
    term_proc: Optional[subprocess.Popen] = None

    try:
        gui_proc = _popen_gui(gui_cmd, env=env)
        print(f"[main] GUI gestartet (pid={gui_proc.pid}) -> {gui_url}")

        ok = _wait_for_gui(gui_url)
        if not ok:
            print("[main] WARN: GUI-API nicht erreichbar (trotz Start). Starte CLI trotzdem…")

        term_proc = _popen_term(term_cmd, env=env)
        print(f"[main] Terminal-Handler gestartet (pid={term_proc.pid})")
        print("[main] Hinweis: Beenden mit STRG+C (stoppt beide Prozesse).")

        # ----------------------------
        # 6) Watchdog: wenn einer stirbt, beende den anderen
        # ----------------------------
        while True:
            time.sleep(0.3)
            gui_dead = (gui_proc.poll() is not None) if gui_proc else True
            term_dead = (term_proc.poll() is not None) if term_proc else True

            if gui_dead or term_dead:
                break

        if gui_proc and gui_proc.poll() is not None:
            print(f"[main] GUI beendet (code={gui_proc.returncode}).")
        if term_proc and term_proc.poll() is not None:
            print(f"[main] CLI beendet (code={term_proc.returncode}).")

        return 0

    except KeyboardInterrupt:
        print("\n[main] STRG+C -> beende Prozesse…")
        return 0

    finally:
        _terminate_process(term_proc)
        _terminate_process(gui_proc)

        t0 = time.time()
        while time.time() - t0 < 2.0:
            if (term_proc is None or term_proc.poll() is not None) and (gui_proc is None or gui_proc.poll() is not None):
                break
            time.sleep(0.1)

        for p in (term_proc, gui_proc):
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
