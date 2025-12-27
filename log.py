# log.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import threading
import sys


@dataclass
class Logger:
    """
    Minimaler Logger für CLI/Tooling.
    - schreibt nach stdout (info/debug) und stderr (warn/error)
    - optional zusätzlich in eine Datei
    """
    level: str = "INFO"               # "DEBUG" | "INFO" | "WARN" | "ERROR"
    logfile: Optional[str] = None     # z.B. "tool.log"
    show_timestamp: bool = True

    def __post_init__(self):
        self._lock = threading.Lock()
        self._level_order = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
        self.level = self.level.upper().strip()
        if self.level not in self._level_order:
            self.level = "INFO"

    # ---------------------
    # Public API
    # ---------------------
    def debug(self, message: str) -> None:
        self._log("DEBUG", message)

    def info(self, message: str) -> None:
        self._log("INFO", message)

    def warn(self, message: str) -> None:
        self._log("WARN", message)

    def error(self, message: str) -> None:
        self._log("ERROR", message)

    # ---------------------
    # Internalsy
    # ---------------------
    def _log(self, lvl: str, message: str) -> None:
        lvl = lvl.upper().strip()
        if self._level_order[lvl] < self._level_order[self.level]:
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.show_timestamp else ""
        prefix = f"[{ts}][{lvl}]" if ts else f"[{lvl}]"
        line = f"{prefix} {message}".rstrip()

        # WARN/ERROR -> stderr, sonst stdout
        stream = sys.stderr if lvl in ("WARN", "ERROR") else sys.stdout

        with self._lock:
            print(line, file=stream, flush=True)
            if self.logfile:
                try:
                    with open(self.logfile, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    # falls logfile nicht geschrieben werden kann, nicht crashen
                    print(f"[WARN] Konnte nicht in logfile schreiben: {self.logfile}", file=sys.stderr, flush=True)
