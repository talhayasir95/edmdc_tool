# terminal_handler.py
from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import List, Optional, Any
import sys

from log import Logger
from registry import ObservableRegistry

HELP = """
Custom-Observables & Wildcards (nur für Custom-Features, NICHT für Gates):
  Pattern ist ein Produkt aus Buchstaben: a*b*c*...
  - unterschiedliche Buchstaben => unterschiedliche x-Indizes
  - wiederholte Buchstaben      => gleicher Index mehrfach (Quadrate/Potenzen)
  - Anzahl verschiedener Buchstaben <= nx

  Beispiele:
    x            -> x1, x2, ..., xN
    x*x          -> x1*x1, x2*x2, ...
    x*y          -> alle Paare x_i*x_j, i<j
    x*y*z        -> alle Tripel x_i*x_j*x_k, i<j<k
    x*x*y        -> (a,a,b) mit a<b
    a*b*c*d*e    -> allgemeine Form, solange #unique<=nx

Gates:
  gate <expr>    Nur mit expliziten Indizes (z.B. heaviside(x1-0.1), relu(x2), tanh(x3)).
                 Wildcards wie 'x*y' sind in Gates verboten.

Befehle:
# Observables / Features hinzufügen
  <expr>                                  | Custom-Ausdruck hinzufügen (validiert + dedup)
                                          | Beispiele:
                                          |   x1*x6
                                          |   x2*x1              (wird kanonisch als x1*x2 gespeichert)
                                          |   exp(sign(x3)*x1)
                                          |   x*y*z*a            (Wildcard)

# Flags
  include_state <0|1>                     | Zustand x in φ aufnehmen
  include_bias <0|1>                      | Bias-Term 1 in φ aufnehmen
  set_trajectory_horizon <k>              | Setzt den Trajektorienhorizont k

# Time Delay
  addtimedelay <n_delays> [tau]           | Time-Delay-Embedding hinzufügen (tau optional)
  rm_time_delay                           | Time-Delay entfernen
  time_delay_on_obs <0|1>                 | 1 => Delay auf Observables anwenden (apply_on_x=False)

# RBF
  addrbf <n_centers>                      | RBF aktivieren/setzen
  change_rbf_params <type|-} <kw|-} <ph|-}| '-' = nicht ändern
  rm_rbf                                  | RBF zurücksetzen
  disable_rbf                             | RBF deaktivieren (Konfig bleibt)
  enable_rbf                              | RBF aktivieren

# Gates
  gate <expr>                             | Gate hinzufügen
  list_gates                              | Gates anzeigen (mit Index)
  rm_gate <i|i..j|i,j,k>                  | Gate(s) löschen
  clear_gates                             | alle Gates löschen
  gates_as_features <0|1>                 | Gates zusätzlich als φ_x-Features
  disable_gates                           | Gates temporär deaktivieren
  enable_gates                            | Gates aktivieren

# EDMDc Parameter
  set_lambda_dyn <val>                    | Setzt λ_dyn
  set_lambda_dec <val>                    | Setzt λ_dec

# U Optionen (nur Config-Flags; keine u-Expressions in Observables)
  u_lift_raw_input <0|1>
  u_bilinear_zxu <0|1>

# Entfernen
  rm <i|i..j|i,j,k>                       | Custom-Expression(en) löschen

# Diagnose
  list_safe_funcs                         | listet erlaubte Funktionen
  info                                    | aktueller Config-Status

# Aktionen
  save <filename>                         | Ergebnisse speichern (TODO)
  undo                                    | letzte Änderung rückgängig
  redo                                    | redo
  clear                                   | komplette Konfiguration resetten
  quit / exit                             | Programm beenden
"""


def _require_args(tokens: List[str], n: int, usage: str) -> None:
    if len(tokens) < n:
        raise ValueError(f"Zu wenige Argumente. Usage: {usage}")


def _to_int(s: str, usage: str) -> int:
    try:
        return int(s)
    except Exception:
        raise ValueError(f"Ungültige Zahl: '{s}'. Usage: {usage}")


def _to_float(s: str, usage: str) -> float:
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Ungültige Zahl: '{s}'. Usage: {usage}")


def _to_bool01(s: str, usage: str) -> bool:
    if s in ("1", "true", "True"):
        return True
    if s in ("0", "false", "False"):
        return False
    raise ValueError(f"Ungültiger Wert: '{s}' (erwartet 0/1 oder true/false). Usage: {usage}")


@dataclass
class TerminalHandler:
    registry: ObservableRegistry
    logger: Optional[Any] = None

    def run(self) -> None:
        self._log(f"CLI gestartet. Config: {self.registry.filepath} (nx={self.registry.nx})")
        self._log("Tippe 'help' für Befehle.")

        while True:
            try:
                raw = input("> ").strip()
                if not raw:
                    continue

                tokens = shlex.split(raw)
                should_exit = self._dispatch(tokens=tokens, raw=raw)
                if should_exit:
                    break

            except (KeyboardInterrupt, EOFError):
                self._log("Beendet.")
                break
            except Exception as e:
                self._err(f"Fehler: {e}")

    def _log(self, msg: str) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(msg)
        else:
            print(msg)

    def _err(self, msg: str) -> None:
        if self.logger is not None and hasattr(self.logger, "error"):
            self.logger.error(msg)
        else:
            print(msg)

    def _dispatch(self, tokens: List[str], raw: str) -> bool:
        cmd = tokens[0]

        # help/quit
        if cmd in ("h", "-h", "--help", "help"):
            print(HELP)
            return False
        if cmd in ("quit", "exit"):
            return True

        match cmd:
            # -------------------------
            # Aktionen
            # -------------------------
            # case "run":
            #     self.registry.run_pipeline()
            #     return False

            case "save":
                _require_args(tokens, 2, "save <filename>")
                self.registry.save_results(tokens[1])
                return False

            case "undo":
                self.registry.undo()
                return False

            case "redo":
                self.registry.redo()
                return False

            case "clear":
                self.registry.clear()
                return False

            case "info":
                print(self.registry.info())
                return False

            # -------------------------
            # Diagnose
            # -------------------------
            case "list_safe_funcs":
                funcs = self.registry.list_safe_funcs()
                print("Safe funcs:", ", ".join(funcs))
                return False

            # -------------------------
            # RBF
            # -------------------------
            case "addrbf":
                _require_args(tokens, 2, "addrbf <n_centers>")
                n_centers = _to_int(tokens[1], "addrbf <n_centers>")
                self.registry.add_rbf(n_centers)
                return False

            case "change_rbf_params":
                _require_args(tokens, 4, "change_rbf_params <type|-} <kw|-} <ph|-}")
                self.registry.change_rbf_params(tokens[1], tokens[2], tokens[3])
                return False

            case "rm_rbf":
                self.registry.rm_rbf()
                return False

            case "disable_rbf":
                self.registry.disable_rbf()
                return False

            case "enable_rbf":
                self.registry.enable_rbf()
                return False

            # -------------------------
            # Time Delay
            # -------------------------
            case "addtimedelay":
                _require_args(tokens, 2, "addtimedelay <n_delays> [tau]")
                n_delays = _to_int(tokens[1], "addtimedelay <n_delays> [tau]")
                tau = _to_int(tokens[2], "addtimedelay <n_delays> [tau]") if len(tokens) >= 3 else None
                self.registry.add_time_delay(n_delays=n_delays, tau=tau)
                return False

            case "rm_time_delay":
                self.registry.rm_time_delay()
                return False

            case "time_delay_on_obs":
                _require_args(tokens, 2, "time_delay_on_obs <0|1>")
                val = _to_bool01(tokens[1], "time_delay_on_obs <0|1>")
                self.registry.time_delay_on_obs(val)
                return False

            case "set_trajectory_horizon":
                _require_args(tokens, 2, "set_trajectory_horizon <k>")
                k = _to_int(tokens[1], "set_trajectory_horizon <k>")
                self.registry.set_trajectory_horizon(k)
                return False

            # -------------------------
            # Gates
            # -------------------------
            case "gate":
                _require_args(tokens, 2, "gate <expr>")
                expr = raw[len("gate "):].strip()
                self.registry.add_gate(expr)
                return False

            case "list_gates":
                items = self.registry.list_gates()
                if not items:
                    print("(keine gates)")
                else:
                    for i, g in enumerate(items):
                        print(f"[{i}] {g}")
                return False

            case "rm_gate":
                _require_args(tokens, 2, "rm_gate <i|i..j|i,j,k>")
                self.registry.rm_gate(tokens[1])
                return False

            case "clear_gates":
                self.registry.clear_gates()
                return False

            case "gates_as_features":
                _require_args(tokens, 2, "gates_as_features <0|1>")
                val = _to_bool01(tokens[1], "gates_as_features <0|1>")
                self.registry.set_gates_as_features(val)
                return False

            case "disable_gates":
                self.registry.disable_gates()
                return False

            case "enable_gates":
                self.registry.enable_gates()
                return False

            # -------------------------
            # EDMDc Parameter
            # -------------------------
            case "set_lambda_dyn":
                _require_args(tokens, 2, "set_lambda_dyn <val>")
                val = _to_float(tokens[1], "set_lambda_dyn <val>")
                self.registry.set_lambda_dyn(val)
                return False

            case "set_lambda_dec":
                _require_args(tokens, 2, "set_lambda_dec <val>")
                val = _to_float(tokens[1], "set_lambda_dec <val>")
                self.registry.set_lambda_dec(val)
                return False

            # -------------------------
            # U Optionen
            # -------------------------
            case "u_lift_raw_input":
                _require_args(tokens, 2, "u_lift_raw_input <0|1>")
                val = _to_bool01(tokens[1], "u_lift_raw_input <0|1>")
                self.registry.set_u_lift_raw_input(val)
                return False

            case "u_bilinear_zxu":
                _require_args(tokens, 2, "u_bilinear_zxu <0|1>")
                val = _to_bool01(tokens[1], "u_bilinear_zxu <0|1>")
                self.registry.set_u_bilinear_zxu(val)
                return False

            # -------------------------
            # Flags
            # -------------------------
            case "include_state":
                _require_args(tokens, 2, "include_state <0|1>")
                val = _to_bool01(tokens[1], "include_state <0|1>")
                self.registry.set_include_state(val)
                return False

            case "include_bias":
                _require_args(tokens, 2, "include_bias <0|1>")
                val = _to_bool01(tokens[1], "include_bias <0|1>")
                self.registry.set_include_bias(val)
                return False

            # -------------------------
            # Custom entfernen
            # -------------------------
            case "rm":
                _require_args(tokens, 2, "rm <i|i..j|i,j,k>")
                self.registry.rm_custom(tokens[1])
                return False

            # -------------------------
            # Default: alles unbekannte => Custom Expr hinzufügen
            # -------------------------
            case _:
                self.registry.add_custom_expr(raw)
                return False


if __name__ == "__main__":
    # Start:
    #   python terminal_handler.py [config.json] [nx]
    cfg_path = sys.argv[1] if len(sys.argv) >= 2 else "config.json"
    nx = int(sys.argv[2]) if len(sys.argv) >= 3 else 2

    logger = Logger()  # wenn du keinen Logger willst: logger=None
    registry = ObservableRegistry(filepath=cfg_path, logger=logger, nx=nx)
    handler = TerminalHandler(registry=registry, logger=logger)
    handler.run()
