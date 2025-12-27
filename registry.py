from __future__ import annotations

"""registry.py

Die CLI (terminal_handler.py) erwartet eine ObservableRegistry, die
  - eine JSON-Config verwaltet (ConfigStore)
  - Expressions/Wildcards validiert
  - Änderungen per Undo/Redo erlaubt
  - optional eine Dash-GUI per HTTP "reload" anstößt

Im ursprünglichen Projekt gab es hierfür mehr Infrastruktur. Für das
hier vorliegende Dateiset implementiert dieses Modul eine schlanke,
robuste Variante, die mit app.py + terminal_handler.py zusammen läuft.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config_store import ConfigStore
from log import Logger
from parser import (
    SAFE_FUNCS,
    ExpressionParserXOnly,
    add_dedup,
    compute_signatures,
    extract_placeholders,
    expand_expression_wildcards,
)


def _deepcopy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # JSON round-trip ist hier ausreichend (nur primitive Typen).
    return json.loads(json.dumps(cfg))


def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_index_spec(spec: str, *, max_len: int) -> List[int]:
    """Parst "3", "1..4" oder "0,2,5" zu einer Indexliste."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Leere Index-Angabe")

    out: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ".." in p:
            a_s, b_s = [x.strip() for x in p.split("..", 1)]
            a = int(a_s)
            b = int(b_s)
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))

    # unique preserving order
    seen = set()
    uniq: List[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)

    # bounds
    for i in uniq:
        if i < 0 or i >= max_len:
            raise ValueError(f"Index {i} außerhalb [0,{max_len-1}]")
    return uniq


@dataclass
class ObservableRegistry:
    filepath: str
    nx: int
    logger: Optional[Any] = None

    store: ConfigStore = field(init=False)
    parser: ExpressionParserXOnly = field(default_factory=ExpressionParserXOnly)

    _undo: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _redo: List[Dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.store = ConfigStore(self.filepath)
        self.store.ensure_dicts()
        self._sync_signatures()
        self.store.save()

    # -----------------------
    # Logging/Notify
    # -----------------------
    def _log(self, msg: str) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(msg)
        else:
            print(msg)

    def _warn(self, msg: str) -> None:
        if self.logger is not None and hasattr(self.logger, "warn"):
            self.logger.warn(msg)
        else:
            print(msg)

    def _push_undo(self) -> None:
        self._undo.append(_deepcopy_config(self.store.config))
        self._redo.clear()

    def _notify_gui(self) -> None:
        """Triggert optional die GUI (wenn OBS_NOTIFY_URL gesetzt ist)."""
        url = os.environ.get("OBS_NOTIFY_URL", "").strip()
        if not url:
            return
        try:
            import requests

            requests.post(url, json={"signature": self.signature()}, timeout=0.4)
        except Exception as e:
            self._warn(f"[registry] GUI notify fehlgeschlagen: {e}")

    # -----------------------
    # Signaturen
    # -----------------------
    def signature(self) -> str:
        """Stabile Signatur für die GUI (relevant: Observables + EDMDc Params)."""
        cfg = self.store.config
        relevant = {
            "include_state": cfg.get("include_state"),
            "include_bias": cfg.get("include_bias"),
            "time_delay": cfg.get("time_delay"),
            "rbf": cfg.get("rbf"),
            "custom_features": cfg.get("custom_features"),
            "gates": cfg.get("gates"),
            "trajectory_horizon": cfg.get("trajectory_horizon"),
            "edmdc_params": cfg.get("edmdc_params"),
            "lift_raw_input": cfg.get("lift_raw_input"),
            "bilinear_zxu": cfg.get("bilinear_zxu"),
        }
        return _sha256_json(relevant)

    def _sync_signatures(self) -> None:
        feats: List[str] = self.store.config.get("custom_features", []) or []
        self.store.config["custom_features_signatures"] = compute_signatures(feats)

    # -----------------------
    # Public API: Info
    # -----------------------
    def list_safe_funcs(self) -> List[str]:
        return sorted(SAFE_FUNCS.keys())

    def info(self) -> str:
        cfg = self.store.config
        n_custom = len(cfg.get("custom_features", []) or [])
        n_gates = len((cfg.get("gates", {}) or {}).get("items", []) or [])
        return (
            f"Config: {self.filepath}\n"
            f"nx={self.nx}\n"
            f"include_state={cfg.get('include_state')}\n"
            f"include_bias={cfg.get('include_bias')}\n"
            f"time_delay={cfg.get('time_delay')}\n"
            f"rbf={cfg.get('rbf')}\n"
            f"custom_features={n_custom}\n"
            f"gates={n_gates} (enabled={cfg.get('gates',{}).get('enabled')}, gates_as_features={cfg.get('gates',{}).get('gates_as_features')})\n"
            f"trajectory_horizon={cfg.get('trajectory_horizon')}\n"
            f"edmdc_params={cfg.get('edmdc_params')}\n"
            f"lift_raw_input={cfg.get('lift_raw_input')}\n"
            f"bilinear_zxu={cfg.get('bilinear_zxu')}\n"
            f"signature={self.signature()}"
        )

    # -----------------------
    # Undo/Redo/Clear
    # -----------------------
    def undo(self) -> None:
        if not self._undo:
            self._warn("[registry] nichts zum Undo")
            return
        self._redo.append(_deepcopy_config(self.store.config))
        self.store.config = self._undo.pop()
        self.store.ensure_dicts()
        self.store.save()
        self._notify_gui()

    def redo(self) -> None:
        if not self._redo:
            self._warn("[registry] nichts zum Redo")
            return
        self._undo.append(_deepcopy_config(self.store.config))
        self.store.config = self._redo.pop()
        self.store.ensure_dicts()
        self.store.save()
        self._notify_gui()

    def clear(self) -> None:
        self._push_undo()
        self.store.clear()
        self._sync_signatures()
        self.store.save()
        self._notify_gui()

    # -----------------------
    # Flags
    # -----------------------
    def set_include_state(self, val: bool) -> None:
        self._push_undo()
        self.store.config["include_state"] = bool(val)
        self.store.save()
        self._notify_gui()

    def set_include_bias(self, val: bool) -> None:
        self._push_undo()
        self.store.config["include_bias"] = bool(val)
        self.store.save()
        self._notify_gui()

    def set_trajectory_horizon(self, k: int) -> None:
        self._push_undo()
        self.store.config["trajectory_horizon"] = int(k)
        self.store.save()
        self._notify_gui()

    def set_u_lift_raw_input(self, val: bool) -> None:
        self._push_undo()
        self.store.config["lift_raw_input"] = bool(val)
        self.store.save()
        self._notify_gui()

    def set_u_bilinear_zxu(self, val: bool) -> None:
        self._push_undo()
        self.store.config["bilinear_zxu"] = bool(val)
        self.store.save()
        self._notify_gui()

    # -----------------------
    # EDMDc params
    # -----------------------
    def set_lambda_dyn(self, val: float) -> None:
        self._push_undo()
        self.store.config["edmdc_params"]["lambda_dyn"] = float(val)
        self.store.save()
        self._notify_gui()

    def set_lambda_dec(self, val: float) -> None:
        self._push_undo()
        self.store.config["edmdc_params"]["lambda_dec"] = float(val)
        self.store.save()
        self._notify_gui()

    # -----------------------
    # Time Delay
    # -----------------------
    def add_time_delay(self, *, n_delays: int, tau: Optional[int] = None) -> None:
        self._push_undo()
        td = self.store.config.get("time_delay", {})
        td["enabled"] = True
        td["n_delays"] = int(n_delays)
        if tau is not None:
            td["tau"] = int(tau)
        self.store.config["time_delay"] = td
        self.store.save()
        self._notify_gui()

    def rm_time_delay(self) -> None:
        self._push_undo()
        td = self.store.config.get("time_delay", {})
        td["enabled"] = False
        td["n_delays"] = 0
        td["tau"] = 1
        self.store.config["time_delay"] = td
        self.store.save()
        self._notify_gui()

    def time_delay_on_obs(self, val: bool) -> None:
        self._push_undo()
        td = self.store.config.get("time_delay", {})
        # CLI semantics: 1 => Delay auf Observables anwenden (apply_on_x=False)
        td["apply_on_x"] = (not bool(val))
        td["enabled"] = True
        self.store.config["time_delay"] = td
        self.store.save()
        self._notify_gui()

    # -----------------------
    # RBF
    # -----------------------
    def add_rbf(self, n_centers: int) -> None:
        """Aktiviert RBF und setzt Centers via KMeans auf Trainingsdaten."""
        self._push_undo()

        from data_provider import DataProvider

        provider = DataProvider(nx=self.nx)
        X = provider.X

        # robuste Grenzen
        n_centers = int(max(1, n_centers))
        n_centers = int(min(n_centers, max(1, X.shape[0] // 10)))

        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_centers, n_init="auto", random_state=0)
            km.fit(X)
            centers = km.cluster_centers_
        except Exception:
            # fallback: random samples
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=n_centers, replace=False)
            centers = X[idx]

        rbf = self.store.config.get("rbf", {})
        rbf["enabled"] = True
        rbf["n_centers"] = int(n_centers)
        rbf.setdefault("type", "gauss")
        rbf.setdefault("kernel_width", 1.0)
        rbf.setdefault("polyharmonic_coeff", 1.0)
        rbf["centers"] = np.asarray(centers, dtype=float).tolist()
        self.store.config["rbf"] = rbf

        self.store.save()
        self._notify_gui()

    def change_rbf_params(self, type_s: str, kw_s: str, ph_s: str) -> None:
        self._push_undo()
        rbf = self.store.config.get("rbf", {})
        if type_s != "-":
            rbf["type"] = str(type_s)
        if kw_s != "-":
            rbf["kernel_width"] = float(kw_s)
        if ph_s != "-":
            rbf["polyharmonic_coeff"] = float(ph_s)
        self.store.config["rbf"] = rbf
        self.store.save()
        self._notify_gui()

    def rm_rbf(self) -> None:
        self._push_undo()
        rbf = self.store.config.get("rbf", {})
        rbf["enabled"] = False
        rbf["n_centers"] = 0
        rbf["centers"] = []
        self.store.config["rbf"] = rbf
        self.store.save()
        self._notify_gui()

    def disable_rbf(self) -> None:
        self._push_undo()
        rbf = self.store.config.get("rbf", {})
        rbf["enabled"] = False
        self.store.config["rbf"] = rbf
        self.store.save()
        self._notify_gui()

    def enable_rbf(self) -> None:
        self._push_undo()
        rbf = self.store.config.get("rbf", {})
        if int(rbf.get("n_centers", 0)) <= 0:
            raise ValueError("rbf.n_centers ist 0. Nutze zuerst 'addrbf <n_centers>'.")
        if not (rbf.get("centers") or []):
            raise ValueError("rbf.centers ist leer. Nutze zuerst 'addrbf <n_centers>'.")
        rbf["enabled"] = True
        self.store.config["rbf"] = rbf
        self.store.save()
        self._notify_gui()

    # -----------------------
    # Gates
    # -----------------------
    def add_gate(self, expr: str) -> None:
        expr = (expr or "").strip()
        if not expr:
            raise ValueError("Leerer Gate-Ausdruck")
        # Wildcards in Gates verbieten
        if extract_placeholders(expr):
            raise ValueError("Gates dürfen keine Wildcards (einzelne Buchstaben) enthalten. Nutze x1..xN.")
        self.parser.validate(expr, nx=self.nx)

        self._push_undo()
        gcfg = self.store.config.get("gates", {})
        gcfg["enabled"] = True
        items: List[str] = gcfg.get("items", []) or []
        if expr not in items:
            items.append(expr)
        gcfg["items"] = items
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    def list_gates(self) -> List[str]:
        gcfg = self.store.config.get("gates", {})
        return list(gcfg.get("items", []) or [])

    def rm_gate(self, spec: str) -> None:
        gcfg = self.store.config.get("gates", {})
        items: List[str] = gcfg.get("items", []) or []
        if not items:
            return
        idxs = _parse_index_spec(spec, max_len=len(items))

        self._push_undo()
        for i in sorted(idxs, reverse=True):
            items.pop(i)
        gcfg["items"] = items
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    def clear_gates(self) -> None:
        self._push_undo()
        gcfg = self.store.config.get("gates", {})
        gcfg["items"] = []
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    def set_gates_as_features(self, val: bool) -> None:
        self._push_undo()
        gcfg = self.store.config.get("gates", {})
        gcfg["gates_as_features"] = bool(val)
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    def disable_gates(self) -> None:
        self._push_undo()
        gcfg = self.store.config.get("gates", {})
        gcfg["enabled"] = False
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    def enable_gates(self) -> None:
        self._push_undo()
        gcfg = self.store.config.get("gates", {})
        gcfg["enabled"] = True
        self.store.config["gates"] = gcfg
        self.store.save()
        self._notify_gui()

    # -----------------------
    # Custom Features
    # -----------------------
    def add_custom_expr(self, expr: str) -> Tuple[int, int]:
        expr = (expr or "").strip()
        if not expr:
            raise ValueError("Leerer Ausdruck")

        # expand wildcards (einzelne Buchstaben, z.B. x*y*z*a) – erlaubt
        expanded = expand_expression_wildcards(expr, nx=self.nx)

        # validate (x-only) und dedup
        for e in expanded:
            self.parser.validate(e, nx=self.nx)

        self._push_undo()
        feats: List[str] = self.store.config.get("custom_features", []) or []
        added, skipped = add_dedup(feats, expanded)
        self.store.config["custom_features"] = feats
        self._sync_signatures()
        self.store.save()
        self._notify_gui()
        return added, skipped

    def rm_custom(self, spec: str) -> None:
        feats: List[str] = self.store.config.get("custom_features", []) or []
        if not feats:
            return
        idxs = _parse_index_spec(spec, max_len=len(feats))
        self._push_undo()
        for i in sorted(idxs, reverse=True):
            feats.pop(i)
        self.store.config["custom_features"] = feats
        self._sync_signatures()
        self.store.save()
        self._notify_gui()

    # -----------------------
    # Pipeline / Results
    # -----------------------
    # def run_pipeline(self) -> None:
    #     """Startet die Pipeline.
    #
    #     In dieser minimalen Repo-Variante bedeutet das:
    #       - Config speichern
    #       - GUI per /api/reload anstoßen (damit diese automatisch neu fitten kann)
    #
    #     (Das eigentliche Fitten passiert in app.py.)
    #     """
    #     self.store.ensure_dicts()
    #     self._sync_signatures()
    #     self.store.save()
    #     self._notify_gui()
    #     self._log("[registry] run_pipeline: config gespeichert + GUI reload getriggert")

    def save_results(self, filename: str) -> None:
        """Speichert aktuell nur die Config als JSON (robuster Default)."""
        filename = (filename or "").strip()
        if not filename:
            raise ValueError("save <filename> benötigt einen Dateinamen")
        if not filename.lower().endswith(".json"):
            filename += ".json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.store.config, f, indent=2, ensure_ascii=False)
        self._log(f"[registry] gespeichert: {filename}")
