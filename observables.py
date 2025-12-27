# features/observables.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable

import numpy as np

from parser import ExpressionParserXOnly


@dataclass
class Observables:
    """
    Observables-Builder passend zur aktuellen config.json:

    Supported config keys:
      - include_state: bool
      - include_bias: bool
      - time_delay: { enabled: bool, apply_on_x: bool, n_delays: int, tau: int }
      - rbf: { enabled: bool, type: str, n_centers: int, kernel_width: float, polyharmonic_coeff: float, centers: [] }
      - custom_features: [ "expr", ... ]   (x-only, z.B. "sin(x1)*x2")
      - gates: { enabled: bool, gates_as_features: bool, items: [ "expr", ... ] }

    Everything else in config is ignored here (edmdc_params, trajectory_horizon, ...).
    """
    config: Dict[str, Any]
    n_input_features: int
    parser: ExpressionParserXOnly = field(default_factory=ExpressionParserXOnly)

    n_output_features: int = field(init=False, default=0)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def calculate_output_features(self) -> int:
        """
        Berechnet die Feature-Anzahl (Spalten) von phi(X) gemäß config.

        Hinweis: Die Zeilenanzahl kann durch Time-Delay reduziert werden.
        """
        include_state = bool(self.config.get("include_state", True))
        include_bias = bool(self.config.get("include_bias", False))

        td = self.config.get("time_delay", {}) or {}
        td_enabled = bool(td.get("enabled", False))
        apply_on_x = bool(td.get("apply_on_x", True))
        n_delays = int(td.get("n_delays", 0))

        # 1) Basis (State)
        n = 0
        if include_state:
            if td_enabled and apply_on_x:
                n += self.n_input_features * (n_delays + 1)
            else:
                n += self.n_input_features

        # 2) RBF
        rbf = self.config.get("rbf", {}) or {}
        if bool(rbf.get("enabled", False)) and int(rbf.get("n_centers", 0)) > 0:
            n += int(rbf.get("n_centers", 0))

        # 3) Custom features
        custom = self.config.get("custom_features", []) or []
        n += len(custom)

        # 4) Bias
        if include_bias:
            n += 1

        # 5) Gates
        gcfg = self.config.get("gates", {}) or {}
        if bool(gcfg.get("enabled", False)):
            gates = gcfg.get("items", []) or []
            n_g = len(gates)

            if n_g > 0 and n > 0:
                # original phi + gated copies pro gate
                n = n * (1 + n_g)

            if bool(gcfg.get("gates_as_features", False)):
                n += n_g

        # 6) Time-delay auf z (wenn nicht apply_on_x)
        if td_enabled and not apply_on_x and n > 0:
            n *= (n_delays + 1)

        return int(n)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Baut phi(X) gemäß config.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X muss 2D sein (N, nx). Bekommen: {X.shape}")
        N, nx = X.shape
        if nx != self.n_input_features:
            raise ValueError(
                f"X hat nx={nx}, erwartet n_input_features={self.n_input_features}."
            )

        include_state = bool(self.config.get("include_state", True))
        include_bias = bool(self.config.get("include_bias", False))

        td = self.config.get("time_delay", {}) or {}
        td_enabled = bool(td.get("enabled", False))
        apply_on_x = bool(td.get("apply_on_x", True))
        n_delays = int(td.get("n_delays", 0))
        tau = int(td.get("tau", 1))

        delay_width = n_delays * tau

        # ------------------------------------------------------------
        # 1) Start: State (ggf. Time-Delay auf x)
        # ------------------------------------------------------------
        if td_enabled and apply_on_x:
            if N <= delay_width:
                raise ValueError(
                    f"N={N} <= n_delays*tau={delay_width}. "
                    f"Kann Time-Delay nicht anwenden (time_delay.apply_on_x=true)."
                )

            if include_state:
                phi = self.apply_time_delay(X, n_delays=n_delays, tau=tau)
            else:
                # keine states, aber Zeilen müssen zu den späteren Features passen
                phi = np.empty((N - delay_width, 0), dtype=float)

            # Für alle späteren Features muss X zeitlich passend getrimmt werden
            X_aligned = X[delay_width:, :]
        else:
            phi = X.copy() if include_state else np.empty((N, 0), dtype=float)
            X_aligned = X

        # ------------------------------------------------------------
        # 2) RBF
        # ------------------------------------------------------------
        rbf = self.config.get("rbf", {}) or {}
        if bool(rbf.get("enabled", False)) and int(rbf.get("n_centers", 0)) > 0:
            rbfs = self.apply_rbf(X_aligned, rbf_cfg=rbf)
            if rbfs.size:
                phi = np.column_stack([phi, rbfs]) if phi.size else rbfs

        # ------------------------------------------------------------
        # 3) Custom features (x-only)
        # ------------------------------------------------------------
        custom_list: List[str] = self.config.get("custom_features", []) or []
        if custom_list:
            feats = []
            for expr in custom_list:
                expr = str(expr).strip()
                if not expr:
                    raise ValueError("Leerer String in custom_features.")
                f = self.parser.compile(expr, nx=self.n_input_features)
                v = np.asarray(f(X_aligned)).reshape(-1)
                if v.shape[0] != X_aligned.shape[0]:
                    raise ValueError(
                        f"Custom expr '{expr}' lieferte Länge {v.shape[0]}, erwartet {X_aligned.shape[0]}."
                    )
                feats.append(v)
            C = np.column_stack(feats) if feats else np.empty((X_aligned.shape[0], 0), dtype=float)
            if C.size:
                phi = np.column_stack([phi, C]) if phi.size else C

        # ------------------------------------------------------------
        # 4) Bias
        # ------------------------------------------------------------
        if include_bias:
            b = np.ones((X_aligned.shape[0], 1), dtype=float)
            phi = np.column_stack([phi, b]) if phi.size else b

        # ------------------------------------------------------------
        # 5) Gates (x-only)
        #    - gated copies: (phi_base, phi_base*g1, ..., phi_base*gK)
        #    - optional: gates_as_features -> G wird am ENDE angehängt (nicht-gated)
        # ------------------------------------------------------------
        gcfg = self.config.get("gates", {}) or {}
        if bool(gcfg.get("enabled", False)):
            gates: List[str] = gcfg.get("items", []) or []
            if gates:
                gate_vals = []
                for gexpr in gates:
                    gexpr = str(gexpr).strip()
                    if not gexpr:
                        raise ValueError("Leerer String in gates.items.")
                    gfun = self.parser.compile(gexpr, nx=self.n_input_features)
                    g = np.asarray(gfun(X_aligned)).reshape(-1)
                    if g.shape[0] != X_aligned.shape[0]:
                        raise ValueError(
                            f"Gate '{gexpr}' lieferte Länge {g.shape[0]}, erwartet {X_aligned.shape[0]}."
                        )
                    gate_vals.append(g)

                # WICHTIG: base_phi einfrieren, damit gates_as_features NICHT mitgegated wird
                base_phi = phi

                # 5.1) Gated copies
                if base_phi.size:
                    blocks = [base_phi] + [base_phi * g[:, None] for g in gate_vals]
                    phi = np.column_stack(blocks)
                else:
                    phi = base_phi

                # 5.2) Gates als Features am Ende (nicht-gated)
                if bool(gcfg.get("gates_as_features", False)):
                    G = np.column_stack(gate_vals)  # (N_aligned, n_gates)
                    phi = np.column_stack([phi, G]) if phi.size else G
        # ------------------------------------------------------------
        # 6) Time-Delay auf z (wenn enabled und apply_on_x=false)
        # ------------------------------------------------------------
        if td_enabled and not apply_on_x:
            if phi.shape[0] <= delay_width:
                raise ValueError(
                    f"phi hat N={phi.shape[0]} <= n_delays*tau={delay_width}. "
                    f"Kann Time-Delay nicht anwenden (time_delay.apply_on_x=false)."
                )
            if phi.size:
                phi = self.apply_time_delay(phi, n_delays=n_delays, tau=tau)
            else:
                # leere Features -> nur Zeilen trimmen
                phi = np.empty((phi.shape[0] - delay_width, 0), dtype=float)

        self.n_output_features = int(phi.shape[1])
        return phi

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def apply_time_delay(self, X: np.ndarray, *, n_delays: int, tau: int) -> np.ndarray:
        if not isinstance(n_delays, int) or n_delays < 0:
            raise ValueError("n_delays muss >= 0 sein.")
        if not isinstance(tau, int) or tau < 1:
            raise ValueError("tau muss >= 1 sein.")
        if X.ndim != 2:
            raise ValueError(f"apply_time_delay erwartet 2D-Array, bekam {X.shape}.")

        delay_width = n_delays * tau
        N = X.shape[0]
        if N <= delay_width:
            raise ValueError(f"N={N} <= n_delays*tau={delay_width}.")

        blocks = [X]
        for i in range(1, n_delays + 1):
            blocks.append(np.roll(X, i * tau, axis=0))

        # Trim vorne, damit keine "rolled" Artefakte bleiben
        return np.column_stack(blocks)[delay_width:, :]

    def apply_rbf(self, X: np.ndarray, *, rbf_cfg: Dict[str, Any]) -> np.ndarray:
        rbf_type = str(rbf_cfg.get("type", "gauss")).strip()
        kernel_width = float(rbf_cfg.get("kernel_width", 1.0))
        polyharmonic_coeff = float(rbf_cfg.get("polyharmonic_coeff", 1.0))

        centers = np.asarray(rbf_cfg.get("centers", []), dtype=float)
        n_centers = int(rbf_cfg.get("n_centers", 0))

        if n_centers <= 0:
            return np.empty((X.shape[0], 0), dtype=float)

        if centers.size == 0:
            # falls n_centers>0 aber keine Centers gesetzt sind
            raise ValueError("rbf.n_centers > 0, aber rbf.centers ist leer.")

        if centers.ndim == 1:
            centers = centers.reshape(1, -1)

        if centers.shape[0] != n_centers:
            raise ValueError(
                f"rbf.n_centers={n_centers}, aber centers hat {centers.shape[0]} Zeilen."
            )

        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"RBF centers dim mismatch: centers={centers.shape[1]} vs X={X.shape[1]}."
            )

        # Paarweise Distanzen
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
        XC = X @ centers.T
        r2 = np.maximum(X_sq - 2.0 * XC + C_sq, 0.0)
        r = np.sqrt(r2)

        if rbf_type == "thinplate":
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = r2 * np.log(r + 1e-16)
            phi = np.where(np.isfinite(phi), phi, 0.0)

        elif rbf_type == "gauss":
            phi = np.exp(-(kernel_width ** 2) * r2)

        elif rbf_type == "invquad":
            phi = 1.0 / (1.0 + (kernel_width ** 2) * r2)

        elif rbf_type == "invmultquad":
            phi = 1.0 / np.sqrt(1.0 + (kernel_width ** 2) * r2)

        elif rbf_type == "polyharmonic":
            k = polyharmonic_coeff
            with np.errstate(divide="ignore", invalid="ignore"):
                if int(k) % 2 == 0:
                    phi = (r ** k) * np.log(r + 1e-16)
                else:
                    phi = r ** k
            phi = np.where(np.isfinite(phi), phi, 0.0)

        else:
            raise ValueError(f"rbf.type '{rbf_type}' unbekannt.")

        return phi
