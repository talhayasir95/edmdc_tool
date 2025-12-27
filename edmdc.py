from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class EDMDcParams:
    """
    Parameter für die Regression (Ridge).
    alpha wird hier als lambda * N skaliert (wie in deinem Code).
    """
    lambda_dyn: float = 1e-6   # Ridge für A,B,(N)
    lambda_dec: float = 1e-6   # Ridge für C


@dataclass
class EDMDcModel:
    """
    EDMDc Modell Klasse
      z_{k+1} = A z_k + B u_k + sum_j (N[:,:,j] @ z_k) * u_{k,j} + b_z
      x_k     = C z_k + d_x
    """
    A: np.ndarray                      # (nZ x nZ)
    B: np.ndarray                      # (nZ x nU)  (kann nU=0 sein)
    C: np.ndarray                      # (nX x nZ)

    b_z: np.ndarray                    # (nZ,)
    d_x: np.ndarray                    # (nX,)

    observables: Any                   # obj mit .transform(X)->Φ und .config dict

    # bilinear optional:
    N: Optional[np.ndarray] = None     # (nZ x nZ x nU) oder None

    def predict_multi_step(self, x0: np.ndarray, U_seq: Optional[np.ndarray]) -> np.ndarray:
        """Multi-step Prediction.

        Unterstützt auch Time-Delay-Observables:
        - Falls time_delay.enabled=True und n_delays*tau>0, muss x0 eine HISTORY enthalten:
            x0 shape (delay_width+1, nX)
          Dann wird z0 als letzte Zeile aus observables.transform(history) genommen.

        x0:
          - ohne Time-Delay: (nX,) oder (1,nX)
          - mit Time-Delay:  (delay_width+1, nX)
        U_seq:
          - (T,nU) oder (T,) bei nU=1
          - oder None wenn nU==0
        """

        cfg: Dict[str, Any] = getattr(self.observables, "config", {}) or {}
        td = cfg.get("time_delay", {}) or {}
        td_enabled = bool(td.get("enabled", False))
        n_delays = int(td.get("n_delays", 0))
        tau = int(td.get("tau", 1))
        delay_width = (n_delays * tau) if td_enabled else 0

        # --- x0 -> History oder Single ---
        x0 = np.asarray(x0, dtype=float)
        if delay_width > 0:
            if x0.ndim != 2 or x0.shape[0] < (delay_width + 1):
                raise ValueError(
                    f"Time-Delay aktiv (delay_width={delay_width}). "
                    f"Bitte x0 als History geben: (delay_width+1,nX) = ({delay_width+1},nX). "
                    f"Bekommen: {x0.shape}"
                )
            x0_hist = x0[-(delay_width + 1):, :]
            x0_2d = x0_hist
        else:
            if x0.ndim == 1:
                x0_2d = x0.reshape(1, -1)
            elif x0.ndim == 2 and x0.shape[0] == 1:
                x0_2d = x0
            else:
                raise ValueError(f"x0 muss (nX,) oder (1,nX) sein, bekam {x0.shape}")

        # --- Horizon bestimmen ---
        # wenn keine Inputs verwendet werden, darf U_seq None sein
        nU = int(self.B.shape[1])
        if nU == 0:
            if U_seq is None:
                T = 1  # minimal 1 step, falls User None gibt (kannst du anpassen)
            else:
                U_seq = np.asarray(U_seq, dtype=float)
                T = int(U_seq.shape[0])
        else:
            if U_seq is None:
                raise ValueError("Dieses Modell erwartet Eingaben (nU>0), aber U_seq ist None.")
            U_seq = np.asarray(U_seq, dtype=float)
            if U_seq.ndim == 1:
                if nU != 1:
                    raise ValueError(f"U_seq ist 1D, aber nU={nU}. Erwartet (T,nU).")
                U_seq = U_seq.reshape(-1, 1)
            if U_seq.shape[1] != nU:
                raise ValueError(f"U_seq hat nU={U_seq.shape[1]}, erwartet {nU}.")
            T = int(U_seq.shape[0])

        nX = int(self.C.shape[0])
        X_hat = np.zeros((T, nX), dtype=float)

        # --- z0 bestimmen ---
        z0_mat = np.asarray(self.observables.transform(x0_2d), dtype=float)
        if z0_mat.ndim != 2 or z0_mat.shape[0] < 1:
            raise ValueError(f"observables.transform(x0) lieferte ungültige Form: {z0_mat.shape}")
        z = z0_mat[-1, :].reshape(-1)  # (nZ,)

        # --- Loop ---
        for t in range(T):
            u = np.zeros((0,), dtype=float) if nU == 0 else U_seq[t].reshape(-1)

            # bilinear term (optional)
            bilinear = 0.0
            if self.N is not None and nU > 0:
                # sum_j (N[:,:,j] @ z) * u_j
                # effizient: über j summieren
                tmp = np.zeros_like(z)
                for j in range(nU):
                    tmp += (self.N[:, :, j] @ z) * u[j]
                bilinear = tmp

            # z_{k+1}
            if nU > 0:
                z = self.A @ z + self.B @ u + bilinear + self.b_z
            else:
                z = self.A @ z + bilinear + self.b_z

            # x_hat
            X_hat[t] = (self.C @ z + self.d_x).reshape(-1)

        return X_hat


class EDMDcTrainer:
    """
    Trainer für EDMDc.
    Liest Flags aus observables.config:
      - lift_raw_input: bool
      - bilinear_zxu: bool
    """
    def __init__(self, params: EDMDcParams):
        self.params = params

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: Optional[np.ndarray],
        observables: Any
    ) -> EDMDcModel:
        """
        X, Y: (N,nX)
        U:    (N,nU) oder (N,) bei nU=1 oder None

        Verhalten:
        - Wenn config["lift_raw_input"] == False -> U wird ignoriert (nU=0).
        - Wenn config["bilinear_zxu"] == True -> zusätzliche Features z*u werden ergänzt (nur wenn nU>0).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"X und Y müssen 2D sein. X={X.shape}, Y={Y.shape}")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"X und Y müssen gleiche nX haben. X={X.shape}, Y={Y.shape}")

        cfg: Dict[str, Any] = getattr(observables, "config", {}) or {}
        lift_raw_input = bool(cfg.get("lift_raw_input", True))   # :contentReference[oaicite:2]{index=2}
        bilinear_zxu = bool(cfg.get("bilinear_zxu", False))      # :contentReference[oaicite:3]{index=3}

        # --- z und z_next ---
        Phi = np.asarray(observables.transform(X), dtype=float)      # (N_eff, nZ)
        Phi_next = np.asarray(observables.transform(Y), dtype=float) # (N_eff, nZ)

        # --- U vorbereiten ---
        if (U is None) or (not lift_raw_input):
            U_feat = np.empty((Phi.shape[0], 0), dtype=float)
        else:
            U = np.asarray(U, dtype=float)
            if U.ndim == 1:
                U = U.reshape(-1, 1)
            if U.shape[0] != X.shape[0]:
                # Bei Time-Delay kann Phi weniger Zeilen haben -> wir trimmen gleich sauber
                pass
            U_feat = U

        # --- alle Matrices auf gemeinsame Länge bringen ---
        N_eff = min(Phi.shape[0], Phi_next.shape[0], U_feat.shape[0])
        if N_eff <= 0:
            raise ValueError("N_eff <= 0: zu wenig Daten nach Observables/Time-Delay.")

        Phi = Phi[:N_eff, :]
        Phi_next = Phi_next[:N_eff, :]
        U_feat = U_feat[:N_eff, :]

        nZ = int(Phi.shape[1])
        nU = int(U_feat.shape[1])

        # --- Regressor-Matrix bauen ---
        # Standard: [z, u]
        # Bilinear: [z, u, vec(z*u)]
        if bilinear_zxu and nU > 0:
            # ZU: (N_eff, nZ*nU) mit Reihenfolge: u0-Block, u1-Block, ...
            # ZU[i, j*nZ:(j+1)*nZ] = Phi[i,:] * U_feat[i,j]
            ZU = (Phi[:, :, None] * U_feat[:, None, :]).reshape(N_eff, nZ * nU)
            Xs = np.hstack([Phi, U_feat, ZU])
        else:
            Xs = np.hstack([Phi, U_feat])

        # --- Dynamik-Regresssion ---
        reg_dyn = Ridge(alpha=self.params.lambda_dyn * N_eff, fit_intercept=True)
        reg_dyn.fit(Xs, Phi_next)

        K = np.asarray(reg_dyn.coef_, dtype=float)         # (nZ, n_features)
        b_z = np.asarray(reg_dyn.intercept_, dtype=float)  # (nZ,)

        # A, B extrahieren
        A = K[:, :nZ]                                      # (nZ, nZ)
        B = K[:, nZ:nZ + nU] if nU > 0 else np.empty((nZ, 0), dtype=float)

        # Bilinear N extrahieren (optional)
        N_tensor = None
        if bilinear_zxu and nU > 0:
            N_flat = K[:, nZ + nU:]                        # (nZ, nZ*nU)
            if N_flat.shape[1] != nZ * nU:
                raise RuntimeError("Interner Fehler: N_flat hat falsche Dimension.")
            # reshape zu (nZ, nZ, nU) mit gleicher Blockreihenfolge wie ZU oben
            N_tensor = N_flat.reshape(nZ, nZ, nU)

        # --- Decoder-Regression (Phi -> X) ---
        reg_dec = Ridge(alpha=self.params.lambda_dec * N_eff, fit_intercept=True)
        reg_dec.fit(Phi, X[:N_eff, :])

        C = np.asarray(reg_dec.coef_, dtype=float)         # (nX, nZ)
        d_x = np.asarray(reg_dec.intercept_, dtype=float)  # (nX,)

        return EDMDcModel(
            A=A,
            B=B,
            C=C,
            b_z=b_z,
            d_x=d_x,
            observables=observables,
            N=N_tensor
        )
