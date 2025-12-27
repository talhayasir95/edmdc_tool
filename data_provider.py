# data_provider.py
from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np


@dataclass
class DataProvider:
    """Erzeugt reproduzierbare synthetische Trainingsdaten (X,Y,U).

    Die ursprünglichen Data-Generator-Module sind in deinem Upload nicht dabei.
    Damit die GUI trotzdem EDMDc fitten kann, erzeugen wir ein stabiles lineares
    System:
        x_{k+1} = A x_k + B u_k + noise

    Parameter werden über ENV gesteuert:
      OBS_NX, OBS_NU, OBS_N, OBS_DT, OBS_SEED
    """

    nx: int = field(default_factory=lambda: int(os.environ.get("OBS_NX", "2")))
    nU: int = field(default_factory=lambda: int(os.environ.get("OBS_NU", "1")))
    N: int = field(default_factory=lambda: int(os.environ.get("OBS_N", "4000")))
    dt: float = field(default_factory=lambda: float(os.environ.get("OBS_DT", "1.0")))
    seed: int = field(default_factory=lambda: int(os.environ.get("OBS_SEED", "7")))

    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)

    # nur fürs Debugging (nicht zwingend genutzt)
    A_true: np.ndarray = field(init=False)
    B_true: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.regenerate()

    def regenerate(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Stabiler A_true (Spektralradius < 1)
        M = rng.normal(size=(self.nx, self.nx))
        eig = np.linalg.eigvals(M)
        rho = float(np.max(np.abs(eig))) if eig.size else 1.0
        scale = 0.95 / max(rho, 1e-9)
        self.A_true = (M * scale).astype(float)

        self.B_true = rng.normal(scale=0.4, size=(self.nx, self.nU)).astype(float)

        U = rng.normal(scale=1.0, size=(self.N, self.nU)).astype(float)

        X = np.zeros((self.N, self.nx), dtype=float)
        x = rng.normal(scale=0.5, size=(self.nx,)).astype(float)
        noise_scale = 0.01

        for k in range(self.N):
            X[k] = x
            x = self.A_true @ x + self.B_true @ U[k] + rng.normal(scale=noise_scale, size=(self.nx,))

        # (X,Y,U) als Paare
        self.X = X[:-1]
        self.Y = X[1:]
        self.U = U[:-1]

    @property
    def shapes(self) -> str:
        return f"X={self.X.shape}, Y={self.Y.shape}, U={self.U.shape}"
