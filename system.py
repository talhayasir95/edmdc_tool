from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from dataclasses import dataclass

class System(ABC):
    @abstractmethod
    def f_x_u(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Dynamics: x_dot = f(x,u)"""
        raise NotImplementedError

    @abstractmethod
    def sampling_time(self) -> float:
        """Return sampling time for system"""
        raise NotImplementedError

    @abstractmethod
    def state_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def input_dimension(self) -> int:
        raise NotImplementedError

    def system_constraints(self) -> Optional[np.ndarray]:
        """
        Constraints for datageneration in the form:
        np.array([
            [x1_min, x1_max],
            [x2_min, x2_max],
            ...
        ])
        """
        return None

    def input_constraints(self) -> Optional[np.ndarray]:
        """
        Constraints for datageneration in the form:
        np.array([
            [u1_min, u1_max],
            [u2_min, u2_max],
            ...
        ])
        """
        return None

@dataclass
class BruntonSystem(System):
    mu: float = -1.0
    lam: float = -1.0

    def f_x_u(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        # autonomes System -> u wird ignoriert
        x1, x2 = x
        return np.array([
            self.mu * x1,
            self.lam * (x2 - x1**2)
        ], dtype=float)

    def state_dimension(self) -> int:
        return 2

    def input_dimension(self) -> int:
        return 0

    def system_constraints(self) -> Optional[np.ndarray]:
        # Beispiel (optional): wenn du nichts willst -> return None
        # return np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=float)
        return None

    def sampling_time(self):
        return .01