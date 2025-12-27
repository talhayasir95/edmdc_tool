from typing import Optional, Tuple

from scipy.stats.qmc import Sobol
import math
import numpy as np
from system import System
from dataclasses import dataclass, field
from utils import prbs_steps

@dataclass
class DataGenerator:
    """
    Generiert Trajektorien-Daten (X, Y, U) für ein System.
    - Unterstützt autonome Systeme (nu=0) und Systeme mit Input (nu>0).
    """
    system: System
    n: int
    train_trajectory_horizon: int
    test_train_ratio: float
    seed: Optional[int] = None

    # wird in __post_init__ gesetzt
    x: np.ndarray = field(init=False)
    x_train: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    n_train: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n muss > 0 sein.")
        if self.system.sampling_time() <= 0:
            raise ValueError("dt muss > 0 sein.")
        if self.train_trajectory_horizon <= 0:
            raise ValueError("train_trajectory_horizon muss > 0 sein.")
        if not (0.0 < self.test_train_ratio < 1.0):
            raise ValueError("test_train_ratio muss zwischen 0 und 1 liegen (z.B. 0.8).")

        self.dt = self.system.sampling_time()
        if math.log2(self.n) % 1 != 0:
            raise ValueError("n sollte eine 2er Potenz sein (z.B. 2**10).")

        nx = self.system.state_dimension()

        # Sobol: Startzustände auf [0,1]^nx
        if self.seed is not None:
            sob = Sobol(d=nx, seed=self.seed)
        else:
            sob = Sobol(d=nx)
        x = sob.random(self.n)

        # Skalieren auf Constraints falls vorhanden
        x_limits = self.system.system_constraints()
        if x_limits is not None:
            x_limits = np.asarray(x_limits, dtype=float)
            if x_limits.shape != (nx, 2):
                raise ValueError(f"system_constraints muss shape (nx,2) haben, expected {(nx,2)} got {x_limits.shape}")
            x_min, x_max = x_limits[:, 0], x_limits[:, 1]
            x = x * (x_max - x_min) + x_min

        self.x = x

        # Split
        self.n_train = int(self.n * self.test_train_ratio)
        self.n_test = self.n - self.n_train
        self.x_train = self.x[:self.n_train]
        self.x_test = self.x[self.n_train:]

    @staticmethod
    def rk4_step(f, dt: float, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        RK4 für x_dot = f(x,u)
        """
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def generate_trajectory(
        self,
        x0: np.ndarray,
        trajectory_horizon: Optional[int] = None,
        control_inputs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Erzeugt eine Trajektorie:
          X[t] = x_t
          Y[t] = x_{t+1}
          U[t] = u_t  (leer wenn nu=0)

        control_inputs:
          - None -> wird je nach System automatisch erzeugt
          - shape (T, nu)
        """
        T = int(trajectory_horizon if trajectory_horizon is not None else self.train_trajectory_horizon)
        if T <= 0:
            raise ValueError("trajectory_horizon muss > 0 sein.")

        nx = self.system.state_dimension()
        nu = self.system.input_dimension()

        x = np.asarray(x0, dtype=float).reshape(nx,)
        X, Y = [], []

        if nu == 0:
            # autonom -> U ist leeres Array
            U = []
            for _ in range(T):
                x_next = self.rk4_step(self.system.f_x_u, self.dt, x, None)
                X.append(x.copy())
                Y.append(x_next.copy())
                U.append(np.zeros((0,), dtype=float))  # consistent placeholder
                x = x_next
            return np.array(X), np.array(Y), np.zeros((T, 0), dtype=float)

        # nu > 0: Inputs erforderlich/erlaubt
        if control_inputs is None:
            # default: zufällige Inputs in constraints oder [-1,1]
            u_limits = self.system.input_constraints()
            if u_limits is None:
                u = np.random.uniform(-1.0, 1.0, size=(T, nu))
            else:
                u_limits = np.asarray(u_limits, dtype=float)
                if u_limits.shape != (nu, 2):
                    raise ValueError(f"input_constraints muss shape (nu,2) haben, expected {(nu,2)} got {u_limits.shape}")
                umin, umax = u_limits[:, 0], u_limits[:, 1]
                u = np.random.uniform(0.0, 1.0, size=(T, nu)) * (umax - umin) + umin
            control_inputs = u
        else:
            control_inputs = np.asarray(control_inputs, dtype=float)
            if control_inputs.shape != (T, nu):
                raise ValueError(f"control_inputs muss shape {(T,nu)} haben, got {control_inputs.shape}")

        U = []
        for t in range(T):
            u_t = control_inputs[t]
            x_next = self.rk4_step(self.system.f_x_u, self.dt, x, u_t)
            X.append(x.copy())
            Y.append(x_next.copy())
            U.append(u_t.copy())
            x = x_next

        return np.array(X), np.array(Y), np.array(U)

    def generate_training_data_trajectories(self):
        X_list, Y_list, U_list = [], [], []
        T = self.train_trajectory_horizon
        nu = self.system.input_dimension()

        for x0 in self.x_train:
            control = None if nu == 0 else prbs_steps(
                T=T, nu=nu, amplitude=1.0, switch_steps=5, seed=self.seed
            )

            X_traj, Y_traj, U_traj = self.generate_trajectory(
                x0=x0,
                trajectory_horizon=T,
                control_inputs=control,
            )
            X_list.append(X_traj)
            Y_list.append(Y_traj)
            U_list.append(U_traj)

        return np.vstack(X_list), np.vstack(Y_list), np.vstack(U_list)

    def true_system_rollout(self, x0: np.ndarray, control_inputs: Optional[np.ndarray] = None, T: Optional[int] = None) -> np.ndarray:
        """
        Rollout des wahren Systems:
        gibt X trajectory (T, nx) zurück.
        """
        X, _, _ = self.generate_trajectory(x0=x0, trajectory_horizon=T, control_inputs=control_inputs)
        return X
