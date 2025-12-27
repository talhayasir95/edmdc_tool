import numpy as np
from typing import Optional
def prbs_steps(T: int, nu: int = 1, amplitude: float = 1.0, switch_steps: int = 5, seed: Optional[int] = None) -> np.ndarray:
    """
    PRBS für diskrete Zeitschritte:
    - liefert U mit shape (T, nu)
    - switch_steps: wie viele Zeitschritte bleibt ein Wert konstant
    """
    rng = np.random.default_rng(seed)
    if T <= 0:
        raise ValueError("T muss > 0 sein.")
    if nu <= 0:
        raise ValueError("nu muss > 0 sein.")
    switch_steps = max(1, int(switch_steps))

    n_intervals = int(np.ceil(T / switch_steps))
    values = rng.choice([-1.0, 1.0], size=(n_intervals, nu))
    U = np.repeat(values, switch_steps, axis=0)[:T, :]
    return amplitude * U

def nrmse_multi_traj_percent(X_truth_list, X_hat_list):
    """
    Berechnet normalisierten rmse für alle Trajektorien in X_truth_list und X_hat_list
    gemäß paper linear predictors for nonlinear dynamical systems (Seite 19 RMSE Definition)
    """
    num = 0.0
    den = 0.0

    for Xt, Xh in zip(X_truth_list, X_hat_list):
        num += np.sum((Xh - Xt)**2)
        den += np.sum(Xt**2)

    return 100.0 * np.sqrt(num / den) if den > 0 else np.nan