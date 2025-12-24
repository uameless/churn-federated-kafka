from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def fedavg(local_updates: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    FedAvg over LogisticRegression weights.

    Each local update dict must contain:
    - "coef": list[list[float]] shape (1, n_features)
    - "intercept": list[float] shape (1,)
    - "n_samples": int

    Returns:
    - global_coef: np.ndarray shape (1, n_features)
    - global_intercept: np.ndarray shape (1,)
    """
    if not local_updates:
        raise ValueError("No local updates provided for FedAvg.")

    total = sum(int(u["n_samples"]) for u in local_updates)
    if total <= 0:
        raise ValueError("Total number of samples must be > 0.")

    coef_sum = None
    intercept_sum = None

    for u in local_updates:
        w = float(u["n_samples"]) / float(total)
        coef = np.array(u["coef"], dtype=float)
        intercept = np.array(u["intercept"], dtype=float)

        if coef_sum is None:
            coef_sum = w * coef
            intercept_sum = w * intercept
        else:
            coef_sum += w * coef
            intercept_sum += w * intercept

    return coef_sum, intercept_sum
