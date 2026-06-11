"""Allocation coefficients derived from predictor outputs.

The allocation core (:mod:`aroll.allocation`) consumes a single coefficient
``a_q`` per prompt.  This module maps a :class:`~aroll.predictor.Prediction`
into that coefficient under several strategies:

* ``"variance"``  — the paper objective ``a_q = 4 sigma_Z^2 p(1-p)``.
* ``"score"``     — the boundary-seeking score (Rec. 1):
                    ``p (1 - p) * uncertainty`` — targets prompts near the
                    *learning boundary* (not too easy, not impossible) that the
                    model is also uncertain about.
* ``"etg"``       — directly use the predicted expected training gain (learning
                    progress); allocates budget toward prompts whose success rate
                    is currently rising fastest.
* ``"blend"``     — geometric-style blend of the variance term and uncertainty.
"""

from __future__ import annotations

import numpy as np

from .predictor import Prediction


def boundary_score(p: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """``score = p (1 - p) * uncertainty`` (Rec. 1)."""
    p = np.asarray(p, dtype=float)
    return p * (1.0 - p) * np.asarray(uncertainty, dtype=float)


def coefficients(
    pred: Prediction,
    strategy: str = "score",
    sigma_z2: float = 1.0,
    uncertainty_weight: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Map predictor outputs to allocation coefficients ``a_q``.

    A small ``eps`` floor keeps every prompt eligible for the minimum budget and
    avoids degenerate all-zero coefficients (which would make λ-bisection moot).
    """
    p, unc, etg = pred.p_success, pred.uncertainty, pred.expected_training_gain
    if strategy == "variance":
        a = 4.0 * sigma_z2 * p * (1.0 - p)
    elif strategy == "score":
        a = boundary_score(p, unc)
    elif strategy == "etg":
        a = np.asarray(etg, dtype=float)
    elif strategy == "blend":
        var = 4.0 * sigma_z2 * p * (1.0 - p)
        a = var * (1.0 + uncertainty_weight * unc)
    else:
        raise ValueError(f"unknown coefficient strategy: {strategy}")
    return np.asarray(a, dtype=float) + eps
