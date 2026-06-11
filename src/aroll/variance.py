"""Per-prompt gradient-variance formulas from the VIP paper.

Implements Propositions 4.2 (Dr. GRPO) and 4.3 (RLOO), which give the variance
of the per-prompt projected gradient estimator as a function of the number of
rollouts ``n``, the per-rollout success probability ``p`` and the variance of
the projected gradient ``sigma_Z^2``::

    Dr. GRPO :  Var(G_q) = (n - 1) / n^2  * 4 sigma_Z^2 p (1 - p)
    RLOO     :  Var(G_q) = 1 / (n - 1)    * 4 sigma_Z^2 p (1 - p)

Following the paper we factor out the prompt-dependent coefficient

    a_q = 4 * sigma_Z^2 * p_q (1 - p_q)

so that the allocation objective only depends on ``a_q`` and ``n_q``.  Because
``sigma_Z^2`` is assumed equal across prompts (paper Sec. 6 / Appendix B.3) it is
a global scale that cancels in the relative allocation; it only matters for
reporting absolute variance.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Estimator(str, Enum):
    """Group-based advantage estimator selecting which variance law to use."""

    DR_GRPO = "dr_grpo"
    RLOO = "rloo"


def variance_coeff(p: np.ndarray, sigma_z2: float = 1.0) -> np.ndarray:
    """Prompt coefficient ``a_q = 4 sigma_Z^2 p (1 - p)`` (paper Eq. 6/8).

    This is the quantity the allocation actually consumes.  It is maximised at
    ``p = 0.5`` (the "learning boundary") and zero at ``p in {0, 1}``.
    """
    p = np.asarray(p, dtype=float)
    return 4.0 * sigma_z2 * p * (1.0 - p)


def per_prompt_variance(
    n: np.ndarray,
    p: np.ndarray,
    estimator: Estimator = Estimator.DR_GRPO,
    sigma_z2: float = 1.0,
) -> np.ndarray:
    """Var(G_q) given rollouts ``n`` and success probability ``p``.

    ``n`` and ``p`` broadcast together.  ``n`` must be >= 2 for Dr. GRPO and
    >= 2 for RLOO (RLOO divides by ``n - 1``).
    """
    n = np.asarray(n, dtype=float)
    a = variance_coeff(p, sigma_z2)
    if estimator == Estimator.DR_GRPO:
        return a * (n - 1.0) / (n * n)
    if estimator == Estimator.RLOO:
        return a / (n - 1.0)
    raise ValueError(f"unknown estimator: {estimator}")


def objective_term(n: np.ndarray, a: np.ndarray, estimator: Estimator) -> np.ndarray:
    """The continuous allocation objective term ``f_q(n)`` minimised in Eq. (6)/(8).

    ``f_q(n) = a_q (n - 1) / n^2``  (Dr. GRPO)  or  ``a_q / (n - 1)``  (RLOO).
    Used by the greedy rounding step (Appendix D) to score marginal rollouts.
    """
    n = np.asarray(n, dtype=float)
    a = np.asarray(a, dtype=float)
    if estimator == Estimator.DR_GRPO:
        return a * (n - 1.0) / (n * n)
    if estimator == Estimator.RLOO:
        return a / (n - 1.0)
    raise ValueError(f"unknown estimator: {estimator}")
