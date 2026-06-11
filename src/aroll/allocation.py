"""Variance-minimising rollout allocation (VIP, paper Section 5.2 / Appendix A.2, D).

Given a coefficient ``a_q`` per prompt (typically ``4 sigma_Z^2 p_q(1-p_q)``,
see :mod:`aroll.variance`) we solve the continuous relaxation of the integer
allocation problem (paper Eq. 6 for Dr. GRPO, Eq. 8 for RLOO)::

    min  sum_q f_q(n_q)   s.t.  sum_q n_q = C,   L <= n_q <= U

The KKT analysis (Theorems 5.1 / 5.2) shows the optimal ``n_q`` is a clamped,
monotone-decreasing function of a single Lagrange multiplier ``lambda``:

    Dr. GRPO :  lambda = a_q (n - 2) / n^3     (interior; solved numerically)
    RLOO     :  n = 1 + sqrt(a_q / lambda)     (interior; closed form)

``S(lambda) = sum_q n_q*(lambda)`` is non-increasing, so the unique ``lambda*``
with ``S(lambda*) = C`` is found by bisection.  We then map the continuous
solution to integers with the greedy marginal-gain rounding of Appendix D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .variance import Estimator, objective_term


@dataclass
class AllocationResult:
    """Outcome of :func:`allocate`."""

    n: np.ndarray            # integer rollouts per prompt, shape (B,)
    n_continuous: np.ndarray  # continuous relaxation solution, shape (B,)
    lam: float               # Lagrange multiplier lambda* found by bisection
    total: int               # sum(n), should equal the budget C
    feasible: bool           # whether BL <= C <= BU held


# --------------------------------------------------------------------------- #
# Continuous per-prompt solution n_q*(lambda)
# --------------------------------------------------------------------------- #
def _interior_dr_grpo(a: float, lam: float, L: int, U: int) -> float:
    """Solve a (n-2)/n^3 = lambda for n in [L, U] (Dr. GRPO).

    The RHS g(n) = a (n-2)/n^3 is strictly decreasing on n >= 3, so we bisect.
    """
    if a <= 0.0 or lam <= 0.0:
        return float(U) if lam <= 0.0 else float(L)

    def g(n: float) -> float:
        return a * (n - 2.0) / (n ** 3)

    lo, hi = float(L), float(U)
    g_lo, g_hi = g(lo), g(hi)        # g_lo >= g_hi (decreasing)
    if lam >= g_lo:
        return float(L)
    if lam <= g_hi:
        return float(U)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) > lam:
            lo = mid                  # need larger n to decrease g
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _n_star(a: np.ndarray, lam: float, L: int, U: int, estimator: Estimator) -> np.ndarray:
    """Vectorised n_q*(lambda), clamped to [L, U] (Theorems 5.1 / 5.2)."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    if estimator == Estimator.RLOO:
        # n = 1 + sqrt(a / lambda), with the threshold clamps from Thm 5.2.
        for i, ai in enumerate(a):
            if ai <= 0.0:
                out[i] = L
                continue
            thr_U = ai / (U - 1.0) ** 2
            thr_L = ai / (L - 1.0) ** 2
            if lam <= thr_U:
                out[i] = U
            elif lam >= thr_L:
                out[i] = L
            else:
                out[i] = 1.0 + np.sqrt(ai / lam)
        return np.clip(out, L, U)
    # Dr. GRPO
    for i, ai in enumerate(a):
        out[i] = _interior_dr_grpo(float(ai), lam, L, U)
    return np.clip(out, L, U)


def _solve_lambda(a: np.ndarray, C: int, L: int, U: int, estimator: Estimator) -> float:
    """Bisection for lambda* such that sum_q n_q*(lambda) = C."""
    a = np.asarray(a, dtype=float)
    a_max = float(np.max(a)) if a.size else 1.0
    # lambda upper bound: large enough that every n_q* hits L.
    if estimator == Estimator.RLOO:
        hi = max(a_max / (L - 1.0) ** 2, 1.0) * 10.0 + 1.0
    else:
        hi = max(a_max * (L - 2.0) / (L ** 3), 1.0) * 10.0 + 1.0
    lo = 0.0
    # S is non-increasing in lambda: S(lo) = BU >= C >= BL = S(hi).
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(_n_star(a, mid, L, U, estimator)))
        if s > C:
            lo = mid                  # too many rollouts -> raise lambda
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Greedy integer rounding (paper Appendix D)
# --------------------------------------------------------------------------- #
def _round(n_cont: np.ndarray, a: np.ndarray, C: int, L: int, U: int,
           estimator: Estimator) -> np.ndarray:
    """Round down, then hand out the remaining budget by largest variance drop."""
    n = np.clip(np.floor(n_cont).astype(int), L, U)
    remaining = int(C - n.sum())

    if remaining > 0:
        # Add rollouts one at a time to whichever prompt yields the largest
        # decrease in f_q (marginal variance reduction), respecting n_q <= U.
        for _ in range(remaining):
            cur = objective_term(n, a, estimator)
            nxt = objective_term(np.minimum(n + 1, U), a, estimator)
            gain = cur - nxt
            gain[n >= U] = -np.inf
            j = int(np.argmax(gain))
            if not np.isfinite(gain[j]):
                break                 # no room left anywhere
            n[j] += 1
    elif remaining < 0:
        # Rare (floors can't exceed C for a feasible problem) but stay safe:
        # remove rollouts where it costs the least variance increase.
        for _ in range(-remaining):
            cur = objective_term(n, a, estimator)
            prv = objective_term(np.maximum(n - 1, L), a, estimator)
            cost = prv - cur
            cost[n <= L] = np.inf
            j = int(np.argmin(cost))
            if not np.isfinite(cost[j]):
                break
            n[j] -= 1
    return n


def allocate(
    a: np.ndarray,
    budget: int,
    n_min: int = 3,
    n_max: int = 64,
    estimator: Estimator = Estimator.DR_GRPO,
) -> AllocationResult:
    """Allocate ``budget`` rollouts across prompts to minimise gradient variance.

    Args:
        a: per-prompt coefficients ``a_q`` (shape ``(B,)``). Larger ``a_q`` ->
            more rollouts. Use :func:`aroll.variance.variance_coeff` for the
            paper objective, or any score (e.g. ``p(1-p)*uncertainty``).
        budget: total rollout budget ``C``.
        n_min, n_max: per-prompt bounds ``L`` and ``U`` (paper requires L >= 3).
        estimator: which variance law (Dr. GRPO or RLOO).

    Returns:
        :class:`AllocationResult` with integer ``n`` summing to ``budget``.
    """
    a = np.asarray(a, dtype=float)
    B = a.shape[0]
    L, U, C = int(n_min), int(n_max), int(budget)
    if L < 3:
        raise ValueError("paper requires per-prompt lower bound L >= 3")
    feasible = B * L <= C <= B * U
    if not feasible:
        # Clamp the budget into the feasible band and proceed best-effort.
        C = int(np.clip(C, B * L, B * U))

    lam = _solve_lambda(a, C, L, U, estimator)
    n_cont = _n_star(a, lam, L, U, estimator)
    n_int = _round(n_cont, a, C, L, U, estimator)
    return AllocationResult(
        n=n_int,
        n_continuous=n_cont,
        lam=lam,
        total=int(n_int.sum()),
        feasible=feasible,
    )


def uniform_allocation(B: int, budget: int, n_min: int = 3, n_max: int = 64) -> np.ndarray:
    """Baseline GRPO-style uniform allocation (every prompt gets C/B rollouts)."""
    base = budget // B
    n = np.full(B, base, dtype=int)
    n[: budget - base * B] += 1       # spread the remainder
    return np.clip(n, n_min, n_max)
