"""Prefix-level allocation for agent / multi-step tasks (Rec. 5, "TRACE"-style).

For single-turn tasks VIP allocates at the *prompt* level.  For multi-step tool
use / ReAct-style agents, the informative unit is an intermediate *prefix*
(a partial trajectory / state): we want more continuation budget on prefixes
that are likely to produce *mixed* outcomes — i.e. those near the learning
boundary where some continuations succeed and some fail.

This module reuses the exact VIP allocation core, but treats each prefix as a
"prompt": its coefficient is its outcome variance ``p(1-p)`` (optionally scaled
by predictor uncertainty), and the budget is the number of continuations to
sample from that prefix.  Prefixes can be weighted by how reachable / on-policy
they are (their visitation probability), so we don't over-invest in rare states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocation import AllocationResult, allocate
from .variance import Estimator


@dataclass
class Prefix:
    """An intermediate trajectory state eligible for continuation rollouts."""

    id: int
    p_success: float           # P(a continuation from here succeeds)
    visitation: float = 1.0    # how on-policy / reachable this prefix is (weight)
    uncertainty: float = 0.0   # optional predictor uncertainty at this prefix
    depth: int = 0             # step index (for diagnostics)


def allocate_prefixes(
    prefixes: list[Prefix],
    budget: int,
    n_min: int = 3,
    n_max: int = 32,
    estimator: Estimator = Estimator.DR_GRPO,
    use_uncertainty: bool = True,
) -> AllocationResult:
    """Allocate continuation budget across intermediate prefixes.

    Coefficient per prefix:  ``visitation * p(1-p) * (1 + uncertainty)``.
    Prefixes with near-certain outcomes (p≈0 or p≈1) or that are rarely visited
    receive the minimum; mixed-outcome, frequently-visited prefixes get more.
    """
    if not prefixes:
        return AllocationResult(np.array([], dtype=int), np.array([]), 0.0, 0, True)
    p = np.array([pf.p_success for pf in prefixes], dtype=float)
    vis = np.array([pf.visitation for pf in prefixes], dtype=float)
    unc = np.array([pf.uncertainty for pf in prefixes], dtype=float)

    a = vis * (p * (1.0 - p))
    if use_uncertainty:
        a = a * (1.0 + unc)
    a = a + 1e-6
    return allocate(a, budget=budget, n_min=n_min, n_max=n_max, estimator=estimator)


def expand_tree_budget(
    root_prefixes: list[Prefix],
    total_budget: int,
    branch_factor: int = 2,
    n_min: int = 3,
    n_max: int = 32,
    estimator: Estimator = Estimator.DR_GRPO,
) -> dict[int, int]:
    """Convenience: allocate a continuation budget over a flat set of prefixes and
    return ``{prefix_id: n_continuations}``.  ``branch_factor`` is informational
    (callers expand each chosen prefix into that many children downstream)."""
    res = allocate_prefixes(root_prefixes, total_budget, n_min, n_max, estimator)
    return {pf.id: int(n) for pf, n in zip(root_prefixes, res.n)}
