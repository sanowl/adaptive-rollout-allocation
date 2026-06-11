"""Difficulty buckets (Rec. 2).

VIP (and any variance/score allocation) tends to concentrate budget on
mid-difficulty prompts (``p ~ 0.5``), which can collapse a batch into a single
difficulty type.  We bucket prompts by predicted success probability and force
the batch to contain a quota of easy / medium / hard prompts, preserving a
healthy curriculum spread.

Convention: success probability ``p`` high  => easy; low => hard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EASY, MEDIUM, HARD = "easy", "medium", "hard"


@dataclass
class BucketConfig:
    """Thresholds and per-batch quotas for difficulty bucketing."""

    hard_below: float = 0.33      # p < hard_below            -> hard
    easy_above: float = 0.67      # p > easy_above            -> easy  (else medium)
    # Fractions of the batch reserved per bucket (need not sum to 1; the
    # remainder is filled by global score ranking).
    quota: dict | None = None

    def quotas_for(self, batch_size: int) -> dict:
        q = self.quota or {HARD: 0.25, MEDIUM: 0.5, EASY: 0.25}
        return {k: int(round(v * batch_size)) for k, v in q.items()}


def assign_buckets(p: np.ndarray, cfg: BucketConfig) -> np.ndarray:
    """Return a string label per prompt: 'easy' / 'medium' / 'hard'."""
    p = np.asarray(p, dtype=float)
    labels = np.full(p.shape, MEDIUM, dtype=object)
    labels[p < cfg.hard_below] = HARD
    labels[p > cfg.easy_above] = EASY
    return labels


def select_balanced_batch(
    p: np.ndarray,
    score: np.ndarray,
    batch_size: int,
    cfg: BucketConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick a difficulty-balanced batch of indices from a candidate pool.

    Within each bucket prompts are ranked by ``score`` (most informative first)
    to satisfy the bucket's quota; any leftover slots are filled by the global
    score ranking.  Returns selected indices into the pool.
    """
    cfg = cfg or BucketConfig()
    rng = rng or np.random.default_rng()
    p = np.asarray(p, dtype=float)
    score = np.asarray(score, dtype=float)
    n = p.shape[0]
    batch_size = min(batch_size, n)

    labels = assign_buckets(p, cfg)
    quotas = cfg.quotas_for(batch_size)
    chosen: list[int] = []
    taken = np.zeros(n, dtype=bool)

    for bucket, q in quotas.items():
        idx = np.where(labels == bucket)[0]
        if idx.size == 0 or q <= 0:
            continue
        order = idx[np.argsort(-score[idx])]   # highest score first
        pick = order[:q]
        chosen.extend(pick.tolist())
        taken[pick] = True

    # Fill any remaining slots by global score among the not-yet-taken prompts.
    if len(chosen) < batch_size:
        rest = np.where(~taken)[0]
        rest = rest[np.argsort(-score[rest])]
        need = batch_size - len(chosen)
        chosen.extend(rest[:need].tolist())

    # If quotas overshot (rounding), trim by lowest score.
    if len(chosen) > batch_size:
        chosen = np.array(chosen)
        chosen = chosen[np.argsort(-score[chosen])][:batch_size].tolist()

    return np.array(sorted(chosen), dtype=int)
