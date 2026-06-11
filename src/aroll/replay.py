"""Rollout reuse with staleness down-weighting (Rec. 3).

Instead of discarding rollouts after one update, we keep recent *verified*
rollouts in a replay buffer and reuse them — but down-weight each entry by how
stale its generating policy is relative to the current policy version::

    weight = decay ** (current_version - entry_version)

These down-weighted rollouts augment the fresh observations when (a) updating
the success predictor and (b) estimating per-prompt success probability, which
is especially valuable when compute-limited (fewer fresh rollouts needed).

Only *on-policy* fresh rollouts should drive the gradient update itself (the
paper's variance analysis is about on-policy samples); replay is used for
*estimation* and to stretch the effective sample size.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutRecord:
    prompt_id: int
    successes: int
    count: int
    version: int                       # policy version when generated
    embedding: np.ndarray | None = None  # only stored when store_embeddings=True


class ReplayBuffer:
    """Fixed-capacity buffer of recent verified rollouts with staleness decay.

    Args:
        capacity: max records (oldest evicted first).
        decay: per-version staleness multiplier in (0, 1].
        max_age: records older than this many versions are ignored (and pruned).
        store_embeddings: keep per-record embeddings. Off by default — the buffer
            blends success/count statistics by prompt id and does not need them,
            so storing d-dim vectors for thousands of records is wasted memory.
            Enable only if you plan to refit a predictor from replayed examples.
    """

    def __init__(self, capacity: int = 4096, decay: float = 0.8, max_age: int = 8,
                 store_embeddings: bool = False):
        self.capacity = capacity
        self.decay = decay
        self.max_age = max_age
        self.store_embeddings = store_embeddings
        self._buf: deque[RolloutRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, prompt_id: int, embedding: np.ndarray | None, successes: int,
            count: int, version: int) -> None:
        emb = (np.asarray(embedding, dtype=float)
               if self.store_embeddings and embedding is not None else None)
        self._buf.append(RolloutRecord(int(prompt_id), int(successes), int(count),
                                       int(version), emb))

    def add_batch(self, prompt_ids, embeddings, successes, counts, version: int) -> None:
        for pid, emb, s, c in zip(prompt_ids, embeddings, successes, counts):
            self.add(pid, emb, s, c, version)

    def staleness_weight(self, record: RolloutRecord, current_version: int) -> float:
        age = current_version - record.version
        if age < 0 or age > self.max_age:
            return 0.0
        return float(self.decay ** age)

    def prune(self, current_version: int) -> None:
        """Drop records older than ``max_age``."""
        self._buf = deque(
            (r for r in self._buf if current_version - r.version <= self.max_age),
            maxlen=self.capacity,
        )

    def for_prompts(self, prompt_ids, current_version: int) -> dict[int, tuple[float, float]]:
        """Aggregate replay evidence per prompt id.

        Returns ``{prompt_id: (weighted_successes, weighted_count)}`` so callers
        can blend it with fresh rollout statistics.
        """
        wanted = set(int(p) for p in prompt_ids)
        agg: dict[int, list[float]] = {p: [0.0, 0.0] for p in wanted}
        for r in self._buf:
            if r.prompt_id not in wanted:
                continue
            w = self.staleness_weight(r, current_version)
            if w <= 0.0:
                continue
            agg[r.prompt_id][0] += w * r.successes
            agg[r.prompt_id][1] += w * r.count
        return {p: (s, c) for p, (s, c) in agg.items()}


def blend_with_replay(
    prompt_ids: np.ndarray,
    fresh_successes: np.ndarray,
    fresh_counts: np.ndarray,
    buffer: ReplayBuffer,
    current_version: int,
    replay_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine fresh stats with staleness-weighted replay stats.

    Returns ``(successes, counts, sample_weight)`` aligned with ``prompt_ids``,
    suitable to pass straight to :meth:`RolloutPredictor.update`.  Fresh rollouts
    keep full weight; replay contributes ``replay_weight``-scaled pseudo-counts.
    """
    prompt_ids = np.asarray(prompt_ids)
    fresh_successes = np.asarray(fresh_successes, dtype=float)
    fresh_counts = np.asarray(fresh_counts, dtype=float)
    replay = buffer.for_prompts(prompt_ids, current_version)

    succ = fresh_successes.copy()
    cnt = fresh_counts.copy()
    for i, pid in enumerate(prompt_ids):
        s, c = replay.get(int(pid), (0.0, 0.0))
        succ[i] += replay_weight * s
        cnt[i] += replay_weight * c
    # Down-weight prompts whose evidence is mostly stale replay.
    fresh_frac = np.divide(fresh_counts, np.maximum(cnt, 1e-9))
    sample_weight = 0.5 + 0.5 * fresh_frac   # in [0.5, 1.0]
    return succ, cnt, sample_weight
