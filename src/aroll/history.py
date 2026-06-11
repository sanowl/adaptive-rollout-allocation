"""Per-prompt history state for tracking non-stationary success probability.

The success probability ``p_q`` is *non-stationary* — it drifts upward as the
policy learns. A predictor that only sees a *static* prompt embedding cannot tell
"prompt q early in training" from "prompt q late in training", so it can only
track the drift through slow online SGD (short memory). The paper's GP sidesteps
this with recursive Bayesian posterior updates.

This module gives the learned predictor an explicit *state* to extrapolate from:
a few recency features per prompt (level, trend, recency, seen-flag) that are
concatenated onto the static embedding. Cheap to maintain, and it lets the
predictor model drift directly instead of chasing it.

It also owns the per-prompt last-observed success rate, so the expected-training-
gain target (learning progress) is computed here rather than duplicated.
"""

from __future__ import annotations

import numpy as np


class PromptHistory:
    """Tracks recent per-prompt success-rate statistics.

    Args:
        num_prompts: size of the prompt set.
        ema_decay: smoothing for the EMA success-rate level (higher = smoother).
        prior: success-rate prior for never-seen prompts.
    """

    N_FEATURES = 4   # [seen, ema_level, trend, recency]

    def __init__(self, num_prompts: int, ema_decay: float = 0.6, prior: float = 0.5):
        self.num_prompts = num_prompts
        self.ema_decay = ema_decay
        self.prior = prior
        self.seen = np.zeros(num_prompts, dtype=bool)
        self.last_phat = np.full(num_prompts, np.nan, dtype=float)
        self.prev_phat = np.full(num_prompts, np.nan, dtype=float)
        self.ema = np.full(num_prompts, prior, dtype=float)
        self.last_version = np.full(num_prompts, -1, dtype=int)

    def features(self, prompt_ids: np.ndarray, current_version: int) -> np.ndarray:
        """Recency feature block for the given prompts, shape ``(B, N_FEATURES)``.

        Reflects state *before* this iteration's update — i.e. exactly what the
        predictor "knew" when it made the prediction it is about to be trained on.
        """
        ids = np.asarray(prompt_ids)
        seen = self.seen[ids].astype(float)
        ema = self.ema[ids]
        both = self.seen[ids]
        trend = np.where(
            both & ~np.isnan(self.prev_phat[ids]),
            self.last_phat[ids] - self.prev_phat[ids],
            0.0,
        )
        age = np.where(self.last_version[ids] >= 0, current_version - self.last_version[ids], 0)
        recency = np.where(self.seen[ids], 1.0 / (1.0 + age), 0.0)
        return np.stack([seen, ema, trend, recency], axis=1)

    def learning_progress(self, prompt_ids: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        """ETG target: rise in success rate since the prompt was last seen.

        First-ever observation has no baseline, so it falls back to the boundary
        proxy ``p_hat(1 - p_hat)``. Uses pre-update state, so call before
        :meth:`update`.
        """
        ids = np.asarray(prompt_ids)
        prev = self.last_phat[ids]
        p_hat = np.asarray(p_hat, dtype=float)
        return np.where(np.isnan(prev), p_hat * (1.0 - p_hat), np.clip(p_hat - prev, 0.0, 1.0))

    def update(self, prompt_ids: np.ndarray, p_hat: np.ndarray, version: int) -> None:
        """Record this iteration's observed success rates. Call after computing
        features/targets for the same step."""
        ids = np.asarray(prompt_ids)
        p_hat = np.asarray(p_hat, dtype=float)
        seen = self.seen[ids]
        # Shift current level into prev (for trend), seed prev=p_hat on first sight.
        self.prev_phat[ids] = np.where(seen, self.last_phat[ids], p_hat)
        self.last_phat[ids] = p_hat
        self.ema[ids] = np.where(seen, self.ema_decay * self.ema[ids] + (1 - self.ema_decay) * p_hat, p_hat)
        self.last_version[ids] = version
        self.seen[ids] = True
