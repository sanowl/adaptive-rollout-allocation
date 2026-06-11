"""Mock RLVR environment for testing VIP without an LLM.

Each prompt has a latent true success probability ``p_q`` and a fixed random
embedding (correlated with ``p_q`` so the predictor has learnable signal).
Calling :meth:`rollout` draws Bernoulli rewards in {-1, +1}.  :meth:`train_step`
simulates policy learning: prompts that received *more* rollouts (and were near
the boundary) improve faster, so their ``p_q`` drifts toward 1 — reproducing the
non-stationary success probabilities the paper's predictor must track.

This lets the whole pipeline (predictor, allocation, replay, buckets) run and be
measured end-to-end on a laptop.
"""

from __future__ import annotations

import numpy as np

from .variance import Estimator, per_prompt_variance


class MockRLVREnv:
    def __init__(self, num_prompts: int = 256, embed_dim: int = 32,
                 learn_rate: float = 0.08, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.learn_rate = learn_rate
        # Latent difficulty spread across [0,1].
        self.p_true = self.rng.uniform(0.02, 0.98, size=num_prompts)
        # Per-prompt learnability ceiling: training pushes p toward this ceiling,
        # not to 1. Drawing ceilings with their own spread keeps the *population*
        # heterogeneous over time (some prompts stay hard), so there is always
        # something for an adaptive allocator to exploit — unlike a model where
        # every prompt saturates to p=1 and difficulty collapses.
        self.ceiling = np.clip(self.p_true + self.rng.uniform(0.0, 0.45, size=num_prompts),
                               self.p_true, 0.99)
        # Embeddings: a signal direction encoding p plus noise (so a predictor
        # can recover p but not trivially).
        signal = np.outer(self._logit(self.p_true), self.rng.normal(size=embed_dim))
        noise = self.rng.normal(scale=1.0, size=(num_prompts, embed_dim))
        self.embeddings = (signal / np.sqrt(embed_dim) + noise).astype(float)

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def rollout(self, prompt_ids: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (successes, counts) for the requested prompts."""
        prompt_ids = np.asarray(prompt_ids)
        counts = np.asarray(counts, dtype=int)
        succ = np.array([self.rng.binomial(c, self.p_true[i]) for i, c in zip(prompt_ids, counts)])
        return succ, counts

    def realized_variance(self, prompt_ids, counts, estimator: Estimator,
                          sigma_z2: float = 1.0) -> float:
        """True minibatch gradient variance under this allocation (ground truth)."""
        prompt_ids = np.asarray(prompt_ids)
        v = per_prompt_variance(np.asarray(counts, dtype=float), self.p_true[prompt_ids],
                                estimator, sigma_z2)
        return float(v.sum())

    def train_step(self, prompt_ids: np.ndarray, counts: np.ndarray) -> None:
        """Advance the policy: boundary prompts with more rollouts improve more,
        but only up to each prompt's learnability ceiling."""
        prompt_ids = np.asarray(prompt_ids)
        counts = np.asarray(counts, dtype=float)
        p = self.p_true[prompt_ids]
        ceil = self.ceiling[prompt_ids]
        # Learning signal ~ gradient magnitude proxy p(1-p), scaled by rollouts,
        # and saturating as p approaches the prompt's ceiling.
        gain = self.learn_rate * np.sqrt(counts) * p * (1 - p) * np.maximum(ceil - p, 0.0)
        self.p_true[prompt_ids] = np.clip(p + gain, 1e-4, ceil)

    def mean_success(self) -> float:
        return float(self.p_true.mean())
