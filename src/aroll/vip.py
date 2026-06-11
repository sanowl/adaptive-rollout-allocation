"""VIP orchestrator: predictor + buckets + replay + variance-minimising allocation.

This ties the paper's allocation core together with the five recommended
extensions into one object that, each training iteration:

1. predicts per-prompt success / uncertainty / ETG with the learned predictor
   (Rec. 1, replacing the GP);
2. optionally selects a difficulty-balanced batch from a candidate pool (Rec. 2);
3. derives allocation coefficients (paper variance or boundary score);
4. solves the variance-minimising budget allocation (paper Sec. 5.2);
5. after rollouts, blends fresh + replayed stats (Rec. 3) and updates the
   predictor online.

Online pruning (Rec. 4, :mod:`aroll.pruning`) and prefix-level allocation
(Rec. 5, :mod:`aroll.prefix`) plug in around this loop and are exercised in the
simulation / tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocation import AllocationResult, allocate
from .buckets import BucketConfig, select_balanced_batch
from .predictor import EnsemblePredictor, Prediction, RolloutPredictor
from .replay import ReplayBuffer, blend_with_replay
from .scoring import boundary_score, coefficients
from .variance import Estimator


@dataclass
class VIPConfig:
    estimator: Estimator = Estimator.DR_GRPO
    coeff_strategy: str = "score"      # "score" (Rec.1), "variance" (paper), "blend", "etg"
    n_min: int = 3                     # L
    n_max: int = 32                    # U
    sigma_z2: float = 1.0
    mc_samples: int = 8                # MC-dropout samples for uncertainty
    predictor_steps: int = 5
    use_buckets: bool = True
    bucket_cfg: BucketConfig = field(default_factory=BucketConfig)
    use_replay: bool = True
    replay_weight: float = 1.0
    predictor_kind: str = "mlp"        # "mlp" (MC-dropout) or "ensemble" (calibrated)
    ensemble_members: int = 5


class VIPAllocator:
    """Stateful VIP controller over a fixed set of training prompts."""

    def __init__(self, embeddings: np.ndarray, config: VIPConfig | None = None,
                 predictor: RolloutPredictor | None = None):
        self.embeddings = np.asarray(embeddings, dtype=float)
        self.num_prompts, self.embed_dim = self.embeddings.shape
        self.cfg = config or VIPConfig()
        if predictor is not None:
            self.predictor = predictor
        elif self.cfg.predictor_kind == "ensemble":
            self.predictor = EnsemblePredictor(self.embed_dim, n_members=self.cfg.ensemble_members)
        else:
            self.predictor = RolloutPredictor(self.embed_dim)
        self.replay = ReplayBuffer() if self.cfg.use_replay else None
        self.version = 0               # policy version (increments per update)
        # Last observed success rate per prompt, used to derive a *real* target
        # for the expected-training-gain head: learning progress = Δ success rate.
        # NaN marks prompts never observed yet.
        self._last_phat = np.full(self.num_prompts, np.nan, dtype=float)

    # -- prediction ----------------------------------------------------------
    def predict(self, prompt_ids: np.ndarray) -> Prediction:
        return self.predictor.predict(self.embeddings[prompt_ids], mc_samples=self.cfg.mc_samples)

    # -- batch selection (Rec. 2) -------------------------------------------
    def select_batch(self, candidate_ids: np.ndarray, batch_size: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        candidate_ids = np.asarray(candidate_ids)
        pred = self.predict(candidate_ids)
        score = boundary_score(pred.p_success, pred.uncertainty)
        if not self.cfg.use_buckets:
            order = np.argsort(-score)[:batch_size]
            return candidate_ids[np.sort(order)]
        local = select_balanced_batch(pred.p_success, score, batch_size, self.cfg.bucket_cfg, rng)
        return candidate_ids[local]

    # -- allocation (paper Sec. 5.2) ----------------------------------------
    def allocate(self, prompt_ids: np.ndarray, budget: int) -> tuple[AllocationResult, Prediction]:
        prompt_ids = np.asarray(prompt_ids)
        pred = self.predict(prompt_ids)
        a = coefficients(pred, strategy=self.cfg.coeff_strategy, sigma_z2=self.cfg.sigma_z2)
        res = allocate(a, budget=budget, n_min=self.cfg.n_min, n_max=self.cfg.n_max,
                       estimator=self.cfg.estimator)
        return res, pred

    # -- online update (Rec. 1 + Rec. 3) ------------------------------------
    def observe(self, prompt_ids: np.ndarray, successes: np.ndarray, counts: np.ndarray) -> float:
        """Record verified rollouts and update the predictor; advances version."""
        prompt_ids = np.asarray(prompt_ids)
        successes = np.asarray(successes, dtype=float)
        counts = np.asarray(counts, dtype=float)
        embs = self.embeddings[prompt_ids]

        if self.replay is not None:
            self.replay.add_batch(prompt_ids, embs, successes.astype(int),
                                  counts.astype(int), self.version)
            succ, cnt, w = blend_with_replay(prompt_ids, successes, counts, self.replay,
                                             self.version, self.cfg.replay_weight)
        else:
            succ, cnt, w = successes, counts, None

        # Real expected-training-gain target: how much this prompt's success rate
        # rose since we last saw it (learning progress). First observation has no
        # baseline, so we fall back to the boundary proxy p_hat(1-p_hat) for it.
        p_hat = np.clip(successes / np.maximum(counts, 1.0), 0.0, 1.0)
        prev = self._last_phat[prompt_ids]
        etg_target = np.where(np.isnan(prev), p_hat * (1.0 - p_hat),
                              np.clip(p_hat - prev, 0.0, 1.0))
        self._last_phat[prompt_ids] = p_hat

        loss = self.predictor.update(embs, succ, cnt, sample_weight=w,
                                     etg_target=etg_target, steps=self.cfg.predictor_steps)
        self.version += 1
        if self.replay is not None:
            self.replay.prune(self.version)
        return loss
