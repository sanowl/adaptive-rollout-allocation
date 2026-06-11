"""End-to-end simulation comparing VIP vs uniform allocation on a mock RLVR env.

Runs the full pipeline (learned predictor + difficulty buckets + replay) against
the GRPO-style uniform baseline under an identical rollout budget, and reports
the realised minibatch gradient variance (lower is better) at each iteration.

Usage:
    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aroll import (  # noqa: E402
    Estimator, MockRLVREnv, VIPAllocator, VIPConfig,
    allocate, coefficients, uniform_allocation,
)
from aroll.pruning import prune_rollouts  # noqa: E402


def run(estimator=Estimator.DR_GRPO, strategy="variance", iters=40,
        batch_size=32, per_prompt=6, seed=0):
    env = MockRLVREnv(num_prompts=256, embed_dim=32, seed=seed)
    budget = batch_size * per_prompt           # C = per_prompt * B, as in the paper
    cfg = VIPConfig(estimator=estimator, coeff_strategy=strategy, n_min=3, n_max=32)
    vip = VIPAllocator(env.embeddings, cfg)
    rng = np.random.default_rng(seed)

    var_vip, var_uni, saved = [], [], []
    for _ in range(iters):
        pool = rng.choice(env.num_prompts, size=batch_size * 3, replace=False)

        # --- VIP path ---
        batch = vip.select_batch(pool, batch_size, rng)
        res, pred = vip.allocate(batch, budget)
        var_vip.append(env.realized_variance(batch, res.n, estimator))

        # Rec. 4: online pruning illustration on the highest-budget prompt.
        j = int(np.argmax(res.n))
        dec = prune_rollouts(np.clip(pred.p_success[j] + rng.normal(0, 0.1, res.n[j]), 0, 1),
                             progress=rng.uniform(0.3, 0.9, res.n[j]), keep_min=cfg.n_min)
        saved.append(dec.est_compute_saved)

        succ, cnt = env.rollout(batch, res.n)
        vip.observe(batch, succ, cnt)
        env.train_step(batch, res.n)

        # --- uniform baseline on the same batch & budget ---
        n_uni = uniform_allocation(len(batch), budget, cfg.n_min, cfg.n_max)
        var_uni.append(env.realized_variance(batch, n_uni, estimator))

    return np.array(var_vip), np.array(var_uni), np.array(saved)


def main():
    for est in (Estimator.DR_GRPO, Estimator.RLOO):
        print(f"\n=== {est.value.upper()} ===")
        # Average the variance ratio over seeds for a stable estimate.
        ratios, saves = [], []
        for seed in range(4):
            v_vip, v_uni, saved = run(est, strategy="variance", seed=seed)
            ratios.append(v_vip.mean() / v_uni.mean())
            saves.append(saved.mean())
        ratio = float(np.mean(ratios))
        print(f"  variance strategy (paper objective): "
              f"VIP/uniform gradient-variance ratio = {ratio:.3f}")
        print(f"  online pruning (Rec.4) avg compute saved: {np.mean(saves)*100:.1f}%")
        assert ratio < 1.0, "VIP should reduce variance vs uniform"

        # The boundary-seeking score (Rec.1) optimises a different objective:
        # it concentrates budget on mid-difficulty, high-uncertainty prompts.
        v_vip_s, _, _ = run(est, strategy="score", seed=0)
        print(f"  score strategy (Rec.1) targets the learning boundary "
              f"(not variance-minimising by design)")
    print("\nVIP reduced gradient variance under an equal budget in both regimes.")


if __name__ == "__main__":
    main()
