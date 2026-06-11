"""End-to-end simulation comparing VIP vs uniform allocation on a mock RLVR env.

Reports three numbers per estimator (lower is better, relative to uniform):

  * uniform        — GRPO-style equal allocation (the 1.000 baseline);
  * learned-VIP    — VIP using the online learned predictor's success estimates;
  * oracle-VIP     — VIP using the env's *true* success probabilities.

Splitting learned vs oracle isolates two things: oracle-VIP shows what the
allocator delivers given perfect estimates, while the learned-vs-oracle gap shows
how much the (deliberately tiny) predictor currently captures. The allocator's
benefit scales with difficulty heterogeneity; the mock env keeps a spread of
per-prompt learnability ceilings so there is always something to exploit.

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
    allocate, uniform_allocation,
)
from aroll.pruning import prune_rollouts  # noqa: E402
from aroll.variance import per_prompt_variance, variance_coeff  # noqa: E402


def run(estimator=Estimator.DR_GRPO, iters=80, num_prompts=64,
        batch_size=24, per_prompt=6, seed=0):
    env = MockRLVREnv(num_prompts=num_prompts, embed_dim=32, seed=seed)
    budget = batch_size * per_prompt           # C = per_prompt * B
    cfg = VIPConfig(estimator=estimator, coeff_strategy="variance", n_min=3, n_max=32)
    vip = VIPAllocator(env.embeddings, cfg)
    rng = np.random.default_rng(seed)

    learned, oracle, uni, saved = [], [], [], []
    for _ in range(iters):
        pool = rng.choice(num_prompts, size=min(batch_size * 2, num_prompts), replace=False)
        batch = vip.select_batch(pool, batch_size, rng)

        # learned-VIP allocation (drives the rollouts and the env update)
        res, pred = vip.allocate(batch, budget)
        learned.append(env.realized_variance(batch, res.n, estimator))

        # oracle-VIP: same budget, but allocate using the env's true p
        ores = allocate(variance_coeff(env.p_true[batch]), budget,
                        cfg.n_min, cfg.n_max, estimator)
        oracle.append(env.realized_variance(batch, ores.n, estimator))

        # uniform baseline on the same batch & budget
        n_uni = uniform_allocation(len(batch), budget, cfg.n_min, cfg.n_max)
        uni.append(env.realized_variance(batch, n_uni, estimator))

        # Rec.4 online pruning on the highest-budget prompt (illustrative)
        j = int(np.argmax(res.n))
        dec = prune_rollouts(np.clip(pred.p_success[j] + rng.normal(0, 0.1, res.n[j]), 0, 1),
                             progress=rng.uniform(0.3, 0.9, res.n[j]), keep_min=cfg.n_min)
        saved.append(dec.est_compute_saved)

        succ, cnt = env.rollout(batch, res.n)
        vip.observe(batch, succ, cnt)
        env.train_step(batch, res.n)

    return (np.mean(learned), np.mean(oracle), np.mean(uni), np.mean(saved))


def main():
    seeds = range(5)
    for est in (Estimator.DR_GRPO, Estimator.RLOO):
        l_r, o_r, sv = [], [], []
        for s in seeds:
            learned, oracle, uni, saved = run(est, seed=s)
            l_r.append(learned / uni)
            o_r.append(oracle / uni)
            sv.append(saved)
        print(f"\n=== {est.value.upper()} (gradient-variance vs uniform, mean over "
              f"{len(seeds)} seeds) ===")
        print(f"  uniform      : 1.000")
        print(f"  learned-VIP  : {np.mean(l_r):.3f}   (tiny online predictor)")
        print(f"  oracle-VIP   : {np.mean(o_r):.3f}   (allocator with true p)")
        print(f"  online pruning (Rec.4) avg compute saved: {np.mean(sv)*100:.1f}%")
        assert np.mean(o_r) < 1.0, "allocator must beat uniform given true p"
        assert np.mean(l_r) <= 1.0 + 1e-3, "learned VIP should not hurt on average"
    print("\nThe allocator reduces gradient variance under an equal budget "
          "(oracle-VIP); the learned predictor captures part of that gap.")


if __name__ == "__main__":
    main()
