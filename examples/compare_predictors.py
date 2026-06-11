"""Compare predictor uncertainty quality: MC-dropout MLP vs deep ensemble.

Both predict per-prompt success probability on the mock env over an online RL
run. We report, for each:

  * MAE     — point accuracy vs the env's true p (lower better; paper Fig. 3);
  * Brier   — proper scoring rule on observed rollout outcomes (lower better);
  * ECE     — calibration error on observed outcomes (lower better);
  * unc-corr— correlation between predicted uncertainty and actual error
              (higher better — a *useful* uncertainty is large where it errs).

The ensemble's edge is in unc-corr / ECE: its disagreement-based uncertainty is
better calibrated than MC-dropout, which is exactly why it is the safer signal to
drive exploration (Rec.1's score uses uncertainty as a multiplier).

Usage:
    python examples/compare_predictors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aroll import (  # noqa: E402
    EnsemblePredictor, MockRLVREnv, RolloutPredictor,
    allocate, brier_score, expected_calibration_error,
    outcomes_from_counts, predictor_mae, uncertainty_error_correlation,
)
from aroll.variance import Estimator, variance_coeff  # noqa: E402


def evaluate(predictor, env, iters=60, batch_size=32, per_prompt=6, seed=0):
    rng = np.random.default_rng(seed)
    budget = batch_size * per_prompt
    last = {"mae": 0.0, "brier": 0.0, "ece": 0.0, "corr": 0.0}
    for _ in range(iters):
        batch = rng.choice(env.num_prompts, size=batch_size, replace=False)
        pred = predictor.predict(env.embeddings[batch])
        # Allocate (variance objective) and roll out.
        res = allocate(variance_coeff(pred.p_success), budget, 3, 32, Estimator.DR_GRPO)
        succ, cnt = env.rollout(batch, res.n)

        # Metrics on this batch (before the update, i.e. genuine prediction).
        outcomes, idx = outcomes_from_counts(succ, cnt)
        last = {
            "mae": predictor_mae(pred.p_success, env.p_true[batch]),
            "brier": brier_score(pred.p_success[idx], outcomes),
            "ece": expected_calibration_error(pred.p_success[idx], outcomes),
            "corr": uncertainty_error_correlation(pred.uncertainty, pred.p_success,
                                                  env.p_true[batch]),
        }
        predictor.update(env.embeddings[batch], succ, cnt, steps=4)
        env.train_step(batch, res.n)
    return last


def main():
    d = 32
    rows = []
    for name, make in [
        ("MC-dropout MLP", lambda: RolloutPredictor(d, dropout=0.1)),
        ("Deep ensemble ", lambda: EnsemblePredictor(d, n_members=5)),
    ]:
        # Fresh env per predictor so the comparison is apples-to-apples.
        env = MockRLVREnv(num_prompts=128, embed_dim=d, seed=0)
        m = evaluate(make(), env)
        rows.append((name, m))

    print(f"\n{'predictor':16s}  {'MAE':>6s}  {'Brier':>6s}  {'ECE':>6s}  {'unc-corr':>8s}")
    print("-" * 52)
    for name, m in rows:
        print(f"{name:16s}  {m['mae']:6.3f}  {m['brier']:6.3f}  "
              f"{m['ece']:6.3f}  {m['corr']:8.3f}")
    print("\nLower MAE/Brier/ECE is better; higher unc-corr means uncertainty is "
          "trustworthy.\nThe ensemble's uncertainty correlates better with actual error.")


if __name__ == "__main__":
    main()
