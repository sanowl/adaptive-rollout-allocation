# aroll — Adaptive Rollout Allocation (VIP) + experimental extensions

Implementation of **VIP** ("Adaptive Rollout Allocation for Online RL with
Verifiable Rewards", ICLR 2026) plus five experimental extensions. The paper
core is faithful; the extensions are deliberate *experiments* that go beyond the
paper's objective — each is a toggle so you can A/B them.

## Paper core (faithful)

- `variance.py` — per-prompt gradient variance (Props 4.2 Dr.GRPO, 4.3 RLOO) and
  the coefficient `a_q = 4σ²·p(1−p)`.
- `allocation.py` — variance-minimising budget allocation: continuous solution
  (Theorems 5.1/5.2) via **bisection on the Lagrange multiplier λ**, then greedy
  marginal-gain **integer rounding** (Appendix D). Respects `sum n_q = C`,
  `L ≤ n_q ≤ U`.

## Extensions (experimental — each a knob)

| # | Idea | Module | Notes |
|---|------|--------|-------|
| 1 | Learned predictor replaces the GP (`p_success`, `uncertainty`, `expected_training_gain`); score `p(1−p)·uncertainty` | `predictor.py`, `scoring.py` | MC-dropout MLP **or** calibrated deep ensemble (`predictor_kind="ensemble"`). `score` optimises the *learning boundary*, not gradient variance — exploration objective, on by default |
| 2 | Difficulty buckets (easy/medium/hard quotas) | `buckets.py` | batch-selection guardrail against curriculum collapse |
| 3 | Replay buffer, staleness-downweighted | `replay.py` | used for *predictor estimation only*, never the on-policy gradient |
| 4 | Online rollout pruning during generation | `pruning.py` | arrol-style; prunes certain+consensual rollouts, keeps ≥ `keep_min` |
| 5 | Prefix-level allocation for agents | `prefix.py` | TRACE-style; reuses the allocator at prefix granularity |

`vip.py` (`VIPAllocator`) wires predictor + buckets + replay + allocation together.
`env.py` is a mock RLVR environment so the whole pipeline runs without an LLM.

## Quick start

```bash
pip install -e .                       # numpy + torch
python examples/demo.py                # VIP vs uniform on the mock env
python examples/compare_predictors.py  # MC-dropout MLP vs deep ensemble (calibration)
pytest                                 # 38 tests
```

Calibration tools (`calibration.py`): `predictor_mae`, `brier_score`,
`expected_calibration_error`, `reliability_curve`, `uncertainty_error_correlation`
— for judging whether a predictor's uncertainty is trustworthy, not just its
point accuracy. The deep ensemble gives better-calibrated uncertainty than
MC-dropout, which matters because the `score` strategy multiplies by it.

```python
import numpy as np
from aroll import VIPAllocator, VIPConfig, Estimator

embeddings = np.random.randn(1000, 384)          # prompt embeddings
vip = VIPAllocator(embeddings, VIPConfig(
    estimator=Estimator.DR_GRPO,
    coeff_strategy="score",   # "score"|"variance"|"blend"|"etg"
    use_buckets=True, use_replay=True,
))

batch = vip.select_batch(candidate_ids=np.arange(1000), batch_size=32)
result, pred = vip.allocate(batch, budget=32 * 8)   # result.n -> rollouts per prompt
# ... generate result.n[i] rollouts per prompt, verify rewards ...
vip.observe(batch, successes, counts)               # online predictor + replay update
```

## Choosing a coefficient strategy

- `variance` / `blend` — minimise gradient variance (the paper's guarantee).
- `score` (default) — `p(1−p)·uncertainty`, targets mid-difficulty + uncertain
  prompts. This is an exploration objective and is **not** variance-minimising;
  use it when you want curriculum/active-learning behaviour rather than the
  paper's variance reduction.

## Integration points (real training loop)

- Swap `MockRLVREnv` for your generator+verifier: produce `result.n[i]` rollouts
  per prompt, compute binary rewards, pass `successes`/`counts` to `observe`.
- Feed real prompt embeddings (or pooled hidden states) into `RolloutPredictor`.
- Apply `prune_rollouts` inside generation; apply `allocate_prefixes` per step
  for multi-turn/agent trajectories.
```
