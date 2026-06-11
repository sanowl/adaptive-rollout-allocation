"""End-to-end: VIP should reduce realised gradient variance vs uniform."""
import numpy as np

from aroll import Estimator, MockRLVREnv, VIPAllocator, VIPConfig, uniform_allocation


def _run(estimator, strategy="variance", seed=0, iters=30):
    # The paper's guarantee is about gradient variance, which the "variance"
    # (and "blend") coefficient strategy minimises. The "score" strategy
    # (Rec. 1) targets the learning boundary instead, a different objective.
    env = MockRLVREnv(num_prompts=200, embed_dim=24, seed=seed)
    batch_size, per_prompt = 24, 6
    budget = batch_size * per_prompt
    cfg = VIPConfig(estimator=estimator, coeff_strategy=strategy)
    vip = VIPAllocator(env.embeddings, cfg)
    rng = np.random.default_rng(seed)
    v_vip, v_uni = [], []
    for _ in range(iters):
        pool = rng.choice(env.num_prompts, size=batch_size * 2, replace=False)
        batch = vip.select_batch(pool, batch_size, rng)
        res, _ = vip.allocate(batch, budget)
        v_vip.append(env.realized_variance(batch, res.n, estimator))
        v_uni.append(env.realized_variance(
            batch, uniform_allocation(len(batch), budget, 3, 32), estimator))
        succ, cnt = env.rollout(batch, res.n)
        vip.observe(batch, succ, cnt)
        env.train_step(batch, res.n)
    return np.mean(v_vip), np.mean(v_uni)


def _mean_over_seeds(estimator, strategy, seeds=(0, 1, 2, 3)):
    ratios = [np.divide(*_run(estimator, strategy, seed=s)) for s in seeds]
    return float(np.mean(ratios))


def test_vip_reduces_variance_dr_grpo():
    assert _mean_over_seeds(Estimator.DR_GRPO, "variance") < 1.0


def test_vip_reduces_variance_rloo():
    assert _mean_over_seeds(Estimator.RLOO, "variance") < 1.0


def test_blend_strategy_also_reduces_variance():
    # "blend" keeps the variance objective while folding in uncertainty.
    assert _mean_over_seeds(Estimator.DR_GRPO, "blend") < 1.0


def test_env_keeps_difficulty_spread():
    # Fix #1: training must not collapse all prompts to p=1; ceilings preserve
    # a heterogeneous population so the allocator always has something to exploit.
    env = MockRLVREnv(num_prompts=128, embed_dim=8, learn_rate=0.2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        batch = rng.choice(128, size=32, replace=False)
        env.train_step(batch, np.full(32, 16))
    assert env.p_true.std() > 0.1            # spread survives
    assert (env.p_true < 0.95).mean() > 0.2  # plenty of non-saturated prompts


def test_etg_target_is_learning_progress():
    # Fix #2: the ETG head target is the *rise* in success rate, not p(1-p).
    env = MockRLVREnv(num_prompts=20, embed_dim=8, seed=1)
    vip = VIPAllocator(env.embeddings, VIPConfig(use_replay=False))
    batch = np.arange(5)
    # First observation seeds the baseline (no progress signal yet).
    vip.observe(batch, successes=np.full(5, 2.0), counts=np.full(5, 8.0))
    assert np.allclose(vip._last_phat[batch], 2.0 / 8.0)
    # Second observation: success rate rose 0.25 -> 0.75, progress = 0.5.
    vip.observe(batch, successes=np.full(5, 6.0), counts=np.full(5, 8.0))
    assert np.allclose(vip._last_phat[batch], 6.0 / 8.0)


def test_observe_advances_version_and_replay():
    env = MockRLVREnv(num_prompts=50, embed_dim=16, seed=3)
    vip = VIPAllocator(env.embeddings, VIPConfig())
    batch = np.arange(10)
    res, _ = vip.allocate(batch, budget=80)
    succ, cnt = env.rollout(batch, res.n)
    vip.observe(batch, succ, cnt)
    assert vip.version == 1
    assert len(vip.replay) == 10
