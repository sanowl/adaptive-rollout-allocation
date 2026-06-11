"""Tests for the five recommended extensions."""
import numpy as np

from aroll.buckets import BucketConfig, assign_buckets, select_balanced_batch
from aroll.predictor import RolloutPredictor
from aroll.prefix import Prefix, allocate_prefixes
from aroll.pruning import corrected_advantages, prune_rollouts
from aroll.replay import ReplayBuffer, blend_with_replay
from aroll.scoring import boundary_score, coefficients
from aroll.variance import Estimator


# --- Rec. 1: learned predictor -------------------------------------------------
def test_predictor_learns_success_prob():
    rng = np.random.default_rng(0)
    d = 16
    w = rng.normal(size=d)
    embs = rng.normal(size=(64, d))
    p_true = 1 / (1 + np.exp(-(embs @ w)))
    pred = RolloutPredictor(d, hidden=64)
    counts = np.full(64, 16)
    for _ in range(60):
        succ = rng.binomial(counts, p_true)
        pred.update(embs, succ, counts, steps=3)
    out = pred.predict(embs, mc_samples=4)
    mae = np.abs(out.p_success - p_true).mean()
    assert mae < 0.2                          # recovers the success probability
    assert np.all(out.uncertainty >= 0)


def test_boundary_score_shape():
    p = np.array([0.5, 0.9, 0.1])
    unc = np.array([0.2, 0.2, 0.2])
    s = boundary_score(p, unc)
    assert s[0] > s[1] and s[0] > s[2]        # boundary prompt scores highest


def test_coefficients_strategies_positive():
    from aroll.predictor import Prediction
    pred = Prediction(np.array([0.5, 0.2]), np.array([0.3, 0.1]), np.array([0.25, 0.16]))
    for strat in ("variance", "score", "etg", "blend"):
        a = coefficients(pred, strategy=strat)
        assert np.all(a > 0)


# --- Rec. 2: difficulty buckets ------------------------------------------------
def test_buckets_assignment():
    p = np.array([0.1, 0.5, 0.9])
    labels = assign_buckets(p, BucketConfig())
    assert list(labels) == ["hard", "medium", "easy"]


def test_balanced_batch_has_all_difficulties():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=120)
    score = boundary_score(p, np.ones_like(p))
    idx = select_balanced_batch(p, score, batch_size=24, rng=rng)
    labels = assign_buckets(p[idx], BucketConfig())
    assert set(labels) == {"easy", "medium", "hard"}   # no collapse
    assert len(idx) == 24


# --- Rec. 3: replay with staleness --------------------------------------------
def test_replay_staleness_decay():
    buf = ReplayBuffer(decay=0.5, max_age=4)
    emb = np.zeros(4)
    buf.add(0, emb, successes=8, count=10, version=0)
    agg = buf.for_prompts([0], current_version=2)         # age 2 -> 0.25 weight
    s, c = agg[0]
    assert np.isclose(s, 8 * 0.25) and np.isclose(c, 10 * 0.25)


def test_replay_drops_too_old():
    buf = ReplayBuffer(decay=0.9, max_age=2)
    buf.add(0, np.zeros(4), 5, 10, version=0)
    agg = buf.for_prompts([0], current_version=5)         # age 5 > max_age
    assert agg[0] == (0.0, 0.0)


def test_blend_with_replay_adds_pseudocounts():
    buf = ReplayBuffer(decay=1.0, max_age=8)
    buf.add(0, np.zeros(2), 4, 8, version=0)
    succ, cnt, w = blend_with_replay(np.array([0]), np.array([2.0]), np.array([4.0]),
                                     buf, current_version=1)
    assert cnt[0] == 4 + 8 and succ[0] == 2 + 4
    assert 0.5 <= w[0] <= 1.0


# --- Rec. 4: online pruning ----------------------------------------------------
def test_pruning_keeps_minimum():
    p = np.full(10, 0.99)                      # all near-certain & consensual
    dec = prune_rollouts(p, progress=np.ones(10), keep_min=3)
    assert dec.keep.sum() >= 3
    assert dec.pruned_count == 10 - dec.keep.sum()


def test_pruning_spares_uncertain_and_unstarted():
    p = np.array([0.5, 0.5, 0.99, 0.99, 0.99, 0.99])
    prog = np.array([0.9, 0.9, 0.9, 0.9, 0.05, 0.9])
    dec = prune_rollouts(p, progress=prog, keep_min=2, value_quantile=0.5)
    assert dec.keep[0] and dec.keep[1]        # uncertain rollouts kept
    assert dec.keep[4]                        # barely-started rollout kept


def test_frozen_baseline_unbiased_after_pruning():
    # Group of 10: 7 near-certain successes (consensus) + 3 genuine failures.
    # Pruning removes consensus successes, biasing the survivor mean downward;
    # the frozen baseline reflects the full started group instead.
    p = np.array([0.97] * 7 + [0.05, 0.05, 0.05])
    full_mean = np.mean(2 * p - 1)            # true group baseline (reward scale)
    dec = prune_rollouts(p, progress=np.ones(10), keep_min=3, value_quantile=0.5)
    assert dec.pruned_count > 0
    # Frozen baseline equals the full-group mean, regardless of who was pruned.
    assert np.isclose(dec.frozen_baseline, full_mean)
    # Survivor mean is biased away from it (consensus successes were dropped).
    survivor_mean = np.mean((2 * p - 1)[dec.keep])
    assert abs(survivor_mean - full_mean) > abs(dec.frozen_baseline - full_mean)


def test_corrected_advantages_use_frozen_baseline():
    p = np.array([0.9] * 6 + [0.1] * 2)
    dec = prune_rollouts(p, progress=np.ones(8), keep_min=3, value_quantile=0.4)
    kept_rewards = np.where(p[dec.keep] > 0.5, 1.0, -1.0)
    adv = corrected_advantages(kept_rewards, dec)
    assert np.allclose(adv, kept_rewards - dec.frozen_baseline)


# --- Rec. 5: prefix-level allocation ------------------------------------------
def test_prefix_allocation_prefers_mixed_outcomes():
    prefixes = [
        Prefix(id=0, p_success=0.01),         # near-certain fail
        Prefix(id=1, p_success=0.99),         # near-certain success
        Prefix(id=2, p_success=0.5),          # mixed -> most informative
    ]
    res = allocate_prefixes(prefixes, budget=30, n_min=3, n_max=20,
                            estimator=Estimator.DR_GRPO)
    assert res.n[2] == res.n.max()
    assert res.total == 30


def test_prefix_visitation_weighting():
    prefixes = [
        Prefix(id=0, p_success=0.5, visitation=0.01),  # informative but rare
        Prefix(id=1, p_success=0.5, visitation=1.0),   # informative and common
    ]
    res = allocate_prefixes(prefixes, budget=20, n_min=3, n_max=15)
    assert res.n[1] >= res.n[0]
