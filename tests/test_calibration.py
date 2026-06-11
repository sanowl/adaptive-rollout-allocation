import numpy as np

from aroll import (
    EnsemblePredictor, MockRLVREnv, VIPAllocator, VIPConfig,
    brier_score, expected_calibration_error, outcomes_from_counts,
    predictor_mae, reliability_curve, uncertainty_error_correlation,
)


# --- calibration metrics -------------------------------------------------------
def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=20000)
    outcomes = (rng.uniform(size=probs.shape) < probs).astype(float)
    assert expected_calibration_error(probs, outcomes, n_bins=10) < 0.02


def test_ece_high_for_overconfident():
    # Always predict 0.99 but only half are correct -> badly calibrated.
    probs = np.full(1000, 0.99)
    outcomes = np.concatenate([np.ones(500), np.zeros(500)])
    assert expected_calibration_error(probs, outcomes) > 0.4


def test_brier_and_mae():
    p_pred = np.array([0.2, 0.8])
    p_true = np.array([0.25, 0.75])
    assert np.isclose(predictor_mae(p_pred, p_true), 0.05)
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_reliability_curve_shapes():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, size=500)
    outcomes = (rng.uniform(size=500) < probs).astype(float)
    conf, acc, cnt = reliability_curve(probs, outcomes, n_bins=10)
    assert conf.shape == acc.shape == cnt.shape
    assert cnt.sum() == 500


def test_outcomes_from_counts_roundtrip():
    succ = np.array([2, 0])
    cnt = np.array([3, 2])
    out, idx = outcomes_from_counts(succ, cnt)
    assert out.sum() == 2 and len(out) == 5
    assert (idx == np.array([0, 0, 0, 1, 1])).all()


# --- ensemble predictor --------------------------------------------------------
def test_ensemble_learns_and_uncertainty_drops_with_data():
    rng = np.random.default_rng(0)
    d = 16
    w = rng.normal(size=d)
    embs = rng.normal(size=(64, d))
    p_true = 1 / (1 + np.exp(-(embs @ w)))
    ens = EnsemblePredictor(d, n_members=4, hidden=64)
    counts = np.full(64, 16)

    early = ens.predict(embs).uncertainty.mean()
    for _ in range(80):
        succ = rng.binomial(counts, p_true)
        ens.update(embs, succ, counts, steps=3)
    out = ens.predict(embs)
    late = out.uncertainty.mean()

    assert predictor_mae(out.p_success, p_true) < 0.2   # members agree on the truth
    assert late < early                                 # disagreement shrinks with data


def test_ensemble_uncertainty_tracks_error():
    # Uncertainty should be positively correlated with actual error.
    rng = np.random.default_rng(2)
    d = 12
    w = rng.normal(size=d)
    embs = rng.normal(size=(80, d))
    p_true = 1 / (1 + np.exp(-(embs @ w)))
    ens = EnsemblePredictor(d, n_members=5, hidden=48)
    counts = np.full(80, 12)
    # Train on only the first half, so the second half stays uncertain.
    for _ in range(60):
        succ = rng.binomial(counts[:40], p_true[:40])
        ens.update(embs[:40], succ, counts[:40], steps=3)
    out = ens.predict(embs)
    corr = uncertainty_error_correlation(out.uncertainty, out.p_success, p_true)
    assert corr > 0.0                                   # higher uncertainty -> higher error


def test_ensemble_is_drop_in_for_vip():
    env = MockRLVREnv(num_prompts=40, embed_dim=16, seed=3)
    vip = VIPAllocator(env.embeddings, VIPConfig(predictor_kind="ensemble", ensemble_members=3))
    assert isinstance(vip.predictor, EnsemblePredictor)
    batch = np.arange(12)
    res, pred = vip.allocate(batch, budget=96)
    assert res.total == 96
    assert np.all(pred.uncertainty >= 0)
    succ, cnt = env.rollout(batch, res.n)
    vip.observe(batch, succ, cnt)                        # update path works
    assert vip.version == 1
