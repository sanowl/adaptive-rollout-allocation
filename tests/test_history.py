"""Change #1: per-prompt history features for non-stationary success prob."""
import numpy as np

from aroll import MockRLVREnv, VIPAllocator, VIPConfig
from aroll.history import PromptHistory


# --- unit: PromptHistory bookkeeping ------------------------------------------
def test_history_features_shape_and_unseen_defaults():
    h = PromptHistory(num_prompts=10)
    feats = h.features(np.array([0, 1, 2]), current_version=0)
    assert feats.shape == (3, PromptHistory.N_FEATURES)
    # Unseen prompts: seen=0, ema=prior(0.5), trend=0, recency=0.
    assert np.allclose(feats[:, 0], 0.0)
    assert np.allclose(feats[:, 1], 0.5)
    assert np.allclose(feats[:, 2:], 0.0)


def test_history_tracks_level_trend_recency():
    h = PromptHistory(num_prompts=5, ema_decay=0.5)
    h.update(np.array([0]), np.array([0.2]), version=0)
    h.update(np.array([0]), np.array([0.6]), version=1)
    f = h.features(np.array([0]), current_version=3)[0]
    assert f[0] == 1.0                       # seen
    assert np.isclose(f[2], 0.6 - 0.2)       # trend = last - prev
    assert np.isclose(f[3], 1.0 / (1.0 + (3 - 1)))   # recency = 1/(1+age)


def test_learning_progress_first_then_delta():
    h = PromptHistory(num_prompts=5)
    # First sight -> proxy p(1-p).
    lp0 = h.learning_progress(np.array([0]), np.array([0.25]))
    assert np.isclose(lp0[0], 0.25 * 0.75)
    h.update(np.array([0]), np.array([0.25]), version=0)
    # Next -> clipped rise in success rate.
    lp1 = h.learning_progress(np.array([0]), np.array([0.75]))
    assert np.isclose(lp1[0], 0.5)


# --- mechanism: history supplies signal when embeddings cannot -----------------
def test_history_enables_discrimination_with_shared_embeddings():
    # Identical embeddings for all prompts => a static predictor must output one
    # value for everyone (no signal). Only history features can tell prompts apart.
    def corr(use_history, seed=0, nprompts=40, iters=60):
        env = MockRLVREnv(num_prompts=nprompts, embed_dim=16, learn_rate=0.03, seed=seed)
        env.embeddings = np.ones_like(env.embeddings)
        vip = VIPAllocator(env.embeddings, VIPConfig(
            coeff_strategy="variance", use_history=use_history,
            use_buckets=False, predictor_steps=4, mc_samples=1))
        rng = np.random.default_rng(seed)
        cs = []
        for it in range(iters):
            batch = rng.choice(nprompts, size=20, replace=False)
            res, pred = vip.allocate(batch, budget=120)
            if it > iters // 2 and pred.p_success.std() > 1e-6 and env.p_true[batch].std() > 1e-6:
                cs.append(np.corrcoef(pred.p_success, env.p_true[batch])[0, 1])
            s, c = env.rollout(batch, res.n)
            vip.observe(batch, s, c)
            env.train_step(batch, res.n)
        return np.mean(cs) if cs else 0.0

    assert corr(use_history=False) < 0.3     # static predictor is blind here
    assert corr(use_history=True) > 0.6      # history recovers the per-prompt signal


def test_history_increases_input_dim():
    env = MockRLVREnv(num_prompts=20, embed_dim=24, seed=0)
    with_h = VIPAllocator(env.embeddings, VIPConfig(use_history=True))
    without_h = VIPAllocator(env.embeddings, VIPConfig(use_history=False))
    assert with_h.embed_dim == 24 + PromptHistory.N_FEATURES
    assert without_h.embed_dim == 24
