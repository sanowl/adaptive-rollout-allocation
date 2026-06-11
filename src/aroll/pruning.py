"""Online rollout pruning during generation (Rec. 4, "arrol"-style).

After VIP decides how many rollouts ``n_q`` to *start* per prompt, we can still
save compute by terminating low-value rollouts *while they are being generated*.
The reported arrol speedup (~1.7x) comes from not finishing rollouts that will
not contribute useful gradient signal.

A group's gradient signal comes from *disagreement* among rollouts (the
advantage is the reward minus the group mean — identical rewards => zero
advantage => zero gradient).  So a partially generated rollout is low-value when
its outcome is already nearly certain *and* it agrees with the emerging group
consensus.  We prune such rollouts subject to keeping at least ``keep_min`` per
prompt so the advantage estimator stays well-defined.

The pruner is verifier/model-agnostic: feed it, per in-flight rollout, a running
estimate of (i) the probability the rollout ends up correct and (ii) optional
confidence/progress.  It returns which rollouts to continue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PruneDecision:
    keep: np.ndarray          # bool mask over in-flight rollouts
    pruned_count: int
    est_compute_saved: float  # fraction of remaining work skipped (0..1)
    frozen_baseline: float    # unbiased group baseline (reward scale), see below


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def prune_rollouts(
    p_correct: np.ndarray,
    progress: np.ndarray | None = None,
    keep_min: int = 3,
    value_quantile: float = 0.3,
    min_progress: float = 0.2,
) -> PruneDecision:
    """Decide which in-flight rollouts to continue for a single prompt.

    Args:
        p_correct: running estimate of P(rollout is correct), one per rollout.
        progress: optional fraction generated so far (0..1); rollouts barely
            started are never pruned (we cannot yet judge them).
        keep_min: never drop below this many rollouts (advantage well-defined).
        value_quantile: prune rollouts whose value falls in the bottom quantile.
        min_progress: only prune rollouts that are at least this far along.

    Value of a rollout = its contribution to group disagreement.  We approximate
    it by how far its predicted outcome is from the group mean, weighted by its
    outcome uncertainty (entropy): a rollout that is *certain and consensual*
    carries the least new gradient signal.

    Baseline-bias correction (Change #2): pruning preferentially removes
    *consensus* rollouts, so the mean of the *surviving* rollouts is a biased
    estimate of the group baseline that the advantage estimator subtracts. We
    therefore freeze an unbiased baseline computed from *all* started rollouts
    before pruning (mapping ``p_correct`` to the reward scale, ``E[R]=2p-1`` for
    rewards in {-1,+1}). The trainer should center advantages with
    :func:`corrected_advantages` against ``decision.frozen_baseline`` rather than
    re-deriving the mean from the survivors.
    """
    p = np.asarray(p_correct, dtype=float)
    m = p.shape[0]
    # Unbiased baseline from the full started group, before any pruning.
    frozen_baseline = float(np.mean(2.0 * p - 1.0)) if m else 0.0
    if m <= keep_min:
        return PruneDecision(np.ones(m, dtype=bool), 0, 0.0, frozen_baseline)

    consensus = float(p.mean())
    disagreement = np.abs(p - consensus)        # high => informative
    uncertainty = _entropy(p)                    # high => outcome still in play
    value = disagreement + uncertainty           # both make a rollout worth finishing

    prog = np.ones(m) if progress is None else np.asarray(progress, dtype=float)
    eligible = prog >= min_progress              # can only prune sufficiently advanced ones

    thr = np.quantile(value, value_quantile)
    prune_mask = eligible & (value <= thr)

    # Enforce keep_min: if pruning would drop below it, restore the most
    # valuable would-be-pruned rollouts.
    keep = ~prune_mask
    if keep.sum() < keep_min:
        deficit = keep_min - int(keep.sum())
        cand = np.where(prune_mask)[0]
        restore = cand[np.argsort(-value[cand])][:deficit]
        keep[restore] = True

    pruned = int((~keep).sum())
    # Compute saved ~ pruned rollouts * their remaining fraction.
    remaining_frac = 1.0 - prog
    saved = float((remaining_frac[~keep]).sum() / max(m, 1)) if pruned else 0.0
    return PruneDecision(keep=keep, pruned_count=pruned, est_compute_saved=saved,
                         frozen_baseline=frozen_baseline)


def corrected_advantages(rewards: np.ndarray, decision: PruneDecision) -> np.ndarray:
    """Center surviving rollouts' rewards against the unbiased frozen baseline.

    Use this instead of ``rewards - rewards.mean()`` after pruning: the survivor
    mean is biased upward/downward because pruning removed consensus rollouts,
    whereas ``decision.frozen_baseline`` reflects the full started group.

    Args:
        rewards: final rewards of the *kept* rollouts (length == decision.keep.sum()).
        decision: the :class:`PruneDecision` from :func:`prune_rollouts`.
    """
    rewards = np.asarray(rewards, dtype=float)
    return rewards - decision.frozen_baseline
