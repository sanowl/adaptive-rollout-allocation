"""Calibration & predictor-quality metrics.

Tools to judge how good a success-probability predictor is — both its point
accuracy (MAE, like the paper's Figure 3) and whether its *uncertainty* is
trustworthy (ECE, Brier, reliability curve). Use these to compare MC-dropout vs
deep-ensemble vs GP predictors honestly, rather than assuming the uncertainty is
meaningful.

* ``predictor_mae``  — mean absolute error vs known true probabilities (sim only).
* ``brier_score``    — mean squared error of probabilistic predictions on binary
                       outcomes; a proper scoring rule (lower = better).
* ``expected_calibration_error`` — gap between confidence and accuracy, binned.
* ``reliability_curve`` — per-bin (confidence, accuracy, count) for plotting.
* ``uncertainty_error_correlation`` — does higher predicted uncertainty actually
                       coincide with larger error? (a sanity check on uncertainty)
"""

from __future__ import annotations

import numpy as np


def predictor_mae(p_pred: np.ndarray, p_true: np.ndarray) -> float:
    """Mean absolute error against ground-truth probabilities (paper Fig. 3)."""
    return float(np.abs(np.asarray(p_pred) - np.asarray(p_true)).mean())


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions on binary outcomes."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Return ``(bin_confidence, bin_accuracy, bin_count)`` over equal-width bins.

    ``probs`` are predicted P(success); ``outcomes`` are binary {0,1} realisations.
    Empty bins are dropped.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        conf.append(float(probs[mask].mean()))
        acc.append(float(outcomes[mask].mean()))
        cnt.append(n)
    return np.array(conf), np.array(acc), np.array(cnt)


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted |confidence - accuracy|."""
    conf, acc, cnt = reliability_curve(probs, outcomes, n_bins)
    if cnt.size == 0:
        return 0.0
    return float(np.sum(cnt / cnt.sum() * np.abs(conf - acc)))


def outcomes_from_counts(successes: np.ndarray, counts: np.ndarray,
                         rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-prompt (successes, counts) into per-rollout binary outcomes and
    a matching repeated-prediction index, for ECE/Brier on observed rollouts.

    Returns ``(outcomes, prompt_index)`` so callers can align with predictions:
    ``ece = expected_calibration_error(p_pred[prompt_index], outcomes)``.
    """
    successes = np.asarray(successes, dtype=int)
    counts = np.asarray(counts, dtype=int)
    outcomes, idx = [], []
    for i, (s, c) in enumerate(zip(successes, counts)):
        outcomes.extend([1] * s + [0] * (c - s))
        idx.extend([i] * c)
    return np.array(outcomes, dtype=float), np.array(idx, dtype=int)


def uncertainty_error_correlation(uncertainty: np.ndarray, p_pred: np.ndarray,
                                  p_true: np.ndarray) -> float:
    """Spearman-free sanity check: Pearson corr between predicted uncertainty and
    actual absolute error. A trustworthy uncertainty is positively correlated
    with error (it is large exactly where the model is wrong)."""
    unc = np.asarray(uncertainty, dtype=float)
    err = np.abs(np.asarray(p_pred, dtype=float) - np.asarray(p_true, dtype=float))
    if unc.std() < 1e-12 or err.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(unc, err)[0, 1])
