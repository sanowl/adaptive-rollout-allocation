import numpy as np

from aroll.variance import Estimator, per_prompt_variance, variance_coeff


def test_coeff_peaks_at_half():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    a = variance_coeff(p)
    assert a[2] == a.max()                 # maximised at p=0.5
    assert a[0] == 0.0 and a[-1] == 0.0    # zero at the extremes


def test_dr_grpo_matches_paper_formula():
    n, p, s2 = 8.0, 0.4, 1.3
    expected = (n - 1) / n**2 * 4 * s2 * p * (1 - p)
    got = per_prompt_variance(n, p, Estimator.DR_GRPO, s2)
    assert np.isclose(got, expected)


def test_rloo_matches_paper_formula():
    n, p, s2 = 10.0, 0.6, 0.9
    expected = 1.0 / (n - 1) * 4 * s2 * p * (1 - p)
    got = per_prompt_variance(n, p, Estimator.RLOO, s2)
    assert np.isclose(got, expected)


def test_variance_monotone_in_n():
    # More rollouts never increases per-prompt variance.
    p = 0.5
    ns = np.arange(3, 33)
    for est in (Estimator.DR_GRPO, Estimator.RLOO):
        v = per_prompt_variance(ns, p, est)
        assert np.all(np.diff(v) <= 1e-12)
