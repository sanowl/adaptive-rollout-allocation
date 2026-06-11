import numpy as np
import pytest

from aroll.allocation import allocate, uniform_allocation
from aroll.variance import Estimator, per_prompt_variance


@pytest.mark.parametrize("est", [Estimator.DR_GRPO, Estimator.RLOO])
def test_budget_and_bounds_respected(est):
    rng = np.random.default_rng(0)
    a = rng.uniform(0.01, 1.0, size=20)
    C, L, U = 200, 3, 32
    res = allocate(a, budget=C, n_min=L, n_max=U, estimator=est)
    assert res.total == C
    assert np.all(res.n >= L) and np.all(res.n <= U)


@pytest.mark.parametrize("est", [Estimator.DR_GRPO, Estimator.RLOO])
def test_more_budget_to_higher_coeff(est):
    # A prompt with larger a_q (nearer p=0.5) should get >= rollouts.
    a = np.array([0.01, 0.05, 0.25, 0.9])  # increasing informativeness
    res = allocate(a, budget=40, n_min=3, n_max=32, estimator=est)
    assert res.n[-1] >= res.n[0]
    assert np.all(np.diff(res.n) >= 0)     # monotone in coefficient


@pytest.mark.parametrize("est", [Estimator.DR_GRPO, Estimator.RLOO])
def test_beats_uniform_in_variance(est):
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=30)
    a = 4 * p * (1 - p)
    C = 30 * 8
    res = allocate(a, budget=C, n_min=3, n_max=32, estimator=est)
    n_uni = uniform_allocation(30, C, 3, 32)
    v_vip = per_prompt_variance(res.n, p, est).sum()
    v_uni = per_prompt_variance(n_uni, p, est).sum()
    assert v_vip <= v_uni + 1e-9          # optimal allocation never worse


def test_zero_coeff_gets_minimum():
    a = np.array([0.0, 0.0, 0.5, 0.5])
    res = allocate(a, budget=30, n_min=3, n_max=32, estimator=Estimator.DR_GRPO)
    assert res.n[0] == 3 and res.n[1] == 3
    assert res.total == 30


def test_allocate_rejects_bad_coefficients():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        allocate(np.array([0.1, np.nan, 0.2]), budget=30)
    with _pytest.raises(ValueError):
        allocate(np.array([0.1, -0.5]), budget=20)
    with _pytest.raises(ValueError):
        allocate(np.array([]), budget=10)
    with _pytest.raises(ValueError):
        allocate(np.array([0.1, 0.2]), budget=10, n_min=5, n_max=4)


def test_infeasible_budget_clamped():
    a = np.ones(5)
    # C below B*L: clamped to feasible band, still returns valid integers.
    res = allocate(a, budget=5, n_min=3, n_max=32, estimator=Estimator.DR_GRPO)
    assert not res.feasible
    assert np.all(res.n >= 3)
