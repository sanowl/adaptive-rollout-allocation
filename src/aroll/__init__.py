"""aroll — Adaptive Rollout Allocation (VIP) with practical extensions.

Faithful implementation of "Adaptive Rollout Allocation for Online RL with
Verifiable Rewards" (VIP, ICLR 2026), plus five recommended extensions:

  1. Learned MLP predictor in place of the Gaussian process   (predictor.py)
  2. Difficulty buckets to prevent batch collapse              (buckets.py)
  3. Rollout reuse via a staleness-weighted replay buffer      (replay.py)
  4. Online rollout pruning during generation (arrol-style)    (pruning.py)
  5. Prefix-level allocation for agent tasks (TRACE-style)     (prefix.py)
"""

from .allocation import AllocationResult, allocate, uniform_allocation
from .buckets import BucketConfig, assign_buckets, select_balanced_batch
from .calibration import (
    brier_score, expected_calibration_error, outcomes_from_counts,
    predictor_mae, reliability_curve, uncertainty_error_correlation,
)
from .env import MockRLVREnv
from .predictor import EMAPredictor, EnsemblePredictor, Prediction, RolloutPredictor
from .prefix import Prefix, allocate_prefixes, expand_tree_budget
from .pruning import PruneDecision, corrected_advantages, prune_rollouts
from .replay import ReplayBuffer, blend_with_replay
from .scoring import boundary_score, coefficients
from .variance import Estimator, per_prompt_variance, variance_coeff
from .vip import VIPAllocator, VIPConfig

__all__ = [
    "allocate", "uniform_allocation", "AllocationResult",
    "Estimator", "per_prompt_variance", "variance_coeff",
    "RolloutPredictor", "EnsemblePredictor", "EMAPredictor", "Prediction",
    "coefficients", "boundary_score",
    "predictor_mae", "brier_score", "expected_calibration_error",
    "reliability_curve", "outcomes_from_counts", "uncertainty_error_correlation",
    "BucketConfig", "assign_buckets", "select_balanced_batch",
    "ReplayBuffer", "blend_with_replay",
    "prune_rollouts", "PruneDecision", "corrected_advantages",
    "Prefix", "allocate_prefixes", "expand_tree_budget",
    "MockRLVREnv",
    "VIPAllocator", "VIPConfig",
]
__version__ = "0.1.0"
