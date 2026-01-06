"""
Penalized-Constrained Regression estimator module.

This module provides the main PenalizedConstrainedRegression estimator class
along with supporting functions for bounds, loss, and penalty computation.
"""

from .estimator import PenalizedConstrainedRegression, PCRegression

# Re-export helper functions for advanced usage
from .bounds import parse_bounds, normalize_bound, build_scipy_bounds
from .loss import get_loss_function, sspe_loss, sse_loss, mse_loss, LOSS_FUNCTIONS
from .penalties import validate_penalty_exclude, compute_elastic_net_penalty

__all__ = [
    # Main class
    'PenalizedConstrainedRegression',
    'PCRegression',
    # Bounds utilities
    'parse_bounds',
    'normalize_bound',
    'build_scipy_bounds',
    # Loss functions
    'get_loss_function',
    'sspe_loss',
    'sse_loss',
    'mse_loss',
    'LOSS_FUNCTIONS',
    # Penalty utilities
    'validate_penalty_exclude',
    'compute_elastic_net_penalty',
]
