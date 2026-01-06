"""
Penalty computation for penalized-constrained regression.

This module handles L1/L2 (ElasticNet) penalty computation with support
for excluding specific features from penalization.
"""

import numpy as np


def validate_penalty_exclude(penalty_exclude, feature_names):
    """Validate penalty_exclude and build a boolean mask.

    Parameters
    ----------
    penalty_exclude : list of str or None
        Feature names to exclude from penalty. If None, all features
        are penalized.
    feature_names : array-like
        All feature names (must be resolved/populated).

    Returns
    -------
    mask : ndarray of bool
        Boolean mask where True means the feature IS penalized,
        False means the feature is excluded from penalty.
    resolved : list
        Resolved list of excluded feature names (empty list if none).

    Raises
    ------
    ValueError
        If penalty_exclude contains feature names not in feature_names.

    Examples
    --------
    >>> mask, resolved = validate_penalty_exclude(['T1'], ['T1', 'LC', 'RC'])
    >>> mask
    array([False,  True,  True])
    >>> resolved
    ['T1']

    >>> mask, resolved = validate_penalty_exclude(None, ['T1', 'LC', 'RC'])
    >>> mask
    array([ True,  True,  True])
    >>> resolved
    []
    """
    n_params = len(feature_names)

    # Default: penalize all features
    if penalty_exclude is None:
        return np.ones(n_params, dtype=bool), []

    # Validate that all excluded names exist
    exclude_set = set(penalty_exclude)
    feature_set = set(feature_names)
    unknown = exclude_set - feature_set

    if unknown:
        raise ValueError(
            f"penalty_exclude contains unknown feature names: {unknown}. "
            f"Valid names are: {list(feature_names)}"
        )

    # Build mask: True = penalize, False = exclude
    mask = np.array([name not in exclude_set for name in feature_names])
    return mask, list(penalty_exclude)


def compute_elastic_net_penalty(coef, alpha, l1_ratio, penalty_mask=None):
    """Compute combined L1 + L2 (ElasticNet) penalty.

    The penalty is computed as:
        penalty = α * l1_ratio * ||β||₁ + 0.5 * α * (1 - l1_ratio) * ||β||₂²

    Where:
    - α (alpha) controls overall penalty strength
    - l1_ratio controls the mix between L1 and L2:
      - l1_ratio = 0: Pure Ridge (L2 only)
      - l1_ratio = 1: Pure Lasso (L1 only)
      - 0 < l1_ratio < 1: ElasticNet mix

    Parameters
    ----------
    coef : ndarray
        Coefficient values to penalize.
    alpha : float
        Overall penalty strength. If 0, returns 0.0 immediately.
    l1_ratio : float
        L1 vs L2 mixing parameter in [0, 1].
    penalty_mask : ndarray of bool, optional
        If provided, only coefficients where mask is True are penalized.
        This enables selective penalty exclusion (e.g., for intercept-like
        parameters in custom prediction functions).

    Returns
    -------
    float
        Total penalty value.

    Examples
    --------
    >>> coef = np.array([1.0, 2.0, 3.0])

    # Pure L2 (Ridge)
    >>> compute_elastic_net_penalty(coef, alpha=0.1, l1_ratio=0.0)
    0.7

    # Pure L1 (Lasso)
    >>> compute_elastic_net_penalty(coef, alpha=0.1, l1_ratio=1.0)
    0.6

    # Exclude first coefficient from penalty
    >>> mask = np.array([False, True, True])
    >>> compute_elastic_net_penalty(coef, alpha=0.1, l1_ratio=0.0, penalty_mask=mask)
    0.65
    """
    # Fast path: no penalty
    if alpha == 0:
        return 0.0

    # Apply mask to select only penalized coefficients
    if penalty_mask is not None:
        coef = coef[penalty_mask]

    # L2 penalty: 0.5 * α * (1 - l1_ratio) * ||β||²
    l2_penalty = 0.5 * alpha * (1 - l1_ratio) * np.sum(coef ** 2)

    # L1 penalty: α * l1_ratio * ||β||₁
    l1_penalty = alpha * l1_ratio * np.sum(np.abs(coef))

    return l2_penalty + l1_penalty
