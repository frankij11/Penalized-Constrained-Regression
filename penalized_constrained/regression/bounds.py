"""
Bounds parsing and validation for penalized-constrained regression.

This module handles the various formats for specifying coefficient bounds
and converts them to a standardized format for scipy.optimize.
"""

import numpy as np


def normalize_bound(bound):
    """Convert bound tuple, replacing None with infinity.

    Parameters
    ----------
    bound : tuple
        A (lower, upper) tuple where None represents unbounded.

    Returns
    -------
    tuple
        Normalized (lower, upper) with None replaced by +/- infinity.

    Examples
    --------
    >>> normalize_bound((0, None))
    (0, inf)
    >>> normalize_bound((None, 1))
    (-inf, 1)
    """
    lower = bound[0] if bound[0] is not None else -np.inf
    upper = bound[1] if bound[1] is not None else np.inf
    return (lower, upper)


def parse_bounds(bounds, n_params, feature_names=None):
    """Parse bounds parameter into list of (lower, upper) tuples.

    Supports multiple input formats for flexibility:
    - None: No bounds (unbounded)
    - Single tuple: Same bounds for all coefficients
    - List of tuples: Individual bounds for each coefficient
    - Dict: Named bounds using feature_names

    Parameters
    ----------
    bounds : None, tuple, list, or dict
        Coefficient bounds in any supported format.
    n_params : int
        Number of parameters to bound.
    feature_names : list of str, optional
        Feature names, required for dict-based bounds.

    Returns
    -------
    list of tuple
        List of (lower, upper) tuples, one per parameter.

    Raises
    ------
    ValueError
        If bounds format is invalid or incompatible with n_params.

    Examples
    --------
    >>> parse_bounds(None, 3)
    [(-inf, inf), (-inf, inf), (-inf, inf)]

    >>> parse_bounds((-1, 0), 3)
    [(-1, 0), (-1, 0), (-1, 0)]

    >>> parse_bounds({'LC': (-1, 0)}, 2, feature_names=['LC', 'RC'])
    [(-1, 0), (-inf, inf)]
    """
    if bounds is None:
        return [(-np.inf, np.inf)] * n_params

    # Dict-based bounds (requires feature_names)
    if isinstance(bounds, dict):
        if feature_names is None:
            raise ValueError(
                "feature_names must be provided when using dict-based bounds"
            )
        return _parse_dict_bounds(bounds, feature_names)

    # Single tuple for all coefficients
    if isinstance(bounds, tuple) and len(bounds) == 2:
        # Check if it's a single bound (not a tuple of tuples)
        if not isinstance(bounds[0], (tuple, list)):
            return [normalize_bound(bounds)] * n_params

    # List of tuples
    if hasattr(bounds, '__iter__'):
        bounds_list = list(bounds)
        if len(bounds_list) != n_params:
            raise ValueError(
                f"bounds has {len(bounds_list)} elements, "
                f"but model has {n_params} parameters"
            )
        return [normalize_bound(b) for b in bounds_list]

    raise ValueError("bounds must be None, tuple, list of tuples, or dict")


def _parse_dict_bounds(bounds, feature_names):
    """Parse dictionary-based bounds using feature names.

    Parameters
    ----------
    bounds : dict
        Dictionary mapping feature names to (lower, upper) tuples.
    feature_names : list of str
        Ordered list of feature names.

    Returns
    -------
    list of tuple
        List of (lower, upper) tuples in feature_names order.
    """
    parsed = []
    for name in feature_names:
        if name in bounds:
            parsed.append(normalize_bound(bounds[name]))
        else:
            # Unbounded if not specified
            parsed.append((-np.inf, np.inf))
    return parsed


def build_scipy_bounds(bounds_parsed, fit_intercept, intercept_bounds, has_prediction_fn):
    """Build bounds list for scipy.optimize.minimize.

    Appends intercept bounds if needed for standard linear models.

    Parameters
    ----------
    bounds_parsed : list of tuple
        Parsed coefficient bounds.
    fit_intercept : bool
        Whether the model includes an intercept.
    intercept_bounds : tuple or None
        Bounds for the intercept term.
    has_prediction_fn : bool
        Whether a custom prediction function is used.
        If True, intercept is handled by the prediction function.

    Returns
    -------
    list of tuple
        Complete bounds list for scipy.optimize.
    """
    scipy_bounds = list(bounds_parsed)

    # Add intercept bounds for standard linear models
    if fit_intercept and not has_prediction_fn:
        if intercept_bounds is not None:
            scipy_bounds.append(normalize_bound(intercept_bounds))
        else:
            scipy_bounds.append((-np.inf, np.inf))

    return scipy_bounds
