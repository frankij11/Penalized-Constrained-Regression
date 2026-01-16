"""
Utility functions for penalized-constrained regression.

Provides:
- Asher's lot midpoint calculation
- Learning rate / slope conversions
- Correlation diagnostics (VIF, condition number)
"""

import numpy as np


def calculate_lot_midpoint(first_unit, last_unit, b):
    """
    Calculate Asher's lot midpoint for learning curve analysis.

    Formula: midpoint = ((1/(last-first+1)) *
              ((last+0.5)^(1+b) - (first-0.5)^(1+b)) / (1+b))^(1/b)

    Parameters
    ----------
    first_unit : float
        First unit in the lot.

    last_unit : float
        Last unit in the lot.

    b : float
        Learning curve slope (typically negative, e.g., ln(0.9)/ln(2) ≈ -0.152).

    Returns
    -------
    midpoint : float
        Lot midpoint.

    References
    ----------
    Asher, H. "Cost-Quantity Relationships in the Airframe Industry."

    Examples
    --------
    >>> # 90% learning curve, lot from unit 1 to 10
    >>> b = np.log(0.9) / np.log(2)
    >>> mp = calculate_lot_midpoint(1, 10, b)
    >>> print(f"Lot midpoint: {mp:.2f}")
    """
    n = last_unit - first_unit + 1
    numerator = ((last_unit + 0.5) ** (1 + b)) - ((first_unit - 0.5) ** (1 + b))
    denominator = (1 + b) * n
    midpoint = (numerator / denominator) ** (1 / b)
    return midpoint


def learning_rate_to_slope(learning_rate):
    """
    Convert learning rate (percentage) to slope exponent.

    Parameters
    ----------
    learning_rate : float
        Learning rate as decimal (e.g., 0.90 for 90%).

    Returns
    -------
    slope : float
        Slope exponent b where Y = T1 * X^b.

    Examples
    --------
    >>> b = learning_rate_to_slope(0.90)  # 90% learning
    >>> print(f"Slope: {b:.4f}")  # ≈ -0.152
    """
    return np.log(learning_rate) / np.log(2)


def slope_to_learning_rate(slope):
    """
    Convert slope exponent to learning rate (percentage).

    Parameters
    ----------
    slope : float
        Slope exponent b.

    Returns
    -------
    learning_rate : float
        Learning rate as decimal.

    Examples
    --------
    >>> lr = slope_to_learning_rate(-0.152)
    >>> print(f"Learning rate: {lr:.1%}")  # ≈ 90%
    """
    return 2 ** slope


def compute_vif(X):
    """
    Compute Variance Inflation Factors for each predictor.

    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing X_j on all other X's.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    vif : ndarray of shape (n_features,)
        VIF for each feature. VIF > 10 suggests harmful multicollinearity.

    Examples
    --------
    >>> vif = compute_vif(X)
    >>> for i, v in enumerate(vif):
    ...     print(f"Feature {i}: VIF = {v:.2f}")
    """
    from sklearn.linear_model import LinearRegression

    X = np.asarray(X)
    n_features = X.shape[1]
    vif = np.zeros(n_features)

    for i in range(n_features):
        # Regress X_i on all other X's
        X_others = np.delete(X, i, axis=1)
        X_i = X[:, i]

        if X_others.shape[1] == 0:
            vif[i] = 1.0
            continue

        model = LinearRegression()
        model.fit(X_others, X_i)
        r2 = model.score(X_others, X_i)

        if r2 >= 1:
            vif[i] = np.inf
        else:
            vif[i] = 1 / (1 - r2)

    return vif


def compute_condition_number(X):
    """
    Compute condition number of X'X matrix.

    κ = λ_max / λ_min from eigenvalue decomposition.
    Condition number > 30 indicates harmful multicollinearity.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    condition_number : float
        Condition number of X'X.

    Examples
    --------
    >>> kappa = compute_condition_number(X)
    >>> print(f"Condition number: {kappa:.2f}")
    >>> if kappa > 30:
    ...     print("Warning: Harmful multicollinearity detected")
    """
    X = np.asarray(X)
    XtX = X.T @ X
    eigenvalues = np.linalg.eigvalsh(XtX)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return np.inf

    return np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))


def multicollinearity_diagnostics(X, feature_names=None):
    """
    Comprehensive multicollinearity diagnostics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    feature_names : list of str or None, default=None
        Names for features.

    Returns
    -------
    diagnostics : dict
        Dictionary with:
        - 'correlation_matrix': Pairwise correlations
        - 'vif': Variance Inflation Factors
        - 'condition_number': Condition number of X'X
        - 'max_correlation': Maximum off-diagonal correlation
        - 'multicollinearity_warning': Boolean flag

    Examples
    --------
    >>> diag = multicollinearity_diagnostics(X, ['LC', 'RC'])
    >>> print(f"Max correlation: {diag['max_correlation']:.2f}")
    >>> print(f"Condition number: {diag['condition_number']:.2f}")
    """
    X = np.asarray(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f'X{i}' for i in range(n_features)]

    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Maximum off-diagonal correlation
    mask = ~np.eye(n_features, dtype=bool)
    max_corr = np.max(np.abs(corr_matrix[mask])) if n_features > 1 else 0

    # VIF
    vif = compute_vif(X)

    # Condition number
    cond_num = compute_condition_number(X)

    # Warning flag
    warning = (max_corr > 0.8) or (np.any(vif > 10)) or (cond_num > 30)

    return {
        'correlation_matrix': corr_matrix,
        'vif': dict(zip(feature_names, vif)),
        'condition_number': cond_num,
        'max_correlation': max_corr,
        'multicollinearity_warning': warning,
        'feature_names': feature_names
    }


def print_multicollinearity_report(X, feature_names=None):
    """Print formatted multicollinearity diagnostics report."""
    diag = multicollinearity_diagnostics(X, feature_names)

    print("=" * 60)
    print("Multicollinearity Diagnostics")
    print("=" * 60)

    print("\nCorrelation Matrix:")
    names = diag['feature_names']
    print("       " + "  ".join(f"{n:>8}" for n in names))
    for i, name in enumerate(names):
        row = "  ".join(f"{diag['correlation_matrix'][i,j]:8.3f}" for j in range(len(names)))
        print(f"{name:>6} {row}")

    print(f"\nMax off-diagonal correlation: {diag['max_correlation']:.3f}")
    print("  (Threshold: > 0.8 suggests multicollinearity)")

    print("\nVariance Inflation Factors:")
    for name, v in diag['vif'].items():
        flag = " WARNING" if v > 10 else ""
        print(f"  {name}: {v:.2f}{flag}")
    print("  (Threshold: > 10 indicates harmful multicollinearity)")

    print(f"\nCondition Number: {diag['condition_number']:.2f}")
    flag = " WARNING" if diag['condition_number'] > 30 else ""
    print(f"  (Threshold: > 30 indicates harmful multicollinearity){flag}")

    if diag['multicollinearity_warning']:
        print("\n  WARNING: Multicollinearity detected!")
        print("    Consider using regularization (Ridge, Lasso, ElasticNet)")
        print("    or constrained regression to stabilize estimates.")
    else:
        print("\n  No severe multicollinearity detected.")

    print("=" * 60)
