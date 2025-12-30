"""
Utility functions for penalized-constrained regression.

Provides:
- Learning curve data generation with controlled correlation
- Asher's lot midpoint calculation
- Correlation diagnostics (VIF, condition number)
"""

import numpy as np
from scipy.optimize import minimize


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


def generate_correlated_learning_data(
    n_lots,
    T1=100,
    b=None,
    c=None,
    target_correlation=0.5,
    cv_error=0.1,
    base_quantity=5,
    growth_rate=1.5,
    random_state=None
):
    """
    Generate learning curve data with controlled correlation between predictors.
    
    Model: Y = T1 × (LotMidpoint)^b × (LotQuantity)^c × ε
    
    Uses optimization to achieve target correlation between log(midpoint) and 
    log(quantity), simulating realistic production ramp-up scenarios.
    
    Parameters
    ----------
    n_lots : int
        Number of lots to generate.
        
    T1 : float, default=100
        Theoretical first unit cost.
        
    b : float or None, default=None
        Learning curve slope. If None, uses 90% learning (≈ -0.152).
        
    c : float or None, default=None
        Lot size (rate) effect slope. If None, uses 95% rate (≈ -0.074).
        
    target_correlation : float, default=0.5
        Target correlation between log(midpoint) and log(quantity).
        Range: typically -0.3 to 0.9 based on SAR data.
        
    cv_error : float, default=0.1
        Coefficient of variation for multiplicative error (e.g., 0.1 = 10%).
        
    base_quantity : float, default=5
        Starting lot quantity.
        
    growth_rate : float, default=1.5
        Approximate growth multiplier for lot sizes.
        
    random_state : int or None, default=None
        Random seed for reproducibility.
        
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'X': Feature matrix [log(midpoint), log(quantity)]
        - 'y': Log of observed costs
        - 'X_original': Original space [midpoint, quantity]
        - 'y_original': Original space costs
        - 'lot_quantities': Array of lot quantities
        - 'lot_midpoints': Array of lot midpoints
        - 'actual_correlation': Achieved correlation
        - 'params': Dict of true parameters
        
    Examples
    --------
    >>> data = generate_correlated_learning_data(
    ...     n_lots=20,
    ...     T1=100,
    ...     target_correlation=0.7,
    ...     cv_error=0.1,
    ...     random_state=42
    ... )
    >>> X, y = data['X'], data['y']
    >>> print(f"Correlation: {data['actual_correlation']:.2f}")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Default slopes
    if b is None:
        b = np.log(0.9) / np.log(2)  # 90% learning ≈ -0.152
    if c is None:
        c = np.log(0.95) / np.log(2)  # 95% rate ≈ -0.074
    
    # Generate lot quantities to achieve target correlation
    lot_quantities = _optimize_lot_quantities(
        n_lots=n_lots,
        target_correlation=target_correlation,
        base_quantity=base_quantity,
        growth_rate=growth_rate,
        b_for_midpoint=b,
        random_state=random_state
    )
    
    # Calculate midpoints
    lot_midpoints = np.zeros(n_lots)
    cumulative = 0
    for i in range(n_lots):
        first_unit = cumulative + 1
        last_unit = first_unit + lot_quantities[i] - 1
        lot_midpoints[i] = calculate_lot_midpoint(first_unit, last_unit, b)
        cumulative = last_unit
    
    # Calculate actual correlation
    log_midpoints = np.log(lot_midpoints)
    log_quantities = np.log(lot_quantities)
    actual_correlation = np.corrcoef(log_midpoints, log_quantities)[0, 1]
    
    # Generate true costs (no error)
    true_costs = T1 * (lot_midpoints ** b) * (lot_quantities ** c)
    
    # Add multiplicative lognormal error
    if cv_error > 0:
        sigma = np.sqrt(np.log(1 + cv_error ** 2))
        errors = np.exp(np.random.normal(0, sigma, n_lots))
        observed_costs = true_costs * errors
    else:
        observed_costs = true_costs.copy()
    
    # Prepare output
    X_original = np.column_stack([lot_midpoints, lot_quantities])
    X_log = np.column_stack([log_midpoints, log_quantities])
    y_log = np.log(observed_costs)
    
    return {
        'X': X_log,
        'y': y_log,
        'X_original': X_original,
        'y_original': observed_costs,
        'lot_quantities': lot_quantities,
        'lot_midpoints': lot_midpoints,
        'actual_correlation': actual_correlation,
        'true_costs': true_costs,
        'params': {
            'T1': T1,
            'b': b,
            'c': c,
            'learning_rate': slope_to_learning_rate(b),
            'rate_effect': slope_to_learning_rate(c)
        }
    }


def _optimize_lot_quantities(n_lots, target_correlation, base_quantity, 
                              growth_rate, b_for_midpoint, random_state=None,
                              min_quantity=1, max_quantity=None):
    """
    Optimize lot quantities to achieve target correlation with midpoints.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if max_quantity is None:
        max_quantity = base_quantity * 100
    
    def calc_midpoints(quantities):
        midpoints = np.zeros(len(quantities))
        cumulative = 0
        for i in range(len(quantities)):
            first_unit = cumulative + 1
            last_unit = first_unit + quantities[i] - 1
            midpoints[i] = calculate_lot_midpoint(first_unit, last_unit, b_for_midpoint)
            cumulative = last_unit
        return midpoints
    
    def objective(quantities):
        try:
            midpoints = calc_midpoints(quantities)
            log_mp = np.log(midpoints)
            log_qty = np.log(quantities)
            
            if np.std(log_mp) == 0 or np.std(log_qty) == 0:
                return 1e6
            
            corr = np.corrcoef(log_mp, log_qty)[0, 1]
            if np.isnan(corr):
                return 1e6
            
            return (corr - target_correlation) ** 2
        except:
            return 1e6
    
    # Initialize with exponential growth + noise
    init_quantities = base_quantity * (growth_rate ** np.arange(n_lots))
    init_quantities += np.random.uniform(-1, 1, n_lots) * init_quantities * 0.2
    init_quantities = np.clip(init_quantities, min_quantity, max_quantity)
    init_quantities = np.round(init_quantities).astype(float)
    
    # Optimize
    bounds = [(min_quantity, max_quantity)] * n_lots
    
    result = minimize(
        objective,
        init_quantities,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    quantities = np.round(np.clip(result.x, min_quantity, max_quantity)).astype(int)
    return quantities


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
        flag = " ⚠️" if v > 10 else ""
        print(f"  {name}: {v:.2f}{flag}")
    print("  (Threshold: > 10 indicates harmful multicollinearity)")
    
    print(f"\nCondition Number: {diag['condition_number']:.2f}")
    flag = " ⚠️" if diag['condition_number'] > 30 else ""
    print(f"  (Threshold: > 30 indicates harmful multicollinearity){flag}")
    
    if diag['multicollinearity_warning']:
        print("\n⚠️  WARNING: Multicollinearity detected!")
        print("    Consider using regularization (Ridge, Lasso, ElasticNet)")
        print("    or constrained regression to stabilize estimates.")
    else:
        print("\n✓ No severe multicollinearity detected.")
    
    print("=" * 60)
