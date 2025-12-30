"""
Diagnostics module for penalized-constrained regression.

Provides:
- Generalized Degrees of Freedom (GDF) computation
- Fit statistics (SEE, SPE, adjusted R²)
- Bootstrap confidence intervals
- Hessian-based standard errors
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import approx_fprime


def compute_gdf_hu(n_samples, n_params, n_constraints, n_redundancies=0):
    """
    Compute Generalized Degrees of Freedom using Hu's formula.
    
    GDF = n - p - (# Constraints) + (# Redundancies)
    
    Parameters
    ----------
    n_samples : int
        Number of observations.
        
    n_params : int
        Number of estimated parameters (including intercept if fitted).
        
    n_constraints : int
        Number of constraints imposed.
        
    n_redundancies : int, default=0
        Number of redundant constraints (constraints derivable from others).
        
    Returns
    -------
    gdf : float
        Generalized degrees of freedom.
        
    References
    ----------
    Hu, S. (2010+). "Generalized Degrees of Freedom for Constrained CERs."
    Tecolote Research, PRT-191.
    """
    return n_samples - n_params - n_constraints + n_redundancies


def compute_gdf_gaines(n_active_predictors, n_equality_constraints=0, 
                        n_binding_inequality=0):
    """
    Compute degrees of freedom using Gaines et al. formula.
    
    df = |Active predictors| - (# equality) - (# binding inequality)
    
    Parameters
    ----------
    n_active_predictors : int
        Number of non-zero coefficients.
        
    n_equality_constraints : int, default=0
        Number of equality constraints.
        
    n_binding_inequality : int, default=0
        Number of binding inequality constraints (at bounds).
        
    Returns
    -------
    df : float
        Effective degrees of freedom.
        
    References
    ----------
    Gaines, B.R., Kim, J., & Zhou, H. (2018). "Algorithms for Fitting 
    the Constrained Lasso." JCGS, 27(4), 861-871.
    """
    return n_active_predictors - n_equality_constraints - n_binding_inequality


class ModelDiagnostics:
    """
    Compute diagnostic statistics for fitted penalized-constrained models.
    
    Parameters
    ----------
    model : PenalizedConstrainedRegression or PenalizedConstrainedCV
        Fitted model.
        
    X : array-like of shape (n_samples, n_features)
        Training data.
        
    y : array-like of shape (n_samples,)
        Target values.
        
    gdf_method : str, default='hu'
        Method for computing GDF: 'hu' or 'gaines'.
        
    Attributes
    ----------
    gdf : float
        Generalized degrees of freedom.
        
    see : float
        Standard Error of Estimate.
        
    spe : float
        Standard Percentage Error.
        
    r2 : float
        R-squared.
        
    adj_r2 : float
        GDF-adjusted R-squared.
        
    mape : float
        Mean Absolute Percentage Error.
        
    Examples
    --------
    >>> from penalized_constrained import PenalizedConstrainedRegression
    >>> from penalized_constrained.diagnostics import ModelDiagnostics
    >>> 
    >>> model = PenalizedConstrainedRegression(bounds=[(-1, 0), (-1, 0)])
    >>> model.fit(X, y)
    >>> 
    >>> diag = ModelDiagnostics(model, X, y)
    >>> print(f"GDF: {diag.gdf}")
    >>> print(f"SPE: {diag.spe:.2%}")
    >>> diag.summary()
    """
    
    def __init__(self, model, X, y, gdf_method='hu'):
        check_is_fitted(model)
        
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.gdf_method = gdf_method
        
        self.n_samples = len(y)
        self.y_pred = model.predict(X)
        self.residuals = self.y - self.y_pred
        
        # Compute all diagnostics
        self._compute_gdf()
        self._compute_fit_statistics()
    
    def _compute_gdf(self):
        """Compute generalized degrees of freedom."""
        # Count parameters
        n_params = len(self.model.coef_)
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept:
            if not hasattr(self.model, 'prediction_fn') or self.model.prediction_fn is None:
                n_params += 1
        
        # Count constraints
        n_constraints = self._count_specified_constraints()
        n_binding = self.model.n_active_constraints_
        
        if self.gdf_method == 'hu':
            # Hu's method: all specified constraints count
            self.gdf = compute_gdf_hu(
                self.n_samples, 
                n_params, 
                n_constraints
            )
        elif self.gdf_method == 'gaines':
            # Gaines' method: only binding constraints count
            n_active = np.sum(np.abs(self.model.coef_) > 1e-10)
            self.gdf = compute_gdf_gaines(
                n_active,
                n_equality_constraints=0,
                n_binding_inequality=n_binding
            )
        else:
            raise ValueError(f"Unknown gdf_method: {self.gdf_method}")
        
        # Ensure GDF is at least 1
        self.gdf = max(1, self.gdf)
    
    def _count_specified_constraints(self):
        """Count total number of specified (non-infinite) bounds."""
        count = 0
        
        if hasattr(self.model, '_bounds_parsed'):
            for lb, ub in self.model._bounds_parsed:
                if np.isfinite(lb):
                    count += 1
                if np.isfinite(ub):
                    count += 1
        
        if hasattr(self.model, 'intercept_bounds') and self.model.intercept_bounds:
            lb, ub = self.model.intercept_bounds
            if lb is not None and np.isfinite(lb):
                count += 1
            if ub is not None and np.isfinite(ub):
                count += 1
        
        return count
    
    def _compute_fit_statistics(self):
        """Compute all fit statistics."""
        n = self.n_samples
        
        # Sum of squared errors
        sse = np.sum(self.residuals ** 2)
        
        # Sum of squared percentage errors
        denom = np.where(np.abs(self.y) < 1e-10, 1e-10, self.y)
        pct_errors = self.residuals / denom
        sspe = np.sum(pct_errors ** 2)
        
        # Standard Error of Estimate (SEE)
        self.see = np.sqrt(sse / self.gdf)
        
        # Standard Percentage Error (SPE)
        self.spe = np.sqrt(sspe / self.gdf)
        
        # R-squared
        ss_total = np.sum((self.y - np.mean(self.y)) ** 2)
        self.r2 = 1 - sse / ss_total if ss_total > 0 else 0.0
        
        # GDF-adjusted R-squared
        if n - 1 > 0 and self.gdf > 0:
            self.adj_r2 = 1 - (1 - self.r2) * (n - 1) / self.gdf
        else:
            self.adj_r2 = self.r2
        
        # Mean Absolute Percentage Error (MAPE)
        self.mape = np.mean(np.abs(pct_errors))
        
        # Root Mean Squared Error
        self.rmse = np.sqrt(np.mean(self.residuals ** 2))
        
        # CV (Coefficient of Variation)
        if np.mean(self.y) != 0:
            self.cv = self.rmse / np.mean(np.abs(self.y))
        else:
            self.cv = np.inf
    
    def summary(self):
        """Print diagnostic summary."""
        print("=" * 60)
        print("Model Diagnostics")
        print("=" * 60)
        print(f"N samples: {self.n_samples}")
        print(f"N parameters: {len(self.model.coef_)}")
        print(f"GDF method: {self.gdf_method}")
        print(f"GDF: {self.gdf:.1f}")
        print(f"Active constraints: {self.model.n_active_constraints_}")
        print()
        print("Fit Statistics:")
        print(f"  R²: {self.r2:.4f}")
        print(f"  Adjusted R² (GDF): {self.adj_r2:.4f}")
        print(f"  SEE: {self.see:.4f}")
        print(f"  SPE: {self.spe:.2%}")
        print(f"  MAPE: {self.mape:.2%}")
        print(f"  RMSE: {self.rmse:.4f}")
        print(f"  CV: {self.cv:.2%}")
        print("=" * 60)
    
    def to_dict(self):
        """Return diagnostics as dictionary."""
        return {
            'n_samples': self.n_samples,
            'gdf': self.gdf,
            'gdf_method': self.gdf_method,
            'r2': self.r2,
            'adj_r2': self.adj_r2,
            'see': self.see,
            'spe': self.spe,
            'mape': self.mape,
            'rmse': self.rmse,
            'cv': self.cv,
            'n_active_constraints': self.model.n_active_constraints_
        }


def bootstrap_confidence_intervals(model_class, X, y, n_bootstrap=1000,
                                    confidence=0.95, random_state=None,
                                    **model_kwargs):
    """
    Compute bootstrap confidence intervals for coefficients.
    
    Parameters
    ----------
    model_class : class
        Model class (PenalizedConstrainedRegression or PenalizedConstrainedCV).
        
    X : array-like of shape (n_samples, n_features)
        Training data.
        
    y : array-like of shape (n_samples,)
        Target values.
        
    n_bootstrap : int, default=1000
        Number of bootstrap samples.
        
    confidence : float, default=0.95
        Confidence level.
        
    random_state : int or None, default=None
        Random seed for reproducibility.
        
    **model_kwargs : dict
        Keyword arguments passed to model constructor.
        
    Returns
    -------
    result : dict
        Dictionary with:
        - 'coef_mean': Mean of bootstrap coefficients
        - 'coef_std': Std of bootstrap coefficients
        - 'coef_ci_lower': Lower CI bound
        - 'coef_ci_upper': Upper CI bound
        - 'intercept_mean': Mean of bootstrap intercepts
        - 'intercept_ci': (lower, upper) tuple for intercept
        - 'bootstrap_coefs': All bootstrap coefficient samples
        
    Notes
    -----
    Bootstrap CIs for penalized models may be narrower than true uncertainty
    because the penalty constrains coefficient variability across resamples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(y)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Suppress verbose output during bootstrap
    model_kwargs['verbose'] = 0
    
    bootstrap_coefs = []
    bootstrap_intercepts = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Fit model
        try:
            model = model_class(**model_kwargs)
            model.fit(X_boot, y_boot)
            bootstrap_coefs.append(model.coef_.copy())
            bootstrap_intercepts.append(model.intercept_)
        except Exception:
            continue
    
    if len(bootstrap_coefs) < 10:
        raise ValueError("Too few successful bootstrap samples")
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    bootstrap_intercepts = np.array(bootstrap_intercepts)
    
    # Compute statistics
    alpha = 1 - confidence
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)
    
    result = {
        'coef_mean': np.mean(bootstrap_coefs, axis=0),
        'coef_std': np.std(bootstrap_coefs, axis=0),
        'coef_ci_lower': np.percentile(bootstrap_coefs, lower_pct, axis=0),
        'coef_ci_upper': np.percentile(bootstrap_coefs, upper_pct, axis=0),
        'intercept_mean': np.mean(bootstrap_intercepts),
        'intercept_std': np.std(bootstrap_intercepts),
        'intercept_ci': (
            np.percentile(bootstrap_intercepts, lower_pct),
            np.percentile(bootstrap_intercepts, upper_pct)
        ),
        'bootstrap_coefs': bootstrap_coefs,
        'n_successful': len(bootstrap_coefs)
    }
    
    return result


def hessian_standard_errors(model, X, y, epsilon=1e-5):
    """
    Estimate standard errors using the Hessian of the objective function.
    
    Parameters
    ----------
    model : PenalizedConstrainedRegression
        Fitted model.
        
    X : array-like of shape (n_samples, n_features)
        Training data.
        
    y : array-like of shape (n_samples,)
        Target values.
        
    epsilon : float, default=1e-5
        Step size for numerical differentiation.
        
    Returns
    -------
    se : ndarray
        Standard errors for coefficients (and intercept if applicable).
        
    Notes
    -----
    Uses numerical approximation of the Hessian. Assumes the objective
    function is approximately quadratic near the optimum.
    """
    check_is_fitted(model)
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    
    # Get current parameters
    if model.prediction_fn is not None:
        params = model.coef_.copy()
    elif model.fit_intercept:
        params = np.append(model.coef_, model.intercept_)
    else:
        params = model.coef_.copy()
    
    n_params = len(params)
    
    # Compute Hessian numerically
    hessian = np.zeros((n_params, n_params))
    
    def objective(p):
        return model._objective(p, X, y)
    
    for i in range(n_params):
        def grad_i(p):
            return approx_fprime(p, objective, epsilon)[i]
        
        hessian[i, :] = approx_fprime(params, grad_i, epsilon)
    
    # Make symmetric
    hessian = (hessian + hessian.T) / 2
    
    try:
        # Covariance is inverse Hessian (scaled by 2/n for MSE)
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(np.abs(cov)))
    except np.linalg.LinAlgError:
        # Hessian not invertible
        se = np.full(n_params, np.nan)
    
    return se
