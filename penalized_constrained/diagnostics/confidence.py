"""
Confidence interval computation methods.

Provides bootstrap and Hessian-based standard error estimation
for penalized-constrained regression models.
"""

import numpy as np
from scipy.optimize import approx_fprime
from sklearn.utils.validation import check_is_fitted


def bootstrap_confidence_intervals(model_class, X, y, n_bootstrap=100,
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
