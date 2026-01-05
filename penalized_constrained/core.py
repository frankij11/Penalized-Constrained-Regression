"""
PenalizedConstrainedRegression: Penalized regression with coefficient constraints.

This module implements the core estimator combining L1/L2 regularization with
bound constraints on coefficients, using SSPE (Sum of Squared Percentage Errors)
as the default loss function for cost estimation applications.

Reference: "Small Data, Big Problems: Can Constraints and Penalties Save Regression?"
           ICEAA 2026 Professional Development & Training Workshop
"""

import numpy as np
import warnings
from datetime import datetime
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


class PenalizedConstrainedRegression(BaseEstimator, RegressorMixin):
    """
    Penalized regression with coefficient constraints.
    
    Minimizes: L(β) + α·l1_ratio·||β||₁ + 0.5·α·(1-l1_ratio)·||β||₂²
    Subject to: bounds[i][0] ≤ β[i] ≤ bounds[i][1]
    
    Parameters
    ----------
    alpha : float, default=0.0
        Overall penalty strength. α=0 gives constrained-only optimization.
        
    l1_ratio : float, default=0.0
        ElasticNet mixing parameter:
        - l1_ratio=0: Ridge (L2 only)
        - l1_ratio=1: Lasso (L1 only)
        - 0 < l1_ratio < 1: ElasticNet
        
    bounds : list, tuple, dict, or None, default=None
        Coefficient bounds. Can be:
        - None: No bounds (uses (-inf, inf) for all)
        - Single tuple (lower, upper): Same bounds for all coefficients
        - List of tuples: [(lb0, ub0), (lb1, ub1), ...]
        - Dict with feature names: {'LC': (-1, 0), 'RC': (-0.5, 0)}
          Requires feature_names to be set.
        Use None in tuple for unbounded: (0, None) means ≥ 0.
        
    feature_names : list of str or None, default=None
        Names for coefficients. Enables dict-based bounds and named access.
        Example: ['T1', 'LC', 'RC'] for learning curve model.
        
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
        
    intercept_bounds : tuple or None, default=None
        Bounds for intercept: (lower, upper). Use None for unbounded.
        
    loss : str or callable, default='sspe'
        Loss function to minimize:
        - 'sspe': Sum of Squared Percentage Errors (default, MUPE-consistent)
        - 'sse': Sum of Squared Errors
        - 'mse': Mean Squared Error
        - callable: Custom loss function with signature loss(y_true, y_pred) -> float
        
    prediction_fn : callable or None, default=None
        Custom prediction function with signature: prediction_fn(X, params) -> y_pred
        where params includes all parameters (coefficients + intercept if applicable).
        If None, uses standard linear prediction: X @ coef + intercept.
        
        Example for learning curve Y = T1 * X1^b * X2^c:
            def lc_func(X, params):
                T1, b, c = params[0], params[1], params[2]
                return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)
        
    scale : bool, default=False
        Whether to standardize X internally before fitting.
        Coefficients are transformed back to original scale after fitting.
        
    init : str or array-like, default='ols'
        Initialization strategy:
        - 'ols': Start from OLS solution (clipped to bounds)
        - 'zeros': Start from zeros
        - array-like: User-provided initial values
        
    method : str, default='SLSQP'
        Optimization method for scipy.optimize.minimize:
        - 'SLSQP': Sequential Least-Squares Quadratic Programming (default)
        - 'L-BFGS-B': Limited-memory BFGS with bounds
        - 'trust-constr': Trust-region constrained optimization
        - 'COBYLA': Constrained Optimization BY Linear Approximation
        
    max_iter : int, default=1000
        Maximum number of optimizer iterations.
        
    tol : float, default=1e-6
        Tolerance for convergence.
        
    verbose : int, default=0
        Verbosity level. 0=silent, 1=warnings, 2=detailed.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
        
    intercept_ : float
        Intercept term. 0.0 if fit_intercept=False.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_names_in_ : ndarray of shape (n_features,) or None
        Feature names seen during fit.
        
    converged_ : bool
        Whether the optimizer converged successfully.
        
    active_constraints_ : list of tuples
        List of (index_or_name, 'lower'|'upper') for binding constraints.
        
    n_active_constraints_ : int
        Number of active (binding) constraints at the solution.
        
    optimization_result_ : scipy.optimize.OptimizeResult
        Full optimization result from scipy.
        
    named_coef_ : dict or None
        Coefficients as dict if feature_names provided.
        
    Examples
    --------
    >>> import numpy as np
    >>> from penalized_constrained import PenalizedConstrainedRegression
    >>> 
    >>> # Basic usage with bounds
    >>> X = np.random.randn(100, 2)
    >>> y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(100)
    >>> 
    >>> model = PenalizedConstrainedRegression(
    ...     alpha=0.1,
    ...     bounds=[(-1, 0), (-1, 0)],  # Both coefficients ≤ 0
    ...     loss='sspe'
    ... )
    >>> model.fit(X, y)
    >>> print(model.coef_)
    
    >>> # With named coefficients
    >>> model = PenalizedConstrainedRegression(
    ...     feature_names=['LC', 'RC'],
    ...     bounds={'LC': (-1, 0), 'RC': (-0.5, 0)},
    ...     alpha=0.1
    ... )
    >>> model.fit(X, y)
    >>> print(model.named_coef_)
    
    >>> # Custom prediction function for learning curve
    >>> def lc_func(X, params):
    ...     T1, b, c = params
    ...     return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)
    >>> 
    >>> model = PenalizedConstrainedRegression(
    ...     prediction_fn=lc_func,
    ...     feature_names=['T1', 'LC', 'RC'],
    ...     bounds={'T1': (0, None), 'LC': (-1, 0), 'RC': (-1, 0)},
    ...     fit_intercept=False
    ... )
    """
    
    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        bounds=None,
        feature_names=None,
        fit_intercept=True,
        intercept_bounds=None,
        loss='sspe',
        prediction_fn=None,
        scale=False,
        init='ols',
        method='SLSQP',
        max_iter=1000,
        tol=1e-6,
        verbose=0
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.bounds = bounds
        self.feature_names = feature_names
        self.fit_intercept = fit_intercept
        self.intercept_bounds = intercept_bounds
        self.loss = loss
        self.prediction_fn = prediction_fn
        self.scale = scale
        self.init = init
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def _get_loss_function(self):
        """Return the loss function based on self.loss parameter."""
        if callable(self.loss):
            return self.loss
        
        if self.loss == 'sspe':
            def sspe(y_true, y_pred):
                # Add small epsilon only where division by zero would occur
                denom = np.where(np.abs(y_true) < 1e-10, 1e-10, y_true)
                return np.sum(((y_true - y_pred) / denom) ** 2)
            return sspe
        
        elif self.loss == 'sse':
            def sse(y_true, y_pred):
                return np.sum((y_true - y_pred) ** 2)
            return sse
        
        elif self.loss == 'mse':
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)
            return mse
        
        else:
            raise ValueError(
                f"Unknown loss '{self.loss}'. Use 'sspe', 'sse', 'mse', or callable."
            )
    
    def _parse_bounds(self, n_params):
        """Parse bounds parameter into list of (lower, upper) tuples.
        
        n_params is the number of parameters to bound, which equals:
        - n_features for standard linear models
        - len(feature_names) for custom prediction functions
        """
        if self.bounds is None:
            return [(-np.inf, np.inf)] * n_params
        
        # Dict-based bounds (requires feature_names)
        if isinstance(self.bounds, dict):
            if self.feature_names is None:
                raise ValueError(
                    "feature_names must be provided when using dict-based bounds"
                )
            if len(self.feature_names) != n_params:
                raise ValueError(
                    f"feature_names has {len(self.feature_names)} elements, "
                    f"but model has {n_params} parameters"
                )
            
            parsed = []
            for name in self.feature_names:
                if name in self.bounds:
                    bound = self.bounds[name]
                else:
                    bound = (-np.inf, np.inf)
                parsed.append(self._normalize_bound(bound))
            return parsed
        
        # Single tuple for all coefficients
        if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
            if not isinstance(self.bounds[0], (tuple, list)):
                return [self._normalize_bound(self.bounds)] * n_params
        
        # List of tuples
        if hasattr(self.bounds, '__iter__'):
            bounds_list = list(self.bounds)
            if len(bounds_list) != n_params:
                raise ValueError(
                    f"bounds has {len(bounds_list)} elements, "
                    f"but model has {n_params} parameters"
                )
            return [self._normalize_bound(b) for b in bounds_list]
        
        raise ValueError(
            "bounds must be None, tuple, list of tuples, or dict"
        )
    
    def _normalize_bound(self, bound):
        """Convert bound tuple, replacing None with inf."""
        lower = bound[0] if bound[0] is not None else -np.inf
        upper = bound[1] if bound[1] is not None else np.inf
        return (lower, upper)
    
    def _predict_internal(self, X, params):
        """Make predictions using custom function or linear model."""
        if self.prediction_fn is not None:
            return self.prediction_fn(X, params)
        
        if self.fit_intercept:
            coef = params[:-1]
            intercept = params[-1]
        else:
            coef = params
            intercept = 0.0
        
        return X @ coef + intercept
    
    def _objective(self, params, X, y):
        """Compute objective: loss + penalty terms."""
        # Get predictions
        y_pred = self._predict_internal(X, params)
        
        # Loss term
        loss_func = self._get_loss_function()
        loss_value = loss_func(y, y_pred)
        
        # Extract coefficients for penalty (exclude intercept if present)
        if self.prediction_fn is not None:
            # For custom functions, penalize all params
            coef = params
        elif self.fit_intercept:
            coef = params[:-1]
        else:
            coef = params
        
        # L2 penalty: 0.5 * α * (1 - l1_ratio) * ||β||²
        l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(coef ** 2)
        
        # L1 penalty: α * l1_ratio * ||β||₁
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(coef))
        
        return loss_value + l2_penalty + l1_penalty
    
    def _get_initial_params(self, X, y, bounds_parsed):
        """Get starting parameters, clipped to bounds."""
        n_features = X.shape[1]
        
        if self.prediction_fn is not None:
            # For custom functions, use zeros or user-provided init
            if isinstance(self.init, str):
                if self.init == 'zeros':
                    params_init = np.zeros(n_features)
                else:
                    # Default initialization for custom functions
                    params_init = np.ones(n_features)
            else:
                params_init = np.array(self.init, dtype=float)
        
        elif self.init == 'ols':
            try:
                ols = LinearRegression(fit_intercept=self.fit_intercept)
                ols.fit(X, y)
                coef_init = ols.coef_.copy()
                intercept_init = ols.intercept_ if self.fit_intercept else 0.0
            except Exception:
                coef_init = np.zeros(n_features)
                intercept_init = np.mean(y) if self.fit_intercept else 0.0
            
            if self.fit_intercept:
                params_init = np.append(coef_init, intercept_init)
            else:
                params_init = coef_init
        
        elif self.init == 'zeros':
            if self.fit_intercept:
                params_init = np.append(np.zeros(n_features), np.mean(y))
            else:
                params_init = np.zeros(n_features)
        
        else:
            params_init = np.array(self.init, dtype=float)
        
        # Clip to bounds
        for i, (lb, ub) in enumerate(bounds_parsed):
            if i < len(params_init):
                params_init[i] = np.clip(params_init[i], lb, ub)
        
        # Clip intercept bounds (last element if fit_intercept and no custom fn)
        if (self.fit_intercept and self.prediction_fn is None 
                and self.intercept_bounds is not None):
            lb, ub = self._normalize_bound(self.intercept_bounds)
            params_init[-1] = np.clip(params_init[-1], lb, ub)
        
        return params_init
    
    def _build_scipy_bounds(self, bounds_parsed):
        """Build bounds in scipy format."""
        scipy_bounds = list(bounds_parsed)
        
        # Add intercept bounds if needed
        if self.fit_intercept and self.prediction_fn is None:
            if self.intercept_bounds is not None:
                scipy_bounds.append(self._normalize_bound(self.intercept_bounds))
            else:
                scipy_bounds.append((-np.inf, np.inf))
        
        return scipy_bounds
    
    def fit(self, X, y):
        """
        Fit the penalized-constrained regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Record fit start time
        self.fit_datetime_ = datetime.now()
        fit_start_time = time.perf_counter()

        # Extract feature names from DataFrame before validation converts to numpy
        _df_columns = None
        if hasattr(X, 'columns'):
            _df_columns = list(X.columns)

        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Determine number of parameters
        if self.prediction_fn is not None:
            # For custom functions, use feature_names to determine n_params
            if self.feature_names is None:
                raise ValueError(
                    "feature_names must be provided when using prediction_fn"
                )
            n_params = len(self.feature_names)
        else:
            n_params = self.n_features_in_

        # Store feature names - auto-generate if not provided
        if self.feature_names is not None:
            self.feature_names_in_ = np.array(self.feature_names)
        elif _df_columns is not None:
            # X was a pandas DataFrame - use column names
            self.feature_names_in_ = np.array(_df_columns)
        else:
            # Generate default names: X1_coef, X2_coef, etc.
            self.feature_names_in_ = np.array([f"X{i+1}_coef" for i in range(n_params)])
        
        # Scale if requested
        if self.scale:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std == 0] = 1.0
            X_work = (X - self._X_mean) / self._X_std
        else:
            X_work = X
        
        # Parse bounds based on number of parameters
        bounds_parsed = self._parse_bounds(n_params)
        self._bounds_parsed = bounds_parsed
        
        # Get initial parameters
        params_init = self._get_initial_params(X_work, y, bounds_parsed)
        
        # Build scipy bounds
        scipy_bounds = self._build_scipy_bounds(bounds_parsed)
        
        # Set up optimizer options
        options = {'maxiter': self.max_iter}
        if self.method in ('SLSQP', 'L-BFGS-B'):
            options['ftol'] = self.tol
        elif self.method == 'trust-constr':
            options['gtol'] = self.tol
        
        if self.verbose >= 2:
            options['disp'] = True
        
        # Optimize
        result = minimize(
            fun=self._objective,
            x0=params_init,
            args=(X_work, y),
            method=self.method,
            bounds=scipy_bounds,
            options=options
        )
        
        self.optimization_result_ = result
        self.converged_ = result.success
        
        # Warn if not converged
        if not self.converged_ and self.verbose >= 1:
            warnings.warn(
                f"Optimizer did not converge: {result.message}. "
                f"Try different initialization (init='zeros') or increase max_iter.",
                UserWarning
            )
        
        # Extract coefficients
        if self.prediction_fn is not None:
            # For custom functions, all params are "coefficients"
            self.coef_ = result.x.copy()
            self.intercept_ = 0.0
        elif self.fit_intercept:
            coef_work = result.x[:-1]
            self.intercept_ = result.x[-1]
        else:
            coef_work = result.x
            self.intercept_ = 0.0
        
        # Unscale coefficients if needed
        if self.scale and self.prediction_fn is None:
            self.coef_ = coef_work / self._X_std
            self.intercept_ = self.intercept_ - np.sum(self.coef_ * self._X_mean)
        elif self.prediction_fn is None:
            self.coef_ = coef_work
        
        # Create named coefficients dict (always available now)
        self.named_coef_ = dict(zip(self.feature_names_in_, self.coef_))
        
        # Compute active constraints
        self._compute_active_constraints()

        # Record fit duration
        self.fit_duration_seconds_ = time.perf_counter() - fit_start_time

        return self
    
    def _compute_active_constraints(self, tol=1e-6):
        """Identify binding constraints at the solution."""
        self.active_constraints_ = []
        
        for i, (lb, ub) in enumerate(self._bounds_parsed):
            if i >= len(self.coef_):
                break
            
            # Get name or index
            if self.feature_names_in_ is not None:
                name = self.feature_names_in_[i]
            else:
                name = i
            
            if np.isfinite(lb) and np.abs(self.coef_[i] - lb) < tol:
                self.active_constraints_.append((name, 'lower'))
            elif np.isfinite(ub) and np.abs(self.coef_[i] - ub) < tol:
                self.active_constraints_.append((name, 'upper'))
        
        # Check intercept bounds
        if (self.fit_intercept and self.prediction_fn is None 
                and self.intercept_bounds is not None):
            lb, ub = self._normalize_bound(self.intercept_bounds)
            if np.isfinite(lb) and np.abs(self.intercept_ - lb) < tol:
                self.active_constraints_.append(('intercept', 'lower'))
            elif np.isfinite(ub) and np.abs(self.intercept_ - ub) < tol:
                self.active_constraints_.append(('intercept', 'upper'))
        
        self.n_active_constraints_ = len(self.active_constraints_)
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values in the same space as training y.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted "
                f"with {self.n_features_in_} features"
            )
        
        if self.prediction_fn is not None:
            return self.prediction_fn(X, self.coef_)
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """
        Return R² score of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True values.
            
        Returns
        -------
        score : float
            R² score.
        """
        from sklearn.metrics import r2_score
        check_is_fitted(self)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'bounds': self.bounds,
            'feature_names': self.feature_names,
            'fit_intercept': self.fit_intercept,
            'intercept_bounds': self.intercept_bounds,
            'loss': self.loss,
            'prediction_fn': self.prediction_fn,
            'scale': self.scale,
            'init': self.init,
            'method': self.method,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def summary(self):
        """Print a summary of the fitted model."""
        check_is_fitted(self)
        
        print("=" * 60)
        print("PenalizedConstrainedRegression Summary")
        print("=" * 60)
        print(f"Loss function: {self.loss}")
        print(f"Alpha (penalty): {self.alpha}")
        print(f"L1 ratio: {self.l1_ratio}")
        print(f"Converged: {self.converged_}")
        print(f"Active constraints: {self.n_active_constraints_}")
        
        print("\nCoefficients:")
        if self.named_coef_ is not None:
            for name, coef in self.named_coef_.items():
                print(f"  {name}: {coef:.6f}")
        else:
            for i, coef in enumerate(self.coef_):
                print(f"  β_{i}: {coef:.6f}")
        
        if self.fit_intercept and self.prediction_fn is None:
            print(f"  Intercept: {self.intercept_:.6f}")
        
        if self.active_constraints_:
            print("\nActive constraints:")
            for name, bound_type in self.active_constraints_:
                print(f"  {name}: {bound_type} bound")
        
        print("=" * 60)


# Alias for convenience
PCRegression = PenalizedConstrainedRegression
