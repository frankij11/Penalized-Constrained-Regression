"""
PenalizedConstrainedCV: Cross-validated penalized-constrained regression.

Automatically selects optimal alpha and l1_ratio using cross-validation.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .core import PenalizedConstrainedRegression


class PenalizedConstrainedCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated PenalizedConstrainedRegression with automatic hyperparameter tuning.
    
    Performs grid search over alpha and l1_ratio values using cross-validation
    to select the best hyperparameters, then refits on the full dataset.
    
    Parameters
    ----------
    alphas : array-like or None, default=None
        Array of alpha values to try. If None, uses np.logspace(-3, 2, 10).
        
    l1_ratios : array-like or None, default=None
        Array of l1_ratio values to try. If None, uses [0.0, 0.5, 1.0].
        
    bounds : list, tuple, dict, or None, default=None
        Coefficient bounds (same format as PenalizedConstrainedRegression).
        
    feature_names : list of str or None, default=None
        Names for coefficients.
        
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
        
    intercept_bounds : tuple or None, default=None
        Bounds for intercept.
        
    loss : str or callable, default='sspe'
        Loss function: 'sspe', 'sse', 'mse', or callable.
        
    prediction_fn : callable or None, default=None
        Custom prediction function.
        
    scale : bool, default=False
        Whether to standardize X internally.
        
    cv : int or cross-validation generator, default=5
        Number of folds or CV splitter.
        
    scoring : str or callable, default='neg_mean_squared_error'
        Scoring metric for cross-validation.
        
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all processors.
        
    method : str, default='SLSQP'
        Optimization method.
        
    max_iter : int, default=1000
        Maximum optimizer iterations.
        
    tol : float, default=1e-6
        Convergence tolerance.
        
    verbose : int, default=0
        Verbosity level.
        
    Attributes
    ----------
    alpha_ : float
        Best alpha selected by cross-validation.
        
    l1_ratio_ : float
        Best l1_ratio selected by cross-validation.
        
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
        
    intercept_ : float
        Intercept term.
        
    best_estimator_ : PenalizedConstrainedRegression
        The best estimator fitted on full data.
        
    cv_results_ : dict
        Cross-validation results from GridSearchCV.
        
    converged_ : bool
        Whether the final fit converged.
        
    named_coef_ : dict or None
        Coefficients as dict if feature_names provided.
        
    Examples
    --------
    >>> from penalized_constrained import PenalizedConstrainedCV
    >>> import numpy as np
    >>> 
    >>> X = np.random.randn(100, 2)
    >>> y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(100)
    >>> 
    >>> model = PenalizedConstrainedCV(
    ...     alphas=np.logspace(-2, 1, 5),
    ...     l1_ratios=[0.0, 0.5, 1.0],
    ...     bounds=[(-1, 0), (-1, 0)],
    ...     cv=5
    ... )
    >>> model.fit(X, y)
    >>> print(f"Best alpha: {model.alpha_:.4f}")
    >>> print(f"Best l1_ratio: {model.l1_ratio_:.2f}")
    """
    
    def __init__(
        self,
        alphas=None,
        l1_ratios=None,
        bounds=None,
        feature_names=None,
        fit_intercept=True,
        intercept_bounds=None,
        loss='sspe',
        prediction_fn=None,
        scale=False,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        method='SLSQP',
        max_iter=1000,
        tol=1e-6,
        verbose=0
    ):
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.bounds = bounds
        self.feature_names = feature_names
        self.fit_intercept = fit_intercept
        self.intercept_bounds = intercept_bounds
        self.loss = loss
        self.prediction_fn = prediction_fn
        self.scale = scale
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Fit using cross-validation to find optimal hyperparameters.
        
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
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Default hyperparameter grids
        if self.alphas is None:
            alphas = np.logspace(-3, 2, 10)
        else:
            alphas = np.array(self.alphas)
        
        if self.l1_ratios is None:
            l1_ratios = [0.0, 0.5, 1.0]
        else:
            l1_ratios = list(self.l1_ratios)
        
        # Include alpha=0 for constrained-only comparison
        if 0.0 not in alphas:
            alphas = np.concatenate([[0.0], alphas])
        
        param_grid = {
            'alpha': alphas,
            'l1_ratio': l1_ratios
        }
        
        # Base estimator
        base_estimator = PenalizedConstrainedRegression(
            bounds=self.bounds,
            feature_names=self.feature_names,
            fit_intercept=self.fit_intercept,
            intercept_bounds=self.intercept_bounds,
            loss=self.loss,
            prediction_fn=self.prediction_fn,
            scale=self.scale,
            method=self.method,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=0  # Suppress during CV
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True
        )
        
        grid_search.fit(X, y)
        
        # Store results
        self.alpha_ = grid_search.best_params_['alpha']
        self.l1_ratio_ = grid_search.best_params_['l1_ratio']
        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_
        self.grid_search_ = grid_search
        
        # Copy attributes from best estimator
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.n_features_in_ = self.best_estimator_.n_features_in_
        self.converged_ = self.best_estimator_.converged_
        self.named_coef_ = self.best_estimator_.named_coef_
        self.active_constraints_ = self.best_estimator_.active_constraints_
        self.n_active_constraints_ = self.best_estimator_.n_active_constraints_
        
        if hasattr(self.best_estimator_, 'feature_names_in_'):
            self.feature_names_in_ = self.best_estimator_.feature_names_in_
        
        return self
    
    def predict(self, X):
        """Predict using the best estimator."""
        check_is_fitted(self)
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        """Return R² score using the best estimator."""
        check_is_fitted(self)
        return self.best_estimator_.score(X, y)
    
    def get_cv_results_df(self):
        """Return cross-validation results as a pandas DataFrame."""
        import pandas as pd
        check_is_fitted(self)
        
        results = {
            'alpha': self.cv_results_['param_alpha'],
            'l1_ratio': self.cv_results_['param_l1_ratio'],
            'mean_score': self.cv_results_['mean_test_score'],
            'std_score': self.cv_results_['std_test_score'],
            'rank': self.cv_results_['rank_test_score']
        }
        
        df = pd.DataFrame(results)
        return df.sort_values('rank')
    
    def summary(self):
        """Print a summary of the cross-validation results."""
        check_is_fitted(self)
        
        print("=" * 60)
        print("PenalizedConstrainedCV Summary")
        print("=" * 60)
        print(f"Best alpha: {self.alpha_:.6f}")
        print(f"Best l1_ratio: {self.l1_ratio_:.2f}")
        print(f"Best CV score: {self.grid_search_.best_score_:.6f}")
        print(f"Converged: {self.converged_}")
        
        # Interpret regularization type
        if self.alpha_ == 0:
            reg_type = "Constrained-only (no penalty)"
        elif self.l1_ratio_ == 0:
            reg_type = "Ridge (L2)"
        elif self.l1_ratio_ == 1:
            reg_type = "Lasso (L1)"
        else:
            reg_type = f"ElasticNet (L1={self.l1_ratio_:.0%}, L2={1-self.l1_ratio_:.0%})"
        print(f"Regularization: {reg_type}")
        
        print("\nCoefficients:")
        if self.named_coef_ is not None:
            for name, coef in self.named_coef_.items():
                print(f"  {name}: {coef:.6f}")
        else:
            for i, coef in enumerate(self.coef_):
                print(f"  β_{i}: {coef:.6f}")
        
        if self.fit_intercept and self.best_estimator_.prediction_fn is None:
            print(f"  Intercept: {self.intercept_:.6f}")
        
        print(f"\nActive constraints: {self.n_active_constraints_}")
        for name, bound_type in self.active_constraints_:
            print(f"  {name}: {bound_type} bound")
        
        print("=" * 60)


# Alias
PCCV = PenalizedConstrainedCV
