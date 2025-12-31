"""
PenalizedConstrainedCV: Cross-validated penalized-constrained regression.

Automatically selects optimal alpha and l1_ratio using cross-validation,
information criteria (AIC/BIC), or generalized cross-validation (GCV).

For small samples, AIC/BIC/GCV are more stable than k-fold CV because they
don't require data splitting.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .core import PenalizedConstrainedRegression


def compute_effective_df(model, X, y):
    """
    Estimate effective degrees of freedom for a penalized/constrained model.

    For constrained optimization, df_effective is approximated by counting
    the number of coefficients not at their bounds (free parameters).

    Parameters
    ----------
    model : PenalizedConstrainedRegression
        Fitted model
    X : array-like
        Training data
    y : array-like
        Target values

    Returns
    -------
    df : float
        Estimated effective degrees of freedom
    """
    if not hasattr(model, 'coef_'):
        return len(model.coef_) + (1 if model.fit_intercept else 0)

    # Count free parameters (not at bounds)
    df = 0
    coef = model.coef_

    if hasattr(model, 'active_constraints_') and model.active_constraints_:
        # Each active constraint reduces df by ~1
        df = len(coef) - len(model.active_constraints_)
    else:
        # Approximation: count non-negligible coefficients
        df = np.sum(np.abs(coef) > 1e-8)

    # Add intercept if present
    if model.fit_intercept and model.prediction_fn is None:
        df += 1

    # For penalized models, df is reduced by regularization
    # Approximate using the shrinkage factor
    if hasattr(model, 'alpha') and model.alpha > 0:
        # Rough approximation: stronger penalty = fewer effective parameters
        shrinkage = 1.0 / (1.0 + model.alpha)
        df = df * shrinkage

    return max(1, df)  # At least 1 df


def compute_aic(rss, n, df):
    """
    Compute Akaike Information Criterion.

    AIC = n * log(RSS/n) + 2 * df

    Lower is better.
    """
    if rss <= 0 or n <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * df


def compute_bic(rss, n, df):
    """
    Compute Bayesian Information Criterion.

    BIC = n * log(RSS/n) + log(n) * df

    Lower is better. More conservative than AIC for large n.
    """
    if rss <= 0 or n <= 0:
        return np.inf
    return n * np.log(rss / n) + np.log(n) * df


def compute_aicc(rss, n, df):
    """
    Compute corrected AIC (AICc) for small samples.

    AICc = AIC + (2*df*(df+1)) / (n - df - 1)

    Recommended when n/df < 40. Converges to AIC as n increases.
    """
    if rss <= 0 or n <= 0 or n <= df + 1:
        return np.inf
    aic = compute_aic(rss, n, df)
    correction = (2 * df * (df + 1)) / (n - df - 1)
    return aic + correction


def compute_gcv(rss, n, df):
    """
    Compute Generalized Cross-Validation score.

    GCV = RSS / (n * (1 - df/n)^2)

    Lower is better. Approximates leave-one-out CV without data splitting.
    """
    if rss <= 0 or n <= 0 or df >= n:
        return np.inf
    return rss / (n * (1 - df / n) ** 2)


class PenalizedConstrainedCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated PenalizedConstrainedRegression with automatic hyperparameter tuning.

    Performs grid search over alpha and l1_ratio values using cross-validation,
    information criteria (AIC/BIC/AICc), or generalized cross-validation (GCV)
    to select the best hyperparameters, then refits on the full dataset.

    For small samples (n < 20), consider using selection='aicc' or selection='gcv'
    which don't require data splitting and are more stable.

    Parameters
    ----------
    alphas : array-like or None, default=None
        Array of alpha values to try. If None, uses np.logspace(-4, 0, 10).
        NOTE: For constrained models, use SMALL alpha values (< 1.0).
        Constraints already regularize; large alphas over-shrink.

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

    selection : str, default='cv'
        Method for selecting best hyperparameters:
        - 'cv': K-fold cross-validation (standard, but unstable for small n)
        - 'loocv': Leave-one-out CV (maximizes training data)
        - 'aic': Akaike Information Criterion
        - 'aicc': Corrected AIC (recommended for small samples)
        - 'bic': Bayesian Information Criterion (more conservative)
        - 'gcv': Generalized CV (approximates LOOCV without splitting)

    cv : int or cross-validation generator, default=5
        Number of folds or CV splitter. Only used when selection='cv'.
        For small samples, consider cv=3 or selection='loocv'.

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
        Best alpha selected by the selection method.

    l1_ratio_ : float
        Best l1_ratio selected by the selection method.

    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.

    intercept_ : float
        Intercept term.

    best_estimator_ : PenalizedConstrainedRegression
        The best estimator fitted on full data.

    cv_results_ : dict
        Results from hyperparameter search.

    converged_ : bool
        Whether the final fit converged.

    selection_scores_ : dict
        Scores for each (alpha, l1_ratio) combination.

    Examples
    --------
    >>> from penalized_constrained import PenalizedConstrainedCV
    >>> import numpy as np
    >>>
    >>> # Small sample - use AICc or GCV instead of CV
    >>> X = np.random.randn(10, 2)
    >>> y = X @ np.array([-0.15, -0.07]) + 4.5 + 0.1 * np.random.randn(10)
    >>>
    >>> model = PenalizedConstrainedCV(
    ...     alphas=np.logspace(-4, 0, 10),  # Small alphas for constrained models
    ...     l1_ratios=[0.0, 0.5, 1.0],
    ...     bounds=[(-1, 0), (-1, 0)],
    ...     selection='aicc'  # Better for small samples
    ... )
    >>> model.fit(X, y)
    >>> print(f"Best alpha: {model.alpha_:.6f}")
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
        init='ols',
        selection='cv',
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
        self.init = init
        self.selection = selection
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Fit using cross-validation or information criteria to find optimal hyperparameters.

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
        n_samples = X.shape[0]

        # Default hyperparameter grids - use smaller alphas for constrained models
        if self.alphas is None:
            alphas = np.logspace(-4, 0, 10)  # 0.0001 to 1.0
        else:
            alphas = np.array(self.alphas)

        if self.l1_ratios is None:
            l1_ratios = [0.0, 0.5, 1.0]
        else:
            l1_ratios = list(self.l1_ratios)

        # Include alpha=0 for constrained-only comparison
        if 0.0 not in alphas:
            alphas = np.concatenate([[0.0], alphas])

        # Route to appropriate selection method
        if self.selection in ['aic', 'aicc', 'bic', 'gcv']:
            return self._fit_information_criterion(X, y, alphas, l1_ratios)
        elif self.selection == 'loocv':
            return self._fit_cv(X, y, alphas, l1_ratios, cv=LeaveOneOut())
        else:  # 'cv' (default)
            return self._fit_cv(X, y, alphas, l1_ratios, cv=self.cv)

    def _fit_cv(self, X, y, alphas, l1_ratios, cv):
        """Fit using cross-validation (K-fold or LOOCV)."""
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
            init=self.init,
            method=self.method,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=0  # Suppress during CV
        )

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=cv,
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
        self.selection_scores_ = None  # Not used for CV

        # Copy attributes from best estimator
        self._copy_estimator_attributes()

        return self

    def _fit_information_criterion(self, X, y, alphas, l1_ratios):
        """
        Fit using information criteria (AIC, AICc, BIC, or GCV).

        These methods don't require data splitting, making them more stable
        for small sample sizes.
        """
        n_samples = X.shape[0]

        # Select criterion function
        criterion_funcs = {
            'aic': compute_aic,
            'aicc': compute_aicc,
            'bic': compute_bic,
            'gcv': compute_gcv
        }
        criterion_func = criterion_funcs[self.selection]

        # Store results for all combinations
        results = []
        best_score = np.inf
        best_model = None
        best_alpha = None
        best_l1_ratio = None

        if self.verbose > 0:
            print(f"Fitting {len(alphas) * len(l1_ratios)} models using {self.selection.upper()}...")

        for alpha in alphas:
            for l1_ratio in l1_ratios:
                # Fit model
                model = PenalizedConstrainedRegression(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    bounds=self.bounds,
                    feature_names=self.feature_names,
                    fit_intercept=self.fit_intercept,
                    intercept_bounds=self.intercept_bounds,
                    loss=self.loss,
                    prediction_fn=self.prediction_fn,
                    scale=self.scale,
                    init=self.init,
                    method=self.method,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    verbose=0
                )

                try:
                    model.fit(X, y)

                    # Compute RSS (residual sum of squares)
                    y_pred = model.predict(X)
                    rss = np.sum((y - y_pred) ** 2)

                    # Compute effective degrees of freedom
                    df = compute_effective_df(model, X, y)

                    # Compute criterion score (lower is better)
                    score = criterion_func(rss, n_samples, df)

                    results.append({
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'score': score,
                        'rss': rss,
                        'df': df,
                        'converged': model.converged_
                    })

                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_alpha = alpha
                        best_l1_ratio = l1_ratio

                except Exception as e:
                    if self.verbose > 0:
                        print(f"  Failed for alpha={alpha:.6f}, l1_ratio={l1_ratio:.2f}: {e}")
                    results.append({
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'score': np.inf,
                        'rss': np.inf,
                        'df': np.nan,
                        'converged': False
                    })

        if best_model is None:
            raise ValueError("All model fits failed. Check bounds and data.")

        # Store results
        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1_ratio
        self.best_estimator_ = best_model
        self.selection_scores_ = results
        self.best_score_ = best_score
        self.grid_search_ = None  # Not used for IC methods

        # Build cv_results_ in similar format to GridSearchCV for compatibility
        self.cv_results_ = {
            'param_alpha': [r['alpha'] for r in results],
            'param_l1_ratio': [r['l1_ratio'] for r in results],
            'mean_test_score': [-r['score'] for r in results],  # Negate so higher is better
            'std_test_score': [0.0] * len(results),  # No std for IC methods
            'rank_test_score': np.argsort([r['score'] for r in results]) + 1
        }

        # Copy attributes from best estimator
        self._copy_estimator_attributes()

        if self.verbose > 0:
            print(f"Best {self.selection.upper()}: {best_score:.4f}")
            print(f"Best alpha: {best_alpha:.6f}, l1_ratio: {best_l1_ratio:.2f}")

        return self

    def _copy_estimator_attributes(self):
        """Copy attributes from best_estimator_ to self."""
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.n_features_in_ = self.best_estimator_.n_features_in_
        self.converged_ = self.best_estimator_.converged_
        self.named_coef_ = self.best_estimator_.named_coef_
        self.active_constraints_ = self.best_estimator_.active_constraints_
        self.n_active_constraints_ = self.best_estimator_.n_active_constraints_

        if hasattr(self.best_estimator_, 'feature_names_in_'):
            self.feature_names_in_ = self.best_estimator_.feature_names_in_
    
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
        print(f"Selection method: {self.selection.upper()}")
        print(f"Best alpha: {self.alpha_:.6f}")
        print(f"Best l1_ratio: {self.l1_ratio_:.2f}")

        # Show score based on selection method
        if self.grid_search_ is not None:
            print(f"Best CV score: {self.grid_search_.best_score_:.6f}")
        elif hasattr(self, 'best_score_'):
            print(f"Best {self.selection.upper()} score: {self.best_score_:.6f}")

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
