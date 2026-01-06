"""
Core diagnostic computation for penalized-constrained regression.

Provides:
- Generalized Degrees of Freedom (GDF) computation
- ModelDiagnostics class for fit statistics
"""

import numpy as np
from typing import Optional
from sklearn.utils.validation import check_is_fitted


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

    alpha_trace : Optional[AlphaTraceResult]
        Cached alpha trace results (computed via compute_alpha_trace method).

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
    >>>
    >>> # Compute alpha trace once, reuse in summaries
    >>> diag.compute_alpha_trace()
    >>> report = diag.summary()  # Includes cached alpha trace
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

        # Alpha trace cache (computed on demand)
        self.alpha_trace = None

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

    def compute_alpha_trace(
        self,
        l1_ratios: Optional[list] = None,
        n_alphas: int = 30,
        force_recompute: bool = False
    ):
        """
        Compute and cache alpha trace analysis.

        This method computes the alpha trace once and stores it in self.alpha_trace.
        Subsequent calls to summary() will use the cached result unless
        force_recompute=True is passed.

        Parameters
        ----------
        l1_ratios : list of float, optional
            L1 ratios to evaluate. Default is [0.0, 0.5, 1.0].
        n_alphas : int, default=30
            Number of alpha values to evaluate per l1_ratio.
        force_recompute : bool, default=False
            If True, recompute even if already cached.

        Returns
        -------
        AlphaTraceResult
            The computed alpha trace result.

        Examples
        --------
        >>> diag = ModelDiagnostics(model, X, y)
        >>> diag.compute_alpha_trace(l1_ratios=[0.0, 0.5, 1.0], n_alphas=50)
        >>> print(diag.alpha_trace.summary_df)
        >>> print(diag.alpha_trace.optimal)
        """
        if self.alpha_trace is not None and not force_recompute:
            return self.alpha_trace

        from .alpha_trace import compute_alpha_trace, summarize_alpha_trace, find_optimal_alpha
        from .dataclasses import AlphaTraceResult

        if l1_ratios is None:
            l1_ratios = [0.0, 0.5, 1.0]

        # Pass model directly - clone approach preserves all settings including prediction_fn
        trace_df = compute_alpha_trace(
            self.model,
            self.X, self.y,
            l1_ratios=l1_ratios,
            n_alphas=n_alphas,
        )

        summary_df = summarize_alpha_trace(trace_df)
        optimal = find_optimal_alpha(trace_df)

        self.alpha_trace = AlphaTraceResult(
            trace_df=trace_df,
            summary_df=summary_df,
            optimal=optimal,
            l1_ratios=l1_ratios,
            n_alphas=n_alphas,
        )

        return self.alpha_trace

    def plot_alpha_trace(self, figsize=(14, 10), save_path=None, show_legend=True):
        """
        Plot the cached alpha trace.

        Must call compute_alpha_trace() first.

        Parameters
        ----------
        figsize : tuple, default=(14, 10)
            Figure size.
        save_path : str, optional
            Path to save the figure.
        show_legend : bool, default=True
            Whether to show legend.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if self.alpha_trace is None:
            raise ValueError("Alpha trace not computed. Call compute_alpha_trace() first.")

        from .alpha_trace import plot_alpha_trace
        return plot_alpha_trace(
            self.alpha_trace.trace_df,
            figsize=figsize,
            save_path=save_path,
            show_legend=show_legend
        )

    def summary(
        self,
        full: bool = True,
        ci_method: str = 'hessian',
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
        include_alpha_trace: bool = False,
    ):
        """
        Generate comprehensive summary report.

        Parameters
        ----------
        full : bool, default=False
            If True, include residual analysis and enable plotting.
        ci_method : str, default='hessian'
            Method for computing confidence intervals:
            - 'hessian': Fast, uses numerical Hessian approximation (default)
            - 'bootstrap': Use bootstrap resampling (slower but more robust)
            - 'none': No confidence intervals
        bootstrap : bool, default=True
            If True, compute bootstrap confidence intervals in addition to ci_method.
            Can be slow for large datasets.
        n_bootstrap : int, default=100
            Number of bootstrap samples.
        confidence : float, default=0.95
            Confidence level for intervals.
        random_state : int, optional
            Random seed for bootstrap.
        include_alpha_trace : bool, default=False
            If True, include alpha trace analysis in the report.
            Uses cached alpha_trace if available (from compute_alpha_trace()),
            otherwise computes it fresh.

        Returns
        -------
        SummaryReport
            Comprehensive summary report with multiple export methods:
            - print_summary(): Formatted console output
            - to_excel(filepath): Export to Excel
            - to_html(filepath): Export to HTML
            - to_pdf(filepath): Export to PDF
            - to_dict(): Convert to dictionary
            - to_dataframe(): Convert coefficients to DataFrame
            - plot_diagnostics(): Generate diagnostic plots (requires full=True)

        Examples
        --------
        >>> diag = ModelDiagnostics(model, X, y)
        >>>
        >>> # Quick summary with Hessian CIs (default)
        >>> report = diag.summary()
        >>> report.print_summary()
        >>>
        >>> # Full summary with both Hessian and bootstrap CIs
        >>> report = diag.summary(full=True, bootstrap=True)
        >>> report.print_summary()
        >>> report.plot_diagnostics()
        >>> report.to_excel('summary.xlsx')
        >>>
        >>> # Include alpha trace (compute once, reuse in reports)
        >>> diag.compute_alpha_trace(l1_ratios=[0.0, 0.5, 1.0])
        >>> report = diag.summary(include_alpha_trace=True)
        >>> report.to_html('report.html')
        """
        from .reporting import generate_summary_report

        return generate_summary_report(
            model=self.model,
            X=self.X,
            y=self.y,
            full=full,
            gdf_method=self.gdf_method,
            ci_method=ci_method,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=random_state,
            include_alpha_trace=include_alpha_trace,
            alpha_trace=self.alpha_trace,  # Pass cached alpha trace
        )

    def __str__(self):
        """Print basic diagnostic summary (legacy method)."""
        output = []
        output.append("=" * 60)
        output.append("Model Diagnostics")
        output.append("=" * 60)
        output.append(f"Model type: {type(self.model).__name__}")
        output.append("Model specifications:")
        output.append(str(self.model.get_params()))
        output.append("Model Run Date (duration):"
                      f" {getattr(self.model, 'fit_datetime_', 'N/A')} "
                      f"({getattr(self.model, 'fit_duration_seconds_', 'N/A')} seconds)")

        output.append(f"N samples: {self.n_samples}")
        output.append(f"N parameters: {len(self.model.coef_)}")
        output.append(f"GDF method: {self.gdf_method}")
        output.append(f"GDF: {self.gdf:.1f}")
        output.append(f"Active constraints: {self.model.n_active_constraints_}")
        output.append("")
        output.append("Fit Statistics:")
        output.append(f"  R2: {self.r2:.4f}")
        output.append(f"  Adjusted R2 (GDF): {self.adj_r2:.4f}")
        output.append(f"  SEE: {self.see:.4f}")
        output.append(f"  SPE: {self.spe:.2%}")
        output.append(f"  MAPE: {self.mape:.2%}")
        output.append(f"  RMSE: {self.rmse:.4f}")
        output.append(f"  CV: {self.cv:.2%}")
        output.append("=" * 60)
        return "\n".join(output)

    def __repr__(self):
        """Print basic diagnostic summary (legacy method)."""
        return self.__str__()

    def to_dict(self):
        """Return diagnostics as dictionary."""
        return {
            'model_type': type(self.model).__name__,
            'model_specs': self.model.get_params(),
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
