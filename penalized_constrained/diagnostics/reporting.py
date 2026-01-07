"""
Summary report generation for penalized-constrained regression.

Provides the SummaryReport dataclass and generate_summary_report function
for creating comprehensive model diagnostic reports.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .dataclasses import (
    CoefficientInfo, FitStatistics, ModelSpecification,
    DataSummary, ConstraintSummary, ResidualAnalysis,
    SampleData, ModelEquation, AlphaTraceResult,
    BootstrapCoefResults, BootstrapResults
)
from .confidence import bootstrap_confidence_intervals, hessian_standard_errors
from .equations import format_model_equation


@dataclass
class SummaryReport:
    """
    Comprehensive summary report for a fitted model.

    This dataclass contains all information needed for various output formats
    (console, Excel, HTML, PDF) and visualizations.

    Attributes
    ----------
    model_spec : ModelSpecification
        Model configuration and optimization details.
    data_summary : DataSummary
        Input data statistics.
    coefficients : List[CoefficientInfo]
        Detailed coefficient information with bounds and CIs.
    intercept : Optional[CoefficientInfo]
        Intercept information if applicable.
    fit_stats : FitStatistics
        Fit quality metrics.
    constraints : ConstraintSummary
        Constraint specification and status.
    residuals : Optional[ResidualAnalysis]
        Residual analysis (only in full mode).
    bootstrap_results : Optional[BootstrapResults]
        Bootstrap CI results (only when bootstrap=True).
    report_datetime : datetime
        When the report was generated.
    ci_method : str
        Method used for confidence intervals ('hessian', 'bootstrap', or 'none').
    equation : Optional[ModelEquation]
        Model equation representation.
    sample_data : Optional[SampleData]
        Sample of input data for reports.
    alpha_trace : Optional[AlphaTraceResult]
        Alpha trace analysis results (only if include_alpha_trace=True).
    """
    model_spec: ModelSpecification
    data_summary: DataSummary
    coefficients: List[CoefficientInfo]
    intercept: Optional[CoefficientInfo]
    fit_stats: FitStatistics
    constraints: ConstraintSummary
    residuals: Optional[ResidualAnalysis] = None
    bootstrap_results: Optional[BootstrapResults] = None
    report_datetime: Optional[datetime] = None
    ci_method: str = 'none'
    equation: Optional[ModelEquation] = None
    sample_data: Optional[SampleData] = None
    alpha_trace: Optional[AlphaTraceResult] = None

    def print_summary(self, decimals: int = 4):
        """
        Print formatted summary to console.

        Parameters
        ----------
        decimals : int, default=4
            Number of decimal places for numeric values.
        """
        width = 70

        def header(title):
            print("=" * width)
            print(f" {title}")
            print("=" * width)

        def section(title):
            print()
            print(f"--- {title} " + "-" * (width - len(title) - 5))

        # Title
        header("PENALIZED-CONSTRAINED REGRESSION SUMMARY")

        # Model Equation
        if self.equation is not None:
            section("Model Equation")
            print(f"  {self.equation.text}")
            if self.equation.is_custom and self.equation.source:
                print("  (Custom model - see source code)")

        # Model Specification
        section("Model Specification")
        print(f"  Model type:     {self.model_spec.model_type}")
        print(f"  Loss function:  {self.model_spec.loss_function}")
        print(f"  Alpha:          {self.model_spec.alpha}")
        print(f"  L1 ratio:       {self.model_spec.l1_ratio}")
        print(f"  Method:         {self.model_spec.method}")
        print(f"  Converged:      {self.model_spec.converged}")
        if self.model_spec.final_objective is not None:
            print(f"  Final objective: {self.model_spec.final_objective:.{decimals}f}")
        if self.model_spec.fit_datetime is not None:
            print(f"  Fit datetime:   {self.model_spec.fit_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.model_spec.fit_duration_seconds is not None:
            if self.model_spec.fit_duration_seconds < 1:
                print(f"  Fit duration:   {self.model_spec.fit_duration_seconds*1000:.1f} ms")
            else:
                print(f"  Fit duration:   {self.model_spec.fit_duration_seconds:.2f} s")

        # Data Summary
        section("Data Summary")
        print(f"  N samples:   {self.data_summary.n_samples}")
        print(f"  N features:  {self.data_summary.n_features}")
        print(f"  Y mean:      {self.data_summary.y_mean:.{decimals}f}")
        print(f"  Y std:       {self.data_summary.y_std:.{decimals}f}")
        print(f"  Y range:     [{self.data_summary.y_min:.{decimals}f}, {self.data_summary.y_max:.{decimals}f}]")

        # Coefficients
        section("Coefficients")
        if self.ci_method != 'none':
            print(f"  (95% CIs via {self.ci_method})")
        print()

        # Determine column widths
        name_width = max(len(c.name) for c in self.coefficients)
        name_width = max(name_width, 12)  # Minimum width

        # Header
        if any(c.ci_lower is not None for c in self.coefficients):
            print(f"  {'Parameter':<{name_width}}  {'Value':>12}  {'95% CI Lower':>12}  {'95% CI Upper':>12}  {'Bounds':>20}  {'Status':>10}")
            print(f"  {'-'*name_width}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*20}  {'-'*10}")
        else:
            print(f"  {'Parameter':<{name_width}}  {'Value':>12}  {'Bounds':>20}  {'Status':>10}")
            print(f"  {'-'*name_width}  {'-'*12}  {'-'*20}  {'-'*10}")

        for coef in self.coefficients:
            lb = f"{coef.lower_bound:.{decimals}f}" if np.isfinite(coef.lower_bound) else "-inf"
            ub = f"{coef.upper_bound:.{decimals}f}" if np.isfinite(coef.upper_bound) else "+inf"
            bounds_str = f"[{lb}, {ub}]"

            if coef.ci_lower is not None:
                print(f"  {coef.name:<{name_width}}  {coef.value:>12.{decimals}f}  {coef.ci_lower:>12.{decimals}f}  {coef.ci_upper:>12.{decimals}f}  {bounds_str:>20}  {coef.bound_status:>10}")
            else:
                print(f"  {coef.name:<{name_width}}  {coef.value:>12.{decimals}f}  {bounds_str:>20}  {coef.bound_status:>10}")

        # Intercept
        if self.intercept is not None:
            lb = f"{self.intercept.lower_bound:.{decimals}f}" if np.isfinite(self.intercept.lower_bound) else "-inf"
            ub = f"{self.intercept.upper_bound:.{decimals}f}" if np.isfinite(self.intercept.upper_bound) else "+inf"
            bounds_str = f"[{lb}, {ub}]"

            if self.intercept.ci_lower is not None:
                print(f"  {'Intercept':<{name_width}}  {self.intercept.value:>12.{decimals}f}  {self.intercept.ci_lower:>12.{decimals}f}  {self.intercept.ci_upper:>12.{decimals}f}  {bounds_str:>20}  {self.intercept.bound_status:>10}")
            else:
                print(f"  {'Intercept':<{name_width}}  {self.intercept.value:>12.{decimals}f}  {bounds_str:>20}  {self.intercept.bound_status:>10}")

        # Fit Statistics
        section("Fit Statistics")
        print(f"  R-squared:       {self.fit_stats.r2:.{decimals}f}")
        print(f"  Adjusted R-sq:   {self.fit_stats.adj_r2:.{decimals}f}")
        print(f"  SEE:             {self.fit_stats.see:.{decimals}f}")
        print(f"  SPE:             {self.fit_stats.spe:.2%}")
        print(f"  MAPE:            {self.fit_stats.mape:.2%}")
        print(f"  RMSE:            {self.fit_stats.rmse:.{decimals}f}")
        if self.fit_stats.mse is not None:
            print(f"  MSE:             {self.fit_stats.mse:.{decimals}f}")
        if self.fit_stats.mae is not None:
            print(f"  MAE:             {self.fit_stats.mae:.{decimals}f}")
        print(f"  CV:              {self.fit_stats.cv:.2%}")
        print(f"  GDF ({self.fit_stats.gdf_method}):     {self.fit_stats.gdf:.1f}")
        if self.fit_stats.aic is not None:
            print(f"  AIC:             {self.fit_stats.aic:.{decimals}f}")
        if self.fit_stats.bic is not None:
            print(f"  BIC:             {self.fit_stats.bic:.{decimals}f}")
        if self.fit_stats.f_statistic is not None:
            pval_str = f"{self.fit_stats.f_pvalue:.4e}" if self.fit_stats.f_pvalue else "N/A"
            print(f"  F-statistic:     {self.fit_stats.f_statistic:.{decimals}f} (p={pval_str})")
        if self.fit_stats.durbin_watson is not None:
            print(f"  Durbin-Watson:   {self.fit_stats.durbin_watson:.{decimals}f}")

        # Constraints
        section("Constraints")
        print(f"  Specified:  {self.constraints.n_specified}")
        print(f"  Active:     {self.constraints.n_active}")
        if self.constraints.active_constraints:
            print("  Active constraints:")
            for name, bound_type in self.constraints.active_constraints:
                print(f"    - {name}: {bound_type} bound")

        # Residual Analysis (if available)
        if self.residuals is not None:
            section("Residual Analysis")
            print(f"  Mean:      {self.residuals.mean:.{decimals}f}")
            print(f"  Std:       {self.residuals.std:.{decimals}f}")
            print(f"  Range:     [{self.residuals.min:.{decimals}f}, {self.residuals.max:.{decimals}f}]")
            print(f"  Skewness:  {self.residuals.skewness:.{decimals}f}")
            print(f"  Kurtosis:  {self.residuals.kurtosis:.{decimals}f}")
            print(f"  Outliers:  {self.residuals.n_outliers} (|r| > 2*std)")

        print()
        if self.report_datetime is not None:
            print(f"Report generated: {self.report_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * width)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire report to nested dictionary.

        Returns
        -------
        dict
            Nested dictionary with all report data.
        """
        result = {
            'report_datetime': self.report_datetime.isoformat() if self.report_datetime else None,
            'ci_method': self.ci_method,
            'model_specification': {
                'model_type': self.model_spec.model_type,
                'loss_function': self.model_spec.loss_function,
                'alpha': self.model_spec.alpha,
                'l1_ratio': self.model_spec.l1_ratio,
                'fit_intercept': self.model_spec.fit_intercept,
                'method': self.model_spec.method,
                'converged': self.model_spec.converged,
                'n_iterations': self.model_spec.n_iterations,
                'final_objective': self.model_spec.final_objective,
                'fit_datetime': self.model_spec.fit_datetime.isoformat() if self.model_spec.fit_datetime else None,
                'fit_duration_seconds': self.model_spec.fit_duration_seconds,
            },
            'data_summary': {
                'n_samples': self.data_summary.n_samples,
                'n_features': self.data_summary.n_features,
                'y_mean': self.data_summary.y_mean,
                'y_std': self.data_summary.y_std,
                'y_min': self.data_summary.y_min,
                'y_max': self.data_summary.y_max,
                'feature_names': self.data_summary.feature_names,
            },
            'coefficients': [
                {
                    'name': c.name,
                    'value': c.value,
                    'lower_bound': c.lower_bound,
                    'upper_bound': c.upper_bound,
                    'is_at_lower': c.is_at_lower,
                    'is_at_upper': c.is_at_upper,
                    'se': c.se,
                    'ci_lower': c.ci_lower,
                    'ci_upper': c.ci_upper,
                }
                for c in self.coefficients
            ],
            'intercept': None if self.intercept is None else {
                'value': self.intercept.value,
                'lower_bound': self.intercept.lower_bound,
                'upper_bound': self.intercept.upper_bound,
                'is_at_lower': self.intercept.is_at_lower,
                'is_at_upper': self.intercept.is_at_upper,
                'se': self.intercept.se,
                'ci_lower': self.intercept.ci_lower,
                'ci_upper': self.intercept.ci_upper,
            },
            'fit_statistics': self.fit_stats.to_dict(),
            'constraints': {
                'n_specified': self.constraints.n_specified,
                'n_active': self.constraints.n_active,
                'active_constraints': self.constraints.active_constraints,
            },
        }

        if self.equation is not None:
            result['equation'] = {
                'text': self.equation.text,
                'latex': self.equation.latex,
                'source': self.equation.source,
                'is_custom': self.equation.is_custom,
            }

        if self.residuals is not None:
            result['residuals'] = {
                'mean': self.residuals.mean,
                'std': self.residuals.std,
                'min': self.residuals.min,
                'max': self.residuals.max,
                'skewness': self.residuals.skewness,
                'kurtosis': self.residuals.kurtosis,
                'n_outliers': self.residuals.n_outliers,
            }

        return result

    def to_dataframe(self):
        """
        Convert coefficients to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficient information.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for c in self.coefficients:
            rows.append({
                'Parameter': c.name,
                'Value': c.value,
                'Lower Bound': c.lower_bound,
                'Upper Bound': c.upper_bound,
                'Status': c.bound_status,
                'SE': c.se,
                'CI Lower': c.ci_lower,
                'CI Upper': c.ci_upper,
            })

        if self.intercept is not None:
            rows.append({
                'Parameter': 'Intercept',
                'Value': self.intercept.value,
                'Lower Bound': self.intercept.lower_bound,
                'Upper Bound': self.intercept.upper_bound,
                'Status': self.intercept.bound_status,
                'SE': self.intercept.se,
                'CI Lower': self.intercept.ci_lower,
                'CI Upper': self.intercept.ci_upper,
            })

        return pd.DataFrame(rows)

    def to_excel(self, filepath: str, sample_n: int = 50):
        """
        Export report to Excel file with multiple sheets.

        Parameters
        ----------
        filepath : str
            Path to output Excel file (.xlsx).
        sample_n : int, default=50
            Number of sample rows to include. -1 for all.
        """
        from .export import to_excel
        to_excel(self, filepath, sample_n=sample_n)

    def to_html(
        self,
        filepath: Optional[str] = None,
        include_plots: bool = True,
        sample_n: int = 50,
        include_equation: bool = True,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> str:
        """
        Export report to HTML format with embedded plots and sample data.

        Parameters
        ----------
        filepath : str, optional
            If provided, write HTML to file. Otherwise return as string.
        include_plots : bool, default=True
            Embed diagnostic plots as base64 images
        sample_n : int, default=50
            Number of sample rows to include in sample data table. -1 for all (warns if >100)
        include_equation : bool, default=True
            Include model equation section
        X : np.ndarray, optional
            Full feature matrix for interactive data explorer. If None, uses sample_data.
        y : np.ndarray, optional
            Full target array for interactive data explorer. If None, uses sample_data.

        Returns
        -------
        str
            HTML content.
        """
        from .export import to_html
        return to_html(
            self, filepath,
            include_plots=include_plots,
            sample_n=sample_n,
            include_equation=include_equation,
            X=X,
            y=y,
        )

    def to_pdf(self, filepath: str, **kwargs):
        """
        Export report to PDF format.

        Parameters
        ----------
        filepath : str
            Path to output PDF file.
        **kwargs
            Additional arguments passed to to_html()

        Notes
        -----
        Requires weasyprint or pdfkit to be installed.
        Falls back to HTML if PDF generation fails.
        """
        from .export import to_pdf
        to_pdf(self, filepath, **kwargs)

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None):
        """
        Generate diagnostic plots.

        Parameters
        ----------
        figsize : tuple, default=(12, 10)
            Figure size.
        save_path : str, optional
            If provided, save figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        from .plotting import plot_diagnostics

        if self.residuals is None or self.residuals.residuals is None:
            raise ValueError("Residual data not available. Use full=True in summary().")

        return plot_diagnostics(
            self.residuals,
            self.coefficients,
            figsize=figsize,
            save_path=save_path
        )


def generate_summary_report(
    model,
    X,
    y,
    full: bool = False,
    gdf_method: str = 'hu',
    ci_method: str = 'hessian',
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    sample_n: int = 50,
    include_alpha_trace: bool = False,
    alpha_trace_l1_ratios: Optional[List[float]] = None,
    alpha_trace_n_alphas: int = 30,
    alpha_trace: Optional['AlphaTraceResult'] = None,
) -> SummaryReport:
    """
    Generate a comprehensive summary report for a fitted model.

    Parameters
    ----------
    model : PenalizedConstrainedRegression or PenalizedConstrainedCV
        Fitted model.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    full : bool, default=False
        If True, include residual analysis and store residual arrays for plotting.
    gdf_method : str, default='hu'
        Method for computing GDF: 'hu' or 'gaines'.
    ci_method : str, default='hessian'
        Method for computing confidence intervals:
        - 'hessian': Fast, uses numerical Hessian approximation (default)
        - 'none': No confidence intervals
    bootstrap : bool, default=False
        If True, also compute bootstrap confidence intervals (in addition to hessian).
        Can be slow for large datasets or many bootstrap samples.
    n_bootstrap : int, default=1000
        Number of bootstrap samples (only used if bootstrap=True).
    confidence : float, default=0.95
        Confidence level for intervals.
    random_state : int, optional
        Random seed for bootstrap.
    sample_n : int, default=50
        Number of sample data rows to include. -1 for all (warns if >100).
    include_alpha_trace : bool, default=False
        If True, include alpha trace analysis in the report.
        If alpha_trace parameter is provided, uses that directly.
        Otherwise computes fresh.
    alpha_trace_l1_ratios : list of float, optional
        L1 ratios to evaluate in alpha trace. Default is [0.0, 0.5, 1.0].
        Only used if alpha_trace is None and include_alpha_trace=True.
    alpha_trace_n_alphas : int, default=30
        Number of alpha values to evaluate per l1_ratio.
        Only used if alpha_trace is None and include_alpha_trace=True.
    alpha_trace : AlphaTraceResult, optional
        Pre-computed alpha trace result (from ModelDiagnostics.compute_alpha_trace()).
        If provided and include_alpha_trace=True, uses this instead of recomputing.

    Returns
    -------
    SummaryReport
        Comprehensive summary report object with multiple export methods.

    Examples
    --------
    >>> from penalized_constrained import PenalizedConstrainedRegression
    >>> from penalized_constrained.diagnostics import generate_summary_report
    >>>
    >>> model = PenalizedConstrainedRegression(bounds=[(-1, 0), (-1, 0)])
    >>> model.fit(X, y)
    >>>
    >>> # Basic summary with Hessian CIs (fast, default)
    >>> report = generate_summary_report(model, X, y)
    >>> report.print_summary()
    >>>
    >>> # Full summary with bootstrap CIs (slower, more robust)
    >>> report = generate_summary_report(model, X, y, full=True, bootstrap=True)
    >>> report.print_summary()
    >>> report.plot_diagnostics()
    >>> report.to_excel('model_summary.xlsx')
    >>>
    >>> # Include alpha trace analysis
    >>> report = generate_summary_report(model, X, y, include_alpha_trace=True)
    >>> print(report.alpha_trace.summary_df)
    """
    from .core import ModelDiagnostics

    check_is_fitted(model)
    X = np.asarray(X)
    y = np.asarray(y)

    # Record report generation time
    report_datetime = datetime.now()

    # Get base diagnostics
    diag = ModelDiagnostics(model, X, y, gdf_method=gdf_method)

    # Model Equation
    equation = format_model_equation(model)

    # Model Specification with all input parameters for reproducibility
    loss_str = model.loss if isinstance(model.loss, str) else 'custom'
    # Handle both CV models (alpha_, l1_ratio_) and base models (alpha, l1_ratio)
    # Note: Can't use `or` here because 0.0 is a valid value but falsy
    alpha = getattr(model, 'alpha_', None)
    if alpha is None:
        alpha = getattr(model, 'alpha', None)
    l1_ratio = getattr(model, 'l1_ratio_', None)
    if l1_ratio is None:
        l1_ratio = getattr(model, 'l1_ratio', None)

    # Get prediction_fn source code if custom
    prediction_fn_source = None
    has_custom_prediction_fn = model.prediction_fn is not None
    if has_custom_prediction_fn:
        prediction_fn_source = equation.source if equation else None

    # Determine x0 value (may be string or array)
    x0_value = getattr(model, 'x0', 'ols')
    if hasattr(x0_value, 'tolist'):  # numpy array
        x0_value = x0_value.tolist()

    model_spec = ModelSpecification(
        model_type=type(model).__name__,
        loss_function=loss_str,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=model.fit_intercept,
        scale=getattr(model, 'scale', False),
        bounds=model.bounds,
        intercept_bounds=getattr(model, 'intercept_bounds', None),
        coef_names=list(model.coef_names) if hasattr(model, 'coef_names') and model.coef_names else None,
        penalty_exclude=getattr(model, 'penalty_exclude', None),
        method=model.method,
        max_iter=getattr(model, 'max_iter', 1000),
        tol=getattr(model, 'tol', 1e-6),
        x0=x0_value,
        prediction_fn_source=prediction_fn_source,
        has_custom_prediction_fn=has_custom_prediction_fn,
        has_custom_loss_fn=callable(model.loss),
        converged=model.converged_,
        n_iterations=getattr(getattr(model, 'optimization_result_', None), 'nit', None),
        final_objective=getattr(getattr(model, 'optimization_result_', None), 'fun', None),
        fit_datetime=getattr(model, 'fit_datetime_', None),
        fit_duration_seconds=getattr(model, 'fit_duration_seconds_', None),
    )

    # Data Summary
    # coef_names are the coefficient/parameter names (e.g., T1, LC, RC)
    coef_names = list(model.coef_names_in_) if hasattr(model, 'coef_names_in_') and model.coef_names_in_ is not None else None
    data_summary = DataSummary(
        n_samples=len(y),
        n_features=X.shape[1],
        y_mean=float(np.mean(y)),
        y_std=float(np.std(y)),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        feature_names=coef_names,  # DataSummary.feature_names refers to coefficient/parameter names
    )

    # Get X column names from feature_names_in_ (sklearn convention)
    # These are the actual column names of the X matrix
    x_column_names = list(model.feature_names_in_)

    # Sample Data
    sample_data = None
    if sample_n != 0:
        actual_n = len(y) if sample_n == -1 else min(sample_n, len(y))
        if sample_n == -1 and len(y) > 100:
            warnings.warn(
                f"Large dataset ({len(y)} rows). Consider using sample_n parameter to limit.",
                UserWarning
            )

        y_pred = model.predict(X)
        sample_data = SampleData(
            X_sample=X[:actual_n].copy(),
            y_sample=y[:actual_n].copy(),
            y_pred_sample=y_pred[:actual_n].copy(),
            n_total=len(y),
            n_sample=actual_n,
            x_column_names=x_column_names,  # Use X column names, not parameter names
        )

    # Determine CI method and compute
    actual_ci_method = 'none'
    bootstrap_results = None
    hessian_se = None

    # Compute hessian SE if requested
    if ci_method == 'hessian':
        actual_ci_method = 'hessian'
        try:
            hessian_se = hessian_standard_errors(model, X, y)
        except Exception as e:
            warnings.warn(f"Hessian SE failed: {e}", UserWarning)
            actual_ci_method = 'none'

    # Compute bootstrap if requested (in addition to hessian)
    if bootstrap:
        constrained_results = None
        unconstrained_results = None

        def _has_effective_constraints(mdl):
            """
            Check if model has any effective constraints or regularization.

            Returns True if model has:
            - alpha > 0 (regularization penalty), OR
            - any coefficient bound that is finite (not ±inf), OR
            - any intercept bound that is finite

            Returns False if model is unconstrained (no bounds, alpha=0).
            """
            # Check alpha (regularization penalty)
            alpha_val = getattr(mdl, 'alpha_', None)
            if alpha_val is None:
                alpha_val = getattr(mdl, 'alpha', 0.0)
            if alpha_val is not None and alpha_val > 0:
                return True

            # Check coefficient bounds
            bounds_parsed = getattr(mdl, '_bounds_parsed', None)
            if bounds_parsed:
                for lb, ub in bounds_parsed:
                    # A constraint exists if either bound is finite (not ±inf)
                    if np.isfinite(lb) or np.isfinite(ub):
                        return True

            # Check intercept bounds
            intercept_bounds = getattr(mdl, 'intercept_bounds', None)
            if intercept_bounds is not None:
                lb, ub = intercept_bounds
                if (lb is not None and np.isfinite(lb)) or (ub is not None and np.isfinite(ub)):
                    return True

            return False

        model_is_constrained = _has_effective_constraints(model)

        # Import base model class for bootstrap (works for both base and CV models)
        from ..regression import PenalizedConstrainedRegression

        def _get_base_model_params(mdl):
            """
            Extract base model parameters for bootstrap.

            For CV models, extracts the best alpha/l1_ratio and base model params.
            For base models, returns get_params() directly.
            """
            # Check if it's a CV model by looking for best_estimator_
            if hasattr(mdl, 'best_estimator_'):
                # CV model - use selected alpha/l1_ratio and base params
                return {
                    'alpha': mdl.alpha_,
                    'l1_ratio': mdl.l1_ratio_,
                    'bounds': mdl.bounds,
                    'coef_names': mdl.coef_names,
                    'penalty_exclude': getattr(mdl, 'penalty_exclude', None),
                    'fit_intercept': mdl.fit_intercept,
                    'intercept_bounds': getattr(mdl, 'intercept_bounds', None),
                    'loss': mdl.loss,
                    'prediction_fn': mdl.prediction_fn,
                    'scale': getattr(mdl, 'scale', False),
                    'x0': mdl.x0,
                    'method': mdl.method,
                    'max_iter': mdl.max_iter,
                    'tol': mdl.tol,
                    'verbose': 0,
                    'safe_mode': getattr(mdl, 'safe_mode', True),
                }
            else:
                # Base model - use get_params() directly
                params = mdl.get_params()
                params['verbose'] = 0
                return params

        if model_is_constrained:
            # Model HAS constraints: run BOTH constrained and unconstrained bootstrap

            # 1. Run constrained bootstrap (with original bounds and alpha)
            try:
                constrained_params = _get_base_model_params(model)
                bootstrap_raw = bootstrap_confidence_intervals(
                    PenalizedConstrainedRegression, X, y,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    random_state=random_state,
                    warm_start_coef=model.coef_,  # Use fit coefficients as starting point
                    **constrained_params
                )
                constrained_results = BootstrapCoefResults(
                    coef_mean=bootstrap_raw['coef_mean'],
                    coef_std=bootstrap_raw['coef_std'],
                    coef_ci_lower=bootstrap_raw['coef_ci_lower'],
                    coef_ci_upper=bootstrap_raw['coef_ci_upper'],
                    intercept_mean=bootstrap_raw.get('intercept_mean'),
                    intercept_std=bootstrap_raw.get('intercept_std'),
                    intercept_ci=bootstrap_raw.get('intercept_ci'),
                    bootstrap_coefs=bootstrap_raw['bootstrap_coefs'],
                    n_successful=bootstrap_raw['n_successful'],
                )
            except Exception as e:
                warnings.warn(f"Constrained bootstrap failed: {e}", UserWarning)

            # 2. Run unconstrained bootstrap (no bounds, alpha=0)
            try:
                unconstrained_params = _get_base_model_params(model)
                unconstrained_params['bounds'] = None
                unconstrained_params['intercept_bounds'] = None
                unconstrained_params['alpha'] = 0.0

                bootstrap_raw_unc = bootstrap_confidence_intervals(
                    PenalizedConstrainedRegression, X, y,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    random_state=random_state,
                    warm_start_coef=model.coef_,  # Use fit coefficients as starting point
                    **unconstrained_params
                )
                unconstrained_results = BootstrapCoefResults(
                    coef_mean=bootstrap_raw_unc['coef_mean'],
                    coef_std=bootstrap_raw_unc['coef_std'],
                    coef_ci_lower=bootstrap_raw_unc['coef_ci_lower'],
                    coef_ci_upper=bootstrap_raw_unc['coef_ci_upper'],
                    intercept_mean=bootstrap_raw_unc.get('intercept_mean'),
                    intercept_std=bootstrap_raw_unc.get('intercept_std'),
                    intercept_ci=bootstrap_raw_unc.get('intercept_ci'),
                    bootstrap_coefs=bootstrap_raw_unc['bootstrap_coefs'],
                    n_successful=bootstrap_raw_unc['n_successful'],
                )
            except Exception as e:
                # Unconstrained bootstrap can fail if model needs constraints to converge
                # or if the model uses custom prediction_fn that requires specific params
                warnings.warn(f"Unconstrained bootstrap failed: {e}", UserWarning)
        else:
            # Model has NO constraints: only run unconstrained bootstrap
            # (constrained_results stays None to indicate no constraints)
            try:
                unconstrained_params = _get_base_model_params(model)
                bootstrap_raw_unc = bootstrap_confidence_intervals(
                    PenalizedConstrainedRegression, X, y,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    random_state=random_state,
                    warm_start_coef=model.coef_,  # Use fit coefficients as starting point
                    **unconstrained_params
                )
                unconstrained_results = BootstrapCoefResults(
                    coef_mean=bootstrap_raw_unc['coef_mean'],
                    coef_std=bootstrap_raw_unc['coef_std'],
                    coef_ci_lower=bootstrap_raw_unc['coef_ci_lower'],
                    coef_ci_upper=bootstrap_raw_unc['coef_ci_upper'],
                    intercept_mean=bootstrap_raw_unc.get('intercept_mean'),
                    intercept_std=bootstrap_raw_unc.get('intercept_std'),
                    intercept_ci=bootstrap_raw_unc.get('intercept_ci'),
                    bootstrap_coefs=bootstrap_raw_unc['bootstrap_coefs'],
                    n_successful=bootstrap_raw_unc['n_successful'],
                )
            except Exception as e:
                warnings.warn(f"Bootstrap failed: {e}", UserWarning)

        # Create BootstrapResults if we have any results
        if constrained_results is not None or unconstrained_results is not None:
            bootstrap_results = BootstrapResults(
                constrained=constrained_results,  # None if model has no constraints
                unconstrained=unconstrained_results,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                feature_names=coef_names,
            )
            actual_ci_method = 'bootstrap'

    # Z-score for confidence interval
    from scipy import stats
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Coefficients
    coefficients = []
    for i, coef_val in enumerate(model.coef_):
        if hasattr(model, 'coef_names_in_') and model.coef_names_in_ is not None:
            name = model.coef_names_in_[i]
        else:
            name = f"beta_{i}"

        lb, ub = model._bounds_parsed[i]

        # Check if at bound
        is_at_lower = np.isfinite(lb) and np.abs(coef_val - lb) < 1e-6
        is_at_upper = np.isfinite(ub) and np.abs(coef_val - ub) < 1e-6

        # Initialize all SE/CI values
        hess_se = None
        hess_ci_lower = None
        hess_ci_upper = None
        boot_se = None
        boot_ci_lower = None
        boot_ci_upper = None

        # Hessian SE and CI
        if hessian_se is not None and i < len(hessian_se) and not np.isnan(hessian_se[i]):
            hess_se = float(hessian_se[i])
            hess_ci_lower = float(coef_val - z_score * hess_se)
            hess_ci_upper = float(coef_val + z_score * hess_se)

        # Bootstrap SE and CI (constrained)
        if bootstrap_results is not None:
            boot_se = float(bootstrap_results.coef_std[i])
            boot_ci_lower = float(bootstrap_results.coef_ci_lower[i])
            boot_ci_upper = float(bootstrap_results.coef_ci_upper[i])

        # Legacy fields: prefer bootstrap if available, otherwise hessian
        if boot_se is not None:
            se, ci_lower, ci_upper = boot_se, boot_ci_lower, boot_ci_upper
        elif hess_se is not None:
            se, ci_lower, ci_upper = hess_se, hess_ci_lower, hess_ci_upper
        else:
            se, ci_lower, ci_upper = None, None, None

        coefficients.append(CoefficientInfo(
            name=str(name),
            value=float(coef_val),
            lower_bound=float(lb),
            upper_bound=float(ub),
            is_at_lower=is_at_lower,
            is_at_upper=is_at_upper,
            hessian_se=hess_se,
            hessian_ci_lower=hess_ci_lower,
            hessian_ci_upper=hess_ci_upper,
            bootstrap_se=boot_se,
            bootstrap_ci_lower=boot_ci_lower,
            bootstrap_ci_upper=boot_ci_upper,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ))

    # Intercept
    intercept_info = None
    if model.fit_intercept and model.prediction_fn is None:
        int_lb = -np.inf
        int_ub = np.inf
        if model.intercept_bounds is not None:
            int_lb = model.intercept_bounds[0] if model.intercept_bounds[0] is not None else -np.inf
            int_ub = model.intercept_bounds[1] if model.intercept_bounds[1] is not None else np.inf

        is_at_lower = np.isfinite(int_lb) and np.abs(model.intercept_ - int_lb) < 1e-6
        is_at_upper = np.isfinite(int_ub) and np.abs(model.intercept_ - int_ub) < 1e-6

        # Initialize all SE/CI values for intercept
        hess_se = None
        hess_ci_lower = None
        hess_ci_upper = None
        boot_se = None
        boot_ci_lower = None
        boot_ci_upper = None

        # Hessian SE and CI for intercept
        if hessian_se is not None:
            int_idx = len(model.coef_)
            if int_idx < len(hessian_se) and not np.isnan(hessian_se[int_idx]):
                hess_se = float(hessian_se[int_idx])
                hess_ci_lower = float(model.intercept_ - z_score * hess_se)
                hess_ci_upper = float(model.intercept_ + z_score * hess_se)

        # Bootstrap SE and CI for intercept
        if bootstrap_results is not None and bootstrap_results.intercept_ci is not None:
            boot_se = float(bootstrap_results.intercept_std)
            boot_ci_lower = float(bootstrap_results.intercept_ci[0])
            boot_ci_upper = float(bootstrap_results.intercept_ci[1])

        # Legacy fields: prefer bootstrap if available, otherwise hessian
        if boot_se is not None:
            se, ci_lower, ci_upper = boot_se, boot_ci_lower, boot_ci_upper
        elif hess_se is not None:
            se, ci_lower, ci_upper = hess_se, hess_ci_lower, hess_ci_upper
        else:
            se, ci_lower, ci_upper = None, None, None

        intercept_info = CoefficientInfo(
            name='Intercept',
            value=float(model.intercept_),
            lower_bound=float(int_lb),
            upper_bound=float(int_ub),
            is_at_lower=is_at_lower,
            is_at_upper=is_at_upper,
            hessian_se=hess_se,
            hessian_ci_lower=hess_ci_lower,
            hessian_ci_upper=hess_ci_upper,
            bootstrap_se=boot_se,
            bootstrap_ci_lower=boot_ci_lower,
            bootstrap_ci_upper=boot_ci_upper,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    # Compute additional statistics
    n = len(y)
    k = X.shape[1] + (1 if model.fit_intercept else 0)  # Number of parameters

    # MSE and MAE
    residuals_for_stats = diag.residuals
    mse = float(np.mean(residuals_for_stats ** 2))
    mae = float(np.mean(np.abs(residuals_for_stats)))

    # AIC and BIC (assuming normally distributed errors)
    # AIC = n * ln(SSE/n) + 2k
    # BIC = n * ln(SSE/n) + k * ln(n)
    sse = float(np.sum(residuals_for_stats ** 2))
    if sse > 0:
        aic = float(n * np.log(sse / n) + 2 * k)
        bic = float(n * np.log(sse / n) + k * np.log(n))
    else:
        aic = None
        bic = None

    # F-statistic and p-value
    # F = (ESS / k) / (RSS / (n - k - 1)) where ESS = TSS - RSS
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss > 0 and n > k + 1:
        rss = sse
        ess = tss - rss
        # Degrees of freedom: k for explained, n-k-1 for residual
        df_regression = k
        df_residual = n - k - 1
        if df_residual > 0 and rss > 0:
            f_statistic = float((ess / df_regression) / (rss / df_residual))
            try:
                f_pvalue = float(1 - stats.f.cdf(f_statistic, df_regression, df_residual))
            except:
                f_pvalue = None
        else:
            f_statistic = None
            f_pvalue = None
    else:
        f_statistic = None
        f_pvalue = None

    # Durbin-Watson statistic for autocorrelation
    # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    if len(residuals_for_stats) > 1 and sse > 0:
        diff_residuals = np.diff(residuals_for_stats)
        durbin_watson = float(np.sum(diff_residuals ** 2) / sse)
    else:
        durbin_watson = None

    # Fit Statistics
    fit_stats = FitStatistics(
        r2=diag.r2,
        adj_r2=diag.adj_r2,
        see=diag.see,
        spe=diag.spe,
        mape=diag.mape,
        rmse=diag.rmse,
        cv=diag.cv,
        gdf=diag.gdf,
        gdf_method=gdf_method,
        # Additional statistics
        aic=aic,
        bic=bic,
        f_statistic=f_statistic,
        f_pvalue=f_pvalue,
        durbin_watson=durbin_watson,
        mse=mse,
        mae=mae,
    )

    # Constraints
    n_specified = diag._count_specified_constraints()
    constraints = ConstraintSummary(
        n_specified=n_specified,
        n_active=model.n_active_constraints_,
        active_constraints=list(model.active_constraints_),
    )

    # Residual Analysis (full mode only)
    residual_analysis = None
    if full:
        residuals = diag.residuals
        try:
            from scipy.stats import skew, kurtosis
            skewness = float(skew(residuals))
            kurt = float(kurtosis(residuals))
        except ImportError:
            # Compute manually
            skewness = float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3))
            kurt = float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 4) - 3)

        n_outliers = int(np.sum(np.abs(residuals) > 2 * np.std(residuals)))

        residual_analysis = ResidualAnalysis(
            mean=float(np.mean(residuals)),
            std=float(np.std(residuals)),
            min=float(np.min(residuals)),
            max=float(np.max(residuals)),
            skewness=skewness,
            kurtosis=kurt,
            n_outliers=n_outliers,
            residuals=residuals.copy(),
            y_pred=diag.y_pred.copy(),
        )

    # Alpha Trace Analysis (optional)
    alpha_trace_result = None
    if include_alpha_trace:
        # Use pre-computed alpha trace if provided
        if alpha_trace is not None:
            alpha_trace_result = alpha_trace
        else:
            # Compute fresh alpha trace
            from .alpha_trace import compute_alpha_trace, summarize_alpha_trace, find_optimal_alpha

            # Default l1_ratios if not specified
            if alpha_trace_l1_ratios is None:
                alpha_trace_l1_ratios = [0.0, 0.5, 1.0]

            # Pass model directly - clone approach preserves all settings including prediction_fn
            trace_df = compute_alpha_trace(
                model,
                X, y,
                l1_ratios=alpha_trace_l1_ratios,
                n_alphas=alpha_trace_n_alphas,
            )

            summary_df = summarize_alpha_trace(trace_df)
            optimal = find_optimal_alpha(trace_df)

            alpha_trace_result = AlphaTraceResult(
                trace_df=trace_df,
                summary_df=summary_df,
                optimal=optimal,
                l1_ratios=alpha_trace_l1_ratios,
                n_alphas=alpha_trace_n_alphas,
            )

    return SummaryReport(
        model_spec=model_spec,
        data_summary=data_summary,
        coefficients=coefficients,
        intercept=intercept_info,
        fit_stats=fit_stats,
        constraints=constraints,
        residuals=residual_analysis,
        bootstrap_results=bootstrap_results,
        report_datetime=report_datetime,
        ci_method=actual_ci_method,
        equation=equation,
        sample_data=sample_data,
        alpha_trace=alpha_trace_result,
    )
