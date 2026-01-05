"""
Diagnostics module for penalized-constrained regression.

Provides:
- Generalized Degrees of Freedom (GDF) computation
- Fit statistics (SEE, SPE, adjusted R²)
- Bootstrap confidence intervals
- Hessian-based standard errors
- Comprehensive summary reports with multiple export formats
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import approx_fprime
import warnings
from io import BytesIO, StringIO


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
    
    def summary(
        self,
        full: bool = False,
        ci_method: str = 'hessian',
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ) -> 'SummaryReport':
        """
        Generate comprehensive summary report.

        Parameters
        ----------
        full : bool, default=False
            If True, include residual analysis and enable plotting.
        ci_method : str, default='hessian'
            Method for computing confidence intervals:
            - 'hessian': Fast, uses numerical Hessian approximation (default)
            - 'none': No confidence intervals
        bootstrap : bool, default=False
            If True, compute bootstrap confidence intervals (overrides ci_method).
            Can be slow for large datasets.
        n_bootstrap : int, default=1000
            Number of bootstrap samples.
        confidence : float, default=0.95
            Confidence level for intervals.
        random_state : int, optional
            Random seed for bootstrap.

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
        >>> # Full summary with bootstrap CIs and plots
        >>> report = diag.summary(full=True, bootstrap=True)
        >>> report.print_summary()
        >>> report.plot_diagnostics()
        >>> report.to_excel('summary.xlsx')
        """
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
        )

    def __repr__(self):
        """Print basic diagnostic summary (legacy method)."""
        print("=" * 60)
        print("Model Diagnostics")
        print("=" * 60)
        print(f"Model type: {type(self.model).__name__}")
        print("Model specifications:")
        print(self.model.get_params())
        print("Model Run Date (duration):" f" {getattr(self.model, 'fit_datetime_', 'N/A')} "
              f"({getattr(self.model, 'fit_duration_seconds_', 'N/A')} seconds)")
        
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


# =============================================================================
# Summary Report Data Classes
# =============================================================================

@dataclass
class CoefficientInfo:
    """Information about a single coefficient."""
    name: str
    value: float
    lower_bound: float
    upper_bound: float
    is_at_lower: bool = False
    is_at_upper: bool = False
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    @property
    def is_constrained(self) -> bool:
        """Whether this coefficient is at a bound."""
        return self.is_at_lower or self.is_at_upper

    @property
    def bound_status(self) -> str:
        """String describing bound status."""
        if self.is_at_lower:
            return "at lower"
        elif self.is_at_upper:
            return "at upper"
        return "free"

    def format_value(self, decimals: int = 6) -> str:
        """Format coefficient value with optional CI."""
        if self.ci_lower is not None and self.ci_upper is not None:
            return f"{self.value:.{decimals}f} [{self.ci_lower:.{decimals}f}, {self.ci_upper:.{decimals}f}]"
        elif self.se is not None:
            return f"{self.value:.{decimals}f} ± {self.se:.{decimals}f}"
        return f"{self.value:.{decimals}f}"


@dataclass
class FitStatistics:
    """Fit quality statistics."""
    r2: float
    adj_r2: float
    see: float
    spe: float
    mape: float
    rmse: float
    cv: float
    gdf: float
    gdf_method: str

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'R²': self.r2,
            'Adjusted R²': self.adj_r2,
            'SEE': self.see,
            'SPE': self.spe,
            'MAPE': self.mape,
            'RMSE': self.rmse,
            'CV': self.cv,
            'GDF': self.gdf,
        }


@dataclass
class ModelSpecification:
    """Model specification details."""
    model_type: str
    loss_function: str
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    method: str
    converged: bool
    n_iterations: Optional[int] = None
    final_objective: Optional[float] = None
    fit_datetime: Optional[datetime] = None
    fit_duration_seconds: Optional[float] = None


@dataclass
class DataSummary:
    """Summary of input data."""
    n_samples: int
    n_features: int
    y_mean: float
    y_std: float
    y_min: float
    y_max: float
    feature_names: Optional[List[str]] = None


@dataclass
class ConstraintSummary:
    """Summary of constraints."""
    n_specified: int
    n_active: int
    active_constraints: List[Tuple[str, str]]  # (name, 'lower'|'upper')


@dataclass
class ResidualAnalysis:
    """Residual analysis statistics."""
    mean: float
    std: float
    min: float
    max: float
    skewness: float
    kurtosis: float
    n_outliers: int  # |residual| > 2*std
    residuals: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None


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
    bootstrap_results : Optional[Dict]
        Bootstrap CI results (only in full mode with bootstrap=True).
    report_datetime : datetime
        When the report was generated.
    ci_method : str
        Method used for confidence intervals ('hessian', 'bootstrap', or 'none').
    """
    model_spec: ModelSpecification
    data_summary: DataSummary
    coefficients: List[CoefficientInfo]
    intercept: Optional[CoefficientInfo]
    fit_stats: FitStatistics
    constraints: ConstraintSummary
    residuals: Optional[ResidualAnalysis] = None
    bootstrap_results: Optional[Dict] = None
    report_datetime: Optional[datetime] = None
    ci_method: str = 'none'

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
        print(f"  CV:              {self.fit_stats.cv:.2%}")
        print(f"  GDF ({self.fit_stats.gdf_method}):     {self.fit_stats.gdf:.1f}")

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

    def to_excel(self, filepath: str):
        """
        Export report to Excel file with multiple sheets.

        Parameters
        ----------
        filepath : str
            Path to output Excel file (.xlsx).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_excel()")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Coefficients sheet
            coef_df = self.to_dataframe()
            coef_df.to_excel(writer, sheet_name='Coefficients', index=False)

            # Fit Statistics sheet
            fit_df = pd.DataFrame([self.fit_stats.to_dict()]).T
            fit_df.columns = ['Value']
            fit_df.index.name = 'Statistic'
            fit_df.to_excel(writer, sheet_name='Fit Statistics')

            # Model Specification sheet
            spec_dict = {
                'Model Type': self.model_spec.model_type,
                'Loss Function': self.model_spec.loss_function,
                'Alpha': self.model_spec.alpha,
                'L1 Ratio': self.model_spec.l1_ratio,
                'Method': self.model_spec.method,
                'Converged': self.model_spec.converged,
                'Final Objective': self.model_spec.final_objective,
            }
            spec_df = pd.DataFrame([spec_dict]).T
            spec_df.columns = ['Value']
            spec_df.index.name = 'Parameter'
            spec_df.to_excel(writer, sheet_name='Model Specification')

            # Data Summary sheet
            data_dict = {
                'N Samples': self.data_summary.n_samples,
                'N Features': self.data_summary.n_features,
                'Y Mean': self.data_summary.y_mean,
                'Y Std': self.data_summary.y_std,
                'Y Min': self.data_summary.y_min,
                'Y Max': self.data_summary.y_max,
            }
            data_df = pd.DataFrame([data_dict]).T
            data_df.columns = ['Value']
            data_df.index.name = 'Statistic'
            data_df.to_excel(writer, sheet_name='Data Summary')

            # Residual Analysis (if available)
            if self.residuals is not None:
                resid_dict = {
                    'Mean': self.residuals.mean,
                    'Std': self.residuals.std,
                    'Min': self.residuals.min,
                    'Max': self.residuals.max,
                    'Skewness': self.residuals.skewness,
                    'Kurtosis': self.residuals.kurtosis,
                    'N Outliers': self.residuals.n_outliers,
                }
                resid_df = pd.DataFrame([resid_dict]).T
                resid_df.columns = ['Value']
                resid_df.index.name = 'Statistic'
                resid_df.to_excel(writer, sheet_name='Residual Analysis')

    def to_html(self, filepath: Optional[str] = None) -> str:
        """
        Export report to HTML format.

        Parameters
        ----------
        filepath : str, optional
            If provided, write HTML to file. Otherwise return as string.

        Returns
        -------
        str
            HTML content.
        """
        html_parts = []

        # CSS Styles
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Penalized-Constrained Regression Summary</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:hover { background-color: #f5f5f5; }
                .stat-value { font-family: 'Consolas', monospace; }
                .at-bound { color: #e74c3c; font-weight: bold; }
                .free { color: #27ae60; }
                .converged-true { color: #27ae60; }
                .converged-false { color: #e74c3c; }
                .metric-card { display: inline-block; background: #ecf0f1; padding: 15px 25px; margin: 5px; border-radius: 5px; text-align: center; }
                .metric-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
        <div class="container">
        """)

        # Title
        html_parts.append("<h1>Penalized-Constrained Regression Summary</h1>")

        # Key Metrics Cards
        html_parts.append('<div style="text-align: center; margin: 20px 0;">')
        html_parts.append(f'<div class="metric-card"><div class="metric-value">{self.fit_stats.r2:.4f}</div><div class="metric-label">R²</div></div>')
        html_parts.append(f'<div class="metric-card"><div class="metric-value">{self.fit_stats.spe:.2%}</div><div class="metric-label">SPE</div></div>')
        html_parts.append(f'<div class="metric-card"><div class="metric-value">{self.fit_stats.mape:.2%}</div><div class="metric-label">MAPE</div></div>')
        html_parts.append(f'<div class="metric-card"><div class="metric-value">{self.constraints.n_active}</div><div class="metric-label">Active Constraints</div></div>')
        html_parts.append('</div>')

        # Model Specification
        html_parts.append("<h2>Model Specification</h2>")
        html_parts.append("<table>")
        html_parts.append(f"<tr><td>Model Type</td><td class='stat-value'>{self.model_spec.model_type}</td></tr>")
        html_parts.append(f"<tr><td>Loss Function</td><td class='stat-value'>{self.model_spec.loss_function}</td></tr>")
        html_parts.append(f"<tr><td>Alpha</td><td class='stat-value'>{self.model_spec.alpha}</td></tr>")
        html_parts.append(f"<tr><td>L1 Ratio</td><td class='stat-value'>{self.model_spec.l1_ratio}</td></tr>")
        html_parts.append(f"<tr><td>Method</td><td class='stat-value'>{self.model_spec.method}</td></tr>")
        converged_class = 'converged-true' if self.model_spec.converged else 'converged-false'
        html_parts.append(f"<tr><td>Converged</td><td class='stat-value {converged_class}'>{self.model_spec.converged}</td></tr>")
        html_parts.append("</table>")

        # Data Summary
        html_parts.append("<h2>Data Summary</h2>")
        html_parts.append("<table>")
        html_parts.append(f"<tr><td>N Samples</td><td class='stat-value'>{self.data_summary.n_samples}</td></tr>")
        html_parts.append(f"<tr><td>N Features</td><td class='stat-value'>{self.data_summary.n_features}</td></tr>")
        html_parts.append(f"<tr><td>Y Mean</td><td class='stat-value'>{self.data_summary.y_mean:.4f}</td></tr>")
        html_parts.append(f"<tr><td>Y Std</td><td class='stat-value'>{self.data_summary.y_std:.4f}</td></tr>")
        html_parts.append(f"<tr><td>Y Range</td><td class='stat-value'>[{self.data_summary.y_min:.4f}, {self.data_summary.y_max:.4f}]</td></tr>")
        html_parts.append("</table>")

        # Coefficients
        html_parts.append("<h2>Coefficients</h2>")
        html_parts.append("<table>")

        has_ci = any(c.ci_lower is not None for c in self.coefficients)
        if has_ci:
            html_parts.append("<tr><th>Parameter</th><th>Value</th><th>95% CI</th><th>Bounds</th><th>Status</th></tr>")
        else:
            html_parts.append("<tr><th>Parameter</th><th>Value</th><th>Bounds</th><th>Status</th></tr>")

        all_coefs = list(self.coefficients)
        if self.intercept is not None:
            intercept_info = CoefficientInfo(
                name='Intercept',
                value=self.intercept.value,
                lower_bound=self.intercept.lower_bound,
                upper_bound=self.intercept.upper_bound,
                is_at_lower=self.intercept.is_at_lower,
                is_at_upper=self.intercept.is_at_upper,
                se=self.intercept.se,
                ci_lower=self.intercept.ci_lower,
                ci_upper=self.intercept.ci_upper,
            )
            all_coefs.append(intercept_info)

        for c in all_coefs:
            lb = f"{c.lower_bound:.4f}" if np.isfinite(c.lower_bound) else "-&infin;"
            ub = f"{c.upper_bound:.4f}" if np.isfinite(c.upper_bound) else "+&infin;"
            bounds_str = f"[{lb}, {ub}]"
            status_class = 'at-bound' if c.is_constrained else 'free'

            if has_ci and c.ci_lower is not None:
                ci_str = f"[{c.ci_lower:.4f}, {c.ci_upper:.4f}]"
                html_parts.append(f"<tr><td>{c.name}</td><td class='stat-value'>{c.value:.6f}</td><td class='stat-value'>{ci_str}</td><td>{bounds_str}</td><td class='{status_class}'>{c.bound_status}</td></tr>")
            elif has_ci:
                html_parts.append(f"<tr><td>{c.name}</td><td class='stat-value'>{c.value:.6f}</td><td>-</td><td>{bounds_str}</td><td class='{status_class}'>{c.bound_status}</td></tr>")
            else:
                html_parts.append(f"<tr><td>{c.name}</td><td class='stat-value'>{c.value:.6f}</td><td>{bounds_str}</td><td class='{status_class}'>{c.bound_status}</td></tr>")

        html_parts.append("</table>")

        # Fit Statistics
        html_parts.append("<h2>Fit Statistics</h2>")
        html_parts.append("<table>")
        html_parts.append(f"<tr><td>R²</td><td class='stat-value'>{self.fit_stats.r2:.4f}</td></tr>")
        html_parts.append(f"<tr><td>Adjusted R²</td><td class='stat-value'>{self.fit_stats.adj_r2:.4f}</td></tr>")
        html_parts.append(f"<tr><td>SEE</td><td class='stat-value'>{self.fit_stats.see:.4f}</td></tr>")
        html_parts.append(f"<tr><td>SPE</td><td class='stat-value'>{self.fit_stats.spe:.2%}</td></tr>")
        html_parts.append(f"<tr><td>MAPE</td><td class='stat-value'>{self.fit_stats.mape:.2%}</td></tr>")
        html_parts.append(f"<tr><td>RMSE</td><td class='stat-value'>{self.fit_stats.rmse:.4f}</td></tr>")
        html_parts.append(f"<tr><td>CV</td><td class='stat-value'>{self.fit_stats.cv:.2%}</td></tr>")
        html_parts.append(f"<tr><td>GDF ({self.fit_stats.gdf_method})</td><td class='stat-value'>{self.fit_stats.gdf:.1f}</td></tr>")
        html_parts.append("</table>")

        # Residual Analysis
        if self.residuals is not None:
            html_parts.append("<h2>Residual Analysis</h2>")
            html_parts.append("<table>")
            html_parts.append(f"<tr><td>Mean</td><td class='stat-value'>{self.residuals.mean:.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std</td><td class='stat-value'>{self.residuals.std:.4f}</td></tr>")
            html_parts.append(f"<tr><td>Range</td><td class='stat-value'>[{self.residuals.min:.4f}, {self.residuals.max:.4f}]</td></tr>")
            html_parts.append(f"<tr><td>Skewness</td><td class='stat-value'>{self.residuals.skewness:.4f}</td></tr>")
            html_parts.append(f"<tr><td>Kurtosis</td><td class='stat-value'>{self.residuals.kurtosis:.4f}</td></tr>")
            html_parts.append(f"<tr><td>Outliers (|r| > 2σ)</td><td class='stat-value'>{self.residuals.n_outliers}</td></tr>")
            html_parts.append("</table>")

        # Active Constraints
        if self.constraints.active_constraints:
            html_parts.append("<h2>Active Constraints</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Parameter</th><th>Bound Type</th></tr>")
            for name, bound_type in self.constraints.active_constraints:
                html_parts.append(f"<tr><td>{name}</td><td>{bound_type}</td></tr>")
            html_parts.append("</table>")

        html_parts.append("</div></body></html>")

        html_content = "\n".join(html_parts)

        if filepath is not None:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return html_content

    def to_pdf(self, filepath: str):
        """
        Export report to PDF format.

        Parameters
        ----------
        filepath : str
            Path to output PDF file.

        Notes
        -----
        Requires weasyprint or pdfkit to be installed.
        Falls back to HTML if PDF generation fails.
        """
        html_content = self.to_html()

        # Try weasyprint first
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(filepath)
            return
        except ImportError:
            pass

        # Try pdfkit
        try:
            import pdfkit
            pdfkit.from_string(html_content, filepath)
            return
        except ImportError:
            pass

        # Fall back to saving HTML with PDF extension warning
        warnings.warn(
            "Neither weasyprint nor pdfkit is installed. "
            "Install with: pip install weasyprint or pip install pdfkit. "
            "Saving as HTML instead.",
            UserWarning
        )
        html_filepath = filepath.replace('.pdf', '.html')
        self.to_html(html_filepath)

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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plot_diagnostics()")

        if self.residuals is None or self.residuals.residuals is None:
            raise ValueError("Residual data not available. Use full=True in summary().")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Diagnostic Plots', fontsize=14, fontweight='bold')

        residuals = self.residuals.residuals
        y_pred = self.residuals.y_pred

        # 1. Residuals vs Fitted
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')

        # Add lowess smoother if statsmodels available
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_idx = np.argsort(y_pred)
            smoothed = lowess(residuals[sorted_idx], y_pred[sorted_idx], frac=0.6)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='LOWESS')
            ax1.legend()
        except ImportError:
            pass

        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Normal Q-Q Plot')
        except ImportError:
            ax2.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
            ax2.set_title('Residual Distribution')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Density')

        # 3. Scale-Location Plot
        ax3 = axes[1, 0]
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        ax3.scatter(y_pred, sqrt_abs_resid, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Residuals|')
        ax3.set_title('Scale-Location')

        # 4. Coefficient Plot with Bounds
        ax4 = axes[1, 1]
        names = [c.name for c in self.coefficients]
        values = [c.value for c in self.coefficients]
        lowers = [c.lower_bound if np.isfinite(c.lower_bound) else min(values) - 1 for c in self.coefficients]
        uppers = [c.upper_bound if np.isfinite(c.upper_bound) else max(values) + 1 for c in self.coefficients]

        y_pos = np.arange(len(names))
        colors = ['red' if c.is_constrained else 'steelblue' for c in self.coefficients]

        ax4.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

        # Add bound markers
        for i, (lb, ub) in enumerate(zip(lowers, uppers)):
            if np.isfinite(self.coefficients[i].lower_bound):
                ax4.plot(lb, i, 'k<', markersize=8)
            if np.isfinite(self.coefficients[i].upper_bound):
                ax4.plot(ub, i, 'k>', markersize=8)

        # Add CI bars if available
        for i, c in enumerate(self.coefficients):
            if c.ci_lower is not None and c.ci_upper is not None:
                ax4.plot([c.ci_lower, c.ci_upper], [i, i], 'k-', linewidth=2)
                ax4.plot([c.ci_lower, c.ci_upper], [i, i], 'k|', markersize=10)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(names)
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Coefficient Value')
        ax4.set_title('Coefficients with Bounds')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Free'),
            Patch(facecolor='red', alpha=0.7, edgecolor='black', label='At Bound'),
        ]
        ax4.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


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
        Note: For bootstrap CIs, use bootstrap=True instead.
    bootstrap : bool, default=False
        If True, compute bootstrap confidence intervals (overrides ci_method).
        Can be slow for large datasets or many bootstrap samples.
    n_bootstrap : int, default=1000
        Number of bootstrap samples (only used if bootstrap=True).
    confidence : float, default=0.95
        Confidence level for intervals.
    random_state : int, optional
        Random seed for bootstrap.

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
    """
    check_is_fitted(model)
    X = np.asarray(X)
    y = np.asarray(y)

    # Record report generation time
    report_datetime = datetime.now()

    # Get base diagnostics
    diag = ModelDiagnostics(model, X, y, gdf_method=gdf_method)

    # Model Specification with timing info
    loss_str = model.loss if isinstance(model.loss, str) else 'custom'
    model_spec = ModelSpecification(
        model_type=type(model).__name__,
        loss_function=loss_str,
        alpha=model.alpha,
        l1_ratio=model.l1_ratio,
        fit_intercept=model.fit_intercept,
        method=model.method,
        converged=model.converged_,
        n_iterations=getattr(model.optimization_result_, 'nit', None),
        final_objective=model.optimization_result_.fun if hasattr(model, 'optimization_result_') else None,
        fit_datetime=getattr(model, 'fit_datetime_', None),
        fit_duration_seconds=getattr(model, 'fit_duration_seconds_', None),
    )

    # Data Summary
    feature_names = list(model.feature_names_in_) if model.feature_names_in_ is not None else None
    data_summary = DataSummary(
        n_samples=len(y),
        n_features=X.shape[1],
        y_mean=float(np.mean(y)),
        y_std=float(np.std(y)),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        feature_names=feature_names,
    )

    # Determine CI method and compute
    actual_ci_method = 'none'
    bootstrap_results = None
    hessian_se = None

    # Bootstrap takes priority if requested
    if bootstrap:
        actual_ci_method = 'bootstrap'
        try:
            bootstrap_results = bootstrap_confidence_intervals(
                type(model), X, y,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                random_state=random_state,
                **model.get_params()
            )
        except Exception as e:
            warnings.warn(f"Bootstrap failed: {e}. Falling back to Hessian.", UserWarning)
            actual_ci_method = 'hessian'
            try:
                hessian_se = hessian_standard_errors(model, X, y)
            except Exception as e2:
                warnings.warn(f"Hessian SE also failed: {e2}", UserWarning)
                actual_ci_method = 'none'
    elif ci_method == 'hessian':
        actual_ci_method = 'hessian'
        try:
            hessian_se = hessian_standard_errors(model, X, y)
        except Exception as e:
            warnings.warn(f"Hessian SE failed: {e}", UserWarning)
            actual_ci_method = 'none'

    # Z-score for confidence interval
    from scipy import stats
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Coefficients
    coefficients = []
    for i, coef_val in enumerate(model.coef_):
        if model.feature_names_in_ is not None:
            name = model.feature_names_in_[i]
        else:
            name = f"beta_{i}"

        lb, ub = model._bounds_parsed[i]

        # Check if at bound
        is_at_lower = np.isfinite(lb) and np.abs(coef_val - lb) < 1e-6
        is_at_upper = np.isfinite(ub) and np.abs(coef_val - ub) < 1e-6

        # Get CI based on method
        ci_lower = None
        ci_upper = None
        se = None

        if bootstrap_results is not None:
            ci_lower = float(bootstrap_results['coef_ci_lower'][i])
            ci_upper = float(bootstrap_results['coef_ci_upper'][i])
            se = float(bootstrap_results['coef_std'][i])
        elif hessian_se is not None and i < len(hessian_se) and not np.isnan(hessian_se[i]):
            se = float(hessian_se[i])
            ci_lower = float(coef_val - z_score * se)
            ci_upper = float(coef_val + z_score * se)

        coefficients.append(CoefficientInfo(
            name=str(name),
            value=float(coef_val),
            lower_bound=float(lb),
            upper_bound=float(ub),
            is_at_lower=is_at_lower,
            is_at_upper=is_at_upper,
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

        ci_lower = None
        ci_upper = None
        se = None

        if bootstrap_results is not None:
            ci_lower = float(bootstrap_results['intercept_ci'][0])
            ci_upper = float(bootstrap_results['intercept_ci'][1])
            se = float(bootstrap_results['intercept_std'])
        elif hessian_se is not None:
            # Intercept is last parameter in hessian_se
            int_idx = len(model.coef_)
            if int_idx < len(hessian_se) and not np.isnan(hessian_se[int_idx]):
                se = float(hessian_se[int_idx])
                ci_lower = float(model.intercept_ - z_score * se)
                ci_upper = float(model.intercept_ + z_score * se)

        intercept_info = CoefficientInfo(
            name='Intercept',
            value=float(model.intercept_),
            lower_bound=float(int_lb),
            upper_bound=float(int_ub),
            is_at_lower=is_at_lower,
            is_at_upper=is_at_upper,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

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
    )


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
