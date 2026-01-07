"""
Data classes for diagnostic reports.

These classes hold structured information about model fits, coefficients,
and diagnostic statistics for use in summary reports and exports.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class CoefficientInfo:
    """Information about a single coefficient with both Hessian and Bootstrap SE/CI."""
    name: str
    value: float
    lower_bound: float
    upper_bound: float
    is_at_lower: bool = False
    is_at_upper: bool = False
    # Hessian-based standard errors and CI
    hessian_se: Optional[float] = None
    hessian_ci_lower: Optional[float] = None
    hessian_ci_upper: Optional[float] = None
    # Bootstrap-based standard errors and CI (constrained)
    bootstrap_se: Optional[float] = None
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    # Legacy fields for backwards compatibility (point to best available)
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
            return f"{self.value:.{decimals}f} +/- {self.se:.{decimals}f}"
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
    # Additional statistics
    aic: Optional[float] = None  # Akaike Information Criterion
    bic: Optional[float] = None  # Bayesian Information Criterion
    f_statistic: Optional[float] = None  # F-statistic for model significance
    f_pvalue: Optional[float] = None  # p-value for F-test
    durbin_watson: Optional[float] = None  # Durbin-Watson statistic for autocorrelation
    mse: Optional[float] = None  # Mean Squared Error
    mae: Optional[float] = None  # Mean Absolute Error

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            'R2': self.r2,
            'Adjusted R2': self.adj_r2,
            'SEE': self.see,
            'SPE': self.spe,
            'MAPE': self.mape,
            'RMSE': self.rmse,
            'CV': self.cv,
            'GDF': self.gdf,
        }
        # Add optional statistics if computed
        if self.aic is not None:
            result['AIC'] = self.aic
        if self.bic is not None:
            result['BIC'] = self.bic
        if self.f_statistic is not None:
            result['F-Statistic'] = self.f_statistic
        if self.f_pvalue is not None:
            result['F p-value'] = self.f_pvalue
        if self.durbin_watson is not None:
            result['Durbin-Watson'] = self.durbin_watson
        if self.mse is not None:
            result['MSE'] = self.mse
        if self.mae is not None:
            result['MAE'] = self.mae
        return result


@dataclass
class ModelSpecification:
    """
    Model specification details - all input parameters needed to recreate the model.

    This dataclass captures all constructor parameters so the model can be
    recreated from the report.
    """
    # Core model type
    model_type: str

    # Loss and penalty settings
    loss_function: str
    alpha: float
    l1_ratio: float

    # Structural settings
    fit_intercept: bool
    scale: bool = False

    # Bounds (stored as original format for reproducibility)
    bounds: Optional[Dict] = None
    intercept_bounds: Optional[Tuple[float, float]] = None

    # Naming
    coef_names: Optional[List[str]] = None
    penalty_exclude: Optional[List[str]] = None

    # Optimization settings
    method: str = 'SLSQP'
    max_iter: int = 1000
    tol: float = 1e-6
    x0: Optional[str] = 'ols'  # Initial values - can be 'ols', 'zeros', or array

    # Custom function (stored as string representation)
    prediction_fn_source: Optional[str] = None
    has_custom_prediction_fn: bool = False
    has_custom_loss_fn: bool = False

    # Fit results
    converged: bool = False
    n_iterations: Optional[int] = None
    final_objective: Optional[float] = None
    fit_datetime: Optional[datetime] = None
    fit_duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type,
            'loss_function': self.loss_function,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'scale': self.scale,
            'bounds': self.bounds,
            'intercept_bounds': self.intercept_bounds,
            'coef_names': self.coef_names,
            'penalty_exclude': self.penalty_exclude,
            'method': self.method,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'x0': self.x0 if isinstance(self.x0, str) else list(self.x0) if self.x0 is not None else None,
            'has_custom_prediction_fn': self.has_custom_prediction_fn,
            'has_custom_loss_fn': self.has_custom_loss_fn,
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'final_objective': self.final_objective,
            'fit_datetime': self.fit_datetime.isoformat() if self.fit_datetime else None,
            'fit_duration_seconds': self.fit_duration_seconds,
        }


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
class SampleData:
    """
    Sample data for inclusion in reports.

    Attributes
    ----------
    X_sample : np.ndarray
        First n rows of feature matrix
    y_sample : np.ndarray
        First n rows of target values
    y_pred_sample : np.ndarray
        Predictions for sample rows
    n_total : int
        Total dataset size
    n_sample : int
        Number of sample rows included
    x_column_names : Optional[List[str]]
        Names of X matrix columns (from DataFrame columns or auto-generated X1, X2, ...).
        These are DIFFERENT from parameter/coefficient names when using custom prediction_fn.
        For example, X columns might be ['midpoint', 'quantity'] while parameters are ['T1', 'LC', 'RC'].
    """
    X_sample: np.ndarray
    y_sample: np.ndarray
    y_pred_sample: np.ndarray
    n_total: int
    n_sample: int
    x_column_names: Optional[List[str]] = None


@dataclass
class ModelEquation:
    """
    Model equation representation.

    Attributes
    ----------
    text : str
        Plain text equation (e.g., "y = 1.5 + 0.3*x1 - 0.2*x2")
    latex : Optional[str]
        LaTeX formatted equation
    source : Optional[str]
        Source code for custom prediction functions
    is_custom : bool
        Whether this is a custom (non-linear) model
    """
    text: str
    latex: Optional[str] = None
    source: Optional[str] = None
    is_custom: bool = False


@dataclass
class AlphaTraceResult:
    """
    Results from alpha trace analysis.

    Attributes
    ----------
    trace_df : pd.DataFrame
        Full trace DataFrame with columns: alpha, l1_ratio, loss_value, converged, coef_*
    summary_df : pd.DataFrame
        Summary by l1_ratio showing best alpha for each
    optimal : dict
        Optimal hyperparameters and coefficients
    l1_ratios : List[float]
        L1 ratios that were evaluated
    n_alphas : int
        Number of alpha values evaluated per l1_ratio
    """
    trace_df: 'pd.DataFrame'  # Forward reference to avoid import
    summary_df: 'pd.DataFrame'
    optimal: Dict
    l1_ratios: List[float]
    n_alphas: int


@dataclass
class BootstrapCoefResults:
    """
    Bootstrap results for a single run (constrained or unconstrained).

    Attributes
    ----------
    coef_mean : np.ndarray
        Mean of bootstrap coefficient estimates
    coef_std : np.ndarray
        Standard deviation of bootstrap coefficient estimates
    coef_ci_lower : np.ndarray
        Lower confidence interval bounds for coefficients
    coef_ci_upper : np.ndarray
        Upper confidence interval bounds for coefficients
    intercept_mean : Optional[float]
        Mean of bootstrap intercept estimates
    intercept_std : Optional[float]
        Standard deviation of bootstrap intercept estimates
    intercept_ci : Optional[Tuple[float, float]]
        Confidence interval for intercept (lower, upper)
    bootstrap_coefs : np.ndarray
        All bootstrap coefficient samples (n_bootstrap x n_features)
    n_successful : int
        Number of successful bootstrap fits
    """
    coef_mean: np.ndarray
    coef_std: np.ndarray
    coef_ci_lower: np.ndarray
    coef_ci_upper: np.ndarray
    intercept_mean: Optional[float]
    intercept_std: Optional[float]
    intercept_ci: Optional[Tuple[float, float]]
    bootstrap_coefs: np.ndarray
    n_successful: int


@dataclass
class BootstrapResults:
    """
    Results from bootstrap confidence interval analysis.

    Contains both constrained (with bounds/alpha) and unconstrained bootstrap results
    to allow comparison of uncertainty estimates.

    Attributes
    ----------
    constrained : BootstrapCoefResults
        Bootstrap results using the original model settings (with bounds and alpha)
    unconstrained : Optional[BootstrapCoefResults]
        Bootstrap results without bounds and alpha=0 (pure OLS bootstrap)
    n_bootstrap : int
        Number of bootstrap samples requested
    confidence : float
        Confidence level used (e.g., 0.95)
    feature_names : Optional[List[str]]
        Names of features for labeling
    """
    constrained: BootstrapCoefResults
    unconstrained: Optional[BootstrapCoefResults]
    n_bootstrap: int
    confidence: float
    feature_names: Optional[List[str]] = None

    # Convenience properties - prefer constrained if available, else unconstrained
    @property
    def _primary(self) -> BootstrapCoefResults:
        """Return the primary results (constrained if available, else unconstrained)."""
        return self.constrained if self.constrained is not None else self.unconstrained

    @property
    def coef_mean(self) -> np.ndarray:
        return self._primary.coef_mean

    @property
    def coef_std(self) -> np.ndarray:
        return self._primary.coef_std

    @property
    def coef_ci_lower(self) -> np.ndarray:
        return self._primary.coef_ci_lower

    @property
    def coef_ci_upper(self) -> np.ndarray:
        return self._primary.coef_ci_upper

    @property
    def intercept_mean(self) -> Optional[float]:
        return self._primary.intercept_mean

    @property
    def intercept_std(self) -> Optional[float]:
        return self._primary.intercept_std

    @property
    def intercept_ci(self) -> Optional[Tuple[float, float]]:
        return self._primary.intercept_ci

    @property
    def bootstrap_coefs(self) -> np.ndarray:
        return self._primary.bootstrap_coefs

    @property
    def n_successful(self) -> int:
        return self._primary.n_successful

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'n_bootstrap': self.n_bootstrap,
            'confidence': self.confidence,
            'feature_names': self.feature_names,
        }
        if self.constrained is not None:
            result['constrained'] = {
                'coef_mean': self.constrained.coef_mean.tolist() if self.constrained.coef_mean is not None else None,
                'coef_std': self.constrained.coef_std.tolist() if self.constrained.coef_std is not None else None,
                'coef_ci_lower': self.constrained.coef_ci_lower.tolist() if self.constrained.coef_ci_lower is not None else None,
                'coef_ci_upper': self.constrained.coef_ci_upper.tolist() if self.constrained.coef_ci_upper is not None else None,
                'intercept_mean': self.constrained.intercept_mean,
                'intercept_std': self.constrained.intercept_std,
                'intercept_ci': list(self.constrained.intercept_ci) if self.constrained.intercept_ci else None,
                'n_successful': self.constrained.n_successful,
            }
        if self.unconstrained is not None:
            result['unconstrained'] = {
                'coef_mean': self.unconstrained.coef_mean.tolist() if self.unconstrained.coef_mean is not None else None,
                'coef_std': self.unconstrained.coef_std.tolist() if self.unconstrained.coef_std is not None else None,
                'coef_ci_lower': self.unconstrained.coef_ci_lower.tolist() if self.unconstrained.coef_ci_lower is not None else None,
                'coef_ci_upper': self.unconstrained.coef_ci_upper.tolist() if self.unconstrained.coef_ci_upper is not None else None,
                'intercept_mean': self.unconstrained.intercept_mean,
                'intercept_std': self.unconstrained.intercept_std,
                'intercept_ci': list(self.unconstrained.intercept_ci) if self.unconstrained.intercept_ci else None,
                'n_successful': self.unconstrained.n_successful,
            }
        return result

    def summary_dataframe(self, include_unconstrained: bool = True) -> 'pd.DataFrame':
        """Return a summary DataFrame of bootstrap statistics by coefficient."""
        import pandas as pd

        names = self.feature_names if self.feature_names else [f'coef_{i}' for i in range(len(self.coef_mean))]

        rows = []
        for i, name in enumerate(names):
            row = {
                'Parameter': name,
                'Constrained Mean': self.constrained.coef_mean[i],
                'Constrained Std': self.constrained.coef_std[i],
                'Constrained CI Lower': self.constrained.coef_ci_lower[i],
                'Constrained CI Upper': self.constrained.coef_ci_upper[i],
            }
            if include_unconstrained and self.unconstrained is not None:
                row['Unconstrained Mean'] = self.unconstrained.coef_mean[i]
                row['Unconstrained Std'] = self.unconstrained.coef_std[i]
                row['Unconstrained CI Lower'] = self.unconstrained.coef_ci_lower[i]
                row['Unconstrained CI Upper'] = self.unconstrained.coef_ci_upper[i]
            rows.append(row)

        if self.constrained.intercept_mean is not None:
            row = {
                'Parameter': 'Intercept',
                'Constrained Mean': self.constrained.intercept_mean,
                'Constrained Std': self.constrained.intercept_std,
                'Constrained CI Lower': self.constrained.intercept_ci[0] if self.constrained.intercept_ci else None,
                'Constrained CI Upper': self.constrained.intercept_ci[1] if self.constrained.intercept_ci else None,
            }
            if include_unconstrained and self.unconstrained is not None and self.unconstrained.intercept_mean is not None:
                row['Unconstrained Mean'] = self.unconstrained.intercept_mean
                row['Unconstrained Std'] = self.unconstrained.intercept_std
                row['Unconstrained CI Lower'] = self.unconstrained.intercept_ci[0] if self.unconstrained.intercept_ci else None
                row['Unconstrained CI Upper'] = self.unconstrained.intercept_ci[1] if self.unconstrained.intercept_ci else None
            rows.append(row)

        return pd.DataFrame(rows)
