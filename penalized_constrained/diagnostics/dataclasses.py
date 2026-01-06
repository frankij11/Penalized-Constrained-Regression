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
    feature_names : Optional[List[str]]
        Names of features
    """
    X_sample: np.ndarray
    y_sample: np.ndarray
    y_pred_sample: np.ndarray
    n_total: int
    n_sample: int
    feature_names: Optional[List[str]] = None


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
