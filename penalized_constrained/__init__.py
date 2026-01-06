"""
Penalized-Constrained Regression
================================

A scikit-learn compatible implementation of penalized regression with 
coefficient constraints, designed for cost estimation with small, 
correlated datasets.

Main Classes
------------
PenalizedConstrainedRegression : Base estimator with L1/L2 penalties and bounds
PenalizedConstrainedCV : Cross-validated version with automatic tuning

Quick Start
-----------
>>> import penalized_constrained as pcreg
>>> from penalized_constrained import PenalizedConstrainedCV
>>> 
>>> # Fit with automatic hyperparameter tuning
>>> model = PenalizedConstrainedCV(
...     bounds=[(-1, 0), (-1, 0)],  # Constrain slopes to be negative
...     loss='sspe'                  # Use MUPE-consistent loss
... )
>>> model.fit(X_log, y_log)
>>> print(model.coef_)

References
----------
"Small Data, Big Problems: Can Constraints and Penalties Save Regression?"
ICEAA 2026 Professional Development & Training Workshop
Kevin Joy, Max Watstein, Herren Associates
"""

from .regression import PenalizedConstrainedRegression, PCRegression
from .cv import PenalizedConstrainedCV, PCCV
from .diagnostics import (
    ModelDiagnostics,
    compute_gdf_hu,
    compute_gdf_gaines,
    bootstrap_confidence_intervals,
    hessian_standard_errors,
    # Summary report classes
    SummaryReport,
    CoefficientInfo,
    FitStatistics,
    ModelSpecification,
    DataSummary,
    ConstraintSummary,
    ResidualAnalysis,
    SampleData,
    ModelEquation,
    AlphaTraceResult,
    generate_summary_report,
    # Equation formatting
    format_model_equation,
    format_linear_equation,
    get_callable_source,
    # Alpha trace analysis
    compute_alpha_trace,
    plot_alpha_trace,
    find_optimal_alpha,
    summarize_alpha_trace,
)
from .utils import (
    generate_correlated_learning_data,
    generate_test_data,
    calculate_lot_midpoint,
    learning_rate_to_slope,
    slope_to_learning_rate,
    compute_vif,
    compute_condition_number,
    multicollinearity_diagnostics,
    print_multicollinearity_report
)

__version__ = "0.1.0"
__author__ = "Kevin Joy, Max Watstein"
__email__ = "kjoy@herrenassociates.com"

__all__ = [
    # Core classes
    'PenalizedConstrainedRegression',
    'PCRegression',
    'PenalizedConstrainedCV',
    'PCCV',

    # Diagnostics
    'ModelDiagnostics',
    'compute_gdf_hu',
    'compute_gdf_gaines',
    'bootstrap_confidence_intervals',
    'hessian_standard_errors',

    # Summary Report
    'SummaryReport',
    'CoefficientInfo',
    'FitStatistics',
    'ModelSpecification',
    'DataSummary',
    'ConstraintSummary',
    'ResidualAnalysis',
    'SampleData',
    'ModelEquation',
    'AlphaTraceResult',
    'generate_summary_report',
    'format_model_equation',
    'format_linear_equation',
    'get_callable_source',

    # Alpha trace
    'compute_alpha_trace',
    'plot_alpha_trace',
    'find_optimal_alpha',
    'summarize_alpha_trace',

    # Utilities
    'generate_correlated_learning_data',
    'generate_test_data',
    'calculate_lot_midpoint',
    'learning_rate_to_slope',
    'slope_to_learning_rate',
    'compute_vif',
    'compute_condition_number',
    'multicollinearity_diagnostics',
    'print_multicollinearity_report',
]
