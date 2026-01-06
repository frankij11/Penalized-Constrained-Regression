"""
Diagnostics module for penalized-constrained regression.

Provides:
- Generalized Degrees of Freedom (GDF) computation
- Fit statistics (SEE, SPE, adjusted R2)
- Bootstrap confidence intervals
- Hessian-based standard errors
- Comprehensive summary reports with multiple export formats
- Model equation formatting
- Diagnostic plots
"""

# Core diagnostics
from .core import (
    ModelDiagnostics,
    compute_gdf_hu,
    compute_gdf_gaines,
)

# Data classes
from .dataclasses import (
    CoefficientInfo,
    FitStatistics,
    ModelSpecification,
    DataSummary,
    ConstraintSummary,
    ResidualAnalysis,
    SampleData,
    ModelEquation,
    AlphaTraceResult,
    BootstrapCoefResults,
    BootstrapResults,
)

# Reporting
from .reporting import (
    SummaryReport,
    generate_summary_report,
)

# Confidence intervals
from .confidence import (
    bootstrap_confidence_intervals,
    hessian_standard_errors,
)

# Plotting
from .plotting import (
    plot_diagnostics,
    figure_to_base64,
    generate_embedded_plots,
)

# Equations
from .equations import (
    format_model_equation,
    format_linear_equation,
    get_callable_source,
    format_loss_function,
)

# Export functions
from .export import (
    to_html,
    to_pdf,
    to_excel,
)

# Alpha trace analysis
from .alpha_trace import (
    compute_alpha_trace,
    plot_alpha_trace,
    find_optimal_alpha,
    summarize_alpha_trace,
)

__all__ = [
    # Core
    'ModelDiagnostics',
    'compute_gdf_hu',
    'compute_gdf_gaines',
    # Data classes
    'CoefficientInfo',
    'FitStatistics',
    'ModelSpecification',
    'DataSummary',
    'ConstraintSummary',
    'ResidualAnalysis',
    'SampleData',
    'ModelEquation',
    'AlphaTraceResult',
    'BootstrapCoefResults',
    'BootstrapResults',
    # Reporting
    'SummaryReport',
    'generate_summary_report',
    # Confidence
    'bootstrap_confidence_intervals',
    'hessian_standard_errors',
    # Plotting
    'plot_diagnostics',
    'figure_to_base64',
    'generate_embedded_plots',
    # Equations
    'format_model_equation',
    'format_linear_equation',
    'get_callable_source',
    'format_loss_function',
    # Export
    'to_html',
    'to_pdf',
    'to_excel',
    # Alpha trace
    'compute_alpha_trace',
    'plot_alpha_trace',
    'find_optimal_alpha',
    'summarize_alpha_trace',
]
