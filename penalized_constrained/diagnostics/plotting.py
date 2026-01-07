"""
Diagnostic plotting utilities.

Generates diagnostic plots for model assessment and provides
utilities for embedding plots in reports.
"""

import io
import base64
from typing import Tuple, Optional, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataclasses import CoefficientInfo, ResidualAnalysis


def plot_diagnostics(
    residuals: 'ResidualAnalysis',
    coefficients: List['CoefficientInfo'],
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    title: str = 'Model Diagnostic Plots'
):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    residuals : ResidualAnalysis
        Residual analysis data (must have residuals and y_pred arrays)
    coefficients : List[CoefficientInfo]
        List of coefficient information
    figsize : tuple, default=(12, 10)
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.
    title : str, default='Model Diagnostic Plots'
        Figure title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_diagnostics()")

    if residuals is None or residuals.residuals is None:
        raise ValueError("Residual data not available. Use full=True in summary().")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    resid_values = residuals.residuals
    y_pred = residuals.y_pred

    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, resid_values, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')

    # Add lowess smoother if statsmodels available
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(y_pred)
        smoothed = lowess(resid_values[sorted_idx], y_pred[sorted_idx], frac=0.6)
        ax1.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='LOWESS')
        ax1.legend()
    except ImportError:
        pass

    # 2. Q-Q Plot
    ax2 = axes[0, 1]
    try:
        from scipy import stats
        stats.probplot(resid_values, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
    except ImportError:
        ax2.hist(resid_values, bins=20, density=True, alpha=0.7, edgecolor='black')
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')

    # 3. Scale-Location Plot
    ax3 = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(resid_values))
    ax3.scatter(y_pred, sqrt_abs_resid, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('sqrt(|Residuals|)')
    ax3.set_title('Scale-Location')

    # 4. Coefficient Plot with Bounds
    # Show each coefficient on its own scale to handle different magnitudes
    ax4 = axes[1, 1]

    n_coefs = len(coefficients)
    if n_coefs == 0:
        ax4.text(0.5, 0.5, 'No coefficients', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Coefficients with Bounds')
    else:
        # Create mini-plots for each coefficient within the subplot
        # This handles coefficients on very different scales (e.g., T1=100, LC=0.9)
        y_positions = np.arange(n_coefs)
        bar_height = 0.6

        for i, c in enumerate(coefficients):
            value = c.value
            lb = c.lower_bound if np.isfinite(c.lower_bound) else None
            ub = c.upper_bound if np.isfinite(c.upper_bound) else None

            # Determine x-axis range for this coefficient
            points = [value]
            if lb is not None:
                points.append(lb)
            if ub is not None:
                points.append(ub)
            if c.ci_lower is not None:
                points.append(c.ci_lower)
            if c.ci_upper is not None:
                points.append(c.ci_upper)

            x_min, x_max = min(points), max(points)
            x_range = x_max - x_min if x_max != x_min else abs(value) * 0.2 or 1.0
            x_min -= x_range * 0.1
            x_max += x_range * 0.1

            # Normalize position within [0, 1] for this coefficient's row
            def normalize(val):
                if x_max == x_min:
                    return 0.5
                return (val - x_min) / (x_max - x_min)

            # Color based on constraint status
            color = 'red' if c.is_constrained else 'steelblue'

            # Draw coefficient value as a dot
            norm_val = normalize(value)
            ax4.plot(norm_val, i, 'o', color=color, markersize=12, zorder=3)

            # Draw bounds as markers
            if lb is not None:
                norm_lb = normalize(lb)
                ax4.plot(norm_lb, i, 'k<', markersize=8, zorder=2)
            if ub is not None:
                norm_ub = normalize(ub)
                ax4.plot(norm_ub, i, 'k>', markersize=8, zorder=2)

            # Draw CI if available
            if c.ci_lower is not None and c.ci_upper is not None:
                norm_ci_low = normalize(c.ci_lower)
                norm_ci_high = normalize(c.ci_upper)
                ax4.plot([norm_ci_low, norm_ci_high], [i, i], 'k-', linewidth=2, zorder=1)
                ax4.plot([norm_ci_low, norm_ci_high], [i, i], 'k|', markersize=10, zorder=2)

            # Draw horizontal line for context
            ax4.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

            # Add value label
            ax4.annotate(f'{value:.4g}', (norm_val, i), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=8)

        ax4.set_yticks(y_positions)
        ax4.set_yticklabels([c.name for c in coefficients])
        ax4.set_xlim(-0.1, 1.1)
        ax4.set_ylim(-0.5, n_coefs - 0.5)
        ax4.set_xlabel('Relative Position (scaled per coefficient)')
        ax4.set_title('Coefficients with Bounds\n(each row scaled independently)')

        # Remove x-axis ticks since each row has different scale
        ax4.set_xticks([])

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Free'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='At Bound'),
            Line2D([0], [0], marker='<', color='black', markersize=8, linestyle='', label='Lower Bound'),
            Line2D([0], [0], marker='>', color='black', markersize=8, linestyle='', label='Upper Bound'),
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def figure_to_base64(fig, format: str = 'png', dpi: int = 150) -> str:
    """
    Convert matplotlib figure to base64 string for HTML embedding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert
    format : str, default='png'
        Image format ('png', 'svg', 'jpg')
    dpi : int, default=150
        Resolution for raster formats

    Returns
    -------
    str
        Base64 encoded image string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_embedded_plots(
    residuals: 'ResidualAnalysis',
    coefficients: List['CoefficientInfo'],
    figsize: Tuple[int, int] = (12, 10),
    format: str = 'png',
    dpi: int = 150,
    close_after: bool = True
) -> dict:
    """
    Generate all plots and return as base64 dict for embedding.

    Parameters
    ----------
    residuals : ResidualAnalysis
        Residual analysis data
    coefficients : List[CoefficientInfo]
        Coefficient information
    figsize : tuple, default=(12, 10)
        Figure size
    format : str, default='png'
        Image format
    dpi : int, default=150
        Image resolution
    close_after : bool, default=True
        Whether to close the figure after encoding

    Returns
    -------
    dict
        {'diagnostics': base64_string, 'format': 'png'}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {'diagnostics': None, 'format': format, 'error': 'matplotlib not available'}

    if residuals is None or residuals.residuals is None:
        return {'diagnostics': None, 'format': format, 'error': 'No residual data'}

    try:
        fig = plot_diagnostics(residuals, coefficients, figsize=figsize)
        encoded = figure_to_base64(fig, format=format, dpi=dpi)

        if close_after:
            plt.close(fig)

        return {
            'diagnostics': encoded,
            'format': format,
            'error': None
        }
    except Exception as e:
        return {
            'diagnostics': None,
            'format': format,
            'error': str(e)
        }
