"""
ICEAA Simulation Analysis Module

Provides data loading and visualization utilities for Quarto integration.

Usage in Quarto .qmd files:
    ```{python}
    from scripts.ICEAA.analysis import load_simulation_results, load_predictions
    df = load_simulation_results()
    ```
"""

from .load_results import (
    load_simulation_results,
    load_predictions,
    load_config,
    RESULTS_DIR,
)

from .visualization import (
    create_overall_boxplot,
    create_boxplot_by_factor,
    create_win_rate_heatmap,
    create_sign_correctness_plot,
    create_model_ranking_chart,
)

__all__ = [
    "load_simulation_results",
    "load_predictions",
    "load_config",
    "RESULTS_DIR",
    "create_overall_boxplot",
    "create_boxplot_by_factor",
    "create_win_rate_heatmap",
    "create_sign_correctness_plot",
    "create_model_ranking_chart",
]
