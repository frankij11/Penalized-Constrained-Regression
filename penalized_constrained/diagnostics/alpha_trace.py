"""
Alpha trace analysis for penalized-constrained regression.

Provides functions to analyze how coefficients and loss change across
different alpha values and l1_ratios, useful for hyperparameter selection.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import warnings
from sklearn.base import clone


def compute_alpha_trace(
    model,
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[Union[np.ndarray, List[float]]] = None,
    l1_ratios: Optional[Union[np.ndarray, List[float]]] = None,
    n_alphas: int = 50,
    alpha_min_ratio: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute coefficient and loss values across a grid of alpha and l1_ratio values.

    Clones the provided model and varies only alpha and l1_ratio, preserving all
    other settings including prediction_fn, bounds, loss function, etc.

    Parameters
    ----------
    model : PenalizedConstrainedRegression or PenalizedConstrainedCV
        A fitted model to clone. All settings (bounds, prediction_fn, loss, etc.)
        are preserved in the cloned models. For CV models, the best_estimator_
        is used as the template.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    alphas : array-like, optional
        Specific alpha values to evaluate. If None, generates logarithmic grid.
    l1_ratios : array-like, optional
        L1 ratio values to evaluate. Default is [0.0, 0.5, 1.0] (Ridge, ElasticNet, Lasso).
    n_alphas : int, default=50
        Number of alpha values to generate if alphas is None.
    alpha_min_ratio : float, default=1e-4
        Ratio of alpha_min to alpha_max when generating alpha grid.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - alpha: Alpha value
        - l1_ratio: L1 ratio value
        - loss_value: Objective value at optimum
        - converged: Whether optimization converged
        - coef_{name}: Coefficient values for each feature
        - intercept: Intercept value (if fit_intercept=True)

    Examples
    --------
    >>> model = PenalizedConstrainedRegression(bounds=[(-1, 0), (-1, 0)])
    >>> model.fit(X, y)
    >>> trace_df = compute_alpha_trace(model, X, y, l1_ratios=[0, 0.5, 1.0])
    >>> print(trace_df[['alpha', 'l1_ratio', 'loss_value']])
    """
    # Ensure X and y are arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # For CV models, use the best_estimator_ as the template for cloning
    # This gives us a PenalizedConstrainedRegression which has alpha/l1_ratio params
    base_model = model
    if hasattr(model, 'best_estimator_'):
        base_model = model.best_estimator_

    # Get feature names from the model
    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        feature_names = list(model.feature_names_in_)
    else:
        # For models with custom prediction_fn, use feature_names param
        if hasattr(model, 'feature_names') and model.feature_names is not None:
            feature_names = list(model.feature_names)
        else:
            feature_names = [f'x{i+1}' for i in range(X.shape[1])]

    n_params = len(feature_names)

    # Get fit_intercept from model
    fit_intercept = getattr(model, 'fit_intercept', True)

    # Default l1_ratios: Ridge, ElasticNet 50/50, and Lasso
    if l1_ratios is None:
        l1_ratios = [0.0, 0.5, 1.0]
    l1_ratios = np.asarray(l1_ratios)

    # Generate alpha grid if not provided
    if alphas is None:
        # Compute alpha_max (value where all coefficients would be zero for Lasso)
        # For SSPE/SSE, approximate using correlation-based heuristic
        y_centered = y - np.mean(y)
        X_centered = X - np.mean(X, axis=0)

        # Approximate alpha_max
        correlations = np.abs(X_centered.T @ y_centered) / len(y)
        alpha_max = np.max(correlations) * 2

        # Ensure alpha_max is reasonable
        alpha_max = max(alpha_max, 1.0)
        alpha_min = alpha_max * alpha_min_ratio

        # Log-spaced alphas from max to min
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
    else:
        alphas = np.asarray(alphas)
        alphas = np.sort(alphas)[::-1]  # Sort descending (large to small)

    # Results storage
    results = []

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            # Clone the base model to preserve all settings (prediction_fn, bounds, etc.)
            # For CV models, we use best_estimator_ which has alpha/l1_ratio params
            cloned = clone(base_model)
            cloned.set_params(alpha=alpha, l1_ratio=l1_ratio)

            # Suppress convergence warnings for speed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cloned.fit(X, y)
                    converged = getattr(cloned, 'converged_', True)

                    # Get loss value
                    loss_value = cloned.optimization_result_.fun if hasattr(cloned, 'optimization_result_') else np.nan

                except Exception:
                    converged = False
                    loss_value = np.nan
                    # Create placeholder for failed fits
                    cloned.coef_ = np.full(n_params, np.nan)
                    if fit_intercept:
                        cloned.intercept_ = np.nan

            # Build result row
            row = {
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'loss_value': loss_value,
                'converged': converged,
            }

            # Add coefficients
            for i, name in enumerate(feature_names):
                row[f'coef_{name}'] = cloned.coef_[i] if hasattr(cloned, 'coef_') else np.nan

            # Add intercept
            if fit_intercept:
                row['intercept'] = cloned.intercept_ if hasattr(cloned, 'intercept_') else np.nan

            results.append(row)

    return pd.DataFrame(results)


def plot_alpha_trace(
    trace_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show_legend: bool = True
):
    """
    Plot alpha trace showing coefficient paths and loss curves.

    Parameters
    ----------
    trace_df : pd.DataFrame
        Output from compute_alpha_trace()
    feature_names : list of str, optional
        Feature names to plot. If None, plots all coefficient columns.
    figsize : tuple, default=(14, 10)
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    show_legend : bool, default=True
        Whether to show legend on plots.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    # Get coefficient columns
    coef_cols = [c for c in trace_df.columns if c.startswith('coef_')]

    if feature_names is None:
        feature_names = [c.replace('coef_', '') for c in coef_cols]

    # Get unique l1_ratios
    l1_ratios = trace_df['l1_ratio'].unique()
    n_ratios = len(l1_ratios)

    # L1 ratio labels
    ratio_labels = {0.0: 'Ridge (L2)', 0.5: 'ElasticNet', 1.0: 'Lasso (L1)'}

    # Create figure: 2 rows (coefficients, loss) x n_ratios columns
    fig, axes = plt.subplots(2, n_ratios, figsize=figsize, squeeze=False)

    # Color palette for coefficients
    colors = plt.cm.tab10(np.linspace(0, 1, len(coef_cols)))

    for col_idx, l1_ratio in enumerate(l1_ratios):
        subset = trace_df[trace_df['l1_ratio'] == l1_ratio].copy()
        # Sort by alpha ascending for proper line plotting
        subset = subset.sort_values('alpha')
        alphas = subset['alpha'].values

        # Plot coefficients (linear x-axis)
        ax_coef = axes[0, col_idx]
        for i, (coef_col, color) in enumerate(zip(coef_cols, colors)):
            coef_name = coef_col.replace('coef_', '')
            ax_coef.plot(alphas, subset[coef_col].values,
                        label=coef_name, color=color, linewidth=2)

        ax_coef.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_coef.set_xlabel('Alpha')
        ax_coef.set_ylabel('Coefficient Value')
        label = ratio_labels.get(l1_ratio, f'L1 Ratio={l1_ratio}')
        ax_coef.set_title(f'{label}\nCoefficient Paths')
        ax_coef.grid(True, alpha=0.3)

        if show_legend and col_idx == n_ratios - 1:
            ax_coef.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

        # Plot loss (linear x-axis)
        ax_loss = axes[1, col_idx]
        ax_loss.plot(alphas, subset['loss_value'].values,
                    color='#e74c3c', linewidth=2)
        ax_loss.set_xlabel('Alpha')
        ax_loss.set_ylabel('Loss Value')
        ax_loss.set_title(f'{label}\nLoss Curve')
        ax_loss.grid(True, alpha=0.3)

        # Mark minimum loss
        min_idx = subset['loss_value'].idxmin()
        if pd.notna(min_idx):
            min_alpha = subset.loc[min_idx, 'alpha']
            min_loss = subset.loc[min_idx, 'loss_value']
            ax_loss.axvline(x=min_alpha, color='green', linestyle='--', alpha=0.7)
            ax_loss.scatter([min_alpha], [min_loss], color='green', s=100, zorder=5,
                          label=f'Min at α={min_alpha:.4f}')
            ax_loss.legend(loc='upper right', fontsize=9)

    plt.suptitle('Alpha Trace Analysis: Regularization Path', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def find_optimal_alpha(
    trace_df: pd.DataFrame,
    criterion: str = 'min_loss',
    l1_ratio: Optional[float] = None
) -> dict:
    """
    Find optimal alpha based on specified criterion.

    Parameters
    ----------
    trace_df : pd.DataFrame
        Output from compute_alpha_trace()
    criterion : str, default='min_loss'
        Selection criterion:
        - 'min_loss': Minimum loss value
        - 'one_se': One standard error rule (requires CV, not implemented yet)
    l1_ratio : float, optional
        Specific l1_ratio to search. If None, searches all.

    Returns
    -------
    dict
        Dictionary containing:
        - alpha: Optimal alpha value
        - l1_ratio: Corresponding l1_ratio
        - loss_value: Loss at optimal
        - coefficients: dict of coefficient values
    """
    if l1_ratio is not None:
        subset = trace_df[trace_df['l1_ratio'] == l1_ratio]
    else:
        subset = trace_df

    if criterion == 'min_loss':
        # Filter converged solutions
        converged = subset[subset['converged'] == True]
        if len(converged) == 0:
            warnings.warn("No converged solutions found. Using all results.")
            converged = subset

        idx = converged['loss_value'].idxmin()
        best_row = converged.loc[idx]
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Extract coefficients
    coef_cols = [c for c in trace_df.columns if c.startswith('coef_')]
    coefficients = {c.replace('coef_', ''): best_row[c] for c in coef_cols}

    result = {
        'alpha': best_row['alpha'],
        'l1_ratio': best_row['l1_ratio'],
        'loss_value': best_row['loss_value'],
        'converged': best_row['converged'],
        'coefficients': coefficients,
    }

    if 'intercept' in best_row:
        result['intercept'] = best_row['intercept']

    return result


def summarize_alpha_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize alpha trace results by l1_ratio.

    Parameters
    ----------
    trace_df : pd.DataFrame
        Output from compute_alpha_trace()

    Returns
    -------
    pd.DataFrame
        Summary with best alpha for each l1_ratio.
    """
    summaries = []

    for l1_ratio in trace_df['l1_ratio'].unique():
        subset = trace_df[trace_df['l1_ratio'] == l1_ratio]
        converged = subset[subset['converged'] == True]

        if len(converged) > 0:
            best_idx = converged['loss_value'].idxmin()
            best_row = converged.loc[best_idx]

            # Get coefficient columns
            coef_cols = [c for c in subset.columns if c.startswith('coef_')]

            # Check how many coefficients are effectively zero at best alpha
            n_zero = sum(1 for c in coef_cols if abs(best_row[c]) < 1e-8)

            summary = {
                'l1_ratio': l1_ratio,
                'best_alpha': best_row['alpha'],
                'min_loss': best_row['loss_value'],
                'n_converged': len(converged),
                'n_total': len(subset),
                'n_zero_coefs': n_zero,
            }
        else:
            summary = {
                'l1_ratio': l1_ratio,
                'best_alpha': np.nan,
                'min_loss': np.nan,
                'n_converged': 0,
                'n_total': len(subset),
                'n_zero_coefs': np.nan,
            }

        summaries.append(summary)

    return pd.DataFrame(summaries)
