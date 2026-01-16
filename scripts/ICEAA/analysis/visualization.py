"""
Visualization functions for ICEAA simulation analysis.

These functions are designed for use in Quarto documents and return
matplotlib figures that can be displayed inline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from typing import Optional, List, Tuple


# Default styling
METRIC = "test_sspe"
METRIC_LABEL = "Test SSPE (Sum of Squared Percentage Errors)"

CATEGORY_LABELS = {
    "cv_error": "CV Error",
    "target_correlation": "Correlation",
    "learning_rate": "Learning Rate",
    "rate_effect": "Rate Effect",
    "n_lots": "Sample Size (n_lots)",
}


def get_model_order(df: pd.DataFrame, metric: str = METRIC) -> List[str]:
    """Get models ordered by mean performance (best first)."""
    return df.groupby("model_name")[metric].mean().sort_values().index.tolist()


def create_overall_boxplot(
    df: pd.DataFrame,
    model_order: Optional[List[str]] = None,
    metric: str = METRIC,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create overall boxplot comparing all models.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe.
    model_order : list, optional
        Order of models on x-axis. If None, ordered by mean performance.
    metric : str
        Metric column to plot.
    figsize : tuple
        Figure size (width, height).

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if model_order is None:
        model_order = get_model_order(df, metric)

    fig, ax = plt.subplots(figsize=figsize)
    data_to_plot = [
        df[df["model_name"] == m][metric].dropna().values for m in model_order
    ]
    bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

    # Color PCReg models differently
    colors = ["lightgreen" if "PCReg" in m else "lightblue" for m in model_order]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Highlight winner
    bp["boxes"][0].set_edgecolor("red")
    bp["boxes"][0].set_linewidth(2)

    ax.set_ylabel(METRIC_LABEL)
    ax.set_xlabel("Model")
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Overall Model Performance (Test SSPE)", fontsize=14, fontweight="bold")

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Penalized-Constrained"),
        Patch(facecolor="lightblue", edgecolor="black", label="Standard Methods"),
        Patch(facecolor="white", edgecolor="red", linewidth=2, label="Winner"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    return fig


def create_boxplot_by_factor(
    df: pd.DataFrame,
    factor: str,
    model_order: Optional[List[str]] = None,
    metric: str = METRIC,
    figsize_per_level: Tuple[int, int] = (5, 6),
) -> plt.Figure:
    """
    Create boxplot panels for each level of a design factor.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe.
    factor : str
        Column name of the factor to split by.
    model_order : list, optional
        Order of models. If None, ordered by mean performance.
    metric : str
        Metric column to plot.
    figsize_per_level : tuple
        Figure size per factor level.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if model_order is None:
        model_order = get_model_order(df, metric)

    factor_values = sorted(df[factor].unique())
    n_levels = len(factor_values)

    fig, axes = plt.subplots(
        1, n_levels, figsize=(figsize_per_level[0] * n_levels, figsize_per_level[1]), sharey=True
    )
    if n_levels == 1:
        axes = [axes]

    for ax, val in zip(axes, factor_values):
        subset = df[df[factor] == val]
        data_to_plot = [
            subset[subset["model_name"] == m][metric].dropna().values for m in model_order
        ]
        bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

        colors = ["lightgreen" if "PCReg" in m else "lightblue" for m in model_order]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(f"{CATEGORY_LABELS.get(factor, factor)} = {val}")
        ax.set_ylabel(METRIC_LABEL if ax == axes[0] else "")
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Mark winner for this level
        means = [subset[subset["model_name"] == m][metric].mean() for m in model_order]
        if means:
            winner_idx = np.argmin(means)
            bp["boxes"][winner_idx].set_edgecolor("red")
            bp["boxes"][winner_idx].set_linewidth(2)

    plt.suptitle(
        f"Model Performance by {CATEGORY_LABELS.get(factor, factor)}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def create_win_rate_heatmap(
    df: pd.DataFrame,
    factor1: str,
    factor2: str,
    model: str = "PCReg_CV",
    baseline: str = "OLS",
    metric: str = METRIC,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Create heatmap showing model improvement over baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe.
    factor1 : str
        Row factor for heatmap.
    factor2 : str
        Column factor for heatmap.
    model : str
        Model to compare.
    baseline : str
        Baseline model.
    metric : str
        Metric to compare.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    scenario_cols = [
        "n_lots",
        "target_correlation",
        "cv_error",
        "learning_rate",
        "rate_effect",
        "replication",
    ]
    df_wide = df.pivot_table(
        index=scenario_cols, columns="model_name", values=metric
    ).reset_index()

    if model not in df_wide.columns or baseline not in df_wide.columns:
        # Return empty figure if models not found
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Models {model} or {baseline} not found", ha="center", va="center")
        return fig

    df_wide["improvement"] = (df_wide[baseline] - df_wide[model]) / df_wide[baseline] * 100
    heatmap_data = df_wide.groupby([factor1, factor2])["improvement"].mean().unstack()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": f"% Improvement\n(positive = {model} better)"},
        ax=ax,
    )
    ax.set_xlabel(CATEGORY_LABELS.get(factor2, factor2))
    ax.set_ylabel(CATEGORY_LABELS.get(factor1, factor1))
    ax.set_title(f"{model} Improvement over {baseline}")
    plt.tight_layout()

    return fig


def create_sign_correctness_plot(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Create bar chart showing sign correctness rate by model.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe with b_correct_sign and c_correct_sign columns.
    models : list, optional
        Models to include. If None, uses all models.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if models is None:
        models = df["model_name"].unique().tolist()

    sign_data = []
    for model in models:
        m = df[df["model_name"] == model]
        if "b_correct_sign" in m.columns and "c_correct_sign" in m.columns:
            both_correct = (m["b_correct_sign"] & m["c_correct_sign"]).mean()
            sign_data.append({"model": model, "both_correct": both_correct})

    if not sign_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sign correctness data not available", ha="center", va="center")
        return fig

    sign_df = pd.DataFrame(sign_data).sort_values("both_correct", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["lightgreen" if "PCReg" in m else "lightblue" for m in sign_df["model"]]
    bars = ax.bar(sign_df["model"], sign_df["both_correct"] * 100, color=colors, edgecolor="black")

    ax.set_ylabel("Both Signs Correct (%)")
    ax.set_xlabel("Model")
    ax.set_title("Sign Correctness by Model", fontsize=14, fontweight="bold")
    ax.set_xticklabels(sign_df["model"], rotation=45, ha="right")
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Perfect")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def create_wrong_sign_heatmap(
    df: pd.DataFrame,
    factor1: str = "n_lots",
    factor2: str = "target_correlation",
    model: str = "OLS",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Create heatmap showing wrong sign rate by two factors.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with sign correctness columns.
    factor1 : str
        Row factor.
    factor2 : str
        Column factor.
    model : str
        Model to analyze.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    m = df[df["model_name"] == model].copy()

    if "b_correct_sign" not in m.columns or "c_correct_sign" not in m.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sign correctness data not available", ha="center", va="center")
        return fig

    m["any_wrong_sign"] = ~(m["b_correct_sign"] & m["c_correct_sign"])
    pivot = m.pivot_table(values="any_wrong_sign", index=factor1, columns=factor2, aggfunc="mean")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar_kws={"label": "Wrong Sign Rate (%)"},
        ax=ax,
    )
    ax.set_xlabel(CATEGORY_LABELS.get(factor2, factor2))
    ax.set_ylabel(CATEGORY_LABELS.get(factor1, factor1))
    ax.set_title(f"{model} Wrong Sign Rate")
    plt.tight_layout()

    return fig


def create_model_ranking_chart(
    df: pd.DataFrame,
    pcreg_model: str = "PCReg_CV",
    metric: str = METRIC,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Create column chart showing average rank for OLS, OLS_LearnOnly, and one PCReg model.

    Since PCReg models are highly correlated, we compare only three models at a time:
    OLS (baseline), OLS_LearnOnly (confluence remedy), and one PCReg variant.

    Computes ranks within each scenario-replication across these 3 models only,
    handling ties with midrank method (average rank for tied values).

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe.
    pcreg_model : str, default "PCReg_CV"
        Which PCReg model to compare (e.g., "PCReg_CV", "PCReg_ConstrainOnly", "PCReg_CV_Wrong").
    metric : str
        Metric to rank by (lower is better).
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Fixed comparison: OLS, OLS_LearnOnly, and specified PCReg model
    models = ["OLS", "OLS_LearnOnly", pcreg_model]

    # Filter to these three models
    df_subset = df[df["model_name"].isin(models)].copy()

    # Define scenario columns
    scenario_cols = [
        "n_lots",
        "target_correlation",
        "cv_error",
        "learning_rate",
        "rate_effect",
        "replication",
    ]

    # Compute rank within each scenario across these 3 models only (ties get average rank)
    df_subset["rank"] = df_subset.groupby(scenario_cols)[metric].rank(method="average")

    # Compute average rank for each model
    avg_ranks = df_subset.groupby("model_name")["rank"].mean().reindex(models)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color by model type
    colors = ["#3498db", "#3498db", "#2ecc71"]  # OLS, OLS_LearnOnly, PCReg

    bars = ax.bar(range(len(avg_ranks)), avg_ranks.values, color=colors, edgecolor="black", linewidth=1.5)

    # Highlight the winner with thicker border
    best_idx = avg_ranks.values.argmin()
    bars[best_idx].set_edgecolor("darkgreen")
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(range(len(avg_ranks)))
    ax.set_xticklabels(models, rotation=0, ha="center")
    ax.set_ylabel("Average Rank (Lower is Better)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Model Ranking: OLS vs Confluence Remedy vs PCReg", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(1, 3)

    # Add value labels on bars
    for bar, rank in zip(bars, avg_ranks.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{rank:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add legend
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Standard Methods"),
        Patch(facecolor="#2ecc71", edgecolor="black", label="Penalized-Constrained"),
        Patch(facecolor="white", edgecolor="darkgreen", linewidth=3, label="Best Performer"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Add subtitle showing what the ranks mean
    subtitle = f"Ranks computed across 6,075 scenarios (lower is better, range: 1-3)"
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    return fig
