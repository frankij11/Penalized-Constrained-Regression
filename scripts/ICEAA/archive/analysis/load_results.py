"""
Data loading utilities for ICEAA simulation analysis.

This module provides functions to load simulation results, predictions,
and configuration data from the output_v2 directory.

Designed for integration with Quarto documents using freeze/cache.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Optional


# Resolve paths relative to this file's location
RESULTS_DIR = Path(__file__).parent.parent / "output_v2"


def load_simulation_results(
    results_dir: Optional[Path] = None,
    converged_only: bool = True,
) -> pd.DataFrame:
    """
    Load main simulation results from parquet file.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing simulation outputs. Defaults to output_v2/.
    converged_only : bool, default True
        If True, filter to only converged model fits.

    Returns
    -------
    pd.DataFrame
        Simulation results with columns:
        - model_name: Name of the regression model
        - n_lots, target_correlation, cv_error, learning_rate, rate_effect: Design factors
        - replication: Replication number (1-25)
        - b, c, T1: Estimated coefficients
        - test_sspe, test_mape, test_mse: Out-of-sample metrics
        - converged: Whether optimization converged
        - b_correct_sign, c_correct_sign: Sign correctness indicators
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    filepath = results_dir / "simulation_results.parquet"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Simulation results not found at {filepath}. "
            "Run the simulation first: python scripts/ICEAA/run_simulation.py"
        )

    df = pd.read_parquet(filepath)

    if converged_only and "converged" in df.columns:
        df = df[df["converged"] == True].copy()

    return df


def load_predictions(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load lot-level predictions from parquet file.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing simulation outputs. Defaults to output_v2/.

    Returns
    -------
    pd.DataFrame
        Lot-level predictions with actual vs. predicted values.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    filepath = results_dir / "predictions_flat.parquet"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Predictions not found at {filepath}. "
            "Run the simulation first: python scripts/ICEAA/run_simulation.py"
        )

    return pd.read_parquet(filepath)


def load_config(results_dir: Optional[Path] = None) -> dict:
    """
    Load simulation configuration from JSON file.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing simulation outputs. Defaults to output_v2/.

    Returns
    -------
    dict
        Configuration dictionary with simulation parameters.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    filepath = results_dir / "simulation_config.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration not found at {filepath}.")

    with open(filepath) as f:
        return json.load(f)


def get_model_comparison(
    df: pd.DataFrame,
    model_a: str = "OLS",
    model_b: str = "PCReg_ConstrainOnly",
    metric: str = "test_sspe",
) -> pd.DataFrame:
    """
    Create a head-to-head comparison of two models across scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results dataframe.
    model_a : str
        First model name (baseline).
    model_b : str
        Second model name (comparison).
    metric : str
        Metric to compare (e.g., 'test_sspe', 'test_mape').

    Returns
    -------
    pd.DataFrame
        Merged dataframe with columns for both models' metrics
        and a 'b_wins' indicator.
    """
    # Create scenario key
    key_cols = [
        "n_lots",
        "target_correlation",
        "cv_error",
        "learning_rate",
        "rate_effect",
        "replication",
    ]

    a = df[df["model_name"] == model_a].set_index(key_cols)[[metric]].rename(
        columns={metric: f"{model_a}_{metric}"}
    )
    b = df[df["model_name"] == model_b].set_index(key_cols)[[metric]].rename(
        columns={metric: f"{model_b}_{metric}"}
    )

    # Also grab sign correctness from model_a
    if model_a == "OLS":
        sign_cols = ["b_correct_sign", "c_correct_sign"]
        signs = df[df["model_name"] == model_a].set_index(key_cols)[sign_cols]
        merged = a.join(b).join(signs)
        merged["any_wrong_sign"] = ~(
            merged["b_correct_sign"] & merged["c_correct_sign"]
        )
    else:
        merged = a.join(b)

    merged["b_wins"] = merged[f"{model_b}_{metric}"] < merged[f"{model_a}_{metric}"]

    return merged.reset_index()
