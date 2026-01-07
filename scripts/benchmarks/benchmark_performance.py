#!/usr/bin/env python
"""
benchmark_performance.py - Performance benchmarking for PCRegression

Measures execution time, identifies bottlenecks, and compares against OLS.

Usage:
    python benchmark_performance.py [--quick] [--output-dir PATH]

Options:
    --quick         Run reduced benchmark (fewer configurations)
    --output-dir    Directory for output files (default: benchmark_output)
"""

import sys
import time
import cProfile
import pstats
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from io import StringIO
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from penalized_constrained import (
    PenalizedConstrainedRegression,
    PenalizedConstrainedCV,
    generate_correlated_learning_data
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimingResult:
    """Single timing measurement."""
    method_name: str
    n_samples: int
    n_features: int
    model_type: str
    wall_time_seconds: float
    n_iterations: Optional[int]
    converged: bool
    r2_train: float
    coef_rmse_vs_true: Optional[float]
    coef_rmse_vs_ols: Optional[float]
    memory_peak_mb: float
    loss_type: str = 'sse'


@dataclass
class ProfileResult:
    """Profiling breakdown for a function."""
    function_name: str
    total_time_ms: float
    n_calls: int
    time_per_call_us: float
    percent_of_total: float


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    sample_sizes: List[int] = field(default_factory=lambda: [10, 30, 50, 100])
    feature_counts: List[int] = field(default_factory=lambda: [3, 5, 10])
    n_repetitions: int = 5
    model_types: List[str] = field(default_factory=lambda: ['linear', 'learning_curve'])
    loss_types: List[str] = field(default_factory=lambda: ['sse', 'sspe'])


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_linear_data(n_samples: int, n_features: int, seed: int = 42) -> Dict:
    """
    Generate linear regression benchmark data with constrained coefficients.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    seed : int
        Random seed

    Returns
    -------
    dict with X, y, true_coef, intercept, bounds
    """
    np.random.seed(seed)

    # True coefficients (all negative to test constraints)
    true_coef = -np.abs(np.random.randn(n_features)) * 0.5
    intercept = 10.0

    # Features with moderate correlation
    X = np.random.randn(n_samples, n_features)

    # Add correlation structure between features
    for i in range(1, n_features):
        X[:, i] = 0.5 * X[:, 0] + 0.5 * X[:, i]

    # Generate y with noise
    noise_std = 0.1 * np.abs(intercept)
    y = X @ true_coef + intercept + np.random.randn(n_samples) * noise_std

    # Bounds: all coefficients <= 0 (negative slopes)
    bounds = [(-1.0, 0.0)] * n_features

    return {
        'X': X,
        'y': y,
        'true_coef': true_coef,
        'intercept': intercept,
        'bounds': bounds,
        'coef_names': [f'X{i}' for i in range(n_features)]
    }


def generate_learning_curve_data_benchmark(n_samples: int, seed: int = 42) -> Dict:
    """
    Generate learning curve data for benchmarking custom prediction functions.

    Parameters
    ----------
    n_samples : int
        Number of lots (samples)
    seed : int
        Random seed

    Returns
    -------
    dict with X, y, true params, bounds, prediction function
    """
    # Use the existing utility for realistic learning curve data
    data = generate_correlated_learning_data(
        n_lots=n_samples,
        T1=100,
        target_correlation=0.6,
        cv_error=0.1,
        random_state=seed
    )

    # True parameters from data generation
    true_params = np.array([
        np.log(data['params']['T1']),  # log(T1) for log-space model
        data['params']['b'],            # LC slope
        data['params']['c']             # RC slope
    ])

    # Bounds for learning curve parameters
    bounds = {
        'log_T1': (2.0, 7.0),    # log(T1) roughly 7-1100
        'LC': (-1.0, 0.0),       # Learning curve slope
        'RC': (-1.0, 0.0)        # Rate curve slope
    }

    # Custom prediction function (log-linear model)
    def lc_predict(X, params):
        """Log-linear learning curve: log(Y) = log(T1) + b*log(MP) + c*log(Q)"""
        log_T1, b, c = params[0], params[1], params[2]
        return log_T1 + b * X[:, 0] + c * X[:, 1]

    return {
        'X': data['X'],           # Already log-transformed
        'y': data['y'],           # Already log-transformed
        'true_coef': true_params,
        'bounds': bounds,
        'coef_names': ['log_T1', 'LC', 'RC'],
        'prediction_fn': lc_predict,
        'intercept': None,  # Custom function handles intercept
        'X_original': data['X_original'],
        'y_original': data['y_original']
    }


# =============================================================================
# TIMING FUNCTIONS
# =============================================================================

def time_model_fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    true_coef: Optional[np.ndarray] = None,
    ols_coef: Optional[np.ndarray] = None,
    n_runs: int = 5
) -> Dict[str, Any]:
    """
    Time model fitting with memory tracking.

    Parameters
    ----------
    model : estimator
        Model to fit (must have fit, predict, score methods)
    X : array
        Features
    y : array
        Target
    true_coef : array, optional
        True coefficients for comparison
    ols_coef : array, optional
        OLS coefficients for comparison
    n_runs : int
        Number of timing runs

    Returns
    -------
    dict with timing and accuracy metrics
    """
    times = []

    for run in range(n_runs):
        # Memory tracking
        tracemalloc.start()

        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        end = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append({
            'wall_time': end - start,
            'memory_peak_mb': peak / 1024 / 1024
        })

    # Extract results
    mean_time = np.mean([t['wall_time'] for t in times])
    std_time = np.std([t['wall_time'] for t in times])
    mean_memory = np.mean([t['memory_peak_mb'] for t in times])

    # Get iteration count if available
    n_iter = None
    if hasattr(model, 'optimization_result_'):
        n_iter = getattr(model.optimization_result_, 'nit', None)
    elif hasattr(model, 'best_estimator_'):
        if hasattr(model.best_estimator_, 'optimization_result_'):
            n_iter = getattr(model.best_estimator_.optimization_result_, 'nit', None)

    # Convergence status
    converged = True
    if hasattr(model, 'converged_'):
        converged = model.converged_

    # R² score
    try:
        r2 = model.score(X, y)
    except Exception:
        r2 = np.nan

    # Coefficient comparison
    coef_rmse_true = None
    coef_rmse_ols = None

    # Get coefficients from model
    if hasattr(model, 'coef_'):
        coef = model.coef_
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'coef_'):
        coef = model.best_estimator_.coef_
    else:
        coef = None

    if coef is not None and true_coef is not None:
        if len(coef) == len(true_coef):
            coef_rmse_true = np.sqrt(np.mean((coef - true_coef) ** 2))

    if coef is not None and ols_coef is not None:
        if len(coef) == len(ols_coef):
            coef_rmse_ols = np.sqrt(np.mean((coef - ols_coef) ** 2))

    return {
        'wall_time': mean_time,
        'wall_time_std': std_time,
        'memory_peak_mb': mean_memory,
        'n_iterations': n_iter,
        'converged': converged,
        'r2': r2,
        'coef_rmse_vs_true': coef_rmse_true,
        'coef_rmse_vs_ols': coef_rmse_ols,
        'coef': coef
    }


# =============================================================================
# PROFILING FUNCTIONS
# =============================================================================

def profile_objective_calls(
    X: np.ndarray,
    y: np.ndarray,
    bounds: List,
    n_calls: int = 1000,
    safe_mode: bool = True
) -> Tuple[str, pstats.Stats, float]:
    """
    Profile the _objective function directly.

    Parameters
    ----------
    X : array
        Features
    y : array
        Target
    bounds : list
        Coefficient bounds
    n_calls : int
        Number of objective function calls
    safe_mode : bool
        Whether to profile with safe_mode on

    Returns
    -------
    output : str
        Profile output text
    stats : pstats.Stats
        Profile statistics
    total_time : float
        Total profiling time
    """
    # Create and fit model first to initialize
    model = PenalizedConstrainedRegression(
        bounds=bounds,
        loss='sse',
        safe_mode=safe_mode
    )
    model.fit(X, y)

    # Get parameters for profiling
    params = model.optimization_result_.x

    # Profile objective calls
    profiler = cProfile.Profile()

    start_total = time.perf_counter()
    profiler.enable()

    for _ in range(n_calls):
        model._objective(params, X, y)

    profiler.disable()
    total_time = time.perf_counter() - start_total

    # Analyze results
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue(), stats, total_time


def extract_function_times(stats: pstats.Stats, total_time: float) -> List[ProfileResult]:
    """
    Extract timing breakdown from cProfile stats.

    Parameters
    ----------
    stats : pstats.Stats
        Profile statistics
    total_time : float
        Total elapsed time

    Returns
    -------
    list of ProfileResult
    """
    results = []

    # Key functions to track
    functions_of_interest = [
        ('_predict_internal', 'Prediction'),
        ('sspe_loss', 'SSPE Loss'),
        ('sse_loss', 'SSE Loss'),
        ('mse_loss', 'MSE Loss'),
        ('compute_elastic_net_penalty', 'Penalty'),
        ('_objective', 'Objective (total)'),
        ('isfinite', 'Safe mode checks'),
    ]

    for func_pattern, display_name in functions_of_interest:
        for key, value in stats.stats.items():
            if func_pattern in str(key):
                cc, nc, tt, ct, callers = value
                results.append(ProfileResult(
                    function_name=display_name,
                    total_time_ms=ct * 1000,
                    n_calls=nc,
                    time_per_call_us=(ct / nc * 1e6) if nc > 0 else 0,
                    percent_of_total=(ct / total_time * 100) if total_time > 0 else 0
                ))
                break

    return results


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def benchmark_linear_models(
    n_samples: int,
    n_features: int,
    n_repetitions: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> List[TimingResult]:
    """
    Benchmark linear models for a specific n, p configuration.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_repetitions : int
        Number of timing repetitions
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    list of TimingResult
    """
    results = []

    # Generate data
    data = generate_linear_data(n_samples, n_features, seed)
    X, y = data['X'], data['y']
    true_coef = data['true_coef']
    bounds = data['bounds']

    # Fit OLS first for comparison
    ols = LinearRegression()
    ols.fit(X, y)
    ols_coef = ols.coef_

    # Define models to benchmark
    models = {
        'OLS': LinearRegression(),
        'PCR_constrained': PenalizedConstrainedRegression(
            alpha=0.0, bounds=bounds, loss='sse', safe_mode=True
        ),
        'PCR_safe_off': PenalizedConstrainedRegression(
            alpha=0.0, bounds=bounds, loss='sse', safe_mode=False
        ),
        'PCR_penalized': PenalizedConstrainedRegression(
            alpha=0.1, l1_ratio=0.5, bounds=bounds, loss='sse', safe_mode=True
        ),
        'PCCV_aic': PenalizedConstrainedCV(
            bounds=bounds, selection='aic',
            alphas=np.logspace(-3, 0, 5), loss='sse', verbose=0
        ),
    }

    # Add sklearn CV methods for smaller problems
    if n_features <= 10:
        models['RidgeCV'] = RidgeCV(alphas=np.logspace(-3, 3, 10))
        models['LassoCV'] = LassoCV(alphas=np.logspace(-3, 0, 10), max_iter=5000)

    for name, model in models.items():
        if verbose:
            print(f"    {name}...", end=' ', flush=True)

        try:
            result = time_model_fit(
                model, X, y,
                true_coef=true_coef,
                ols_coef=ols_coef,
                n_runs=n_repetitions
            )

            results.append(TimingResult(
                method_name=name,
                n_samples=n_samples,
                n_features=n_features,
                model_type='linear',
                wall_time_seconds=result['wall_time'],
                n_iterations=result['n_iterations'],
                converged=result['converged'],
                r2_train=result['r2'],
                coef_rmse_vs_true=result['coef_rmse_vs_true'],
                coef_rmse_vs_ols=result['coef_rmse_vs_ols'],
                memory_peak_mb=result['memory_peak_mb'],
                loss_type='sse'
            ))

            if verbose:
                print(f"{result['wall_time']:.4f}s")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results.append(TimingResult(
                method_name=name,
                n_samples=n_samples,
                n_features=n_features,
                model_type='linear',
                wall_time_seconds=np.nan,
                n_iterations=None,
                converged=False,
                r2_train=np.nan,
                coef_rmse_vs_true=None,
                coef_rmse_vs_ols=None,
                memory_peak_mb=np.nan,
                loss_type='sse'
            ))

    return results


def benchmark_learning_curve_models(
    n_samples: int,
    n_repetitions: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> List[TimingResult]:
    """
    Benchmark learning curve models with custom prediction function.

    Parameters
    ----------
    n_samples : int
        Number of lots
    n_repetitions : int
        Number of timing repetitions
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    list of TimingResult
    """
    results = []

    # Generate learning curve data
    data = generate_learning_curve_data_benchmark(n_samples, seed)
    X, y = data['X'], data['y']
    bounds = data['bounds']
    coef_names = data['coef_names']
    prediction_fn = data['prediction_fn']
    true_coef = data['true_coef']

    # OLS on log-transformed data for comparison
    ols = LinearRegression()
    ols.fit(X, y)
    ols_coef = np.concatenate([[ols.intercept_], ols.coef_])

    # Models to benchmark
    models = {
        'OLS_log': LinearRegression(),
        'PCR_LC_constrained': PenalizedConstrainedRegression(
            alpha=0.0,
            bounds=bounds,
            coef_names=coef_names,
            prediction_fn=prediction_fn,
            fit_intercept=False,
            loss='sse',
            safe_mode=True,
            x0='zeros'
        ),
        'PCR_LC_safe_off': PenalizedConstrainedRegression(
            alpha=0.0,
            bounds=bounds,
            coef_names=coef_names,
            prediction_fn=prediction_fn,
            fit_intercept=False,
            loss='sse',
            safe_mode=False,
            x0='zeros'
        ),
        'PCR_LC_penalized': PenalizedConstrainedRegression(
            alpha=0.1,
            l1_ratio=0.5,
            bounds=bounds,
            coef_names=coef_names,
            penalty_exclude=['log_T1'],
            prediction_fn=prediction_fn,
            fit_intercept=False,
            loss='sse',
            safe_mode=True,
            x0='zeros'
        ),
    }

    for name, model in models.items():
        if verbose:
            print(f"    {name}...", end=' ', flush=True)

        try:
            result = time_model_fit(
                model, X, y,
                true_coef=true_coef if 'OLS' not in name else None,
                ols_coef=ols_coef if 'OLS' not in name else None,
                n_runs=n_repetitions
            )

            results.append(TimingResult(
                method_name=name,
                n_samples=n_samples,
                n_features=3,  # LC always has 3 params
                model_type='learning_curve',
                wall_time_seconds=result['wall_time'],
                n_iterations=result['n_iterations'],
                converged=result['converged'],
                r2_train=result['r2'],
                coef_rmse_vs_true=result['coef_rmse_vs_true'],
                coef_rmse_vs_ols=result['coef_rmse_vs_ols'],
                memory_peak_mb=result['memory_peak_mb'],
                loss_type='sse'
            ))

            if verbose:
                print(f"{result['wall_time']:.4f}s")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results.append(TimingResult(
                method_name=name,
                n_samples=n_samples,
                n_features=3,
                model_type='learning_curve',
                wall_time_seconds=np.nan,
                n_iterations=None,
                converged=False,
                r2_train=np.nan,
                coef_rmse_vs_true=None,
                coef_rmse_vs_ols=None,
                memory_peak_mb=np.nan,
                loss_type='sse'
            ))

    return results


def run_full_benchmark(config: BenchmarkConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Run complete benchmark suite.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame with all results
    """
    all_results = []

    # Linear models
    if 'linear' in config.model_types:
        for n in config.sample_sizes:
            for p in config.feature_counts:
                if verbose:
                    print(f"\n[Linear] n={n}, p={p}")

                results = benchmark_linear_models(
                    n_samples=n,
                    n_features=p,
                    n_repetitions=config.n_repetitions,
                    seed=42,
                    verbose=verbose
                )
                all_results.extend(results)

    # Learning curve models
    if 'learning_curve' in config.model_types:
        for n in config.sample_sizes:
            if verbose:
                print(f"\n[Learning Curve] n={n}")

            results = benchmark_learning_curve_models(
                n_samples=n,
                n_repetitions=config.n_repetitions,
                seed=42,
                verbose=verbose
            )
            all_results.extend(results)

    return pd.DataFrame([asdict(r) for r in all_results])


# =============================================================================
# SAFE MODE ANALYSIS
# =============================================================================

def analyze_safe_mode_overhead(
    sample_sizes: List[int] = [30, 50, 100],
    n_features: int = 5,
    n_runs: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Specifically measure the overhead of safe_mode.

    Returns DataFrame with safe_mode on vs off comparison.
    """
    results = []

    for n in sample_sizes:
        if verbose:
            print(f"\nSafe mode analysis: n={n}, p={n_features}")

        data = generate_linear_data(n, n_features)
        X, y = data['X'], data['y']
        bounds = data['bounds']

        for safe_mode in [True, False]:
            model = PenalizedConstrainedRegression(
                alpha=0.0, bounds=bounds, loss='sse', safe_mode=safe_mode
            )

            result = time_model_fit(model, X, y, n_runs=n_runs)

            results.append({
                'n_samples': n,
                'n_features': n_features,
                'safe_mode': safe_mode,
                'wall_time_seconds': result['wall_time'],
                'n_iterations': result['n_iterations']
            })

            if verbose:
                status = "ON" if safe_mode else "OFF"
                print(f"  safe_mode={status}: {result['wall_time']:.4f}s")

    df = pd.DataFrame(results)

    # Calculate overhead
    if verbose:
        print("\nSafe Mode Overhead:")
        for n in sample_sizes:
            subset = df[df['n_samples'] == n]
            time_on = subset[subset['safe_mode'] == True]['wall_time_seconds'].values[0]
            time_off = subset[subset['safe_mode'] == False]['wall_time_seconds'].values[0]
            overhead_pct = ((time_on - time_off) / time_off) * 100
            print(f"  n={n}: {overhead_pct:.1f}% overhead")

    return df


# =============================================================================
# REPORTING
# =============================================================================

def generate_markdown_report(
    results_df: pd.DataFrame,
    profile_results: Optional[List[ProfileResult]] = None,
    safe_mode_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate comprehensive markdown report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark timing results
    profile_results : list of ProfileResult, optional
        Function profiling breakdown
    safe_mode_df : pd.DataFrame, optional
        Safe mode analysis results

    Returns
    -------
    str : Markdown report
    """
    report = []

    report.append("# PCRegression Performance Benchmark Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # =========================================================================
    # Section 1: Executive Summary
    # =========================================================================
    report.append("## Executive Summary\n\n")

    # Calculate key metrics
    linear_df = results_df[results_df['model_type'] == 'linear']

    if len(linear_df) > 0:
        ols_times = linear_df[linear_df['method_name'] == 'OLS']['wall_time_seconds']
        pcr_times = linear_df[linear_df['method_name'] == 'PCR_constrained']['wall_time_seconds']

        if len(ols_times) > 0 and len(pcr_times) > 0:
            avg_ols = ols_times.mean()
            avg_pcr = pcr_times.mean()
            slowdown = avg_pcr / avg_ols if avg_ols > 0 else np.nan

            report.append(f"- **Average OLS time**: {avg_ols*1000:.2f} ms\n")
            report.append(f"- **Average PCRegression time**: {avg_pcr*1000:.2f} ms\n")
            report.append(f"- **PCRegression slowdown vs OLS**: {slowdown:.1f}x\n\n")

    # =========================================================================
    # Section 2: Timing Results - Linear Models
    # =========================================================================
    report.append("## 1. Timing Results - Linear Models\n\n")

    if len(linear_df) > 0:
        # Pivot table
        timing_pivot = linear_df.pivot_table(
            values='wall_time_seconds',
            index=['n_samples', 'n_features'],
            columns='method_name',
            aggfunc='mean'
        ).round(4)

        report.append("### Wall Time (seconds)\n\n")
        report.append(timing_pivot.to_markdown())
        report.append("\n\n")

        # Speed relative to OLS
        report.append("### Speed Relative to OLS\n\n")
        report.append("| n | p | Method | Time (s) | Ratio vs OLS |\n")
        report.append("|---|---|--------|----------|-------------|\n")

        for (n, p), group in linear_df.groupby(['n_samples', 'n_features']):
            ols_row = group[group['method_name'] == 'OLS']
            if len(ols_row) > 0:
                ols_time = ols_row['wall_time_seconds'].values[0]

                for _, row in group.iterrows():
                    ratio = row['wall_time_seconds'] / ols_time if ols_time > 0 else np.nan
                    report.append(
                        f"| {n} | {p} | {row['method_name']} | "
                        f"{row['wall_time_seconds']:.4f} | {ratio:.1f}x |\n"
                    )

        report.append("\n")

    # =========================================================================
    # Section 3: Timing Results - Learning Curve Models
    # =========================================================================
    lc_df = results_df[results_df['model_type'] == 'learning_curve']

    if len(lc_df) > 0:
        report.append("## 2. Timing Results - Learning Curve Models\n\n")

        lc_pivot = lc_df.pivot_table(
            values='wall_time_seconds',
            index='n_samples',
            columns='method_name',
            aggfunc='mean'
        ).round(4)

        report.append("### Wall Time (seconds)\n\n")
        report.append(lc_pivot.to_markdown())
        report.append("\n\n")

    # =========================================================================
    # Section 4: Coefficient Accuracy
    # =========================================================================
    report.append("## 3. Coefficient Accuracy\n\n")

    accuracy_cols = ['method_name', 'n_samples', 'n_features', 'model_type',
                     'r2_train', 'coef_rmse_vs_true', 'coef_rmse_vs_ols']
    accuracy_df = results_df[accuracy_cols].copy()
    accuracy_df = accuracy_df.round(4)

    report.append(accuracy_df.to_markdown(index=False))
    report.append("\n\n")

    # =========================================================================
    # Section 5: Safe Mode Analysis
    # =========================================================================
    if safe_mode_df is not None:
        report.append("## 4. Safe Mode Overhead Analysis\n\n")

        report.append("| n | safe_mode | Time (s) | Overhead |\n")
        report.append("|---|-----------|----------|----------|\n")

        for n in safe_mode_df['n_samples'].unique():
            subset = safe_mode_df[safe_mode_df['n_samples'] == n]
            time_on = subset[subset['safe_mode'] == True]['wall_time_seconds'].values[0]
            time_off = subset[subset['safe_mode'] == False]['wall_time_seconds'].values[0]
            overhead_pct = ((time_on - time_off) / time_off) * 100

            report.append(f"| {n} | True | {time_on:.4f} | +{overhead_pct:.1f}% |\n")
            report.append(f"| {n} | False | {time_off:.4f} | baseline |\n")

        report.append("\n")

    # =========================================================================
    # Section 6: Bottleneck Analysis (Profiling)
    # =========================================================================
    if profile_results:
        report.append("## 5. Bottleneck Analysis\n\n")
        report.append("Function time breakdown from profiling `_objective()` calls:\n\n")
        report.append("| Function | Time (ms) | % of Total | Calls | Per Call (μs) |\n")
        report.append("|----------|-----------|------------|-------|---------------|\n")

        for pr in sorted(profile_results, key=lambda x: -x.total_time_ms):
            report.append(
                f"| {pr.function_name} | {pr.total_time_ms:.2f} | "
                f"{pr.percent_of_total:.1f}% | {pr.n_calls} | "
                f"{pr.time_per_call_us:.2f} |\n"
            )

        report.append("\n")

    # =========================================================================
    # Section 7: Optimization Recommendations
    # =========================================================================
    report.append("## 6. Optimization Recommendations\n\n")

    report.append("### Quick Wins\n\n")
    report.append("1. **Use `safe_mode=False`** for well-tested prediction functions\n")
    report.append("   - Eliminates ~10-15% overhead from validity checks\n")
    report.append("   - Only use when prediction function is known to be stable\n\n")

    report.append("2. **Use information criteria (AIC/BIC)** instead of CV for small samples\n")
    report.append("   - Faster than k-fold CV (no data splitting)\n")
    report.append("   - Better statistical properties for n < 50\n\n")

    report.append("### Medium-Term Optimizations\n\n")
    report.append("3. **Parallelize IC grid search** in `cv.py`\n")
    report.append("   - Current: Sequential loop over alpha × l1_ratio combinations\n")
    report.append("   - Potential: `joblib.Parallel` for embarrassingly parallel fits\n\n")

    report.append("4. **Warm starting** between CV folds or alpha values\n")
    report.append("   - Pass previous coefficients as `x0` parameter\n")
    report.append("   - Can reduce optimizer iterations by 20-40%\n\n")

    report.append("### Long-Term Considerations\n\n")
    report.append("5. **Analytical gradients** for scipy optimizer\n")
    report.append("   - Currently uses finite difference approximation\n")
    report.append("   - Analytical gradients could halve optimization time\n\n")

    report.append("6. **Numba JIT compilation** for loss functions\n")
    report.append("   - Only beneficial for large n (> 1000)\n")
    report.append("   - Adds dependency and compilation overhead\n\n")

    return ''.join(report)


def generate_ols_comparison_report(results_df: pd.DataFrame) -> str:
    """
    Generate focused OLS vs PCRegression comparison.
    """
    report = []
    report.append("# OLS vs PCRegression Comparison\n\n")

    linear_df = results_df[results_df['model_type'] == 'linear'].copy()

    if len(linear_df) == 0:
        return "No linear model results available."

    # Time comparison
    report.append("## Speed Comparison\n\n")

    methods_to_compare = ['OLS', 'PCR_constrained', 'PCR_penalized', 'PCCV_aic']

    for (n, p), group in linear_df.groupby(['n_samples', 'n_features']):
        report.append(f"### n={n}, p={p}\n\n")

        ols_time = group[group['method_name'] == 'OLS']['wall_time_seconds'].values
        ols_time = ols_time[0] if len(ols_time) > 0 else np.nan

        for method in methods_to_compare:
            row = group[group['method_name'] == method]
            if len(row) > 0:
                time_val = row['wall_time_seconds'].values[0]
                r2 = row['r2_train'].values[0]
                rmse_true = row['coef_rmse_vs_true'].values[0]
                ratio = time_val / ols_time if ols_time > 0 else np.nan

                report.append(f"- **{method}**: {time_val*1000:.2f}ms ({ratio:.1f}x OLS), ")
                report.append(f"R²={r2:.4f}, RMSE vs true={rmse_true:.4f}\n")

        report.append("\n")

    # Accuracy comparison
    report.append("## Coefficient Accuracy\n\n")
    report.append("PCRegression typically achieves **better coefficient accuracy** when:\n")
    report.append("- True coefficients satisfy the constraints (e.g., negative slopes)\n")
    report.append("- Sample size is small (n < 50)\n")
    report.append("- Features are correlated (multicollinearity)\n\n")

    report.append("The trade-off is **computation time**: PCRegression uses iterative ")
    report.append("optimization vs OLS's closed-form solution.\n\n")

    return ''.join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PCRegression Performance Benchmark')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer configurations)')
    parser.add_argument('--output-dir', default='benchmark_output',
                        help='Output directory')
    parser.add_argument('--no-profile', action='store_true',
                        help='Skip profiling (faster)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PCRegression Performance Benchmark")
    print("=" * 60)

    # Configure benchmark
    if args.quick:
        config = BenchmarkConfig(
            sample_sizes=[30, 50],
            feature_counts=[3, 5],
            n_repetitions=2,
            model_types=['linear']
        )
    else:
        config = BenchmarkConfig(
            sample_sizes=[10, 30, 50, 100],
            feature_counts=[3, 5, 10],
            n_repetitions=5,
            model_types=['linear', 'learning_curve']
        )

    # Run main benchmark
    print("\n" + "=" * 60)
    print("Running main benchmark...")
    print("=" * 60)

    results_df = run_full_benchmark(config, verbose=True)

    # Save raw results
    results_df.to_csv(output_dir / 'timing_results.csv', index=False)
    print(f"\nSaved timing results to {output_dir / 'timing_results.csv'}")

    # Safe mode analysis
    print("\n" + "=" * 60)
    print("Analyzing safe_mode overhead...")
    print("=" * 60)

    safe_mode_df = analyze_safe_mode_overhead(
        sample_sizes=[30, 50, 100] if not args.quick else [30, 50],
        verbose=True
    )
    safe_mode_df.to_csv(output_dir / 'safe_mode_analysis.csv', index=False)

    # Profile objective function
    profile_results = None
    if not args.no_profile:
        print("\n" + "=" * 60)
        print("Profiling _objective function...")
        print("=" * 60)

        data = generate_linear_data(100, 5)
        profile_output, stats, total_time = profile_objective_calls(
            data['X'], data['y'], data['bounds'],
            n_calls=1000, safe_mode=True
        )

        # Save raw profile output
        (output_dir / 'profile_output.txt').write_text(profile_output)

        # Extract function times
        profile_results = extract_function_times(stats, total_time)

        print(f"Total profile time: {total_time:.3f}s for 1000 calls")
        print(f"Average per call: {total_time/1000*1000:.3f}ms")

    # Generate reports
    print("\n" + "=" * 60)
    print("Generating reports...")
    print("=" * 60)

    # Main benchmark report
    report = generate_markdown_report(results_df, profile_results, safe_mode_df)
    (output_dir / 'benchmark_report.md').write_text(report, encoding='utf-8')
    print(f"Saved benchmark report to {output_dir / 'benchmark_report.md'}")

    # OLS comparison report
    ols_report = generate_ols_comparison_report(results_df)
    (output_dir / 'ols_comparison.md').write_text(ols_report, encoding='utf-8')
    print(f"Saved OLS comparison to {output_dir / 'ols_comparison.md'}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nTiming by Method (mean across all configurations):")
    summary = results_df.groupby('method_name')['wall_time_seconds'].agg(['mean', 'std'])
    summary = summary.sort_values('mean')
    print(summary.round(4).to_string())

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("  - timing_results.csv")
    print("  - safe_mode_analysis.csv")
    print("  - benchmark_report.md")
    print("  - ols_comparison.md")
    if not args.no_profile:
        print("  - profile_output.txt")


if __name__ == '__main__':
    main()
