"""
05_simulation_study_v2.py
=========================
Refactored simulation study with SimulationModel base class architecture.

Key Design:
- SimulationModel base class handles X transforms (via Pipeline) and y transforms
- All models receive identical raw (unit space) X, y data
- Log-space models: transform X and y to log, predict, exp back
- Unit-space models (PCReg): fit directly in unit space with custom prediction_fn
- Subclasses override only what differs
- Scenario config passed through __init__
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import warnings
from typing import Dict, List, Any, Tuple, Optional, Callable
import json
import os
import hashlib

# Suppress all warnings including sklearn's R² warnings for small samples
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Specifically suppress sklearn UndefinedMetricWarning for R² with few samples
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Add parent to path for pcreg import
sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg

# Import simulation data module
from simulation_data import get_scenario_data, get_test_data, load_or_generate_simulation_data

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score


# ============================================================================
# CONFIGURATION
# ============================================================================
from dataclasses import dataclass
@dataclass
class SimulationConfig:
    """
    Configuration dictionary for simulation study parameters.
    """
    pass

CONFIG = {
    # Experimental design
    'sample_sizes': [5, 10, 30],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],
    'rate_effects': [0.80, 0.85, 0.90],
    'n_replications': 100,

    # Fixed parameters
    'T1': 100,
    'base_seed': 42,

    # Model parameters
    # NOTE: For constrained models, use SMALL alpha values!
    # Large alphas (>1) over-regularize when constraints already limit the solution space.
    # With small samples, prefer near-zero alpha to let constraints do the work.
    'cv_folds': 3,
    'alpha_grid': np.logspace(-5, 0, 10),  # 0.00001 to 1.0 (was -2 to 2 = 0.01 to 100)
    'l1_ratio_grid': [0.0, 0.5, 1.0],  # Include pure Ridge (0.0) and pure Lasso (1.0)

    # Constraint bounds (in log space, i.e., slopes)
    'correct_loose_bounds': [(-0.5, 0), (-0.5, 0)],
    'wrong_constraint_a': 0.5,

    # Parallelization
    'n_jobs': -1,  # -1 = all cores, 1 = sequential
    'batch_size': 100,  # Scenarios per batch file
    'backend': 'loky',  # 'loky' (default), 'threading', or 'multiprocessing'

    # Output
    'output_dir': Path(__file__).parent / "output_v2",
    'save_predictions': True,
}

# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from X by index."""
    def __init__(self, columns: List[int]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Works with both numpy arrays and pandas DataFrames
        if hasattr(X, 'iloc'):
            return X.iloc[:, self.columns]
        return X[:, self.columns]

class ColumnNameSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from X by index."""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Works with both numpy arrays and pandas DataFrames
        return X[self.columns]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def unscale_coefficients(coef: np.ndarray, intercept: float, scaler: StandardScaler) -> Tuple[np.ndarray, float]:
    """
    Transform coefficients from scaled space back to original space.

    For standardized data: y = sum(coef_scaled * (x - mean) / scale) + intercept
    Returns coefficients for: y = sum(coef_original * x) + intercept_original

    Parameters
    ----------
    coef : ndarray
        Coefficients from regressor fitted on scaled data
    intercept : float
        Intercept from regressor fitted on scaled data
    scaler : StandardScaler
        Fitted scaler with mean_ and scale_ attributes

    Returns
    -------
    coef_original : ndarray
        Coefficients in original (unscaled) feature space
    intercept_original : float
        Intercept in original space
    """
    coef_original = coef / scaler.scale_
    intercept_original = intercept - np.sum(coef * scaler.mean_ / scaler.scale_)
    return coef_original, intercept_original


def extract_coefficients(model, model_name: str) -> Dict[str, float]:
    """
    Extract b, c, T1 from any fitted model.

    Handles:
    - TransformedTargetRegressor wrapping
    - StandardScaler unscaling
    - Different coefficient orderings (log-space vs unit-space)
    - Column selection (OLS_LearnOnly)

    Parameters
    ----------
    model : fitted estimator
        Fitted model (Pipeline, TransformedTargetRegressor, or PCReg)
    model_name : str
        Name of the model for determining extraction logic

    Returns
    -------
    dict with keys: 'b', 'c', 'intercept', 'T1_est'
    """
    # Unwrap TransformedTargetRegressor if present
    if hasattr(model, 'regressor_'):
        pipe = model.regressor_
    else:
        pipe = model

    # Handle PCReg models (not pipelines - have coef_ directly)
    if hasattr(pipe, 'coef_') and not hasattr(pipe, 'named_steps'):
        coef = pipe.coef_
        if model_name.startswith('PCReg') and 'Log' not in model_name:
            # Unit-space PCReg: coef = [T1, b, c]
            return {
                'b': coef[1] if len(coef) > 1 else np.nan,
                'c': coef[2] if len(coef) > 2 else 0.0,
                'intercept': 0.0,
                'T1_est': coef[0] if len(coef) > 0 else np.nan,
            }
        # Log-space PCReg: coef = [b, c]
        intercept = getattr(pipe, 'intercept_', 0.0)
        return {
            'b': coef[0] if len(coef) > 0 else np.nan,
            'c': coef[1] if len(coef) > 1 else 0.0,
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }

    # Pipeline models - access regressor via named_steps
    reg = pipe.named_steps['reg']
    coef = reg.coef_
    intercept = getattr(reg, 'intercept_', 0.0)

    # Unscale if StandardScaler present
    if 'scale' in pipe.named_steps:
        coef, intercept = unscale_coefficients(coef, intercept, pipe.named_steps['scale'])

    # Handle column selection (OLS_LearnOnly) - only b estimated, c=0
    if 'select' in pipe.named_steps:
        return {
            'b': coef[0] if len(coef) > 0 else np.nan,
            'c': 0.0,
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }

    # Standard log-space model: coef = [b, c]
    return {
        'b': coef[0] if len(coef) > 0 else np.nan,
        'c': coef[1] if len(coef) > 1 else 0.0,
        'intercept': intercept,
        'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
    }


def extract_regularization_params(model, model_name: str) -> Dict[str, Any]:
    """
    Extract regularization parameters (alpha, l1_ratio) from fitted model.

    Parameters
    ----------
    model : fitted estimator
        Fitted model
    model_name : str
        Name of the model

    Returns
    -------
    dict with keys: 'alpha', 'l1_ratio', 'converged' (as available)
    """
    # Unwrap TransformedTargetRegressor
    if hasattr(model, 'regressor_'):
        pipe = model.regressor_
    else:
        pipe = model

    # Get regressor (either from pipeline or directly)
    if hasattr(pipe, 'named_steps'):
        reg = pipe.named_steps['reg']
    else:
        reg = pipe

    params = {}
    if hasattr(reg, 'alpha_'):
        params['alpha'] = reg.alpha_
    if hasattr(reg, 'l1_ratio_'):
        params['l1_ratio'] = reg.l1_ratio_
    if hasattr(reg, 'converged_'):
        params['converged'] = reg.converged_
    else:
        params['converged'] = True  # Assume converged if not tracked

    return params


# ============================================================================
# UNIT-SPACE PREDICTION FUNCTION (for PCReg)
# ============================================================================
def unit_space_prediction_fn(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learning curve prediction function for unit-space fitting.

    Y = T1 * X1^b * X2^c

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Features where X[:, 0] is lot midpoint and X[:, 1] is lot quantity.
    params : ndarray of shape (3,)
        Parameters [T1, b, c].

    Returns
    -------
    y_pred : ndarray
        Predicted costs in unit space.
    """
    T1, b, c = params[0], params[1], params[2]
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)

def generate_models(scenario_config: Dict = {}) -> Dict[str, Any]:
    """
    Generate model instances based on scenario configuration.

    This is the SINGLE source of truth for model creation. Each model is
    independently created with scenario-specific parameters.

    Parameters
    ----------
    scenario_config : dict
        Scenario-specific configuration containing:
        - alpha_grid: Array of alpha values for CV selection
        - l1_ratio_grid: Array of l1_ratio values
        - cv_folds: Number of CV folds
        - correct_loose_bounds: Default bounds for PCReg [(b_lo, b_hi), (c_lo, c_hi)]
        - tight_bounds: Tight bounds for PCReg_*_Tight variants
        - wrong_bounds: Wrong bounds for PCReg_*_Wrong variants
        - T1: Initial T1 estimate
        - b_true, c_true: True parameter values (for x0 initialization)

    Returns
    -------
    models : dict
        Dictionary of {model_name: configured_model_instance}
    """
    # Extract config with defaults
    alpha_grid = scenario_config.get('alpha_grid', np.logspace(-5, 0, 10))
    l1_ratio_grid = scenario_config.get('l1_ratio_grid', [0.0, 0.5, 1.0])
    cv_folds = scenario_config.get('cv_folds', 3)
    correct_bounds = scenario_config.get('correct_loose_bounds', [(-0.5, 0), (-0.5, 0)])
    tight_bounds = scenario_config.get('tight_bounds', [(-0.5, 0), (-0.5, 0)])
    wrong_bounds = scenario_config.get('wrong_bounds', [(-0.5, 0), (-0.5, 0)])
    T1 = scenario_config.get('T1', 100)
    b_init = scenario_config.get('b_true', -0.15)
    c_init = scenario_config.get('c_true', -0.23)

    # Unit-space bounds include T1 as first parameter
    unit_bounds_correct = [(0, None)] + correct_bounds
    unit_bounds_tight = [(0, None)] + tight_bounds
    unit_bounds_wrong = [(0, None)] + wrong_bounds

    # x0 for unit-space models
    x0_unit = [T1, b_init, c_init]

    models = {}

    # -------------------------------------------------------------------------
    # Log-space models (TransformedTargetRegressor wrapping pipeline)
    # -------------------------------------------------------------------------

    # OLS: log(y) = intercept + b*log(x1) + c*log(x2)
    models['OLS'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('reg', LinearRegression()),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    # OLS with only learning variable (column 0)
    models['OLS_LearnOnly'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('select', ColumnSelector([0])),
            ('log', FunctionTransformer(np.log)),
            ('reg', LinearRegression()),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    # RidgeCV with StandardScaler for proper regularization
    models['RidgeCV'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('scale', StandardScaler()),
            ('reg', RidgeCV(alphas=alpha_grid, cv=cv_folds)),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    # LassoCV
    models['LassoCV'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('reg', LassoCV(alphas=alpha_grid, cv=cv_folds, max_iter=5000)),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    # BayesianRidge
    models['BayesianRidge'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('reg', BayesianRidge()),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    # -------------------------------------------------------------------------
    # Unit-space PCReg models (direct fitting with custom prediction_fn)
    # -------------------------------------------------------------------------

    # PCReg constrain-only (no penalty, alpha=0)
    models['PCReg_ConstrainOnly'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_correct,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=[0],
        l1_ratios=[0],
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # PCReg with CV selection
    models['PCReg_CV'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_correct,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid,
        cv=cv_folds,
        selection='cv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # PCReg with GCV selection
    models['PCReg_GCV'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_correct,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid,
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # PCReg with AICc selection
    models['PCReg_AICc'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_correct,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid,
        selection='aicc',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # PCReg with GCV and tight bounds
    models['PCReg_GCV_Tight'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_tight,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid,
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # PCReg with GCV and wrong bounds
    models['PCReg_GCV_Wrong'] = pcreg.PenalizedConstrainedCV(
        bounds=unit_bounds_wrong,
        coef_names=['T1', 'b', 'c'],
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=x0_unit,
        loss='sspe',
        alphas=alpha_grid,
        l1_ratios=l1_ratio_grid,
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
    )

    # -------------------------------------------------------------------------
    # Log-space constrained model (PCReg with MSE loss, fits in log space)
    # -------------------------------------------------------------------------

    # PCReg in log-space with GCV (constrained OLS-like)
    models['PCRegGCV_LogMSE'] = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('scale', StandardScaler()),
            ('reg', pcreg.PenalizedConstrainedCV(
                bounds=correct_bounds,  # Only [b, c] bounds (no T1)
                coef_names=['b', 'c'],
                fit_intercept=True,
                loss='mse',
                prediction_fn=None,
                alphas=alpha_grid,
                l1_ratios=l1_ratio_grid,
                selection='gcv',
                n_jobs=1,
                verbose=0,
            )),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )

    return models


# ============================================================================
# MODEL REGISTRY
# ============================================================================
def get_model_registry() -> List[str]:
    """
    Get list of all available model names.

    Generates a dummy scenario_config to extract model names from generate_models().
    """
    dummy_config = {
        'alpha_grid': np.logspace(-5, 0, 10),
        'l1_ratio_grid': [0.0, 0.5, 1.0],
        'cv_folds': 3,
        'correct_loose_bounds': [(-0.5, 0), (-0.5, 0)],
        'tight_bounds': [(-0.5, 0), (-0.5, 0)],
        'wrong_bounds': [(-0.5, 0), (-0.5, 0)],
        'T1': 100,
        'b_true': -0.15,
        'c_true': -0.23,
    }
    return list(generate_models(dummy_config).keys())


# Module-level constant (generated once at import)
MODEL_REGISTRY = get_model_registry()


# ============================================================================
# MODEL HASHING (for cache invalidation)
# ============================================================================
def get_model_hash(model_name: str) -> str:
    """
    Generate a hash for a model based on its name.

    Simple scenario-independent hashing. Model structure changes are
    tracked by code version, not parameter values.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    hash : str
        12-character hash of the model name
    """
    return hashlib.md5(model_name.encode()).hexdigest()[:12]


def get_all_model_hashes() -> Dict[str, str]:
    """Get hashes for all registered models."""
    return {name: get_model_hash(name) for name in MODEL_REGISTRY}


# ============================================================================
# WRONG BOUNDS GENERATION
# ============================================================================
def generate_wrong_bounds(true_b: float,  true_c: float,  a: float = 0.1,
    random_state: Optional[int] = None,
    upper_cap: float = -1e-12,  # keep upper bound strictly negative; set to 0.0 if non-positive is OK
) -> List[Tuple[float, float]]:
    """
    Generate wrong bounds using one uncertainty parameter 'a'.
    The interval direction (wrongness) is chosen at random:
      - Downward: interval lies below the true value (upper < true)
      - Upward:   interval lies above the true value (lower > true)
    Bounds are kept strictly negative.

    Parameters
    ----------
    true_b, true_c : float
        True parameter values; expected in [-0.5, 0).
    a : float
        Uncertainty scale (same units as the parameters). Controls maximum random gap/width.
    random_state : Optional[int]
        Seed for reproducibility.
    upper_cap : float
        The maximum allowed upper bound; use 0.0 if you allow upper to be exactly zero.

    Returns
    -------
    List[Tuple[float, float]]
        [(lower_b, upper_b), (lower_c, upper_c)]
    """
    
    rng = np.random.default_rng(random_state)

    def one_interval(true_val: float) -> Tuple[float, float]:
        # Randomly choose direction: -1 => downward (below truth), +1 => upward (above truth)
        direction = rng.choice([-1, 1])
        # Random gap and width, both in [0, a]
        gap = rng.uniform(0.0, a)
        width = rng.uniform(0.0, .5)
        if direction == -1:
            # Downward: upper is below true by 'gap', then lower extends further down by 'width'
            upper = true_val - gap
            lower = upper - width
        else:
            # Upward: lower is above true by 'gap', then upper extends above lower by 'width'
            lower = true_val + gap
            upper = upper_cap

        # Enforce strictly negative bounds
                  # cap upper at <= 0 (or tiny negative)
        if lower >= 0:
            # If lower drifted non-negative, pull it just below upper
            lower = upper - max(width, 1e-6)

        # Ensure ordering and a tiny gap
        if lower >= upper:
            lower = upper - 1e-6

        return (lower, upper)

    b_bounds = one_interval(true_b)
    c_bounds = one_interval(true_c)
    return [b_bounds, c_bounds]

# ============================================================================
# FITTING AND METRICS
# ============================================================================
def fit_and_extract(
    model,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    true_params: Dict[str, float],
) -> Tuple[Dict[str, Any], Any]:
    """
    Fit a model and extract results using helper functions.

    Parameters
    ----------
    model : estimator
        Model instance to fit (Pipeline, TransformedTargetRegressor, or PCReg)
    model_name : str
        Name of the model for coefficient extraction logic
    X : ndarray
        Features in unit space.
    y : ndarray
        Target in unit space.
    true_params : dict
        True parameter values {'b', 'c', 'T1'}.

    Returns
    -------
    result : dict
        Fit results and metrics.
    model : estimator
        Fitted model.
    """
    result = {
        'model_name': model_name,
        'converged': True,
        'fit_time': 0.0,
    }

    try:
        start_time = time.time()
        model.fit(X, y)
        result['fit_time'] = time.time() - start_time

        # Get coefficients using helper function
        coefs = extract_coefficients(model, model_name)
        result.update(coefs)

        # Derive learning curve slopes from b and c (LC_est = 2^b, RC_est = 2^c)
        b = coefs.get('b', np.nan)
        c = coefs.get('c', np.nan)
        result['LC_est'] = 2 ** b if not np.isnan(b) else np.nan
        result['RC_est'] = 2 ** c if not np.isnan(c) else np.nan

        # R² in unit space
        y_pred = model.predict(X)
        result['r2'] = r2_score(y, y_pred)

        # Regularization params using helper function
        reg_params = extract_regularization_params(model, model_name)
        result.update(reg_params)

        if 'converged' not in result:
            result['converged'] = True

    except Exception as e:
        result['converged'] = False
        result['error'] = str(e)
        for key in ['b', 'c', 'intercept', 'T1_est', 'LC_est', 'RC_est', 'r2']:
            result[key] = np.nan

    # Compute errors vs true values
    if not np.isnan(result.get('b', np.nan)):
        result['b_error'] = abs(result['b'] - true_params['b'])
        result['c_error'] = abs(result['c'] - true_params['c'])
        result['b_correct_sign'] = result['b'] <= 0
        result['c_correct_sign'] = result['c'] <= 0
        if not np.isnan(result.get('T1_est', np.nan)):
            result['T1_error'] = abs(result['T1_est'] - true_params['T1'])
            result['T1_pct_error'] = result['T1_error'] / true_params['T1']
        else:
            result['T1_error'] = np.nan
            result['T1_pct_error'] = np.nan
    else:
        result['b_error'] = np.nan
        result['c_error'] = np.nan
        result['b_correct_sign'] = False
        result['c_correct_sign'] = False
        result['T1_error'] = np.nan
        result['T1_pct_error'] = np.nan

    return result, model


def compute_oos_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute out-of-sample metrics in unit space."""
    residuals = y_true - y_pred
    pct_errors = residuals / y_true

    return {
        'test_sspe': np.sum(pct_errors ** 2),
        'test_mape': np.mean(np.abs(pct_errors)),
        'test_mse': np.mean(residuals ** 2),
        'test_n_lots': len(y_true),
    }


# ============================================================================
# SCENARIO RUNNER
# ============================================================================
def run_single_scenario(
    scenario: Tuple,
    config: Dict,
    models_to_run: Optional[List[str]] = None,
    simulation_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run models on a single scenario.

    All models receive the SAME raw X, y data in unit space.

    Parameters
    ----------
    scenario : tuple
        (scenario_id,) or (n_lots, cv_error, learning_rate, rate_effect, replication)
    config : dict
        Simulation configuration.
    models_to_run : list of str, optional
        List of model names to run. If None, runs all models in MODEL_REGISTRY.
    simulation_df : pd.DataFrame, optional
        Pre-loaded simulation data. If None, loads from cache.
    """
    # Get scenario data from simulation_data module
    scenario_id = scenario[0] if len(scenario) == 1 else scenario
    lot_data = get_scenario_data(scenario_id, simulation_df)

    # Extract training data from lot_data DataFrame
    X_train = lot_data[['lot_midpoint', 'lot_quantity']].values
    y_train = lot_data['observed_cost'].values

    # Extract parameters from lot_data
    b_true = lot_data['b_true'].iloc[0]
    c_true = lot_data['c_true'].iloc[0]
    T1_true = lot_data['T1_true'].iloc[0]
    n_lots = int(lot_data['n_lots'].iloc[0])
    cv_error = lot_data['cv_error'].iloc[0]
    learning_rate = lot_data['learning_rate'].iloc[0]
    rate_effect = lot_data['rate_effect'].iloc[0]
    replication = int(lot_data['replication'].iloc[0])
    seed = int(lot_data['seed'].iloc[0])
    actual_corr = lot_data['actual_correlation'].iloc[0]

    true_params = {'b': b_true, 'c': c_true, 'T1': T1_true}

    # Get real holdout test data (may be empty if program has exactly n_lots)
    test_df = get_test_data(scenario_id, simulation_df)
    has_test = len(test_df) > 0

    if has_test:
        X_test = test_df[['lot_midpoint', 'lot_quantity']].values
        y_test = test_df['true_cost'].values  # Use true cost (no error) for clean comparison
        test_n_lots = len(test_df)
    else:
        X_test = None
        y_test = None
        test_n_lots = 0

    # Build scenario config for models
    tight_margin = 0.05

    tight_bounds = [
        (b_true - tight_margin, b_true + tight_margin),
        (c_true - tight_margin, c_true + tight_margin)
    ]
    wrong_bounds = generate_wrong_bounds(b_true, c_true,
                                          a=config['wrong_constraint_a'],
                                          random_state=seed)

    scenario_config = {
        'b_true': b_true,
        'c_true': c_true,
        'T1': T1_true,
        'correct_bounds': config['correct_loose_bounds'],
        'tight_bounds': tight_bounds,
        'wrong_bounds': wrong_bounds,
        'alpha_grid': config['alpha_grid'],
        'l1_ratio_grid': config['l1_ratio_grid'],
        'cv_folds': config['cv_folds'],
    }

    # Determine which models to run
    if models_to_run is None:
        models_to_run = MODEL_REGISTRY

    # Generate all models for this scenario
    all_models = generate_models(scenario_config)

    # Fit models
    results = []
    predictions = []
    save_predictions = config.get('save_predictions', False)

    for model_name in models_to_run:
        # Get configured model from generated models
        model = all_models[model_name]

        # Fit model and extract results
        result, fitted_model = fit_and_extract(model, model_name, X_train, y_train, true_params)

        # Compute OOS metrics (only if test data available and model converged)
        if has_test and result['converged'] and not np.isnan(result.get('b', np.nan)):
            try:
                y_pred = fitted_model.predict(X_test)
                oos_metrics = compute_oos_metrics(y_pred, y_test)
                result.update(oos_metrics)

                # Lot-level predictions
                if save_predictions:
                    pred_errors = y_pred - y_test
                    pct_errors = pred_errors / y_test

                    for i in range(test_n_lots):
                        predictions.append({
                            'model_name': model_name,
                            'n_lots': n_lots,
                            'actual_correlation': actual_corr,
                            'cv_error': cv_error,
                            'learning_rate': learning_rate,
                            'rate_effect': rate_effect,
                            'replication': replication,
                            'seed': seed,
                            'test_lot_number': i + 1,
                            'test_lot_midpoint': test_df['lot_midpoint'].iloc[i],
                            'test_lot_quantity': test_df['lot_quantity'].iloc[i],
                            'true_cost': y_test[i],
                            'predicted_cost': y_pred[i],
                            'prediction_error': pred_errors[i],
                            'pct_error': pct_errors[i],
                        })

            except Exception as e:
                result.update({
                    'test_sspe': np.nan,
                    'test_mape': np.nan,
                    'test_mse': np.nan,
                    'test_n_lots': 0,
                    'test_error': str(e),
                })
        else:
            result.update({
                'test_sspe': np.nan,
                'test_mape': np.nan,
                'test_mse': np.nan,
                'test_n_lots': 0,
            })

        # Add scenario metadata and model hash
        result.update({
            'n_lots': n_lots,
            'actual_correlation': actual_corr,
            'cv_error': cv_error,
            'learning_rate': learning_rate,
            'rate_effect': rate_effect,
            'b_true': b_true,
            'c_true': c_true,
            'T1_true': T1_true,
            'replication': replication,
            'seed': seed,
            'has_test_data': has_test,
            'model_hash': get_model_hash(model_name),
        })

        results.append(result)

    return results, predictions


# ============================================================================
# PARALLEL PROCESSING WITH BATCHING
# ============================================================================
def _scenario_worker(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """Worker function for parallel execution."""
    scenario_id, config, models_to_run, simulation_df = args
    return run_single_scenario((scenario_id,), config, models_to_run, simulation_df)


def save_batch_file(
    results: List[Dict],
    predictions: List[Dict],
    output_dir: Path,
    batch_id: int,
):
    """Save batch to separate files."""
    if results:
        results_file = output_dir / f'batch_results_{batch_id:05d}.parquet'
        pd.DataFrame(results).to_parquet(results_file, engine='pyarrow', index=False)

    if predictions:
        pred_file = output_dir / f'batch_predictions_{batch_id:05d}.parquet'
        pd.DataFrame(predictions).to_parquet(pred_file, engine='pyarrow', index=False)


def merge_batch_files(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all batch files into final output, appending to existing results."""
    results_file = output_dir / 'simulation_results.parquet'
    pred_file = output_dir / 'predictions_flat.parquet'

    # Merge results - include existing file if present
    results_files = sorted(output_dir.glob('batch_results_*.parquet'))
    if results_files:
        results_dfs = [pd.read_parquet(f) for f in results_files]

        # Append to existing results if file exists
        if results_file.exists():
            existing_df = pd.read_parquet(results_file)
            results_dfs.insert(0, existing_df)

        results_df = pd.concat(results_dfs, ignore_index=True)
        results_df.to_parquet(results_file, engine='pyarrow', index=False)

        # Clean up batch files
        for f in results_files:
            f.unlink()
    else:
        # No new batches, just return existing
        if results_file.exists():
            results_df = pd.read_parquet(results_file)
        else:
            results_df = pd.DataFrame()

    # Merge predictions - include existing file if present
    pred_batch_files = sorted(output_dir.glob('batch_predictions_*.parquet'))
    if pred_batch_files:
        pred_dfs = [pd.read_parquet(f) for f in pred_batch_files]

        # Append to existing predictions if file exists
        if pred_file.exists():
            existing_pred_df = pd.read_parquet(pred_file)
            pred_dfs.insert(0, existing_pred_df)

        predictions_df = pd.concat(pred_dfs, ignore_index=True)
        predictions_df.to_parquet(pred_file, engine='pyarrow', index=False)

        # Clean up batch files
        for f in pred_batch_files:
            f.unlink()
    else:
        # No new batches, just return existing
        if pred_file.exists():
            predictions_df = pd.read_parquet(pred_file)
        else:
            predictions_df = pd.DataFrame()

    return results_df, predictions_df


def load_completed_scenarios(output_dir: Path) -> Tuple[set, set]:
    """
    Load scenario+model combinations that have already been computed.

    Returns
    -------
    completed_scenario_models : set
        Set of (n_lots, correlation, cv_error, learning_rate, rate_effect, replication, model_hash) tuples.
    completed_scenarios : set
        Set of (n_lots, correlation, cv_error, learning_rate, rate_effect, replication) tuples
        where ALL current models have been run.
    """
    results_file = output_dir / 'simulation_results.parquet'

    if not results_file.exists():
        return set(), set()

    try:
        df = pd.read_parquet(results_file)
        if df.empty:
            return set(), set()

        # Get current model hashes
        current_hashes = get_all_model_hashes()

        # Check if model_hash column exists (for migration)
        if 'model_hash' not in df.columns:
            print("Warning: model_hash column not found. Run migrate_add_model_hashes() first.")
            return set(), set()

        # Build set of completed (scenario, model_hash) combinations
        scenario_cols = ['n_lots', 'actual_correlation', 'cv_error',
                         'learning_rate', 'rate_effect', 'replication', 'model_hash']

        completed_scenario_models = set()
        for _, row in df[scenario_cols].drop_duplicates().iterrows():
            key = (row['n_lots'], row['actual_correlation'], row['cv_error'],
                   row['learning_rate'], row['rate_effect'], row['replication'],
                   row['model_hash'])
            completed_scenario_models.add(key)

        # Find scenarios where ALL current models have been run
        scenario_only_cols = ['n_lots', 'actual_correlation', 'cv_error',
                              'learning_rate', 'rate_effect', 'replication']
        completed_scenarios = set()

        for _, row in df[scenario_only_cols].drop_duplicates().iterrows():
            scenario_key = (row['n_lots'], row['actual_correlation'], row['cv_error'],
                           row['learning_rate'], row['rate_effect'], row['replication'])

            # Check if all current model hashes are present for this scenario
            scenario_hashes = {h for (n, cor, cv, lr, re, rep, h) in completed_scenario_models
                              if (n, cor, cv, lr, re, rep) == scenario_key}

            if set(current_hashes.values()).issubset(scenario_hashes):
                completed_scenarios.add(scenario_key)

        return completed_scenario_models, completed_scenarios

    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set(), set()


def get_models_to_run(scenario: Tuple, completed_scenario_models: set) -> List[str]:
    """
    Determine which models need to be run for a scenario.

    Parameters
    ----------
    scenario : tuple
        (n_lots, correlation, cv_error, learning_rate, rate_effect, replication)
    completed_scenario_models : set
        Set of completed (scenario..., model_hash) tuples.

    Returns
    -------
    models_to_run : list
        List of model names that need to be run.
    """
    current_hashes = get_all_model_hashes()
    models_to_run = []

    for model_name, model_hash in current_hashes.items():
        key = (*scenario, model_hash)
        if key not in completed_scenario_models:
            models_to_run.append(model_name)

    return models_to_run


def run_simulation(config: Dict, parallel: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full simulation study.

    Parameters
    ----------
    config : dict
        Simulation configuration
    parallel : bool, default=True
        If True, use joblib.Parallel for parallel execution.
        If False, run sequentially (useful for debugging).

    Returns
    -------
    results_df : pd.DataFrame
        Results for all model fits
    predictions_df : pd.DataFrame
        Lot-level predictions (if save_predictions=True)
    """
    output_dir = config['output_dir']
    output_dir.mkdir(exist_ok=True)

    n_models = len(MODEL_REGISTRY)

    # Load simulation data (generates if needed, uses cache otherwise)
    print("Loading simulation data...")
    simulation_df = load_or_generate_simulation_data(
        sample_sizes=config['sample_sizes'],
        cv_errors=config['cv_errors'],
        learning_rates=config['learning_rates'],
        rate_effects=config['rate_effects'],
        n_replications=config['n_replications'],
        T1=config['T1'],
        base_seed=config['base_seed'],
    )

    # Get unique scenario IDs
    all_scenario_ids = simulation_df['scenario_id'].unique().tolist()
    n_total = len(all_scenario_ids)

    # Resume capability - check completed scenarios
    completed_scenario_models, completed_scenarios = load_completed_scenarios(output_dir)

    # Build list of (scenario_id, models_to_run) pairs
    scenarios_with_models = []
    total_model_fits = 0
    for scenario_id in all_scenario_ids:
        # Get scenario tuple for compatibility with completed tracking
        row = simulation_df[simulation_df['scenario_id'] == scenario_id].iloc[0]
        scenario_tuple = (int(row['n_lots']), row['actual_correlation'], row['cv_error'],
                         row['learning_rate'], row['rate_effect'], int(row['replication']))

        if scenario_tuple in completed_scenarios:
            continue
        models_needed = get_models_to_run(scenario_tuple, completed_scenario_models)
        if models_needed:
            scenarios_with_models.append((scenario_id, models_needed))
            total_model_fits += len(models_needed)

    n_to_run = len(scenarios_with_models)
    n_skipped = n_total - n_to_run

    # Print summary
    print(f"Total scenarios: {n_total}")
    print(f"Fully completed scenarios: {n_skipped}")
    print(f"Scenarios needing models: {n_to_run}")
    print(f"Total models in registry: {n_models}")
    print(f"Total model fits needed: {total_model_fits}")
    if parallel:
        print(f"Parallel workers: {config['n_jobs']} (backend: {config['backend']})")
    else:
        print("Running sequentially (debug mode)")
    print(f"Batch size: {config['batch_size']}")
    print()

    if n_to_run == 0:
        print("All scenarios and models completed!")
        return merge_batch_files(output_dir)

    # Execute scenarios
    start_time = time.time()
    batch_size = config['batch_size']
    n_batches = (n_to_run + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_to_run)
        batch_items = scenarios_with_models[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{n_batches} "
              f"(scenarios {batch_start + 1}-{batch_end})...")

        # Build worker args
        worker_args = [(s_id, config, models, simulation_df) for s_id, models in batch_items]

        # Execute batch - parallel or sequential
        if parallel:
            from joblib import Parallel, delayed
            batch_results_list = Parallel(
                n_jobs=config['n_jobs'],
                backend=config['backend'],
                verbose=0,
            )(delayed(_scenario_worker)(args) for args in worker_args)
        else:
            batch_results_list = [_scenario_worker(args) for args in worker_args]

        # Collect results
        batch_results = []
        batch_predictions = []
        for results, predictions in batch_results_list:
            batch_results.extend(results)
            batch_predictions.extend(predictions)

        # Save batch
        save_batch_file(batch_results, batch_predictions, output_dir, batch_idx)

        # Progress reporting
        elapsed = time.time() - start_time
        scenarios_done = batch_end
        rate = scenarios_done / elapsed if elapsed > 0 else 0
        remaining = (n_to_run - scenarios_done) / rate if rate > 0 else 0

        print(f"  Saved {len(batch_results)} model fits")
        print(f"  Progress: {scenarios_done}/{n_to_run} ({100*scenarios_done/n_to_run:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min")
        print()

    # Merge all batch files
    print("Merging batch files...")
    results_df, predictions_df = merge_batch_files(output_dir)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Average per scenario: {total_time/n_to_run:.3f} seconds")

    return results_df, predictions_df


# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_results(df: pd.DataFrame) -> Dict:
    """Compute summary statistics."""
    analysis = {}

    df_conv = df[df['converged'] == True].copy()
    has_oos = 'test_sspe' in df_conv.columns

    agg_dict = {
        'b_error': ['mean', 'std'],
        'c_error': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'b_correct_sign': 'mean',
        'c_correct_sign': 'mean',
        'fit_time': 'mean',
        'converged': 'sum'
    }

    if has_oos:
        agg_dict['test_sspe'] = ['mean', 'std']
        agg_dict['test_mape'] = ['mean', 'std']

    if 'T1_pct_error' in df_conv.columns:
        agg_dict['T1_pct_error'] = ['mean', 'std']

    method_stats = df_conv.groupby('model_name').agg(agg_dict).round(4)
    analysis['method_stats'] = method_stats

    # Winner analysis
    winners = {}
    metrics_to_analyze = ['b_error', 'c_error']
    if has_oos:
        metrics_to_analyze.append('test_sspe')

    scenario_cols = ['n_lots', 'actual_correlation', 'cv_error',
                     'learning_rate', 'rate_effect', 'replication']

    for metric in metrics_to_analyze:
        win_counts = {}
        for _, group in df_conv.groupby(scenario_cols):
            group_valid = group.dropna(subset=[metric])
            if len(group_valid) > 0:
                winner = group_valid.loc[group_valid[metric].idxmin(), 'model_name']
                win_counts[winner] = win_counts.get(winner, 0) + 1
        winners[metric] = win_counts

    analysis['winners'] = winners

    return analysis


def print_summary(df: pd.DataFrame, analysis: Dict):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal observations: {len(df)}")
    print(f"Converged: {df['converged'].sum()} ({100*df['converged'].mean():.1f}%)")

    print("\n" + "-" * 80)
    print("OVERALL METHOD PERFORMANCE")
    print("-" * 80)

    stats = analysis['method_stats']
    has_oos = ('test_sspe', 'mean') in stats.columns

    if has_oos:
        print(f"{'Method':<20} {'b_err':>8} {'c_err':>8} {'R2':>6} {'SSPE':>10} {'MAPE':>8}")
        print("-" * 80)

        for method in stats.index:
            row = stats.loc[method]
            print(f"{method:<20} "
                  f"{row[('b_error', 'mean')]:>8.4f} "
                  f"{row[('c_error', 'mean')]:>8.4f} "
                  f"{row[('r2', 'mean')]:>6.3f} "
                  f"{row[('test_sspe', 'mean')]:>10.4f} "
                  f"{row[('test_mape', 'mean')]:>8.4f}")

    print("\n" + "-" * 80)
    print("WINNER COUNTS")
    print("-" * 80)

    for metric, counts in analysis['winners'].items():
        print(f"\n{metric}:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for method, count in sorted_counts[:5]:
            print(f"  {method}: {count}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run simulation study v2')
    parser.add_argument('--show-hashes', action='store_true',
                        help='Show current model hashes and exit')
    parser.add_argument('--run-doe', action='store_true',
                        help='Run DOE analysis after simulation completes')
    parser.add_argument('--doe-only', action='store_true',
                        help='Run only DOE analysis (skip simulation)')
    args = parser.parse_args()

    if args.show_hashes:
        print("Current model hashes:")
        print("-" * 50)
        for name, hash_val in sorted(get_all_model_hashes().items()):
            print(f"  {name:<25} {hash_val}")
        sys.exit(0)

    if args.doe_only:
        print("Running DOE analysis only...")
        from doe_analysis import run_full_doe_analysis
        run_full_doe_analysis()
        sys.exit(0)

    print("=" * 80)
    print("SIMULATION STUDY v2: SimulationModel Base Class Architecture")
    print("Log-space models: OLS, Ridge, Lasso, etc.")
    print("Unit-space models: PCReg variants")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Sample sizes: {CONFIG['sample_sizes']}")
    print(f"  CV errors: {CONFIG['cv_errors']}")
    print(f"  Learning rates: {CONFIG['learning_rates']}")
    print(f"  Rate effects: {CONFIG['rate_effects']}")
    print(f"  Replications: {CONFIG['n_replications']}")
    print(f"  Parallel workers: {CONFIG['n_jobs']}")
    print()

    # Run simulation
    results_df, predictions_df = run_simulation(CONFIG, parallel=(CONFIG['n_jobs'] != 1))

    # Save config
    output_dir = CONFIG['output_dir']
    config_save = {k: str(v) if isinstance(v, (Path, np.ndarray)) else v
                   for k, v in CONFIG.items()}
    with open(output_dir / 'simulation_config.json', 'w') as f:
        json.dump(config_save, f, indent=2, default=str)

    # Analyze
    print("\nAnalyzing results...")
    analysis = analyze_results(results_df)
    print_summary(results_df, analysis)

    # Leaderboard
    print("\n" + "=" * 80)
    print("LEADERBOARD BY TEST_SSPE")
    print("=" * 80)
    print(results_df.groupby('model_name')['test_sspe'].mean()
          .sort_values(ascending=True).to_string())

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

    # Run DOE analysis if requested
    if args.run_doe:
        print("\n" + "=" * 80)
        print("RUNNING DOE ANALYSIS")
        print("=" * 80)
        from doe_analysis import run_full_doe_analysis
        run_full_doe_analysis()
