"""
05_simulation_study.py
======================
Full factorial simulation study for the ICEAA 2026 paper.

Compares multiple regression methods across:
- Sample sizes: {5, 10, 30}
- Correlations: {0, 0.5, 0.9}
- Error CVs: {0.01, 0.1, 0.2}
- Learning slopes: {85%, 90%, 95%}
- Rate slopes: {80%, 85%, 90%}
- Replications: 50 per scenario

Run without arguments for full reproducibility.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings
from itertools import product
from typing import Dict, List, Any, Tuple
import json

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, BayesianRidge

# Model name constants for faster membership checks
PCREG_MODELS = frozenset(['PCRegConstrain_Only', 'PCRegCV', 'PCRegCV_Tight', 'PCRegCV_Wrong'])

# Scenario key columns for identifying unique scenarios
SCENARIO_KEY_COLS = ('n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication')

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION - All parameters defined here for reproducibility
# ============================================================================
CONFIG = {
    # Experimental design
    'sample_sizes': [5, 10, 30],
    'correlations': [0.0, 0.5, 0.9],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],  # Converts to b slope
    'rate_effects': [0.80, 0.85, 0.90],     # Converts to c slope
    'n_replications': 11,

    # Fixed parameters
    'T1': 100,
    'base_seed': 42,

    # Model parameters
    'cv_folds': 3,
    'alpha_grid': np.logspace(-2, 2, 10),
    'l1_ratio_grid': [0.0, 0.5, 1.0],

    # Penalized-constrained model options
    'fit_in_unit_space': False,  # If True, use lc_func for prediction in unit space

    # Constraint error magnitudes to test per PAC paper (James et al. 2020)
    # a=1.0 corresponds to 50% average error in constraints
    'wrong_constraint_a': 0.5,  # Default perturbation magnitude for wrong constraints

    # Base correct bounds (both slopes negative for improvement)
    'correct_loose_bounds': [(-0.5, 0), (-0.5, 0)],

    # Out-of-sample testing parameters
    'test_n_lots': 5,  # Number of lots for out-of-sample testing
    'test_quantity_multiplier': 2,  # Test lot quantity = max(train_quantity) * multiplier

    # Batch processing for incremental saves
    'batch_size': 100,  # Number of scenarios per batch before saving
    'resume': False,  # If True, skip scenarios that have already been computed

    # Output
    'output_dir': Path(__file__).parent / "output",
    'save_intermediate': True,
    'save_predictions': True,  # Save lot-level predictions to parquet
}


def generate_wrong_bounds(true_b: float, true_c: float, a: float = 0.5,
                          random_state: int = None) -> List[tuple]:
    """
    Generate 'wrong' constraints following James et al. (2020) PAC methodology.

    Perturbs the true bounds multiplicatively:
    constraint_wrong = constraint_true * (1 + U(0, a))

    Parameters
    ----------
    true_b : float
        True learning slope (negative, e.g., ln(0.90)/ln(2) ≈ -0.152)
    true_c : float
        True rate slope (negative, e.g., ln(0.85)/ln(2) ≈ -0.234)
    a : float
        Magnitude of constraint error (0.25, 0.50, 0.75, 1.00).
        a=1.0 corresponds to 50% average error in constraints.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    bounds : list of tuples
        Perturbed bounds for [b, c]
    """
    rng = np.random.RandomState(random_state)

    # Generate uniform perturbations
    u_b = rng.uniform(0, a)
    u_c = rng.uniform(0, a)

    # Apply multiplicative perturbation to bounds
    # For the loose bounds [-1, 0] and [-0.5, 0], perturb the lower bound
    # This widens the constraint region by making it more negative
    perturbed_lower_b = -1 * (1 + u_b)  # More negative than -1
    perturbed_lower_c = -0.5 * (1 + u_c)  # More negative than -0.5

    return [(perturbed_lower_b, 0), (perturbed_lower_c, 0)]


def lc_prediction_fn(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learning curve prediction function for unit-space fitting.

    Computes: Y = T1 * X1^b * X2^c

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Input features where X[:, 0] is lot midpoint and X[:, 1] is rate variable.
    params : ndarray of shape (3,)
        Parameters [T1, b, c] where T1 is the theoretical first unit cost,
        b is the learning slope, and c is the rate slope.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted costs.
    """
    T1, b, c = params[0], params[1], params[2]
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)


def get_models(learning_rate: float, rate_effect: float, cv_folds: int = 3,
               fit_in_unit_space: bool = False, wrong_constraint_a: float = 0.5,
               random_state: int = None) -> Dict[str, Any]:
    """
    Define all models to compare, including scenario-aware penalized-constrained variants.

    Parameters
    ----------
    learning_rate : float
        Learning rate (e.g., 0.90 for 90% learning).
    rate_effect : float
        Rate effect (e.g., 0.85 for 85% rate effect).
    cv_folds : int
        Number of CV folds for cross-validated models.
    fit_in_unit_space : bool
        If True, fit penalized-constrained models in unit space using lc_prediction_fn.
    wrong_constraint_a : float
        Constraint error magnitude for "wrong" bounds per PAC paper.
    random_state : int, optional
        Random seed for wrong bounds generation.

    Returns
    -------
    models : dict
        Dictionary of model_name -> model_instance or 'special' for custom handling.
    """
    # Convert rates to slopes for scenario-specific bounds
    true_b = pcreg.learning_rate_to_slope(learning_rate)
    true_c = pcreg.learning_rate_to_slope(rate_effect)

    # Tight bounds: narrow region around true values
    tight_margin = 0.05
    tight_bounds = [
        (true_b - tight_margin, true_b + tight_margin),
        (true_c - tight_margin, true_c + tight_margin)
    ]

    # Generate wrong bounds per PAC methodology
    wrong_bounds = generate_wrong_bounds(true_b, true_c, a=wrong_constraint_a,
                                          random_state=random_state)

    # Correct loose bounds from config
    correct_loose_bounds = CONFIG['correct_loose_bounds']

    # Build penalized-constrained models
    if fit_in_unit_space:
        # Unit-space fitting with lc_func prediction function
        # For unit-space, we need 3 parameters: T1, b, c
        # Bounds: T1 > 0 (unconstrained upper), b in correct range, c in correct range
        unit_bounds = {'T1': (0, None), 'LC': correct_loose_bounds[0], 'RC': correct_loose_bounds[1]}
        unit_tight_bounds = {'T1': (0, None), 'LC': tight_bounds[0], 'RC': tight_bounds[1]}
        unit_wrong_bounds = {'T1': (0, None), 'LC': wrong_bounds[0], 'RC': wrong_bounds[1]}

        penalized_constrained_only = pcreg.PenalizedConstrainedRegression(
            bounds=unit_bounds,
            feature_names=['T1', 'LC', 'RC'],
            alpha=0.0,
            loss='sspe',
            prediction_fn=lc_prediction_fn,
            fit_intercept=False,
            init=[CONFIG['T1'], true_b, true_c]  # Initialize near true values for better convergence
        )

        # Good starting point for optimizer convergence
        init_params = [CONFIG['T1'], true_b, true_c]

        penalized_constrained_cv = pcreg.PenalizedConstrainedCV(
            bounds=unit_bounds,
            feature_names=['T1', 'LC', 'RC'],
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            prediction_fn=lc_prediction_fn,
            fit_intercept=False,
            init=init_params,
            cv=cv_folds,
            verbose=0
        )

        penalized_constrained_cv_tight = pcreg.PenalizedConstrainedCV(
            bounds=unit_tight_bounds,
            feature_names=['T1', 'LC', 'RC'],
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            prediction_fn=lc_prediction_fn,
            fit_intercept=False,
            init=init_params,
            cv=cv_folds,
            verbose=0
        )

        penalized_constrained_cv_wrong = pcreg.PenalizedConstrainedCV(
            bounds=unit_wrong_bounds,
            feature_names=['T1', 'LC', 'RC'],
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            prediction_fn=lc_prediction_fn,
            fit_intercept=False,
            init=init_params,
            cv=cv_folds,
            verbose=0
        )
    else:
        # Log-linear (fit) space - standard approach
        penalized_constrained_only = pcreg.PenalizedConstrainedRegression(
            bounds=correct_loose_bounds,
            alpha=0.0,
            loss='sspe'
        )

        penalized_constrained_cv = pcreg.PenalizedConstrainedCV(
            bounds=correct_loose_bounds,
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            cv=cv_folds,
            verbose=0
        )

        penalized_constrained_cv_tight = pcreg.PenalizedConstrainedCV(
            bounds=tight_bounds,
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            cv=cv_folds,
            verbose=0
        )

        penalized_constrained_cv_wrong = pcreg.PenalizedConstrainedCV(
            bounds=wrong_bounds,
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            cv=cv_folds,
            verbose=0
        )

    return {
        'OLS': LinearRegression(),

        'OLS_LearnOnly': 'special',  # Handled separately (drops rate variable)

        'RidgeCV': RidgeCV(
            alphas=np.logspace(-3, 3, 20),
            cv=cv_folds
        ),

        'LassoCV': LassoCV(
            alphas=np.logspace(-3, 0, 20),
            cv=cv_folds,
            max_iter=5000
        ),

        'BayesianRidge': BayesianRidge(),

        'PLS': PLSRegression(n_components=2, scale=False),

        'PCA_Linear': Pipeline([
            ('pca', PCA(n_components=2)),
            ('linear', LinearRegression())
        ]),

        'PCRegConstrain_Only': penalized_constrained_only,

        'PCRegCV': penalized_constrained_cv,

        'PCRegCV_Tight': penalized_constrained_cv_tight,

        'PCRegCV_Wrong': penalized_constrained_cv_wrong,
    }


def fit_single_model(model_name: str, model, X: np.ndarray, y: np.ndarray,
                     b_true: float, c_true: float, T1_true: float,
                     fit_in_unit_space: bool = False) -> Tuple[Dict, Any]:
    """
    Fit a single model and return results plus the fitted model.

    Parameters
    ----------
    model_name : str
        Name of the model being fit.
    model : estimator or str
        Model instance or 'special' for custom handling.
    X : ndarray
        Feature matrix (log-transformed for fit space, raw for unit space).
    y : ndarray
        Target values (log-transformed for fit space, raw for unit space).
    b_true : float
        True learning slope.
    c_true : float
        True rate slope.
    T1_true : float
        True theoretical first unit cost.
    fit_in_unit_space : bool
        If True, penalized-constrained models are fit in unit space.

    Returns
    -------
    result : dict
        Dictionary containing fit results and metrics.
    fitted_model : estimator or None
        The fitted model for making predictions, or None if fitting failed.
    """
    result = {
        'model_name': model_name,
        'converged': True,
        'fit_time': 0.0
    }
    fitted_model = None

    start_time = time.time()

    try:
        if model_name == 'OLS_LearnOnly':
            # Use only first column (learning variable)
            model = LinearRegression()
            model.fit(X[:, [0]], y)
            result['LC_est'] = np.log(model.coef_[0])/ np.log(2)  # Convert coef to learning slope
            result['RC_est'] = 1  # Not estimated
            result['intercept'] = model.intercept_
            result['T1_est'] = np.exp(model.intercept_)  # exp(intercept) = T1
            result['r2'] = model.score(X[:, [0]], y)
            # Store for compatibility
            result['b'] = model.coef_[0]
            result['c'] = 0.0
            fitted_model = model

        elif model_name == 'PLS':
            model.fit(X, y)
            # PLS coef_ shape is (n_features, n_targets) - flatten for single target
            coef = model.coef_.ravel()
            result['LC_est'] = np.log(coef[0])/ np.log(2)  # Convert coef to learning slope
            result['RC_est'] = np.log(coef[1]) / np.log(2) if len(coef) > 1 else 0.0
            result['intercept'] = 0.0
            result['T1_est'] = np.nan  # PLS doesn't give a clean T1 estimate
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred.ravel()) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            result['b'] = coef[0]
            result['c'] = coef[1] if len(coef) > 1 else 0.0
            fitted_model = model

        elif model_name == 'PCA_Linear':
            # Pipeline: extract coefficients from the linear regression step
            model.fit(X, y)
            # Get the linear regression from the pipeline
            linear_model = model.named_steps['linear']
            pca_model = model.named_steps['pca']
            # Transform PCA coefficients back to original space
            coef_pca_space = linear_model.coef_
            coef_original = pca_model.components_.T @ coef_pca_space
            result['LC_est'] = np.log(coef_original[0]) / np.log(2)  # Convert coef to learning slope
            result['RC_est'] = np.log(coef_original[1]) / np.log(2) if len(coef_original) > 1 else 0.0
            result['intercept'] = linear_model.intercept_
            result['T1_est'] = np.exp(linear_model.intercept_)
            result['r2'] = model.score(X, y)
            result['b'] = coef_original[0]
            result['c'] = coef_original[1] if len(coef_original) > 1 else 0.0
            fitted_model = model

        else:
            model.fit(X, y)

            # Extract coefficients based on model type
            if model_name in PCREG_MODELS and fit_in_unit_space:
                # Unit-space models have [T1, b, c] as coef_ where b,c are slopes directly
                result['T1_est'] = model.coef_[0]
                result['LC_est'] = 2**model.coef_[1]  # This IS the slope (e.g., -0.152)
                result['RC_est'] = 2**model.coef_[2]  # This IS the slope (e.g., -0.234)
                result['b'] = model.coef_[1]  # Slope, not learning rate
                result['c'] = model.coef_[2]  # Slope, not rate effect
                result['intercept'] = 0.0
            else:
                result['LC_est'] = 2**model.coef_[1]  # This IS the slope (e.g., -0.152)
                result['RC_est'] = 2**model.coef_[2]  # This IS the slope (e.g., -0.234)
                result['b'] = model.coef_[1]  # Slope, not learning rate
                result['c'] = model.coef_[2]  # Slope, not rate effect


                if hasattr(model, 'intercept_'):
                    result['intercept'] = model.intercept_
                    result['T1_est'] = np.exp(model.intercept_)
                else:
                    result['intercept'] = 0.0
                    result['T1_est'] = np.nan

            result['r2'] = model.score(X, y)
            fitted_model = model

            # Get regularization parameters if available
            if hasattr(model, 'alpha_'):
                result['alpha'] = model.alpha_
            if hasattr(model, 'l1_ratio_'):
                result['l1_ratio'] = model.l1_ratio_
            if hasattr(model, 'converged_'):
                result['converged'] = model.converged_

    except Exception as e:
        result['converged'] = False
        result['error'] = str(e)
        result['LC_est'] = np.nan
        result['RC_est'] = np.nan
        result['T1_est'] = np.nan
        result['intercept'] = np.nan
        result['r2'] = np.nan
        result['b'] = np.nan
        result['c'] = np.nan

    result['fit_time'] = time.time() - start_time

    # Compute coefficient errors
    if not np.isnan(result['LC_est']):
        result['b_error'] = abs(result['b'] - b_true)
        result['c_error'] = abs(result['c'] - c_true)
        result['b_correct_sign'] = result['b'] <= 0
        result['c_correct_sign'] = result['c'] <= 0
        if not np.isnan(result['T1_est']):
            result['T1_error'] = abs(result['T1_est'] - T1_true)
            result['T1_pct_error'] = abs(result['T1_est'] - T1_true) / T1_true
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

    return result, fitted_model


def predict_in_unit_space(
    fitted_model,
    model_name: str,
    X_test_log: np.ndarray,
    X_test_original: np.ndarray,
    fit_in_unit_space: bool = False
) -> np.ndarray:
    """
    Generate predictions in unit space from a fitted model.

    Parameters
    ----------
    fitted_model : estimator
        Fitted model to use for predictions.
    model_name : str
        Name of the model.
    X_test_log : ndarray
        Test features in log space (for log-linear models).
    X_test_original : ndarray
        Test features in original space (for unit-space models).
    fit_in_unit_space : bool
        If True, PCReg models predict directly in unit space.

    Returns
    -------
    y_pred_unit : ndarray
        Predictions in unit space.
    """
    if model_name == 'OLS_LearnOnly':
        # Only uses first column (learning variable)
        y_pred_log = fitted_model.predict(X_test_log[:, [0]])
        return np.exp(y_pred_log.ravel())
    elif fit_in_unit_space and model_name in PCREG_MODELS:
        # Unit-space models predict directly in unit space using original X
        return fitted_model.predict(X_test_original).ravel()
    else:
        # Log-linear models: predict in log space, transform to unit
        y_pred_log = fitted_model.predict(X_test_log)
        return np.exp(y_pred_log.ravel())


def compute_oos_metrics(
    y_pred_unit: np.ndarray,
    y_test_true: np.ndarray
) -> Dict:
    """
    Compute out-of-sample prediction metrics.

    Parameters
    ----------
    y_pred_unit : ndarray
        Predicted costs in unit space.
    y_test_true : ndarray
        True costs in unit space.

    Returns
    -------
    metrics : dict
        Dictionary with test_sspe, test_mape, test_mse.
    """
    # Ensure arrays are 1D
    y_pred_unit = np.asarray(y_pred_unit).ravel()
    y_test_true = np.asarray(y_test_true).ravel()

    # Compute metrics in unit space
    residuals = y_test_true - y_pred_unit
    pct_errors = residuals / y_test_true

    # SSPE: Sum of Squared Percentage Errors
    sspe = np.sum(pct_errors ** 2)

    # MAPE: Mean Absolute Percentage Error (not multiplied by 100)
    mape = np.mean(np.abs(pct_errors))

    # MSE: Mean Squared Error in unit space
    mse = np.mean(residuals ** 2)

    return {
        'test_sspe': sspe,
        'test_mape': mape,
        'test_mse': mse,
        'test_n_lots': len(y_test_true)
    }


def run_single_scenario(
    n_lots: int,
    correlation: float,
    cv_error: float,
    learning_rate: float,
    rate_effect: float,
    replication: int,
    config: dict
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run all models on a single scenario with out-of-sample testing.

    Returns
    -------
    results : list of dict
        Model fit results with in-sample and out-of-sample metrics.
    predictions : list of dict
        Lot-level predictions for each model (for parquet output).
    """
    # Convert rates to slopes
    b_true = pcreg.learning_rate_to_slope(learning_rate)
    c_true = pcreg.learning_rate_to_slope(rate_effect)
    T1_true = config['T1']

    # Generate seed deterministically
    seed = config['base_seed'] + hash((n_lots, correlation, cv_error,
                                        learning_rate, rate_effect, replication)) % (2**31)

    # Generate training data
    data = pcreg.generate_correlated_learning_data(
        n_lots=n_lots,
        T1=T1_true,
        b=b_true,
        c=c_true,
        target_correlation=correlation,
        cv_error=cv_error,
        random_state=seed
    )

    X_log, y_log = data['X'], data['y']
    X_original, y_original = data['X_original'], data['y_original']
    actual_corr = data['actual_correlation']

    # Get fit_in_unit_space setting
    fit_in_unit_space = config.get('fit_in_unit_space', False)

    # Select appropriate X and y based on fitting space
    # PCReg with unit-space fitting needs original (non-log) data
    # All other models use log-transformed data
    X_train = X_log
    y_train = y_log

    # Generate test data for out-of-sample evaluation
    test_n_lots = config.get('test_n_lots', 5)
    test_quantity_multiplier = config.get('test_quantity_multiplier', 2)

    # Test quantity = max training quantity * multiplier
    max_train_quantity = int(np.max(data['lot_quantities']))
    test_quantity = int(max_train_quantity * test_quantity_multiplier)

    # First test unit starts after last training unit
    first_test_unit = int(data['last_units'][-1]) + 1

    test_data = pcreg.generate_test_data(
        first_unit_start=first_test_unit,
        n_lots=test_n_lots,
        base_quantity=test_quantity,
        T1=T1_true,
        b=b_true,
        c=c_true,
        cv_error=0.0,  # Use true values for test (no error)
        random_state=seed + 1000  # Different seed for test data
    )

    X_test_log = test_data['X']  # Log space
    X_test_original = test_data['X_original']  # Original space for unit-space models
    y_test_true = test_data['true_costs']  # Unit space true values

    # Get models with scenario-specific bounds
    models = get_models(
        learning_rate=learning_rate,
        rate_effect=rate_effect,
        cv_folds=config['cv_folds'],
        fit_in_unit_space=fit_in_unit_space,
        wrong_constraint_a=config.get('wrong_constraint_a', 0.5),
        random_state=seed  # Use same seed for reproducible wrong bounds
    )

    # Fit all models and compute OOS metrics
    results = []
    predictions = []
    save_predictions = config.get('save_predictions', False)

    for model_name, model in models.items():
        # Select training data based on model type
        # PCReg models with unit-space fitting need original (non-log) data
        if fit_in_unit_space and model_name in PCREG_MODELS:
            X_fit, y_fit = X_original, y_original
        else:
            X_fit, y_fit = X_train, y_train

        result, fitted_model = fit_single_model(
            model_name, model, X_fit, y_fit, b_true, c_true, T1_true,
            fit_in_unit_space=fit_in_unit_space
        )

        # Compute out-of-sample metrics and predictions
        if fitted_model is not None:
            try:
                # Get predictions in unit space (single call, used for both metrics and parquet)
                y_pred_unit = predict_in_unit_space(
                    fitted_model=fitted_model,
                    model_name=model_name,
                    X_test_log=X_test_log,
                    X_test_original=X_test_original,
                    fit_in_unit_space=fit_in_unit_space
                )

                # Compute OOS metrics
                oos_metrics = compute_oos_metrics(y_pred_unit, y_test_true)
                result.update(oos_metrics)

                # Collect lot-level predictions for parquet output (vectorized)
                if save_predictions:
                    prediction_errors = y_pred_unit - y_test_true
                    pct_errors = prediction_errors / y_test_true

                    # Build all records at once using list comprehension with pre-computed arrays
                    lot_predictions = [
                        {
                            'model_name': model_name,
                            'n_lots': n_lots,
                            'target_correlation': correlation,
                            'cv_error': cv_error,
                            'learning_rate': learning_rate,
                            'rate_effect': rate_effect,
                            'replication': replication,
                            'seed': seed,
                            'test_lot_number': i + 1,
                            'test_lot_midpoint': test_data['lot_midpoints'][i],
                            'test_lot_quantity': test_data['lot_quantities'][i],
                            'true_cost': y_test_true[i],
                            'predicted_cost': y_pred_unit[i],
                            'prediction_error': prediction_errors[i],
                            'pct_error': pct_errors[i]
                        }
                        for i in range(test_n_lots)
                    ]
                    predictions.extend(lot_predictions)

            except Exception as e:
                # Prediction failed
                result.update({
                    'test_sspe': np.nan,
                    'test_mape': np.nan,
                    'test_mse': np.nan,
                    'test_n_lots': 0,
                    'test_error': str(e)
                })
        else:
            # Model failed to fit, add NaN OOS metrics
            result.update({
                'test_sspe': np.nan,
                'test_mape': np.nan,
                'test_mse': np.nan,
                'test_n_lots': 0
            })

        # Add scenario metadata
        result.update({
            'n_lots': n_lots,
            'target_correlation': correlation,
            'actual_correlation': actual_corr,
            'cv_error': cv_error,
            'learning_rate': learning_rate,
            'rate_effect': rate_effect,
            'b_true': b_true,
            'c_true': c_true,
            'T1_true': T1_true,
            'replication': replication,
            'seed': seed,
            'fit_in_unit_space': fit_in_unit_space,
            'test_quantity': test_quantity,
            'first_test_unit': first_test_unit
        })

        results.append(result)

    return results, predictions


def _run_scenario_wrapper(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """
    Wrapper function for parallel execution of run_single_scenario.

    This function is defined at module level to be picklable for multiprocessing.

    Returns
    -------
    results : list of dict
        Model fit results with metrics.
    predictions : list of dict
        Lot-level predictions for parquet output.
    """
    n_lots, corr, cv_error, lr, re, rep, config = args
    return run_single_scenario(n_lots, corr, cv_error, lr, re, rep, config)


def make_scenario_key(n_lots: int, correlation: float, cv_error: float,
                      learning_rate: float, rate_effect: float, replication: int) -> tuple:
    """
    Create a unique key tuple for a scenario.

    Returns a tuple that can be used for set membership testing.
    """
    return (n_lots, correlation, cv_error, learning_rate, rate_effect, replication)


def load_completed_scenarios(output_dir: Path) -> set:
    """
    Load scenario keys that have already been computed from existing results.

    Parameters
    ----------
    output_dir : Path
        Directory containing simulation results.

    Returns
    -------
    completed : set
        Set of scenario key tuples that have already been computed.
    """
    results_file = output_dir / 'simulation_results.parquet'

    if not results_file.exists():
        return set()

    try:
        df = pd.read_parquet(results_file)
        if df.empty:
            return set()

        # Extract unique scenarios (each scenario has multiple models)
        # Use the first model's row for each scenario
        scenario_cols = list(SCENARIO_KEY_COLS)

        # Check all required columns exist
        missing_cols = [c for c in scenario_cols if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in existing results: {missing_cols}")
            return set()

        completed = set()
        for _, row in df[scenario_cols].drop_duplicates().iterrows():
            key = make_scenario_key(
                n_lots=row['n_lots'],
                correlation=row['target_correlation'],
                cv_error=row['cv_error'],
                learning_rate=row['learning_rate'],
                rate_effect=row['rate_effect'],
                replication=row['replication']
            )
            completed.add(key)

        return completed

    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set()


def save_batch_results(
    results: List[Dict],
    predictions: List[Dict],
    output_dir: Path,
    batch_num: int,
    append: bool = True
):
    """
    Save a batch of results incrementally.

    Parameters
    ----------
    results : list of dict
        Model fit results for this batch.
    predictions : list of dict
        Lot-level predictions for this batch.
    output_dir : Path
        Output directory.
    batch_num : int
        Batch number for logging.
    append : bool
        If True, append to existing file; otherwise overwrite.
    """
    results_file = output_dir / 'simulation_results.parquet'
    predictions_file = output_dir / 'predictions_flat.parquet'

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()

    if append and results_file.exists():
        # Load existing and concatenate
        existing_df = pd.read_parquet(results_file)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    # Save results
    results_df.to_parquet(results_file, engine='pyarrow', index=False)

    # Save predictions if we have them
    if not predictions_df.empty:
        if append and predictions_file.exists():
            existing_pred = pd.read_parquet(predictions_file)
            predictions_df = pd.concat([existing_pred, predictions_df], ignore_index=True)
        predictions_df.to_parquet(predictions_file, engine='pyarrow', index=False)

    print(f"  Batch {batch_num} saved: {len(results)} model fits")


def run_full_simulation(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete factorial simulation with batch processing and resume capability.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - n_jobs: Number of parallel workers (-1 = all CPUs, 1 = sequential)
        - save_predictions: Whether to collect lot-level predictions
        - batch_size: Number of scenarios to process before saving (default 100)
        - resume: If True, skip scenarios that have already been computed
        - All other simulation parameters

    Returns
    -------
    results_df : pd.DataFrame
        Results dataframe with all model fits across all scenarios.
    predictions_df : pd.DataFrame
        Lot-level predictions for parquet output (empty if save_predictions=False).
    """
    output_dir = config['output_dir']
    output_dir.mkdir(exist_ok=True)

    # Calculate total scenarios
    n_scenarios_total = (
        len(config['sample_sizes']) *
        len(config['correlations']) *
        len(config['cv_errors']) *
        len(config['learning_rates']) *
        len(config['rate_effects']) *
        config['n_replications']
    )
    # Get model count using first scenario's parameters
    n_models = len(get_models(
        learning_rate=config['learning_rates'][0],
        rate_effect=config['rate_effects'][0],
        cv_folds=config['cv_folds']
    ))

    save_predictions = config.get('save_predictions', False)
    batch_size = config.get('batch_size', 100)
    resume = config.get('resume', True)

    # Build list of all scenario arguments
    factor_combinations = list(product(
        config['sample_sizes'],
        config['correlations'],
        config['cv_errors'],
        config['learning_rates'],
        config['rate_effects'],
        range(config['n_replications'])
    ))

    # Check for completed scenarios if resuming
    completed_scenarios = set()
    if resume:
        completed_scenarios = load_completed_scenarios(output_dir)
        if completed_scenarios:
            print(f"Resuming: Found {len(completed_scenarios)} completed scenarios")

    # Filter out completed scenarios
    scenarios_to_run = []
    for n_lots, corr, cv, lr, re, rep in factor_combinations:
        key = make_scenario_key(n_lots, corr, cv, lr, re, rep)
        if key not in completed_scenarios:
            scenarios_to_run.append((n_lots, corr, cv, lr, re, rep, config))

    n_scenarios = len(scenarios_to_run)
    n_skipped = n_scenarios_total - n_scenarios

    print(f"Total scenarios: {n_scenarios_total}")
    print(f"Already completed: {n_skipped}")
    print(f"Scenarios to run: {n_scenarios}")
    print(f"Models per scenario: {n_models}")
    print(f"Total model fits remaining: {n_scenarios * n_models}")
    print(f"Batch size: {batch_size}")
    print(f"Save predictions: {save_predictions}")
    print()

    if n_scenarios == 0:
        print("All scenarios already completed!")
        # Load and return existing results
        results_file = output_dir / 'simulation_results.parquet'
        predictions_file = output_dir / 'predictions_flat.parquet'
        results_df = pd.read_parquet(results_file) if results_file.exists() else pd.DataFrame()
        predictions_df = pd.read_parquet(predictions_file) if predictions_file.exists() else pd.DataFrame()
        return results_df, predictions_df

    start_time = time.time()
    batch_results = []
    batch_predictions = []
    batch_num = 0
    scenarios_processed = 0

    # Process scenarios in batches
    for i, args in enumerate(scenarios_to_run):
        # Run scenario
        results, predictions = _run_scenario_wrapper(args)
        batch_results.extend(results)
        if save_predictions:
            batch_predictions.extend(predictions)
        scenarios_processed += 1

        # Save batch when full or at end
        if len(batch_results) >= batch_size * n_models or i == len(scenarios_to_run) - 1:
            batch_num += 1
            save_batch_results(
                results=batch_results,
                predictions=batch_predictions,
                output_dir=output_dir,
                batch_num=batch_num,
                append=True
            )
            batch_results = []
            batch_predictions = []

        # Progress update
        if (i + 1) % 10 == 0 or i == len(scenarios_to_run) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_scenarios - i - 1) / rate if rate > 0 else 0
            total_done = n_skipped + i + 1
            print(f"Progress: {total_done}/{n_scenarios_total} ({100*total_done/n_scenarios_total:.1f}%) "
                  f"- ETA: {remaining/60:.1f} min"
                  f" - Elapsed: {elapsed/60:.1f} min")

    # Load final results
    results_file = output_dir / 'simulation_results.parquet'
    predictions_file = output_dir / 'predictions_flat.parquet'
    results_df = pd.read_parquet(results_file) if results_file.exists() else pd.DataFrame()
    predictions_df = pd.read_parquet(predictions_file) if predictions_file.exists() else pd.DataFrame()

    elapsed_total = time.time() - start_time
    print(f"\nTotal time for this run: {elapsed_total/60:.1f} minutes")
    if n_scenarios > 0:
        print(f"Average per scenario: {elapsed_total/n_scenarios:.3f} seconds")

    return results_df, predictions_df


def analyze_results(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics from simulation results.
    """
    analysis = {}

    # Filter to converged models only
    df_conv = df[df['converged'] == True].copy()

    # Check if OOS metrics are available
    has_oos = 'test_sspe' in df_conv.columns

    # Overall statistics by method
    agg_dict = {
        'b_error': ['mean', 'std'],
        'c_error': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'b_correct_sign': 'mean',
        'c_correct_sign': 'mean',
        'fit_time': 'mean',
        'converged': 'sum'
    }

    # Add OOS metrics if available
    if has_oos:
        agg_dict['test_sspe'] = ['mean', 'std']
        agg_dict['test_mape'] = ['mean', 'std']
        agg_dict['test_mse'] = ['mean', 'std']

    # Add T1 error if available
    if 'T1_pct_error' in df_conv.columns:
        agg_dict['T1_pct_error'] = ['mean', 'std']

    method_stats = df_conv.groupby('model_name').agg(agg_dict).round(4)

    analysis['method_stats'] = method_stats

    # Winner analysis - which method has lowest error most often
    winners = {}
    metrics_to_analyze = ['b_error', 'c_error']

    # Use test_sspe as primary winner metric if available
    if has_oos:
        metrics_to_analyze.append('test_sspe')

    for metric in metrics_to_analyze:
        # Group by scenario (excluding model_name)
        scenario_cols = ['n_lots', 'target_correlation', 'cv_error',
                        'learning_rate', 'rate_effect', 'replication']

        win_counts = {}
        for name, group in df_conv.groupby(scenario_cols):
            if len(group) > 0:
                # Filter out NaN values for this metric
                group_valid = group.dropna(subset=[metric])
                if len(group_valid) > 0:
                    winner = group_valid.loc[group_valid[metric].idxmin(), 'model_name']
                    win_counts[winner] = win_counts.get(winner, 0) + 1

        winners[metric] = win_counts

    analysis['winners'] = winners

    # Statistics by design factors
    factor_agg = {'b_error': 'mean', 'c_error': 'mean', 'r2': 'mean'}
    if has_oos:
        factor_agg['test_sspe'] = 'mean'
        factor_agg['test_mape'] = 'mean'

    for factor in ['n_lots', 'target_correlation', 'cv_error']:
        factor_stats = df_conv.groupby(['model_name', factor]).agg(factor_agg).round(4)
        analysis[f'by_{factor}'] = factor_stats

    return analysis


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create and save visualization plots.
    """
    df_conv = df[df['converged'] == True].copy()

    # Figure 1: Overall method comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods_order = ['OLS', 'RidgeCV', 'LassoCV', 'BayesianRidge',
                     'PenalizedConstrainedOnly', 'PenalizedConstrainedCV',
                     'PenalizedConstrainedCV_Tight']

    # Filter to main methods
    df_main = df_conv[df_conv['model_name'].isin(methods_order)]

    for ax, metric, title in zip(
        axes,
        ['b_error', 'c_error', 'r2'],
        ['Learning Slope Error', 'Rate Slope Error', 'R²']
    ):
        data = [df_main[df_main['model_name'] == m][metric].dropna() for m in methods_order]
        bp = ax.boxplot(data, labels=methods_order, patch_artist=True)
        ax.set_xticklabels(methods_order, rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Color penalized-constrained methods
        colors = ['lightblue' if 'PenalizedConstrained' not in m else 'lightgreen' for m in methods_order]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Effect of correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ['b_error', 'c_error'], ['LC Slope Error', 'RC Slope Error']):
        for method in ['OLS', 'RidgeCV', 'PenalizedConstrainedCV']:
            df_m = df_conv[df_conv['model_name'] == method]
            means = df_m.groupby('target_correlation')[metric].mean()
            ax.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Target Correlation')
        ax.set_ylabel(f'Mean {title}')
        ax.set_title(f'{title} vs Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correlation_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Effect of sample size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ['b_error', 'c_error'], ['LC Slope Error', 'RC Slope Error']):
        for method in ['OLS', 'RidgeCV', 'PenalizedConstrainedCV']:
            df_m = df_conv[df_conv['model_name'] == method]
            means = df_m.groupby('n_lots')[metric].mean()
            ax.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sample Size (N lots)')
        ax.set_ylabel(f'Mean {title}')
        ax.set_title(f'{title} vs Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'sample_size_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Correct sign rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sign_rates = df_conv.groupby('model_name').agg({
        'b_correct_sign': 'mean',
        'c_correct_sign': 'mean'
    })
    
    x = np.arange(len(sign_rates))
    width = 0.35
    
    ax.bar(x - width/2, sign_rates['b_correct_sign'] * 100, width, label='LC slope (b)', color='steelblue')
    ax.bar(x + width/2, sign_rates['c_correct_sign'] * 100, width, label='RC slope (c)', color='coral')
    
    ax.set_ylabel('Correct Sign Rate (%)')
    ax.set_title('Percentage of Simulations with Correct Coefficient Sign')
    ax.set_xticks(x)
    ax.set_xticklabels(sign_rates.index, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correct_sign_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def print_summary(df: pd.DataFrame, analysis: Dict):
    """
    Print summary of simulation results.
    """
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal observations: {len(df)}")
    print(f"Converged: {df['converged'].sum()} ({100*df['converged'].mean():.1f}%)")

    print("\n" + "-"*80)
    print("OVERALL METHOD PERFORMANCE (mean ± std)")
    print("-"*80)

    stats = analysis['method_stats']

    # Check if OOS metrics are available
    has_oos = ('test_sspe', 'mean') in stats.columns

    if has_oos:
        print(f"{'Method':<18} {'b_err':>10} {'c_err':>10} {'R²':>8} {'SSPE':>10} {'MAPE':>8}")
        print("-"*80)

        for method in stats.index:
            row = stats.loc[method]
            b_err = f"{row[('b_error', 'mean')]:.4f}"
            c_err = f"{row[('c_error', 'mean')]:.4f}"
            r2 = f"{row[('r2', 'mean')]:.3f}"
            sspe = f"{row[('test_sspe', 'mean')]:.4f}"
            mape = f"{row[('test_mape', 'mean')]:.4f}"
            print(f"{method:<18} {b_err:>10} {c_err:>10} {r2:>8} {sspe:>10} {mape:>8}")
    else:
        print(f"{'Method':<20} {'b error':>12} {'c error':>12} {'R²':>10} {'b sign%':>10} {'c sign%':>10}")
        print("-"*80)

        for method in stats.index:
            row = stats.loc[method]
            b_err = f"{row[('b_error', 'mean')]:.4f}±{row[('b_error', 'std')]:.4f}"
            c_err = f"{row[('c_error', 'mean')]:.4f}±{row[('c_error', 'std')]:.4f}"
            r2 = f"{row[('r2', 'mean')]:.3f}"
            b_sign = f"{100*row[('b_correct_sign', 'mean')]:.0f}%"
            c_sign = f"{100*row[('c_correct_sign', 'mean')]:.0f}%"
            print(f"{method:<20} {b_err:>12} {c_err:>12} {r2:>10} {b_sign:>10} {c_sign:>10}")

    # Print estimated parameter accuracy if available
    if 'T1_pct_error' in df.columns:
        print("\n" + "-"*80)
        print("PARAMETER ESTIMATION ACCURACY")
        print("-"*80)
        print(f"{'Method':<18} {'T1_%err':>10} {'LC_err':>10} {'RC_err':>10}")
        for method in stats.index:
            row = stats.loc[method]
            t1_err = f"{row[('T1_pct_error', 'mean')]:.4f}" if ('T1_pct_error', 'mean') in row.index else "N/A"
            b_err = f"{row[('b_error', 'mean')]:.4f}"
            c_err = f"{row[('c_error', 'mean')]:.4f}"
            print(f"{method:<18} {t1_err:>10} {b_err:>10} {c_err:>10}")

    print("\n" + "-"*80)
    print("WINNER COUNTS (lowest error)")
    print("-"*80)

    for metric, counts in analysis['winners'].items():
        metric_label = metric
        if metric == 'test_sspe':
            metric_label = 'test_sspe (OOS prediction)'
        print(f"\n{metric_label}:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for method, count in sorted_counts[:5]:
            print(f"  {method}: {count}")

    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("SIMULATION STUDY: Penalized-Constrained Regression")
    print("ICEAA 2026 Paper")
    print("="*80)
    print()
    print("Configuration:")
    print(f"  Sample sizes: {CONFIG['sample_sizes']}")
    print(f"  Correlations: {CONFIG['correlations']}")
    print(f"  CV errors: {CONFIG['cv_errors']}")
    print(f"  Learning rates: {CONFIG['learning_rates']}")
    print(f"  Rate effects: {CONFIG['rate_effects']}")
    print(f"  Replications: {CONFIG['n_replications']}")
    print(f"  Test lots: {CONFIG['test_n_lots']}")
    print(f"  Test quantity multiplier: {CONFIG['test_quantity_multiplier']}x")
    print(f"  Save predictions: {CONFIG['save_predictions']}")
    print()

    # Create output directory
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(exist_ok=True)

    # Run simulation
    print("Starting simulation...")
    print()

    results_df, predictions_df = run_full_simulation(CONFIG)

    # Save raw results as parquet (more efficient than CSV)
    results_df.to_parquet(output_dir / 'simulation_results.parquet', engine='pyarrow', index=False)
    print(f"\nResults saved to: {output_dir / 'simulation_results.parquet'}")

    # Overall winners by test_sspe
    print("\n" + "="*80)
    print("OVERALL WINNERS BY TEST_SSPE")
    print("="*80)
    print(results_df.groupby('model_name')['test_sspe'].mean().sort_values(ascending=True).reset_index())

    # Winners by CV error
    print("\n" + "="*80)
    print("WINNERS BY CV ERROR")
    print("="*80)
    cv_winners = results_df.groupby(['cv_error', 'model_name'])['test_sspe'].mean().reset_index()
    cv_winners = cv_winners.sort_values(by=['cv_error', 'test_sspe'], ascending=True)
    print(cv_winners)

    # Winners by correlation
    print("\n" + "="*80)
    print("WINNERS BY CORRELATION")
    print("="*80)
    corr_winners = results_df.groupby(['target_correlation', 'model_name'])['test_sspe'].mean().reset_index()
    corr_winners = corr_winners.sort_values(by=['target_correlation', 'test_sspe'], ascending=True)
    print(corr_winners)

    # Winners by rate effect (RC slope)
    print("\n" + "="*80)
    print("WINNERS BY RATE EFFECT (RC SLOPE)")
    print("="*80)
    rc_winners = results_df.groupby(['rate_effect', 'model_name'])['test_sspe'].mean().reset_index()
    rc_winners = rc_winners.sort_values(by=['rate_effect', 'test_sspe'], ascending=True)
    print(rc_winners)


    # Save predictions to partitioned parquet if enabled
    if CONFIG['save_predictions'] and not predictions_df.empty:
        #predictions_path = output_dir / 'predictions'
        #predictions_path.mkdir(exist_ok=True)

        # Save as partitioned parquet by n_lots and model_name for efficient querying
        #predictions_df.to_parquet(
        #    predictions_path / 'predictions.parquet',
        #    engine='pyarrow',
        #    index=False,
        #    partition_cols=['n_lots', 'model_name']
        #)
        #print(f"Predictions saved to: {predictions_path}")

        # Also save a flat parquet for easy loading
        predictions_df.to_parquet(
            output_dir / 'predictions_flat.parquet',
            engine='pyarrow',
            index=False
        )
        print(f"Flat predictions saved to: {output_dir / 'predictions_flat.parquet'}")

    # Save config
    config_save = {k: str(v) if isinstance(v, (Path, np.ndarray)) else v
                   for k, v in CONFIG.items()}
    with open(output_dir / 'simulation_config.json', 'w') as f:
        json.dump(config_save, f, indent=2, default=str)

    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results_df)

    # Print summary
    print_summary(results_df, analysis)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, output_dir)

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)
