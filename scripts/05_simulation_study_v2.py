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
from itertools import product
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

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.metrics import r2_score


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Experimental design
    'sample_sizes': [5, 10, 30],
    'correlations': [0.0, 0.5, 0.9],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],
    'rate_effects': [0.80, 0.85, 0.90],
    'n_replications': 25,

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

    # Out-of-sample testing
    'test_n_lots': 5,
    'test_quantity_multiplier': 2,

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
    """Select specific columns from X."""
    def __init__(self, columns: List[int]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]


# ============================================================================
# SIMULATION MODEL BASE CLASS
# ============================================================================
class SimulationModel(BaseEstimator, RegressorMixin):
    """
    Base class for all simulation models.

    Handles:
    - X transforms via sklearn Pipeline (class attribute x_transforms)
    - y transforms via y_transform/y_inverse functions
    - Regressor instantiation and fitting
    - Coefficient extraction via get_coefficients()
    - Scenario-specific config passed through __init__

    Subclasses override class attributes and methods as needed.

    Attributes (class-level, override in subclasses)
    -------------------------------------------------
    x_transforms : list of (name, transformer) tuples
        Pipeline steps for X transformation. Default: log transform.
    y_transform : callable or None
        Function to transform y before fitting. Default: np.log.
    y_inverse : callable or None
        Function to inverse-transform predictions. Default: np.exp.
    regressor_class : class
        The sklearn-compatible regressor class to use.
    regressor_params : dict
        Default parameters for the regressor.
    """

    # Class-level defaults (override in subclasses)
    x_transforms: List[Tuple[str, Any]] = [('log', FunctionTransformer(np.log))]
    y_transform: Optional[Callable] = staticmethod(np.log)
    y_inverse: Optional[Callable] = staticmethod(np.exp)
    regressor_class: type = LinearRegression
    regressor_params: Dict[str, Any] = {}

    def __init__(self, **scenario_config):
        """
        Initialize with scenario-specific configuration.

        Parameters
        ----------
        **scenario_config : dict
            Scenario-specific parameters like bounds, alpha_grid, cv_folds, etc.
        """
        self.scenario_config = scenario_config

    def _build_regressor_params(self) -> Dict[str, Any]:
        """
        Build regressor parameters. Override for scenario-dependent params.

        Returns
        -------
        params : dict
            Parameters to pass to regressor_class.
        """
        return self.regressor_params.copy()

    def fit(self, X, y):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features in unit space.
        y : ndarray of shape (n_samples,)
            Target in unit space.

        Returns
        -------
        self
        """
        # Build X pipeline (create fresh instances)
        if self.x_transforms:
            steps = [(name, clone(t)) for name, t in self.x_transforms]
            self.x_pipeline_ = Pipeline(steps)
            X_t = self.x_pipeline_.fit_transform(X)
        else:
            self.x_pipeline_ = None
            X_t = X

        # Transform y
        if self.y_transform is not None:
            y_t = self.y_transform(y)
        else:
            y_t = y

        # Build and fit regressor
        params = self._build_regressor_params()
        self.regressor_ = self.regressor_class(**params)
        self.regressor_.fit(X_t, y_t)

        return self

    def predict(self, X):
        """
        Predict in unit space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features in unit space.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predictions in unit space.
        """
        # Transform X
        if self.x_pipeline_ is not None:
            X_t = self.x_pipeline_.transform(X)
        else:
            X_t = X

        # Predict
        y_pred = self.regressor_.predict(X_t)

        # Inverse transform predictions
        if self.y_inverse is not None:
            y_pred = self.y_inverse(y_pred)

        return np.asarray(y_pred).ravel()

    def score(self, X, y):
        """R² score in unit space."""
        return r2_score(y, self.predict(X))

    def get_coefficients(self) -> Dict[str, float]:
        """
        Extract model coefficients.

        Override in subclasses for non-standard coefficient extraction.

        Returns
        -------
        coefs : dict
            Dictionary with keys 'b', 'c', 'intercept', 'T1_est'.
        """
        coef = self.regressor_.coef_
        intercept = getattr(self.regressor_, 'intercept_', 0.0)
        return {
            'b': coef[0] if len(coef) > 0 else np.nan,
            'c': coef[1] if len(coef) > 1 else 0.0,
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }

    def get_regularization_params(self) -> Dict[str, Any]:
        """
        Extract regularization parameters if available.

        Returns
        -------
        params : dict
            Dictionary with 'alpha', 'l1_ratio', 'converged' if available.
        """
        params = {}
        if hasattr(self.regressor_, 'alpha_'):
            params['alpha'] = self.regressor_.alpha_
        if hasattr(self.regressor_, 'l1_ratio_'):
            params['l1_ratio'] = self.regressor_.l1_ratio_
        if hasattr(self.regressor_, 'converged_'):
            params['converged'] = self.regressor_.converged_
        return params


# ============================================================================
# LOG-SPACE MODEL SUBCLASSES
# ============================================================================
class OLS(SimulationModel):
    """Ordinary Least Squares in log-log space."""
    regressor_class = LinearRegression


class OLS_LearnOnly(SimulationModel):
    """OLS using only the learning variable (column 0)."""
    x_transforms = [
        ('select', ColumnSelector([0])),
        ('log', FunctionTransformer(np.log)),
    ]
    regressor_class = LinearRegression

    def get_coefficients(self) -> Dict[str, float]:
        intercept = getattr(self.regressor_, 'intercept_', 0.0)
        return {
            'b': self.regressor_.coef_[0],
            'c': 0.0,  # Not estimated
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }


class Ridge(SimulationModel):
    """Ridge regression with CV in log-log space."""
    regressor_class = RidgeCV

    def _build_regressor_params(self) -> Dict[str, Any]:
        return {
            'alphas': np.logspace(-3, 3, 20),
            'cv': self.scenario_config.get('cv_folds', 3),
        }


class Lasso(SimulationModel):
    """Lasso regression with CV in log-log space."""
    regressor_class = LassoCV

    def _build_regressor_params(self) -> Dict[str, Any]:
        return {
            'alphas': np.logspace(-3, 0, 20),
            'cv': self.scenario_config.get('cv_folds', 3),
            'max_iter': 5000,
        }


class BayesianRidgeModel(SimulationModel):
    """Bayesian Ridge regression in log-log space."""
    regressor_class = BayesianRidge


class PLS(SimulationModel):
    """Partial Least Squares in log-log space."""
    regressor_class = PLSRegression
    regressor_params = {'n_components': 2, 'scale': False}

    def get_coefficients(self) -> Dict[str, float]:
        coef = self.regressor_.coef_.ravel()
        return {
            'b': coef[0] if len(coef) > 0 else np.nan,
            'c': coef[1] if len(coef) > 1 else 0.0,
            'intercept': 0.0,
            'T1_est': np.nan,  # PLS doesn't give clean T1
        }

    def predict(self, X):
        # PLS returns 2D array, need to ravel
        if self.x_pipeline_ is not None:
            X_t = self.x_pipeline_.transform(X)
        else:
            X_t = X

        y_pred = self.regressor_.predict(X_t).ravel()

        if self.y_inverse is not None:
            y_pred = self.y_inverse(y_pred)

        return y_pred


class PCA_OLS(SimulationModel):
    """PCA + OLS in log-log space with coefficients transformed back."""
    x_transforms = [('log', FunctionTransformer(np.log))]
    regressor_class = LinearRegression

    def fit(self, X, y):
        # Log transform X
        if self.x_transforms:
            steps = [(name, clone(t)) for name, t in self.x_transforms]
            self.x_pipeline_ = Pipeline(steps)
            X_log = self.x_pipeline_.fit_transform(X)
        else:
            self.x_pipeline_ = None
            X_log = X

        # Transform y
        y_log = self.y_transform(y) if self.y_transform else y

        # PCA
        self.pca_ = PCA(n_components=2)
        X_pca = self.pca_.fit_transform(X_log)

        # Linear regression on PCA components
        self.regressor_ = LinearRegression()
        self.regressor_.fit(X_pca, y_log)

        # Transform coefficients back to original log-space
        self.coef_original_ = self.pca_.components_.T @ self.regressor_.coef_

        return self

    def predict(self, X):
        if self.x_pipeline_ is not None:
            X_log = self.x_pipeline_.transform(X)
        else:
            X_log = X

        X_pca = self.pca_.transform(X_log)
        y_log_pred = self.regressor_.predict(X_pca)

        if self.y_inverse is not None:
            return self.y_inverse(y_log_pred)
        return y_log_pred

    def get_coefficients(self) -> Dict[str, float]:
        intercept = self.regressor_.intercept_
        return {
            'b': self.coef_original_[0],
            'c': self.coef_original_[1] if len(self.coef_original_) > 1 else 0.0,
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }


# ============================================================================
# UNIT-SPACE PCREG MODEL SUBCLASSES
# ============================================================================
def unit_space_prediction_fn(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learning curve prediction function for unit-space fitting.

    Y = T1 * X1^b * X2^c

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Features where X[:, 0] is lot midpoint and X[:, 1] is rate variable.
    params : ndarray of shape (3,)
        Parameters [T1, b, c].

    Returns
    -------
    y_pred : ndarray
        Predicted costs in unit space.
    """
    T1, b, c = params[0], params[1], params[2]
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)


class PCRegBase(SimulationModel):
    """
    Base class for Penalized-Constrained Regression models.

    Fits in UNIT SPACE using custom prediction function.
    No X or y transforms needed.
    """
    x_transforms = []  # No X transform
    y_transform = None  # No y transform
    y_inverse = None  # No inverse needed

    # Subclasses set these
    bounds_key: Optional[str] = None  # 'correct', 'tight', or 'wrong'
    use_cv: bool = False
    selection_method: str = 'cv'  # 'cv', 'loocv', 'aic', 'aicc', 'bic', 'gcv'

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds based on bounds_key."""
        if self.bounds_key == 'correct':
            return self.scenario_config.get('correct_bounds', [(-0.5, 0), (-0.5, 0)])
        elif self.bounds_key == 'tight':
            return self.scenario_config['tight_bounds']
        elif self.bounds_key == 'wrong':
            return self.scenario_config['wrong_bounds']
        else:
            return [(-0.5, 0), (-0.5, 0)]

    def _build_regressor_params(self) -> Dict[str, Any]:
        """Build PCReg parameters for unit-space fitting."""
        bounds = self._get_bounds()

        # For unit space: bounds are [T1, b, c]
        # T1 > 0, b and c from config
        unit_bounds = [(0, None), bounds[0], bounds[1]]

        # Initial guess near true values for convergence
        b_true = self.scenario_config.get('b_true', -0.15)
        c_true = self.scenario_config.get('c_true', -0.23)
        T1_init = self.scenario_config.get('T1', 100)
        init = [T1_init, b_true, c_true]

        params = {
            'bounds': unit_bounds,
            'feature_names': ['T1', 'b', 'c'],
            'prediction_fn': unit_space_prediction_fn,
            'fit_intercept': False,
            'init': init,
            'loss': 'sspe',
        }

        if self.use_cv:
            params.update({
                'alphas': self.scenario_config.get('alpha_grid', np.logspace(-4, 0, 10)),
                'l1_ratios': self.scenario_config.get('l1_ratio_grid', [0.0, 0.5, 1.0]),
                'cv': self.scenario_config.get('cv_folds', 3),
                'selection': self.selection_method,
                'n_jobs': 1,  # Disable nested parallelism
                'verbose': 0,
            })
        else:
            params['alpha'] = 0.0

        return params

    @property
    def regressor_class(self):
        if self.use_cv:
            return pcreg.PenalizedConstrainedCV
        return pcreg.PenalizedConstrainedRegression

    def fit(self, X, y):
        """Fit directly in unit space."""
        # No transforms
        self.x_pipeline_ = None

        # Build and fit regressor
        params = self._build_regressor_params()
        if self.use_cv:
            self.regressor_ = pcreg.PenalizedConstrainedCV(**params)
        else:
            self.regressor_ = pcreg.PenalizedConstrainedRegression(**params)

        self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        """Predict in unit space."""
        return self.regressor_.predict(X).ravel()

    def get_coefficients(self) -> Dict[str, float]:
        """Extract T1, b, c from unit-space model."""
        coef = self.regressor_.coef_
        return {
            'b': coef[1],  # b is second param
            'c': coef[2],  # c is third param
            'intercept': 0.0,  # No intercept in unit-space model
            'T1_est': coef[0],  # T1 is first param
        }

    def get_regularization_params(self) -> Dict[str, Any]:
        params = {}
        if hasattr(self.regressor_, 'alpha_'):
            params['alpha'] = self.regressor_.alpha_
        if hasattr(self.regressor_, 'l1_ratio_'):
            params['l1_ratio'] = self.regressor_.l1_ratio_
        if hasattr(self.regressor_, 'converged_'):
            params['converged'] = self.regressor_.converged_
        else:
            params['converged'] = True
        return params


class PCReg_ConstrainOnly(PCRegBase):
    """PCReg with constraints only (no penalty), correct bounds."""
    bounds_key = 'correct'
    use_cv = False


class PCReg_CV(PCRegBase):
    """PCReg with K-fold CV for alpha/l1_ratio selection, correct bounds."""
    bounds_key = 'correct'
    use_cv = True
    selection_method = 'cv'


class PCReg_AICc(PCRegBase):
    """PCReg with AICc selection for alpha/l1_ratio, correct bounds.
    Better for small samples than CV (no data splitting required)."""
    bounds_key = 'correct'
    use_cv = True
    selection_method = 'aicc'


class PCReg_GCV(PCRegBase):
    """PCReg with GCV selection for alpha/l1_ratio, correct bounds.
    Approximates leave-one-out CV without data splitting."""
    bounds_key = 'correct'
    use_cv = True
    selection_method = 'gcv'


class PCReg_CV_Tight(PCRegBase):
    """PCReg CV with tight bounds around true values (oracle model)."""
    bounds_key = 'tight'
    use_cv = True
    selection_method = 'cv'


class PCReg_CV_Wrong(PCRegBase):
    """PCReg CV with wrong (perturbed) bounds."""
    bounds_key = 'wrong'
    use_cv = True
    selection_method = 'cv'


class PCRegMSEBase(PCRegBase):
    """
    PCReg with MSE loss instead of SSPE.

    Fits in UNIT SPACE with Y = T1 * X1^b * X2^c but minimizes MSE instead of SSPE.
    """
    loss_type: str = 'mse'  # 'mse' or 'sse'

    def _build_regressor_params(self) -> Dict[str, Any]:
        """Build PCReg parameters with MSE loss."""
        params = super()._build_regressor_params()
        params['loss'] = self.loss_type
        return params


class PCReg_MSE(PCRegMSEBase):
    """PCReg with MSE loss on Y (unit space), constraints only, no penalty."""
    bounds_key = 'correct'
    use_cv = False
    loss_type = 'mse'


class PCReg_MSE_CV(PCRegMSEBase):
    """PCReg with MSE loss on Y (unit space), CV-tuned penalty."""
    bounds_key = 'correct'
    use_cv = True
    selection_method = 'cv'
    loss_type = 'mse'


class PCRegLogSpaceBase(SimulationModel):
    """
    PCReg in LOG SPACE - like OLS but with constraints on b and c.

    Transforms: log(Y) = log(T1) + b*log(X1) + c*log(X2)
    This is a linear model in log space with constraints.
    """
    x_transforms = [('log', FunctionTransformer(np.log))]
    y_transform = staticmethod(np.log)
    y_inverse = staticmethod(np.exp)

    bounds_key: Optional[str] = 'correct'
    use_cv: bool = False

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for b and c (log-space coefficients)."""
        if self.bounds_key == 'correct':
            return self.scenario_config.get('correct_bounds', [(-0.5, 0), (-0.5, 0)])
        elif self.bounds_key == 'tight':
            return self.scenario_config['tight_bounds']
        elif self.bounds_key == 'wrong':
            return self.scenario_config['wrong_bounds']
        else:
            return [(-0.5, 0), (-0.5, 0)]

    def _build_regressor_params(self) -> Dict[str, Any]:
        """Build PCReg parameters for log-space fitting."""
        bounds = self._get_bounds()

        params = {
            'bounds': bounds,  # bounds on b and c
            'feature_names': ['b', 'c'],
            'fit_intercept': True,  # log(T1) is the intercept
            'intercept_bounds': (0, None),  # T1 > 0 means log(T1) can be anything, but typically positive
            'loss': 'mse',  # MSE on log(Y)
        }

        if self.use_cv:
            params.update({
                'alphas': self.scenario_config.get('alpha_grid', np.logspace(-4, 0, 10)),
                'l1_ratios': self.scenario_config.get('l1_ratio_grid', [0.0, 0.5, 1.0]),
                'cv': self.scenario_config.get('cv_folds', 3),
                'selection': 'cv',
                'n_jobs': 1,
                'verbose': 0,
            })
        else:
            params['alpha'] = 0.0

        return params

    def fit(self, X, y):
        """Fit in log space."""
        # Transform X
        if self.x_transforms:
            steps = [(name, clone(t)) for name, t in self.x_transforms]
            self.x_pipeline_ = Pipeline(steps)
            X_t = self.x_pipeline_.fit_transform(X)
        else:
            self.x_pipeline_ = None
            X_t = X

        # Transform y
        y_t = self.y_transform(y) if self.y_transform else y

        # Build and fit regressor
        params = self._build_regressor_params()
        if self.use_cv:
            self.regressor_ = pcreg.PenalizedConstrainedCV(**params)
        else:
            self.regressor_ = pcreg.PenalizedConstrainedRegression(**params)

        self.regressor_.fit(X_t, y_t)
        return self

    def predict(self, X):
        """Predict in unit space (transform back from log)."""
        if self.x_pipeline_ is not None:
            X_t = self.x_pipeline_.transform(X)
        else:
            X_t = X

        y_pred_log = self.regressor_.predict(X_t)

        if self.y_inverse is not None:
            return self.y_inverse(y_pred_log)
        return y_pred_log

    def get_coefficients(self) -> Dict[str, float]:
        """Extract b, c, and T1 from log-space model."""
        coef = self.regressor_.coef_
        intercept = self.regressor_.intercept_
        return {
            'b': coef[0],
            'c': coef[1] if len(coef) > 1 else 0.0,
            'intercept': intercept,
            'T1_est': np.exp(intercept) if intercept != 0 else np.nan,
        }

    def get_regularization_params(self) -> Dict[str, Any]:
        params = {}
        if hasattr(self.regressor_, 'alpha_'):
            params['alpha'] = self.regressor_.alpha_
        if hasattr(self.regressor_, 'l1_ratio_'):
            params['l1_ratio'] = self.regressor_.l1_ratio_
        if hasattr(self.regressor_, 'converged_'):
            params['converged'] = self.regressor_.converged_
        else:
            params['converged'] = True
        return params


class PCReg_LogMSE(PCRegLogSpaceBase):
    """PCReg with MSE on log(Y) - like OLS but with constraints on b and c."""
    bounds_key = 'correct'
    use_cv = False


class PCReg_LogMSE_CV(PCRegLogSpaceBase):
    """PCReg with MSE on log(Y) and CV-tuned penalty."""
    bounds_key = 'correct'
    use_cv = True


# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODEL_CLASSES: Dict[str, type] = {
    'OLS': OLS,
    'OLS_LearnOnly': OLS_LearnOnly,
    'RidgeCV': Ridge,
    'LassoCV': Lasso,
    'BayesianRidge': BayesianRidgeModel,
    # 'PLS': PLS,  # Commented out - not needed for analysis
    # 'PCA_OLS': PCA_OLS,  # Commented out - not needed for analysis
    'PCReg_ConstrainOnly': PCReg_ConstrainOnly,  # SSPE loss, unit space, constraints only
    'PCReg_CV': PCReg_CV,  # SSPE loss, unit space, CV-tuned penalty
    # 'PCReg_AICc': PCReg_AICc,  # Removed - poor performance with small samples
    'PCReg_GCV': PCReg_GCV,
    'PCReg_CV_Tight': PCReg_CV_Tight,
    'PCReg_CV_Wrong': PCReg_CV_Wrong,
    # New loss function variants
    'PCReg_MSE': PCReg_MSE,  # MSE loss, unit space, constraints only
    'PCReg_MSE_CV': PCReg_MSE_CV,  # MSE loss, unit space, CV-tuned penalty
    'PCReg_LogMSE': PCReg_LogMSE,  # MSE loss, log space (like constrained OLS)
    'PCReg_LogMSE_CV': PCReg_LogMSE_CV,  # MSE loss, log space, CV-tuned penalty
}


def get_model_config(model_class: type) -> Dict[str, Any]:
    """
    Extract configuration from a model class for hashing.

    Captures class attributes that define model behavior.
    """
    config = {'class_name': model_class.__name__}

    # Extract relevant class attributes
    for attr in ['x_transforms', 'y_transform', 'y_inverse',
                 'regressor_class', 'regressor_params',
                 'bounds_key', 'use_cv', 'selection_method', 'loss_type']:
        if hasattr(model_class, attr):
            val = getattr(model_class, attr)
            # Convert to string representation for hashing
            if attr == 'regressor_class':
                config[attr] = val.__name__ if val else None
            elif attr == 'x_transforms':
                # Serialize transform pipeline structure
                config[attr] = [(name, t.__class__.__name__) for name, t in val] if val else []
            elif callable(val) and val is not None:
                # For transform functions, use their name
                config[attr] = getattr(val, '__name__', str(val))
            else:
                config[attr] = val

    return config


def get_model_hash(model_name: str) -> str:
    """
    Generate a hash for a model based on its configuration.

    The hash changes when model parameters change, triggering re-run.

    Parameters
    ----------
    model_name : str
        Name of the model in MODEL_CLASSES.

    Returns
    -------
    hash : str
        12-character hash of the model configuration.
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = MODEL_CLASSES[model_name]
    config = get_model_config(model_class)

    # Create deterministic string representation
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def get_all_model_hashes() -> Dict[str, str]:
    """Get hashes for all registered models."""
    return {name: get_model_hash(name) for name in MODEL_CLASSES}


def migrate_add_model_hashes(output_dir: Path = None) -> None:
    """
    One-time migration: add model_hash column to existing results.

    Run this once after updating to the new model-hash-based tracking.
    Creates a backup before modifying.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory. If None, uses CONFIG['output_dir'].
    """
    if output_dir is None:
        output_dir = CONFIG['output_dir']

    results_file = output_dir / 'simulation_results.parquet'

    if not results_file.exists():
        print(f"No results file found at {results_file}")
        return

    print(f"Migrating {results_file}...")

    # Read existing results
    df = pd.read_parquet(results_file)

    if 'model_hash' in df.columns:
        print("model_hash column already exists. Skipping migration.")
        return

    # Build hash lookup from current model definitions
    hash_lookup = get_all_model_hashes()

    # Check for unknown models
    unknown_models = set(df['model_name'].unique()) - set(hash_lookup.keys())
    if unknown_models:
        print(f"Warning: Found models not in current registry: {unknown_models}")
        print("These rows will have NaN model_hash and will be re-run.")

    # Add hash column
    df['model_hash'] = df['model_name'].map(hash_lookup)

    # Create backup
    backup_file = output_dir / 'simulation_results_backup.parquet'
    df_original = pd.read_parquet(results_file)
    df_original.to_parquet(backup_file, engine='pyarrow', index=False)
    print(f"Created backup at {backup_file}")

    # Save updated results
    df.to_parquet(results_file, engine='pyarrow', index=False)
    print(f"Added model_hash to {len(df)} rows")
    print(f"Hash mapping:")
    for name, hash_val in sorted(hash_lookup.items()):
        count = (df['model_name'] == name).sum()
        print(f"  {name}: {hash_val} ({count} rows)")


def generate_wrong_bounds(true_b: float, true_c: float, a: float = 0.5,
                          random_state: int = None) -> List[Tuple[float, float]]:
    """Generate 'wrong' constraints following James et al. (2020) PAC methodology."""
    rng = np.random.RandomState(random_state)
    u_b = rng.uniform(0, a)
    u_c = rng.uniform(0, a)
    perturbed_lower_b = -1 * (1 + u_b)
    perturbed_lower_c = -0.5 * (1 + u_c)
    return [(perturbed_lower_b, 0), (perturbed_lower_c, 0)]


def create_models(scenario_config: Dict) -> Dict[str, SimulationModel]:
    """
    Instantiate all models with scenario configuration.

    Parameters
    ----------
    scenario_config : dict
        Contains: b_true, c_true, T1, tight_bounds, wrong_bounds,
        correct_bounds, alpha_grid, l1_ratio_grid, cv_folds, etc.

    Returns
    -------
    models : dict
        Model name -> instantiated SimulationModel.
    """
    return {name: cls(**scenario_config) for name, cls in MODEL_CLASSES.items()}


# ============================================================================
# FITTING AND METRICS
# ============================================================================
def fit_model(
    model: SimulationModel,
    X: np.ndarray,
    y: np.ndarray,
    true_params: Dict[str, float],
) -> Tuple[Dict[str, Any], SimulationModel]:
    """
    Fit a model and extract results.

    Parameters
    ----------
    model : SimulationModel
        Model instance to fit.
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
    model : SimulationModel
        Fitted model.
    """
    result = {
        'model_name': model.__class__.__name__,
        'converged': True,
        'fit_time': 0.0,
    }

    try:
        start_time = time.time()
        model.fit(X, y)
        result['fit_time'] = time.time() - start_time

        # Get coefficients
        coefs = model.get_coefficients()
        result.update(coefs)

        # Derive learning curve slopes from b and c (LC_est = 2^b, RC_est = 2^c)
        b = coefs.get('b', np.nan)
        c = coefs.get('c', np.nan)
        result['LC_est'] = 2 ** b if not np.isnan(b) else np.nan
        result['RC_est'] = 2 ** c if not np.isnan(c) else np.nan

        # R² in unit space
        result['r2'] = model.score(X, y)

        # Regularization params
        reg_params = model.get_regularization_params()
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
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run models on a single scenario.

    All models receive the SAME raw X, y data in unit space.

    Parameters
    ----------
    scenario : tuple
        (n_lots, correlation, cv_error, learning_rate, rate_effect, replication)
    config : dict
        Simulation configuration.
    models_to_run : list of str, optional
        List of model names to run. If None, runs all models in MODEL_CLASSES.
    """
    n_lots, correlation, cv_error, learning_rate, rate_effect, replication = scenario

    # Compute true parameters
    b_true = pcreg.learning_rate_to_slope(learning_rate)
    c_true = pcreg.learning_rate_to_slope(rate_effect)
    T1_true = config['T1']

    true_params = {'b': b_true, 'c': c_true, 'T1': T1_true}

    # Deterministic seed
    seed = config['base_seed'] + hash((n_lots, correlation, cv_error,
                                        learning_rate, rate_effect, replication)) % (2**31)

    # Generate training data in unit space
    data = pcreg.generate_correlated_learning_data(
        n_lots=n_lots,
        T1=T1_true,
        b=b_true,
        c=c_true,
        target_correlation=correlation,
        cv_error=cv_error,
        random_state=seed
    )

    X_train = data['X_original']  # Unit space
    y_train = data['y_original']  # Unit space
    actual_corr = data['actual_correlation']

    # Generate test data in unit space
    test_n_lots = config['test_n_lots']
    max_train_qty = int(np.max(data['lot_quantities']))
    test_quantity = int(max_train_qty * config['test_quantity_multiplier'])
    first_test_unit = int(data['last_units'][-1]) + 1

    test_data = pcreg.generate_test_data(
        first_unit_start=first_test_unit,
        n_lots=test_n_lots,
        base_quantity=test_quantity,
        T1=T1_true,
        b=b_true,
        c=c_true,
        cv_error=0.0,
        random_state=seed + 1000
    )

    X_test = test_data['X_original']
    y_test = test_data['true_costs']

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

    # Create models (only those requested, or all if not specified)
    all_models = create_models(scenario_config)
    if models_to_run is None:
        models = all_models
    else:
        models = {name: all_models[name] for name in models_to_run if name in all_models}

    # Fit models
    results = []
    predictions = []
    save_predictions = config.get('save_predictions', False)

    for model_name, model in models.items():
        # Fit model
        result, fitted_model = fit_model(model, X_train, y_train, true_params)

        # Compute OOS metrics
        if result['converged'] and not np.isnan(result.get('b', np.nan)):
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
                            'target_correlation': correlation,
                            'cv_error': cv_error,
                            'learning_rate': learning_rate,
                            'rate_effect': rate_effect,
                            'replication': replication,
                            'seed': seed,
                            'test_lot_number': i + 1,
                            'test_lot_midpoint': test_data['lot_midpoints'][i],
                            'test_lot_quantity': test_data['lot_quantities'][i],
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
            'test_quantity': test_quantity,
            'first_test_unit': first_test_unit,
            'model_hash': get_model_hash(model_name),
        })

        results.append(result)

    return results, predictions


# ============================================================================
# PARALLEL PROCESSING WITH BATCHING
# ============================================================================
def _scenario_worker(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """Worker function for parallel execution."""
    scenario, config, models_to_run = args
    return run_single_scenario(scenario, config, models_to_run)


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
        scenario_cols = ['n_lots', 'target_correlation', 'cv_error',
                         'learning_rate', 'rate_effect', 'replication', 'model_hash']

        completed_scenario_models = set()
        for _, row in df[scenario_cols].drop_duplicates().iterrows():
            key = (row['n_lots'], row['target_correlation'], row['cv_error'],
                   row['learning_rate'], row['rate_effect'], row['replication'],
                   row['model_hash'])
            completed_scenario_models.add(key)

        # Find scenarios where ALL current models have been run
        scenario_only_cols = ['n_lots', 'target_correlation', 'cv_error',
                              'learning_rate', 'rate_effect', 'replication']
        completed_scenarios = set()

        for _, row in df[scenario_only_cols].drop_duplicates().iterrows():
            scenario_key = (row['n_lots'], row['target_correlation'], row['cv_error'],
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


def run_simulation_parallel(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full simulation with parallel processing."""
    from joblib import Parallel, delayed

    output_dir = config['output_dir']
    output_dir.mkdir(exist_ok=True)

    n_models = len(MODEL_CLASSES)

    # Build all scenarios
    all_scenarios = list(product(
        config['sample_sizes'],
        config['correlations'],
        config['cv_errors'],
        config['learning_rates'],
        config['rate_effects'],
        range(config['n_replications'])
    ))
    n_total = len(all_scenarios)

    # Resume capability - now tracks at model level
    completed_scenario_models, completed_scenarios = load_completed_scenarios(output_dir)

    # Build list of (scenario, models_to_run) pairs
    scenarios_with_models = []
    total_model_fits = 0
    for scenario in all_scenarios:
        if scenario in completed_scenarios:
            # All models done for this scenario
            continue
        models_needed = get_models_to_run(scenario, completed_scenario_models)
        if models_needed:
            scenarios_with_models.append((scenario, models_needed))
            total_model_fits += len(models_needed)

    n_to_run = len(scenarios_with_models)
    n_skipped = n_total - n_to_run

    print(f"Total scenarios: {n_total}")
    print(f"Fully completed scenarios: {n_skipped}")
    print(f"Scenarios needing models: {n_to_run}")
    print(f"Total models in registry: {n_models}")
    print(f"Total model fits needed: {total_model_fits}")
    print(f"Parallel workers: {config['n_jobs']} (backend: {config['backend']})")
    print(f"Batch size: {config['batch_size']}")
    print()

    if n_to_run == 0:
        print("All scenarios and models completed!")
        return merge_batch_files(output_dir)

    start_time = time.time()
    batch_size = config['batch_size']
    n_jobs = config['n_jobs']
    backend = config['backend']

    n_batches = (n_to_run + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_to_run)
        batch_items = scenarios_with_models[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{n_batches} "
              f"(scenarios {batch_start + 1}-{batch_end})...")

        worker_args = [(s, config, models) for s, models in batch_items]

        batch_results_list = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=0,
        )(delayed(_scenario_worker)(args) for args in worker_args)

        batch_results = []
        batch_predictions = []

        for results, predictions in batch_results_list:
            batch_results.extend(results)
            batch_predictions.extend(predictions)

        save_batch_file(batch_results, batch_predictions, output_dir, batch_idx)

        elapsed = time.time() - start_time
        scenarios_done = batch_end
        rate = scenarios_done / elapsed if elapsed > 0 else 0
        remaining = (n_to_run - scenarios_done) / rate if rate > 0 else 0

        total_expected = elapsed + remaining
        print(f"  Saved {len(batch_results)} model fits")
        print(f"  Progress: {scenarios_done}/{n_to_run} ({100*scenarios_done/n_to_run:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min | Total Expected: {total_expected/60:.1f} min")
        print()

    print("Merging batch files...")
    results_df, predictions_df = merge_batch_files(output_dir)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Average per scenario: {total_time/n_to_run:.3f} seconds")

    return results_df, predictions_df


def run_simulation_sequential(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation sequentially (for debugging)."""
    output_dir = config['output_dir']
    output_dir.mkdir(exist_ok=True)

    all_scenarios = list(product(
        config['sample_sizes'],
        config['correlations'],
        config['cv_errors'],
        config['learning_rates'],
        config['rate_effects'],
        range(config['n_replications'])
    ))

    # Resume capability - now tracks at model level
    completed_scenario_models, completed_scenarios = load_completed_scenarios(output_dir)

    # Build list of (scenario, models_to_run) pairs
    scenarios_with_models = []
    total_model_fits = 0
    for scenario in all_scenarios:
        if scenario in completed_scenarios:
            continue
        models_needed = get_models_to_run(scenario, completed_scenario_models)
        if models_needed:
            scenarios_with_models.append((scenario, models_needed))
            total_model_fits += len(models_needed)

    print(f"Running {len(scenarios_with_models)} scenarios sequentially...")
    print(f"Total model fits needed: {total_model_fits}")

    start_time = time.time()
    batch_size = config['batch_size']
    batch_results = []
    batch_predictions = []
    batch_idx = 0

    for i, (scenario, models_to_run) in enumerate(scenarios_with_models):
        results, predictions = run_single_scenario(scenario, config, models_to_run)
        batch_results.extend(results)
        batch_predictions.extend(predictions)

        if len(batch_results) >= batch_size * len(MODEL_CLASSES):
            save_batch_file(batch_results, batch_predictions, output_dir, batch_idx)
            batch_results = []
            batch_predictions = []
            batch_idx += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i+1}/{len(scenarios_with_models)} - {elapsed/60:.1f} min elapsed")

    if batch_results:
        save_batch_file(batch_results, batch_predictions, output_dir, batch_idx)

    return merge_batch_files(output_dir)


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

    scenario_cols = ['n_lots', 'target_correlation', 'cv_error',
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
    parser.add_argument('--migrate', action='store_true',
                        help='Run one-time migration to add model_hash column')
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

    if args.migrate:
        migrate_add_model_hashes()
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
    print(f"  Correlations: {CONFIG['correlations']}")
    print(f"  CV errors: {CONFIG['cv_errors']}")
    print(f"  Learning rates: {CONFIG['learning_rates']}")
    print(f"  Rate effects: {CONFIG['rate_effects']}")
    print(f"  Replications: {CONFIG['n_replications']}")
    print(f"  Parallel workers: {CONFIG['n_jobs']}")
    print()

    # Run simulation
    if CONFIG['n_jobs'] == 1:
        results_df, predictions_df = run_simulation_sequential(CONFIG)
    else:
        results_df, predictions_df = run_simulation_parallel(CONFIG)

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
