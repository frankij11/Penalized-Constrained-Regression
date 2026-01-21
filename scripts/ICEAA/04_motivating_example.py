'''
Docstring for scripts.ICEAA.04_motivating_example
This module contains a motivating example for the ICEAA framework.
OLS vs. PCRegGCV
'''
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
import numpy as np


# custom library
import penalized_constrained as pcreg
DIR = Path(__file__).resolve().parent

df = pd.read_csv(DIR / 'output_v2/motivational_example_data.csv')

df_train = df.query("lot_type == 'train'").copy()
df_test = df.query("lot_type == 'test'").copy()
X = df_train[['lot_midpoint', 'lot_quantity']]
y = df_train['observed_cost']


print(df.head())

ols = TransformedTargetRegressor(
        regressor=Pipeline([
            ('log', FunctionTransformer(np.log)),
            ('reg', LinearRegression()),
        ]),
        func=np.log,
        inverse_func=np.exp,
    )
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

pc = pcreg.PenalizedConstrainedCV(
        coef_names=['T1', 'b', 'c'],
        bounds={'T1': (0, None), 'b': (-0.5, 0), 'c': (-0.5, 0)},
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=[1, 0, 0],
        loss='sspe',
        alphas=np.logspace(-5, 0, 10),  # Match simulation
        l1_ratios=[0.0, 0.5, 1.0],       # Match simulation
        cv=3,                             # Match simulation cv_folds
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
        safe_mode=True
    )

pc_enet = pcreg.PenalizedConstrainedCV(
        coef_names=['T1', 'b', 'c'],
        bounds={'T1': (None, None), 'b': (None, None), 'c': (None, None)},
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=[1, 0, 0],
        loss='sspe',
        alphas=np.logspace(-5, 0, 10),  # Match simulation
        l1_ratios=[0.0, 0.5, 1.0],       # Match simulation
        cv=3,                             # Match simulation cv_folds
        selection='gcv',
        penalty_exclude=['T1'],
        n_jobs=1,
        verbose=0,
        safe_mode=True
    )


ols.fit(X, y)
ols_lc, ols_rc=2**ols.regressor_.named_steps['reg'].coef_
ols_t1 = np.exp(ols.regressor_.named_steps['reg'].intercept_)

print("OLS R^2:", ols.score(X, y))
print("OLS Coefficients (T1, LC, RC):", ols_t1, ols_lc, ols_rc)

# Check if OLS produces "bad" coefficients (LC > 1 or RC > 1 is economically impossible)
bad_ols = (ols_lc > 1) or (ols_rc > 1)
print(f"OLS has impossible coefficients (LC>1 or RC>1): {bad_ols}")


pc.fit(X, y)
pc_enet.fit(X, y)

# Hyperparameters selected by GCV (may vary slightly from simulation due to numeric precision)
# Simulation results for this seed: alpha=1.0, l1_ratio=0.0
print(f"PCRegGCV selected alpha: {pc.alpha_}")
print(f"PCRegGCV selected l1_ratio: {pc.l1_ratio_}")

pc_lc, pc_rc = 2**pc.coef_[1], 2**pc.coef_[2]
pc_t1 = pc.coef_[0]

pc_enet_lc, pc_enet_rc = 2**pc_enet.coef_[1], 2**pc_enet.coef_[2]
pc_enet_t1 = pc_enet.coef_[0]


# print Score and coefficients
print("PCRegGCV R^2:", pc.score(X, y))
print("PCRegGCV Coefficients (T1, LC, RC):", pc_t1, pc_lc, pc_rc)

# print Score and coefficients
print("PCRegGCV Enet (No Constraints) R^2:", pc_enet.score(X, y))
print("PCRegGCV Enet (No Constraints) Coefficients (T1, LC, RC):", pc_enet_t1, pc_enet_lc, pc_enet_rc)


# Check if PCReg respects constraints (LC <= 1 and RC <= 1)
pcreg_valid = (pc_lc <= 1) and (pc_rc <= 1)
print(f"PCReg respects economic constraints (LC<=1 and RC<=1): {pcreg_valid}")

pc.summary()

# Summary of the motivating example
print("\n" + "="*60)
print("MOTIVATING EXAMPLE SUMMARY")
print("="*60)
print(f"OLS produces impossible coefficients: {bad_ols}")
print(f"  - OLS LC (learning curve): {ols_lc:.4f} {'> 1 (INVALID)' if ols_lc > 1 else '<= 1 (valid)'}")
print(f"  - OLS RC (rate effect): {ols_rc:.4f} {'> 1 (INVALID)' if ols_rc > 1 else '<= 1 (valid)'}")
print(f"PCReg respects constraints: {pcreg_valid}")
print(f"  - PCReg LC: {pc_lc:.4f} <= 1 (valid)")
print(f"  - PCReg RC: {pc_rc:.4f} <= 1 (valid)")
print(f"\nBoth models achieve similar R^2 on training data:")
print(f"  - OLS R^2: {ols.score(X, y):.4f}")
print(f"  - PCReg R^2: {pc.score(X, y):.4f}")
print("="*60)