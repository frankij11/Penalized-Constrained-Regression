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

pc = pcreg.PenalizedConstrainedRegression(
        coef_names=['T1', 'b', 'c'],
        bounds={'T1': (0, None), 'b': (-.5, 0), 'c': (-5, 0)},
        prediction_fn=unit_space_prediction_fn,
        fit_intercept=False,
        x0=[1,0,0],
        loss='sspe',
        alpha=1, #np.logspace(-5, 0, 20)
        l1_ratio=0,#[0, 0.5, 1],
        #selection='gcv',
        penalty_exclude=['T1'],
        #n_jobs=-1,
        #verbose=0,
        safe_mode=True
    )


ols.fit(X, y)
ols_lc, ols_rc=2**ols.regressor_.named_steps['reg'].coef_
ols_t1 = np.exp(ols.regressor_.named_steps['reg'].intercept_)

print("OLS R^2:", ols.score(X, y))
print("OLS Coefficients:", ols_t1, ols_lc, ols_rc)


pc.fit(X, y)
pc_lc, pc_rc = 2**pc.coef_[1], 2**pc.coef_[2]
pc_t1 = pc.coef_[0]
# print Score and coefficients
print("PCRegGCV R^2:", pc.score(X, y))
print("PCRegGCV Coefficients:", pc_t1, pc_lc, pc_rc)

pc.summary()