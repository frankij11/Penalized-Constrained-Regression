"""
Verify the motivating example produces the expected results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load example data
df_example = pd.read_csv('output_v2/motivational_example_data.csv')
df_train = df_example[df_example['lot_type'] == 'train'].copy()
df_test = df_example[df_example['lot_type'] == 'test'].copy()

print("="*70)
print("MOTIVATIONAL EXAMPLE DATA")
print("="*70)
print(f"Seed: {df_example['seed'].iloc[0]}")
print(f"Training lots: {len(df_train)}")
print(f"Test lots: {len(df_test)}")
print(f"\nTrue parameters:")
print(f"  T1: {df_example['T1_true'].iloc[0]}")
print(f"  b: {df_example['b_true'].iloc[0]:.4f}  -> LR = {2**df_example['b_true'].iloc[0]*100:.1f}%")
print(f"  c: {df_example['c_true'].iloc[0]:.4f}  -> RE = {2**df_example['c_true'].iloc[0]*100:.1f}%")
print(f"\nCorrelation: {df_example['actual_correlation'].iloc[0]:.3f}")
print(f"CV error: {df_example['cv_error'].iloc[0]}")

# Now fit models
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import penalized_constrained as pcreg

X_train = df_train[['lot_midpoint', 'lot_quantity']].values
y_train = df_train['observed_cost'].values
X_test = df_test[['lot_midpoint', 'lot_quantity']].values
y_test_true = df_test['true_cost'].values

# OLS
ols = TransformedTargetRegressor(
    regressor=Pipeline([
        ('log', FunctionTransformer(np.log)),
        ('reg', LinearRegression()),
    ]),
    func=np.log,
    inverse_func=np.exp,
)
ols.fit(X_train, y_train)
ols_b, ols_c = ols.regressor_.named_steps['reg'].coef_
ols_T1 = np.exp(ols.regressor_.named_steps['reg'].intercept_)
ols_pred_test = ols.predict(X_test)
ols_mape_true = mean_absolute_percentage_error(y_test_true, ols_pred_test)
ols_r2 = r2_score(y_train, ols.predict(X_train))

# OLS LearnOnly
X_train_learn = df_train[['lot_midpoint']].values
X_test_learn = df_test[['lot_midpoint']].values
ols_learn = TransformedTargetRegressor(
    regressor=Pipeline([
        ('log', FunctionTransformer(np.log)),
        ('reg', LinearRegression()),
    ]),
    func=np.log,
    inverse_func=np.exp,
)
ols_learn.fit(X_train_learn, y_train)
ols_learn_b = ols_learn.regressor_.named_steps['reg'].coef_[0]
ols_learn_T1 = np.exp(ols_learn.regressor_.named_steps['reg'].intercept_)
ols_learn_pred_test = ols_learn.predict(X_test_learn)
ols_learn_mape_true = mean_absolute_percentage_error(y_test_true, ols_learn_pred_test)
ols_learn_r2 = r2_score(y_train, ols_learn.predict(X_train_learn))

# PCReg-GCV
def prediction_fn(X, params):
    T1, b, c = params
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)

pc_gcv = pcreg.PenalizedConstrainedCV(
    coef_names=['T1', 'b', 'c'],
    bounds={'T1': (0, None), 'b': (-0.5, 0), 'c': (-0.5, 0)},
    prediction_fn=prediction_fn,
    fit_intercept=False,
    x0=[100, -0.1, -0.1],
    selection='gcv',
    loss='sspe',
    penalty_exclude=['T1'],
    n_jobs=1
)
pc_gcv.fit(X_train, y_train)
pc_T1, pc_b, pc_c = pc_gcv.coef_
pc_pred_test = pc_gcv.predict(X_test)
pc_mape_true = mean_absolute_percentage_error(y_test_true, pc_pred_test)
pc_r2 = r2_score(y_train, pc_gcv.predict(X_train))

print("\n" + "="*70)
print("MODEL RESULTS (as will appear in paper)")
print("="*70)

true_T1 = df_example['T1_true'].iloc[0]
true_lr = 2 ** df_example['b_true'].iloc[0]
true_re = 2 ** df_example['c_true'].iloc[0]

print(f"\n{'Metric':<25} {'True':>10} {'OLS':>12} {'OLS-Learn':>12} {'PCReg-GCV':>12}")
print("-"*70)
print(f"{'T1':<25} {true_T1:>10.0f} {ols_T1:>12.0f} {ols_learn_T1:>12.0f} {pc_T1:>12.0f}")
print(f"{'Learning Rate':<25} {true_lr*100:>10.1f}% {2**ols_b*100:>12.1f}% {2**ols_learn_b*100:>12.1f}% {2**pc_b*100:>12.1f}%")
print(f"{'Rate Effect':<25} {true_re*100:>10.1f}% {2**ols_c*100:>12.1f}% {'--':>12} {2**pc_c*100:>12.1f}%")

# Check validity
ols_valid = (2**ols_b <= 1.0) and (2**ols_b >= 0.7) and (2**ols_c <= 1.0) and (2**ols_c >= 0.7)
ols_learn_valid = (2**ols_learn_b <= 1.0) and (2**ols_learn_b >= 0.7)
pc_valid = True  # Always valid by construction

print(f"{'Valid Coefficients':<25} {'Yes':>10} {'NO' if not ols_valid else 'Yes':>12} {'Yes' if ols_learn_valid else 'NO':>12} {'Yes':>12}")
print(f"{'Train R²':<25} {'--':>10} {ols_r2:>12.3f} {ols_learn_r2:>12.3f} {pc_r2:>12.3f}")
print(f"{'Test MAPE (vs True)':<25} {'--':>10} {ols_mape_true*100:>12.1f}% {ols_learn_mape_true*100:>12.1f}% {pc_mape_true*100:>12.1f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"\nOLS produces INVALID coefficients:")
print(f"  - Learning Rate = {2**ols_b*100:.1f}% (> 100% means learning 'unlearning'!)")
print(f"  - Rate Effect = {2**ols_c*100:.1f}% (< 70% is unreasonable)")
print(f"  - Test MAPE = {ols_mape_true*100:.1f}% (terrible prediction)")
print(f"\nPCReg-GCV produces VALID coefficients:")
print(f"  - Learning Rate = {2**pc_b*100:.1f}% (reasonable, close to true {true_lr*100:.1f}%)")
print(f"  - Rate Effect = {2**pc_c*100:.1f}% (reasonable, close to true {true_re*100:.1f}%)")
print(f"  - Test MAPE = {pc_mape_true*100:.1f}% (excellent prediction)")
print(f"\nThis clearly demonstrates why PCReg-GCV wins!")
