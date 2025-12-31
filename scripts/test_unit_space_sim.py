"""Quick test: PCReg in unit space within simulation framework."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg

# Simulation-like setup
T1_true = 100
learning_rate = 0.90
rate_effect = 0.85
b_true = pcreg.learning_rate_to_slope(learning_rate)  # -0.152
c_true = pcreg.learning_rate_to_slope(rate_effect)    # -0.234

print(f"True slopes: b={b_true:.4f}, c={c_true:.4f}")

# Generate data
data = pcreg.generate_correlated_learning_data(
    n_lots=10, T1=T1_true, b=b_true, c=c_true,
    target_correlation=0.5, cv_error=0.1, random_state=42
)
X_original, y_original = data['X_original'], data['y_original']

# Define prediction function
def lc_func(X, params):
    T1, b, c = params
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)

# Test PCRegConstrain_Only
model = pcreg.PenalizedConstrainedRegression(
    bounds={'T1': (0, None), 'LC': (-0.5, 0), 'RC': (-0.5, 0)},
    feature_names=['T1', 'LC', 'RC'],
    alpha=0.0,
    loss='sspe',
    prediction_fn=lc_func,
    fit_intercept=False,
    init=[T1_true, b_true, c_true]
)
model.fit(X_original, y_original)

print(f"\nPCRegConstrain_Only:")
print(f"  Converged: {model.converged_}")
print(f"  T1={model.coef_[0]:.2f}, LC={model.coef_[1]:.4f}, RC={model.coef_[2]:.4f}")

# Test PCRegCV
model_cv = pcreg.PenalizedConstrainedCV(
    bounds={'T1': (0, None), 'LC': (-0.5, 0), 'RC': (-0.5, 0)},
    feature_names=['T1', 'LC', 'RC'],
    alphas=np.logspace(-2, 2, 5),
    l1_ratios=[0.0, 0.5, 1.0],
    loss='sspe',
    prediction_fn=lc_func,
    fit_intercept=False,
    init=[T1_true, b_true, c_true],
    cv=3,
    verbose=0
)
model_cv.fit(X_original, y_original)

print(f"\nPCRegCV:")
print(f"  Converged: {model_cv.converged_}")
print(f"  T1={model_cv.coef_[0]:.2f}, LC={model_cv.coef_[1]:.4f}, RC={model_cv.coef_[2]:.4f}")
print(f"  Best alpha: {model_cv.alpha_:.4f}, l1_ratio: {model_cv.l1_ratio_:.2f}")

# Check error calculation (simulating fit_single_model logic)
LC_est = model_cv.coef_[1]  # This IS the slope
RC_est = model_cv.coef_[2]  # This IS the slope
b = LC_est  # Slope, not learning rate!
c = RC_est  # Slope, not rate effect!

b_error = abs(b - b_true)
c_error = abs(c - c_true)

print(f"\nError calculation check:")
print(f"  b_true={b_true:.4f}, b_est={b:.4f}, b_error={b_error:.4f}")
print(f"  c_true={c_true:.4f}, c_est={c:.4f}, c_error={c_error:.4f}")
print(f"  b_correct_sign: {b <= 0}")
print(f"  c_correct_sign: {c <= 0}")
