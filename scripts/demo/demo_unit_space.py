"""Demo: PCReg in unit space with custom prediction function."""
import numpy as np
import penalized_constrained as pcreg

# Generate sample data
data = pcreg.generate_correlated_learning_data(n_lots=10, T1=100, b=-0.152, c=-0.234, cv_error=0.1, random_state=42)
X, y = data['X_original'], data['y_original']  # Unit space data
print(f"X shape: {X.shape}, X[0]: {X[0]}")  # Should be [midpoint, quantity], not logs
print(f"y shape: {y.shape}, y[0]: {y[0]:.2f}")  # Should be actual cost, not log

# Define prediction function: Y = T1 * X1^b * X2^c
def lc_func(X, params):
    T1, b, c = params
    return T1 * (X[:, 0] ** b) * (X[:, 1] ** c)

# Fit model
model = pcreg.PenalizedConstrainedRegression(
    prediction_fn=lc_func,
    feature_names=['T1', 'LC', 'RC'],
    bounds={'T1': (0, None), 'LC': (-0.5, 0), 'RC': (-0.5, 0)},
    loss='sspe',
    fit_intercept=False,
    init=[100, -0.15, -0.23]  # Good starting point near true values
)
model.fit(X, y)

print(f"\nConverged: {model.converged_}")
print(f"Coefficients: T1={model.coef_[0]:.2f}, LC={model.coef_[1]:.4f}, RC={model.coef_[2]:.4f}")
print(f"True values:  T1=100, LC=-0.152, RC=-0.234")
print(f"Predictions sample: {model.predict(X[:3])}")
print(f"Actual sample: {y[:3]}")
