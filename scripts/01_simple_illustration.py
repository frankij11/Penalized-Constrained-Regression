"""
01_simple_illustration.py
=========================
Simple illustration of Penalized-Constrained Regression WITHOUT the library.
Demonstrates the core concept: combining ElasticNet penalty with bound constraints.

This script is intentionally concise (<100 lines) for paper illustration.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge

# === DATA GENERATION ===
np.random.seed(42)
n, T1, b_true, c_true = 20, 100, -0.152, -0.074  # 90% learning, 95% rate

# Simulate correlated lot data
lot_qty = np.round(5 * 1.3 ** np.arange(n) + np.random.uniform(-2, 2, n)).astype(int)
lot_qty = np.clip(lot_qty, 1, None)
midpoints = np.cumsum(lot_qty) - lot_qty/2  # Approximate midpoints

# Generate costs with 10% CV error
true_cost = T1 * (midpoints ** b_true) * (lot_qty ** c_true)
observed_cost = true_cost * np.exp(np.random.normal(0, 0.1, n))

# Log transform for linear regression
X = np.column_stack([np.log(midpoints), np.log(lot_qty)])
y = np.log(observed_cost)

print("="*70)
print("SIMPLE ILLUSTRATION: OLS vs Ridge vs Constrained vs Penalized-Constrained")
print("="*70)
print(f"True parameters: T1={T1}, b={b_true:.4f}, c={c_true:.4f}")
print(f"Correlation between log(midpoint) and log(quantity): {np.corrcoef(X[:,0], X[:,1])[0,1]:.2f}\n")

# === METHOD 1: OLS ===
ols = LinearRegression().fit(X, y)
print(f"OLS:         b={ols.coef_[0]:+.4f}, c={ols.coef_[1]:+.4f}, T1={np.exp(ols.intercept_):.1f}")

# === METHOD 2: Ridge ===
ridge = Ridge(alpha=0.1).fit(X, y)
print(f"Ridge:       b={ridge.coef_[0]:+.4f}, c={ridge.coef_[1]:+.4f}, T1={np.exp(ridge.intercept_):.1f}")

# === METHOD 3: Constrained Only (no penalty) ===
def objective_sse(params, X, y):
    coef, intercept = params[:2], params[2]
    return np.sum((y - X @ coef - intercept) ** 2)

bounds_constrained = [(-1, 0), (-1, 0), (None, None)]  # b≤0, c≤0, intercept free
x0 = np.append(ols.coef_, ols.intercept_)
x0[:2] = np.clip(x0[:2], -1, 0)  # Clip to bounds

res_constrained = minimize(objective_sse, x0, args=(X, y), method='SLSQP', bounds=bounds_constrained)
print(f"Constrained: b={res_constrained.x[0]:+.4f}, c={res_constrained.x[1]:+.4f}, "
      f"T1={np.exp(res_constrained.x[2]):.1f}")

# === METHOD 4: Penalized-Constrained (ElasticNet + Bounds) ===
def objective_penalized(params, X, y, alpha=0.1, l1_ratio=0.5):
    coef, intercept = params[:2], params[2]
    sse = np.sum((y - X @ coef - intercept) ** 2)
    l1 = alpha * l1_ratio * np.sum(np.abs(coef))
    l2 = 0.5 * alpha * (1 - l1_ratio) * np.sum(coef ** 2)
    return sse + l1 + l2

res_penalized = minimize(objective_penalized, x0, args=(X, y, 0.1, 0.5), 
                         method='SLSQP', bounds=bounds_constrained)
print(f"Pen+Constr:  b={res_penalized.x[0]:+.4f}, c={res_penalized.x[1]:+.4f}, "
      f"T1={np.exp(res_penalized.x[2]):.1f}")

# === COMPARISON TABLE ===
print("\n" + "="*70)
print("COEFFICIENT ERROR COMPARISON")
print("="*70)
print(f"{'Method':<15} {'b error':>10} {'c error':>10} {'R²':>8}")
print("-"*45)

methods = [
    ('OLS', ols.coef_[0], ols.coef_[1], ols.score(X, y)),
    ('Ridge', ridge.coef_[0], ridge.coef_[1], ridge.score(X, y)),
    ('Constrained', res_constrained.x[0], res_constrained.x[1], 
     1 - np.sum((y - X @ res_constrained.x[:2] - res_constrained.x[2])**2) / np.sum((y - y.mean())**2)),
    ('Pen+Constr', res_penalized.x[0], res_penalized.x[1],
     1 - np.sum((y - X @ res_penalized.x[:2] - res_penalized.x[2])**2) / np.sum((y - y.mean())**2)),
]

for name, b_est, c_est, r2 in methods:
    b_err = abs(b_est - b_true)
    c_err = abs(c_est - c_true)
    sign_ok = "✓" if (b_est <= 0 and c_est <= 0) else "✗"
    print(f"{name:<15} {b_err:>10.4f} {c_err:>10.4f} {r2:>8.4f}  {sign_ok}")

print("\n" + "="*70)
print("KEY INSIGHT: Penalized-Constrained combines regularization stability")
print("with domain knowledge (negative slopes), giving the best of both worlds.")
print("="*70)
