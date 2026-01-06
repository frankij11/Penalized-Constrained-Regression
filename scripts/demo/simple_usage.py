import penalized_constrained as pcreg
import numpy as np
import pandas as pd


# Generate realistic learning curve data
data = pcreg.generate_correlated_learning_data(
    n_lots=20,
    T1=100,
    target_correlation=0.7,  # Correlation between predictors
    cv_error=0.2,            # 20% CV error
    random_state=42
)

def lc_func(X, params):
    """Learning curve prediction function.

    X can be a DataFrame or numpy array with columns [lot_midpoint, lot_quantity]
    params are [T1, LC, RC] where LC/RC are learning/rate curve percentages
    """
    T1, LC, RC = params
    b = np.log(LC) / np.log(2)
    c = np.log(RC) / np.log(2)
    # Handle both DataFrame and numpy array input
    if hasattr(X, 'lot_midpoint'):
        midpoint = X.lot_midpoint.values
        quantity = X.lot_quantity.values
    else:
        midpoint = X[:, 0]
        quantity = X[:, 1]
    return T1 * (midpoint ** b) * (quantity ** c)

X, y = data['lot_data'][['lot_midpoint', 'lot_quantity']], data['lot_data']['observed_cost']  # Unit space data [midpoint, quantity]
X_log, y_log = data['lot_data'][['log_midpoint', 'log_quantity']], data['log_lot_data']['log_observed_cost']  # Log space data [midpoint, quantity]

true_params = data['params']

# Fit with named coefficients
model = pcreg.PenalizedConstrainedCV(
    coef_names=['T1', 'LC', 'RC'],
    bounds={'T1': (0,None), 'LC': (.7, 1), 'RC': (.7, 1)},
    alphas=np.logspace(-4, .5, 10),
    l1_ratios=[0.0, 0.5, 1.0],
    loss='sspe',
    prediction_fn=lc_func,
    fit_intercept=False,
    penalty_exclude=['T1'],
    x0=[100, 0.99, 0.99]  # Starting point
)
model.fit(X, y)

ols_model = pcreg.PCRegression(
    coef_names=['b', 'r'],
    bounds={'b': (None,None), 'r': (None, None)},
    loss='sse',
    alpha=0.0,
    prediction_fn=None, # OLS
    fit_intercept=True
)

ols_model.fit(X_log, y_log)

print(f"True Parameters:  T1={true_params['T1']}, LC={2**true_params['b']}, RC={2**true_params['c']}")
print(f"PCReg Parameters: T1={model.named_coef_['T1']:.2f}, LC={model.named_coef_['LC']:.4f}, RC={model.named_coef_['RC']:.4f}")
print(f"OLS Parameters:   T1={np.exp(ols_model.intercept_):.2f}, LC={2**ols_model.named_coef_['b']:.4f}, RC={2**ols_model.named_coef_['r']:.4f}")

diag = pcreg.ModelDiagnostics(model, X, y)
# Note: alpha_trace not supported for custom prediction_fn with different param count than X features
report = diag.summary(bootstrap=True, n_bootstrap=100, include_alpha_trace=True)
report.plot_diagnostics()
report.to_html("scripts/demo/simple_usage_diagnostics.html")
report.to_excel("scripts/demo/simple_usage_diagnostics.xlsx")
#report.to_pdf("scripts/demo/simple_usage_diagnostics.pdf")

report_ols = pcreg.ModelDiagnostics(ols_model, np.log(X), np.log(y))
report_ols_summary = report_ols.summary(bootstrap=True, n_bootstrap=100, include_alpha_trace=True)
report_ols_summary.plot_diagnostics()
report_ols_summary.to_html("scripts/demo/simple_usage_ols_diagnostics.html")