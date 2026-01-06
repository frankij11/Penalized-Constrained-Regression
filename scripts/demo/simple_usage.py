import penalized_constrained as pcreg
import numpy as np


# Generate realistic learning curve data
data = pcreg.generate_correlated_learning_data(
    n_lots=20,
    T1=100,
    target_correlation=0.7,  # Correlation between predictors
    cv_error=0.1,            # 10% CV error
    random_state=42
)

def lc_func(X, params):
    """Learning curve prediction function.

    X is a numpy array with columns [midpoint, quantity]
    params are [T1, LC, RC] where LC/RC are learning/rate curve percentages
    """
    T1, LC, RC = params
    b = np.log(LC) / np.log(2)
    c = np.log(RC) / np.log(2)
    midpoint = X[:, 0]
    quantity = X[:, 1]
    return T1 * (midpoint ** b) * (quantity ** c)

X, y = data['X_original'], data['y_original']  # Unit space data [midpoint, quantity]


true_params = data['params']

# Fit with named coefficients
model = pcreg.PenalizedConstrainedCV(
    feature_names=['T1', 'LC', 'RC'],
    bounds={'T1': (0,None), 'LC': (.7, 1), 'RC': (.7, 1)},
    alphas=np.logspace(-4, .5, 10),
    l1_ratios=[0.0, 0.5, 1.0],
    loss='sspe',
    prediction_fn=lc_func,
    fit_intercept=False,
    init=[100, 0.99, 0.99]  # Starting point
)
model.fit(X, y)

# Compare to true values
print(f"True b: {true_params['b']:.4f} ({2**true_params['b']:.4f})" )
print(f"Estimated b: {np.log(model.named_coef_['LC'])/np.log(2):.4f} ({model.named_coef_['LC']:.4f})")
print(f"True c: {true_params['c']:.4f} ({2**true_params['c']:.4f})")
print(f"Estimated c: {np.log(model.named_coef_['RC'])/np.log(2):.4f} ({model.named_coef_['RC']:.4f})")
print(f"True T1: {true_params['T1']:.2f}")
print(f"Estimated T1: {model.named_coef_['T1']:.2f}")

diag = pcreg.ModelDiagnostics(model, X, y)
# Note: alpha_trace not supported for custom prediction_fn with different param count than X features
report = diag.summary(bootstrap=True, n_bootstrap=100, include_alpha_trace=True)
report.plot_diagnostics()
report.to_html("scripts/demo/simple_usage_diagnostics.html")
report.to_excel("scripts/demo/simple_usage_diagnostics.xlsx")
report.to_pdf("scripts/demo/simple_usage_diagnostics.pdf")
print(report)
