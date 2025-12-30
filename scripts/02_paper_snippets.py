"""
02_paper_snippets.py
====================
Code snippets for the ICEAA 2026 paper:
"Small Data, Big Problems: Can Constraints and Penalties Save Regression?"

These snippets are designed to be copy-pasted into the paper with minimal modification.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import penalized_constrained as pcreg

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# SNIPPET 1: Basic Usage Example
# ============================================================================
def snippet_basic_usage():
    """Basic usage example for Section 3 of the paper."""
    print("\n" + "="*60)
    print("SNIPPET 1: Basic Usage")
    print("="*60)
    
    code = '''
from penalized_constrained import PenalizedConstrainedCV
import numpy as np

# Fit with cross-validated hyperparameter selection
model = PenalizedConstrainedCV(
    alphas=np.logspace(-3, 2, 10),
    l1_ratios=[0.0, 0.5, 1.0],
    bounds=[(-1, 0), (-0.5, 0)],  # LC slope ≤ 0, RC slope ≤ 0
    loss='sspe',                   # MUPE-consistent
    cv=5
)
model.fit(X_log, y_log)

print(f"Best α: {model.alpha_:.4f}")
print(f"Best l1_ratio: {model.l1_ratio_:.2f}")
print(f"Coefficients: {model.coef_}")
'''
    print(code)
    return code


# ============================================================================
# SNIPPET 2: Learning Curve with Named Coefficients
# ============================================================================
def snippet_learning_curve():
    """Learning curve example with named coefficients."""
    print("\n" + "="*60)
    print("SNIPPET 2: Learning Curve Analysis")
    print("="*60)
    
    # Generate data
    data = pcreg.generate_correlated_learning_data(
        n_lots=20,
        T1=100,
        target_correlation=0.7,
        cv_error=0.1,
        random_state=42
    )
    X, y = data['X'], data['y']
    
    # Fit model
    model = pcreg.PenalizedConstrainedCV(
        feature_names=['LC', 'RC'],
        bounds={'LC': (-1, 0), 'RC': (-0.5, 0)},
        alphas=np.logspace(-2, 1, 10),
        l1_ratios=[0.0, 0.5, 1.0],
        loss='sspe',
        cv=5
    )
    model.fit(X, y)
    
    print(f"True LC slope (b): {data['params']['b']:.4f}")
    print(f"Estimated LC:      {model.named_coef_['LC']:.4f}")
    print(f"True RC slope (c): {data['params']['c']:.4f}")
    print(f"Estimated RC:      {model.named_coef_['RC']:.4f}")
    print(f"Best α: {model.alpha_:.4f}, l1_ratio: {model.l1_ratio_:.2f}")
    
    return model, data


# ============================================================================
# SNIPPET 3: Multicollinearity Diagnostics
# ============================================================================
def snippet_diagnostics():
    """Multicollinearity diagnostics example."""
    print("\n" + "="*60)
    print("SNIPPET 3: Multicollinearity Diagnostics")
    print("="*60)
    
    # Generate highly correlated data
    data = pcreg.generate_correlated_learning_data(
        n_lots=20,
        target_correlation=0.85,
        random_state=42
    )
    X = data['X']
    
    # Run diagnostics
    pcreg.print_multicollinearity_report(X, feature_names=['LC', 'RC'])
    
    return data


# ============================================================================
# SNIPPET 4: GDF-Adjusted Statistics
# ============================================================================
def snippet_gdf_statistics():
    """GDF-adjusted fit statistics example."""
    print("\n" + "="*60)
    print("SNIPPET 4: GDF-Adjusted Statistics")
    print("="*60)
    
    # Generate and fit
    data = pcreg.generate_correlated_learning_data(n_lots=20, random_state=42)
    X, y = data['X'], data['y']
    
    model = pcreg.PenalizedConstrainedRegression(
        bounds=[(-1, 0), (-0.5, 0)],
        alpha=0.1,
        l1_ratio=0.5,
        loss='sspe'
    )
    model.fit(X, y)
    
    # Compute diagnostics
    diag = pcreg.ModelDiagnostics(model, X, y, gdf_method='hu')
    diag.summary()
    
    # Also show Gaines method
    diag_gaines = pcreg.ModelDiagnostics(model, X, y, gdf_method='gaines')
    print(f"\nGaines GDF: {diag_gaines.gdf:.1f} (only binding constraints counted)")
    
    return diag


# ============================================================================
# SNIPPET 5: Comparison Table Generation
# ============================================================================
def snippet_comparison_table():
    """Generate comparison table for different methods."""
    print("\n" + "="*60)
    print("SNIPPET 5: Method Comparison Table")
    print("="*60)
    
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    
    # Generate data
    data = pcreg.generate_correlated_learning_data(
        n_lots=30,
        target_correlation=0.7,
        cv_error=0.1,
        random_state=42
    )
    X, y = data['X'], data['y']
    b_true, c_true = data['params']['b'], data['params']['c']
    
    # Methods to compare
    results = []
    
    # OLS
    ols = LinearRegression().fit(X, y)
    results.append({
        'Method': 'OLS',
        'b': ols.coef_[0],
        'c': ols.coef_[1],
        'b_error': abs(ols.coef_[0] - b_true),
        'c_error': abs(ols.coef_[1] - c_true),
        'R²': ols.score(X, y),
        'Correct_Signs': ols.coef_[0] <= 0 and ols.coef_[1] <= 0
    })
    
    # RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X, y)
    results.append({
        'Method': 'RidgeCV',
        'b': ridge.coef_[0],
        'c': ridge.coef_[1],
        'b_error': abs(ridge.coef_[0] - b_true),
        'c_error': abs(ridge.coef_[1] - c_true),
        'R²': ridge.score(X, y),
        'Correct_Signs': ridge.coef_[0] <= 0 and ridge.coef_[1] <= 0
    })
    
    # LassoCV
    lasso = LassoCV(alphas=np.logspace(-3, 0, 20)).fit(X, y)
    results.append({
        'Method': 'LassoCV',
        'b': lasso.coef_[0],
        'c': lasso.coef_[1],
        'b_error': abs(lasso.coef_[0] - b_true),
        'c_error': abs(lasso.coef_[1] - c_true),
        'R²': lasso.score(X, y),
        'Correct_Signs': lasso.coef_[0] <= 0 and lasso.coef_[1] <= 0
    })
    
    # Constrained Only
    constrained = pcreg.PenalizedConstrainedRegression(
        bounds=[(-1, 0), (-0.5, 0)],
        alpha=0.0,
        loss='sspe'
    ).fit(X, y)
    results.append({
        'Method': 'Constrained',
        'b': constrained.coef_[0],
        'c': constrained.coef_[1],
        'b_error': abs(constrained.coef_[0] - b_true),
        'c_error': abs(constrained.coef_[1] - c_true),
        'R²': constrained.score(X, y),
        'Correct_Signs': True
    })
    
    # Penalized + Constrained CV
    pc_cv = pcreg.PenalizedConstrainedCV(
        bounds=[(-1, 0), (-0.5, 0)],
        alphas=np.logspace(-2, 1, 10),
        l1_ratios=[0.0, 0.5, 1.0],
        loss='sspe',
        cv=5
    ).fit(X, y)
    results.append({
        'Method': 'Pen+Constr CV',
        'b': pc_cv.coef_[0],
        'c': pc_cv.coef_[1],
        'b_error': abs(pc_cv.coef_[0] - b_true),
        'c_error': abs(pc_cv.coef_[1] - c_true),
        'R²': pc_cv.score(X, y),
        'Correct_Signs': True
    })
    
    # Create table
    df = pd.DataFrame(results)
    print(f"\nTrue values: b = {b_true:.4f}, c = {c_true:.4f}")
    print(f"Correlation: {data['actual_correlation']:.2f}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    return df


# ============================================================================
# SNIPPET 6: Visualization for Paper
# ============================================================================
def snippet_visualization(save_path=None):
    """Create visualization comparing methods."""
    print("\n" + "="*60)
    print("SNIPPET 6: Visualization")
    print("="*60)
    
    from sklearn.linear_model import LinearRegression, RidgeCV
    
    # Generate data
    data = pcreg.generate_correlated_learning_data(
        n_lots=20,
        target_correlation=0.75,
        cv_error=0.15,
        random_state=123
    )
    X, y = data['X'], data['y']
    b_true, c_true = data['params']['b'], data['params']['c']
    
    # Fit models
    ols = LinearRegression().fit(X, y)
    ridge = RidgeCV().fit(X, y)
    pc = pcreg.PenalizedConstrainedCV(
        bounds=[(-1, 0), (-0.5, 0)],
        loss='sspe'
    ).fit(X, y)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    methods = [
        ('OLS', ols.coef_),
        ('RidgeCV', ridge.coef_),
        ('Pen+Constr', pc.coef_)
    ]
    
    for ax, (name, coef) in zip(axes, methods):
        # Bar chart of coefficients
        x_pos = [0, 1]
        bars = ax.bar(x_pos, [coef[0], coef[1]], color=['steelblue', 'coral'], alpha=0.7)
        
        # True values as horizontal lines
        ax.axhline(y=b_true, color='steelblue', linestyle='--', linewidth=2, label='True b')
        ax.axhline(y=c_true, color='coral', linestyle='--', linewidth=2, label='True c')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['LC (b)', 'RC (c)'])
        ax.set_ylabel('Coefficient Value')
        ax.set_title(name)
        ax.set_ylim(-0.3, 0.1)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Add error annotation
        b_err = abs(coef[0] - b_true)
        c_err = abs(coef[1] - c_true)
        ax.text(0.5, 0.02, f'Errors: b={b_err:.3f}, c={c_err:.3f}', 
                transform=ax.transAxes, ha='center', fontsize=9)
    
    axes[0].legend(loc='upper right', fontsize=8)
    plt.suptitle(f'Coefficient Estimates (True: b={b_true:.3f}, c={c_true:.3f})', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# RUN ALL SNIPPETS
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("PAPER SNIPPETS: ICEAA 2026")
    print("Small Data, Big Problems: Can Constraints and Penalties Save Regression?")
    print("="*70)
    
    snippet_basic_usage()
    snippet_learning_curve()
    snippet_diagnostics()
    snippet_gdf_statistics()
    snippet_comparison_table()
    
    # Save visualization
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    snippet_visualization(save_path=output_dir / "method_comparison.png")
    
    print("\n" + "="*70)
    print("All snippets completed successfully!")
    print("="*70)
