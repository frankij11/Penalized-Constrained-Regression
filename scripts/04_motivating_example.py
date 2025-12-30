"""
04_motivating_example.py
========================
Find and analyze the "motivating example": a dataset where:
- OLS has good R² but WRONG coefficient signs
- Ridge doesn't fully fix the problem
- Constrained-only may miss optimal penalty
- Penalized+Constrained gives correct signs AND good fit

This script automatically searches for such cases and reports frequency.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


def evaluate_dataset(data, verbose=False):
    """
    Evaluate a single dataset with multiple methods.
    
    Returns dict with method results and whether it's a "motivating example".
    """
    X, y = data['X'], data['y']
    b_true, c_true = data['params']['b'], data['params']['c']
    
    results = {}
    
    # OLS
    ols = LinearRegression().fit(X, y)
    results['OLS'] = {
        'b': ols.coef_[0],
        'c': ols.coef_[1],
        'r2': ols.score(X, y),
        'b_correct_sign': ols.coef_[0] <= 0,
        'c_correct_sign': ols.coef_[1] <= 0,
        'b_error': abs(ols.coef_[0] - b_true),
        'c_error': abs(ols.coef_[1] - c_true)
    }
    
    # RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X, y)
    results['RidgeCV'] = {
        'b': ridge.coef_[0],
        'c': ridge.coef_[1],
        'r2': ridge.score(X, y),
        'alpha': ridge.alpha_,
        'b_correct_sign': ridge.coef_[0] <= 0,
        'c_correct_sign': ridge.coef_[1] <= 0,
        'b_error': abs(ridge.coef_[0] - b_true),
        'c_error': abs(ridge.coef_[1] - c_true)
    }
    
    # Constrained Only (alpha=0)
    constrained = pcreg.PenalizedConstrainedRegression(
        bounds=[(-1, 0), (-0.5, 0)],
        alpha=0.0,
        loss='sspe'
    ).fit(X, y)
    results['Constrained'] = {
        'b': constrained.coef_[0],
        'c': constrained.coef_[1],
        'r2': constrained.score(X, y),
        'b_correct_sign': True,
        'c_correct_sign': True,
        'b_error': abs(constrained.coef_[0] - b_true),
        'c_error': abs(constrained.coef_[1] - c_true)
    }
    
    # Penalized + Constrained CV
    pc = pcreg.PenalizedConstrainedCV(
        bounds=[(-1, 0), (-0.5, 0)],
        alphas=np.logspace(-2, 1, 10),
        l1_ratios=[0.0, 0.5, 1.0],
        loss='sspe',
        cv=3,
        verbose=0
    ).fit(X, y)
    results['Pen+Constr'] = {
        'b': pc.coef_[0],
        'c': pc.coef_[1],
        'r2': pc.score(X, y),
        'alpha': pc.alpha_,
        'l1_ratio': pc.l1_ratio_,
        'b_correct_sign': True,
        'c_correct_sign': True,
        'b_error': abs(pc.coef_[0] - b_true),
        'c_error': abs(pc.coef_[1] - c_true)
    }
    
    # Check if this is a "motivating example"
    is_motivating = (
        results['OLS']['r2'] > 0.7 and  # Good fit
        (not results['OLS']['b_correct_sign'] or not results['OLS']['c_correct_sign']) and  # Wrong sign
        (not results['RidgeCV']['b_correct_sign'] or not results['RidgeCV']['c_correct_sign']) and  # Ridge doesn't fix
        results['Pen+Constr']['r2'] > 0.6  # PC still has decent fit
    )
    
    # Alternative: also count cases where signs are right but errors are large
    ols_large_error = (results['OLS']['b_error'] > 0.05 or results['OLS']['c_error'] > 0.05)
    pc_lower_error = (
        results['Pen+Constr']['b_error'] < results['OLS']['b_error'] and
        results['Pen+Constr']['c_error'] < results['OLS']['c_error']
    )
    
    is_improvement = results['OLS']['r2'] > 0.6 and ols_large_error and pc_lower_error
    
    return {
        'results': results,
        'is_motivating': is_motivating,
        'is_improvement': is_improvement,
        'data': data
    }


def search_for_motivating_example(
    n_search=1000,
    n_lots=15,
    correlation_range=(0.6, 0.9),
    cv_error_range=(0.1, 0.25),
    verbose=True
):
    """
    Search for motivating examples through random sampling.
    
    Returns the best example and statistics on how often problems occur.
    """
    motivating_examples = []
    improvement_examples = []
    all_results = []
    
    for i in range(n_search):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Searched {i+1}/{n_search}...")
        
        # Random parameters
        seed = 1000 + i
        np.random.seed(seed)
        
        corr = np.random.uniform(*correlation_range)
        cv_error = np.random.uniform(*cv_error_range)
        
        # Generate data
        data = pcreg.generate_correlated_learning_data(
            n_lots=n_lots,
            T1=100,
            target_correlation=corr,
            cv_error=cv_error,
            random_state=seed
        )
        data['seed'] = seed
        
        # Evaluate
        eval_result = evaluate_dataset(data)
        eval_result['seed'] = seed
        eval_result['correlation'] = data['actual_correlation']
        eval_result['cv_error'] = cv_error
        
        all_results.append(eval_result)
        
        if eval_result['is_motivating']:
            motivating_examples.append(eval_result)
        
        if eval_result['is_improvement']:
            improvement_examples.append(eval_result)
    
    # Statistics
    n_ols_wrong_sign = sum(
        1 for r in all_results 
        if not r['results']['OLS']['b_correct_sign'] or not r['results']['OLS']['c_correct_sign']
    )
    n_ridge_wrong_sign = sum(
        1 for r in all_results 
        if not r['results']['RidgeCV']['b_correct_sign'] or not r['results']['RidgeCV']['c_correct_sign']
    )
    
    stats = {
        'n_searched': n_search,
        'n_motivating': len(motivating_examples),
        'n_improvement': len(improvement_examples),
        'pct_ols_wrong_sign': 100 * n_ols_wrong_sign / n_search,
        'pct_ridge_wrong_sign': 100 * n_ridge_wrong_sign / n_search,
        'pct_motivating': 100 * len(motivating_examples) / n_search,
        'pct_improvement': 100 * len(improvement_examples) / n_search
    }
    
    return motivating_examples, improvement_examples, all_results, stats


def visualize_motivating_example(eval_result, save_path=None):
    """Create comprehensive visualization of a motivating example."""
    data = eval_result['data']
    results = eval_result['results']
    
    X, y = data['X'], data['y']
    b_true, c_true = data['params']['b'], data['params']['c']
    
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Data visualization
    ax1 = fig.add_subplot(2, 2, 1)
    cumulative = np.cumsum(data['lot_quantities'])
    ax1.scatter(cumulative, data['y_original'], s=50, alpha=0.7, label='Observed')
    ax1.plot(cumulative, data['true_costs'], 'r--', linewidth=2, label='True')
    ax1.set_xlabel('Cumulative Units', fontsize=11)
    ax1.set_ylabel('Cost', fontsize=11)
    ax1.set_title('Learning Curve Data', fontsize=12)
    ax1.legend()
    
    # Plot 2: Coefficient comparison
    ax2 = fig.add_subplot(2, 2, 2)
    methods = ['OLS', 'RidgeCV', 'Constrained', 'Pen+Constr']
    x_pos = np.arange(len(methods))
    width = 0.35
    
    b_values = [results[m]['b'] for m in methods]
    c_values = [results[m]['c'] for m in methods]
    
    bars1 = ax2.bar(x_pos - width/2, b_values, width, label='LC slope (b)', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, c_values, width, label='RC slope (c)', color='coral', alpha=0.8)
    
    ax2.axhline(y=b_true, color='steelblue', linestyle='--', linewidth=2, label=f'True b={b_true:.3f}')
    ax2.axhline(y=c_true, color='coral', linestyle='--', linewidth=2, label=f'True c={c_true:.3f}')
    ax2.axhline(y=0, color='black', linewidth=1)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylabel('Coefficient Value', fontsize=11)
    ax2.set_title('Coefficient Estimates by Method', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-0.4, 0.15)
    
    # Highlight wrong signs with red boxes
    for i, m in enumerate(methods):
        if not results[m]['b_correct_sign']:
            ax2.add_patch(plt.Rectangle((i - width, 0), width, results[m]['b'], 
                                         fill=False, edgecolor='red', linewidth=2))
        if not results[m]['c_correct_sign']:
            ax2.add_patch(plt.Rectangle((i, 0), width, results[m]['c'],
                                         fill=False, edgecolor='red', linewidth=2))
    
    # Plot 3: R² comparison
    ax3 = fig.add_subplot(2, 2, 3)
    r2_values = [results[m]['r2'] for m in methods]
    colors = ['red' if not results[m]['b_correct_sign'] or not results[m]['c_correct_sign'] 
              else 'green' for m in methods]
    bars = ax3.bar(methods, r2_values, color=colors, alpha=0.7)
    ax3.set_ylabel('R²', fontsize=11)
    ax3.set_title('Model Fit (R²) - Red = Wrong Sign', fontsize=12)
    ax3.set_ylim(0, 1)
    
    for bar, r2 in zip(bars, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{r2:.3f}', ha='center', fontsize=10)
    
    # Plot 4: Error comparison
    ax4 = fig.add_subplot(2, 2, 4)
    b_errors = [results[m]['b_error'] for m in methods]
    c_errors = [results[m]['c_error'] for m in methods]
    
    bars1 = ax4.bar(x_pos - width/2, b_errors, width, label='b error', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, c_errors, width, label='c error', color='coral', alpha=0.8)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, fontsize=10)
    ax4.set_ylabel('Absolute Error', fontsize=11)
    ax4.set_title('Coefficient Errors (lower is better)', fontsize=12)
    ax4.legend()
    
    # Add summary text
    corr = data['actual_correlation']
    cv = data.get('cv_error', 'N/A')
    fig.suptitle(f'Motivating Example: OLS Good Fit (R²={results["OLS"]["r2"]:.2f}) but Wrong Signs\n'
                 f'Correlation={corr:.2f}, N={len(y)} lots', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def print_example_summary(eval_result):
    """Print detailed summary of a motivating example."""
    results = eval_result['results']
    data = eval_result['data']
    
    print("\n" + "="*70)
    print("MOTIVATING EXAMPLE SUMMARY")
    print("="*70)
    print(f"Seed: {eval_result.get('seed', 'N/A')}")
    print(f"N lots: {len(data['y'])}")
    print(f"Correlation: {data['actual_correlation']:.3f}")
    print(f"True parameters: b={data['params']['b']:.4f}, c={data['params']['c']:.4f}")
    
    print("\n" + "-"*70)
    print(f"{'Method':<15} {'b':>10} {'c':>10} {'R²':>8} {'b sign':>8} {'c sign':>8}")
    print("-"*70)
    
    for method in ['OLS', 'RidgeCV', 'Constrained', 'Pen+Constr']:
        r = results[method]
        b_sign = "✓" if r['b_correct_sign'] else "✗ WRONG"
        c_sign = "✓" if r['c_correct_sign'] else "✗ WRONG"
        print(f"{method:<15} {r['b']:>10.4f} {r['c']:>10.4f} {r['r2']:>8.4f} {b_sign:>8} {c_sign:>8}")
    
    print("-"*70)
    
    # Key insight
    print("\nKEY INSIGHT:")
    if not results['OLS']['b_correct_sign'] or not results['OLS']['c_correct_sign']:
        print("  • OLS produces WRONG coefficient signs despite good R²")
    if not results['RidgeCV']['b_correct_sign'] or not results['RidgeCV']['c_correct_sign']:
        print("  • Ridge regularization alone does NOT fix the sign problem")
    print("  • Penalized+Constrained enforces correct signs while maintaining fit")
    print("="*70)


# ============================================================================
# MAIN: Search and Analyze
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("SEARCHING FOR MOTIVATING EXAMPLES")
    print("Good R² + Wrong Coefficients → Penalized-Constrained Fixes It")
    print("="*70)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Search for examples
    print("\nSearching for motivating examples...")
    motivating, improvement, all_results, stats = search_for_motivating_example(
        n_search=500,
        n_lots=15,
        correlation_range=(0.6, 0.9),
        cv_error_range=(0.1, 0.25),
        verbose=True
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("SEARCH RESULTS")
    print("="*70)
    print(f"Datasets searched: {stats['n_searched']}")
    print(f"OLS wrong sign: {stats['pct_ols_wrong_sign']:.1f}%")
    print(f"Ridge wrong sign: {stats['pct_ridge_wrong_sign']:.1f}%")
    print(f"Perfect motivating examples: {stats['n_motivating']} ({stats['pct_motivating']:.1f}%)")
    print(f"PC improvement examples: {stats['n_improvement']} ({stats['pct_improvement']:.1f}%)")
    
    # Show best motivating example
    if motivating:
        print("\n" + "="*70)
        print("BEST MOTIVATING EXAMPLE")
        print("="*70)
        
        # Pick example with highest OLS R² but wrong signs
        best = max(motivating, key=lambda x: x['results']['OLS']['r2'])
        print_example_summary(best)
        
        # Visualize
        fig = visualize_motivating_example(best, save_path=output_dir / "motivating_example.png")
        
        # Save the dataset
        save_df = pd.DataFrame({
            'lot_midpoint': best['data']['lot_midpoints'],
            'lot_quantity': best['data']['lot_quantities'],
            'log_midpoint': best['data']['X'][:, 0],
            'log_quantity': best['data']['X'][:, 1],
            'log_cost': best['data']['y'],
            'cost': best['data']['y_original']
        })
        save_df.to_csv(output_dir / "motivating_example_data.csv", index=False)
        print(f"\nData saved to: {output_dir / 'motivating_example_data.csv'}")
    
    elif improvement:
        print("\n" + "="*70)
        print("BEST IMPROVEMENT EXAMPLE (signs correct but errors reduced)")
        print("="*70)
        best = max(improvement, key=lambda x: 
                   x['results']['OLS']['b_error'] + x['results']['OLS']['c_error'])
        print_example_summary(best)
        fig = visualize_motivating_example(best, save_path=output_dir / "improvement_example.png")
    
    else:
        print("\nNo motivating examples found in this search.")
        print("Try increasing n_search or adjusting correlation/error ranges.")
    
    print("\n" + "="*70)
    print("Search completed!")
    print("="*70)
    
    plt.show()
