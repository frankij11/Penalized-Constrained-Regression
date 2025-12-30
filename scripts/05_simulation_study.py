"""
05_simulation_study.py
======================
Full factorial simulation study for the ICEAA 2026 paper.

Compares multiple regression methods across:
- Sample sizes: {5, 10, 30}
- Correlations: {0, 0.5, 0.9}
- Error CVs: {0.01, 0.1, 0.2}
- Learning slopes: {85%, 90%, 95%}
- Rate slopes: {80%, 85%, 90%}
- Replications: 50 per scenario

Run without arguments for full reproducibility.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings
from itertools import product
from typing import Dict, List, Any
import json

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION - All parameters defined here for reproducibility
# ============================================================================
CONFIG = {
    # Experimental design
    'sample_sizes': [5, 10, 30],
    'correlations': [0.0, 0.5, 0.9],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],  # Converts to b slope
    'rate_effects': [0.80, 0.85, 0.90],     # Converts to c slope
    'n_replications': 50,
    
    # Fixed parameters
    'T1': 100,
    'base_seed': 42,
    
    # Model parameters
    'cv_folds': 3,
    'alpha_grid': np.logspace(-2, 2, 10),
    'l1_ratio_grid': [0.0, 0.5, 1.0],
    
    # Output
    'output_dir': Path(__file__).parent / "output",
    'save_intermediate': True,
}


def get_models(cv_folds: int = 3) -> Dict[str, Any]:
    """
    Define all models to compare.
    
    Returns dict of model_name -> model_instance
    """
    return {
        'OLS': LinearRegression(),
        
        'OLS_LearnOnly': 'special',  # Handled separately (drops rate variable)
        
        'RidgeCV': RidgeCV(
            alphas=np.logspace(-3, 3, 20),
            cv=cv_folds
        ),
        
        'LassoCV': LassoCV(
            alphas=np.logspace(-3, 0, 20),
            cv=cv_folds,
            max_iter=5000
        ),
        
        'BayesianRidge': BayesianRidge(),
        
        'PLS': PLSRegression(n_components=2, scale=False),
        
        'PCA_Linear': Pipeline([
            ('pca', PCA(n_components=2)),
            ('linear', LinearRegression())
        ]),
        
        'ConstrainedOnly': pcreg.PenalizedConstrainedRegression(
            bounds=[(-1, 0), (-0.5, 0)],
            alpha=0.0,
            loss='sspe'
        ),
        
        'ConstrainedCV': pcreg.PenalizedConstrainedCV(
            bounds=[(-1, 0), (-0.5, 0)],
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            cv=cv_folds,
            verbose=0
        ),
        
        'ConstrainedCV_Tight': 'special',  # Bounds set based on true values
        
        'ConstrainedCV_Wrong': pcreg.PenalizedConstrainedCV(
            bounds=[(0, 1), (0, 1)],  # Incorrect: positive bounds
            alphas=CONFIG['alpha_grid'],
            l1_ratios=CONFIG['l1_ratio_grid'],
            loss='sspe',
            cv=cv_folds,
            verbose=0
        ),
    }


def fit_single_model(model_name: str, model, X: np.ndarray, y: np.ndarray,
                     b_true: float, c_true: float) -> Dict:
    """
    Fit a single model and return results.
    """
    result = {
        'model_name': model_name,
        'converged': True,
        'fit_time': 0.0
    }
    
    start_time = time.time()
    
    try:
        if model_name == 'OLS_LearnOnly':
            # Use only first column (learning variable)
            model = LinearRegression()
            model.fit(X[:, [0]], y)
            result['b'] = model.coef_[0]
            result['c'] = 0.0  # Not estimated
            result['intercept'] = model.intercept_
            result['r2'] = model.score(X[:, [0]], y)
            
        elif model_name == 'ConstrainedCV_Tight':
            # Set bounds near true values
            tight_bounds = [
                (b_true - 0.05, b_true + 0.05),
                (c_true - 0.05, c_true + 0.05)
            ]
            model = pcreg.PenalizedConstrainedCV(
                bounds=tight_bounds,
                alphas=CONFIG['alpha_grid'],
                l1_ratios=CONFIG['l1_ratio_grid'],
                loss='sspe',
                cv=CONFIG['cv_folds'],
                verbose=0
            )
            model.fit(X, y)
            result['b'] = model.coef_[0]
            result['c'] = model.coef_[1]
            result['intercept'] = model.intercept_
            result['r2'] = model.score(X, y)
            result['alpha'] = model.alpha_
            result['l1_ratio'] = model.l1_ratio_
            result['converged'] = model.converged_
            
        elif model_name == 'PLS':
            model.fit(X, y)
            result['b'] = model.coef_[0, 0]
            result['c'] = model.coef_[1, 0]
            result['intercept'] = 0.0
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred.ravel()) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result['r2'] = 1 - ss_res / ss_tot
            
        else:
            model.fit(X, y)
            result['b'] = model.coef_[0]
            result['c'] = model.coef_[1]
            
            if hasattr(model, 'intercept_'):
                result['intercept'] = model.intercept_
            else:
                result['intercept'] = 0.0
            
            result['r2'] = model.score(X, y)
            
            # Get regularization parameters if available
            if hasattr(model, 'alpha_'):
                result['alpha'] = model.alpha_
            if hasattr(model, 'l1_ratio_'):
                result['l1_ratio'] = model.l1_ratio_
            if hasattr(model, 'converged_'):
                result['converged'] = model.converged_
    
    except Exception as e:
        result['converged'] = False
        result['error'] = str(e)
        result['b'] = np.nan
        result['c'] = np.nan
        result['intercept'] = np.nan
        result['r2'] = np.nan
    
    result['fit_time'] = time.time() - start_time
    
    # Compute errors
    if not np.isnan(result['b']):
        result['b_error'] = abs(result['b'] - b_true)
        result['c_error'] = abs(result['c'] - c_true)
        result['b_correct_sign'] = result['b'] <= 0
        result['c_correct_sign'] = result['c'] <= 0
    else:
        result['b_error'] = np.nan
        result['c_error'] = np.nan
        result['b_correct_sign'] = False
        result['c_correct_sign'] = False
    
    return result


def run_single_scenario(
    n_lots: int,
    correlation: float,
    cv_error: float,
    learning_rate: float,
    rate_effect: float,
    replication: int,
    config: dict
) -> List[Dict]:
    """
    Run all models on a single scenario.
    """
    # Convert rates to slopes
    b_true = pcreg.learning_rate_to_slope(learning_rate)
    c_true = pcreg.learning_rate_to_slope(rate_effect)
    
    # Generate seed deterministically
    seed = config['base_seed'] + hash((n_lots, correlation, cv_error, 
                                        learning_rate, rate_effect, replication)) % (2**31)
    
    # Generate data
    data = pcreg.generate_correlated_learning_data(
        n_lots=n_lots,
        T1=config['T1'],
        b=b_true,
        c=c_true,
        target_correlation=correlation,
        cv_error=cv_error,
        random_state=seed
    )
    
    X, y = data['X'], data['y']
    actual_corr = data['actual_correlation']
    
    # Get models
    models = get_models(config['cv_folds'])
    
    # Fit all models
    results = []
    for model_name, model in models.items():
        result = fit_single_model(model_name, model, X, y, b_true, c_true)
        
        # Add scenario metadata
        result.update({
            'n_lots': n_lots,
            'target_correlation': correlation,
            'actual_correlation': actual_corr,
            'cv_error': cv_error,
            'learning_rate': learning_rate,
            'rate_effect': rate_effect,
            'b_true': b_true,
            'c_true': c_true,
            'replication': replication,
            'seed': seed
        })
        
        results.append(result)
    
    return results


def run_full_simulation(config: dict) -> pd.DataFrame:
    """
    Run the complete factorial simulation.
    """
    # Calculate total scenarios
    n_scenarios = (
        len(config['sample_sizes']) *
        len(config['correlations']) *
        len(config['cv_errors']) *
        len(config['learning_rates']) *
        len(config['rate_effects']) *
        config['n_replications']
    )
    n_models = len(get_models())
    
    print(f"Total scenarios: {n_scenarios}")
    print(f"Models per scenario: {n_models}")
    print(f"Total model fits: {n_scenarios * n_models}")
    print()
    
    all_results = []
    start_time = time.time()
    scenario_count = 0
    
    # Iterate over all factor combinations
    factor_combinations = product(
        config['sample_sizes'],
        config['correlations'],
        config['cv_errors'],
        config['learning_rates'],
        config['rate_effects'],
        range(config['n_replications'])
    )
    
    for n_lots, corr, cv, lr, re, rep in factor_combinations:
        scenario_count += 1
        
        if scenario_count % 100 == 0:
            elapsed = time.time() - start_time
            rate = scenario_count / elapsed
            remaining = (n_scenarios - scenario_count) / rate
            print(f"Progress: {scenario_count}/{n_scenarios} ({100*scenario_count/n_scenarios:.1f}%) "
                  f"- ETA: {remaining/60:.1f} min")
        
        results = run_single_scenario(n_lots, corr, cv, lr, re, rep, config)
        all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print(f"Average per scenario: {elapsed_total/n_scenarios:.3f} seconds")
    
    return df


def analyze_results(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics from simulation results.
    """
    analysis = {}
    
    # Filter to converged models only
    df_conv = df[df['converged'] == True].copy()
    
    # Overall statistics by method
    method_stats = df_conv.groupby('model_name').agg({
        'b_error': ['mean', 'std'],
        'c_error': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'b_correct_sign': 'mean',
        'c_correct_sign': 'mean',
        'fit_time': 'mean',
        'converged': 'sum'
    }).round(4)
    
    analysis['method_stats'] = method_stats
    
    # Winner analysis - which method has lowest error most often
    winners = {}
    for metric in ['b_error', 'c_error']:
        # Group by scenario (excluding model_name)
        scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 
                        'learning_rate', 'rate_effect', 'replication']
        
        win_counts = {}
        for name, group in df_conv.groupby(scenario_cols):
            if len(group) > 0:
                winner = group.loc[group[metric].idxmin(), 'model_name']
                win_counts[winner] = win_counts.get(winner, 0) + 1
        
        winners[metric] = win_counts
    
    analysis['winners'] = winners
    
    # Statistics by design factors
    for factor in ['n_lots', 'target_correlation', 'cv_error']:
        factor_stats = df_conv.groupby(['model_name', factor]).agg({
            'b_error': 'mean',
            'c_error': 'mean',
            'r2': 'mean'
        }).round(4)
        analysis[f'by_{factor}'] = factor_stats
    
    return analysis


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create and save visualization plots.
    """
    df_conv = df[df['converged'] == True].copy()
    
    # Figure 1: Overall method comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods_order = ['OLS', 'RidgeCV', 'LassoCV', 'BayesianRidge', 
                     'ConstrainedOnly', 'ConstrainedCV', 'ConstrainedCV_Tight']
    
    # Filter to main methods
    df_main = df_conv[df_conv['model_name'].isin(methods_order)]
    
    for ax, metric, title in zip(
        axes,
        ['b_error', 'c_error', 'r2'],
        ['Learning Slope Error', 'Rate Slope Error', 'R²']
    ):
        data = [df_main[df_main['model_name'] == m][metric].dropna() for m in methods_order]
        bp = ax.boxplot(data, labels=methods_order, patch_artist=True)
        ax.set_xticklabels(methods_order, rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Color constrained methods
        colors = ['lightblue' if 'Constrained' not in m else 'lightgreen' for m in methods_order]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Effect of correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, metric, title in zip(axes, ['b_error', 'c_error'], ['LC Slope Error', 'RC Slope Error']):
        for method in ['OLS', 'RidgeCV', 'ConstrainedCV']:
            df_m = df_conv[df_conv['model_name'] == method]
            means = df_m.groupby('target_correlation')[metric].mean()
            ax.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Target Correlation')
        ax.set_ylabel(f'Mean {title}')
        ax.set_title(f'{title} vs Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correlation_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Effect of sample size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, metric, title in zip(axes, ['b_error', 'c_error'], ['LC Slope Error', 'RC Slope Error']):
        for method in ['OLS', 'RidgeCV', 'ConstrainedCV']:
            df_m = df_conv[df_conv['model_name'] == method]
            means = df_m.groupby('n_lots')[metric].mean()
            ax.plot(means.index, means.values, 'o-', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sample Size (N lots)')
        ax.set_ylabel(f'Mean {title}')
        ax.set_title(f'{title} vs Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'sample_size_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Correct sign rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sign_rates = df_conv.groupby('model_name').agg({
        'b_correct_sign': 'mean',
        'c_correct_sign': 'mean'
    })
    
    x = np.arange(len(sign_rates))
    width = 0.35
    
    ax.bar(x - width/2, sign_rates['b_correct_sign'] * 100, width, label='LC slope (b)', color='steelblue')
    ax.bar(x + width/2, sign_rates['c_correct_sign'] * 100, width, label='RC slope (c)', color='coral')
    
    ax.set_ylabel('Correct Sign Rate (%)')
    ax.set_title('Percentage of Simulations with Correct Coefficient Sign')
    ax.set_xticks(x)
    ax.set_xticklabels(sign_rates.index, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correct_sign_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def print_summary(df: pd.DataFrame, analysis: Dict):
    """
    Print summary of simulation results.
    """
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal observations: {len(df)}")
    print(f"Converged: {df['converged'].sum()} ({100*df['converged'].mean():.1f}%)")
    
    print("\n" + "-"*80)
    print("OVERALL METHOD PERFORMANCE (mean ± std)")
    print("-"*80)
    
    stats = analysis['method_stats']
    print(f"{'Method':<20} {'b error':>12} {'c error':>12} {'R²':>10} {'b sign%':>10} {'c sign%':>10}")
    print("-"*80)
    
    for method in stats.index:
        row = stats.loc[method]
        b_err = f"{row[('b_error', 'mean')]:.4f}±{row[('b_error', 'std')]:.4f}"
        c_err = f"{row[('c_error', 'mean')]:.4f}±{row[('c_error', 'std')]:.4f}"
        r2 = f"{row[('r2', 'mean')]:.3f}"
        b_sign = f"{100*row[('b_correct_sign', 'mean')]:.0f}%"
        c_sign = f"{100*row[('c_correct_sign', 'mean')]:.0f}%"
        print(f"{method:<20} {b_err:>12} {c_err:>12} {r2:>10} {b_sign:>10} {c_sign:>10}")
    
    print("\n" + "-"*80)
    print("WINNER COUNTS (lowest error)")
    print("-"*80)
    
    for metric, counts in analysis['winners'].items():
        print(f"\n{metric}:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for method, count in sorted_counts[:5]:
            print(f"  {method}: {count}")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("SIMULATION STUDY: Penalized-Constrained Regression")
    print("ICEAA 2026 Paper")
    print("="*80)
    print()
    print("Configuration:")
    print(f"  Sample sizes: {CONFIG['sample_sizes']}")
    print(f"  Correlations: {CONFIG['correlations']}")
    print(f"  CV errors: {CONFIG['cv_errors']}")
    print(f"  Learning rates: {CONFIG['learning_rates']}")
    print(f"  Rate effects: {CONFIG['rate_effects']}")
    print(f"  Replications: {CONFIG['n_replications']}")
    print()
    
    # Create output directory
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    # Run simulation
    print("Starting simulation...")
    print()
    
    results_df = run_full_simulation(CONFIG)
    
    # Save raw results
    results_df.to_csv(output_dir / 'simulation_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'simulation_results.csv'}")
    
    # Save config
    config_save = {k: str(v) if isinstance(v, (Path, np.ndarray)) else v 
                   for k, v in CONFIG.items()}
    with open(output_dir / 'simulation_config.json', 'w') as f:
        json.dump(config_save, f, indent=2, default=str)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results_df)
    
    # Print summary
    print_summary(results_df, analysis)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, output_dir)
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)
