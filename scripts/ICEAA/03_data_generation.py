"""
03_data_generation.py
=====================
Helper script for generating correlated learning curve data.

This script provides utilities and examples for creating realistic
learning curve datasets with controlled correlation between predictors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import penalized_constrained as pcreg


def generate_single_dataset(
    n_lots=20,
    T1=100,
    learning_rate=0.90,
    rate_effect=0.95,
    target_correlation=0.5,
    cv_error=0.1,
    random_state=None
):
    """
    Generate a single learning curve dataset with specified parameters.
    
    Parameters
    ----------
    n_lots : int
        Number of lots.
    T1 : float
        First unit cost.
    learning_rate : float
        Learning rate (e.g., 0.90 for 90%).
    rate_effect : float
        Rate effect (e.g., 0.95 for 95%).
    target_correlation : float
        Target correlation between log(midpoint) and log(quantity).
    cv_error : float
        Coefficient of variation for error.
    random_state : int
        Random seed.
        
    Returns
    -------
    data : dict
        Dictionary with X, y, parameters, etc.
    """
    b = pcreg.learning_rate_to_slope(learning_rate)
    c = pcreg.learning_rate_to_slope(rate_effect)
    
    data = pcreg.generate_correlated_learning_data(
        n_lots=n_lots,
        T1=T1,
        b=b,
        c=c,
        target_correlation=target_correlation,
        cv_error=cv_error,
        random_state=random_state
    )
    
    return data


def generate_dataset_grid(
    n_lots_list=[10, 20, 30],
    correlation_list=[0.0, 0.5, 0.9],
    cv_error_list=[0.05, 0.1, 0.2],
    n_replications=50,
    base_seed=42
):
    """
    Generate a grid of datasets for simulation study.
    
    Parameters
    ----------
    n_lots_list : list
        Sample sizes to test.
    correlation_list : list
        Correlation levels to test.
    cv_error_list : list
        Error levels to test.
    n_replications : int
        Number of replications per scenario.
    base_seed : int
        Base random seed for reproducibility.
        
    Returns
    -------
    datasets : list of dict
        List of dataset dictionaries with metadata.
    """
    datasets = []
    
    for n_lots in n_lots_list:
        for corr in correlation_list:
            for cv in cv_error_list:
                for rep in range(n_replications):
                    seed = base_seed + hash((n_lots, corr, cv, rep)) % (2**31)
                    
                    data = generate_single_dataset(
                        n_lots=n_lots,
                        target_correlation=corr,
                        cv_error=cv,
                        random_state=seed
                    )
                    
                    data['metadata'] = {
                        'n_lots': n_lots,
                        'target_correlation': corr,
                        'cv_error': cv,
                        'replication': rep,
                        'seed': seed
                    }
                    
                    datasets.append(data)
    
    print(f"Generated {len(datasets)} datasets")
    print(f"  n_lots: {n_lots_list}")
    print(f"  correlations: {correlation_list}")
    print(f"  cv_errors: {cv_error_list}")
    print(f"  replications: {n_replications}")
    
    return datasets


def save_dataset(data, filepath):
    """Save dataset to CSV file."""
    df = pd.DataFrame({
        'lot_midpoint': data['lot_midpoints'],
        'lot_quantity': data['lot_quantities'],
        'log_midpoint': data['X'][:, 0],
        'log_quantity': data['X'][:, 1],
        'log_cost': data['y'],
        'cost': data['y_original'],
        'true_cost': data['true_costs']
    })
    
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    return df


def load_dataset(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    
    data = {
        'X': df[['log_midpoint', 'log_quantity']].values,
        'y': df['log_cost'].values,
        'X_original': df[['lot_midpoint', 'lot_quantity']].values,
        'y_original': df['cost'].values,
        'lot_midpoints': df['lot_midpoint'].values,
        'lot_quantities': df['lot_quantity'].values
    }
    
    if 'true_cost' in df.columns:
        data['true_costs'] = df['true_cost'].values
    
    return data


def visualize_dataset(data, title="Learning Curve Data"):
    """Create visualization of learning curve dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cost vs Cumulative Units
    ax = axes[0, 0]
    cumulative = np.cumsum(data['lot_quantities'])
    ax.scatter(cumulative, data['y_original'], alpha=0.7, label='Observed')
    if 'true_costs' in data:
        ax.plot(cumulative, data['true_costs'], 'r--', label='True', linewidth=2)
    ax.set_xlabel('Cumulative Units')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Cumulative Production')
    ax.legend()
    
    # Plot 2: Log-Log Space
    ax = axes[0, 1]
    ax.scatter(data['X'][:, 0], data['y'], alpha=0.7, label='log(midpoint)')
    ax.scatter(data['X'][:, 1], data['y'], alpha=0.7, label='log(quantity)')
    ax.set_xlabel('Log Predictor Value')
    ax.set_ylabel('Log Cost')
    ax.set_title('Log-Log Space')
    ax.legend()
    
    # Plot 3: Correlation between predictors
    ax = axes[1, 0]
    ax.scatter(data['X'][:, 0], data['X'][:, 1], alpha=0.7)
    corr = np.corrcoef(data['X'][:, 0], data['X'][:, 1])[0, 1]
    ax.set_xlabel('log(Lot Midpoint)')
    ax.set_ylabel('log(Lot Quantity)')
    ax.set_title(f'Predictor Correlation: {corr:.3f}')
    
    # Add regression line
    z = np.polyfit(data['X'][:, 0], data['X'][:, 1], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data['X'][:, 0].min(), data['X'][:, 0].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2)
    
    # Plot 4: Lot size progression
    ax = axes[1, 1]
    lots = np.arange(1, len(data['lot_quantities']) + 1)
    ax.bar(lots, data['lot_quantities'], alpha=0.7)
    ax.set_xlabel('Lot Number')
    ax.set_ylabel('Lot Quantity')
    ax.set_title('Production Ramp-Up')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN: Examples
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("DATA GENERATION EXAMPLES")
    print("="*70)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Single dataset
    print("\n1. Generating single dataset...")
    data = generate_single_dataset(
        n_lots=20,
        T1=100,
        learning_rate=0.90,
        rate_effect=0.95,
        target_correlation=0.7,
        cv_error=0.1,
        random_state=42
    )
    
    print(f"   N lots: {len(data['y'])}")
    print(f"   True parameters: b={data['params']['b']:.4f}, c={data['params']['c']:.4f}")
    print(f"   Achieved correlation: {data['actual_correlation']:.3f}")
    
    # Save to CSV
    save_dataset(data, output_dir / "example_dataset.csv")
    
    # Visualize
    fig = visualize_dataset(data, "Example Learning Curve Dataset")
    fig.savefig(output_dir / "example_dataset.png", dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to: {output_dir / 'example_dataset.png'}")
    
    # Example 2: Multiple correlation levels
    print("\n2. Generating datasets with different correlations...")
    for corr in [0.0, 0.5, 0.9]:
        data = generate_single_dataset(
            n_lots=20,
            target_correlation=corr,
            random_state=42
        )
        print(f"   Target: {corr:.1f}, Achieved: {data['actual_correlation']:.3f}")
    
    # Example 3: Dataset grid for simulation
    print("\n3. Generating small grid for testing...")
    datasets = generate_dataset_grid(
        n_lots_list=[10, 20],
        correlation_list=[0.5],
        cv_error_list=[0.1],
        n_replications=5,
        base_seed=42
    )
    
    print(f"   Total datasets: {len(datasets)}")
    
    # Check correlation distribution
    correlations = [d['actual_correlation'] for d in datasets]
    print(f"   Correlation range: [{min(correlations):.3f}, {max(correlations):.3f}]")
    
    print("\n" + "="*70)
    print("Data generation examples completed!")
    print("="*70)
    
    plt.show()
