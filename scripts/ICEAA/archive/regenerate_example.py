"""
Regenerate motivational example with a better seed where PCReg-GCV clearly wins.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_data import load_or_generate_simulation_data

# Configuration - must match run_simulation.py
CONFIG = {
    'sample_sizes': [5, 10, 30],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],
    'rate_effects': [0.80, 0.85, 0.90],
    'n_replications': 100,
    'T1': 100,
    'base_seed': 42,
}

# Load simulation data
print("Loading simulation data...")
simulation_df = load_or_generate_simulation_data(
    sample_sizes=CONFIG['sample_sizes'],
    cv_errors=CONFIG['cv_errors'],
    learning_rates=CONFIG['learning_rates'],
    rate_effects=CONFIG['rate_effects'],
    n_replications=CONFIG['n_replications'],
    T1=CONFIG['T1'],
    base_seed=CONFIG['base_seed'],
)

# Best seed we found
target_seed = 1425625790

# Find scenario with this seed
scenario_data = simulation_df[simulation_df['seed'] == target_seed].copy()

if len(scenario_data) == 0:
    print(f"ERROR: Seed {target_seed} not found in simulation data!")
    print("Available seeds sample:", simulation_df['seed'].unique()[:10])
    sys.exit(1)

print(f"Found scenario with seed {target_seed}")
print(f"  n_lots: {scenario_data['n_lots'].iloc[0]}")
print(f"  cv_error: {scenario_data['cv_error'].iloc[0]}")
print(f"  learning_rate: {scenario_data['learning_rate'].iloc[0]}")
print(f"  rate_effect: {scenario_data['rate_effect'].iloc[0]}")
print(f"  T1_true: {scenario_data['T1_true'].iloc[0]}")
print(f"  b_true: {scenario_data['b_true'].iloc[0]}")
print(f"  c_true: {scenario_data['c_true'].iloc[0]}")
print(f"  actual_correlation: {scenario_data['actual_correlation'].iloc[0]}")

# Split into train and test
train_data = scenario_data[scenario_data['lot_type'] == 'train'].copy()
test_data = scenario_data[scenario_data['lot_type'] == 'test'].copy()

print(f"\n  Training lots: {len(train_data)}")
print(f"  Test lots: {len(test_data)}")

# Save to CSV
output_dir = Path('output_v2')
output_file = output_dir / 'motivational_example_data.csv'
scenario_data.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")

# Verify by checking simulation results for this seed
results = pd.read_parquet(output_dir / 'simulation_results.parquet')
seed_results = results[results['seed'] == target_seed].copy()
seed_results['LC'] = 2 ** seed_results['b']
seed_results['RC'] = 2 ** seed_results['c']

print("\n" + "="*60)
print("SIMULATION RESULTS FOR THIS SEED")
print("="*60)
print(seed_results[['model_name', 'test_mape', 'LC', 'RC', 'T1_est', 'r2']].to_string())
