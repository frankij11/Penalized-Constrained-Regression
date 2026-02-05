import pandas as pd
import numpy as np

# Load simulation results
results = pd.read_parquet('output_v2/simulation_results.parquet')

# Current example seed
current_seed = 1987917204

print("Results for current seed (1987917204):")
current_results = results[results['seed'] == current_seed]
print(current_results[['model_name', 'test_mape', 'b', 'c', 'T1_est', 'r2']].to_string())

# Calculate LC/RC
current_results = current_results.copy()
current_results['LC'] = 2 ** current_results['b']
current_results['RC'] = 2 ** current_results['c']
print("\nWith LC/RC:")
print(current_results[['model_name', 'test_mape', 'LC', 'RC', 'T1_est', 'r2']].to_string())

# Also check the best seed
best_seed = 814700337
print(f"\n\nResults for best seed ({best_seed}):")
best_results = results[results['seed'] == best_seed]
best_results = best_results.copy()
best_results['LC'] = 2 ** best_results['b']
best_results['RC'] = 2 ** best_results['c']
print(best_results[['model_name', 'test_mape', 'LC', 'RC', 'T1_est', 'r2']].to_string())

# Check another good candidate - 1425625790
alt_seed = 1425625790
print(f"\n\nResults for alt seed ({alt_seed}):")
alt_results = results[results['seed'] == alt_seed]
alt_results = alt_results.copy()
alt_results['LC'] = 2 ** alt_results['b']
alt_results['RC'] = 2 ** alt_results['c']
print(alt_results[['model_name', 'test_mape', 'LC', 'RC', 'T1_est', 'r2']].to_string())

# Check seed 567400748 - lower cv_error
alt_seed2 = 567400748
print(f"\n\nResults for seed with cv_error=0.1 ({alt_seed2}):")
alt_results2 = results[results['seed'] == alt_seed2]
alt_results2 = alt_results2.copy()
alt_results2['LC'] = 2 ** alt_results2['b']
alt_results2['RC'] = 2 ** alt_results2['c']
print(alt_results2[['model_name', 'test_mape', 'LC', 'RC', 'T1_est', 'r2']].to_string())

# Get scenario info for this seed
scenario_info = results[results['seed'] == alt_seed2][['learning_rate', 'rate_effect', 'cv_error', 'n_lots', 'T1_true', 'b_true', 'c_true']].iloc[0]
print("\nScenario info:")
print(scenario_info)
