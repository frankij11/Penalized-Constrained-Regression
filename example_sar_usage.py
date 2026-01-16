"""
Example: Using SAR-based Learning Curve Data Generation

This example shows how to use generate_sar_based_learning_data() to create
learning curve training data from actual procurement quantities.
"""

import penalized_constrained as pcreg
import numpy as np

print("=" * 80)
print("SAR-BASED DATA GENERATION EXAMPLES")
print("=" * 80)

# Example 1: Random program selection
print("\nExample 1: Random Program Selection")
print("-" * 80)

data = pcreg.generate_sar_based_learning_data(
    n_lots=20,
    T1=100,
    cv_error=0.1,
    random_state=42
)

print(f"Program ID: {data['program_id']}")
print(f"Lots generated: {len(data['lot_quantities'])}")
print(f"Actual correlation: {data['actual_correlation']:.3f}")
print(f"Available observations: {data['sar_n_available']}")
print(f"Year range: {min(data['sar_fiscal_years']):.0f} - {max(data['sar_fiscal_years']):.0f}")
print(f"\nQuantity statistics:")
print(f"  Mean: {np.mean(data['lot_quantities']):.1f}")
print(f"  Std: {np.std(data['lot_quantities']):.1f}")
print(f"  Range: {data['lot_quantities'].min()} - {data['lot_quantities'].max()}")

# Example 2: Specific program with recent data only
print("\n\nExample 2: Specific Program with Recent Data Filter")
print("-" * 80)

# First, find a valid program ID
sample_data = pcreg.generate_sar_based_learning_data(n_lots=5, random_state=1)
program_id = sample_data['program_id']
print(f"Using program ID: {program_id}")

data = pcreg.generate_sar_based_learning_data(
    n_lots=15,
    program_name=program_id,
    min_year=2010,
    T1=100,
    random_state=42
)

print(f"Lots generated: {len(data['lot_quantities'])}")
print(f"Year range: {min(data['sar_fiscal_years']):.0f} - {max(data['sar_fiscal_years']):.0f}")
print(f"All years >= 2010: {all(y >= 2010 for y in data['sar_fiscal_years'])}")

# Example 3: Train a model with SAR data
print("\n\nExample 3: Training a Model with SAR Data")
print("-" * 80)

# Generate training data
data = pcreg.generate_sar_based_learning_data(
    n_lots=20,
    T1=100,
    cv_error=0.1,
    random_state=42
)

X_train = data['X_original']  # Unit space: [midpoint, quantity]
y_train = data['y_original']

# Train constrained model
model = pcreg.PenalizedConstrainedRegression(
    bounds=[(-0.5, 0), (-0.5, 0)],  # Constrain both slopes to be negative
    alpha=0.01,
    penalty_exclude=[],  # Penalize both coefficients
    loss='sspe'
)
model.fit(X_train, y_train)

print(f"True parameters:")
print(f"  b = {data['params']['b']:.4f} (learning: {data['params']['learning_rate']:.1%})")
print(f"  c = {data['params']['c']:.4f} (rate effect: {data['params']['rate_effect']:.1%})")
print(f"\nEstimated parameters:")
print(f"  b = {model.coef_[0]:.4f} (learning: {pcreg.slope_to_learning_rate(model.coef_[0]):.1%})")
print(f"  c = {model.coef_[1]:.4f} (rate effect: {pcreg.slope_to_learning_rate(model.coef_[1]):.1%})")
print(f"\nModel fit:")
print(f"  R² = {model.score(X_train, y_train):.3f}")
print(f"  Converged: {model.converged_}")

# Example 4: Compare SAR vs Synthetic data
print("\n\nExample 4: Comparing SAR vs Synthetic Data")
print("-" * 80)

# SAR data (natural correlation)
data_sar = pcreg.generate_sar_based_learning_data(
    n_lots=20,
    random_state=42
)

# Synthetic data (target correlation)
data_synth = pcreg.generate_correlated_learning_data(
    n_lots=20,
    target_correlation=0.7,
    random_state=42
)

print("SAR data:")
print(f"  Correlation: {data_sar['actual_correlation']:.3f} (natural)")
print(f"  Quantity CV: {np.std(data_sar['lot_quantities']) / np.mean(data_sar['lot_quantities']):.3f}")
print(f"  Quantity range: {data_sar['lot_quantities'].min()} - {data_sar['lot_quantities'].max()}")

print("\nSynthetic data:")
print(f"  Correlation: {data_synth['actual_correlation']:.3f} (targeted)")
print(f"  Quantity CV: {np.std(data_synth['lot_quantities']) / np.mean(data_synth['lot_quantities']):.3f}")
print(f"  Quantity range: {data_synth['lot_quantities'].min()} - {data_synth['lot_quantities'].max()}")

print("\nKey difference: SAR data uses real procurement patterns,")
print("while synthetic data is optimized to achieve target correlation.")

# Example 5: Using lot_data DataFrame
print("\n\nExample 5: Exploring the lot_data DataFrame")
print("-" * 80)

data = pcreg.generate_sar_based_learning_data(n_lots=10, random_state=42)
lot_data = data['lot_data']

print("Available columns:")
print(f"  {list(lot_data.columns)}")

print("\nFirst 5 lots:")
cols_to_show = ['lot_number', 'lot_quantity', 'lot_midpoint', 'observed_cost',
                'program_id', 'sar_fiscal_year']
print(lot_data[cols_to_show].head().to_string(index=False))

print("\n\nSummary statistics:")
print(lot_data[['lot_quantity', 'lot_midpoint', 'observed_cost']].describe())

print("\n" + "=" * 80)
print("Examples complete!")
print("=" * 80)
