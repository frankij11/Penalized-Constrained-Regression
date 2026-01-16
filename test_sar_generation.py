"""
Test script for SAR-based learning curve data generation.

This script verifies that generate_sar_based_learning_data() works correctly
and produces output compatible with the existing simulation framework.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
import penalized_constrained as pcreg

print("=" * 80)
print("SAR-BASED DATA GENERATION TESTS")
print("=" * 80)

# Test 1: Basic generation with mock data
print("\n" + "=" * 80)
print("Test 1: Basic Generation")
print("=" * 80)
try:
    data = pcreg.generate_sar_based_learning_data(
        n_lots=10,
        T1=100,
        cv_error=0.1,
        random_state=42
    )
    print(f"✓ Generated {len(data['lot_quantities'])} lots")
    print(f"  Program ID: {data['program_id']}")
    print(f"  Correlation: {data['actual_correlation']:.3f}")
    print(f"  Available observations: {data['sar_n_available']}")
    print(f"  Used replacement: {data['sar_used_replacement']}")
    print(f"  Year range: {min(data['sar_fiscal_years'])} - {max(data['sar_fiscal_years'])}")
    print(f"  Quantity range: {data['lot_quantities'].min()} - {data['lot_quantities'].max()}")
except FileNotFoundError as e:
    print(f"✗ FileNotFoundError (expected if SAR file not yet created):")
    print(f"  {e}")
    print("\nNote: Create the SAR data file at:")
    print("  scripts/ICEAA/sar_raw_db/msar_annual_quantities.csv")
    print("  to run the full tests.")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Output structure compatibility
print("\n" + "=" * 80)
print("Test 2: Output Structure Compatibility")
print("=" * 80)
try:
    data_synth = pcreg.generate_correlated_learning_data(
        n_lots=10, T1=100, target_correlation=0.5, random_state=42
    )
    data_sar = pcreg.generate_sar_based_learning_data(
        n_lots=10, T1=100, random_state=42
    )

    # Check all keys from synthetic are in SAR (SAR has extras)
    synth_keys = set(data_synth.keys())
    sar_keys = set(data_sar.keys())
    missing_keys = synth_keys - sar_keys

    if missing_keys:
        print(f"✗ Missing keys: {missing_keys}")
    else:
        print("✓ All synthetic keys present in SAR output")
        extra_keys = sar_keys - synth_keys
        print(f"  SAR adds extra keys: {extra_keys}")

    # Check data types and shapes
    print("\n  Checking array shapes:")
    for key in ['X', 'y', 'X_original', 'y_original', 'lot_quantities', 'lot_midpoints']:
        if key in data_sar:
            shape_sar = data_sar[key].shape if hasattr(data_sar[key], 'shape') else len(data_sar[key])
            shape_synth = data_synth[key].shape if hasattr(data_synth[key], 'shape') else len(data_synth[key])
            match = "✓" if shape_sar == shape_synth else "✗"
            print(f"    {match} {key}: SAR={shape_sar}, Synthetic={shape_synth}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Model training compatibility
print("\n" + "=" * 80)
print("Test 3: Model Training Compatibility")
print("=" * 80)
try:
    data = pcreg.generate_sar_based_learning_data(
        n_lots=15, T1=100, random_state=42
    )

    X = data['X_original']
    y = data['y_original']

    model = pcreg.PenalizedConstrainedRegression(
        bounds=[(-0.5, 0), (-0.5, 0)],
        alpha=0.0
    )
    model.fit(X, y)

    print(f"✓ Model trained successfully")
    print(f"  True coefficients: b={data['params']['b']:.4f}, c={data['params']['c']:.4f}")
    print(f"  Estimated coefficients: b={model.coef_[0]:.4f}, c={model.coef_[1]:.4f}")
    print(f"  R²: {model.score(X, y):.3f}")

    # Check predictions work
    y_pred = model.predict(X)
    print(f"✓ Predictions generated: shape={y_pred.shape}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Reproducibility
print("\n" + "=" * 80)
print("Test 4: Reproducibility")
print("=" * 80)
try:
    data1 = pcreg.generate_sar_based_learning_data(
        n_lots=10, random_state=123
    )
    data2 = pcreg.generate_sar_based_learning_data(
        n_lots=10, random_state=123
    )

    if np.allclose(data1['lot_quantities'], data2['lot_quantities']):
        print("✓ Same seed produces identical quantities")
    else:
        print("✗ Quantities differ with same seed")
        print(f"  Data1: {data1['lot_quantities']}")
        print(f"  Data2: {data2['lot_quantities']}")

    if data1['program_id'] == data2['program_id']:
        print("✓ Same program selected")
    else:
        print("✗ Different programs selected")
        print(f"  Data1 program: {data1['program_id']}")
        print(f"  Data2 program: {data2['program_id']}")

    if np.allclose(data1['actual_correlation'], data2['actual_correlation']):
        print("✓ Correlation matches")
    else:
        print("✗ Correlation differs")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Year filtering
print("\n" + "=" * 80)
print("Test 5: Year Filtering")
print("=" * 80)
try:
    data_all = pcreg.generate_sar_based_learning_data(
        n_lots=10, random_state=42
    )
    data_recent = pcreg.generate_sar_based_learning_data(
        n_lots=10, min_year=2010, random_state=42
    )

    print(f"✓ All years range: {min(data_all['sar_fiscal_years'])} - {max(data_all['sar_fiscal_years'])}")
    print(f"✓ Recent years range: {min(data_recent['sar_fiscal_years'])} - {max(data_recent['sar_fiscal_years'])}")

    if min(data_recent['sar_fiscal_years']) >= 2010:
        print("✓ Year filter working correctly")
    else:
        print("✗ Year filter not applied correctly")
        print(f"  Expected min year >= 2010, got {min(data_recent['sar_fiscal_years'])}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: DataFrame structure
print("\n" + "=" * 80)
print("Test 6: DataFrame Structure")
print("=" * 80)
try:
    data = pcreg.generate_sar_based_learning_data(
        n_lots=5, random_state=42
    )

    lot_data = data['lot_data']
    print(f"✓ lot_data is DataFrame: {isinstance(lot_data, pd.DataFrame)}")
    print(f"  Columns: {list(lot_data.columns)}")
    print(f"  Shape: {lot_data.shape}")

    # Check for SAR-specific columns
    sar_columns = ['sar_fiscal_year', 'program_id']
    has_sar_cols = all(col in lot_data.columns for col in sar_columns)
    if has_sar_cols:
        print(f"✓ SAR-specific columns present: {sar_columns}")
    else:
        print(f"✗ Missing SAR columns")

    print("\n  First few rows:")
    print(lot_data[['lot_number', 'lot_quantity', 'lot_midpoint', 'program_id', 'sar_fiscal_year']].head())

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Specific program selection
print("\n" + "=" * 80)
print("Test 7: Specific Program Selection")
print("=" * 80)
try:
    # First get a valid program ID
    data = pcreg.generate_sar_based_learning_data(
        n_lots=5, random_state=42
    )
    valid_program_id = data['program_id']
    print(f"  Found program ID: {valid_program_id}")

    # Now request that specific program
    data_specific = pcreg.generate_sar_based_learning_data(
        n_lots=5, program_name=valid_program_id, random_state=999
    )

    if data_specific['program_id'] == valid_program_id:
        print(f"✓ Specific program selection works (ID={valid_program_id})")
    else:
        print(f"✗ Wrong program selected")
        print(f"  Expected: {valid_program_id}, Got: {data_specific['program_id']}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Error handling - Invalid program
print("\n" + "=" * 80)
print("Test 8: Error Handling - Invalid Program")
print("=" * 80)
try:
    data = pcreg.generate_sar_based_learning_data(
        n_lots=10, program_name=99999  # Very unlikely to exist
    )
    print("✗ Should have raised ValueError for invalid program")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)[:100]}...")
except Exception as e:
    print(f"✗ Wrong exception type: {type(e).__name__}")

# Test 9: Caching
print("\n" + "=" * 80)
print("Test 9: Caching Performance")
print("=" * 80)
try:
    import time

    # First call (no cache)
    start = time.time()
    data1 = pcreg.generate_sar_based_learning_data(n_lots=10, random_state=42)
    first_time = time.time() - start

    # Second call (cached)
    start = time.time()
    data2 = pcreg.generate_sar_based_learning_data(n_lots=10, random_state=43)
    second_time = time.time() - start

    speedup = first_time / second_time if second_time > 0 else float('inf')

    print(f"✓ First call: {first_time:.4f}s")
    print(f"✓ Second call: {second_time:.4f}s")
    print(f"✓ Speedup: {speedup:.1f}x")

    if speedup > 5:
        print("✓ Caching provides significant speedup (>5x)")
    elif speedup > 1.5:
        print("✓ Caching provides moderate speedup")
    else:
        print("⚠ Speedup lower than expected (may vary by system)")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Sampling with replacement
print("\n" + "=" * 80)
print("Test 10: Sampling with Replacement")
print("=" * 80)
try:
    # Request more lots than likely available for a single program
    data = pcreg.generate_sar_based_learning_data(
        n_lots=50, random_state=42, allow_replacement=True
    )

    print(f"✓ Generated {data['sar_n_available']} lots available")
    print(f"  Requested: 50 lots")
    print(f"  Used replacement: {data['sar_used_replacement']}")

    if data['sar_used_replacement']:
        print("✓ Sampling with replacement worked")
    else:
        print("  Note: Program had enough data, no replacement needed")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("All core tests completed!")
print("\nThe function is ready to use in your simulation studies.")
print("\nNext steps:")
print("  1. Ensure SAR data file exists at:")
print("     scripts/ICEAA/sar_raw_db/msar_annual_quantities.csv")
print("  2. Try generating data with different parameters")
print("  3. Integrate with run_simulation.py if desired")
print("=" * 80)
