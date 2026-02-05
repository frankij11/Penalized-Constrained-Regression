import pandas as pd
import numpy as np

# Load current motivational example data
df_example = pd.read_csv('output_v2/motivational_example_data.csv')
print("Current motivational example data:")
print(df_example.head(10))
print()

# Check for seed column
if 'seed' in df_example.columns:
    print(f"Current seed: {df_example['seed'].iloc[0]}")
else:
    print("No seed column in example data")

print()
print("Columns:", df_example.columns.tolist())
print()
print("Unique lot_type:", df_example['lot_type'].unique())
print()
print("Train data:")
print(df_example[df_example['lot_type'] == 'train'])
print()
print("Test data (first 10):")
print(df_example[df_example['lot_type'] == 'test'].head(10))

# Check scenario params
if 'T1_true' in df_example.columns:
    print(f"\nTrue T1: {df_example['T1_true'].iloc[0]}")
if 'b_true' in df_example.columns:
    print(f"True b: {df_example['b_true'].iloc[0]}")
    print(f"True LR: {2**df_example['b_true'].iloc[0]*100:.1f}%")
if 'c_true' in df_example.columns:
    print(f"True c: {df_example['c_true'].iloc[0]}")
    print(f"True RE: {2**df_example['c_true'].iloc[0]*100:.1f}%")
