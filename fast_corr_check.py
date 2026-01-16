import sys
from pathlib import Path
import numpy as np
import pandas as pd
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))
import penalized_constrained as pcreg

print("Quick Correlation Analysis for n = [5, 10, 15, 20]")
print("=" * 60)

sample_sizes = [5, 10, 15, 20]
results = []

for seed in range(1, 11):  # Just 10 programs
    data = pcreg.generate_sar_based_learning_data(n_lots=20, cv_error=0.0, random_state=seed)
    for n_lots in sample_sizes:
        subset = data['lot_data'].iloc[:n_lots]
        corr = np.corrcoef(subset['log_midpoint'], subset['log_quantity'])[0, 1]
        results.append({'n_lots': n_lots, 'correlation': corr, 'program': data['program_id']})
    print(f"Program {data['program_id']:2d}: n=5:{results[-4]['correlation']:6.3f}, n=10:{results[-3]['correlation']:6.3f}, n=15:{results[-2]['correlation']:6.3f}, n=20:{results[-1]['correlation']:6.3f}")

df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Summary by Sample Size:")
print(df.groupby('n_lots')['correlation'].agg(['mean', 'std', 'min', 'max']).round(3))
