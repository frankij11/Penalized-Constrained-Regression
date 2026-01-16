"""
Quick SAR Correlation Analysis by Sample Size

Analyzes how correlation varies with number of lots sampled.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
import penalized_constrained as pcreg

print("=" * 80)
print("QUICK SAR CORRELATION ANALYSIS")
print("=" * 80)

# Sample sizes to analyze
sample_sizes = [5, 10, 15, 20]
n_programs = 20  # Reduced for speed
results = []

print(f"\nAnalyzing {n_programs} programs for sample sizes: {sample_sizes}\n")

for seed in range(1, n_programs + 1):
    try:
        # Generate data for max sample size
        data = pcreg.generate_sar_based_learning_data(
            n_lots=20,
            T1=100,
            cv_error=0.0,
            random_state=seed,
            allow_replacement=True
        )

        program_id = data['program_id']

        # Calculate correlation for each sample size
        for n_lots in sample_sizes:
            # Use first n lots
            subset = data['lot_data'].iloc[:n_lots]
            log_midpoints = subset['log_midpoint'].values
            log_quantities = subset['log_quantity'].values

            # Calculate correlation
            if len(log_midpoints) >= 2:
                correlation = np.corrcoef(log_midpoints, log_quantities)[0, 1]

                results.append({
                    'seed': seed,
                    'program_id': program_id,
                    'n_lots': n_lots,
                    'correlation': correlation
                })

        if seed % 5 == 0:
            print(f"  Processed {seed}/{n_programs} programs...")

    except Exception as e:
        print(f"  Warning: Failed for seed {seed}: {e}")

df = pd.DataFrame(results)
print(f"\n✓ Analyzed {df['program_id'].nunique()} unique programs")
print("=" * 80)

# Summary statistics
print("\n" + "=" * 80)
print("CORRELATION STATISTICS BY SAMPLE SIZE")
print("=" * 80)

summary = df.groupby('n_lots')['correlation'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(3)

print("\n", summary)

# Detailed breakdown
print("\n" + "=" * 80)
print("DISTRIBUTION BY SAMPLE SIZE")
print("=" * 80)

for n_lots in sample_sizes:
    subset = df[df['n_lots'] == n_lots]['correlation']
    print(f"\nn_lots = {n_lots}:")
    print(f"  Mean:  {subset.mean():6.3f}")
    print(f"  Std:   {subset.std():6.3f}")
    print(f"  Range: [{subset.min():6.3f}, {subset.max():6.3f}]")

    # Count by bins
    neg_strong = (subset < -0.5).sum()
    neg_mod = ((subset >= -0.5) & (subset < -0.1)).sum()
    weak = ((subset >= -0.1) & (subset < 0.1)).sum()
    pos_mod = ((subset >= 0.1) & (subset < 0.5)).sum()
    pos_strong = (subset >= 0.5).sum()

    print(f"  Distribution:")
    if neg_strong > 0:
        print(f"    Strong negative (r<-0.5):  {neg_strong:2d} ({100*neg_strong/len(subset):5.1f}%)")
    if neg_mod > 0:
        print(f"    Moderate negative:         {neg_mod:2d} ({100*neg_mod/len(subset):5.1f}%)")
    if weak > 0:
        print(f"    Weak (-0.1 to 0.1):        {weak:2d} ({100*weak/len(subset):5.1f}%)")
    if pos_mod > 0:
        print(f"    Moderate positive:         {pos_mod:2d} ({100*pos_mod/len(subset):5.1f}%)")
    if pos_strong > 0:
        print(f"    Strong positive (r>0.5):   {pos_strong:2d} ({100*pos_strong/len(subset):5.1f}%)")

# Correlation change analysis
print("\n" + "=" * 80)
print("CORRELATION STABILITY ACROSS SAMPLE SIZES")
print("=" * 80)

# For each seed, calculate change from n=5 to n=20
changes = []
for seed in df['seed'].unique():
    seed_data = df[df['seed'] == seed]
    corr_5 = seed_data[seed_data['n_lots'] == 5]['correlation'].values
    corr_20 = seed_data[seed_data['n_lots'] == 20]['correlation'].values

    if len(corr_5) > 0 and len(corr_20) > 0:
        change = corr_20[0] - corr_5[0]
        changes.append({
            'seed': seed,
            'corr_5': corr_5[0],
            'corr_20': corr_20[0],
            'change': change,
            'abs_change': abs(change)
        })

changes_df = pd.DataFrame(changes)

print(f"\nAverage absolute change (n=5 to n=20): {changes_df['abs_change'].mean():.3f}")
print(f"Max absolute change: {changes_df['abs_change'].max():.3f}")
print(f"Min absolute change: {changes_df['abs_change'].min():.3f}")

print(f"\nPrograms with stable correlation (|Δ| < 0.1): {(changes_df['abs_change'] < 0.1).sum()}")
print(f"Programs with changing correlation (|Δ| > 0.3): {(changes_df['abs_change'] > 0.3).sum()}")

print("\nTop 5 most stable programs:")
print(changes_df.nsmallest(5, 'abs_change')[['seed', 'corr_5', 'corr_20', 'change']].to_string(index=False))

print("\nTop 5 most changing programs:")
print(changes_df.nlargest(5, 'abs_change')[['seed', 'corr_5', 'corr_20', 'change']].to_string(index=False))

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR SIMULATION ANALYSIS")
print("=" * 80)

all_corr = df['correlation']
p33 = all_corr.quantile(0.33)
p67 = all_corr.quantile(0.67)

print(f"\nSuggested correlation bins (based on all {len(all_corr)} observations):")
print(f"  Low:    r < {p33:.2f}")
print(f"  Medium: {p33:.2f} ≤ r < {p67:.2f}")
print(f"  High:   r ≥ {p67:.2f}")

print("\nDistribution by sample size:")
for n_lots in sample_sizes:
    subset = df[df['n_lots'] == n_lots]['correlation']
    low = (subset < p33).sum()
    med = ((subset >= p33) & (subset < p67)).sum()
    high = (subset >= p67).sum()
    print(f"  n={n_lots:2d}: Low={low:2d}, Medium={med:2d}, High={high:2d}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

mean_5 = df[df['n_lots'] == 5]['correlation'].mean()
mean_20 = df[df['n_lots'] == 20]['correlation'].mean()
std_5 = df[df['n_lots'] == 5]['correlation'].std()
std_20 = df[df['n_lots'] == 20]['correlation'].std()

print(f"\n1. Mean correlation:")
print(f"   n=5:  {mean_5:6.3f} (std={std_5:.3f})")
print(f"   n=20: {mean_20:6.3f} (std={std_20:.3f})")

if std_20 < std_5:
    print(f"   → Correlation estimates are MORE STABLE with larger samples")
    print(f"   → Variability decreases by {100*(1 - std_20/std_5):.1f}%")
else:
    print(f"   → Variability similar across sample sizes")

if abs(mean_20 - mean_5) < 0.05:
    print(f"\n2. Mean correlation is STABLE across sample sizes")
else:
    trend = "increases" if mean_20 > mean_5 else "decreases"
    print(f"\n2. Mean correlation {trend} with sample size")
    print(f"   → Change of {abs(mean_20 - mean_5):.3f}")

if all_corr.mean() < 0:
    print(f"\n3. SAR programs show predominantly NEGATIVE correlation")
    print(f"   → Suggests quantities tend to decrease over time")
    print(f"   → Reflects end-of-program tapering or steady-state production")
elif all_corr.mean() > 0:
    print(f"\n3. SAR programs show predominantly POSITIVE correlation")
    print(f"   → Suggests ramp-up pattern (increasing quantities)")
else:
    print(f"\n3. SAR programs show MIXED correlation patterns")

print(f"\n4. For simulation studies:")
print(f"   → SAR data provides natural, realistic correlations")
print(f"   → Range observed: [{all_corr.min():.2f}, {all_corr.max():.2f}]")
print(f"   → Recommend binning by actual correlation for analysis")
print(f"   → Larger samples (n=20) give more stable estimates")

# Save results
csv_path = Path(__file__).parent / 'sar_correlation_quick_analysis.csv'
df.to_csv(csv_path, index=False)
print(f"\n✓ Data saved to: {csv_path}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
