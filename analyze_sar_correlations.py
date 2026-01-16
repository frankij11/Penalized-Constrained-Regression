"""
Analyze SAR Correlation Patterns by Sample Size

This script explores how correlation between log(midpoint) and log(quantity)
varies with the number of lots sampled from SAR data. Typically, early lots
show ramp-up patterns (increasing quantities) leading to different correlations
than steady-state production.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
import penalized_constrained as pcreg

print("=" * 80)
print("SAR CORRELATION ANALYSIS BY SAMPLE SIZE")
print("=" * 80)

# Sample sizes to analyze
sample_sizes = [5, 10, 15, 20]
n_programs = 30  # Number of different programs to sample
random_seeds = range(1, n_programs + 1)

# Storage for results
results = []

print(f"\nAnalyzing {n_programs} different SAR programs...")
print(f"Sample sizes: {sample_sizes}")
print(f"\nThis may take a moment...\n")

for seed in random_seeds:
    try:
        # Generate data for maximum sample size
        max_n_lots = max(sample_sizes)
        data = pcreg.generate_sar_based_learning_data(
            n_lots=max_n_lots,
            T1=100,
            cv_error=0.0,  # No error for pure correlation analysis
            random_state=seed,
            allow_replacement=True
        )

        program_id = data['program_id']
        n_available = data['sar_n_available']

        # Calculate correlation for different sample sizes
        for n_lots in sample_sizes:
            # Use first n lots
            log_midpoints = data['lot_data']['log_midpoint'].iloc[:n_lots].values
            log_quantities = data['lot_data']['log_quantity'].iloc[:n_lots].values

            # Calculate correlation
            correlation = np.corrcoef(log_midpoints, log_quantities)[0, 1]

            # Store result
            results.append({
                'program_id': program_id,
                'n_available': n_available,
                'n_lots': n_lots,
                'correlation': correlation,
                'seed': seed
            })

        if seed % 10 == 0:
            print(f"  Processed {seed}/{n_programs} programs...")

    except Exception as e:
        print(f"  Warning: Failed for seed {seed}: {e}")
        continue

# Convert to DataFrame
df = pd.DataFrame(results)

print(f"\nSuccessfully analyzed {df['program_id'].nunique()} unique programs")
print("=" * 80)

# Summary statistics by sample size
print("\n" + "=" * 80)
print("CORRELATION STATISTICS BY SAMPLE SIZE")
print("=" * 80)

summary = df.groupby('n_lots')['correlation'].agg([
    'count', 'mean', 'std', 'min',
    ('25%', lambda x: x.quantile(0.25)),
    ('50%', lambda x: x.quantile(0.50)),
    ('75%', lambda x: x.quantile(0.75)),
    'max'
]).round(3)

print("\n", summary)

# Distribution by sample size
print("\n" + "=" * 80)
print("CORRELATION DISTRIBUTION BY SAMPLE SIZE")
print("=" * 80)

for n_lots in sample_sizes:
    subset = df[df['n_lots'] == n_lots]['correlation']
    print(f"\nn_lots = {n_lots}:")
    print(f"  Mean: {subset.mean():.3f}")
    print(f"  Std:  {subset.std():.3f}")
    print(f"  Range: [{subset.min():.3f}, {subset.max():.3f}]")

    # Correlation bins
    bins = [(-1, -0.5, 'Strong Negative'),
            (-0.5, -0.1, 'Moderate Negative'),
            (-0.1, 0.1, 'Weak/None'),
            (0.1, 0.5, 'Moderate Positive'),
            (0.5, 1, 'Strong Positive')]

    print(f"  Distribution:")
    for low, high, label in bins:
        count = ((subset >= low) & (subset < high)).sum()
        pct = 100 * count / len(subset)
        if count > 0:
            print(f"    {label:20s}: {count:2d} ({pct:5.1f}%)")

# Program-level analysis
print("\n" + "=" * 80)
print("CORRELATION CHANGE WITH SAMPLE SIZE (By Program)")
print("=" * 80)

# Calculate change in correlation from smallest to largest sample
# Group by both seed and program_id to handle duplicates
df['seed_prog'] = df['seed'].astype(str) + '_' + df['program_id'].astype(str)
pivot = df.pivot(index='seed_prog', columns='n_lots', values='correlation')
pivot['change_5_to_20'] = pivot[20] - pivot[5]
pivot['abs_change'] = pivot['change_5_to_20'].abs()

print(f"\nProgram instances with largest correlation changes (n=5 to n=20):")
top_changes = pivot.nlargest(10, 'abs_change')[['change_5_to_20', 5, 20]].round(3)
print(top_changes)

print(f"\nAverage absolute change: {pivot['abs_change'].mean():.3f}")
print(f"Max absolute change: {pivot['abs_change'].max():.3f}")

# Identify programs with stable vs. changing correlations
stable_instances = pivot[pivot['abs_change'] < 0.1]
changing_instances = pivot[pivot['abs_change'] > 0.3]

print(f"\nProgram instances with stable correlation (|Δ| < 0.1): {len(stable_instances)}")
print(f"Program instances with changing correlation (|Δ| > 0.3): {len(changing_instances)}")

# Typical correlation ranges for simulation scenarios
print("\n" + "=" * 80)
print("RECOMMENDED CORRELATION BINS FOR SIMULATION ANALYSIS")
print("=" * 80)

all_corr = df['correlation']
low_threshold = all_corr.quantile(0.33)
high_threshold = all_corr.quantile(0.67)

print(f"\nBased on {len(all_corr)} observations across all sample sizes:")
print(f"  Low correlation:    r < {low_threshold:.2f}")
print(f"  Medium correlation: {low_threshold:.2f} ≤ r < {high_threshold:.2f}")
print(f"  High correlation:   r ≥ {high_threshold:.2f}")

for n_lots in sample_sizes:
    subset = df[df['n_lots'] == n_lots]['correlation']
    low = (subset < low_threshold).sum()
    med = ((subset >= low_threshold) & (subset < high_threshold)).sum()
    high = (subset >= high_threshold).sum()
    print(f"\nn_lots={n_lots}: Low={low}, Medium={med}, High={high}")

# Create visualization if matplotlib available
print("\n" + "=" * 80)
print("GENERATING VISUALIZATION")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAR Correlation Analysis by Sample Size', fontsize=16, fontweight='bold')

    # Plot 1: Box plot by sample size
    ax = axes[0, 0]
    df.boxplot(column='correlation', by='n_lots', ax=ax)
    ax.set_xlabel('Number of Lots')
    ax.set_ylabel('Correlation (log-midpoint vs log-quantity)')
    ax.set_title('Correlation Distribution by Sample Size')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero correlation')
    ax.get_figure().suptitle('')  # Remove automatic title
    ax.legend()

    # Plot 2: Histogram for each sample size
    ax = axes[0, 1]
    for n_lots in sample_sizes:
        subset = df[df['n_lots'] == n_lots]['correlation']
        ax.hist(subset, bins=20, alpha=0.5, label=f'n={n_lots}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Correlation Histograms')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    ax.legend()

    # Plot 3: Correlation change by program
    ax = axes[1, 0]
    for idx, (prog_id, row) in enumerate(pivot.iterrows()):
        if idx < 20:  # Plot first 20 programs for clarity
            ax.plot(sample_sizes, [row[n] for n in sample_sizes],
                   marker='o', alpha=0.5, linewidth=1)
    ax.set_xlabel('Number of Lots')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Evolution by Program (first 20 programs)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Plot 4: Mean correlation with confidence bands
    ax = axes[1, 1]
    mean_corr = df.groupby('n_lots')['correlation'].mean()
    std_corr = df.groupby('n_lots')['correlation'].std()
    ax.plot(sample_sizes, mean_corr, marker='o', linewidth=2, color='blue', label='Mean')
    ax.fill_between(sample_sizes,
                     mean_corr - std_corr,
                     mean_corr + std_corr,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax.set_xlabel('Number of Lots')
    ax.set_ylabel('Correlation')
    ax.set_title('Mean Correlation with Variability')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'sar_correlation_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    # Also save CSV
    csv_path = Path(__file__).parent / 'sar_correlation_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Raw data saved to: {csv_path}")

    plt.show()

except Exception as e:
    print(f"Could not create visualization: {e}")
    print("(matplotlib may not be available)")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. CORRELATION VARIES WITH SAMPLE SIZE:")
mean_by_size = df.groupby('n_lots')['correlation'].mean()
print(f"   n=5:  Mean correlation = {mean_by_size[5]:.3f}")
print(f"   n=10: Mean correlation = {mean_by_size[10]:.3f}")
print(f"   n=15: Mean correlation = {mean_by_size[15]:.3f}")
print(f"   n=20: Mean correlation = {mean_by_size[20]:.3f}")

trend = "increases" if mean_by_size[20] > mean_by_size[5] else "decreases"
print(f"   → Correlation generally {trend} with more lots")

print("\n2. VARIABILITY:")
std_by_size = df.groupby('n_lots')['correlation'].std()
print(f"   Highest variability: n={std_by_size.idxmax()} (std={std_by_size.max():.3f})")
print(f"   Lowest variability:  n={std_by_size.idxmin()} (std={std_by_size.min():.3f})")

print("\n3. TYPICAL SAR PATTERNS:")
if all_corr.mean() < -0.1:
    print("   → Most SAR programs show NEGATIVE correlation")
    print("   → This suggests quantity decreases as production progresses")
elif all_corr.mean() > 0.1:
    print("   → Most SAR programs show POSITIVE correlation")
    print("   → This suggests ramp-up pattern (increasing quantities)")
else:
    print("   → SAR programs show WEAK correlation on average")
    print("   → Mix of ramp-up and steady-state patterns")

print("\n4. IMPLICATIONS FOR SIMULATION:")
print("   → When using SAR data, correlation is NOT controlled")
print("   → Should analyze results by actual correlation bins, not target")
print("   → Consider stratifying by n_lots in analysis")
print(f"   → Sample size n={sample_sizes[-1]} provides most stable estimates")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
