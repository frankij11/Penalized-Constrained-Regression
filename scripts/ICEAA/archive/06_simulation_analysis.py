'''
Generate comprehensive analysis reports from simulation study results. Given the configurations
used in `05_simulation_study_v2.py`, this script reads the saved results and predictions, and
produces summary statistics and visualizations to evaluate model performance across different scenarios.

For each category (cv_error, correlation, learning_rate, rate_effect):
- Generate descriptive statistics showing which model wins
- Create boxplots comparing model performance

DOE Analysis Features:
- Repeated Measures ANOVA with Greenhouse-Geisser correction
- Pairwise comparisons with Holm-Bonferroni correction
- Effect size analysis (Hedges' g)
- Conditional win rate analysis
'''

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Try importing optional packages for DOE analysis
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Note: pingouin not installed. DOE statistical tests will be limited.")
    print("Install with: pip install pingouin")

pd.options.display.float_format = '{:.4f}'.format

# Categories to analyze (design parameters from simulation study)
CATEGORIES = ['cv_error', 'target_correlation', 'learning_rate', 'rate_effect']
CATEGORY_LABELS = {
    'cv_error': 'CV Error',
    'target_correlation': 'Correlation',
    'learning_rate': 'Learning Rate',
    'rate_effect': 'Rate Effect'
}

# Primary metric for comparison
METRIC = 'test_sspe'
METRIC_LABEL = 'Test SSPE (Sum of Squared Percentage Errors)'

# Results directory
results_dir = pathlib.Path(__file__).parent / "output_v2"
output_dir = results_dir / "analysis_plots"
output_dir.mkdir(exist_ok=True)

# Load results
print("Loading simulation results...")
df_results = pd.read_parquet(results_dir / "simulation_results.parquet")
print(f"Loaded {len(df_results):,} rows")
print(f"Models: {df_results['model_name'].unique().tolist()}")
print(f"Categories: {CATEGORIES}")
print()

# Filter to converged models only
df = df_results[df_results['converged'] == True].copy()
print(f"After filtering converged: {len(df):,} rows")
print()

# ==============================================================================
# OVERALL WINNER ANALYSIS
# ==============================================================================
print("=" * 80)
print("OVERALL WINNER ANALYSIS")
print("=" * 80)
print()

# Overall mean test_sspe by model
overall_stats = df.groupby('model_name')[METRIC].describe().reset_index()
overall_stats = overall_stats.sort_values((METRIC, 'mean'))
print("Overall Performance (sorted by mean test_sspe, lower is better):")
print(overall_stats.to_string())
print()

overall_winner = overall_stats.iloc[0]['model_name']
print(f"*** OVERALL WINNER: {overall_winner} (mean {METRIC} = {overall_stats.iloc[0][('mean', METRIC)]:.4f}) ***")
print()

# ==============================================================================
# ANALYSIS BY CATEGORY
# ==============================================================================
for category in CATEGORIES:
    print("=" * 80)
    print(f"ANALYSIS BY {CATEGORY_LABELS[category].upper()}")
    print("=" * 80)
    print()

    # Get unique values for this category
    category_values = sorted(df[category].unique())
    print(f"Values: {category_values}")
    print()

    # Descriptive statistics by category and model
    stats_by_cat = df.groupby([category, 'model_name'])[METRIC].describe()
    print(f"Descriptive Statistics ({METRIC}):")
    print(stats_by_cat.to_string())
    print()

    # Find winner for each category value
    print(f"Winner by {CATEGORY_LABELS[category]}:")
    print("-" * 40)
    winners = []
    for val in category_values:
        subset = df[df[category] == val]
        means = subset.groupby('model_name')[METRIC].mean().sort_values()
        winner = means.index[0]
        winner_score = means.iloc[0]
        winners.append({
            category: val,
            'winner': winner,
            f'mean_{METRIC}': winner_score
        })
        print(f"  {CATEGORY_LABELS[category]} = {val}: {winner} ({METRIC} = {winner_score:.4f})")
    print()

    # Create summary table for this category
    pivot_table = df.groupby([category, 'model_name'])[METRIC].mean().unstack()
    print(f"Mean {METRIC} by {CATEGORY_LABELS[category]} and Model:")
    print(pivot_table.to_string())
    print()

    # -------------------------------------------------------------------------
    # CREATE BOXPLOT FOR THIS CATEGORY
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(category_values), figsize=(5 * len(category_values), 6),
                              sharey=True)
    if len(category_values) == 1:
        axes = [axes]

    # Get model order (sorted by overall mean performance)
    model_order = overall_stats.index.tolist()

    for ax, val in zip(axes, category_values):
        subset = df[df[category] == val]

        # Prepare data for boxplot
        data_to_plot = [subset[subset['model_name'] == m][METRIC].dropna().values
                        for m in model_order]

        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

        # Color boxes - highlight PCReg variants
        colors = []
        for m in model_order:
            if 'PCReg' in m:
                colors.append('lightgreen')
            else:
                colors.append('lightblue')

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'{CATEGORY_LABELS[category]} = {val}')
        ax.set_ylabel(METRIC_LABEL if ax == axes[0] else '')
        ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        # Mark winner
        means = [subset[subset['model_name'] == m][METRIC].mean() for m in model_order]
        winner_idx = np.argmin(means)
        bp['boxes'][winner_idx].set_edgecolor('red')
        bp['boxes'][winner_idx].set_linewidth(2)

    plt.suptitle(f'Model Performance by {CATEGORY_LABELS[category]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'boxplot_by_{category}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: boxplot_by_{category}.png")
    print()

# ==============================================================================
# COMBINED BOXPLOT (ALL MODELS, OVERALL)
# ==============================================================================
print("=" * 80)
print("CREATING OVERALL BOXPLOT")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

model_order = overall_stats.index.tolist()
data_to_plot = [df[df['model_name'] == m][METRIC].dropna().values for m in model_order]

bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

# Color boxes
colors = ['lightgreen' if 'PCReg' in m else 'lightblue' for m in model_order]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Mark winner
bp['boxes'][0].set_edgecolor('red')
bp['boxes'][0].set_linewidth(2)

ax.set_ylabel(METRIC_LABEL)
ax.set_xlabel('Model')
ax.set_xticklabels(model_order, rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Overall Model Performance (Test SSPE)', fontsize=14, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', label='Penalized-Constrained'),
    Patch(facecolor='lightblue', edgecolor='black', label='Standard Methods'),
    Patch(facecolor='white', edgecolor='red', linewidth=2, label='Winner')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
fig.savefig(output_dir / 'boxplot_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: boxplot_overall.png")
print()

# ==============================================================================
# WINNER HEATMAP BY CATEGORY COMBINATIONS
# ==============================================================================
print("=" * 80)
print("WINNER FREQUENCY ANALYSIS")
print("=" * 80)
print()

# Count how often each model wins across all scenarios
scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
win_counts = {}

for name, group in df.groupby(scenario_cols):
    if len(group) > 0:
        group_valid = group.dropna(subset=[METRIC])
        if len(group_valid) > 0:
            winner = group_valid.loc[group_valid[METRIC].idxmin(), 'model_name']
            win_counts[winner] = win_counts.get(winner, 0) + 1

total_scenarios = sum(win_counts.values())
print(f"Total scenarios: {total_scenarios}")
print()
print("Win counts (how often each model has lowest test_sspe):")
print("-" * 40)
for model, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / total_scenarios
    print(f"  {model}: {count} ({pct:.1f}%)")
print()

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("=" * 80)
print("SUMMARY TABLE FOR PAPER")
print("=" * 80)
print()

# Create a nice summary table
summary_data = []
for model in model_order:
    model_data = df[df['model_name'] == model]
    summary_data.append({
        'Model': model,
        'Mean SSPE': model_data[METRIC].mean(),
        'Std SSPE': model_data[METRIC].std(),
        'Median SSPE': model_data[METRIC].median(),
        'Win Count': win_counts.get(model, 0),
        'Win %': 100 * win_counts.get(model, 0) / total_scenarios if total_scenarios > 0 else 0
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Mean SSPE')
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv(output_dir / 'model_summary.csv', index=False)
print(f"\nSaved: model_summary.csv")

# ==============================================================================
# DOE STATISTICAL ANALYSIS
# ==============================================================================
print()
print("=" * 80)
print("DOE STATISTICAL ANALYSIS")
print("=" * 80)
print()

# Create scenario_id for repeated measures analysis
scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
df['scenario_id'] = df.groupby(scenario_cols).ngroup()

# Models for focused comparison
DOE_MODELS = ['OLS', 'OLS_LearnOnly', 'PCReg_ConstrainOnly', 'PCReg_CV', 'PCReg_CV_Tight', 'PCReg_CV_Wrong']
df_doe = df[df['model_name'].isin(DOE_MODELS)].copy()

print(f"DOE Analysis on {len(df_doe):,} observations")
print(f"Models: {DOE_MODELS}")
print(f"Unique scenarios: {df_doe['scenario_id'].nunique():,}")
print()

# -----------------------------------------------------------------------------
# REPEATED MEASURES ANOVA
# -----------------------------------------------------------------------------
if HAS_PINGOUIN:
    print("-" * 80)
    print("REPEATED MEASURES ANOVA (Model as within-subject factor)")
    print("-" * 80)

    rm_aov = pg.rm_anova(
        data=df_doe,
        dv=METRIC,
        within='model_name',
        subject='scenario_id',
        correction=True  # Greenhouse-Geisser
    )

    print(rm_aov.to_string())
    print()

    # Handle different pingouin versions - column may be 'np2' or 'ng2'
    eta_sq_col = 'np2' if 'np2' in rm_aov.columns else 'ng2'
    eta_sq = rm_aov[eta_sq_col].values[0]
    eta_type = 'partial' if eta_sq_col == 'np2' else 'generalized'

    if eta_sq < 0.01:
        effect_interp = 'negligible'
    elif eta_sq < 0.06:
        effect_interp = 'small'
    elif eta_sq < 0.14:
        effect_interp = 'medium'
    else:
        effect_interp = 'large'

    print(f"Effect Size: {eta_type} eta-squared = {eta_sq:.4f} ({effect_interp})")
    print(f"Interpretation: Model choice explains {eta_sq*100:.1f}% of variance in {METRIC}")
    print()

    # Sphericity test
    sphericity = pg.sphericity(
        data=df_doe,
        dv=METRIC,
        within='model_name',
        subject='scenario_id'
    )
    print(f"Sphericity Test (Mauchly): W = {sphericity.W:.4f}, p = {sphericity.pval:.4f}")
    if sphericity.pval < 0.05:
        print("  -> Sphericity violated. Greenhouse-Geisser correction applied.")
    print()

    # Save ANOVA results
    rm_aov.to_csv(output_dir / 'doe_rm_anova.csv', index=False)

    # -----------------------------------------------------------------------------
    # PAIRWISE COMPARISONS
    # -----------------------------------------------------------------------------
    print("-" * 80)
    print("PAIRWISE COMPARISONS (Holm-Bonferroni Corrected)")
    print("-" * 80)

    pairwise = pg.pairwise_tests(
        data=df_doe,
        dv=METRIC,
        within='model_name',
        subject='scenario_id',
        padjust='holm',
        effsize='hedges'
    )

    # Add significance stars
    def sig_stars(p):
        if pd.isna(p):
            return ''
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return 'ns'

    pairwise['sig'] = pairwise['p-corr'].apply(sig_stars)

    # Key comparisons to highlight
    key_pairs = [
        ('PCReg_CV', 'OLS'),
        ('PCReg_CV', 'OLS_LearnOnly'),
        ('PCReg_CV_Tight', 'OLS'),
        ('PCReg_ConstrainOnly', 'OLS'),
    ]

    print("\nKey Comparisons:")
    print(f"{'Comparison':<35} {'Hedges g':>10} {'p-value':>12} {'Sig':>6}")
    print("-" * 70)

    for a, b in key_pairs:
        row = pairwise[(pairwise['A'] == a) & (pairwise['B'] == b)]
        if len(row) == 0:
            row = pairwise[(pairwise['A'] == b) & (pairwise['B'] == a)]

        if len(row) > 0:
            row = row.iloc[0]
            comparison = f"{a} vs {b}"
            print(f"{comparison:<35} {row['hedges']:>10.4f} {row['p-corr']:>12.4f} {row['sig']:>6}")

    print()
    print("Hedges' g interpretation: |g| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large")
    print("Negative g means first model has LOWER SSPE (better)")
    print()

    # Save pairwise results
    pairwise.to_csv(output_dir / 'doe_pairwise_comparisons.csv', index=False)
    print("Saved: doe_pairwise_comparisons.csv")

else:
    print("Skipping RM-ANOVA and pairwise tests (pingouin not installed)")
    print("Install with: pip install pingouin")

# -----------------------------------------------------------------------------
# CONDITIONAL WIN RATE ANALYSIS
# -----------------------------------------------------------------------------
print()
print("-" * 80)
print("CONDITIONAL WIN RATE ANALYSIS (PCReg_CV vs OLS)")
print("-" * 80)
print()

# Pivot to wide format
df_wide = df_doe.pivot_table(
    index=scenario_cols,
    columns='model_name',
    values=METRIC
).reset_index()

if 'PCReg_CV' in df_wide.columns and 'OLS' in df_wide.columns:
    df_wide['pcreg_wins'] = df_wide['PCReg_CV'] < df_wide['OLS']

    # Overall win rate
    overall_wins = df_wide['pcreg_wins'].sum()
    overall_total = len(df_wide)
    overall_rate = overall_wins / overall_total

    # Binomial test
    binom_result = stats.binomtest(overall_wins, overall_total, p=0.5, alternative='greater')

    print(f"Overall: PCReg_CV wins {overall_wins}/{overall_total} ({overall_rate:.1%})")
    print(f"Binomial test (H0: win rate = 50%): p = {binom_result.pvalue:.4f}")
    if binom_result.pvalue < 0.05:
        print("  -> PCReg_CV significantly outperforms OLS overall")
    print()

    # Win rates by factor
    print("Win Rates by Design Factor:")
    print("-" * 50)

    win_rate_results = []
    for factor in CATEGORIES + ['n_lots']:
        print(f"\n{factor}:")
        for level in sorted(df_wide[factor].unique()):
            mask = df_wide[factor] == level
            level_wins = df_wide.loc[mask, 'pcreg_wins'].sum()
            level_total = mask.sum()
            level_rate = level_wins / level_total

            # Binomial test
            binom = stats.binomtest(level_wins, level_total, p=0.5, alternative='greater')
            sig = '*' if binom.pvalue < 0.05 else ''

            print(f"  {level}: {level_rate:.1%} ({level_wins}/{level_total}) {sig}")

            win_rate_results.append({
                'factor': factor,
                'level': level,
                'wins': level_wins,
                'total': level_total,
                'win_rate': level_rate,
                'p_value': binom.pvalue,
                'significant': binom.pvalue < 0.05
            })

    # Save win rates
    pd.DataFrame(win_rate_results).to_csv(output_dir / 'doe_win_rates.csv', index=False)
    print("\nSaved: doe_win_rates.csv")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print(f"All plots saved to: {output_dir}")
print("=" * 80)
