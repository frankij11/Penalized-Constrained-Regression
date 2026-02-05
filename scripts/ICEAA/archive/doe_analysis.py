"""
doe_analysis.py
===============
Comprehensive Design of Experiments (DOE) statistical analysis for PCReg simulation study.

Implements:
1. Repeated Measures ANOVA with Greenhouse-Geisser correction
2. Pairwise comparisons with Holm-Bonferroni correction
3. Effect size analysis (partial eta-squared, Hedges' g)
4. Conditional superiority analysis ("when to use PCReg")
5. Decision rule generation via decision trees
6. Interaction visualizations (heatmaps)

Usage:
    python doe_analysis.py

Output:
    - CSV files with statistical test results
    - Heatmaps showing PCReg improvement by condition
    - Decision rules for when to use PCReg
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try importing optional packages
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Some analyses will be skipped.")
    print("Install with: pip install pingouin")

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ALPHA = 0.05  # Significance level
PRIMARY_METRIC = 'test_sspe'  # Primary outcome variable
METRIC_LABEL = 'Test SSPE (Sum of Squared Percentage Errors)'

# Models to focus on for DOE analysis
# Set to None to auto-discover from results, or specify a list to filter
# PCReg_CV_Tight uses true parameter values (oracle) - shown separately
MODELS_OF_INTEREST = None  # Auto-discover from results

# Practical models (excludes oracle/cheating models)
# Set to None to auto-discover (will exclude ORACLE_MODELS)
PRACTICAL_MODELS = None  # Auto-discover from results

# Oracle models (use true parameter values - for reference only)
# These will be excluded from practical analysis
ORACLE_MODELS = ['PCReg_CV_Tight']

# Design factors from simulation study
DESIGN_FACTORS = [
    'n_lots',
    'target_correlation',
    'cv_error',
    'learning_rate',
    'rate_effect'
]

FACTOR_LABELS = {
    'n_lots': 'Sample Size (n lots)',
    'target_correlation': 'Feature Correlation',
    'cv_error': 'Error Magnitude (CV)',
    'learning_rate': 'Learning Rate',
    'rate_effect': 'Rate Effect'
}

# Results directory
RESULTS_DIR = Path(__file__).parent / "output_v2"
OUTPUT_DIR = RESULTS_DIR / "doe_analysis"


# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================
def load_and_prepare_data(results_dir: Path,
                          models_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load simulation results and prepare for DOE analysis.

    Parameters
    ----------
    results_dir : Path
        Directory containing simulation_results.parquet
    models_filter : list of str, optional
        List of model names to include. If None, includes all models.

    Returns
    -------
    df : DataFrame
        Prepared data with scenario_id for repeated measures
    """
    print("Loading simulation results...")
    df = pd.read_parquet(results_dir / 'simulation_results.parquet')

    # Filter to converged models only
    df = df[df['converged'] == True].copy()

    # Filter to models of interest (if specified)
    if models_filter is not None:
        df = df[df['model_name'].isin(models_filter)].copy()

    # Create unique scenario identifier for repeated measures analysis
    # Each scenario is defined by design factors + replication
    df['scenario_id'] = df.groupby(
        DESIGN_FACTORS + ['replication']
    ).ngroup()

    print(f"Loaded {len(df):,} observations")
    print(f"Models: {sorted(df['model_name'].unique().tolist())}")
    print(f"Unique scenarios: {df['scenario_id'].nunique():,}")

    return df


def discover_models(results_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Discover all models in results and categorize as practical vs oracle.

    Parameters
    ----------
    results_dir : Path
        Directory containing simulation_results.parquet

    Returns
    -------
    all_models : list
        All model names found in results
    practical_models : list
        Models excluding oracle models
    """
    df = pd.read_parquet(results_dir / 'simulation_results.parquet')
    all_models = sorted(df['model_name'].unique().tolist())
    practical_models = [m for m in all_models if m not in ORACLE_MODELS]

    return all_models, practical_models


# ==============================================================================
# EFFECT SIZE INTERPRETATION
# ==============================================================================
def interpret_eta_squared(eta_sq: float) -> str:
    """Interpret partial eta-squared effect size."""
    if pd.isna(eta_sq):
        return 'undefined'
    if eta_sq < 0.01:
        return 'negligible'
    elif eta_sq < 0.06:
        return 'small'
    elif eta_sq < 0.14:
        return 'medium'
    else:
        return 'large'


def interpret_hedges_g(g: float) -> str:
    """Interpret Hedges' g effect size."""
    if pd.isna(g):
        return 'undefined'
    g_abs = abs(g)
    if g_abs < 0.2:
        size = 'negligible'
    elif g_abs < 0.5:
        size = 'small'
    elif g_abs < 0.8:
        size = 'medium'
    else:
        size = 'large'

    direction = 'favors_first' if g > 0 else 'favors_second'
    return f'{size} ({direction})'


# ==============================================================================
# REPEATED MEASURES ANOVA
# ==============================================================================
def run_repeated_measures_anova(df: pd.DataFrame) -> Dict:
    """
    Run repeated measures ANOVA with model as within-subject factor.

    The repeated measures structure accounts for the fact that all models
    are fit on the same data within each scenario-replication.

    Parameters
    ----------
    df : DataFrame
        Data with 'model_name', 'scenario_id', and PRIMARY_METRIC columns

    Returns
    -------
    results : dict
        Dictionary with ANOVA table, sphericity test, and effect sizes
    """
    if not HAS_PINGOUIN:
        print("Skipping RM-ANOVA: pingouin not installed")
        return {}

    print("\n" + "=" * 80)
    print("REPEATED MEASURES ANOVA")
    print("=" * 80)

    # Repeated measures ANOVA
    # Within-subject factor: model_name
    # Subject/blocking factor: scenario_id
    rm_aov = pg.rm_anova(
        data=df,
        dv=PRIMARY_METRIC,
        within='model_name',
        subject='scenario_id',
        correction=True  # Greenhouse-Geisser correction
    )

    # Handle different pingouin versions - column may be 'np2' or 'ng2'
    eta_sq_col = 'np2' if 'np2' in rm_aov.columns else 'ng2'
    eta_sq = rm_aov[eta_sq_col].values[0]

    # Add effect size interpretation
    rm_aov['effect_interpretation'] = rm_aov[eta_sq_col].apply(interpret_eta_squared)

    print("\nANOVA Results:")
    print(rm_aov.to_string())

    # Sphericity test
    sphericity = pg.sphericity(
        data=df,
        dv=PRIMARY_METRIC,
        within='model_name',
        subject='scenario_id'
    )

    print("\nSphericity Test (Mauchly):")
    print(f"  W = {sphericity.W:.4f}")
    print(f"  Chi-sq = {sphericity.chi2:.4f}")
    print(f"  p-value = {sphericity.pval:.4f}")
    print(f"  Sphericity violated: {sphericity.pval < ALPHA}")

    if sphericity.pval < ALPHA:
        print("  -> Greenhouse-Geisser correction applied (eps column)")

    # Handle p-value column names (may be 'p-GG-corr' or just use eps-corrected)
    p_col = 'p-GG-corr' if 'p-GG-corr' in rm_aov.columns else 'p-unc'
    p_value = rm_aov[p_col].values[0] if sphericity.pval < ALPHA else rm_aov['p-unc'].values[0]

    return {
        'anova_table': rm_aov,
        'sphericity': sphericity,
        'f_statistic': rm_aov['F'].values[0],
        'p_value': p_value,
        'eta_squared': eta_sq,
        'eta_sq_type': 'generalized' if eta_sq_col == 'ng2' else 'partial',
        'effect_interpretation': rm_aov['effect_interpretation'].values[0]
    }


# ==============================================================================
# PAIRWISE COMPARISONS
# ==============================================================================
def run_pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run pairwise comparisons between all models with Holm-Bonferroni correction.

    Parameters
    ----------
    df : DataFrame
        Data with 'model_name', 'scenario_id', and PRIMARY_METRIC columns

    Returns
    -------
    pairwise : DataFrame
        Pairwise comparison results with effect sizes and corrected p-values
    """
    if not HAS_PINGOUIN:
        print("Skipping pairwise comparisons: pingouin not installed")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS (Holm-Bonferroni Corrected)")
    print("=" * 80)

    # Pairwise tests with Holm correction
    pairwise = pg.pairwise_tests(
        data=df,
        dv=PRIMARY_METRIC,
        within='model_name',
        subject='scenario_id',
        padjust='holm',
        effsize='hedges',
        return_desc=True
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
        return ''

    pairwise['sig'] = pairwise['p-corr'].apply(sig_stars)

    # Add effect size interpretation
    pairwise['effect_interpretation'] = pairwise['hedges'].apply(interpret_hedges_g)

    # Display key comparisons
    print("\nKey Comparisons (PCReg_CV vs baselines):")
    print("-" * 80)

    key_pairs = [
        ('PCReg_CV', 'OLS'),
        ('PCReg_CV', 'OLS_LearnOnly'),
        ('PCReg_CV_Tight', 'OLS'),
        ('PCReg_CV_Tight', 'PCReg_CV'),
        ('PCReg_ConstrainOnly', 'OLS'),
        ('PCReg_CV_Wrong', 'PCReg_CV')
    ]

    for a, b in key_pairs:
        row = pairwise[(pairwise['A'] == a) & (pairwise['B'] == b)]
        if len(row) == 0:
            row = pairwise[(pairwise['A'] == b) & (pairwise['B'] == a)]

        if len(row) > 0:
            row = row.iloc[0]
            print(f"  {a} vs {b}:")
            print(f"    Hedges' g = {row['hedges']:.4f} ({row['effect_interpretation']})")
            print(f"    p-value (Holm) = {row['p-corr']:.4f} {row['sig']}")

    return pairwise


# ==============================================================================
# CONDITIONAL WIN RATE ANALYSIS
# ==============================================================================
def compute_conditional_win_rates(df: pd.DataFrame,
                                   baseline: str = 'OLS') -> pd.DataFrame:
    """
    Compute win rates for each model vs baseline across conditions.

    For each design factor level, this computes:
    - What % of scenarios does PCReg beat OLS?
    - Is this significantly > 50%? (binomial test)

    Parameters
    ----------
    df : DataFrame
        Simulation results
    baseline : str
        Model to compare against (default: 'OLS')

    Returns
    -------
    results : DataFrame
        Win rates and statistical tests by condition
    """
    print("\n" + "=" * 80)
    print(f"CONDITIONAL WIN RATES (vs {baseline})")
    print("=" * 80)

    # Pivot to wide format: one column per model
    df_wide = df.pivot_table(
        index=DESIGN_FACTORS + ['replication'],
        columns='model_name',
        values=PRIMARY_METRIC
    ).reset_index()

    results = []

    # Use models actually in the data
    models_in_data = [col for col in df_wide.columns if col not in DESIGN_FACTORS + ['replication']]

    for model in models_in_data:
        if model == baseline:
            continue

        # Check if model column exists
        if model not in df_wide.columns or baseline not in df_wide.columns:
            continue

        # Create win indicator
        wins = df_wide[model] < df_wide[baseline]

        # Overall win rate
        n_wins = wins.sum()
        n_total = len(wins)
        win_rate = n_wins / n_total

        # Binomial test: Is win rate significantly > 50%?
        p_value = stats.binomtest(n_wins, n_total, p=0.5, alternative='greater').pvalue

        results.append({
            'model': model,
            'baseline': baseline,
            'factor': 'Overall',
            'level': 'All',
            'n_wins': int(n_wins),
            'n_total': int(n_total),
            'win_rate': win_rate,
            'p_value': p_value,
            'significant': p_value < ALPHA
        })

        # Win rates by each factor
        for factor in DESIGN_FACTORS:
            for level in df_wide[factor].unique():
                mask = df_wide[factor] == level
                level_wins = wins[mask]

                n_wins_level = level_wins.sum()
                n_total_level = len(level_wins)

                if n_total_level > 0:
                    win_rate_level = n_wins_level / n_total_level
                    p_value_level = stats.binomtest(
                        n_wins_level, n_total_level, p=0.5, alternative='greater'
                    ).pvalue

                    results.append({
                        'model': model,
                        'baseline': baseline,
                        'factor': factor,
                        'level': level,
                        'n_wins': int(n_wins_level),
                        'n_total': int(n_total_level),
                        'win_rate': win_rate_level,
                        'p_value': p_value_level,
                        'significant': p_value_level < ALPHA
                    })

    results_df = pd.DataFrame(results)

    # Display summary
    print("\nOverall Win Rates:")
    print("-" * 60)
    overall = results_df[results_df['factor'] == 'Overall']
    for _, row in overall.iterrows():
        sig = '*' if row['significant'] else ''
        print(f"  {row['model']} vs {row['baseline']}: "
              f"{row['win_rate']:.1%} ({row['n_wins']}/{row['n_total']}) {sig}")

    return results_df


# ==============================================================================
# EFFECT SIZES BY CONDITION
# ==============================================================================
def compute_effect_sizes_by_condition(df: pd.DataFrame,
                                       model1: str = 'PCReg_CV',
                                       model2: str = 'OLS') -> pd.DataFrame:
    """
    Compute Hedges' g effect sizes for model1 vs model2 by design factors.

    Parameters
    ----------
    df : DataFrame
        Simulation results
    model1 : str
        First model (typically PCReg variant)
    model2 : str
        Second model (typically baseline)

    Returns
    -------
    effect_sizes : DataFrame
        Effect sizes with confidence intervals by condition
    """
    if not HAS_PINGOUIN:
        print("Skipping effect size analysis: pingouin not installed")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print(f"EFFECT SIZES BY CONDITION ({model1} vs {model2})")
    print("=" * 80)
    print("Positive Hedges' g = model1 has higher SSPE (model2 is better)")
    print("Negative Hedges' g = model1 has lower SSPE (model1 is better)")

    results = []

    for factor in DESIGN_FACTORS:
        print(f"\n{FACTOR_LABELS.get(factor, factor)}:")
        print("-" * 40)

        for level in sorted(df[factor].unique()):
            subset = df[df[factor] == level]

            m1_data = subset[subset['model_name'] == model1][PRIMARY_METRIC].dropna()
            m2_data = subset[subset['model_name'] == model2][PRIMARY_METRIC].dropna()

            if len(m1_data) > 10 and len(m2_data) > 10:
                # Compute Hedges' g (model1 - model2, so negative means model1 is better)
                g = pg.compute_effsize(m1_data, m2_data, eftype='hedges')

                # Bootstrap CI
                try:
                    ci = pg.compute_bootci(m1_data, m2_data, func='hedges',
                                           seed=42, n_boot=1000)
                    ci_low, ci_high = ci[0], ci[1]
                except:
                    ci_low, ci_high = np.nan, np.nan

                interp = interpret_hedges_g(g)

                results.append({
                    'factor': factor,
                    'level': level,
                    'model1': model1,
                    'model2': model2,
                    'hedges_g': g,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'interpretation': interp,
                    'n1': len(m1_data),
                    'n2': len(m2_data)
                })

                # Direction for interpretation
                if g < 0:
                    better = model1
                else:
                    better = model2

                print(f"  {level}: g = {g:+.4f} [{ci_low:.4f}, {ci_high:.4f}] -> {better} better")

    return pd.DataFrame(results)


# ==============================================================================
# DECISION RULE GENERATION
# ==============================================================================
def generate_decision_rules(df: pd.DataFrame,
                            model: str = 'PCReg_CV',
                            baseline: str = 'OLS') -> Dict:
    """
    Generate interpretable decision rules for when to use PCReg.

    Uses a decision tree classifier trained on scenario features
    to predict which model wins.

    Parameters
    ----------
    df : DataFrame
        Simulation results
    model : str
        Model to evaluate
    baseline : str
        Baseline to compare against

    Returns
    -------
    rules : dict
        Decision tree rules, feature importance, and accuracy
    """
    if not HAS_SKLEARN:
        print("Skipping decision rules: scikit-learn not installed")
        return {}

    print("\n" + "=" * 80)
    print(f"DECISION RULES: When to use {model} vs {baseline}")
    print("=" * 80)

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=DESIGN_FACTORS + ['replication'],
        columns='model_name',
        values=PRIMARY_METRIC
    ).reset_index()

    if model not in df_wide.columns or baseline not in df_wide.columns:
        print(f"Error: {model} or {baseline} not found in data")
        return {}

    # Binary outcome: Does model beat baseline?
    y = (df_wide[model] < df_wide[baseline]).astype(int)

    # Features
    X = df_wide[DESIGN_FACTORS].copy()

    # Convert categorical to numeric if needed
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes

    # Train decision tree
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=100,
        min_samples_split=200,
        random_state=42
    )
    tree.fit(X, y)

    # Extract rules
    rules_text = export_text(tree, feature_names=list(X.columns))

    # Feature importance
    importance = pd.DataFrame({
        'factor': DESIGN_FACTORS,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)

    accuracy = tree.score(X, y)

    print(f"\nDecision Tree Accuracy: {accuracy:.1%}")
    print("\nFeature Importance (which factors matter most):")
    print("-" * 40)
    for _, row in importance.iterrows():
        bar = '*' * int(row['importance'] * 40)
        print(f"  {row['factor']:<25} {row['importance']:.3f} {bar}")

    print("\nDecision Rules:")
    print("-" * 40)
    print(rules_text)

    return {
        'decision_tree': tree,
        'rules': rules_text,
        'feature_importance': importance,
        'accuracy': accuracy
    }


# ==============================================================================
# INTERACTION HEATMAPS
# ==============================================================================
def create_interaction_heatmap(df: pd.DataFrame,
                                factor1: str,
                                factor2: str,
                                model: str = 'PCReg_CV',
                                baseline: str = 'OLS',
                                output_path: Path = None):
    """
    Create heatmap showing model improvement over baseline by two factors.

    Parameters
    ----------
    df : DataFrame
        Simulation results
    factor1, factor2 : str
        Design factors for x and y axes
    model : str
        Model to evaluate
    baseline : str
        Baseline to compare against
    output_path : Path
        Directory to save the plot
    """
    # Pivot to wide format
    df_wide = df.pivot_table(
        index=DESIGN_FACTORS + ['replication'],
        columns='model_name',
        values=PRIMARY_METRIC
    ).reset_index()

    if model not in df_wide.columns or baseline not in df_wide.columns:
        print(f"Warning: Cannot create heatmap - {model} or {baseline} not in data")
        return

    # Compute improvement: (baseline - model) / baseline * 100
    # Positive = model is better (lower SSPE)
    df_wide['improvement'] = (df_wide[baseline] - df_wide[model]) / df_wide[baseline] * 100

    # Aggregate by factor1 x factor2
    heatmap_data = df_wide.groupby([factor1, factor2])['improvement'].mean().unstack()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': f'% Improvement ({model} vs {baseline})\nPositive = {model} better'},
        ax=ax
    )

    ax.set_xlabel(FACTOR_LABELS.get(factor2, factor2))
    ax.set_ylabel(FACTOR_LABELS.get(factor1, factor1))
    ax.set_title(f'{model} Improvement over {baseline}\nby {FACTOR_LABELS.get(factor1, factor1)} and {FACTOR_LABELS.get(factor2, factor2)}')

    plt.tight_layout()

    if output_path:
        filename = output_path / f'heatmap_{factor1}_{factor2}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.close(fig)


def create_all_heatmaps(df: pd.DataFrame, output_path: Path):
    """Create heatmaps for key factor combinations."""
    print("\n" + "=" * 80)
    print("CREATING INTERACTION HEATMAPS")
    print("=" * 80)

    # Key factor combinations to visualize
    factor_pairs = [
        ('n_lots', 'target_correlation'),
        ('n_lots', 'cv_error'),
        ('target_correlation', 'cv_error'),
        ('n_lots', 'learning_rate'),
        ('n_lots', 'rate_effect')
    ]

    for f1, f2 in factor_pairs:
        create_interaction_heatmap(df, f1, f2, output_path=output_path)


# ==============================================================================
# SUMMARY STATISTICS TABLE
# ==============================================================================
def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics table for paper.

    Parameters
    ----------
    df : DataFrame
        Simulation results

    Returns
    -------
    summary : DataFrame
        Summary statistics by model
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS TABLE")
    print("=" * 80)

    # Compute win counts
    scenario_cols = DESIGN_FACTORS + ['replication']
    win_counts = {}
    total_scenarios = 0

    for name, group in df.groupby(scenario_cols):
        group_valid = group.dropna(subset=[PRIMARY_METRIC])
        if len(group_valid) > 0:
            winner = group_valid.loc[group_valid[PRIMARY_METRIC].idxmin(), 'model_name']
            win_counts[winner] = win_counts.get(winner, 0) + 1
            total_scenarios += 1

    # Summary stats - use models actually in the data
    summary_data = []
    for model in sorted(df['model_name'].unique()):
        model_data = df[df['model_name'] == model][PRIMARY_METRIC].dropna()

        if len(model_data) > 0:
            summary_data.append({
                'Model': model,
                'Mean SSPE': model_data.mean(),
                'Std SSPE': model_data.std(),
                'Median SSPE': model_data.median(),
                'Q25 SSPE': model_data.quantile(0.25),
                'Q75 SSPE': model_data.quantile(0.75),
                'Win Count': win_counts.get(model, 0),
                'Win %': 100 * win_counts.get(model, 0) / total_scenarios if total_scenarios > 0 else 0,
                'N': len(model_data)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean SSPE')

    print("\n" + summary_df.to_string(index=False))

    return summary_df


# ==============================================================================
# RECOMMENDATIONS GENERATOR
# ==============================================================================
def generate_recommendations(win_rates_df: pd.DataFrame,
                             effect_sizes_df: pd.DataFrame) -> str:
    """
    Generate actionable recommendations based on analysis results.

    Parameters
    ----------
    win_rates_df : DataFrame
        Conditional win rates
    effect_sizes_df : DataFrame
        Effect sizes by condition

    Returns
    -------
    recommendations : str
        Text recommendations for when to use each method
    """
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS: When to Use Each Method")
    print("=" * 80)

    recommendations = []
    recommendations.append("Based on the DOE analysis, here are recommendations for method selection:\n")

    # Analyze PCReg_CV vs OLS
    if len(win_rates_df) > 0:
        pcreg_vs_ols = win_rates_df[
            (win_rates_df['model'] == 'PCReg_CV') &
            (win_rates_df['baseline'] == 'OLS')
        ]

        # Find conditions where PCReg clearly wins
        strong_pcreg = pcreg_vs_ols[
            (pcreg_vs_ols['win_rate'] > 0.6) &
            (pcreg_vs_ols['significant'] == True)
        ]

        if len(strong_pcreg) > 0:
            recommendations.append("USE PCReg_CV when:")
            for _, row in strong_pcreg.iterrows():
                if row['factor'] != 'Overall':
                    recommendations.append(f"  - {FACTOR_LABELS.get(row['factor'], row['factor'])} = {row['level']} "
                                          f"(win rate: {row['win_rate']:.0%})")

        # Find conditions where OLS is competitive
        ols_competitive = pcreg_vs_ols[
            (pcreg_vs_ols['win_rate'] < 0.55) &
            (pcreg_vs_ols['factor'] != 'Overall')
        ]

        if len(ols_competitive) > 0:
            recommendations.append("\nOLS is competitive when:")
            for _, row in ols_competitive.iterrows():
                recommendations.append(f"  - {FACTOR_LABELS.get(row['factor'], row['factor'])} = {row['level']} "
                                      f"(PCReg win rate: {row['win_rate']:.0%})")

    # Key factors from effect sizes
    if len(effect_sizes_df) > 0:
        recommendations.append("\nKEY FACTORS affecting method performance:")

        for factor in DESIGN_FACTORS:
            factor_effects = effect_sizes_df[effect_sizes_df['factor'] == factor]
            if len(factor_effects) > 0:
                g_range = factor_effects['hedges_g'].max() - factor_effects['hedges_g'].min()
                if g_range > 0.2:
                    recommendations.append(f"  - {FACTOR_LABELS.get(factor, factor)}: Effect varies substantially (range: {g_range:.2f})")

    recommendations.append("\nGENERAL GUIDANCE:")
    recommendations.append("  - PCReg benefits most from incorporating prior knowledge via constraints")
    recommendations.append("  - With small samples (n < 10), constraints become more valuable")
    recommendations.append("  - PCReg_CV_Tight (well-specified constraints) is optimal when bounds are known")
    recommendations.append("  - PCReg_CV_Wrong shows robustness to constraint misspecification")

    rec_text = '\n'.join(recommendations)
    print(rec_text)

    return rec_text


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def run_full_doe_analysis():
    """Run complete DOE analysis and save all outputs."""

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("DOE ANALYSIS FOR PCREG SIMULATION STUDY")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Significance Level (alpha): {ALPHA}")
    print("=" * 80)

    # Auto-discover models from results
    all_models, practical_models = discover_models(RESULTS_DIR)

    # Use configured lists if specified, otherwise use discovered
    models_of_interest = MODELS_OF_INTEREST if MODELS_OF_INTEREST else all_models
    practical_model_list = PRACTICAL_MODELS if PRACTICAL_MODELS else practical_models

    print(f"\nDiscovered {len(all_models)} models in results:")
    for m in all_models:
        oracle_tag = " (oracle)" if m in ORACLE_MODELS else ""
        print(f"  - {m}{oracle_tag}")

    # Load data (all models including oracle)
    df_all = load_and_prepare_data(RESULTS_DIR, models_filter=models_of_interest)

    # Check if we have enough data
    if len(df_all) == 0:
        print("ERROR: No data found. Run simulation study first.")
        return

    # Create practical-only dataset (excludes oracle models)
    df_practical = df_all[df_all['model_name'].isin(practical_model_list)].copy()

    print("\n" + "=" * 80)
    print("PART 1: PRACTICAL MODELS ONLY (Excludes Oracle)")
    print(f"Models ({len(practical_model_list)}): {practical_model_list}")
    print("=" * 80)
    print(f"\nNote: Oracle models {ORACLE_MODELS} are excluded because they use true")
    print("      parameter values which are not available in practice.")

    # 1. Repeated Measures ANOVA - PRACTICAL MODELS
    print("\n" + "-" * 80)
    print("PRACTICAL MODELS ANALYSIS")
    print("-" * 80)
    anova_practical = run_repeated_measures_anova(df_practical)
    if anova_practical and 'anova_table' in anova_practical:
        anova_practical['anova_table'].to_csv(OUTPUT_DIR / 'rm_anova_practical.csv', index=False)

    # 2. Pairwise Comparisons - PRACTICAL MODELS
    pairwise_practical = run_pairwise_comparisons(df_practical)
    if len(pairwise_practical) > 0:
        pairwise_practical.to_csv(OUTPUT_DIR / 'pairwise_practical.csv', index=False)

    # 3. Conditional Win Rates - PRACTICAL MODELS
    win_rates = compute_conditional_win_rates(df_practical, baseline='OLS')
    if len(win_rates) > 0:
        win_rates.to_csv(OUTPUT_DIR / 'win_rates_by_condition.csv', index=False)

    # 4. Effect Sizes by Condition
    effect_sizes = compute_effect_sizes_by_condition(df_practical, model1='PCReg_CV', model2='OLS')
    if len(effect_sizes) > 0:
        effect_sizes.to_csv(OUTPUT_DIR / 'effect_sizes_by_condition.csv', index=False)

    # 5. Decision Rules
    decision_rules = generate_decision_rules(df_practical, model='PCReg_CV', baseline='OLS')
    if decision_rules and 'rules' in decision_rules:
        with open(OUTPUT_DIR / 'decision_rules.txt', 'w') as f:
            f.write(f"Decision Rules for PCReg_CV vs OLS\n")
            f.write(f"(Practical models only - excludes oracle)\n")
            f.write(f"Accuracy: {decision_rules['accuracy']:.1%}\n\n")
            f.write(decision_rules['rules'])

        decision_rules['feature_importance'].to_csv(
            OUTPUT_DIR / 'feature_importance.csv', index=False
        )

    # 6. Interaction Heatmaps (practical models)
    create_all_heatmaps(df_practical, OUTPUT_DIR)

    # 7. Summary Statistics Table - PRACTICAL MODELS
    print("\n" + "=" * 80)
    print("SUMMARY: PRACTICAL MODELS")
    print("=" * 80)
    summary_practical = generate_summary_table(df_practical)
    summary_practical.to_csv(OUTPUT_DIR / 'summary_practical.csv', index=False)

    # 8. Recommendations (based on practical models)
    recommendations = generate_recommendations(win_rates, effect_sizes)
    with open(OUTPUT_DIR / 'recommendations.txt', 'w') as f:
        f.write(recommendations)

    # =========================================================================
    # PART 2: ORACLE MODEL REFERENCE (for comparison only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: ORACLE MODEL REFERENCE (PCReg_CV_Tight)")
    print("=" * 80)
    print("\nNote: PCReg_CV_Tight uses TRUE parameter values for constraint bounds.")
    print("      This represents the theoretical best-case for PCReg with perfect")
    print("      prior knowledge. Included for reference only - not practical.")

    # Summary including oracle
    print("\n" + "-" * 80)
    print("ALL MODELS (Including Oracle)")
    print("-" * 80)
    summary_all = generate_summary_table(df_all)
    summary_all.to_csv(OUTPUT_DIR / 'summary_all_models.csv', index=False)

    # Run ANOVA with all models for comparison
    print("\n" + "-" * 80)
    print("RM-ANOVA WITH ORACLE (for reference)")
    print("-" * 80)
    anova_all = run_repeated_measures_anova(df_all)
    if anova_all and 'anova_table' in anova_all:
        anova_all['anova_table'].to_csv(OUTPUT_DIR / 'rm_anova_all_models.csv', index=False)

    # Oracle vs practical comparison
    print("\n" + "-" * 80)
    print("ORACLE vs PRACTICAL COMPARISON")
    print("-" * 80)

    oracle_data = df_all[df_all['model_name'] == ORACLE_MODEL][PRIMARY_METRIC]
    practical_best = df_practical.groupby('model_name')[PRIMARY_METRIC].mean().idxmin()
    practical_data = df_practical[df_practical['model_name'] == practical_best][PRIMARY_METRIC]

    print(f"\nOracle (PCReg_CV_Tight) Mean SSPE: {oracle_data.mean():.4f}")
    print(f"Best Practical ({practical_best}) Mean SSPE: {practical_data.mean():.4f}")
    print(f"Gap: {((practical_data.mean() - oracle_data.mean()) / oracle_data.mean() * 100):.1f}% higher than oracle")

    # Final summary
    print("\n" + "=" * 80)
    print("DOE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print("  PRACTICAL MODELS (use these for paper):")
    print("    - rm_anova_practical.csv")
    print("    - pairwise_practical.csv")
    print("    - summary_practical.csv")
    print("    - win_rates_by_condition.csv")
    print("    - recommendations.txt")
    print("  ORACLE REFERENCE:")
    print("    - rm_anova_all_models.csv")
    print("    - summary_all_models.csv")
    print("\nAll files generated:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    run_full_doe_analysis()
