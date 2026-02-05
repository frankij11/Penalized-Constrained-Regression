# %%
import pandas as pd
import numpy as np
from pathlib import Path

PARENT = Path(__file__).resolve().parent
RESULTS_PATH = PARENT / "Output_v2" / "simulation_results.parquet"

class CONFIG:
    all_models=True

if CONFIG.all_models:
    MODELS_TO_COMPARE = [
        "OLS","PCReg_GCV","OLS_LearnOnly","RidgeCV", "BayesianRidge", "LassoCV"
    ]
else:
    MODELS_TO_COMPARE = [
        "OLS","PCReg_GCV"#,"OLS_LearnOnly","RidgeCV", "BayesianRidge", "LassoCV"
]

METRICS_TO_COMPARE = [
    "test_mape","test_mse", "b_error","c_error","T1_error",
    "b_bias", "c_bias", "T1_bias"
]


# find seeds where OLS produces "bad" coefficients but good R2
def find_bad_ols_coefs(df, r2=.8):
    seeds = df.query("(LC_est>1 | RC_est>1 | LC_est <.7 | RC_est<.7) and r2>0.8 and model_name=='OLS'")["seed"].unique()
    seeds_all = df.query("(LC_est>1 | RC_est>1 | LC_est <.7 | RC_est<.7) and r2>0.8 and not model_name.str.contains('PC')")["seed"].unique()
    df = df.assign(bad_ols_coefs=lambda x: x["seed"].isin(seeds), bad_non_pc_coefs=lambda x: x["seed"].isin(seeds_all))
    return df

def find_good_pcreg_fits(df, r2=.8):
    seeds=df.query("bad_ols_coefs==1 and rank_test_mape==1 and model_name.str.lower().str.contains('pc') and alpha>0")["seed"].unique()
    df = df.assign(pcreg_improves_bad_ols_coef=lambda x: x["seed"].isin(seeds))
    seeds=df.query("bad_ols_coefs==0 and rank_test_mape==1 and model_name.str.lower().str.contains('pc') and alpha>0")["seed"].unique()
    df = df.assign(pcreg_improves_good_ols_coef=lambda x: x["seed"].isin(seeds))

    return df

def find_pcreg_beats_all_baselines(df):
    """
    Find scenarios where PCReg_GCV outperforms both constraints-only AND Ridge-only.

    Primary criterion: lower true_coefs_error (T1_error + b_error + c_error)
    Secondary criterion: lower test_mape

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with model_name, seed, test_mape, LC_est, RC_est,
        T1_error, b_error, c_error columns

    Returns
    -------
    pd.DataFrame
        Analysis dataframe with one row per seed, containing comparison metrics
    """
    # Compute true_coefs_error if not present
    if 'true_coefs_error' not in df.columns:
        df = df.assign(true_coefs_error=lambda x: x['T1_error'] + x['b_error'] + x['c_error'])

    # Models to compare
    models = ['OLS', 'PCReg_ConstrainOnly', 'RidgeCV', 'PCReg_GCV']
    df_subset = df[df['model_name'].isin(models)].copy()

    # Pivot to get one row per seed for each metric
    pivot_mape = df_subset.pivot_table(index='seed', columns='model_name', values='test_mape')
    pivot_coef_err = df_subset.pivot_table(index='seed', columns='model_name', values='true_coefs_error')
    pivot_lc = df_subset.pivot_table(index='seed', columns='model_name', values='LC_est')
    pivot_rc = df_subset.pivot_table(index='seed', columns='model_name', values='RC_est')
    pivot_test_n = df_subset.pivot_table(index='seed', columns='model_name', values='test_n_lots')

    # Build analysis dataframe
    analysis = pd.DataFrame({
        # Test MAPE
        'OLS_mape': pivot_mape.get('OLS'),
        'ConstrainOnly_mape': pivot_mape.get('PCReg_ConstrainOnly'),
        'RidgeCV_mape': pivot_mape.get('RidgeCV'),
        'PCReg_GCV_mape': pivot_mape.get('PCReg_GCV'),
        # True coefficient error
        'OLS_coef_err': pivot_coef_err.get('OLS'),
        'ConstrainOnly_coef_err': pivot_coef_err.get('PCReg_ConstrainOnly'),
        'RidgeCV_coef_err': pivot_coef_err.get('RidgeCV'),
        'PCReg_GCV_coef_err': pivot_coef_err.get('PCReg_GCV'),
        # Learning curve estimates
        'PCReg_GCV_LC': pivot_lc.get('PCReg_GCV'),
        'PCReg_GCV_RC': pivot_rc.get('PCReg_GCV'),
        'OLS_LC': pivot_lc.get('OLS'),
        'OLS_RC': pivot_rc.get('OLS'),
        'RidgeCV_LC': pivot_lc.get('RidgeCV'),
        'RidgeCV_RC': pivot_rc.get('RidgeCV'),
        'ConstrainOnly_LC': pivot_lc.get('PCReg_ConstrainOnly'),
        'ConstrainOnly_RC': pivot_rc.get('PCReg_ConstrainOnly'),
        # Test lots
        'test_n_lots': pivot_test_n.get('OLS'),
    })

    # Calculate improvements in coefficient error
    analysis['coef_improve_vs_constrain'] = (
        (analysis['ConstrainOnly_coef_err'] - analysis['PCReg_GCV_coef_err'])
        / analysis['ConstrainOnly_coef_err'].replace(0, np.nan) * 100
    )
    analysis['coef_improve_vs_ridge'] = (
        (analysis['RidgeCV_coef_err'] - analysis['PCReg_GCV_coef_err'])
        / analysis['RidgeCV_coef_err'].replace(0, np.nan) * 100
    )
    analysis['coef_improve_vs_ols'] = (
        (analysis['OLS_coef_err'] - analysis['PCReg_GCV_coef_err'])
        / analysis['OLS_coef_err'].replace(0, np.nan) * 100
    )

    # Calculate improvements in test MAPE
    analysis['mape_improve_vs_constrain'] = (
        (analysis['ConstrainOnly_mape'] - analysis['PCReg_GCV_mape'])
        / analysis['ConstrainOnly_mape'].replace(0, np.nan) * 100
    )
    analysis['mape_improve_vs_ridge'] = (
        (analysis['RidgeCV_mape'] - analysis['PCReg_GCV_mape'])
        / analysis['RidgeCV_mape'].replace(0, np.nan) * 100
    )

    # Conditions for a "compelling" example
    # PRIMARY: Must beat both on coefficient error
    analysis['beats_both_coef'] = (
        (analysis['PCReg_GCV_coef_err'] < analysis['ConstrainOnly_coef_err']) &
        (analysis['PCReg_GCV_coef_err'] < analysis['RidgeCV_coef_err'])
    )
    # SECONDARY: Also beats both on test MAPE
    analysis['beats_both_mape'] = (
        (analysis['PCReg_GCV_mape'] < analysis['ConstrainOnly_mape']) &
        (analysis['PCReg_GCV_mape'] < analysis['RidgeCV_mape'])
    )
    # Sensible coefficients for PCReg_GCV (stricter: <= 0.96 to avoid boundary solutions)
    analysis['sensible_coef'] = (
        (analysis['PCReg_GCV_LC'] >= 0.7) & (analysis['PCReg_GCV_LC'] <= 0.96) &
        (analysis['PCReg_GCV_RC'] >= 0.7) & (analysis['PCReg_GCV_RC'] <= 0.96)
    )
    # OLS produces problematic coefficients
    analysis['ols_problematic'] = (
        (analysis['OLS_LC'] > 1) | (analysis['OLS_RC'] > 1) |
        (analysis['OLS_LC'] < 0.7) | (analysis['OLS_RC'] < 0.7)
    )

    # Combined filter: must beat both on coef error, prefer if also beats on MAPE
    mask = (
        analysis['beats_both_coef'] &
        analysis['sensible_coef'] &
        analysis['ols_problematic']
    )

    return analysis[mask].sort_values('PCReg_GCV_coef_err', ascending=True)


def select_best_motivating_example(analysis_df, df_results, df_study_data):
    """
    Select the single best motivating example.

    Selection criteria:
    1. Lowest true_coefs_error for PCReg_GCV (50% weight)
    2. Largest improvement margin over both baselines in coef error (30% weight)
    3. OLS clearly wrong with impossible coefficients >1 (20% weight)

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Output from find_pcreg_beats_all_baselines()
    df_results : pd.DataFrame
        Full simulation results
    df_study_data : pd.DataFrame
        Study data with raw observations

    Returns
    -------
    seed : int
        Selected seed
    example_results : pd.DataFrame
        All model results for this seed
    example_data : pd.DataFrame
        Raw data for this seed
    summary : pd.Series
        Summary statistics for selected example
    """
    if len(analysis_df) == 0:
        raise ValueError("No candidates found matching criteria")

    # Score candidates
    candidates = analysis_df.copy()

    # Prefer examples where OLS produces clearly impossible coefficients (>1)
    candidates['ols_clearly_wrong'] = (
        (candidates['OLS_LC'] > 1) | (candidates['OLS_RC'] > 1)
    ).astype(int)

    # Min improvement over both baselines
    candidates['min_coef_improvement'] = candidates[['coef_improve_vs_constrain', 'coef_improve_vs_ridge']].min(axis=1)

    # Normalize for scoring
    max_coef_err = candidates['PCReg_GCV_coef_err'].max()
    min_coef_err = candidates['PCReg_GCV_coef_err'].min()
    candidates['norm_coef_err'] = 1 - (candidates['PCReg_GCV_coef_err'] - min_coef_err) / (max_coef_err - min_coef_err + 1e-10)

    max_improvement = candidates['min_coef_improvement'].max()
    min_improvement = candidates['min_coef_improvement'].min()
    candidates['norm_improvement'] = (candidates['min_coef_improvement'] - min_improvement) / (max_improvement - min_improvement + 1e-10)

    # Composite score
    candidates['score'] = (
        candidates['norm_coef_err'] * 0.5 +
        candidates['norm_improvement'] * 0.3 +
        candidates['ols_clearly_wrong'] * 0.2
    )

    # Also prefer if beats both on MAPE (bonus)
    candidates.loc[candidates['beats_both_mape'], 'score'] += 0.1

    # Select best
    best_seed = candidates['score'].idxmax()

    # Extract results and data
    example_results = df_results[df_results['seed'] == best_seed].copy()
    example_data = df_study_data[df_study_data['seed'] == best_seed].copy()

    return best_seed, example_results, example_data, candidates.loc[best_seed]


def export_motivating_example(seed, example_results, example_data, summary_stats,
                               output_dir, prefix="pcr_beats_all"):
    """
    Export motivating example to CSV files.

    Creates:
    - {prefix}_example_data.csv - Raw lot data
    - {prefix}_example_results.csv - All model results for this seed
    - {prefix}_comparison.csv - Head-to-head comparison table
    - {prefix}_summary.csv - Summary comparison table

    Parameters
    ----------
    seed : int
        Selected seed
    example_results : pd.DataFrame
        All model results for this seed
    example_data : pd.DataFrame
        Raw data for this seed
    summary_stats : pd.Series
        Summary statistics from selection
    output_dir : Path
        Output directory
    prefix : str
        File prefix (default "pcr_beats_all")
    """
    output_dir = Path(output_dir)

    # 1. Export raw data
    example_data.to_csv(output_dir / f"{prefix}_example_data.csv", index=False)

    # 2. Export all model results for this seed
    example_results.to_csv(output_dir / f"{prefix}_example_results.csv", index=False)

    # 3. Create comparison table for key models
    key_models = ['OLS', 'RidgeCV', 'PCReg_ConstrainOnly', 'PCReg_GCV']

    # Add true_coefs_error if not present
    if 'true_coefs_error' not in example_results.columns:
        example_results = example_results.assign(
            true_coefs_error=lambda x: x['T1_error'] + x['b_error'] + x['c_error']
        )

    comparison = example_results[example_results['model_name'].isin(key_models)][
        ['model_name', 'test_mape', 'true_coefs_error', 'T1_error', 'b_error', 'c_error',
         'LC_est', 'RC_est', 'T1_est', 'r2', 'alpha']
    ].sort_values('true_coefs_error')
    comparison.to_csv(output_dir / f"{prefix}_comparison.csv", index=False)

    # 4. Export summary
    summary_df = pd.DataFrame([{
        'seed': seed,
        'coef_improve_vs_constrainonly_pct': summary_stats['coef_improve_vs_constrain'],
        'coef_improve_vs_ridge_pct': summary_stats['coef_improve_vs_ridge'],
        'coef_improve_vs_ols_pct': summary_stats['coef_improve_vs_ols'],
        'mape_improve_vs_constrainonly_pct': summary_stats.get('mape_improve_vs_constrain', np.nan),
        'mape_improve_vs_ridge_pct': summary_stats.get('mape_improve_vs_ridge', np.nan),
        'PCReg_GCV_coef_err': summary_stats['PCReg_GCV_coef_err'],
        'ConstrainOnly_coef_err': summary_stats['ConstrainOnly_coef_err'],
        'RidgeCV_coef_err': summary_stats['RidgeCV_coef_err'],
        'OLS_coef_err': summary_stats['OLS_coef_err'],
        'PCReg_GCV_mape': summary_stats['PCReg_GCV_mape'],
        'ConstrainOnly_mape': summary_stats['ConstrainOnly_mape'],
        'RidgeCV_mape': summary_stats['RidgeCV_mape'],
        'OLS_mape': summary_stats['OLS_mape'],
        'PCReg_GCV_LC': summary_stats['PCReg_GCV_LC'],
        'PCReg_GCV_RC': summary_stats['PCReg_GCV_RC'],
        'test_n_lots': summary_stats['test_n_lots'],
        'beats_both_mape': summary_stats['beats_both_mape'],
    }])
    summary_df.to_csv(output_dir / f"{prefix}_summary.csv", index=False)

    print(f"\nMotivating Example Exported (seed={seed})")
    print("=" * 70)
    print(f"  PRIMARY CRITERION - Coefficient Error:")
    print(f"    PCReg_GCV:          {summary_stats['PCReg_GCV_coef_err']:.4f}")
    print(f"    PCReg_ConstrainOnly:{summary_stats['ConstrainOnly_coef_err']:.4f} ({summary_stats['coef_improve_vs_constrain']:.1f}% worse)")
    print(f"    RidgeCV:            {summary_stats['RidgeCV_coef_err']:.4f} ({summary_stats['coef_improve_vs_ridge']:.1f}% worse)")
    print(f"    OLS:                {summary_stats['OLS_coef_err']:.4f} ({summary_stats['coef_improve_vs_ols']:.1f}% worse)")
    print(f"  SECONDARY - Test MAPE:")
    print(f"    PCReg_GCV:          {summary_stats['PCReg_GCV_mape']:.4f}")
    print(f"    PCReg_ConstrainOnly:{summary_stats['ConstrainOnly_mape']:.4f}")
    print(f"    RidgeCV:            {summary_stats['RidgeCV_mape']:.4f}")
    print(f"    OLS:                {summary_stats['OLS_mape']:.4f}")
    print(f"  Beats both on MAPE too: {summary_stats['beats_both_mape']}")
    print(f"  PCReg_GCV coefficients: LC={summary_stats['PCReg_GCV_LC']:.3f}, RC={summary_stats['PCReg_GCV_RC']:.3f}")
    print(f"  Test lots: {int(summary_stats['test_n_lots']) if not pd.isna(summary_stats['test_n_lots']) else 'N/A'}")
    print("=" * 70)


def calculate_bias(df, df_study_data=None):
    """
    Calculate bias for b, c, and T1 estimates.

    Bias = estimated - true

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with model estimates (b, c, T1_est).
        May already contain b_true, c_true, T1_true columns.
    df_study_data : pd.DataFrame, optional
        Study data with true values (b_true, c_true, T1_true).
        Only used if true values are not already in df.

    Returns
    -------
    pd.DataFrame
        DataFrame with bias columns added (b_bias, c_bias, T1_bias)
    """
    df = df.copy()

    # Check if true values already exist in df
    has_true_values = all(col in df.columns for col in ['b_true', 'c_true', 'T1_true'])

    if not has_true_values and df_study_data is not None:
        # Get unique true values per seed from study data
        true_values = (df_study_data
                       .groupby('seed')[['b_true', 'c_true', 'T1_true']]
                       .first()
                       .reset_index())
        # Merge true values into results
        df = df.merge(true_values, on='seed', how='left')

    # Calculate bias (estimated - true)
    df['b_bias'] = df['b'] - df['b_true']
    df['c_bias'] = df['c'] - df['c_true']
    df['T1_bias'] = df['T1_est'] - df['T1_true']

    return df

def rank_models(df, criteria='test_mape'):
    df = df.copy()
    df["rank_"+ criteria] = df.groupby(["seed"])[criteria].rank(method="average", ascending=True)
    return df

def pct_beats_ols(df, criteria='test_mape'):
    rank_col = f"rank_{criteria}"
    
    # Get OLS rank for each seed
    ols_ranks = (df.query("model_name == 'OLS'")
                   .set_index("seed")[rank_col]
                   .rename("ols_rank"))
    
    # Join OLS rank back and compare
    df = df.join(ols_ranks, on="seed")
    df[f"beats_ols_{criteria}"] = df[rank_col] < df["ols_rank"]
    df = df.drop(columns=["ols_rank"])
    
    return df
def get_subset_of_models(df, models, all_models=True):
    if all_models:
        return df
    else:
        return df.query("model_name in @models")


if __name__ == "__main__":
    # Load study data first (needed for bias calculation)
    df_study_data = pd.read_parquet(RESULTS_PATH.parent / "simulation_study_data.parquet")

    df = (pd.read_parquet(RESULTS_PATH)
          .pipe(get_subset_of_models, MODELS_TO_COMPARE, CONFIG.all_models)
          .pipe(calculate_bias, df_study_data)
          .pipe(find_bad_ols_coefs)
          .pipe(rank_models, criteria='test_mape')
          .pipe(pct_beats_ols, criteria='test_mape')
          .pipe(find_good_pcreg_fits)
          .pipe(rank_models, criteria='b_error')
          .pipe(rank_models, criteria='c_error')
          .pipe(rank_models, criteria='T1_error')
          .pipe(rank_models, criteria='b_bias')
          .pipe(rank_models, criteria='c_bias')
          .pipe(rank_models, criteria='T1_bias')
    )
    filename = "All_Models" if CONFIG.all_models else "OLS_vs_PCReg"
    df.to_csv(PARENT / "Output_v2" / f"simulation_results_extended_{filename}.csv", index=False)

    # =============================================================================
    # MOTIVATING EXAMPLE: PCReg beats BOTH Constraints-Only AND Ridge-Only
    # =============================================================================
    # Primary criterion: lower true_coefs_error (coefficient recovery)
    # Secondary criterion: lower test_mape (predictive accuracy)
    print("\n" + "="*70)
    print("Finding Motivating Example: PCReg vs Constraints-Only vs Ridge")
    print("="*70)

    # Add true_coefs_error to df for analysis
    df = df.assign(true_coefs_error=lambda x: x['T1_error'] + x['b_error'] + x['c_error'])

    # Find candidates where PCReg_GCV beats both baselines on coefficient error
    candidates = find_pcreg_beats_all_baselines(df)

    print(f"Found {len(candidates)} candidates where PCReg_GCV beats both baselines on coef error")
    print(f"  - Of these, {candidates['beats_both_mape'].sum()} also beat both on test_mape")

    if len(candidates) > 0:
        # Select best example
        best_seed, example_results, example_data, summary = select_best_motivating_example(
            candidates, df, df_study_data
        )

        # Export
        export_motivating_example(
            best_seed, example_results, example_data, summary,
            output_dir=PARENT / "Output_v2",
            prefix="pcr_beats_all"
        )

        # Also export top 10 candidates for reference
        candidates.head(10).to_csv(PARENT / "Output_v2" / "pcr_beats_all_top_candidates.csv")
        print(f"\nTop 10 candidates exported to pcr_beats_all_top_candidates.csv")

        # Print comparison table
        print("\nModel Comparison for Best Example (sorted by true_coefs_error):")
        key_models = ['OLS', 'RidgeCV', 'PCReg_ConstrainOnly', 'PCReg_GCV']
        print(example_results[example_results['model_name'].isin(key_models)][
            ['model_name', 'true_coefs_error', 'test_mape', 'LC_est', 'RC_est', 'T1_est', 'r2']
        ].sort_values('true_coefs_error').to_string())

        # =============================================================================
        # HEAD-TO-HEAD: OLS vs PCReg_GCV for the motivating example
        # =============================================================================
        print("\n" + "="*70)
        print(f"HEAD-TO-HEAD: OLS vs PCReg_GCV (seed={best_seed})")
        print("="*70)

        ols_row = example_results[example_results['model_name'] == 'OLS'].iloc[0]
        pcreg_row = example_results[example_results['model_name'] == 'PCReg_GCV'].iloc[0]

        # Get true parameters from the study data
        true_b = example_data['b_true'].iloc[0]
        true_c = example_data['c_true'].iloc[0]
        true_T1 = example_data['T1_true'].iloc[0]
        true_LC = 2 ** true_b
        true_RC = 2 ** true_c

        print(f"\n  TRUE PARAMETERS:")
        print(f"    T1 = {true_T1:.2f}, b = {true_b:.4f}, c = {true_c:.4f}")
        print(f"    LC = {true_LC:.4f}, RC = {true_RC:.4f}")

        print(f"\n  OLS ESTIMATES:")
        print(f"    T1 = {ols_row['T1_est']:.2f}, b = {ols_row['b']:.4f}, c = {ols_row['c']:.4f}")
        print(f"    LC = {ols_row['LC_est']:.4f}, RC = {ols_row['RC_est']:.4f}")
        print(f"    Coefficient Error = {ols_row['true_coefs_error']:.4f}")
        print(f"    Test MAPE = {ols_row['test_mape']:.4f}")
        print(f"    R² = {ols_row['r2']:.4f}")

        print(f"\n  PCReg_GCV ESTIMATES:")
        print(f"    T1 = {pcreg_row['T1_est']:.2f}, b = {pcreg_row['b']:.4f}, c = {pcreg_row['c']:.4f}")
        print(f"    LC = {pcreg_row['LC_est']:.4f}, RC = {pcreg_row['RC_est']:.4f}")
        print(f"    Coefficient Error = {pcreg_row['true_coefs_error']:.4f}")
        print(f"    Test MAPE = {pcreg_row['test_mape']:.4f}")
        print(f"    R² = {pcreg_row['r2']:.4f}")
        print(f"    Alpha = {pcreg_row['alpha']:.6f}")

        # Calculate improvements
        coef_improvement = (ols_row['true_coefs_error'] - pcreg_row['true_coefs_error']) / ols_row['true_coefs_error'] * 100
        mape_improvement = (ols_row['test_mape'] - pcreg_row['test_mape']) / ols_row['test_mape'] * 100

        print(f"\n  IMPROVEMENT (PCReg_GCV vs OLS):")
        print(f"    Coefficient Error: {coef_improvement:.1f}% better")
        print(f"    Test MAPE: {mape_improvement:.1f}% better")
        print(f"    OLS LC > 1 (impossible): {ols_row['LC_est'] > 1}")
        print(f"    OLS RC < 0.7 (implausible): {ols_row['RC_est'] < 0.7}")
        print("="*70)

        # Export head-to-head comparison
        h2h_comparison = pd.DataFrame([
            {'Parameter': 'T1_true', 'True': true_T1, 'OLS': ols_row['T1_est'], 'PCReg_GCV': pcreg_row['T1_est']},
            {'Parameter': 'b_true', 'True': true_b, 'OLS': ols_row['b'], 'PCReg_GCV': pcreg_row['b']},
            {'Parameter': 'c_true', 'True': true_c, 'OLS': ols_row['c'], 'PCReg_GCV': pcreg_row['c']},
            {'Parameter': 'LC (2^b)', 'True': true_LC, 'OLS': ols_row['LC_est'], 'PCReg_GCV': pcreg_row['LC_est']},
            {'Parameter': 'RC (2^c)', 'True': true_RC, 'OLS': ols_row['RC_est'], 'PCReg_GCV': pcreg_row['RC_est']},
            {'Parameter': 'Coef_Error', 'True': 0, 'OLS': ols_row['true_coefs_error'], 'PCReg_GCV': pcreg_row['true_coefs_error']},
            {'Parameter': 'Test_MAPE', 'True': 0, 'OLS': ols_row['test_mape'], 'PCReg_GCV': pcreg_row['test_mape']},
            {'Parameter': 'R2', 'True': 1, 'OLS': ols_row['r2'], 'PCReg_GCV': pcreg_row['r2']},
        ])
        h2h_comparison.to_csv(PARENT / "Output_v2" / "pcr_beats_all_ols_vs_pcreg.csv", index=False)
        print(f"\nOLS vs PCReg_GCV comparison exported to pcr_beats_all_ols_vs_pcreg.csv")

    else:
        print("WARNING: No candidates found where PCReg_GCV beats both baselines.")
        print("Consider checking if PCReg_ConstrainOnly model ran in simulation.")

    # =============================================================================
    # LEGACY: Original Motivating Example (OLS vs PCReg with good coefficients)
    # =============================================================================
    print("\n" + "="*70)
    print("Legacy Motivating Example: PCReg vs OLS (original criteria)")
    print("="*70)
    pcreg_beats_ols_test_mape=df.pivot_table(index="seed", columns="model_name", values="test_mape").assign(
        pcreg_beats_ols_test_mape=lambda x: x['PCReg_GCV'] < x['OLS']
    ).query("pcreg_beats_ols_test_mape==True").reset_index()
    motivational_example_results = df.assign(pcreg_beats_ols_test_mape=lambda x: x.seed.isin(pcreg_beats_ols_test_mape.seed.unique())).sort_values('true_coefs_error').query(
        "pcreg_beats_ols_test_mape==1 and bad_ols_coefs==1 and model_name=='PCReg_GCV' and (.80<LC_est <=.999) and (.80<RC_est<=.999) and alpha>0 and r2>.7"
    ).reset_index()
    print(f"Found {len(motivational_example_results)} candidates for legacy motivational example")
    print(motivational_example_results[['seed','pcreg_beats_ols_test_mape', 'r2', 'test_mape', "learning_rate", "rate_effect", 'LC_est', 'RC_est', 'alpha']])
    motivational_example_results.to_csv(PARENT / "Output_v2" / "motivational_example_simulation_results.csv", index=False)
    example_seed = motivational_example_results.loc[2, 'seed']
    df_motivational = df_study_data.query("seed==@example_seed")

    # write the motivational example data to csv
    df_motivational.to_csv(PARENT / "Output_v2" / "motivational_example_data.csv", index=False)
    # %%
    print("Legacy Motivational example Results:", df.query("seed==@example_seed and model_name.isin(['OLS','PCReg_GCV'])").T)

    # %%
    print("Legacy motivational example study data:")
    print(df_study_data.query("seed==@example_seed"))

    # %%

    def DecisionTree(df, feature_columns=None):
        '''Create a decision tree to determine what model to use based on simulation parameters'''
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        feature_cols = ['bad_ols_coefs', 'T', 'b_sd', 'c_sd', 'T1_sd', 'LC_true', 'RC_true', 'sigma']
        X = df[feature_cols]


    print("Number of simulations where OLS produces bad coefficients but PCReg improves:",
    df.query("bad_ols_coefs==1 and pcreg_improves_bad_ols_coef==1").shape[0],
            "out of", df.query("bad_ols_coefs==1").shape[0],
            f"({df.query('bad_ols_coefs==1 and pcreg_improves_bad_ols_coef==1').shape[0]/df.query('bad_ols_coefs==1').shape[0]*100:.2f}%)"
    )
    print("Test MAPE gorupby bad ols coefs:")
    print(df.groupby(["bad_ols_coefs", "model_name"])['test_mape'].describe().sort_values(['bad_ols_coefs','mean']))
    print("Rank Test MAPE groupby bad ols coefs:")
    print(df.groupby(['bad_ols_coefs','model_name'])['rank_test_mape'].describe().sort_values(['bad_ols_coefs','mean']))
    print("Percentage beats OLS Test MAPE groupby bad ols coefs:")
    print(df.query("model_name!='OLS'").assign(beats_ols_test_mape = lambda x: x.beats_ols_test_mape.astype(int)).groupby(['bad_ols_coefs','model_name'])['beats_ols_test_mape'].describe().sort_values(['bad_ols_coefs','mean'], ascending=(True,False)))

    print("Number of times each model is ranked 1:")
    print((df.query("rank_test_mape==1").groupby("model_name").size()).sort_values(ascending=False))

    print("Number of times each model is ranked 1:")
    print((df.query("rank_test_mape==1").groupby(['bad_ols_coefs',"model_name"]).size()).sort_values(ascending=False))


    # percentage of time each model is ranked 1
    # need to add the ability to see when they tied for first place
    print("Percentage of time each model is ranked 1:")
    print((df.query("rank_test_mape==1").groupby(["bad_ols_coefs","model_name"]).size() / df.query('rank_test_mape.notna()')["seed"].nunique()).sort_values(ascending=False))

    print("Average Test MAPE by n_lots:")
    print(df.groupby(['n_lots','model_name'])['test_mape'].describe().sort_values(['n_lots','mean']))

    print("Average Test MAPE by correlation:")
    print(df.assign(actual_correlation=lambda x: np.round(x.actual_correlation,1)).groupby(['actual_correlation','model_name'])['test_mape'].describe().sort_values(['actual_correlation','mean']))

    # Visualization options for comparing model variation
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = ["test_mape", "test_mse", "test_sspe"]
    bad_ols_values = sorted(df["bad_ols_coefs"].unique())
    CLIP_PERCENTILE = 99

    # ============================================================
    # Option 1: Violin plots - shows full distribution shape + quartiles
    # ============================================================
    fig, axes = plt.subplots(len(metrics), len(bad_ols_values),
                             figsize=(10, 3.5 * len(metrics)),
                             sharex='row', sharey='row')

    for i, metric in enumerate(metrics):
        clip_val = df[metric].quantile(CLIP_PERCENTILE / 100)
        df[f"{metric}_clipped"] = df[metric].clip(upper=clip_val)

        for j, bad_ols in enumerate(bad_ols_values):
            ax = axes[i, j]
            subset = df.query("bad_ols_coefs == @bad_ols")

            sns.violinplot(data=subset, x="model_name", y=f"{metric}_clipped",
                           ax=ax, cut=0, inner="quartile", palette="Set2")

            if i == 0:
                ax.set_title(f"bad_ols_coefs={bad_ols}", fontsize=11, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel(metric if j == 0 else "")

    fig.suptitle("Violin Plots: Distribution of Test Metrics by Model", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(PARENT / "Output_v2" / "violin_plots.png", dpi=150, bbox_inches='tight')
    #plt.show()

    # ============================================================
    # Option 2: ECDF (Cumulative Distribution) - shows % below threshold
    # ============================================================
    fig, axes = plt.subplots(len(metrics), len(bad_ols_values),
                             figsize=(10, 3.5 * len(metrics)),
                             sharex='row')

    for i, metric in enumerate(metrics):
        clip_val = df[metric].quantile(CLIP_PERCENTILE / 100)

        for j, bad_ols in enumerate(bad_ols_values):
            ax = axes[i, j]
            subset = df.query("bad_ols_coefs == @bad_ols")

            for model in df["model_name"].unique():
                model_data = subset.query("model_name == @model")[metric].clip(upper=clip_val)
                sns.ecdfplot(data=model_data, ax=ax, label=model, linewidth=2)

            if i == 0:
                ax.set_title(f"bad_ols_coefs={bad_ols}", fontsize=11, fontweight='bold')
            ax.set_xlabel(metric if i == len(metrics) - 1 else "")
            ax.set_ylabel("Cumulative %" if j == 0 else "")
            if i == 0 and j == 1:
                ax.legend(loc='lower right', fontsize=9)

    fig.suptitle("ECDF: Cumulative Distribution of Test Metrics", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(PARENT / "Output_v2" / "ecdf_plots.png", dpi=150, bbox_inches='tight')
    #plt.show()

    # ============================================================
    # Option 3: KDE overlay - smooth density comparison
    # ============================================================
    fig, axes = plt.subplots(len(metrics), len(bad_ols_values),
                             figsize=(10, 3.5 * len(metrics)),
                             sharex='row')

    for i, metric in enumerate(metrics):
        clip_val = df[metric].quantile(CLIP_PERCENTILE / 100)

        for j, bad_ols in enumerate(bad_ols_values):
            ax = axes[i, j]
            subset = df.query("bad_ols_coefs == @bad_ols")

            for model in ["OLS", "PCReg_GCV"]:
                model_data = subset.query("model_name == @model")[metric].clip(upper=clip_val)
                sns.kdeplot(data=model_data, ax=ax, label=model, linewidth=2, fill=True, alpha=0.3)

            if i == 0:
                ax.set_title(f"bad_ols_coefs={bad_ols}", fontsize=11, fontweight='bold')
            ax.set_xlabel(metric if i == len(metrics) - 1 else "")
            ax.set_ylabel("Density" if j == 0 else "")
            if i == 0 and j == 1:
                ax.legend(loc='upper right', fontsize=9)

    fig.suptitle("KDE: Density of Test Metrics by Model", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(PARENT / "Output_v2" / "kde_plots.png", dpi=150, bbox_inches='tight')
    #plt.show()

    # ============================================================
    # Bias Analysis
    # ============================================================
    print("\n" + "="*70)
    print("BIAS ANALYSIS: Estimated - True")
    print("="*70)

    print("\nBias Summary by Model (mean bias across all simulations):")
    bias_summary = df.groupby('model_name')[['b_bias', 'c_bias', 'T1_bias']].agg(['mean', 'std', 'median'])
    print(bias_summary.round(4))

    print("\nBias Summary by Model and Bad OLS Coefficients:")
    bias_by_bad_ols = df.groupby(['bad_ols_coefs', 'model_name'])[['b_bias', 'c_bias', 'T1_bias']].mean()
    print(bias_by_bad_ols.round(4))

    # Export bias summary
    bias_summary.to_csv(PARENT / "Output_v2" / "bias_summary_by_model.csv")
    bias_by_bad_ols.to_csv(PARENT / "Output_v2" / "bias_summary_by_model_and_bad_ols.csv")
    print(f"\nBias summaries exported to Output_v2/")
