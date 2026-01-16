"""
Simulation Data Generation Module
=================================

Generates simulation study data using SAR procurement quantities.
Uses pipe operations and groupby transforms for clean, pythonic code.

Usage
-----
>>> from simulation_data import load_or_generate_simulation_data, get_scenario_data
>>> df = load_or_generate_simulation_data()
>>> lot_data = get_scenario_data(scenario_id=1, df=df)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import json
import hashlib
from datetime import datetime
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    'sample_sizes': [5, 10, 30],
    'cv_errors': [0.01, 0.1, 0.2],
    'learning_rates': [0.85, 0.90, 0.95],
    'rate_effects': [0.80, 0.85, 0.90],
    'n_replications': 25,
    'T1': 100,
    'base_seed': 42,
}

_THIS_DIR = Path(__file__).parent
_DEFAULT_SAR_PATH = _THIS_DIR / "sar_raw_db" / "msar_annual_quantities.csv"
_DEFAULT_OUTPUT_DIR = _THIS_DIR / "output_v2" 


# ============================================================================
# CORE UTILITIES
# ============================================================================
def learning_rate_to_slope(lr):
    """Convert learning rate to slope exponent: b = ln(lr)/ln(2)."""
    return np.log(lr) / np.log(2)


def calculate_lot_midpoint(first, last, b):
    """Asher's lot midpoint formula (vectorized)."""
    n = last - first + 1
    num = ((last + 0.5) ** (1 + b)) - ((first - 0.5) ** (1 + b))
    return (num / ((1 + b) * n)) ** (1 / b)


# ============================================================================
# CACHING
# ============================================================================
def compute_config_hash(config: dict) -> str:
    """SHA256 hash of config for cache invalidation."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def load_manifest(path: Path) -> Optional[dict]:
    """Load manifest if exists."""
    return json.load(open(path)) if path.exists() else None


def save_manifest(path: Path, config: dict, hash: str, stats: dict):
    """Save manifest with config and stats."""
    json.dump({'hash': hash, 'config': config, 'created_at': datetime.now().isoformat(), **stats},
              open(path, 'w'), indent=2, default=str)


# ============================================================================
# SAR DATA
# ============================================================================
def load_sar_data(sar_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess SAR procurement quantities."""
    sar_path = Path(sar_path or _DEFAULT_SAR_PATH)
    if not sar_path.exists():
        raise FileNotFoundError(f"SAR data not found: {sar_path}")

    return (
        pd.read_csv(sar_path)
        .assign(
            fiscal_year=lambda d: pd.to_numeric(d['fiscal_year'], errors='coerce'),
            total=lambda d: pd.to_numeric(d['total'], errors='coerce')
        )
        .query('fiscal_year > 0 and total.notna() and total > 0')
        .pipe(lambda d: d[d['appn_account'].str.lower().str.contains('procurement', na=False)])
        .groupby(['program', 'fiscal_year'], as_index=False)
        .agg({'total': 'sum'})
        .sort_values(['program', 'fiscal_year'])
        .assign(program_id=lambda d: d.groupby('program').ngroup() + 1)
        .drop(columns=['program'])
        .reset_index(drop=True)
    )


def select_program_data(sar_df: pd.DataFrame, n_lots: int, seed: int) -> dict:
    """
    Select SAR program and split into train/test preserving temporal order.

    Parameters
    ----------
    sar_df : pd.DataFrame
        SAR data with program_id and total columns
    n_lots : int
        Number of training lots needed
    seed : int
        Random seed for program selection

    Returns
    -------
    dict with keys:
        'program_id': int
        'train_quantities': array of first n_lots quantities (chronological)
        'test_quantities': array of remaining quantities (may be empty)
        'total_lots': total program lots
    """
    rng = np.random.RandomState(seed)
    counts = sar_df.groupby('program_id').size()
    eligible = counts[counts >= n_lots]
    if len(eligible) == 0:
        eligible = counts[counts >= max(3, min(n_lots, 5))]
    if len(eligible) == 0:
        raise ValueError(f"No SAR programs with sufficient data for {n_lots} lots")

    program_id = eligible.index[rng.randint(len(eligible))]
    quantities = sar_df[sar_df['program_id'] == program_id]['total'].values
    quantities = np.maximum(1, np.round(quantities).astype(int))

    return {
        'program_id': program_id,
        'train_quantities': quantities[:n_lots],
        'test_quantities': quantities[n_lots:],  # May be empty
        'total_lots': len(quantities),
    }


# ============================================================================
# PIPE FUNCTIONS
# ============================================================================
def expand_to_lots(scenarios_df: pd.DataFrame, sar_df: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    """Expand each scenario to lots with train/test split preserving temporal order."""
    rows = []
    for _, row in scenarios_df.iterrows():
        seed = base_seed + hash((row['n_lots'], row['cv_error'], row['learning_rate'],
                                  row['rate_effect'], row['replication'])) % (2**31)
        data = select_program_data(sar_df, int(row['n_lots']), seed)

        # Training lots (first n_lots chronologically)
        for lot_num, qty in enumerate(data['train_quantities'], 1):
            rows.append({
                **row.to_dict(),
                'lot_type': 'train',
                'lot_number': lot_num,
                'lot_quantity': int(qty),
                'program_id': data['program_id'],
                'total_program_lots': data['total_lots'],
                'seed': seed,
            })

        # Test lots (remaining lots, may be empty)
        first_test = len(data['train_quantities']) + 1
        for lot_num, qty in enumerate(data['test_quantities'], first_test):
            rows.append({
                **row.to_dict(),
                'lot_type': 'test',
                'lot_number': lot_num,
                'lot_quantity': int(qty),
                'program_id': data['program_id'],
                'total_program_lots': data['total_lots'],
                'seed': seed,
            })

    return pd.DataFrame(rows)


def add_lot_calculations(df: pd.DataFrame, T1: float) -> pd.DataFrame:
    """Add lot calculations using groupby transform for cumsum."""
    return (
        df
        .assign(
            b_true=lambda d: learning_rate_to_slope(d['learning_rate']),
            c_true=lambda d: learning_rate_to_slope(d['rate_effect']),
            T1_true=T1,
            last_unit=lambda d: d.groupby('scenario_id')['lot_quantity'].transform('cumsum'),
            first_unit=lambda d: d['last_unit'] - d['lot_quantity'] + 1,
        )
        .assign(
            lot_midpoint=lambda d: calculate_lot_midpoint(d['first_unit'], d['last_unit'], d['b_true']),
            log_midpoint=lambda d: np.log(d['lot_midpoint']),
            log_quantity=lambda d: np.log(d['lot_quantity']),
        )
    )


def add_costs_and_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Add costs with lognormal error and per-scenario correlation (training only)."""
    # True cost
    df = df.assign(
        true_cost=lambda d: d['T1_true'] * (d['lot_midpoint'] ** d['b_true']) * (d['lot_quantity'] ** d['c_true'])
    )

    # Lognormal error per scenario (using seed for reproducibility)
    observed_costs = []
    for _, group in df.groupby('scenario_id'):
        rng = np.random.RandomState(int(group['seed'].iloc[0]))
        cv = group['cv_error'].iloc[0]
        if cv > 0:
            #sigma = np.sqrt(np.log(1 + cv**2))
            #errors = rng.normal(1, cv, len(group))
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = -sigma**2 / 2  # This ensures E[exp(X)] = 1
            errors = np.exp(rng.normal(mu, sigma, len(group)))

        else:
            errors = np.ones(len(group))
        observed_costs.extend(group['true_cost'].values * errors)

    df['observed_cost'] = observed_costs
    df['log_observed_cost'] = np.log(df['observed_cost'])

    # Correlation per scenario (training data only)
    train_df = df[df['lot_type'] == 'train']
    corr = train_df.groupby('scenario_id')[['log_midpoint', 'log_quantity']].apply(
        lambda g: g['log_midpoint'].corr(g['log_quantity'])
    )
    df['actual_correlation'] = df['scenario_id'].map(corr)

    return df


# ============================================================================
# MAIN GENERATION
# ============================================================================
def generate_simulation_data(config: dict, verbose: bool = True) -> pd.DataFrame:
    """Generate full simulation data using pipe operations."""
    if verbose:
        print("Loading SAR data...")
    sar_df = load_sar_data()

    # Build scenario grid
    scenarios = (
        pd.DataFrame(
            product(config['sample_sizes'], config['cv_errors'],
                    config['learning_rates'], config['rate_effects'],
                    range(1, config['n_replications'] + 1)),
            columns=['n_lots', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
        )
        .assign(scenario_id=lambda d: range(1, len(d) + 1))
    )

    if verbose:
        print(f"Generating {len(scenarios)} scenarios...")

    # Pipe chain: expand → calculate → costs
    df = (
        scenarios
        .pipe(expand_to_lots, sar_df=sar_df, base_seed=config['base_seed'])
        .pipe(add_lot_calculations, T1=config['T1'])
        .pipe(add_costs_and_correlation)
    )

    # Reorder columns
    col_order = [
        'scenario_id', 'n_lots', 'learning_rate', 'rate_effect', 'cv_error',
        'replication', 'seed', 'program_id', 'total_program_lots', 'actual_correlation',
        'b_true', 'c_true', 'T1_true',
        'lot_type', 'lot_number', 'first_unit', 'last_unit', 'lot_quantity', 'lot_midpoint',
        'log_midpoint', 'log_quantity',
        'true_cost', 'observed_cost', 'log_observed_cost'
    ]
    df = df[col_order]

    if verbose:
        print(f"Generated {len(df)} rows across {len(scenarios)} scenarios")

    return df


def load_or_generate_simulation_data(
    output_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    force_regenerate: bool = False,
    verbose: bool = True,
    **config_overrides
) -> pd.DataFrame:
    """Load cached simulation data or generate new."""
    output_path = Path(output_path or _DEFAULT_OUTPUT_DIR / "simulation_study_data.parquet")
    manifest_path = Path(manifest_path or _DEFAULT_OUTPUT_DIR / "simulation_study_data_manifest.json")

    config = {**DEFAULT_CONFIG, **config_overrides}
    config_hash = compute_config_hash(config)

    # Check cache
    if not force_regenerate and output_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest and manifest.get('hash') == config_hash:
            if verbose:
                print(f"Loading cached data from {output_path}")
            return pd.read_parquet(output_path)
        elif verbose:
            print("Config changed, regenerating...")

    # Generate
    df = generate_simulation_data(config, verbose=verbose)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_manifest(manifest_path, config, config_hash,
                  {'n_scenarios': df['scenario_id'].nunique(), 'n_rows': len(df)})

    if verbose:
        print(f"Saved to {output_path}")

    return df


# ============================================================================
# SCENARIO ACCESS (for run_simulation.py)
# ============================================================================
def get_scenario_data(scenario_id: int, df: pd.DataFrame = None) -> pd.DataFrame:
    """Get training lot_data DataFrame for a specific scenario."""
    if df is None:
        df = load_or_generate_simulation_data()
    return df[(df['scenario_id'] == scenario_id) & (df['lot_type'] == 'train')].copy()


def get_test_data(scenario_id: int, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Get test lot_data DataFrame for a specific scenario.

    Returns empty DataFrame if no test lots available.
    """
    if df is None:
        df = load_or_generate_simulation_data()
    return df[(df['scenario_id'] == scenario_id) & (df['lot_type'] == 'test')].copy()


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulation data")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--replications", type=int, help="Override n_replications")
    args = parser.parse_args()

    kwargs = {'force_regenerate': args.force}
    if args.replications:
        kwargs['n_replications'] = args.replications

    df = load_or_generate_simulation_data(**kwargs)

    print(f"\nTotal rows: {len(df)}, Scenarios: {df['scenario_id'].nunique()}")
    print(f"Correlation range: [{df['actual_correlation'].min():.3f}, {df['actual_correlation'].max():.3f}]")
    print(f"\nSample:\n{df.head(10).to_string(index=False)}")
