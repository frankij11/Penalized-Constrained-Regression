"""
Utility script to create sample lot data from SAR annual quantities.

Reads SAR data, calculates lot information using learning curve parameters,
and computes rolling correlations between midpoint and lot quantity.

This is a simplified demo script. For full simulation data generation,
see scripts/ICEAA/simulation_data.py
"""

import sys
from pathlib import Path

# Add ICEAA scripts to path for simulation_data import
sys.path.insert(0, str(Path(__file__).parent.parent / "ICEAA"))

import pandas as pd
import numpy as np
from simulation_data import load_sar_data, create_lot_data


def calculate_cumulative_correlation(df, min_lots=5):
    """
    Calculate cumulative correlation between lot_midpoint and lot_quantity.

    For each row, calculates the correlation using all lots up to and including
    that row, starting when there are at least min_lots of data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'lot_midpoint' and 'lot_quantity' columns.
    min_lots : int, optional
        Minimum number of lots required before calculating correlation (default 5).

    Returns
    -------
    pd.Series
        Cumulative correlations, with NaN for rows before min_lots threshold.
    """
    correlations = []

    for i in range(len(df)):
        if i + 1 < min_lots:
            correlations.append(np.nan)
        else:
            subset = df.iloc[:i + 1]
            corr = subset['lot_midpoint'].corr(subset['lot_quantity'])
            correlations.append(corr)

    return pd.Series(correlations, index=df.index)


def process_sar_data(output_path, lc_rate=0.95, rc_rate=0.95, T1=1000,
                     cv_error=0.1, min_lots=5, random_seed=42):
    """
    Process SAR quantity data and create sample dataset with lot information.

    Parameters
    ----------
    output_path : str
        Path to save the output CSV file.
    lc_rate : float, optional
        Learning curve rate (default 0.95 for 95% learning curve).
    rc_rate : float, optional
        Rate curve rate (default 0.95 for 95% rate curve).
    T1 : float, optional
        Theoretical first unit cost (default 1000).
    cv_error : float, optional
        Coefficient of variation for random error (default 0.1).
    min_lots : int, optional
        Minimum lots before calculating correlation (default 5).
    random_seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with lot information and correlations.
    """
    np.random.seed(random_seed)

    # Load preprocessed SAR data using simulation_data module
    sar_df = load_sar_data()

    # Process each program separately
    all_results = []

    for program_id in sar_df['program_id'].unique():
        program_data = sar_df[sar_df['program_id'] == program_id].copy()

        # Skip programs with insufficient data
        if len(program_data) < min_lots:
            continue

        # Get quantities from SAR data
        quantities = program_data['total'].values.astype(int)

        # Calculate lot data using simulation_data module
        lot_data = create_lot_data(
            quantities=quantities,
            learning_rate=lc_rate,
            rate_effect=rc_rate,
            T1=T1,
            cv_error=cv_error,
            seed=random_seed + program_id
        )

        # Calculate cumulative correlation
        lot_data['correlation'] = calculate_cumulative_correlation(lot_data, min_lots=min_lots)

        # Add program_id
        lot_data['program'] = program_id

        all_results.append(lot_data)

    # Combine all programs
    result_df = pd.concat(all_results, ignore_index=True)

    # Select and order final columns
    output_columns = [
        'program',
        'lot_number',
        'lot_quantity',
        'first_unit',
        'last_unit',
        'lot_midpoint',
        'correlation'
    ]

    result_df = result_df[output_columns]

    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Sample data saved to: {output_path}")
    print(f"Total programs: {result_df['program'].nunique()}")
    print(f"Total lots: {len(result_df)}")

    return result_df


if __name__ == "__main__":
    # Default paths
    output_file = "sample_data.csv"

    # Process data with 95% learning curve
    df = process_sar_data(
        output_path=output_file,
        lc_rate=0.95,
        rc_rate=0.95,
        T1=1000,
        cv_error=0.1,
        min_lots=5,
        random_seed=42
    )

    # Display summary
    print("\nSample of output data:")
    print(df.head(20).to_string(index=False))

    print("\nCorrelation statistics by program:")
    print(df.groupby('lot_number')['correlation'].describe())
