#!/usr/bin/env python
"""
Master Pipeline for ICEAA Analysis
===================================

Reproduces the full analysis for the penalized-constrained regression paper.

Usage:
    python master_pipeline.py --full     # Run everything (~1-2 hours)
    python master_pipeline.py --quick    # Skip simulation, use cached data
    python master_pipeline.py --check    # Just verify data exists
    python master_pipeline.py --render   # Render Quarto paper to Word/HTML
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output_v2"

REQUIRED_FILES = [
    'simulation_results.parquet',
    'simulation_study_data.parquet',
    'simulation_config.json',
]


def check_data():
    """Check if required data files exist."""
    print("Checking required data files...")
    missing = []
    for f in REQUIRED_FILES:
        path = OUTPUT_DIR / f
        if not path.exists():
            missing.append(f)
            print(f"  [X] {f} - MISSING")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
    return missing


def run_simulation():
    """Run Monte Carlo simulation (~50 min on 16 cores)."""
    print("\n" + "=" * 60)
    print("Running simulation (this may take 50+ minutes)...")
    print("=" * 60 + "\n")
    subprocess.run([sys.executable, 'run_simulation.py'], cwd=SCRIPT_DIR, check=True)


def run_analysis():
    """Run post-simulation analysis."""
    print("\n" + "=" * 60)
    print("Running analysis scripts...")
    print("=" * 60 + "\n")
    subprocess.run([sys.executable, 'simulation_analysis.py'], cwd=SCRIPT_DIR, check=True)


def render_paper(format='docx'):
    """Render Quarto paper to specified format."""
    print(f"\n" + "=" * 60)
    print(f"Rendering paper to {format}...")
    print("=" * 60 + "\n")

    qmd_file = SCRIPT_DIR / 'penalized_constrained_regression.qmd'
    if not qmd_file.exists():
        print(f"[X] Quarto file not found: {qmd_file}")
        print("  Create the .qmd file first before rendering.")
        return False

    subprocess.run([
        'quarto', 'render', str(qmd_file),
        '--to', format
    ], cwd=SCRIPT_DIR, check=True)

    output_file = qmd_file.with_suffix(f'.{format}')
    print(f"\n[OK] Paper rendered: {output_file.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='ICEAA Analysis Pipeline - Reproduces full paper analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python master_pipeline.py --check    # Check if data files exist
    python master_pipeline.py --full     # Run everything from scratch
    python master_pipeline.py --quick    # Use cached data, render paper
    python master_pipeline.py --render   # Just render the paper
        """
    )
    parser.add_argument('--full', action='store_true',
                        help='Run complete pipeline (simulation + analysis + render)')
    parser.add_argument('--quick', action='store_true',
                        help='Use cached data, render paper')
    parser.add_argument('--check', action='store_true',
                        help='Check if data files exist')
    parser.add_argument('--render', action='store_true',
                        help='Render paper only')
    parser.add_argument('--format', default='docx',
                        choices=['docx', 'html', 'pdf'],
                        help='Output format for paper (default: docx)')
    args = parser.parse_args()

    # Default to --check if no arguments provided
    if not any([args.full, args.quick, args.check, args.render]):
        args.check = True

    print("\n" + "=" * 60)
    print("ICEAA Paper Pipeline")
    print("=" * 60)

    if args.check:
        missing = check_data()
        if missing:
            print(f"\n[X] Missing files: {missing}")
            print("  Run with --full to generate.")
        else:
            print("\n[OK] All data files present!")
        return

    if args.full:
        run_simulation()
        run_analysis()
        render_paper(args.format)
        print("\n" + "=" * 60)
        print("[OK] Full pipeline complete!")
        print("=" * 60)

    if args.quick:
        missing = check_data()
        if missing:
            print(f"\n[X] Missing: {missing}")
            print("  Run --full first to generate simulation data.")
            return
        render_paper(args.format)
        print("\n[OK] Quick render complete!")

    if args.render:
        render_paper(args.format)


if __name__ == "__main__":
    main()
