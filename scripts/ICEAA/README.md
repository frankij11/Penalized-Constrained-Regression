# ICEAA 2026: Penalized-Constrained Regression Paper

This folder contains the analysis and paper for the ICEAA 2026 Professional Development & Training Workshop.

**Paper:** *Penalized-Constrained Regression: Combining Regularization and Domain Constraints for Cost Estimation*

**Authors:** Kevin Joy, Max Watstein (Herren Associates)

## Quick Start

```bash
# Check if data files exist
python master_pipeline.py --check

# Full reproduction (~1-2 hours, runs simulation)
python master_pipeline.py --full

# Render paper to Word (requires existing simulation data)
python master_pipeline.py --quick

# Render paper only
quarto render penalized_constrained_regression.qmd --to docx
```

## Folder Structure

```
ICEAA/
├── penalized_constrained_regression.qmd   # THE PAPER (Quarto document)
├── references.bib                          # BibTeX citations
├── master_pipeline.py                      # Single-command reproduction
│
├── run_simulation.py                       # Monte Carlo simulation (~50 min)
├── simulation_data.py                      # Data generation utilities
├── simulation_analysis.py                  # Post-simulation analysis
│
├── ICEAA2026_Paper_DRAFT.md               # Full paper draft (reference)
├── SIMULATION_FINDINGS.md                  # Key results summary
│
├── output_v2/                             # Generated data and figures
│   ├── simulation_results.parquet         # Main simulation results
│   ├── simulation_study_data.parquet      # Scenario data
│   └── predictions_flat.parquet           # Model predictions
│
├── analysis/                              # Helper modules
│   ├── load_results.py                    # Data loading utilities
│   └── visualization.py                   # Plotting utilities
│
├── notebooks/                             # Interactive examples
│   ├── 01_simple_illustration.ipynb       # Simple concept demo
│   └── 04_motivating_example.ipynb        # Detailed motivating example
│
└── archive/                               # Deprecated/old files
```

## Reproduction Steps

### 1. Install Dependencies

```bash
pip install penalized-constrained
pip install plotly pandas numpy scikit-learn
pip install quarto  # For rendering paper
```

### 2. Run Simulation (Optional, ~50 minutes)

```bash
python run_simulation.py
```

This generates 8,100 scenarios (81 factor combinations × 100 replications).

### 3. Run Analysis

```bash
python simulation_analysis.py
```

### 4. Render Paper

```bash
quarto render penalized_constrained_regression.qmd --to docx
```

## Key Findings

- **PCReg outperforms OLS in 58.2% of scenarios** overall
- **Strongest advantage when OLS produces unreasonable coefficients** (learning rate outside 70-100%)
- **Constraints alone often sufficient** - PCReg with α=0 beats PCReg-CV 62.1% of the time
- **Best for small samples** - advantage increases with n_lots ≤ 10

## Software

The `penalized-constrained` Python package was developed specifically for the cost estimating community:

```bash
pip install penalized-constrained
```

```python
import penalized_constrained as pcreg

model = pcreg.PenalizedConstrainedCV(
    bounds={'T1': (0, None), 'b': (-0.5, 0), 'c': (-0.5, 0)},
    selection='gcv',
    loss='sspe'
)
model.fit(X, y)
```

## Citation

```bibtex
@inproceedings{joy2026pcreg,
  title={Penalized-Constrained Regression: Combining Regularization and Domain Constraints for Cost Estimation},
  author={Joy, Kevin and Watstein, Max},
  booktitle={ICEAA Professional Development \& Training Workshop},
  year={2026}
}
```
