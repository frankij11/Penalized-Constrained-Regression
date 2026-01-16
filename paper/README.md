# ICEAA 2026 Research Paper

This directory contains the Quarto-based research paper for "Penalized-Constrained Regression for Learning Curve Estimation".

## Prerequisites

1. **Install Quarto**: Download from https://quarto.org/docs/get-started/
2. **Install Python dependencies**:
   ```bash
   pip install -e ".[paper]"
   ```
3. **Install TinyTeX for PDF output** (optional):
   ```bash
   quarto install tinytex
   ```

## Workflow

### Step 1: Run Simulation (if needed)

The simulation takes several hours to run. Only run if you need to regenerate results:

```bash
cd scripts/ICEAA
python run_simulation.py
```

Results are cached in `scripts/ICEAA/output_v2/`.

### Step 2: Render Paper

```bash
# Render to all formats (HTML, PDF, Word)
quarto render paper/

# Or render specific format
quarto render paper/ --to html
quarto render paper/ --to pdf
quarto render paper/ --to docx

# Preview with live reload
quarto preview paper/

# Force re-run all code (ignore freeze cache)
quarto render paper/ --no-freeze
```

Output files are saved to `paper/_output/`.

## Directory Structure

```
paper/
├── _quarto.yml          # Project configuration
├── paper.qmd            # Main document (includes sections)
├── sections/
│   ├── 01-introduction.qmd
│   ├── 02-methodology.qmd
│   ├── 03-simulation-design.qmd
│   ├── 04-results.qmd       # Dynamic - loads parquet data
│   ├── 05-doe-analysis.qmd  # Dynamic - statistical tests
│   └── 06-discussion.qmd
├── references.bib       # Bibliography
├── figures/             # Output directory for figures
└── _freeze/             # Cached computations (auto-generated)
```

## Freeze Feature

The `_quarto.yml` is configured with `freeze: auto`:

- **First render**: Executes all Python code and caches results
- **Subsequent renders**: Uses cached outputs unless source code changes
- **Fast iteration**: Text-only edits render instantly

To force a fresh run: `quarto render paper/ --no-freeze`

## Adding Citations

Add BibTeX entries to `references.bib`, then cite in text with `@key`:

```markdown
As shown by @wright1936factors, learning curves follow a power law.
```

## Cross-References

Reference figures and tables with `@fig-label` and `@tbl-label`:

```markdown
@fig-overall shows the overall performance comparison.
```
