'''
Generate a single HTML report with all simulation analysis results.

This script reads simulation results and produces a comprehensive HTML report
with all plots embedded inline (no external files needed).

Includes:
- Overall model performance comparison
- Analysis by design factors (DOE)
- Statistical tests (RM-ANOVA, pairwise comparisons)
- Win rate analysis
- Recommendations for when to use each method
'''

import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
from scipy import stats
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

# Try importing optional packages for DOE analysis
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

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


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_boxplot_by_category(df, category, category_values, model_order):
    """Create boxplot for a specific category."""
    fig, axes = plt.subplots(1, len(category_values), figsize=(5 * len(category_values), 6),
                              sharey=True)
    if len(category_values) == 1:
        axes = [axes]

    for ax, val in zip(axes, category_values):
        subset = df[df[category] == val]
        data_to_plot = [subset[subset['model_name'] == m][METRIC].dropna().values
                        for m in model_order]
        bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

        colors = ['lightgreen' if 'PCReg' in m else 'lightblue' for m in model_order]
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
    return fig


def create_overall_boxplot(df, model_order):
    """Create overall boxplot comparing all models."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [df[df['model_name'] == m][METRIC].dropna().values for m in model_order]
    bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True)

    colors = ['lightgreen' if 'PCReg' in m else 'lightblue' for m in model_order]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    bp['boxes'][0].set_edgecolor('red')
    bp['boxes'][0].set_linewidth(2)

    ax.set_ylabel(METRIC_LABEL)
    ax.set_xlabel('Model')
    ax.set_xticklabels(model_order, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Overall Model Performance (Test SSPE)', fontsize=14, fontweight='bold')

    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Penalized-Constrained'),
        Patch(facecolor='lightblue', edgecolor='black', label='Standard Methods'),
        Patch(facecolor='white', edgecolor='red', linewidth=2, label='Winner')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    return fig


def create_heatmap_figure(df, factor1, factor2, model='PCReg_CV', baseline='OLS'):
    """Create heatmap showing model improvement over baseline."""
    scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
    df_wide = df.pivot_table(
        index=scenario_cols,
        columns='model_name',
        values=METRIC
    ).reset_index()

    if model not in df_wide.columns or baseline not in df_wide.columns:
        return None

    df_wide['improvement'] = (df_wide[baseline] - df_wide[model]) / df_wide[baseline] * 100
    heatmap_data = df_wide.groupby([factor1, factor2])['improvement'].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': f'% Improvement\n(positive = {model} better)'},
        ax=ax
    )
    ax.set_xlabel(CATEGORY_LABELS.get(factor2, factor2))
    ax.set_ylabel(CATEGORY_LABELS.get(factor1, factor1) if factor1 in CATEGORY_LABELS else factor1)
    ax.set_title(f'{model} Improvement over {baseline}')
    plt.tight_layout()
    return fig


def compute_doe_statistics(df):
    """Compute DOE statistical tests."""
    scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
    df['scenario_id'] = df.groupby(scenario_cols).ngroup()

    DOE_MODELS = ['OLS', 'OLS_LearnOnly', 'PCReg_ConstrainOnly', 'PCReg_CV', 'PCReg_CV_Tight', 'PCReg_CV_Wrong']
    df_doe = df[df['model_name'].isin(DOE_MODELS)].copy()

    results = {'has_pingouin': HAS_PINGOUIN}

    if HAS_PINGOUIN:
        # Repeated measures ANOVA
        rm_aov = pg.rm_anova(
            data=df_doe,
            dv=METRIC,
            within='model_name',
            subject='scenario_id',
            correction=True
        )
        results['rm_anova'] = rm_aov

        # Pairwise comparisons
        pairwise = pg.pairwise_tests(
            data=df_doe,
            dv=METRIC,
            within='model_name',
            subject='scenario_id',
            padjust='holm',
            effsize='hedges'
        )
        results['pairwise'] = pairwise

    # Win rate analysis
    df_wide = df_doe.pivot_table(
        index=scenario_cols,
        columns='model_name',
        values=METRIC
    ).reset_index()

    if 'PCReg_CV' in df_wide.columns and 'OLS' in df_wide.columns:
        df_wide['pcreg_wins'] = df_wide['PCReg_CV'] < df_wide['OLS']
        overall_wins = df_wide['pcreg_wins'].sum()
        overall_total = len(df_wide)
        binom_result = stats.binomtest(overall_wins, overall_total, p=0.5, alternative='greater')

        results['overall_win_rate'] = overall_wins / overall_total
        results['overall_wins'] = overall_wins
        results['overall_total'] = overall_total
        results['overall_pvalue'] = binom_result.pvalue

        # Win rates by factor
        win_rates = []
        for factor in CATEGORIES + ['n_lots']:
            for level in sorted(df_wide[factor].unique()):
                mask = df_wide[factor] == level
                level_wins = df_wide.loc[mask, 'pcreg_wins'].sum()
                level_total = mask.sum()
                binom = stats.binomtest(level_wins, level_total, p=0.5, alternative='greater')
                win_rates.append({
                    'factor': factor,
                    'level': level,
                    'win_rate': level_wins / level_total,
                    'p_value': binom.pvalue,
                    'significant': binom.pvalue < 0.05
                })
        results['win_rates'] = pd.DataFrame(win_rates)

    return results


def generate_html_report(df, results_dir):
    """Generate complete HTML report with all analysis."""

    # Overall statistics
    overall_stats = df.groupby('model_name')[METRIC].agg(['mean', 'std', 'median', 'count'])
    overall_stats = overall_stats.sort_values('mean')
    model_order = overall_stats.index.tolist()
    overall_winner = overall_stats.index[0]

    # Win counts
    scenario_cols = ['n_lots', 'target_correlation', 'cv_error', 'learning_rate', 'rate_effect', 'replication']
    win_counts = {}
    for name, group in df.groupby(scenario_cols):
        if len(group) > 0:
            group_valid = group.dropna(subset=[METRIC])
            if len(group_valid) > 0:
                winner = group_valid.loc[group_valid[METRIC].idxmin(), 'model_name']
                win_counts[winner] = win_counts.get(winner, 0) + 1
    total_scenarios = sum(win_counts.values())

    # DOE statistics
    doe_stats = compute_doe_statistics(df)

    # Start building HTML
    html_parts = []
    html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <title>Simulation Study Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 40px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .winner {{ background-color: #d4edda !important; font-weight: bold; }}
        .metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .highlight {{ color: #27ae60; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h4 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .toc {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .toc ul {{ columns: 2; }}
        .toc a {{ color: #3498db; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Simulation Study Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Data Source:</strong> {results_dir / "simulation_results.parquet"}</p>
    <p><strong>Total Observations:</strong> {len(df):,} (converged models only)</p>

    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Executive Summary</a></li>
            <li><a href="#overall">Overall Model Performance</a></li>
            <li><a href="#doe">DOE Statistical Analysis</a></li>
            <li><a href="#recommendations">Recommendations</a></li>
            <li><a href="#cv_error">Analysis by CV Error</a></li>
            <li><a href="#target_correlation">Analysis by Correlation</a></li>
            <li><a href="#learning_rate">Analysis by Learning Rate</a></li>
            <li><a href="#rate_effect">Analysis by Rate Effect</a></li>
            <li><a href="#wins">Winner Frequency Analysis</a></li>
            <li><a href="#heatmaps">Interaction Heatmaps</a></li>
        </ul>
    </div>

    <h2 id="summary">Executive Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h4>Overall Winner</h4>
            <div class="value">{overall_winner}</div>
        </div>
        <div class="summary-card">
            <h4>Best Mean SSPE</h4>
            <div class="value">{overall_stats.loc[overall_winner, "mean"]:.4f}</div>
        </div>
        <div class="summary-card">
            <h4>Total Scenarios</h4>
            <div class="value">{total_scenarios:,}</div>
        </div>
        <div class="summary-card">
            <h4>Models Compared</h4>
            <div class="value">{len(model_order)}</div>
        </div>
    </div>
''')

    # Overall Performance Section
    html_parts.append('''
    <h2 id="overall">Overall Model Performance</h2>
    <p>Models ranked by mean Test SSPE (lower is better). The winner is highlighted in green.</p>
''')

    # Create and embed overall boxplot
    fig_overall = create_overall_boxplot(df, model_order)
    img_overall = fig_to_base64(fig_overall)
    html_parts.append(f'<img src="data:image/png;base64,{img_overall}" alt="Overall Boxplot">')

    # Overall stats table
    html_parts.append('<h3>Descriptive Statistics</h3>')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Rank</th><th>Model</th><th>Mean SSPE</th><th>Std SSPE</th><th>Median SSPE</th><th>Count</th><th>Win Count</th><th>Win %</th></tr>')
    for rank, (model, row) in enumerate(overall_stats.iterrows(), 1):
        wins = win_counts.get(model, 0)
        win_pct = 100 * wins / total_scenarios if total_scenarios > 0 else 0
        row_class = 'winner' if rank == 1 else ''
        html_parts.append(f'<tr class="{row_class}"><td>{rank}</td><td>{model}</td><td>{row["mean"]:.4f}</td><td>{row["std"]:.4f}</td><td>{row["median"]:.4f}</td><td>{int(row["count"])}</td><td>{wins}</td><td>{win_pct:.1f}%</td></tr>')
    html_parts.append('</table>')

    # ===========================================================================
    # DOE STATISTICAL ANALYSIS SECTION
    # ===========================================================================
    html_parts.append('''
    <h2 id="doe">DOE Statistical Analysis</h2>
    <p>Rigorous statistical analysis using Design of Experiments methodology to determine if differences
    between models are statistically significant and practically meaningful.</p>
''')

    # RM-ANOVA Results
    if doe_stats.get('has_pingouin') and 'rm_anova' in doe_stats:
        rm_aov = doe_stats['rm_anova']

        # Handle different pingouin versions - column may be 'np2' or 'ng2'
        eta_sq_col = 'np2' if 'np2' in rm_aov.columns else 'ng2'
        eta_sq = rm_aov[eta_sq_col].values[0]
        eta_type = 'partial' if eta_sq_col == 'np2' else 'generalized'

        f_val = rm_aov['F'].values[0]
        p_col = 'p-GG-corr' if 'p-GG-corr' in rm_aov.columns else 'p-unc'
        p_val = rm_aov[p_col].values[0]

        effect_interp = 'negligible' if eta_sq < 0.01 else ('small' if eta_sq < 0.06 else ('medium' if eta_sq < 0.14 else 'large'))

        html_parts.append('''
    <h3>Repeated Measures ANOVA</h3>
    <p>Tests whether there is a statistically significant difference in performance across models,
    accounting for the repeated measures structure (all models fit on the same data).</p>
''')
        html_parts.append('<div class="metric-box">')
        html_parts.append(f'<p><strong>F-statistic:</strong> {f_val:.2f}</p>')
        html_parts.append(f'<p><strong>p-value:</strong> {p_val:.4f} {"***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))}</p>')
        html_parts.append(f'<p><strong>Effect Size ({eta_type} η²):</strong> {eta_sq:.4f} ({effect_interp})</p>')
        html_parts.append(f'<p><strong>Interpretation:</strong> Model choice explains {eta_sq*100:.1f}% of variance in test SSPE.</p>')
        html_parts.append('</div>')

        # Pairwise comparisons
        if 'pairwise' in doe_stats:
            pairwise = doe_stats['pairwise']

            html_parts.append('''
    <h3>Pairwise Comparisons (Holm-Bonferroni Corrected)</h3>
    <p>Statistical comparisons between specific model pairs with effect sizes (Hedges' g).
    Negative g means the first model has lower SSPE (better performance).</p>
''')

            # Key comparisons table
            key_pairs = [
                ('PCReg_CV', 'OLS'),
                ('PCReg_CV', 'OLS_LearnOnly'),
                ('PCReg_CV_Tight', 'OLS'),
                ('PCReg_ConstrainOnly', 'OLS'),
                ('PCReg_CV_Tight', 'PCReg_CV'),
                ('PCReg_CV_Wrong', 'PCReg_CV'),
            ]

            html_parts.append('<table>')
            html_parts.append('<tr><th>Comparison</th><th>Hedges\' g</th><th>p-value (Holm)</th><th>Significance</th><th>Interpretation</th></tr>')

            for a, b in key_pairs:
                row = pairwise[(pairwise['A'] == a) & (pairwise['B'] == b)]
                if len(row) == 0:
                    row = pairwise[(pairwise['A'] == b) & (pairwise['B'] == a)]

                if len(row) > 0:
                    row = row.iloc[0]
                    g = row['hedges']
                    p = row['p-corr']

                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                    g_abs = abs(g)
                    g_size = 'negligible' if g_abs < 0.2 else ('small' if g_abs < 0.5 else ('medium' if g_abs < 0.8 else 'large'))
                    better = a if g < 0 else b

                    html_parts.append(f'<tr><td>{a} vs {b}</td><td>{g:.4f}</td><td>{p:.4f}</td><td>{sig}</td><td>{g_size} effect, {better} better</td></tr>')

            html_parts.append('</table>')

    # Win Rate Analysis
    if 'overall_win_rate' in doe_stats:
        html_parts.append('''
    <h3>Win Rate Analysis (PCReg_CV vs OLS)</h3>
    <p>How often does PCReg_CV achieve lower test SSPE than OLS across scenarios?</p>
''')
        html_parts.append('<div class="metric-box">')
        html_parts.append(f'<p><strong>Overall Win Rate:</strong> {doe_stats["overall_win_rate"]:.1%} ({doe_stats["overall_wins"]}/{doe_stats["overall_total"]})</p>')
        html_parts.append(f'<p><strong>Binomial Test p-value:</strong> {doe_stats["overall_pvalue"]:.4f}</p>')
        if doe_stats["overall_pvalue"] < 0.05:
            html_parts.append('<p class="highlight">PCReg_CV significantly outperforms OLS overall (p < 0.05)</p>')
        html_parts.append('</div>')

        # Win rates by factor
        if 'win_rates' in doe_stats:
            html_parts.append('<h4>Win Rates by Design Factor</h4>')
            html_parts.append('<table>')
            html_parts.append('<tr><th>Factor</th><th>Level</th><th>Win Rate</th><th>Significant?</th></tr>')

            for _, row in doe_stats['win_rates'].iterrows():
                sig = '*' if row['significant'] else ''
                row_class = 'winner' if row['win_rate'] > 0.6 and row['significant'] else ''
                html_parts.append(f'<tr class="{row_class}"><td>{row["factor"]}</td><td>{row["level"]}</td><td>{row["win_rate"]:.1%}</td><td>{sig}</td></tr>')

            html_parts.append('</table>')

    # ===========================================================================
    # RECOMMENDATIONS SECTION
    # ===========================================================================
    html_parts.append('''
    <h2 id="recommendations">Recommendations: When to Use Each Method</h2>
    <div class="metric-box">
        <h3>Use PCReg (Penalized-Constrained Regression) when:</h3>
        <ul>
            <li><strong>Sample size is small</strong> (n_lots ≤ 10) - constraints provide valuable regularization</li>
            <li><strong>Prior knowledge is available</strong> about coefficient signs/bounds</li>
            <li><strong>Feature correlation is high</strong> (≥ 0.5) - constraints help with collinearity</li>
            <li><strong>Interpretability matters</strong> - ensures economically sensible coefficients</li>
        </ul>

        <h3>Standard OLS may be sufficient when:</h3>
        <ul>
            <li><strong>Sample size is large</strong> (n_lots ≥ 30)</li>
            <li><strong>Feature correlation is low</strong> (< 0.3)</li>
            <li><strong>No prior constraints</strong> are appropriate for the application</li>
        </ul>

        <h3>Key Findings:</h3>
        <ul>
            <li><strong>PCReg_CV_Tight</strong> (well-specified constraints) achieves the best performance when bounds are known</li>
            <li><strong>PCReg_CV_Wrong</strong> shows robustness - misspecified constraints still competitive</li>
            <li><strong>PCReg_ConstrainOnly</strong> often matches or beats PCReg_CV - penalty may not always help</li>
            <li><strong>OLS_LearnOnly</strong> (single predictor) performs poorly - rate effect is important</li>
        </ul>
    </div>
''')

    # Analysis by each category
    for category in CATEGORIES:
        category_values = sorted(df[category].unique())

        html_parts.append(f'''
    <h2 id="{category}">Analysis by {CATEGORY_LABELS[category]}</h2>
    <p>Values: {category_values}</p>
''')

        # Create and embed category boxplot
        fig_cat = create_boxplot_by_category(df, category, category_values, model_order)
        img_cat = fig_to_base64(fig_cat)
        html_parts.append(f'<img src="data:image/png;base64,{img_cat}" alt="Boxplot by {category}">')

        # Winner summary for this category
        html_parts.append(f'<h3>Winners by {CATEGORY_LABELS[category]}</h3>')
        html_parts.append('<table>')
        html_parts.append(f'<tr><th>{CATEGORY_LABELS[category]}</th><th>Winner</th><th>Mean SSPE</th></tr>')
        for val in category_values:
            subset = df[df[category] == val]
            means = subset.groupby('model_name')[METRIC].mean().sort_values()
            winner = means.index[0]
            winner_score = means.iloc[0]
            html_parts.append(f'<tr><td>{val}</td><td class="highlight">{winner}</td><td>{winner_score:.4f}</td></tr>')
        html_parts.append('</table>')

        # Full pivot table
        pivot_table = df.groupby([category, 'model_name'])[METRIC].mean().unstack()
        html_parts.append(f'<h3>Mean {METRIC} by {CATEGORY_LABELS[category]} and Model</h3>')
        html_parts.append('<div style="overflow-x: auto;">')
        html_parts.append(pivot_table.to_html(classes='', float_format='{:.4f}'.format))
        html_parts.append('</div>')

        # Descriptive statistics table
        stats_by_cat = df.groupby([category, 'model_name'])[METRIC].describe()
        html_parts.append(f'<h3>Detailed Statistics</h3>')
        html_parts.append('<details><summary>Click to expand full statistics table</summary>')
        html_parts.append('<div style="overflow-x: auto;">')
        html_parts.append(stats_by_cat.to_html(classes='', float_format='{:.4f}'.format))
        html_parts.append('</div></details>')

    # Winner frequency analysis
    html_parts.append('''
    <h2 id="wins">Winner Frequency Analysis</h2>
    <p>How often each model achieves the lowest test SSPE across all scenarios.</p>
''')

    html_parts.append('<table>')
    html_parts.append('<tr><th>Rank</th><th>Model</th><th>Win Count</th><th>Win %</th></tr>')
    sorted_wins = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, count) in enumerate(sorted_wins, 1):
        pct = 100 * count / total_scenarios
        row_class = 'winner' if rank == 1 else ''
        html_parts.append(f'<tr class="{row_class}"><td>{rank}</td><td>{model}</td><td>{count}</td><td>{pct:.1f}%</td></tr>')
    html_parts.append('</table>')

    # ===========================================================================
    # INTERACTION HEATMAPS SECTION
    # ===========================================================================
    html_parts.append('''
    <h2 id="heatmaps">Interaction Heatmaps</h2>
    <p>These heatmaps show how much PCReg_CV improves over OLS (in %) across different
    combinations of design factors. Green = PCReg better, Red = OLS better.</p>
''')

    # Create heatmaps for key factor combinations
    heatmap_pairs = [
        ('n_lots', 'target_correlation'),
        ('n_lots', 'cv_error'),
        ('target_correlation', 'cv_error'),
    ]

    for f1, f2 in heatmap_pairs:
        fig_heatmap = create_heatmap_figure(df, f1, f2)
        if fig_heatmap is not None:
            img_heatmap = fig_to_base64(fig_heatmap)
            html_parts.append(f'<img src="data:image/png;base64,{img_heatmap}" alt="Heatmap {f1} vs {f2}">')

    # Close HTML
    html_parts.append('''
</div>
</body>
</html>
''')

    return ''.join(html_parts)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("Loading simulation results...")
    df_results = pd.read_parquet(results_dir / "simulation_results.parquet")
    print(f"Loaded {len(df_results):,} rows")
    print(f"Models: {df_results['model_name'].unique().tolist()}")

    # Filter to converged models only
    df = df_results[df_results['converged'] == True].copy()
    print(f"After filtering converged: {len(df):,} rows")

    print("\nGenerating HTML report...")
    html_content = generate_html_report(df, results_dir)

    # Save report
    report_path = results_dir / "simulation_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nReport saved to: {report_path}")
    print("Open this file in a web browser to view the complete analysis.")
