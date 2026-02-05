import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path

# Load data
OUTPUT_DIR = Path('output_v2')
results_all = pd.read_parquet(OUTPUT_DIR / 'simulation_results.parquet')

# Filter to OLS and PCReg_GCV
MODELS = ['OLS', 'PCReg_GCV']
results = results_all[results_all['model_name'].isin(MODELS)].copy()

# Calculate OLS reasonableness
results['LC_est'] = 2 ** results['b']
results['RC_est'] = 2 ** results['c']
ols_data = results[results['model_name'] == 'OLS'][['seed', 'LC_est', 'RC_est']].copy()
ols_data['ols_reasonable'] = ((ols_data['LC_est'] >= 0.70) & (ols_data['LC_est'] <= 1.0) &
                               (ols_data['RC_est'] >= 0.70) & (ols_data['RC_est'] <= 1.0))

# Build comparison df
ols_df = results[results['model_name'] == 'OLS'][['seed', 'test_mape', 'n_lots', 'cv_error', 'actual_correlation']].copy()
ols_df = ols_df.rename(columns={'test_mape': 'ols_mape'})
pcreg_df = results[results['model_name'] == 'PCReg_GCV'][['seed', 'test_mape']].copy()
pcreg_df = pcreg_df.rename(columns={'test_mape': 'pcreg_mape'})

comparison = ols_df.merge(pcreg_df, on='seed').merge(ols_data[['seed', 'ols_reasonable']], on='seed')
comparison['pcreg_wins'] = comparison['pcreg_mape'] < comparison['ols_mape']

# Features WITH ols_reasonable
X_with = comparison[['n_lots', 'actual_correlation', 'ols_reasonable']].copy()
X_with['ols_reasonable'] = X_with['ols_reasonable'].astype(int)

# Features WITHOUT ols_reasonable (only truly observable)
X_without = comparison[['n_lots', 'actual_correlation', 'cv_error']].copy()

y = comparison['pcreg_wins'].astype(int)

print('='*60)
print('DECISION TREE ANALYSIS')
print('='*60)

# Decision Tree WITH ols_reasonable
dt_with = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=42)
cv_with = cross_val_score(dt_with, X_with, y, cv=5)
print(f'DT with ols_reasonable:    CV Accuracy = {cv_with.mean():.1%} (+/- {cv_with.std():.1%})')

# Decision Tree WITHOUT ols_reasonable
dt_without = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=42)
cv_without = cross_val_score(dt_without, X_without, y, cv=5)
print(f'DT without ols_reasonable: CV Accuracy = {cv_without.mean():.1%} (+/- {cv_without.std():.1%})')

# Baseline (just predict majority class)
baseline = max(y.mean(), 1 - y.mean())
print(f'Baseline (majority class): Accuracy = {baseline:.1%}')

print()
print('='*60)
print('RANDOM FOREST ANALYSIS')
print('='*60)

# Random Forest WITH ols_reasonable
rf_with = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=50, random_state=42)
cv_rf_with = cross_val_score(rf_with, X_with, y, cv=5)
print(f'RF with ols_reasonable:    CV Accuracy = {cv_rf_with.mean():.1%} (+/- {cv_rf_with.std():.1%})')

# Random Forest WITHOUT ols_reasonable
rf_without = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=50, random_state=42)
cv_rf_without = cross_val_score(rf_without, X_without, y, cv=5)
print(f'RF without ols_reasonable: CV Accuracy = {cv_rf_without.mean():.1%} (+/- {cv_rf_without.std():.1%})')

print()
print('='*60)
print('FEATURE IMPORTANCE (RF without ols_reasonable)')
print('='*60)
rf_without.fit(X_without, y)
for feat, imp in zip(['n_lots', 'correlation', 'cv_error'], rf_without.feature_importances_):
    print(f'{feat:20s}: {imp:.3f}')

print()
print('='*60)
print('WIN RATES BY OBSERVABLE FACTORS')
print('='*60)

# Win rate by n_lots
print('\nBy n_lots:')
for n in sorted(comparison['n_lots'].unique()):
    subset = comparison[comparison['n_lots'] == n]
    print(f"  n={n:2d}: PCReg wins {subset['pcreg_wins'].mean()*100:.1f}%  (n={len(subset)})")

# Win rate by cv_error  
print('\nBy cv_error:')
for cv in sorted(comparison['cv_error'].unique()):
    subset = comparison[comparison['cv_error'] == cv]
    print(f"  cv={cv:.2f}: PCReg wins {subset['pcreg_wins'].mean()*100:.1f}%  (n={len(subset)})")

# Cross-tabulation
print('\n' + '='*60)
print('PCReg WIN RATE: n_lots x cv_error')
print('='*60)
pivot = comparison.pivot_table(values='pcreg_wins', index='n_lots', columns='cv_error', aggfunc='mean') * 100
print(pivot.round(1).to_string())
