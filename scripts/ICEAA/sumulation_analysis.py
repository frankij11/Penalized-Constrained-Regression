import pandas as pd
import numpy as np
from pathlib import Path

PARENT = Path(__file__).resolve().parent
RESULTS_PATH = PARENT / "Output_v2" / "simulation_results.parquet"

class CONFIG:
    all_models=False

if CONFIG.all_models:
    MODELS_TO_COMPARE = [
        "OLS","PCReg_GCV","OLS_LearnOnly","RidgeCV", "BayesianRidge", "LassoCV"
    ]
else:
    MODELS_TO_COMPARE = [
        "OLS","PCReg_GCV"#,"OLS_LearnOnly","RidgeCV", "BayesianRidge", "LassoCV"
]

METRICS_TO_COMPARE = [
    "test_mape","test_mse", "b_error","c_error","T1_error"
]


# find seeds where OLS produces "bad" coefficients but good R2
def find_bad_ols_coefs(df, r2=.8):
    seeds = df.query("(LC_est>1 | RC_est>1) and r2>0.8 and model_name=='OLS'")["seed"].unique()
    df = df.assign(bad_ols_coefs=lambda x: x["seed"].isin(seeds))
    return df

def find_good_pcreg_fits(df, r2=.8):
    seeds=df.query("bad_ols_coefs==1 and rank_test_mape==1 and model_name.str.lower().str.contains('pc') and alpha>0")["seed"].unique()
    df = df.assign(pcreg_improves_bad_ols_coef=lambda x: x["seed"].isin(seeds))
    seeds=df.query("bad_ols_coefs==0 and rank_test_mape==1 and model_name.str.lower().str.contains('pc') and alpha>0")["seed"].unique()
    df = df.assign(pcreg_improves_good_ols_coef=lambda x: x["seed"].isin(seeds))

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

df = (pd.read_parquet(RESULTS_PATH)
      .pipe(get_subset_of_models, MODELS_TO_COMPARE, CONFIG.all_models)
      .pipe(find_bad_ols_coefs)
      .pipe(rank_models, criteria='test_mape')
      .pipe(pct_beats_ols, criteria='test_mape')
      .pipe(find_good_pcreg_fits)
      .pipe(rank_models, criteria='b_error')
      .pipe(rank_models, criteria='c_error')
      .pipe(rank_models, criteria='T1_error')
)
filename = "All_Models" if CONFIG.all_models else "OLS_vs_PCReg"
df.to_csv(PARENT / "Output_v2" / f"simulation_results_extended_{filename}.csv", index=False)


df_study_data = pd.read_parquet(RESULTS_PATH.parent / "simulation_study_data.parquet")

# motivational example is a run where OLS has very bad test_mape but PCReg is best and has a good test_mape
motivational_example_results = df.sort_values('test_mape').query("bad_ols_coefs==1 and model_name.str.contains('PC') and LC_est <1 and RC_est<1 and alpha>0").reset_index().loc[0]
df_motivational = df_study_data.query("seed==@motivational_example_results.seed")
df_motivational.to_csv(PARENT / "Output_v2" / "motivational_example_data.csv", index=False)
print("Motivational example Results:", df.query("seed==@motivational_example_results.seed").T)



print("motivational example study data:")
print(df_study_data.query("seed==@motivational_example_results.seed"))
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
plt.show()

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
plt.show()

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

        for model in df["model_name"].unique():
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
plt.show()
