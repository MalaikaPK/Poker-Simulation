import pandas as pd
from scipy.stats import ttest_rel, chi2_contingency, chisquare, f_oneway, ttest_1samp
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.iolib.summary2 import summary_col

"""
This module includes several statistical tests, although, not all are used in the analysis.
When running the Chi_square_test.py file, the results extracted from poker_simulation.py are saved in CSV files.
Note, that when running it, the CSV files are overwritten if not named something different in between the simulations,
why the CSV files not include all raw results from the main and robustness simulations.
Regressions etc. will also be overwritten if not saved separately.
"""

# Load simulation results from Chi_square_test.py
file_path_bne = '/Users/Malaika/Desktop/BA Projekt/Python/poker-sim-master/poker_simulation_results_BNE.csv'
file_path_pbe = '/Users/Malaika/Desktop/BA Projekt/Python/poker-sim-master/poker_simulation_results_PBE.csv'
file_path_earnings = '/Users/Malaika/Desktop/BA Projekt/Python/poker-sim-master/earnings_raw_per_simulation.csv'
file_path_raw = '/Users/Malaika/Desktop/BA Projekt/Python/poker-sim-master/player_stats_per_simulation.csv'

results_bne = pd.read_csv(file_path_bne)
results_pbe = pd.read_csv(file_path_pbe)
earnings_raw= pd.read_csv(file_path_earnings)
results_raw= pd.read_csv(file_path_raw)

print("Simulation Results BNE:\n", results_bne)
print("\nSimulation Results PBE:\n", results_pbe)

# Linear Regression on Earnings
print("\n--- Linear Regression ---")
results_bne['Model'] = 'BNE'
results_pbe['Model'] = 'PBE'
combined = pd.concat([results_bne, results_pbe])

# Rename columns for consistency
results_bne = results_bne.rename(columns={
    'Total Earnings': 'Earnings',
    'Avg Earnings (Per Round)': 'EarningsPerRound',
    'Raises': 'Raises',
    'Win Rate': 'WinRate'
})

results_pbe = results_pbe.rename(columns={
    'Total Earnings': 'Earnings',
    'Avg Earnings (Per Round)': 'EarningsPerRound',
    'Raises': 'Raises',
    'Win Rate': 'WinRate'
})

# Combine the datasets
combined = pd.concat([results_bne, results_pbe])


# Rename columns for consistency in earnings_raw and results_raw
results_raw = results_raw.rename(columns={
    'Total Earnings': 'Earnings',
    'Avg Earnings (Per Round)': 'EarningsPerRound',
    'Raises': 'Raises',
    'Win Rate': 'WinRate'
})

# Rename columns for consistency in earnings_raw and results_raw
earnings_raw = results_raw.rename(columns={
    'Total Earnings': 'Earnings',
    'Avg Earnings (Per Round)': 'EarningsPerRound',
    'Raises': 'Raises',
    'Win Rate': 'WinRate'
})


# Add 'Action' column based on Raises
results_raw['Action'] = np.where(results_raw['Raises'] > 0, 'Raise', 'Fold')

print(results_raw['Player'].unique())

results_raw['Player'] = pd.Categorical(results_raw['Player'], 
                                       categories=['Rational', 'Loss Averse', 'Optimistic', 'Risk Averse', 'Risk Seeking'], 
                                       ordered=False)



"""
Several regressions. Not all are used for Latex tables.
"""

# Earnings explained by model, win rate, and raises
model1 = smf.ols("Earnings ~ C(Model) * WinRate + Raises", data=results_raw).fit()
print("\nEarnnings Model Win interaction:")
print(model1.summary())

# Earnings explained by players
model = smf.ols("Earnings ~ C(Player)", data=results_raw).fit()
print("\nTotal Earnings Player:")
print(model.summary())

# Win Rate explained by players
model2 = smf.ols("WinRate ~ C(Player)", data=results_raw).fit()
print("\nWin Rate Player:")
print(model2.summary())

# Earnings by raises, win rate and model
model_total = smf.ols('Earnings ~ Raises + WinRate + C(Model)', data=results_raw).fit()
print("\nTotal Earnings Regression:")
print(model_total.summary())

# Average earnings by raises, win rate and model.
model_avg = smf.ols('EarningsPerRound ~ Raises + WinRate + C(Model)', data=results_raw).fit()
print("\nAvg Earnings Per Round Regression:")
print(model_avg.summary())


# Win rate by raises and avg earnings. 
model_winrate = smf.ols('WinRate ~ Raises + EarningsPerRound', data=results_raw).fit()
print("\nWin Rate Regression:")
print(model_winrate.summary())


# Avg earningns by model, players, and raises. 
model_simple = smf.ols('EarningsPerRound ~ C(Model) * C(Player) + C(Model) + Raises', data=results_raw).fit()
print("\nEarningsPerRound with Model interaction:")
print(model_simple.summary())


# Win Rate by avg earnings and player.
model_int = smf.ols('WinRate ~ EarningsPerRound * C(Player) + EarningsPerRound', data=results_raw).fit()
print("\Win Rate with player interaction")
print(model_int.summary())


[model1, model, model2, model_winrate, model_int, model_avg, model_simple],


anova_results = anova_lm(model_avg, model_simple)
print("\nModel Comparison (Additive vs. Interaction):")
print(anova_results)


# Per-Player Paired T-Tests
print("\n--- Per-Player T-Tests ---")
for player in combined['Player'].unique():
    bne_total = combined[(combined['Player'] == player) & (combined['Model'] == 'BNE')]['Earnings']
    pbe_total = combined[(combined['Player'] == player) & (combined['Model'] == 'PBE')]['Earnings']

    bne_avg = combined[(combined['Player'] == player) & (combined['Model'] == 'BNE')]['EarningsPerRound']
    pbe_avg = combined[(combined['Player'] == player) & (combined['Model'] == 'PBE')]['EarningsPerRound']

    if len(bne_total) == len(pbe_total) and len(bne_total) > 1:
        t_total, p_total = ttest_rel(bne_total, pbe_total)
        t_round, p_round = ttest_rel(bne_avg, pbe_avg)
        t_total, p_total = wilcoxon(bne_total, pbe_total)
        t_round, p_round = wilcoxon(bne_avg, pbe_avg)
        print(f"{player}:")
        print(f"  Total Earnings: t = {t_total:.4f}, p = {p_total:.4f} {'✅' if p_total < 0.05 else '❌'}")
        print(f"  Avg Earnings/Round: t = {t_round:.4f}, p = {p_round:.4f} {'✅' if p_round < 0.05 else '❌'}")
    elif len(bne_total) == len(pbe_total) == 1:
        print(f"{player}: ΔTotal = {pbe_total.iloc[0] - bne_total.iloc[0]:.2f}, ΔAvg/Round = {pbe_avg.iloc[0] - bne_avg.iloc[0]:.2f}")
    else:
        print(f"{player}: Insufficient data.")




def chi_square_test(df, label="", print_expected=False):
    """
    Perform Chi-Square test for independence on categorical data in a dataframe.
    
    Parameters:
    - df: DataFrame containing categorical data (e.g., Player, Raises, Folds)
    - print_expected: If True, prints the expected frequencies
    """
    # Convert the categorical columns into a contingency table
    contingency_table = pd.crosstab(df['Player'], df['Raises'])
    
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\n--- Chi-Square Test ({label}) ---")
        print(f"Chi² = {chi2:.4f}, p = {p:.5f}, DoF = {dof}")
        if print_expected:
            print("Expected frequencies:\n", expected)
    except Exception as e:
        print(f"ERROR in Chi-Square test ({label}): {e}")


# Combine the results to analyse both models together
results_bne['Model'] = 'BNE'
results_pbe['Model'] = 'PBE'
combined = pd.concat([results_bne, results_pbe])

# Testing "Raises" by Players
chi_square_test(combined, label="Raises by Player Type", print_expected=True)

# Perform the ANOVA test for "EarningsPerRound"
anova_result = stats.f_oneway(
    combined[combined['Player'] == 'Risk Averse']['EarningsPerRound'],
    combined[combined['Player'] == 'Risk Seeking']['EarningsPerRound'],
    combined[combined['Player'] == 'Loss Averse']['EarningsPerRound'],
    combined[combined['Player'] == 'Optimistic']['EarningsPerRound'],
    combined[combined['Player'] == 'Rational']['EarningsPerRound']
)

print(f"ANOVA Test Results: F = {anova_result.statistic:.4f}, p = {anova_result.pvalue:.5f}")


# Separate the data for BNE and PBE
bne = combined[combined['Model'] == 'BNE']
pbe = combined[combined['Model'] == 'PBE']

metrics = ['Earnings', 'EarningsPerRound', 'Raises', 'WinRate']

# For BNE
print("T-Test and Mann-Whitney U Results for BNE:")

for metric in metrics:
    t_stat_bne, p_val_bne = ttest_1samp(bne[metric], 0)  # Test against zero
    u_stat_bne, p_u_bne = mannwhitneyu(bne[metric], [0] * len(bne[metric]), alternative='two-sided')  # Non-parametric test
    print(f"\n{metric} (BNE):\n  T-test: t = {t_stat_bne:.4f}, p = {p_val_bne:.5f}\n  Mann-Whitney U: U = {u_stat_bne}, p = {p_u_bne:.5f}")

# For PBE
print("\nT-Test and Mann-Whitney U Results for PBE:")

for metric in metrics:
    t_stat_pbe, p_val_pbe = ttest_1samp(pbe[metric], 0)  # Test against zero
    u_stat_pbe, p_u_pbe = mannwhitneyu(pbe[metric], [0] * len(pbe[metric]), alternative='two-sided')  # Non-parametric test
    print(f"\n{metric} (PBE):\n  T-test: t = {t_stat_pbe:.4f}, p = {p_val_pbe:.5f}\n  Mann-Whitney U: U = {u_stat_pbe}, p = {p_u_pbe:.5f}")


# Proportion Test: Raise vs Fold Proportions by Model
print("\n--- Proportion Test: Raise vs. Fold by Model ---")
combined['Action'] = np.where(combined['Raises'] > 0, 'Raise', 'Fold')
action_counts = pd.crosstab(combined['Model'], combined['Action'])

# Fisher Exact only for 2x2 tables
if action_counts.shape == (2, 2):
    oddsratio, p_fisher = fisher_exact(action_counts)
    print(f"Fisher's Exact: OR = {oddsratio:.4f}, p = {p_fisher:.5f} {'✅' if p_fisher < 0.05 else '❌'}")
else:
    chi2, p_chi, dof, ex = chi2_contingency(action_counts)
    print(f"Chi² = {chi2:.4f}, p = {p_chi:.5f} {'✅' if p_chi < 0.05 else '❌'}")


# Shapiro-Wilk Normality Test on Earnings 
print("\n--- Shapiro-Wilk Normality Test ---")
for model in ['BNE', 'PBE']:
    stat, p = stats.shapiro(combined[combined['Model'] == model]['Earnings'])
    print(f"{model}: W = {stat:.4f}, p = {p:.5f} {'✅' if p > 0.05 else '❌'}")


# Levene's Test
print("\n--- Levene's Test for Equal Variance ---")
stat, p = stats.levene(
    combined[combined['Model'] == 'BNE']['Earnings'],
    combined[combined['Model'] == 'PBE']['Earnings']
)
print(f"Levene's Test: F = {stat:.4f}, p = {p:.5f} {'✅' if p > 0.05 else '❌'}")


# Effect Size: Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

print("\n--- Cohen's d (Effect Size) ---")
d = cohens_d(
    combined[combined['Model'] == 'BNE']['Earnings'],
    combined[combined['Model'] == 'PBE']['Earnings']
)
print(f"Cohen's d: {d:.4f}")


# Bootstrapped Confidence Interval
print("\n--- Bootstrapped 95% CI for Mean Earnings ---")
def bootstrap_ci(data, n_boot=1000, ci=95):
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper

for model in ['BNE', 'PBE']:
    earnings = combined[combined['Model'] == model]['Earnings']
    lower, upper = bootstrap_ci(earnings)
    print(f"{model}: 95% CI = [{lower:.2f}, {upper:.2f}]")



def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * (1 - ci) / 2)
    upper = np.percentile(boot_means, 100 * (1 + ci) / 2)
    return lower, upper



# Subgroup t-tests, effect sizes, and CIs
player_types = earnings_raw['Player'].unique()


print("\n=== Subgroup Analysis by Player Type ===")
for player in player_types:
    subset = earnings_raw[earnings_raw['Player'] == player]
    earnings_bne = subset[subset['Model'] == 'BNE']['Earnings'].values
    earnings_pbe = subset[subset['Model'] == 'PBE']['Earnings'].values

    # T-test
    t_stat, p_val = ttest_ind(earnings_bne, earnings_pbe, equal_var=False)

    # Effect size
    d = cohens_d(earnings_bne, earnings_pbe)

    # Bootstrap CI
    ci_bne = bootstrap_ci(earnings_bne)
    ci_pbe = bootstrap_ci(earnings_pbe)


# T-test for comparing earnings between BNE and PBE for each player
for player_type in earnings_raw['Player'].unique():
    earnings_bne = earnings_raw[(earnings_raw['Model'] == 'BNE') & (earnings_raw['Player'] == player_type)]['Earnings']
    earnings_pbe = earnings_raw[(earnings_raw['Model'] == 'PBE') & (earnings_raw['Player'] == player_type)]['Earnings']
 

# ANOVA for earnings by players
earnings_by_type = [results_raw[results_raw['Player'] == ptype]['Earnings'] for ptype in player_types]
f_stat, p_val = f_oneway(*earnings_by_type)
print(f"ANOVA for earnings by player type: F-statistic = {f_stat}, p-value = {p_val}")

# ANOVA for Winrate by players
earnings_by_type = [results_raw[results_raw['Player'] == ptype]['WinRate'] for ptype in player_types]
f_stat, p_val = f_oneway(*earnings_by_type)
print(f"ANOVA for WinRate by player type: F-statistic = {f_stat}, p-value = {p_val}")


# Mann-Whitney U test between BNE and PBE for a specific player
player_type = 'Rational'  # This can be changed as needed
earnings_bne = results_raw[(results_raw['Model'] == 'BNE') & (results_raw['Player'] == player_type)]['Earnings']
earnings_pbe = results_raw[(results_raw['Model'] == 'PBE') & (results_raw['Player'] == player_type)]['Earnings']
u_stat, p_val = mannwhitneyu(earnings_bne, earnings_pbe)
print(f"Mann-Whitney U Test for {player_type} earnings: U-statistic = {u_stat}, p-value = {p_val}")

# Wilcoxon test for paired data (BNE vs PBE for a player)
earnings_bne = results_raw[(results_raw['Model'] == 'BNE') & (results_raw['Player'] == player_type)]['Earnings']
earnings_pbe = results_raw[(results_raw['Model'] == 'PBE') & (results_raw['Player'] == player_type)]['Earnings']
w_stat, p_val = wilcoxon(earnings_bne, earnings_pbe)
print(f"Wilcoxon Signed-Rank Test for {player_type} earnings: w-statistic = {w_stat}, p-value = {p_val}")

# Tukey HSD test for pairwise comparisons between players' earnings
model_data = pd.concat([results_raw[['Earnings']], results_raw['Player']], axis=1)  # Concatenate player types
tukey_results = pairwise_tukeyhsd(endog=model_data['Earnings'], groups=model_data['Player'], alpha=0.05)
print(tukey_results)

# Cohen's d for effect size between two groups (BNE vs PBE for a player)
from statsmodels.stats.weightstats import DescrStatsW
bne_stats = DescrStatsW(earnings_bne)
pbe_stats = DescrStatsW(earnings_pbe)
cohen_d = (bne_stats.mean - pbe_stats.mean) / np.sqrt((bne_stats.var + pbe_stats.var) / 2)
print(f"Cohen's d for {player_type} earnings: {cohen_d}")


# Separate the data for BNE and PBE
bne = combined[combined['Model'] == 'BNE']
pbe = combined[combined['Model'] == 'PBE']

metrics = ['Earnings', 'EarningsPerRound', 'Raises', 'WinRate']

# Reference value for one-sample t-test (e.g., 0 if testing whether significantly different from 0)
ref_value = 0

print("T-Test and Mann-Whitney U Results for BNE:")
for metric in metrics:
    t_stat_bne, p_val_bne = ttest_1samp(bne[metric], popmean=ref_value, nan_policy='omit')
    u_stat_bne, u_p_val_bne = mannwhitneyu(bne[metric], [ref_value] * len(bne[metric]), alternative='two-sided')
    print(f"{metric} - t = {t_stat_bne:.4f}, p = {p_val_bne:.4f} | U = {u_stat_bne:.4f}, p = {u_p_val_bne:.4f}")

print("\nT-Test and Mann-Whitney U Results for PBE:")
for metric in metrics:
    t_stat_pbe, p_val_pbe = ttest_1samp(pbe[metric], popmean=ref_value, nan_policy='omit')
    u_stat_pbe, u_p_val_pbe = mannwhitneyu(pbe[metric], [ref_value] * len(pbe[metric]), alternative='two-sided')
    print(f"{metric} - t = {t_stat_pbe:.4f}, p = {p_val_pbe:.4f} | U = {u_stat_pbe:.4f}, p = {u_p_val_pbe:.4f}")


"""
More regressions.
"""

# Earnings by win rate and player.
model = smf.ols('Earnings ~ WinRate * C(Player)', data=results_raw).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# Earnings by raises and players.
model = smf.ols('Earnings ~ Raises * C(Player)', data=results_raw).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


#Earnings by model and players.
model = smf.ols('Earnings ~ C(Model) * C(Player)', data=results_raw).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)



# Create the summary table with all customisations
summary = summary_col(
    [model1, model, model2, model_winrate, model_int, model_avg, model_simple],
    stars=True,
    float_format='%0.2f',
    model_names=[
        "Earnings I",
        "Earnings II",
        "WinRate I",
        "WinRate II",
        "WinRate III",
        "Avg Earnings I",
        "Avg Earnings II"
    ],
    drop_omitted=True,  # drops rows for coefficients not used in a model
    info_dict={
        'N': lambda x: f"{int(x.nobs)}",
        'R2': lambda x: f"{x.rsquared:.2f}"
    }
)

# Print to console
print(summary)

# Save to LaTeX file. 
# A new file regression_table.tex will occur to copy into LaTeX (although not pretty so must be costumised in LaTeX)
with open("regression_table.tex", "w") as f:
    f.write(summary.as_latex())


# Subset for Rational player
rational_bne = results_raw[(results_raw['Model'] == 'BNE') & (results_raw['Player'] == 'Rational')]
rational_pbe = results_raw[(results_raw['Model'] == 'PBE') & (results_raw['Player'] == 'Rational')]


# Paired t-test
t_stat, p_val = ttest_rel(rational_pbe['EarningsPerRound'].values, rational_bne['EarningsPerRound'].values)
print(f"Paired t-test (Rational): t = {t_stat:.4f}, p = {p_val:.4f}")

# Wilcoxon if non-normal
w_stat, p_val_w = wilcoxon(rational_pbe['EarningsPerRound'].values, rational_bne['EarningsPerRound'].values)
print(f"Wilcoxon test (Rational): W = {w_stat:.4f}, p = {p_val_w:.4f}")


# Compute delta earnings per simulation for each player
results_pivot = results_raw.pivot_table(index=['Simulation', 'Player'], columns='Model', values='EarningsPerRound').reset_index()
results_pivot['delta'] = results_pivot['PBE'] - results_pivot['BNE']

# Split into Rational vs Non-Rational
delta_rational = results_pivot[results_pivot['Player'] == 'Rational']['delta']
delta_others = results_pivot[results_pivot['Player'] != 'Rational']['delta']

# t-test or Mann-Whitney
t_stat_d, p_val_d = ttest_ind(delta_rational, delta_others)
u_stat, p_val_u = mannwhitneyu(delta_rational, delta_others)

print(f"Independent t-test on Δ Earnings: t = {t_stat_d:.4f}, p = {p_val_d:.4f}")
print(f"Mann-Whitney on Δ Earnings: U = {u_stat:.4f}, p = {p_val_u:.4f}")

model = smf.ols('EarningsPerRound ~ C(Model) * C(Player)', data=results_raw).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

np.var(rational_bne['EarningsPerRound']), np.var(rational_pbe['EarningsPerRound'])
print("Variance (BNE):", np.var(rational_bne['EarningsPerRound']))
print("Variance (PBE):", np.var(rational_pbe['EarningsPerRound']))



t_stat, p_val = ttest_rel(rational_pbe['WinRate'], rational_bne['WinRate'])
print(f"Paired t-test on WinRate (Rational): t = {t_stat:.4f}, p = {p_val:.4f}")

rational_data = results_raw[results_raw['Player'] == 'Rational']

"""
ANOVA and models
"""

# Avg earnings by model and raises.
model = smf.ols('EarningsPerRound ~ C(Model) * Raises', data=rational_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# Win rate by avg earnings and raises.
model = smf.ols('WinRate ~ EarningsPerRound * Raises', data=rational_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# Raises by avg earnings and win rate
model = smf.ols('Raises ~ EarningsPerRound * WinRate', data=rational_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# Avg earnings by raises
model = smf.ols('EarningsPerRound ~ Raises', data=rational_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# Avg earnings by win rate
model = smf.ols('EarningsPerRound ~ WinRate', data=rational_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)



"""
Manual tables for eanings results.
Note that these tables are similar to the ones in Chi_square_test.py, however they are based on total earnings.
These are based on average earnings, abstracted from the performance results used in the Analysis.
To ensure proper figures without running the simulation and overwrite previous results, I made new manual figures.
"""

# Define custom colors
copenhagen_red = "#990000"
pbe_gray = "#A5ACAF"

# Input data from the user's table
data = {
    'Player': ['Risk Averse', 'Risk Seeking', 'Loss Averse', 'Optimistic', 'Rational'],
    'BNE': [165.0, -57.0, 97.0, -96.0, -109.0],
    'PBE': [4.0, -46.0, 59.0, -79.0, 62.0]
}

# Create separate dataframes for BNE and PBE
df_bne = pd.DataFrame({
    'Player': data['Player'],
    'Earnings': data['BNE'],
    'Type': 'BNE'
})
df_pbe = pd.DataFrame({
    'Player': data['Player'],
    'Earnings': data['PBE'],
    'Type': 'PBE'
})

# Combine for plotting
df = pd.concat([df_bne, df_pbe])

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Player', y='Earnings', hue='Type', data=df, palette=[copenhagen_red, pbe_gray])
plt.title("Average Earnings Comparison (BNE vs PBE)")
plt.ylabel("Average Chips Earned")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Define custom colors
copenhagen_red = "#990000"
pbe_gray = "#A5ACAF"

# Input data from the user's table
data = {
    'Player': ['Risk Averse', 'Risk Seeking', 'Loss Averse', 'Optimistic', 'Rational'],
    'BNE': [6.8, -2.6, -43.0, 15.2, 23.6],
    'PBE': [-6.8, -4.4, 11.0, -7.0, 7.2]
}

# Create separate dataframes for BNE and PBE
df_bne = pd.DataFrame({
    'Player': data['Player'],
    'Earnings': data['BNE'],
    'Type': 'BNE'
})
df_pbe = pd.DataFrame({
    'Player': data['Player'],
    'Earnings': data['PBE'],
    'Type': 'PBE'
})

# Combine for plotting
df = pd.concat([df_bne, df_pbe])

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Player', y='Earnings', hue='Type', data=df, palette=[copenhagen_red, pbe_gray])
plt.title("Average Earnings Comparison (BNE vs PBE)")
plt.ylabel("Average Chips Earned")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()