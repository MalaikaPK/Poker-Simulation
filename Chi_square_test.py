from poker_simulation import simulate_poker_games
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import time
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as smp
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import levene, shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats import weightstats as stests
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import csv



def compute_percentages(simulation_result):
    """
    Convert raw counts into percentages using player names and extract Raise/Fold counts for Chi-Square test.
    """
    percentages = {}
    observed_counts = []
    earnings_dict = {}
    avg_earnings_per_round = {}

    for player_obj, (raises, folds, wins) in simulation_result.results.items():
        player_name = player_obj.name
        total_decisions = raises + folds
        raise_pct = (raises / total_decisions) * 100 if total_decisions else 0
        fold_pct = (folds / total_decisions) * 100 if total_decisions else 0
        win_pct = (wins / total_decisions) * 100 if total_decisions else 0
        avg_earnings = simulation_result.earnings.get(player_obj, 0)
        avg_earnings_round = sum(simulation_result.earnings_per_round.get(player_obj, [])) / len(simulation_result.earnings_per_round.get(player_obj, [])) if simulation_result.earnings_per_round.get(player_obj) else 0
        
        percentages[player_name] = (raise_pct, fold_pct, win_pct)
        earnings_dict[player_name] = avg_earnings
        avg_earnings_per_round[player_name] = avg_earnings_round

        if raises > 0 or folds > 0:
            observed_counts.append([raises, folds])
    return percentages, np.array(observed_counts), earnings_dict, avg_earnings_per_round



def chi_square_test(observed_counts, label=""):
    """ Perform a Chi-Square test for independence on raise/fold behavior. """
    if observed_counts.size == 0:
        print(f"ERROR: No data for Chi-Square test ({label})")
        return
    chi2, p, dof, expected_freqs = stats.chi2_contingency(observed_counts)
    print(f"\nChi-Square Test ({label}):\nChi2: {chi2:.3f}, p-value: {p:.5f}, DoF: {dof}")

def compute_confidence_intervals(simulation_result, n_trials, label=""):
    """ Compute 95% confidence intervals for win rates. """
    for player, (_, _, wins) in simulation_result.results.items():
        if wins > 0:
            ci = smp.proportion_confint(wins, n_trials, alpha=0.05, method='wilson')
        else:
            ci = (0, 0)
        print(f"{player.name}: {ci}")



def t_tests_within_equilibrium(results, label="BNE"):
    """Perform pairwise t-tests between player types within the same equilibrium."""
    print(f"\nT-tests for {label} Earnings Between Player Types:")
    for (player1, player2) in combinations(results.keys(), 2):
        earnings1 = np.array(results[player1])
        earnings2 = np.array(results[player2])

        t_stat, p_value = stats.ttest_ind(earnings1, earnings2, nan_policy='omit', alternative='two-sided')
        print(f"{player1} vs {player2}: t={t_stat:.3f}, p={p_value:.3f}")


def t_test_per_player(bne_results, pbe_results):
    """ Perform t-tests for each player type separately. """
    for player in bne_results.keys():
        # Extract win rates as arrays (ensure no missing values)
        bne_win_rates = np.array(bne_results[player])
        pbe_win_rates = np.array(pbe_results[player])

        # Perform independent t-test for win rates
        t_stat, p_value = stats.ttest_ind(bne_win_rates, pbe_win_rates, nan_policy='omit', alternative="two-sided")
        
        # Output the results for each player
        print(f"{player} Win Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")

def t_test_per_player_earnings(bne_results, pbe_results):
    """ Perform t-tests for earnings for each player type separately. """
    for player in bne_results.keys():
        bne_earnings = np.array(bne_results[player])
        pbe_earnings = np.array(pbe_results[player])
        t_stat, p_value = stats.ttest_ind(bne_earnings, pbe_earnings, nan_policy='omit')
        d = cohen_d(bne_earnings, pbe_earnings)
        print(f"{player} Earnings t-test: t={t_stat:.3f}, p={p_value:.3f}, Cohen's d={d:.2f}")



def combine_results(bne_results, earnings_dict, pbe_results):
    # Create DataFrames
    bne_df = pd.DataFrame(bne_results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    bne_df['Avg Earnings'] = bne_df.index.map(lambda player: earnings_dict.get(player, 0))
    pbe_df = pd.DataFrame(pbe_results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    pbe_df['Avg Earnings'] = pbe_df.index.map(lambda player: earnings_dict.get(player, 0))

    # Add strategy labels
    bne_df['PBE'] = 0
    pbe_df['PBE'] = 1

    # Combine both
    combined_df = pd.concat([bne_df, pbe_df], ignore_index=True)

    # Add interaction terms
    combined_df['Earnings_PBE_interaction'] = combined_df['Avg Earnings'] * combined_df['PBE']
    combined_df['win_raise_interaction'] = combined_df['Raises (%)'] * combined_df['Win Rate (%)']
    combined_df['Intercept'] = 1

    return combined_df

# Regressions, not used for latex tables but tests
def run_regression_interaction_1(combined_df):
    X = combined_df[["Win Rate (%)", "PBE", "win_raise_interaction", "Intercept"]]
    y = combined_df["Avg Earnings"]

    model = sm.OLS(y, X).fit()
    print("\nRegression Results  interaction:\n", model.summary())

def run_regression(results, earnings_dict):
    """
    Run a regression to analyse how different players influence win rates.
    """
    df = pd.DataFrame(results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    df['Avg Earnings'] = df.index.map(lambda player: earnings_dict.get(player, 0))
    df["Intercept"] = 1
    X = df[["Avg Earnings", "Intercept"]]
    y = df["Win Rate (%)"]
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results:\n", model.summary())

def run_regression_interaction(results, earnings_dict, is_pbe):
    """
    Run a regression to analyse how different players influence win rates.
    """
    df = pd.DataFrame(results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    df['Avg Earnings'] = df.index.map(lambda player: earnings_dict.get(player, 0))

    df["Intercept"] = 1
    df["PBE"] = int(is_pbe)
    df["Earnings_PBE_interaction"] = df["Avg Earnings"] * df["PBE"]
    df["Raises_PBE_interaction"] = df["Raises (%)"] * df["PBE"]

    X = df[["Avg Earnings", "PBE", "Earnings_PBE_interaction", "Raises_PBE_interaction", "Intercept"]]
    y = df["Win Rate (%)"]
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results:\n", model.summary())

def cohen_d(x, y):
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate standard deviations
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    
    # Calculate sample sizes
    n_x = len(x)
    n_y = len(y)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n_x - 1) * std_x**2 + (n_y - 1) * std_y**2) / (n_x + n_y - 2))
    
    # Calculate Cohen's d
    d = (mean_x - mean_y) / pooled_std
    return d


def cohen_d(a, b):
    """Compute Cohen's d for independent samples."""
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

def print_results(results, earnings, avg_earnings_per_round, label=""):
    """ Print formatted table of poker simulation results. """
    player_results = []
    
    for player_name, (raise_pct, fold_pct, win_pct) in results.items():
        avg_earnings = earnings.get(player_name, 0)  # Default earnings to 0 if not found
        avg_earnings_round = avg_earnings_per_round.get(player_name, 0)  # Default to 0 if not found
        player_results.append({
            'Player': player_name,
            'Raises': raise_pct,
            'Folds': fold_pct,
            'Win Rate': win_pct,
            'Total Earnings': avg_earnings,
            'Avg Earnings (Per Round)': avg_earnings_round
        })
    
    df = pd.DataFrame(player_results)
    print(f"\nPoker Simulation Results ({label}):\n", df)
    # Save the dataframe to a CSV file used for tests.py
    csv_filename = f"poker_simulation_results_{label}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename



def plot_results(results, title="Poker Simulation Results"):
    """ Generate a figure with raise/fold percentages and win rates. """
    df = pd.DataFrame(results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    players = df.index
    raises, folds, win_rates = df["Raises (%)"], df["Folds (%)"], df["Win Rate (%)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(players))
    axes[0].bar(x, raises, width=0.4, label="Raises (%)", color="#A6192E", alpha=0.8)
    axes[0].bar(x + 0.4, folds, width=0.4, label="Folds (%)", color="#D6D6D6", alpha=0.8)
    axes[0].set_xticks(x + 0.2)
    axes[0].set_xticklabels(players, rotation=45, ha="right")
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Raise and Fold Percentages by Player Type")
    axes[0].legend()
    sns.lineplot(x=players, y=win_rates, marker="o", color="#A6192E", linewidth=2.5, ax=axes[1])
    axes[1].set_title("Win Rate by Player Type")
    axes[1].set_ylabel("Win Rate (%)")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Custom color palette
copenhagen_red = "#A6192E"

def plot_earnings(earnings_bne, earnings_pbe, title="Average Earnings by Player Type"):
    """ Plot bar chart for earnings comparison between BNE and PBE. """
    names_bne = list(earnings_bne.keys())
    values_bne = list(earnings_bne.values())
    names_pbe = list(earnings_pbe.keys())
    values_pbe = list(earnings_pbe.values())

    # Combine both datasets for plotting
    df_bne = pd.DataFrame({'Player': names_bne, 'Earnings': values_bne, 'Type': 'BNE'})
    df_pbe = pd.DataFrame({'Player': names_pbe, 'Earnings': values_pbe, 'Type': 'PBE'})
    df = pd.concat([df_bne, df_pbe])

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Player', y='Earnings', hue='Type', data=df, palette=[copenhagen_red, "#A5ACAF"])
    plt.title(title)
    plt.ylabel("Average Chips Earned")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Dummy data setup: player types and simulated earnings
np.random.seed(42)
data = {
    'player_type': ['rational'] * 100 + ['risk_averse'] * 100 + ['risk_seeking'] * 100 + ['loss_averse'] * 100 + ['optimistic'] * 100,
    'earnings': (
        np.random.normal(10, 5, 100).tolist() +
        np.random.normal(5, 5, 100).tolist() +
        np.random.normal(8, 5, 100).tolist() +
        np.random.normal(8, 5, 100).tolist() +
        np.random.normal(8, 5, 100).tolist()
    )
}

df = pd.DataFrame(data)

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    bootstraps = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    lower = np.percentile(bootstraps, (100 - ci) / 2)
    upper = np.percentile(bootstraps, 100 - ((100 - ci) / 2))
    return lower, upper


# Example usage with a DataFrame `df` grouped by player type
for ptype in df['player_type'].unique():
    earnings = df[df['player_type'] == ptype]['earnings'].values
    lower, upper = bootstrap_ci(earnings)
    print(f"{ptype}: 95% CI for mean earnings = ({lower:.2f}, {upper:.2f})")
    print(f"{ptype}: mean = {np.mean(earnings):.2f}, std = {np.std(earnings):.2f}, n = {len(earnings)}")

def anova_player_earnings(df, player_col='player_type', value_col='earnings'):
    model = ols(f"{value_col} ~ C({player_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA Results:")
    print(anova_table)


if __name__ == "__main__":
    start_time = time.time()

    player_types = ['Risk Averse', 'Risk Seeking', 'Loss Averse', 'Optimistic', 'Rational']
    bne_data = {ptype: [] for ptype in player_types}
    pbe_data = {ptype: [] for ptype in player_types}
    # Compute averages across all simulated earnings
    bne_avg = {ptype: np.mean(bne_data[ptype]) for ptype in player_types}
    pbe_avg = {ptype: np.mean(pbe_data[ptype]) for ptype in player_types}

    n_simulations = 10
    rounds_per_sim = 100

    
    earnings_by_model_and_player = {
        'BNE': defaultdict(list),
        'PBE': defaultdict(list)
    }


    all_player_stats = []  # For the big CSV with percentages + earnings. Also used for tests.py.

    for i in range(n_simulations):
        print(f"Simulation {i+1}/{n_simulations}", end='\r')

        # ----- BNE Simulation -----
        simulation_bne = simulate_poker_games(rounds_per_sim, belief_updating=False)
        bne_results, bne_observed, bne_earnings, bne_avg_earnings_per_round = compute_percentages(simulation_bne)

        for player_name in player_types:
            raise_pct, fold_pct, win_pct = bne_results.get(player_name, (0, 0, 0))
            total_earnings = bne_earnings.get(player_name, 0)
            avg_per_round = bne_avg_earnings_per_round.get(player_name, 0)

            earnings_by_model_and_player['BNE'][player_name].append(total_earnings)

            all_player_stats.append({
                'Simulation': i + 1,
                'Model': 'BNE',
                'Player': player_name,
                'Raises': raise_pct,
                'Folds': fold_pct,
                'Win Rate': win_pct,
                'Total Earnings': total_earnings,
                'Avg Earnings (Per Round)': avg_per_round
            })

        # ----- PBE Simulation -----
        simulation_pbe = simulate_poker_games(rounds_per_sim, belief_updating=True)
        pbe_results, pbe_observed, pbe_earnings, pbe_avg_earnings_per_round = compute_percentages(simulation_pbe)

        for player_name in player_types:
            raise_pct, fold_pct, win_pct = pbe_results.get(player_name, (0, 0, 0))
            total_earnings = pbe_earnings.get(player_name, 0)
            avg_per_round = pbe_avg_earnings_per_round.get(player_name, 0)

            earnings_by_model_and_player['PBE'][player_name].append(total_earnings)

            all_player_stats.append({
                'Simulation': i + 1,
                'Model': 'PBE',
                'Player': player_name,
                'Raises': raise_pct,
                'Folds': fold_pct,
                'Win Rate': win_pct,
                'Total Earnings': total_earnings,
                'Avg Earnings (Per Round)': avg_per_round
            })

    # Save full player stats to CSV
    player_stats_df = pd.DataFrame(all_player_stats)
    player_stats_df.to_csv("player_stats_per_simulation.csv", index=False)

    print("\nSaved full player stats to 'player_stats_per_simulation.csv'")

    # Save raw earnings separately for bootstraps or t-tests
    raw_earnings_records = []
    for model in ['BNE', 'PBE']:
        for player, earning_list in earnings_by_model_and_player[model].items():
            for earning in earning_list:
                raw_earnings_records.append({
                    'Model': model,
                    'Player': player,
                    'Total Earnings': total_earnings,
                    'Avg Earnings (Per Round)': avg_per_round
                })

    raw_earnings_df = pd.DataFrame(raw_earnings_records)
    raw_earnings_df.to_csv("earnings_raw_per_simulation.csv", index=False)

    print("Saved raw earnings to 'earnings_raw_per_simulation.csv'")
    print(f"Completed in {round(time.time() - start_time, 2)} seconds.")

 

    
    combined_player_data = []

    # Use bne_earnings and pbe_earnings to get the earnings data
    for strategy_label, results, earnings_dict in [("BNE", bne_results, bne_earnings), ("PBE", pbe_results, pbe_earnings)]:
        for player_name, (raise_pct, fold_pct, win_pct) in results.items():
            avg_earnings = earnings_dict.get(player_name, 0)
            combined_player_data.append({
                "Player": player_name,
                "Strategy": strategy_label,
                "Raises (%)": raise_pct,
                "Folds (%)": fold_pct,
                "Win Rate (%)": win_pct,
                "Avg Earnings": avg_earnings
            })

    combined_df = pd.DataFrame(combined_player_data)
    combined_df = combined_df.dropna()

    #Chi-Square Test for Raise/Fold Distributions Players
    chi_square_test(bne_observed, label="BNE")
    compute_confidence_intervals(simulation_bne, 100, label="BNE")

    chi_square_test(pbe_observed, label="PBE")
    compute_confidence_intervals(simulation_pbe, 100, label="PBE")
    # ANOVA for Player Differences in Raising Frequency
    player_types = {}
    for player in bne_results:
        if player not in player_types:
            player_types[player] = []
        player_types[player].append(bne_results[player][0])
        player_types[player].append(pbe_results[player][0])
    
    f_stat, anova_p_value = stats.f_oneway(*player_types.values())
    print(f"ANOVA for Player Type Differences: F-stat={f_stat:.4f}, p-value={anova_p_value:.4f}")

    # Tukey's HSD for Player Comparisons
    all_raises = []
    labels = []
    for player, raises in player_types.items():
        all_raises.extend(raises)
        labels.extend([player] * len(raises))
    tukey_results = pairwise_tukeyhsd(all_raises, labels, alpha=0.05)
    print("Tukey HSD Test:")
    print(tukey_results)

    player_types_earnings = {}
    for player in bne_results:
        if player not in player_types_earnings:
            player_types_earnings[player] = []
        player_types_earnings[player].extend(bne_results[player])  # Earnings from BNE
        player_types_earnings[player].extend(pbe_results[player])  # Earnings from PBE

    # Run ANOVA
    f_stat, anova_p_value = stats.f_oneway(*player_types_earnings.values())
    print(f"ANOVA for Player Type Differences in Earnings: F-stat={f_stat:.4f}, p-value={anova_p_value:.4f}")

    # Tukey's HSD for Earnings
    all_earnings = []
    labels = []
    for player, earnings in player_types_earnings.items():
        all_earnings.extend(earnings)
        labels.extend([player] * len(earnings))

    tukey_results = pairwise_tukeyhsd(all_earnings, labels, alpha=0.05)
    print("Tukey HSD Test for Earnings:")
    print(tukey_results)

    # Goodness-of-Fit Chi-Square for CPT Predictions
    expected_raises = [bne_results[player][0] for player in bne_results]
    observed_raises = [pbe_results[player][0] for player in pbe_results]
    #expected_raises = np.array(expected_raises) * (sum(observed_raises) / sum(expected_raises))
    if sum(expected_raises) > 0:
        expected_raises = np.array(expected_raises) * (sum(observed_raises) / sum(expected_raises))
    else:
        expected_raises = np.array(expected_raises)  # Avoid division by zero

    chi2_gof_stat, chi2_gof_p_value = stats.chisquare(observed_raises, f_exp=expected_raises)
    print(f"Goodness-of-Fit Chi-Square for CPT Predictions: chi2={chi2_gof_stat:.4f}, p-value={chi2_gof_p_value:.4f}")

    chi2_stat, chi2_p_value = stats.chisquare(bne_observed.sum(axis=0), f_exp=pbe_observed.sum(axis=0))
    print(f"Chi-Square Test for Raise/Fold: chi2={chi2_stat:.4f}, p-value={chi2_p_value:.4f}")


    # Plot Results for Tables
    print_results(bne_results, bne_earnings, bne_avg_earnings_per_round, label="BNE")
    plot_results(bne_results, title="BNE Results")

    print_results(pbe_results, pbe_earnings, pbe_avg_earnings_per_round, label="PBE")
    plot_results(pbe_results, title="PBE Results")


    plot_earnings(bne_earnings, pbe_earnings, title="Average Earnings Comparison (BNE vs PBE)")

    # Cohen's d for Earnings
    print("\nCohen's d for BNE Earnings Between Players:")
    for p1, p2 in combinations(bne_results.keys(), 2):
        d = cohen_d(bne_results[p1], bne_results[p2])
        print(f"{p1} vs {p2}: d={d:.3f}")

    print("\nCohen's d for PBE Earnings Between Players:")
    for p1, p2 in combinations(pbe_results.keys(), 2):
        d = cohen_d(pbe_results[p1], pbe_results[p2])
        print(f"{p1} vs {p2}: d={d:.3f}")

    # Time
    end_time = time.time()
    print(f"\nTotal Simulation Time: {end_time - start_time:.2f} seconds")