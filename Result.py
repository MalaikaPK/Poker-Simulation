import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import statsmodels.stats.proportion as smp
import statsmodels.api as sm
from poker_simulation import simulate_poker_games

def compute_percentages(simulation_result):
    """
    Convert raw counts into percentages using player names and extract Raise/Fold counts for Chi-Square test.
    """
    percentages = {}
    observed_counts = []
    for player_obj, (raises, folds, wins) in simulation_result.results.items():
        player_name = player_obj.name
        total_decisions = raises + folds
        raise_pct = (raises / total_decisions) * 100 if total_decisions else 0
        fold_pct = (folds / total_decisions) * 100 if total_decisions else 0
        win_pct = (wins / total_decisions) * 100 if total_decisions else 0
        percentages[player_name] = (raise_pct, fold_pct, win_pct)
        if raises > 0 or folds > 0:
            observed_counts.append([raises, folds])
    return percentages, np.array(observed_counts)

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

def print_results(results, label=""):
    """ Print formatted table of poker simulation results. """
    df = pd.DataFrame(results, index=['Raises (%)', 'Folds (%)', 'Win Rate (%)']).T
    print(f"\nPoker Simulation Results ({label}):\n", df)

def run_regression(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Raises', 'Folds', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results:\n", model.summary())

def run_regression_with_types(results, is_pbe):
    """
    Run regression with player types and PBE indicator.
    """
    df = pd.DataFrame(results, index=['Raises', 'Folds', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    df["PBE"] = int(is_pbe)  # 1 if PBE, 0 if BNE
    df["PlayerType"] = df.index  # Use player name as a categorical variable
    df = pd.get_dummies(df, columns=["PlayerType"], drop_first=True)  # Create dummies

    df = df.dropna()

    # Run OLS regression
    #X["Intercept"] = 1
    X = df[["Raises", "PBE", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results:\n", model.summary())

def run_regression_all(results):
    """
    Run a regression analyzing how Raise %, Fold %, and Player Type affect Win Rate.
    """
    df = pd.DataFrame(results, index=['Raises', 'Folds', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    #df["PBE"] = int(is_pbe)  # 1 if PBE, 0 if BNE
    df["PlayerType"] = df.index  # Use player name as a categorical variable
    df = pd.get_dummies(df, columns=["PlayerType"], drop_first=True)  # Create dummies

    df = df.dropna()

    X = df[["Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results (All Player Types):\n", model.summary())

def run_regression_with_interactions(results, is_pbe):
    """
    Run regression with player types and interaction effects with PBE indicator.
    """
    df = pd.DataFrame(results, index=['Raises', 'Folds', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    df["PBE"] = int(is_pbe)  # 1 if PBE, 0 if BNE
    df["PlayerType"] = df.index  # Use player name as a categorical variable
    df = pd.get_dummies(df, columns=["PlayerType"], drop_first=True)  # Create dummies

    df = df.dropna()

    # Interaction term: PlayerType * PBE
    for col in df.columns:
        if "PlayerType" in col:
            df[f"{col}_PBE"] = df[col] * df["PBE"]  
    
    X = df[["Raises", "Wins", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results with Interactions:\n", model.summary())





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

def t_test_comparison(bne_results, pbe_results):
    """ Perform t-tests comparing BNE and PBE results. """
    for metric, idx in zip(["Win Rate", "Raise %", "Fold %"], [2, 0, 1]):
        bne_vals = [bne_results[p][idx] for p in bne_results]
        pbe_vals = [pbe_results[p][idx] for p in pbe_results]
        t_stat, p_value = stats.ttest_ind(bne_vals, pbe_vals, alternative="two-sided")
        print(f"{metric} t-test: t={t_stat:.3f}, p={p_value:.3f}")

def t_test_per_player(bne_results, pbe_results):
    """ Perform t-tests for each player type separately. """
    for player in bne_results.keys():
        bne_win_rates = np.array(bne_results[player])  # Extract win rates as an array
        pbe_win_rates = np.array(pbe_results[player])  # Extract win rates as an array
        
        # Perform independent t-test for win rates
        t_stat, p_value = stats.ttest_ind(bne_win_rates, pbe_win_rates, nan_policy='omit', alternative="two-sided")
        
        print(f"{player} Win Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")


# def t_test_per_player(bne_results, pbe_results):
#     """ Perform t-tests for each player type separately. """
#     for player in bne_results.keys():
#         bne_win = bne_results[player]  # Array of win rates for BNE (not just a single value)
#         pbe_win = pbe_results[player]  # Array of win rates for PBE (not just a single value)
        
#         # Perform t-test on the arrays of win rates
#         t_stat, p_value = stats.ttest_ind(bne_win, pbe_win, nan_policy='omit', alternative="two-sided")  # Ensure NaN values are ignored
        
#         print(f"{player} Win Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")

# def t_test_per_player(bne_results, pbe_results):
#     """ Perform t-tests for each player type separately. """
#     for player in bne_results.keys():
#         bne_win = bne_results[player][2]  # Win rate for BNE
#         pbe_win = pbe_results[player][2]  # Win rate for PBE
#         t_stat, p_value = stats.ttest_ind([bne_win], [pbe_win])
#         print(f"{player} Win Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")

def t_test_rational_vs_others(results):
    """ Compare Rational player's win rate against others. """
    rational_win = results["Rational"][2]  # Win rate for Rational player
    other_wins = [results[p][2] for p in results if p != "Rational"]
    t_stat, p_value = stats.ttest_1samp(other_wins, rational_win, alternative="less")
    print(f"Rational vs Others Win Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")

def t_test_rational_vs_others_raise(results):
    """ Compare Rational player's raise rate against others. """
    rational_raise = results["Rational"][2]  # Win rate for Rational player
    other_raise = [results[p][2] for p in results if p != "Rational"]
    t_stat, p_value = stats.ttest_1samp(other_raise, rational_raise, alternative="less")
    print(f"Rational vs Others raise Rate t-test: t={t_stat:.3f}, p={p_value:.3f}")

if __name__ == "__main__":
    start_time = time.time()
    print("\nRunning BNE Simulation...")
    simulation_bne = simulate_poker_games(100, belief_updating=False)
    bne_results, bne_observed = compute_percentages(simulation_bne)
    print_results(bne_results, label="BNE")
    chi_square_test(bne_observed, label="BNE")
    compute_confidence_intervals(simulation_bne, 100, label="BNE")
    plot_results(bne_results, title="BNE Results")
    print("\nRunning PBE Simulation...")
    simulation_pbe = simulate_poker_games(100, belief_updating=True)
    pbe_results, pbe_observed = compute_percentages(simulation_pbe)
    print_results(pbe_results, label="PBE")
    chi_square_test(pbe_observed, label="PBE")
    compute_confidence_intervals(simulation_pbe, 100, label="PBE")
    plot_results(pbe_results, title="PBE Results")
    
    t_test_comparison(bne_results, pbe_results)
    t_test_per_player(bne_results, pbe_results)
    t_test_rational_vs_others(bne_results)
    t_test_rational_vs_others(pbe_results)

    t_test_rational_vs_others_raise(bne_results)
    t_test_rational_vs_others_raise(pbe_results)

    run_regression(bne_results)
    run_regression(pbe_results)

    # Run for both BNE and PBE
    run_regression_with_types(bne_results, is_pbe=False)
    run_regression_with_types(pbe_results, is_pbe=True)

    run_regression_all(bne_results)
    run_regression_all(pbe_results)

    run_regression_with_interactions(bne_results, is_pbe=False)
    run_regression_with_interactions(bne_results, is_pbe=True)


    end_time = time.time()
    print(f"\nTotal Simulation Time: {end_time - start_time:.2f} seconds")
