import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import statsmodels.stats.proportion as smp
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
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


def run_regression_rational_risk(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Rational', 'Risk-Averse', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Rational", "Risk-Averse", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Rational Risk-Averse:\n", model.summary())

def run_regression_rational(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Rational', 'Raises', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Rational", "Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Rational:\n", model.summary())

def run_regression_loss_averse(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Loss-Averse', 'Raises', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Loss-Averse", "Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Loss-Averse:\n", model.summary())

def run_regression_risk_averse(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Risk-Averse', 'Raises', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Risk-Averse", "Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Risk-Averse:\n", model.summary())

def run_regression_risk_seeking(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Risk-Seeking', 'Raises', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Risk-Seeking", "Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Risk-Seeking:\n", model.summary())

def run_regression_optimistic(results):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Optimistic', 'Raises', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    X = df[["Optimistic", "Raises", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results Optimistic:\n", model.summary())

def run_regression_with_interactions(results):
    """
    Run a regression incorporating interaction terms to analyze player behaviors.
    """
    df = pd.DataFrame(results).T  # Transpose so player types are rows
    df.columns = ['Raises', 'Folds', 'Wins']  # Adjust based on your actual data structure
    df["Intercept"] = 1
    df = df.dropna()
    player_types = ['Risk Averse', 'Risk Seeking', 'Loss Averse', 'Optimistic', 'Rational']

    for player in player_types:
        df[player] = (df.index == player).astype(int)

    for player in player_types:
        df[f'{player}_Raises'] = df['Raises'] * df[player]
        df[f'{player}_Folds'] = df['Folds'] * df[player]
        df[f'{player}_Wins'] = df['Wins'] * df[player]

    y = df['Wins']
    X = df[
        ["Raises", "Intercept"] + 
        player_types + 
        [f'{player}_Raises' for player in player_types] + 
        #[f'{player}_Folds' for player in player_types] + 
        [f'{player}_Wins' for player in player_types]
    ]

    X = X.drop(columns=[f'{player}_Wins' for player in player_types], errors='ignore')

    # Ensure matrix is not singular
    X = X.loc[:, X.nunique() > 1]  # Drop columns with only one unique value


    # X = X.dropna()
    # y = y.loc[X.index]  # Align y with X

    # Run the regression
    model = sm.OLS(y, X).fit()
    print("\nRegression Results with Interactions:\n", model.summary())

def run_regression_interaction(results, is_pbe):
    """
    Run a regression to analyze how different player types influence win rates.
    """
    df = pd.DataFrame(results, index=['Raises', 'Folds', 'Wins']).T
    df["Intercept"] = 1  # Add intercept
    # Create an interaction term
    df["PBE"] = int(is_pbe)  # 1 if PBE, 0 if BNE
    df['Raises_PBE_interaction'] = df['Raises'] * df['PBE']
    X = df[["Raises", "PBE", "Raises_PBE_interaction", "Intercept"]]  # Independent variables
    y = df["Wins"]  # Dependent variable (win rate)
    
    model = sm.OLS(y, X).fit()
    print("\nRegression Results:\n", model.summary())

def run_regression_interaction_players(results, is_pbe):
    """
    Run a regression with interactions between PBE and player types.
    """
    df = pd.DataFrame(results).T  # Transpose so player types are rows
    df.columns = ['Raises', 'Folds', 'Wins']  # Adjust based on your actual data structure
    df["Intercept"] = 1
    df = df.dropna()
    df["PBE"] = int(is_pbe)  
    player_types = ["Risk Averse", "Risk Seeking", "Loss Averse", "Optimistic", "Rational"]
    
    for player in player_types:
        df[player] = (df.index == player).astype(int)

    # # Ensure all columns exist in df before creating interactions
    # for player in player_types:
    #     df[f'{player}_PBE'] = df['PBE'] * df[player]
    #     df[f'{player}_PBE_Raises'] = df[f'{player}_PBE'] * df['Raises']

    # X = df[
    #     ["Raises", "PBE", "Intercept"] + 
    #     player_types + 
    #     [f'{player}_PBE' for player in player_types] + 
    #     [f'{player}_PBE_Raises' for player in player_types]
    # ]
    y = df['Wins']
    X = df[["PBE", "Intercept"]]



    # Fit regression
    model = sm.OLS(y, X).fit()
    print("\nRegression Results BNE:\n", model.summary())



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

# def t_test_per_player(bne_results, pbe_results):
#     """ Perform t-tests for each player type based on raise percentages. """
#     for player in bne_results.keys():
#         bne_raises = bne_results[player][0]  # Extract raise counts
#         pbe_raises = pbe_results[player][0]  # Extract raise counts
        
#         # Debugging: Print the data to check its structure
#         print(f"Player: {player}")
#         print(f"BNE Raises: {bne_raises}")
#         print(f"PBE Raises: {pbe_raises}")
        
#         # Check if bne_raises and pbe_raises are lists or arrays
#         if not isinstance(bne_raises, (list, np.ndarray)) or not isinstance(pbe_raises, (list, np.ndarray)):
#             print(f"Error: Expected list or array, but got {type(bne_raises)} and {type(pbe_raises)} for player {player}.")
#             continue
        
#         # Ensure there are valid raise counts (i.e., no division by zero)
#         if len(bne_raises) == 0 or len(pbe_raises) == 0:
#             print(f"{player} has no raise data to test.")
#             continue
        
#         # Perform independent t-test for raise counts
#         t_stat, p_value = stats.ttest_ind(bne_raises, pbe_raises, nan_policy='omit', alternative="two-sided")
        
#         print(f"{player} Raise Count t-test: t={t_stat:.3f}, p={p_value:.3f}")


# def chi_square_test_per_player(bne_results, pbe_results):
#     """ Perform Chi-Square test for raise actions. """
#     for player in bne_results.keys():
#         # Ensure you're extracting the correct data (raises, folds)
#         bne_data = bne_results[player]
#         pbe_data = pbe_results[player]
        
#         # Check that there are exactly two values (raises, folds) for each simulation
#         if len(bne_data) != 2 or len(pbe_data) != 2:
#             print(f"Error: Incorrect data format for {player}. Expected 2 values (raises, folds), got {len(bne_data)} and {len(pbe_data)}.")
#             continue
        
#         bne_raises, bne_folds = bne_data
#         pbe_raises, pbe_folds = pbe_data
        
#         # Create contingency table for Chi-Square test: [raises, folds] for both BNE and PBE
#         observed = np.array([[bne_raises, bne_folds], [pbe_raises, pbe_folds]])
        
#         chi2_stat, p_val, dof, expected = stats.chi2_contingency(observed)
        
#         print(f"{player} Raise Chi-Square Test: Chi2 Stat={chi2_stat:.3f}, p={p_val:.3f}, dof={dof}")

# def f_test_per_player(bne_results, pbe_results):
#     """ Perform F-test for raise percentages. """
#     for player in bne_results.keys():
#         bne_raises_pct = np.array([bne_results[player][0] / (bne_results[player][0] + bne_results[player][1]) * 100])  # Calculate raise percentage
#         pbe_raises_pct = np.array([pbe_results[player][0] / (pbe_results[player][0] + pbe_results[player][1]) * 100])  # Calculate raise percentage
        
#         # Perform F-test for variance in raise percentages
#         f_stat, p_value = stats.f_oneway(bne_raises_pct, pbe_raises_pct)
        
#         print(f"{player} Raise F-test: F-statistic={f_stat:.3f}, P-value={p_value:.3f}")


def chi_square_test_per_player(bne_results, pbe_results):
    """ Perform Chi-Square tests for each player type separately. """
    for player in bne_results.keys():
        # Unpack raises, folds, and wins for both BNE and PBE
        bne_raises, bne_folds, bne_wins = bne_results[player]  # Assuming you have wins too
        pbe_raises, pbe_folds, pbe_wins = pbe_results[player]
        
        # Create a contingency table for Chi-Square test
        observed = np.array([[bne_raises, bne_folds], [pbe_raises, pbe_folds]])
        
        # Perform the Chi-Square test
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(observed)
        
        # Output the results for each player
        print(f"{player} Chi-Square Test: Chi2 Stat={chi2_stat:.3f}, p={p_val:.3f}, dof={dof}")



def f_test_for_results(bne_results, pbe_results):
    for player in bne_results.keys():
        # Get the win rates for the player from both BNE and PBE
        bne_win_rates = np.array(bne_results[player])  # Convert to NumPy array
        pbe_win_rates = np.array(pbe_results[player])  # Convert to NumPy array
        
        # Perform One-way ANOVA (F-test)
        f_stat, p_value = stats.f_oneway(bne_win_rates, pbe_win_rates)
        
        # Print the results for each player type
        print(f"{player} F-test: F-statistic={f_stat:.3f}, P-value={p_value:.3f}")


if __name__ == "__main__":
    start_time = time.time()
    print("\nRunning BNE Simulation...")
    simulation_bne = simulate_poker_games(100, belief_updating=False)
    bne_results, bne_observed = compute_percentages(simulation_bne)
    
    
    print("\nRunning PBE Simulation...")
    simulation_pbe = simulate_poker_games(100, belief_updating=True)
    pbe_results, pbe_observed = compute_percentages(simulation_pbe)
    
    
    t_test_per_player(bne_results, pbe_results)

    chi_square_test_per_player(bne_results, pbe_results)

    f_test_for_results(bne_results, pbe_results)

    # run_regression(bne_results)
    # run_regression(pbe_results)

    # run_regression_rational_risk(bne_results)
    # run_regression_rational_risk(pbe_results)

    # run_regression_interaction(bne_results, is_pbe=False)
    # run_regression_interaction(pbe_results, is_pbe=True)

    # run_regression_interaction_players(pbe_results, is_pbe=True)


    # run_regression_rational(bne_results)
    # run_regression_rational(pbe_results)

    # run_regression_loss_averse(bne_results)
    # run_regression_loss_averse(pbe_results)

    # run_regression_risk_averse(bne_results)
    # run_regression_risk_averse(pbe_results)

    # run_regression_risk_seeking(bne_results)
    # run_regression_risk_seeking(pbe_results)

    # run_regression_optimistic(bne_results)
    # run_regression_optimistic(pbe_results)


    # run_regression_with_interactions(bne_results)
    # run_regression_with_interactions(pbe_results)
