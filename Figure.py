import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def value_function(x, alpha, beta, lambda_):
    """ CPT value function for gains and losses """
    return np.where(x >= 0, x**alpha, -lambda_ * (-x)**beta)

def probability_weighting(p, param):
    """ CPT probability weighting function with parameter gamma or delta """
    return (p**param) / ((p**param + (1 - p)**param)**(1 / param))

# Parameters (as you defined)
tversky_params = {"alpha": 0.88, "beta": 0.88, "lambda_": 2.5, "gamma": 0.65}

players_params = {
    "Risk-Averse": {"alpha": 0.75, "beta": 0.85, "lambda_": 2.25, "gamma": 0.70, "delta": 0.55},
    "Risk-Seeking": {"alpha": 1.1, "beta": 1.1, "lambda_": 1.2, "gamma": 0.70, "delta": 0.85},
    "Loss-Averse": {"alpha": 0.88, "beta": 0.88, "lambda_": 3.0, "gamma": 0.61, "delta": 0.69},
    "Optimistic": {"alpha": 0.9, "beta": 0.9, "lambda_": 1.2, "gamma": 0.5, "delta": 0.85},
    "Rational": {"alpha": 1.0, "beta": 1.00, "lambda_": 1.00, "gamma": 1.00},
}

x_values = np.linspace(-10, 10, 500)
p_values = np.linspace(0, 1, 500)

def plot_cpt_functions(title, alpha, beta, lambda_, gamma, delta=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Value function plot
    v_values = value_function(x_values, alpha, beta, lambda_)
    axs[0].plot(x_values, v_values, color="#A6192E",
                label=f"$\\alpha={alpha}$, $\\beta={beta}$, $\\lambda={lambda_}$")
    axs[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axs[0].set_title("CPT Value Function", fontsize=14)
    axs[0].set_xlabel("Outcome ($x$)", fontsize=12)
    axs[0].set_ylabel("Value ($v(x)$)", fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Probability weighting plot
    w_gain = probability_weighting(p_values, gamma)
    axs[1].plot(p_values, w_gain, label=f"Gain weighting ($\\gamma={gamma}$)", color="#A6192E")
    
    if delta is not None:
        w_loss = probability_weighting(p_values, delta)
        axs[1].plot(p_values, w_loss, label=f"Loss weighting ($\\delta={delta}$)", color="#0E7C7B")
    
    axs[1].plot(p_values, p_values, color="#D6D6D6", linestyle="--", label="Linear weighting")
    axs[1].set_title("CPT Probability Weighting Function", fontsize=14)
    axs[1].set_xlabel("Probability ($p$)", fontsize=12)
    axs[1].set_ylabel("Weighted Probability ($w(p)$)", fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Plot general Tversky & Kahneman parameters
figures = []
figures.append(plot_cpt_functions("General CPT Functions (Tversky & Kahneman)", **tversky_params, delta=None))

# Plot each player, including delta where present
for player, params in players_params.items():
    delta = params.get("delta", None)
    figures.append(plot_cpt_functions(f"{player} Player CPT Functions", **params))


for fig in figures:
    plt.show()



import numpy as np

# Define the CPT functions
def probability_weighting(p, gamma):
    return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1/gamma))

def value_function(x, alpha, beta, lambda_):
    if x >= 0:
        return x**alpha
    else:
        return -lambda_ * ((-x)**beta)

# Define player parameters
risk_averse_params = {
    "alpha": 0.75,
    "beta": 0.85,
    "gamma": 0.70,
    "delta": 0.55,
    "lambda_": 2.25,
    "rational": False
}

rational_params = {
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "delta": 1.0,
    "lambda_": 1.0,
    "rational": True
}

# Setup scenario: 100/200 betting, win probability 0.7 (strong hand)
win_prob = 0.7
potential_gain = 200  # chips won
potential_loss = 100  # chips risked

# Define decision function
def cpt_decision(player_params, win_prob, gain, loss):
    if player_params["rational"]:
        utility_gain = gain
        utility_loss = -loss
        expected_utility_raise = win_prob * utility_gain + (1 - win_prob) * utility_loss
        decision = "Raise" if expected_utility_raise > 0 else "Fold"
        return expected_utility_raise, 0, decision

    weighted_win = probability_weighting(win_prob, player_params["gamma"])
    weighted_loss = probability_weighting(1 - win_prob, player_params["delta"])

    utility_gain = value_function(gain, player_params["alpha"], player_params["beta"], player_params["lambda_"])
    utility_loss = value_function(-loss, player_params["alpha"], player_params["beta"], player_params["lambda_"])

    expected_utility_raise = weighted_win * utility_gain + weighted_loss * utility_loss
    decision = "Raise" if expected_utility_raise > 0 else "Fold"
    return expected_utility_raise, 0, decision

# Compute for both players
risk_averse_result = cpt_decision(risk_averse_params, win_prob, potential_gain, potential_loss)
rational_result = cpt_decision(rational_params, win_prob, potential_gain, potential_loss)

risk_averse_result, rational_result

