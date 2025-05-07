import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from poker_simulation import simulate_poker_games, init_game

num_games = 100  # Number of simulations

# Initialize the game
game = init_game()

# Run the poker simulation
sim_result = simulate_poker_games(game, num_games)

# Print simulation results
sim_result.print_results()

# Store results in a list for visualization
results_data = []

for player, (raises, folds, wins) in sim_result.results.items():
    results_data.append({
        "Player": player.name,
        "Raises %": round(raises / (raises + folds) * 100, 1),
        "Folds %": round(folds / (raises + folds) * 100, 1),
        "Wins %": round(wins / (raises + folds) * 100, 1),
    })

# Convert to DataFrame
df = pd.DataFrame(results_data)

# Ensure data exists
if df.empty:
    print("No data to visualize.")
else:
    # 💖 PINK COLOR PALETTE 💖
    pink_palette = ["#ff69b4", "#ff1493", "#db7093"]

    # 📊 Visualization 1: Win % by Player Type
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Player", y="Wins %", data=df, palette=pink_palette)
    plt.title("Win Percentage by Player Type", fontsize=14)
    plt.xlabel("Player Type", fontsize=12)
    plt.ylabel("Win Percentage", fontsize=12)
    plt.xticks(rotation=45)
    plt.show(block=False)  # Show without blocking execution
    plt.pause(0.1)  # Ensure the second plot appears

    # 📊 Visualization 2: Raises vs. Folds
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Raises %", y="Folds %", hue="Player", data=df, palette=pink_palette, s=100, alpha=0.7)
    plt.title("Raises vs. Folds by Player Type", fontsize=14)
    plt.xlabel("Raises (%)", fontsize=12)
    plt.ylabel("Folds (%)", fontsize=12)
    plt.legend(title="Player Type")
    plt.show()
