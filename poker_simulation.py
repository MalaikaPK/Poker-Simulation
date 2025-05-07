from player import PokerPlayer, CPT
from poker_game import PokerGame
import time


class SimulationResult:
    """
    Represents the result of a poker simulation.
    """
    results: dict[PokerPlayer, tuple[int, int, int]]  # Player -> (Raises, Folds, Wins)
    earnings_per_round: dict[PokerPlayer, list[int]]


    def __init__(self):
        self.results = {}
        self.earnings: dict[PokerPlayer, int] = {}
        self.earnings_per_round: dict[PokerPlayer, list[int]] = {}  # Store per-round earnings

    def add_game_result(self, game_result: dict[PokerPlayer, tuple[int, int, int]], payoff_result: dict[PokerPlayer, int]):
        """
        Add the result of a poker game to the simulation results.
        """
        for player, (r, f, w) in game_result.items():
            if player not in self.results:
                self.results[player] = (r, f, w)
            else:
                (curr_r, curr_f, curr_w) = self.results[player]
                self.results[player] = (curr_r + r, curr_f + f, curr_w + w)

        for player, payoff in payoff_result.items():
            self.earnings[player] = self.earnings.get(player, 0) + payoff

            # Store earnings per round (used for average earnings per round calculation)
            if player not in self.earnings_per_round:
                self.earnings_per_round[player] = []
            self.earnings_per_round[player].append(payoff)
                
    def print_results(self):
        """
        Print the results of the simulation.
        <Player Name>: Raises: <Number of Raises>    Folds: <Number of Folds>    Wins: <Number of Wins>
        """
        longest_name_length = max(len(player.name) for player in self.results)
        print("\n".join(
            f"{player.name}:{" "*(longest_name_length - len(player.name))} \
              Raises: {round(raises/(raises+folds)*100, 1)} %\t \
            \tFolds: {round(folds/(raises+folds)*100, 1)} %\t \
            \tWins: {round(wins/(raises+folds)*100, 1)} %\t \
            Avg Earnings (Total): {round(self.earnings.get(player, 0), 2)} chips \
            \tAvg Earnings (Per Round): {round(sum(self.earnings_per_round[player]) / len(self.earnings_per_round[player]), 2)} chips"
            for player, (raises, folds, wins) in self.results.items()))
    
def simulate_poker_games(number_of_games: int, belief_updating=False) -> SimulationResult:
    """
    Simulate a number of poker games and return the results.
    """
    game = init_game(belief_updating)
    simulation_result = SimulationResult()
    for _ in range(number_of_games):
        game.play_game()
        game_result = game.generate_game_result()
        payoff_result = game.generate_payoff_result()
        simulation_result.add_game_result(game_result, payoff_result)
        #simulation_result.add_game_result(game.generate_game_result())

    # Add empirical priors from simulated games
    if belief_updating:
        priors_by_type = {}
        for ptype, hand_counts in game.hand_class_counts_by_type.items():
            total = sum(hand_counts.values())
            if total > 0:
                priors_by_type[ptype] = {
                    hand_class: count / total for hand_class, count in hand_counts.items()
                }

        # Assign to each player
        for player in game.players:
            player.opponent_hand_class_probs = {
                other.name: priors_by_type.get(other.name, {}) for other in game.players if other != player
            }
    return simulation_result

def init_game(belief_updating):
    # Define CPTs for different player types
    risk_averse_cpt = CPT(alpha=0.75, beta=0.85, gamma=0.70, delta=0.55, lambda_=2.25)
    risk_seeking_cpt = CPT(alpha=1.1, beta=1.1, gamma=0.70, delta=0.85, lambda_=1.2)
    loss_averse_cpt = CPT(alpha=0.88, beta=0.88, gamma=0.61, delta=0.69, lambda_=3.0)
    optimistic_cpt = CPT(alpha=0.9, beta=0.9, gamma=0.50, delta=0.85, lambda_=1.2)
    rational_cpt = CPT(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, lambda_=1.0, rational=True)  # Fully rational benchmark player

    players = [
        PokerPlayer("Risk Averse", risk_averse_cpt),
        PokerPlayer("Risk Seeking", risk_seeking_cpt),
        PokerPlayer("Loss Averse", loss_averse_cpt),
        PokerPlayer("Optimistic", optimistic_cpt),
        PokerPlayer("Rational", rational_cpt)
    ]

    # Create a poker game with the players
    return PokerGame(players, belief_updating)

    

if __name__ == '__main__':
    start_time = time.time()
    # Run the BNE version (fixed beliefs)
    print("\nRunning BNE Simulation...")
    bne_results = simulate_poker_games(1000, False)
    bne_results.print_results()

    # Run the PBE version (updated beliefs)
    print("\nRunning PBE Simulation...")
    pbe_results = simulate_poker_games(1000, True)
    pbe_results.print_results()

    end_time = time.time()
    print(f"\nTotal Simulation Time: {end_time - start_time:.2f} seconds")

