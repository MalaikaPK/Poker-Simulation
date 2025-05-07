import cProfile
from poker_simulation import simulate_poker_games, init_game

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    game = init_game()
    simulate_poker_games(game, number_of_games=100_000)

    profiler.disable()

    # dump to profile file
    profiler.dump_stats("simulation.prof")