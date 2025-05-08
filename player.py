import math
from poker_cards import PokerHand
from functools import lru_cache
from joblib import Parallel, delayed
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

# Priors drawn from a Monte Carlo Simulation
hand_distributions = {
    "highest card": 0.1766613,
    "pair": 0.4483240,
    "two pair": 0.2363353,
    "three of a kind": 0.0676977,
    "straight": 0.0318850,
    "flush": 0.0304791,
    "full house": 0.0067738,
    "four of a kind": 0.0016922,
    "straight flush": 0.0001516,
    "royal flush": 0.0000015}

class PokerPlayer:
    hand: PokerHand
    win_probability_cache={}

   
    def __init__(self, name: str, cpt: 'CPT', active_players=None):
        self.name = name
        self.cpt = cpt
        self.hand = PokerHand()
        self.decisions = {"Raise": 0, "Fold": 0}  
    
        self.opponent_actions = []  # Store opponent actions to update beliefs
        self.total_contribution = 0  # Will be updated by PokerGame each round
        self.active_players = active_players or []
        self.opponent_hand_class_probs = hand_distributions.copy()  # Set the priors directly from the hand_distributions
        self.belief_about_opponent = {}  # Store Posterior beliefs
        

    

    def get_win_probability(self, table: PokerHand, nb_player) -> float:
        """
        Calculate the player's win probability based on the table.
        This method will calculate the likelihood of winning the hand using CPT.
        """
        hole_card_tuple = tuple(sorted((c.rank, c.suit) for c in self.hand.cards))
        community_card_tuple = tuple(sorted((c.rank, c.suit) for c in table.cards))
        cache_key = (hole_card_tuple, community_card_tuple, nb_player)

        # Check if probability is already computed
        if cache_key in self.win_probability_cache:
            return self.win_probability_cache[cache_key]

        # Convert to pypokerengine card format
        hole_card = [c.to_pypokercard() for c in self.hand.cards]
        community_card = [c.to_pypokercard() for c in table.cards]

        # Determine number of simulations based on the game stage.
        nb_simulation = (
            500 if len(table.cards) == 0 else
            1_000 if len(table.cards) <= 3 else
            1_500
        )
        # Run Monte Carlo simulations in parallel to increase speed
        win_prob = self.parallel_monte_carlo(nb_simulation, nb_player, hole_card, community_card)

        # Store in cache to increase speed
        self.win_probability_cache[cache_key] = win_prob
        return win_prob


    @staticmethod
    def parallel_monte_carlo(nb_simulation, nb_player, hole_card, community_card):
        """
        Runs Monte Carlo simulations in parallel using joblib.
        """
        num_jobs = min(6, nb_simulation)  # Use up to 4 CPU cores
        num_sim_per_job = max(1, nb_simulation // num_jobs)  # Ensure at least 1 per job

        results = Parallel(n_jobs=num_jobs)(
            delayed(estimate_hole_card_win_rate)(num_sim_per_job, nb_player, hole_card, community_card)
            for _ in range(num_jobs)
        )

        return sum(results) / len(results)  # Compute average win probability
    

    def compute_action_likelihood(self, observed_action: str, table: PokerHand, nb_players: int) -> float:
        """
        Computes the denominator of the belief update equation:
        ∑_{H_{-i}'} w_i(P(a_{-i} | H_{-i}', I_i)) * μ_i(H_{-i}' | I_i)
        This is the total distorted likelihood of the observed action under all possible opponent hands,
        weighted by the player's prior belief over those hands.
        """
        
        action_likelihood = 0.0

        for hand_class, prior_prob in self.belief_about_opponent.items():
            # Estimate win probability for hand class, already presented by prior_prob
            # \mu_i(H_{-i}' | I_i): prior belief over opponent's hand class

            # Estimate P(win | H_{-i}', I_i)
            win_prob = self.estimate_win_probability_for_hand_class(hand_class, table, nb_players)

            # Apply CPT distortion to win probability, w_i(P(...))
            if self.cpt:
                win_prob = self.cpt.probability_weighting(win_prob, self.cpt.gamma)

            # P(a_{-i} | H_{-i}', I_i), Calculate action likelihood based on distorted win probability given opponent's type
            p_raise_given_class = self.raise_probability_given_winrate(win_prob)

            if observed_action == "Raise":
                likelihood = p_raise_given_class
            elif observed_action == "Fold":
                likelihood = 1 - p_raise_given_class
            else:
                continue  # skip unsupported actions

            # Accumulate likelihood, weighted by the prior probability of the hand class, \mu_i(H_{-i}'|I_i) * w_i(P(a_{-i}|H_{-i}', I_i))
            action_likelihood += prior_prob * likelihood

        # Return full denominator
        return action_likelihood

    
    # Logistic function to map a probability of how good hand is, into an action likelihood
    def raise_probability_given_winrate(self, win_prob, k=10, threshold=0.5): 
        '''
        P(a_{-i} = Raise | H_{-i}, I_i), Logistic raise probability based on objectiv win rates. 
        k: steepness of decision threshold, higher implies sharper.
        threshold: win rates at indifference, at which raise probability = 0.5
        '''
        return 1 / (1 + math.exp(-k * (win_prob - threshold))) 
    
    def estimate_win_probability_for_hand_class(self, hand_class, table: PokerHand, nb_players): 
        """
        Estimate P(win | H_{-i}, I_i): Objective win probability for the given hand class.
        """
        return self.get_win_probability(table, nb_players)

    def update_beliefs(self, belief_updating, table: PokerHand, nb_players: int):
        """
        Beliefs are updated using standard Bayesian formula with CPT distortion applied only to the likelihood of the observed action.
        Numerator: distorted likelihood of action given hand H_{-i} * prior over H_{-i}
        Denominator: sum over all H_{-i}' across all opponent hand classes
        Return posterior μ_i(H_{-i} | a_{-i}, I_i)

        This mirrors Equation 4.5 and 4.6 in my Thesis
        
        """
        if not belief_updating or not self.opponent_actions:
            return  # Skip belief updating if not in PBE or no observations

        # Get the most recent opponent action
        observed_action = self.opponent_actions[-1]  
        if observed_action not in {"Raise", "Fold"}:
            return  # Skip if unsupported action

        # Compute denominator of Bayes' Rule (normalizer): \sum w_i(P(a_{-i}|H_{-i}', I_i)) * \mu_i(H_{-i}'|I_i)
        normalizer = self.compute_action_likelihood(observed_action, table, nb_players)
        if normalizer == 0:
            return  # Avoid division by zero

        new_probs = {}
        for hand_class, prior_prob in self.opponent_hand_class_probs.items():
            # \mu_i(H_{-i} | I_i): prior belief over opponent hand class
            # Already given by `prior_prob`

            # Estimate win probability for this hand class, H_{-i}
            win_prob = self.estimate_win_probability_for_hand_class(hand_class, table, nb_players)

            # Apply CPT distortion to the likelihood: w_i(P(a_{-i} | H_{-i}, I_i))
            if self.cpt:
                win_prob = self.cpt.probability_weighting(win_prob, self.cpt.gamma)

            # Estimate likelihood of the observed action given this hand class: P(a_{-i} | H_{-i}, I_i), Modeled using a logistic function over win_prob
            if observed_action == "Raise":
                likelihood = self.raise_probability_given_winrate(win_prob)
            elif observed_action == "Fold":
                likelihood = 1 - self.raise_probability_given_winrate(win_prob)

            # Bayesian update. Numerator: w_i(P(a_{-i}|H_{-i}, I_i)) * \mu_i(H_{-i}|I_i)
            posterior = (likelihood * prior_prob) / (normalizer + 1e-10)
            new_probs[hand_class] = posterior

        # Normalise the posterior probabilities, \tilde\mu_i(...)
        total_prob = sum(new_probs.values())
        for hand_class in new_probs:
            new_probs[hand_class] /= total_prob

        # Update belief distribution, \tilde\mu_i(H_{-i}| I_i, a_{-i})
        self.belief_about_opponent = new_probs

            
    def make_decision(self, table: PokerHand, nb_player, current_bet, belief_updating) -> str:
        if belief_updating:
            self.update_beliefs(belief_updating, table, nb_player)

        win_probability = self.get_win_probability(table, nb_player)
        total_bet = current_bet  # Only bet for this round now

        # potential_loss = current_bet      # What you lose if you fold or lose after raising
        potential_gain = (current_bet * (nb_player - 1))  # What you win net of your own bet
        potential_loss = max(self.total_contribution, 1)  # total across all rounds


            
        decision = self.cpt.cpt_decision(win_probability, potential_gain, potential_loss)
        self.decisions[decision] += 1
        return decision


class CPT:
    def __init__(self, alpha: float, beta: float, gamma: float, delta: float, lambda_: float, rational: bool = False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_ = lambda_
        self.rational = rational

    # Simulate a poker player's decision to raise or fold
    def cpt_decision(self, win_probability, potential_gain, potential_loss):
        """
        player: dict containing player parameters (alpha, beta, lambda_, gamma, delta)
        win_probability: probability of winning the hand
        potential_gain: the monetary gain if the player wins
        potential_loss: the monetary loss if the player loses
        """
        if self.rational:
            # Rational decision-making (no probability distortion or value transformation)
            utility_gain = potential_gain
            utility_loss = -potential_loss
            expected_utility_raise = win_probability * utility_gain + (1 - win_probability) * utility_loss
            # utility_fold = -potential_loss
            utility_fold = 0

            decision = "Raise" if expected_utility_raise > utility_fold else "Fold"
            return decision
        
        
        #Biased decision-making using CPT
        # Calculate weighted probabilities
        weighted_win_prob = self.probability_weighting(win_probability, self.gamma)
        weighted_loss_prob = self.probability_weighting(1 - win_probability, self.delta)

        # Calculate CPT utilities
        utility_gain = self.value_function(potential_gain, self.alpha, self.beta, self.lambda_)
        utility_loss = self.value_function(-potential_loss, self.alpha, self.beta, self.lambda_)

        # Expected utility of raising
        expected_utility_raise = (weighted_win_prob * utility_gain) + (1-weighted_win_prob) * utility_loss
        

        
        # Utility of folding (status quo)
        utility_fold = 0
        # utility_fold = utility_loss

        # Decision based on higher utility
        decision = "Raise" if expected_utility_raise > utility_fold else "Fold"
        return decision
        
        
    @staticmethod
    def probability_weighting(p, gamma):
        """
        Implements the probability weighting function.
        """
        return (p**gamma) / ((p**gamma + (1 - p)**gamma)**(1 / gamma))
        
    
    @staticmethod
    def value_function(x, alpha, beta, lambda_):
        """
        Implements the value function for gains and losses.
        """
        if x >= 0:
            return x**alpha
        else:
            return -lambda_ * ((-x)**beta)
        
