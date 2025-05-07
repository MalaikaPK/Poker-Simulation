import math
from poker_cards import PokerHand
from functools import lru_cache
from joblib import Parallel, delayed
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

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
        #self.belief_updating = belief_updating  # Flag for PBE
        self.opponent_actions = []  # Store opponent actions to update beliefs
        # self.belief_about_opponent = 0.5
        self.total_contribution = 0  # Will be updated by PokerGame each round
        self.active_players = active_players or []

       # Initialize opponent's prior beliefs from hand_distributions
        self.opponent_hand_class_probs = hand_distributions.copy()  # Set the priors directly from the hand_distributions

        # This will store the updated beliefs (posterior)
        self.belief_about_opponent = {}  # Posterior beliefs
        

    

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

        # Determine number of simulations based on the game stage
        nb_simulation = (
            500 if len(table.cards) == 0 else
            1_000 if len(table.cards) <= 3 else
            1_500
        )

        # Run Monte Carlo simulations in parallel
        win_prob = self.parallel_monte_carlo(nb_simulation, nb_player, hole_card, community_card)

        # Store in cache
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
        action_likelihood = 0.0

        for hand_class, prior_prob in self.belief_about_opponent.items():
            # Estimate win probability for this hand class
            win_prob = self.estimate_win_probability_for_hand_class(hand_class, table, nb_players)

            # Apply CPT distortion to win probability (if applicable)
            if self.cpt:
                win_prob = self.cpt.probability_weighting(win_prob, self.cpt.gamma)

            # Calculate action likelihood based on distorted win probability
            p_raise_given_class = self.raise_probability_given_winrate(win_prob)

            if observed_action == "Raise":
                likelihood = p_raise_given_class
            elif observed_action == "Fold":
                likelihood = 1 - p_raise_given_class
            else:
                continue  # skip unsupported actions

            # Weight by the prior probability of the hand class
            action_likelihood += prior_prob * likelihood

        return action_likelihood

    
    # Logistic function to map a probability of how good hand is, into an action likelihood
    def raise_probability_given_winrate(self, win_prob, k=10, threshold=0.5): 
        '''
        Rational Logistic raise probability based on objectiv win rates.
        k: steepness of decision threshold, higher implies sharper.
        threshold: win rates at indifference, at which raise probability = 0.5
        '''
        return 1 / (1 + math.exp(-k * (win_prob - threshold))) 
    
    def estimate_win_probability_for_hand_class(self, hand_class, table: PokerHand, nb_players): 
        return self.get_win_probability(table, nb_players)

    def update_beliefs(self, belief_updating, table: PokerHand, nb_players: int):
        if not belief_updating or not self.opponent_actions:
            return  # Skip belief updating if not in PBE or no observations

        # Get the most recent opponent action
        observed_action = self.opponent_actions[-1]  
        if observed_action not in {"Raise", "Fold"}:
            return  # Skip if unsupported action

        # Compute denominator of Bayes' Rule (normalizer)
        normalizer = self.compute_action_likelihood(observed_action, table, nb_players)
        if normalizer == 0:
            return  # Avoid division by zero

        new_probs = {}
        for hand_class, prior_prob in self.opponent_hand_class_probs.items():
            # Estimate win probability for this hand class
            win_prob = self.estimate_win_probability_for_hand_class(hand_class, table, nb_players)

            # Apply CPT distortion if needed
            if self.cpt:
                win_prob = self.cpt.probability_weighting(win_prob, self.cpt.gamma)

            # Estimate likelihood of the observed action given this hand class
            if observed_action == "Raise":
                likelihood = self.raise_probability_given_winrate(win_prob)
            elif observed_action == "Fold":
                likelihood = 1 - self.raise_probability_given_winrate(win_prob)

            # Bayesian update
            posterior = (likelihood * prior_prob) / (normalizer + 1e-10)
            new_probs[hand_class] = posterior

        # Normalize the posterior probabilities
        total_prob = sum(new_probs.values())
        for hand_class in new_probs:
            new_probs[hand_class] /= total_prob

        # Update belief distribution
        self.belief_about_opponent = new_probs





    # def update_beliefs(self, belief_updating, table):
    #     if not belief_updating or not self.opponent_actions:
    #         return  # Skip belief updating if not in PBE or no observations
        
    #     total_actions = len(self.opponent_actions)
    #     raise_count = sum(1 for action in self.opponent_actions if action == "Raise")
    #     fold_count = sum(1 for action in self.opponent_actions if action == "Fold")

    #     if total_actions == 0:
    #         return  # Avoid division by zero

    #     # Estimate likelihoods based on observed frequencies
    #     likelihood_raise = (raise_count + 1) / (total_actions + 2)
    #     likelihood_fold = (fold_count + 1) / (total_actions + 2)


    #     # # Apply CPT probability weighting to distort perceived likelihoods
    #     weighted_likelihood_raise = self.cpt.probability_weighting(likelihood_raise, self.cpt.gamma)
    #     weighted_likelihood_fold = self.cpt.probability_weighting(likelihood_fold, self.cpt.delta)

    #     # Use stored prior or initialize at 0.5 if it's the first round
    #     prior = getattr(self, "belief_about_opponent", 0.5)

    #     # Apply distorted Bayesian rule
    #     posterior_numerator = weighted_likelihood_raise * prior
    #     posterior_denominator = (weighted_likelihood_raise * prior) + (weighted_likelihood_fold * (1 - prior))


    #     # # Apply Bayes' rule
    #     # posterior_numerator = likelihood_raise * prior
    #     # posterior_denominator = (likelihood_raise * prior) + (likelihood_fold * (1 - prior))

    #     if posterior_denominator > 0:
    #         self.belief_about_opponent = posterior_numerator / posterior_denominator
    #     else:
    #         self.belief_about_opponent = prior  # Avoid NaN issues



    # def update_beliefs(self):
    #     """ Updates beliefs based on observed opponent actions using Bayes' rule. """
    #     if not self.belief_updating or not self.opponent_actions:
    #         return  # Skip belief updating if not in PBE or no observations
        
    #     # Count how often opponents raised
    #     raise_count = sum(1 for action in self.opponent_actions if action == "Raise")
    #     fold_count = sum(1 for action in self.opponent_actions if action == "Fold")

    #     # Compute updated probability of opponent holding a strong hand
    #     prior = 0.5  # Initial belief (neutral prior, can be improved)
    #     likelihood_raise = 0.7  # Probability that a strong hand raises
    #     likelihood_fold = 0.3   # Probability that a weak hand folds

    #     # Apply Bayes' rule
    #     updated_belief = (likelihood_raise * prior) / (
    #         likelihood_raise * prior + likelihood_fold * (1 - prior)
    #     )

    #     # Store the new belief as the probability that an opponent is strong
    #     self.belief_about_opponent = updated_belief
            
    def make_decision(self, table: PokerHand, nb_player, current_bet, belief_updating) -> str:
        if belief_updating:
            self.update_beliefs(belief_updating, table, nb_player)

        win_probability = self.get_win_probability(table, nb_player)
        
        
        # total_bet = 0
        # small_blind = 50
        # big_blind = 100
        # total_bet = small_blind+ big_blind

        # # Determine stage and set betting amount accordingly
        # if len(table.cards) == 0:  # Pre-flop
        #     current_bet = big_blind
        # elif len(table.cards) <= 3:  # Flop
        #     current_bet = big_blind * 1.5
        # else:  # Turn or River
        #     current_bet = big_blind * 3

        # total_bet += current_bet

        # # Pot includes everyone's bet
        # total_pot = total_bet * nb_player

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
            # utility_fold= win_probability*utility_loss

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
        # utility_fold = -potential_loss
        utility_fold = 0
        # utility_fold = utility_loss

        # print("Weighted win prob:", weighted_win_prob)
        # print("Weighted loss prob:", weighted_loss_prob)
        # print("Utility gain:", utility_gain)
        # print("Utility loss:", utility_loss)
        # print("Expected utility of raise:", expected_utility_raise)
        # print("Utility of fold:", utility_fold)

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
        
