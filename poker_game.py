from poker_cards import PokerHand, PokerDeck
from player import PokerPlayer, hand_distributions


class PokerGame:
    """
    Represents a game of poker.
    """
    players: list[PokerPlayer]
    active_players: list[PokerPlayer]  # Players still in the game (have not folded)
    deck: PokerDeck
    table: PokerHand
    player_decisions: dict[PokerPlayer, list[str]]  # Track player decisions Player -> [decisions]
    winner: PokerPlayer
    player_hand_classes: dict[PokerPlayer, str] = {}



    def __init__(self, players: list[PokerPlayer], belief_updating):
        self.players: list[PokerPlayer] = players
        self._reset_game()
        self.belief_updating= belief_updating  
        self.hand_class_counts_by_type = {}
        self.player_contributions = {player: 0 for player in players}
        

        

    def _reset_game(self):
        """
        Reset the game state for a new round.
        """
        self.player_hand_classes = {}
        self.player_contributions = {player: 0 for player in self.players}
        self.deck = PokerDeck()
        self.table = PokerHand("table")
        self.active_players = self.players.copy()
        self.player_decisions = {}
        self.winner = None

        # Reset the opponent's prior belief to the original hand distributions
        for player in self.players:
            player.opponent_hand_class_probs = hand_distributions.copy()  # Reset the prior belief

    def _deal_hands(self):
        """
        Deal two cards to each player
        """
        for player in self.players:
            player.hand = PokerHand(player.name)
            self.deck.move_cards(player.hand, 2)

    def _deal_flop(self):
        """
        Remove one card from the deck (burn card) and deal three cards to the table (the flop).
        """
        self.deck.pop_card()
        self.deck.move_cards(self.table, 3)

    def _deal_turn(self):
        """
        Deal 4th card to the table (the turn):
        Remove one card from the deck (burn card) and deal one card to the table.
        """
        self.deck.pop_card()
        self.deck.move_cards(self.table, 1)

    def _deal_river(self):
        """
        Deal last card to the table (the river):
        Remove one card from the deck (burn card) and deal one card to the table.
        """
        self.deck.pop_card()
        self.deck.move_cards(self.table, 1)

    @staticmethod
    def get_current_bet(table):
        small_blind = 50
        big_blind = 100

        if len(table.cards) == 0:  # Pre-flop
            return big_blind
        elif len(table.cards) <= 3:  # Flop
            return big_blind
        else:  # Turn or River
            return big_blind * 2

    def _process_player_decisions(self):
        """
        Simulate decisions for each player.
        """
       
        # current_bet = 1000 if len(self.table.cards) <= 3 else 40
        current_bet = PokerGame.get_current_bet(self.table)
        for idx, player in enumerate(self.active_players):
            player.total_contribution = self.player_contributions[player]
            decision = player.make_decision(self.table, len(self.active_players), current_bet, self.belief_updating)

            # Store opponent actions for belief updating in PBE
            for other_player in self.active_players:
                if other_player != player:
                    other_player.opponent_actions.append(decision)

            # Save the decision
            self._save_player_decisions(player, decision)

            fold_idxs = []
            if decision == "Raise":
                self.player_contributions[player] += current_bet
            elif decision == "Fold":
                fold_idxs.append(idx)

         # Remove folded players after decisions are made
        for idx in reversed(fold_idxs):  # Iterate in reverse to avoid skipping players
            self.active_players.pop(idx)

            
    def _save_player_decisions(self, player: PokerPlayer, decision: str):
        if player not in self.player_decisions:
            self.player_decisions[player] = []
        self.player_decisions[player].append(decision)

    def generate_payoff_result(self) -> dict[PokerPlayer, int]:
        """
        Returns each player's net chip gain/loss from this hand.
        Positive = gain, Negative = loss.
        """
        return self.player_payoffs  # This must be tracked during play_game()


    def _process_winner(self):
        """
        Determine the winner of the game.
        """
        self.winner = max(self.active_players, key=lambda player: player.hand + self.table)

        
    def play_game(self):
        """
        Play a game of poker.
        """
        self._reset_game()
        self._deal_hands()
        self._deal_flop()
        self._process_player_decisions()
        self._deal_turn()
        self._process_player_decisions()
        self._deal_river()
        self._process_player_decisions()

         # Save final player hand categories
        for player in self.players:
            full_hand = player.hand + self.table
            hand_category = full_hand.get_hand_category()
            self.player_hand_classes[player] = hand_category

        self._process_winner()

        total_pot = sum(self.player_contributions.values())
        self.player_payoffs = {}

        for player in self.players:
            if player == self.winner:
                self.player_payoffs[player] = total_pot - self.player_contributions[player]
            else:
                self.player_payoffs[player] = -self.player_contributions[player]



            
    def generate_game_result(self) -> dict[PokerPlayer, tuple[int, int, int]]:
        """
        Generate the result of the game.
        :returns: A dictionary of player stats: Player -> (Raises, Folds, Wins)
        """
        game_result = {}
        for player in self.players:
            last_decision = self.player_decisions[player][-1]
            raises = int(last_decision == "Raise")
            folds = int(last_decision == "Fold")
            wins = int(player == self.winner)
            game_result[player] = (raises, folds, wins)
            
        return game_result
        
