from collections import defaultdict
import cards
from pypokerengine.engine.card import Card as PyPokerCard

class PokerCard(cards.Card):
    def __eq__(self, other):
        return self.rank == other.rank

    def __lt__(self, other):
        return self.rank < other.rank
   
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def to_pypokercard(self):
        return PyPokerCard(2**(self.suit+1), self.rank)
    
class PokerHand(cards.Hand):
    cards: list[PokerCard]
    _hand_ranking_functions = [
        "is_royal_flush", 
        "is_straight_flush", 
        "is_four_of_a_kind", 
        "is_full_house", 
        "is_flush", 
        "is_straight", 
        "is_three_of_a_kind", 
        "is_two_pair", 
        "is_pair"]
    
    def __eq__(self, other):
        return set(self.cards) == set(other.cards)

    def __gt__(self, other):
        
        if self == other:
            return False
        
        self.cards.sort(reverse=True)
        other.cards.sort(reverse=True)

        # Precompute rankings
        self_rankings = [getattr(self, func)() for func in self._hand_ranking_functions]
        other_rankings = [getattr(other, func)() for func in self._hand_ranking_functions]

        # Find the highest ranking hand for each
        self_best_rank, self_best_hand = max(self_rankings, key=lambda x: x[0])
        other_best_rank, other_best_hand = max(other_rankings, key=lambda x: x[0])

        # Compare hand rankings
        if self_best_rank != other_best_rank:
            return self_best_rank > other_best_rank

        # Compare on tiebreaker rules if both hands have the same rank
        match self_best_rank:
            case 6:  # Full house 
                # Compare the three of a kind cards first, then the pair cards
                return self_best_hand.cards[0] > other_best_hand.cards[0] or self_best_hand.cards[-1] > other_best_hand.cards[-1]
            case 5:  # Flush
                # Compare the highest cards of each flush
                return self.has_highest_card(other_best_hand)
            case 4:  # Straight
                # Compare the highest card of each straight
                return self_best_hand.cards[0] > other_best_hand.cards[0]
            case 3:  # Three of a kind
                # Compare the three of a kind cards first, then the highest kicker (cards from the original hands)
                return self_best_hand.cards[0] > other_best_hand.cards[0] or self.has_highest_card(other)
            case 2:  # Two pair
                # Compare the higher pair first, then the lower pair, then the kicker (cards from the original hands)
                return self_best_hand.cards[0] > other_best_hand.cards[0] or self_best_hand.cards[2] > other_best_hand.cards[2] or self.has_highest_card(other)
            case 1:  # Pair
                # Compare the pair cards first, then the kickers (cards from the original hands)
                return self_best_hand.cards[0] > other_best_hand.cards[0] or self.has_highest_card(other)
            case _:  # Highest card tiebreaker
                return self.has_highest_card(other)
            
    def get_hand_category(self) -> str:
        """
        Returns the string name of the best hand category.
        E.g., "Pair", "Flush", "Full House", etc.
        """
        for func_name in self._hand_ranking_functions:
            rank, _ = getattr(self, func_name)()
            if rank > 0:
                return func_name.replace("is_", "").replace("_", " ").title()
        return "High Card"

         
        
    def __add__(self, other):
        """Returns a hand with the cards from this hand and the other."""
        hand = PokerHand()
        hand.cards = self.cards + other.cards
        return hand
    
    def __sub__(self, other):
        """Returns a hand with the cards from this hand that are not in the other."""
        hand = PokerHand()
        hand.cards = [card for card in self.cards if card not in other.cards]
        return hand
    
    def is_royal_flush(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 5:
            _is_straight_flush, straight_flush_hand = self.is_straight_flush()
            if _is_straight_flush and straight_flush_hand.cards[0].rank == 14:  # Rank 14 = Ace
                return 9, straight_flush_hand 
        return 0, None

    def is_straight_flush(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 5:
            _is_straight, _ = self.is_straight()
            if _is_straight:
                _is_flush, straight_flush_hand = self.is_flush()
                if _is_flush:
                    return 8, straight_flush_hand
        return 0, None

    def is_four_of_a_kind(self) -> tuple[int, 'PokerHand']:
        assert len(self.cards) == 7
        # If there are four cards of the same rank, then the fourth card (index 3) must be part of the four of a kind
        four_of_a_kind_cards = [card for card in self.cards if card.rank == self.cards[3].rank]
        if len(four_of_a_kind_cards) == 4:
            four_of_a_kind_hand = PokerHand()
            four_of_a_kind_hand.cards = four_of_a_kind_cards
            return 7, four_of_a_kind_hand
        return 0, None

    def is_full_house(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 5:
            _is_three_of_a_kind, three_of_a_kind_hand = self.is_three_of_a_kind()
            if _is_three_of_a_kind:
                _is_pair, pair_hand = (self - three_of_a_kind_hand).is_pair()
                if _is_pair:
                    return 6, three_of_a_kind_hand + pair_hand
        return 0, None

    def is_flush(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 5:
            # Group cards by suit
            suit_counts = defaultdict(list)
            for card in self.cards:
                suit_counts[card.suit].append(card)
            
            # Check if any suit has 5 or more cards
            for _, suited_cards in suit_counts.items():
                if len(suited_cards) >= 5:
                    flush_hand = PokerHand()
                    # Only keep the top 5 cards of the flush
                    flush_hand.cards = sorted(suited_cards, reverse=True)[:5]
                    return 5, flush_hand
        return 0, None

    def is_straight(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 5:
            # For each card, check if the next four cards have decreasing ranks
            for idx in range(len(self.cards)-4):
                for i in range(1, 5):
                    if self.cards[idx].rank != self.cards[idx+i].rank + 1:
                        break
                else:  # If the inner loop completes without breaking, then all cards are in a straight
                    straight_cards = self.cards[idx:idx+5]
                    straight_hand = PokerHand()
                    straight_hand.cards = straight_cards
                    return 4, straight_hand
        return 0, None

    def is_three_of_a_kind(self) -> tuple[int, 'PokerHand']:
        assert len(self.cards) == 7
        # If there are three cards of the same rank, then the third card (index 2) or the fifth card (index 4) must be part of the three of a kind
        card_group_1 = [card for card in self.cards if card.rank == self.cards[2].rank]
        card_group_2 = [card for card in self.cards if card.rank == self.cards[4].rank]
        # Choose the first group with 3 or more cards (group 1 has priority because it contains the higher ranked cards)
        three_of_a_kind_cards = card_group_1 if len(card_group_1) >= 3 else card_group_2 if len(card_group_2) >= 3 else []
        if three_of_a_kind_cards:
            three_of_a_kind_hand = PokerHand()
            three_of_a_kind_hand.cards = three_of_a_kind_cards[:3]
            return 3, three_of_a_kind_hand

        return 0, None

    def is_two_pair(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 4:
            is_pair1, high_pair = self.is_pair()
            if is_pair1:
                is_pair2, low_pair = (self - high_pair).is_pair()
                if is_pair2:
                    return 2, high_pair + low_pair
        return 0, None

    def is_pair(self) -> tuple[int, 'PokerHand']:
        if len(self.cards) >= 2:
            for idx, card in enumerate(self.cards[:-1]):
                if card.rank == self.cards[idx+1].rank:
                    pair_hand = PokerHand()
                    pair_hand.cards = [card, self.cards[idx+1]]
                    return 1, pair_hand
        return 0, None
    
    def has_highest_card(self, other) -> bool:
        """ Returns True if self has a higher card than other. """
        return self.cards > other.cards
    
        
class PokerDeck(cards.Deck):
    cards: list[PokerCard]
    
    def create_card(self, suit, rank):
        return PokerCard(suit, rank)
