"""This module contains a code example related to

Think Python, 2nd Edition
by Allen Downey
http://thinkpython2.com

Copyright 2015 Allen Downey

License: http://creativecommons.org/licenses/by/4.0/
"""

import random


class Card:
    """Represents a standard playing card.
    
    Attributes:
      suit: integer 0-3
      rank: integer 2-14
    """

    suit_names = ["Clubs", "Diamonds", "Hearts", "Spades"]
    rank_names = [None, None, "2", "3", "4", "5", "6", "7", 
              "8", "9", "10", "Jack", "Queen", "King", "Ace"]

    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank
        assert self.suit >= 0 and self.suit <= 3 and self.rank >= 2 and self.rank <= 14

    def __str__(self):
        """Returns a human-readable string representation."""
        return '%s of %s' % (Card.rank_names[self.rank],
                             Card.suit_names[self.suit])
    
    def __repr__(self):
        """Returns a formal string representation."""
        return str(self)

    def __eq__(self, other):
        """Checks whether self and other have the same rank and suit.

        returns: boolean
        """
        return self.suit == other.suit and self.rank == other.rank

    def __lt__(self, other):
        """Compares this card to other, first by suit, then rank.

        returns: boolean
        """
        t1 = self.suit, self.rank
        t2 = other.suit, other.rank
        return t1 < t2
                    
class Deck:
    """Represents a deck of cards.

    Attributes:
      cards: list of Card objects.
    """
    cards: list[Card]
    
    def __init__(self):
        """Initializes the Deck with 52 cards.
        """
        self.cards = []
        for suit in range(4):
            for rank in range(2, 15):
                card = self.create_card(suit, rank)
                self.cards.append(card)

    def __str__(self):
        """Returns a string representation of the deck.
        """
        return "Deck: " + str(self.cards)
    
    def __repr__(self):
        return str(self)
    
    def create_card(self, suit: int, rank: int):
        return Card(suit, rank)

    def add_card(self, card: Card):
        """Adds a card to the deck.

        card: Card
        """
        self.cards.append(card)

    def remove_card(self, card: Card):
        """Removes a card from the deck or raises exception if it is not there.
        
        card: Card
        """
        self.cards.remove(card)

    def pop_card(self, i: int = -1):
        """Removes and returns a card from the deck.

        i: index of the card to pop; by default, pops the last card.
        """
        return self.cards.pop(i)

    def sort(self, reverse: bool = False):
        """Sorts the cards."""
        self.cards.sort(reverse=reverse)

    def move_cards(self, hand, number_of_cards: int):
        """Moves the given number of random cards from the deck into the Hand.

        hand: destination Hand object
        num: integer number of cards to move
        """
        for _ in range(number_of_cards):
            hand.add_card(self.pop_card(random.randint(0, len(self.cards)-1)))
            

class Hand(Deck):
    """Represents a hand of playing cards."""
    
    def __init__(self, label=''):
        self.label = label
        self.cards: list[Card] = []
        
    def __str__(self):
        return "Hand (%s): %s" % (self.label, str(self.cards))

