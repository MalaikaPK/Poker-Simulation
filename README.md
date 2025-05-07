
# Understanding Poker Decisions Through the Lens of Cumulative Prospect Theory

Bachelor Thesis - Malaika Pohl Khader

Reprository for simulating a Limit Texas Hold'em Poker game for five player types under Bayesian Nash Equilibrium (BNE) and Perfect Bayesian Nash Equilibrium (PBE), evaluated with Cumulative Prospect Theory and statistical analysis. The following is an overview of the moodules in the project.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/MalaikaPK/poker-sim.git
cd poker-sim/poker-2.0
pip install -r requirements.txt
`

## Requirements
Python 3.12


## Project Structure

### 1. `cards.py`
The `cards.py` module provides foundational classes for representing playing cards, decks, and hands. It defines three key components: 
- **Card Class**: Represents individual playing cards with a suit (e.g., Spades, Hearts) and rank (e.g., Ace, 2, 3… King).
- **Deck Class**: Constructs and manipulates a 52-card deck, supporting actions like shuffling and dealing.
- **Hand Class**: Represents a players hand, inheriting from the Deck class, and tracks cards drawn from the deck.

This module abstracts the card manipulation logic, providing a clean structure for the game environment.

---

### 2. `player.py`
The `player.py` module defines the `PokerPlayer` class and implements decision-making based on Cumulative Prospect Theory (CPT) alongside Bayesian Nash Equilibrium (BNE) and Perfect Bayesian Equilibrium (PBE). 
The module simulates a poker player’s decision-making, incorporating Monte Carlo simulations to estimate win probabilities, updating beliefs about opponents strategies, and choosing actions based on CPT-weighted expected utility. 
The belief updating system reflects beliefs based on observed actions and hands, where the priors are drawn from a probability distribution based on previous monte-carlo simulations of hand class probabilities with 10,000,000 iterations. 
In order to estimate P(a=raise|H,I), I have used a logistic function to approximate the likelihood that a player would raise given the win rate.

---

### 3. `poker_cards.py`
The `poker_cards.py` module extends the base card framework to suit poker-specific hand evaluation. It introduces:
- **PokerCard**: A subclass of `Card`, tailored for poker hands.
- **PokerHand**: A subclass of `Hand`, which implements the logic for evaluating and comparing poker hands.
- **PokerDeck**: A subclass of `Deck`, generating `PokerCard` instances for use in games.

This module handles the rules for determining hand rankings in poker.

---

### 4. `poker_game.py`
The `poker_game.py` module governs the flow of a single poker game round. It defines the `PokerGame` class and manages various game elements:
- It initialises the game with players and a poker deck.
- Deals cards in three stages (Flop, Turn, and River), simulating player decisions (raise or fold).
- Tracks game outcomes, including updating player beliefs based on opponent actions, and determining the winner based on the best hand.

This module serves as the core structure for running a poker game in the simulation.

---

### 5. `poker_simulation.py`
The `poker_simulation.py` module is responsible for executing multiple poker game simulations and collecting results. It runs poker games under BNE and PBE assumptions, comparing player strategies.
Furthernore, it initialises diverse players with distinct CPT parameters and aggregates statistics on game outcomes, including total raises, folds, and wins.

This module is key to the experimental framework for testing poker strategies under uncertainty. To run the poker simulation, execute `python poker_simulation.py`. The simulation will run with the specified settings.

---

### 6. `chi_square_test.py`
The `chi_square_test.py` module handles the statistical analysis of the poker simulation results. It conducts several tests, including:
- **Chi-Square Tests**: Evaluates the independence of raise/fold distributions across player types and between BNE and PBE.
- **Statistical Analysis**: Runs t-tests, ANOVA, and Tukey HSD tests to examine differences in player behavior.
- **Visualisations**: Generates bar charts and line plots to illustrate raise/fold percentages and win rates for each player type.

This module is crucial for drawing insights from the simulation data and understanding the strategic behavior of players. To run the statistical tests, execute `python chi_square_test.py`. The simulation will run with the specified settings from the `poker_simulation.py` module as well as the tests from the `chi_square_test.py` module.



## License Information

The following modules are adapted from open-source projects and are licensed under the **MIT License**:

1. **`cards.py`**  
   Adapted from [Allen Downeys "Think Python"](https://github.com/AllenDowney/ThinkPython).  
   License: MIT License  
   Copyright (c) 2017 Allen Downey.  
   You may freely use, copy, modify, and distribute the software, provided the copyright and license are included in all copies or substantial portions of the software.
   
    'The MIT License

        Copyright (c) 2017 Allen Downey

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.'


2. **`pypokerengine`**  
   The `pypokerengine` library is also licensed under the MIT License.  
   License: MIT License  
   Copyright (c) 2017-2025.  
   As with `cards.py`, the software can be freely used, modified, and distributed, provided the copyright and license are included.