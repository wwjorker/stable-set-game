"""
Parameterised evaluation functions for the Stable Set Game.

When the minimax search cannot reach a terminal state within its depth
limit, it relies on a heuristic evaluation function to estimate how
favourable the current position is for the maximising player.

Each feature function takes a game state and returns a numeric score.
The ``WeightedEvaluator`` combines several features via a linear
combination with configurable weights; the self-play framework is used
to compare different weight settings.

Convention:
    - Positive scores favour the **maximising** player (the player whose
      turn it is when the AI begins its search — always the AI itself).
    - Negative scores favour the opponent.
    - Terminal win  → +∞  (represented by ``SCORE_WIN``).
    - Terminal loss → −∞  (represented by ``SCORE_LOSS``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict

import networkx as nx

from stable_set_game.engine import StableSetGame


# Sentinel scores for terminal states
SCORE_WIN: float = math.inf
SCORE_LOSS: float = -math.inf
SCORE_DRAW: float = 0.0  # never happens in Node Kayles, kept for generality


# ======================================================================
# Individual feature functions
# ======================================================================
# Each feature is computed from the perspective of the *current* player
# (the one about to move).  The minimax wrapper negates appropriately.

def feature_move_difference(game: StableSetGame) -> float:
    """Number of legal moves available to the current player.

    A simple but effective heuristic: more available moves generally
    means more flexibility and a better position.
    """
    return float(len(game.legal_moves()))


def feature_blocked_ratio(game: StableSetGame) -> float:
    """Fraction of remaining (unselected, unblocked) vertices that are
    adjacent to at least one selected vertex, normalised to [0, 1].

    Higher values mean the board is more constrained — potentially
    bad for the player who must move next.  We return the negative
    so that a *higher* blocked ratio yields a *lower* score.
    """
    moves = game.legal_moves()
    total_free = len([v for v in game.graph.nodes if v not in game.selected])
    if total_free == 0:
        return 0.0
    return -(1.0 - len(moves) / total_free)


def feature_degree_sum(game: StableSetGame) -> float:
    """Sum of degrees (in the original graph) of all legal-move vertices.

    Vertices with higher degree block more neighbours when selected,
    so a position where legal moves have high total degree gives the
    current player more powerful options.
    """
    return float(sum(game.graph.degree(v) for v in game.legal_moves()))


def feature_move_parity(game: StableSetGame) -> float:
    """Parity of the number of legal moves.

    In many combinatorial games, the parity of remaining moves
    determines the winner under optimal play.  Returns +1.0 if the
    count is odd (current player can potentially take the last move)
    and -1.0 if even.
    """
    return 1.0 if len(game.legal_moves()) % 2 == 1 else -1.0


def feature_component_count(game: StableSetGame) -> float:
    """Number of connected components of the subgraph induced by the
    currently *available* vertices (legal moves).

    Intuition: when the remaining playing area splits into several
    independent components, the game decomposes into a disjoint sum of
    smaller sub-games (in the Sprague–Grundy sense).  More components
    therefore means a more fragmented, often more tractable position.
    Isolated available vertices also count as single-vertex components,
    so a position with many isolated moves scores high.

    Returns 0.0 when no moves are available (terminal position).
    """
    available = game.legal_moves()
    if not available:
        return 0.0
    subgraph = game.graph.subgraph(available)
    return float(nx.number_connected_components(subgraph))


# Registry of built-in features (name → function)
BUILTIN_FEATURES: Dict[str, Callable[[StableSetGame], float]] = {
    "move_count": feature_move_difference,
    "blocked_ratio": feature_blocked_ratio,
    "degree_sum": feature_degree_sum,
    "move_parity": feature_move_parity,
    "component_count": feature_component_count,
}


# ======================================================================
# Weighted evaluator
# ======================================================================

@dataclass
class WeightedEvaluator:
    """Combines multiple feature functions via a weighted linear sum.

    Attributes:
        weights: Mapping from feature name to its scalar weight.
                 Only features present in this dict are evaluated.

    Example::

        evaluator = WeightedEvaluator(weights={
            "move_count":   1.0,
            "blocked_ratio": 0.5,
            "move_parity":  2.0,
        })
        score = evaluator(game)
    """

    weights: Dict[str, float] = field(default_factory=lambda: {
        "move_count": 1.0,
        "move_parity": 0.5,
    })

    def __call__(self, game: StableSetGame) -> float:
        """Evaluate the game state.

        Returns:
            ``SCORE_LOSS`` (−∞) if the current player has no legal moves;
            otherwise the weighted sum of the active features.
        """
        # Terminal check
        if game.is_game_over():
            # Current player cannot move → they lose
            return SCORE_LOSS

        score = 0.0
        for name, weight in self.weights.items():
            feature_fn = BUILTIN_FEATURES[name]
            score += weight * feature_fn(game)
        return score
