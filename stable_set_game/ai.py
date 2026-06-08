"""
AI player for the Stable Set Game.

Implements minimax search with alpha-beta pruning and a configurable
evaluation function.  The ``AIPlayer`` class encapsulates all parameters
needed for one AI configuration, making it easy to pit different configs
against each other in the self-play framework.

Key design decisions
--------------------
* **Negamax formulation** — We use the negamax variant of minimax, where
  the score is always from the perspective of the *current* player.
  This eliminates the need for separate "max" and "min" branches.
* **In-place undo** — Instead of copying the game state at every node,
  we use ``make_move`` / ``undo_move`` for efficiency.
* **Move ordering** — Moves are optionally sorted by a quick heuristic
  (degree in the original graph, descending) to improve pruning.
* **Search statistics** — The ``SearchResult`` dataclass records the
  number of nodes explored, useful for performance analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from stable_set_game.engine import StableSetGame
from stable_set_game.evaluation import (
    SCORE_LOSS,
    WeightedEvaluator,
)


# ======================================================================
# Search result container
# ======================================================================

@dataclass
class SearchResult:
    """Stores the outcome of a minimax search.

    Attributes:
        best_move: The vertex the AI recommends (None only if no moves).
        score: The minimax score of the best move.
        nodes_explored: Total game-tree nodes visited during the search.
        depth_reached: The maximum depth actually searched.
    """
    best_move: Optional[object] = None
    score: float = 0.0
    nodes_explored: int = 0
    depth_reached: int = 0


# ======================================================================
# Core minimax with alpha-beta pruning (negamax formulation)
# ======================================================================

def _negamax(
    game: StableSetGame,
    depth: int,
    alpha: float,
    beta: float,
    evaluate: Callable[[StableSetGame], float],
    order_moves: bool,
    stats: Dict[str, int],
) -> float:
    """Negamax search with alpha-beta pruning.

    The score is always from the perspective of the **current** player.

    Args:
        game:        The current game state (mutated in place, then undone).
        depth:       Remaining search depth.
        alpha:       Best score the current player can guarantee so far.
        beta:        Best score the opponent can guarantee so far.
        evaluate:    Heuristic evaluation function.
        order_moves: If True, sort moves by vertex degree (descending)
                     before searching to improve pruning.
        stats:       Mutable dict for collecting ``nodes_explored``.

    Returns:
        The negamax score of this position.
    """
    stats["nodes"] += 1

    moves = game.legal_moves()

    # Terminal state: current player has no moves → they lose.
    if not moves:
        return SCORE_LOSS

    # Depth limit reached: fall back to heuristic evaluation.
    if depth == 0:
        return evaluate(game)

    # Optional move ordering: try high-degree vertices first, as they
    # tend to block more of the opponent's options.
    if order_moves:
        moves = sorted(moves, key=lambda v: game.graph.degree(v), reverse=True)

    best_score = -math.inf

    for move in moves:
        game.make_move(move)
        # Negamax: opponent's score is the negation of ours
        child_score = -_negamax(
            game, depth - 1, -beta, -alpha, evaluate, order_moves, stats
        )
        game.undo_move()

        if child_score > best_score:
            best_score = child_score

        alpha = max(alpha, best_score)
        if alpha >= beta:
            break  # beta cutoff — opponent won't allow this branch

    return best_score


# ======================================================================
# AI Player
# ======================================================================

@dataclass
class AIPlayer:
    """A configurable AI player for the Stable Set Game.

    Attributes:
        max_depth:   Maximum search depth (plies). Use ``math.inf`` for
                     unlimited depth (exact solve — only feasible on
                     small graphs).
        evaluator:   The heuristic evaluation function. Defaults to
                     a ``WeightedEvaluator`` with sensible weights.
        order_moves: Whether to sort moves by vertex degree before
                     searching, to improve alpha-beta pruning.
        name:        A human-readable label for this configuration.
    """

    max_depth: int | float = 10
    evaluator: WeightedEvaluator = field(default_factory=WeightedEvaluator)
    order_moves: bool = True
    name: str = "AI"

    def choose_move(self, game: StableSetGame) -> SearchResult:
        """Select the best move for the current player.

        Args:
            game: The current game state. It is **not** modified.

        Returns:
            A ``SearchResult`` containing the chosen move and search
            statistics.

        Raises:
            ValueError: If the game is already over (no legal moves).
        """
        moves = game.legal_moves()
        if not moves:
            raise ValueError("No legal moves — game is already over.")

        # Use in-place mutation on a copy to avoid side-effects
        working_copy = game.copy()
        stats: Dict[str, int] = {"nodes": 0}

        # Optionally order root moves
        if self.order_moves:
            moves = sorted(
                moves, key=lambda v: game.graph.degree(v), reverse=True
            )

        best_move = moves[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Determine effective depth (cap inf at a large int for the loop)
        effective_depth = (
            self.max_depth if math.isfinite(self.max_depth)
            else 1000
        )

        for move in moves:
            working_copy.make_move(move)
            score = -_negamax(
                working_copy,
                effective_depth - 1,
                -beta,
                -alpha,
                self.evaluator,
                self.order_moves,
                stats,
            )
            working_copy.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            # No pruning at root — we always want the best move

        return SearchResult(
            best_move=best_move,
            score=best_score,
            nodes_explored=stats["nodes"],
            depth_reached=effective_depth,
        )

    def __repr__(self) -> str:
        return (
            f"AIPlayer(name={self.name!r}, max_depth={self.max_depth}, "
            f"order_moves={self.order_moves})"
        )
