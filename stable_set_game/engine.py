"""
Game engine for the Stable Set Game (Node Kayles).

Rules:
    - Two players alternate turns on an undirected graph.
    - On each turn the current player selects a vertex that is NOT adjacent
      to any previously selected vertex (i.e. the selected set must remain
      an independent / stable set).
    - The player who cannot make a legal move loses.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Set

import networkx as nx


class StableSetGame:
    """Represents a single game of Stable Set (Node Kayles).

    Attributes:
        graph: The underlying undirected graph.
        selected: Set of vertices already chosen by players.
        current_player: 1 or 2, indicating whose turn it is.
        history: Ordered list of (player, vertex) moves made so far.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Initialise a new game on the given graph.

        Args:
            graph: An undirected NetworkX graph. Nodes can be any hashable
                   type, but integers are recommended.

        Raises:
            ValueError: If the graph is empty (has no vertices).
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot create a game on an empty graph.")

        # Store an independent copy so external mutations don't affect the game
        self.graph: nx.Graph = graph.copy()
        self.selected: Set = set()
        self.current_player: int = 1
        self.history: List[tuple[int, object]] = []

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def legal_moves(self) -> List:
        """Return a sorted list of vertices that the current player may select.

        A vertex *v* is legal iff:
        1. *v* has not been selected yet, AND
        2. *v* is not adjacent to any already-selected vertex.
        """
        blocked = set()
        for v in self.selected:
            blocked.add(v)
            blocked.update(self.graph.neighbors(v))

        return sorted(v for v in self.graph.nodes if v not in blocked)

    def is_legal_move(self, vertex) -> bool:
        """Check whether selecting *vertex* is a legal move."""
        return vertex in self.legal_moves()

    def is_game_over(self) -> bool:
        """Return True if no legal moves remain (current player loses)."""
        return len(self.legal_moves()) == 0

    def get_winner(self) -> Optional[int]:
        """Return the winning player (1 or 2), or None if the game is ongoing.

        The winner is the player who made the *last* move, because the
        current player is the one who cannot move.
        """
        if not self.is_game_over():
            return None
        # The other player wins
        return 2 if self.current_player == 1 else 1

    def get_loser(self) -> Optional[int]:
        """Return the losing player (1 or 2), or None if the game is ongoing."""
        if not self.is_game_over():
            return None
        return self.current_player

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def make_move(self, vertex) -> None:
        """Apply a move: current player selects *vertex*.

        Args:
            vertex: The vertex to select.

        Raises:
            ValueError: If the game is already over.
            ValueError: If *vertex* is not a legal move.
        """
        if self.is_game_over():
            raise ValueError("Game is already over — no moves can be made.")
        if not self.is_legal_move(vertex):
            raise ValueError(
                f"Vertex {vertex} is not a legal move for player "
                f"{self.current_player}."
            )

        self.selected.add(vertex)
        self.history.append((self.current_player, vertex))
        # Switch turns
        self.current_player = 2 if self.current_player == 1 else 1

    def undo_move(self) -> tuple[int, object]:
        """Undo the most recent move and return it as (player, vertex).

        Raises:
            ValueError: If there are no moves to undo.
        """
        if not self.history:
            raise ValueError("No moves to undo.")

        player, vertex = self.history.pop()
        self.selected.discard(vertex)
        self.current_player = player
        return player, vertex

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> "StableSetGame":
        """Return a deep copy of the current game state."""
        return deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"StableSetGame(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"selected={self.selected}, "
            f"current_player={self.current_player})"
        )
