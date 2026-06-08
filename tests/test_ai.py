"""Tests for the AI player (minimax with alpha-beta pruning)."""

import math

import pytest
import networkx as nx

from stable_set_game.engine import StableSetGame
from stable_set_game.evaluation import WeightedEvaluator
from stable_set_game.ai import AIPlayer, SearchResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def exact_ai() -> AIPlayer:
    """AI with unlimited depth — solves the game exactly."""
    return AIPlayer(max_depth=math.inf, name="Exact")


@pytest.fixture
def shallow_ai() -> AIPlayer:
    """AI with depth-2 search — tests heuristic behaviour."""
    return AIPlayer(max_depth=2, name="Shallow")


# ======================================================================
# SearchResult basics
# ======================================================================

class TestSearchResult:

    def test_default_values(self):
        r = SearchResult()
        assert r.best_move is None
        assert r.score == 0.0
        assert r.nodes_explored == 0


# ======================================================================
# Exact solve on small graphs (unlimited depth)
# ======================================================================

class TestExactSolve:
    """On small graphs the AI should find the game-theoretic optimal move."""

    def test_single_node(self, exact_ai):
        """K_1: only one move, player 1 wins."""
        game = StableSetGame(nx.complete_graph(1))
        result = exact_ai.choose_move(game)
        assert result.best_move == 0
        assert result.score == math.inf  # guaranteed win

    def test_path3_centre_wins(self, exact_ai):
        """P_3: picking vertex 1 (centre) wins immediately for player 1."""
        game = StableSetGame(nx.path_graph(3))
        result = exact_ai.choose_move(game)
        assert result.best_move == 1
        assert result.score == math.inf

    def test_path4_player1_loses(self, exact_ai):
        """P_4 (0--1--2--3): player 1 is in a losing position.

        The maximum independent set of P_4 has size 2 (e.g. {0,2} or
        {1,3}).  Since two moves are made in total and player 2 makes
        the last one, player 1 always loses under optimal play.
        """
        game = StableSetGame(nx.path_graph(4))
        result = exact_ai.choose_move(game)
        assert result.score == -math.inf

    def test_triangle_any_move_wins(self, exact_ai):
        """K_3: any move wins for player 1 (all block everything)."""
        game = StableSetGame(nx.complete_graph(3))
        result = exact_ai.choose_move(game)
        assert result.best_move in [0, 1, 2]
        assert result.score == math.inf

    def test_cycle4_player1_loses(self, exact_ai):
        """C_4: player 1 is in a losing position.

        The maximum independent set of C_4 has size 2.  Both moves
        are consumed by player 2 last, so player 1 always loses.
        """
        game = StableSetGame(nx.cycle_graph(4))
        result = exact_ai.choose_move(game)
        assert result.score == -math.inf

    def test_path5_player1_wins(self, exact_ai):
        """P_5: player 1 can win by picking an endpoint."""
        game = StableSetGame(nx.path_graph(5))
        result = exact_ai.choose_move(game)
        assert result.score == math.inf

    def test_two_isolated_player2_wins(self, exact_ai):
        """Two isolated nodes: P1 picks one, P2 picks the other, P1 loses.

        So from P1's perspective, the best score is -inf (forced loss).
        """
        g = nx.Graph()
        g.add_nodes_from([0, 1])
        game = StableSetGame(g)
        result = exact_ai.choose_move(game)
        assert result.score == -math.inf


# ======================================================================
# Game is not modified by AI
# ======================================================================

class TestNoSideEffects:

    def test_game_unchanged_after_search(self, exact_ai):
        """The AI must not modify the game state passed to it."""
        game = StableSetGame(nx.path_graph(5))
        original_selected = game.selected.copy()
        original_player = game.current_player
        original_history = list(game.history)

        exact_ai.choose_move(game)

        assert game.selected == original_selected
        assert game.current_player == original_player
        assert game.history == original_history


# ======================================================================
# Depth-limited search
# ======================================================================

class TestDepthLimited:

    def test_depth_1_returns_a_legal_move(self):
        """Even at depth 1, the AI should return a valid move."""
        ai = AIPlayer(max_depth=1, name="Depth1")
        game = StableSetGame(nx.path_graph(5))
        result = ai.choose_move(game)
        assert result.best_move in game.legal_moves()

    def test_deeper_search_explores_more_nodes(self):
        """Deeper search should visit more nodes."""
        game = StableSetGame(nx.path_graph(6))
        shallow = AIPlayer(max_depth=2)
        deep = AIPlayer(max_depth=4)

        r_shallow = shallow.choose_move(game)
        r_deep = deep.choose_move(game)

        assert r_deep.nodes_explored > r_shallow.nodes_explored

    def test_shallow_still_finds_obvious_win(self):
        """On K_3, even depth-1 should find the winning move."""
        ai = AIPlayer(max_depth=1)
        game = StableSetGame(nx.complete_graph(3))
        result = ai.choose_move(game)
        assert result.best_move in [0, 1, 2]


# ======================================================================
# Move ordering
# ======================================================================

class TestMoveOrdering:

    def test_ordering_reduces_nodes(self):
        """Move ordering should generally reduce nodes explored
        (or at least not increase them significantly) via better pruning."""
        game = StableSetGame(nx.path_graph(7))

        ai_ordered = AIPlayer(max_depth=6, order_moves=True)
        ai_unordered = AIPlayer(max_depth=6, order_moves=False)

        r_ordered = ai_ordered.choose_move(game)
        r_unordered = ai_unordered.choose_move(game)

        # Both should find the same best score
        assert r_ordered.score == r_unordered.score

        # Ordered should explore fewer or equal nodes
        assert r_ordered.nodes_explored <= r_unordered.nodes_explored


# ======================================================================
# Custom evaluator
# ======================================================================

class TestCustomEvaluator:

    def test_different_weights_can_change_move(self):
        """Different evaluation weights may lead to different move choices
        at limited depth (demonstrating that the evaluator is parameterised)."""
        game = StableSetGame(nx.path_graph(8))

        ai_a = AIPlayer(
            max_depth=2,
            evaluator=WeightedEvaluator(weights={"move_count": 1.0}),
        )
        ai_b = AIPlayer(
            max_depth=2,
            evaluator=WeightedEvaluator(weights={"degree_sum": 1.0}),
        )

        r_a = ai_a.choose_move(game)
        r_b = ai_b.choose_move(game)

        # Both moves must be legal
        assert r_a.best_move in game.legal_moves()
        assert r_b.best_move in game.legal_moves()


# ======================================================================
# Error handling
# ======================================================================

class TestErrors:

    def test_no_moves_raises(self, exact_ai):
        """Calling choose_move on a finished game should raise."""
        game = StableSetGame(nx.complete_graph(3))
        game.make_move(0)
        with pytest.raises(ValueError, match="No legal moves"):
            exact_ai.choose_move(game)


# ======================================================================
# Full game simulation: AI vs AI
# ======================================================================

class TestAIvsAI:

    def test_full_game_terminates(self):
        """Two AIs playing each other must produce a valid winner."""
        game = StableSetGame(nx.path_graph(6))
        ai1 = AIPlayer(max_depth=4, name="P1-AI")
        ai2 = AIPlayer(max_depth=4, name="P2-AI")

        players = {1: ai1, 2: ai2}

        while not game.is_game_over():
            current_ai = players[game.current_player]
            result = current_ai.choose_move(game)
            game.make_move(result.best_move)

        assert game.get_winner() in [1, 2]
        assert len(game.history) >= 1

    def test_exact_vs_exact_on_path5(self):
        """Two exact AIs on P_5: the winner should be deterministic."""
        game = StableSetGame(nx.path_graph(5))
        ai = AIPlayer(max_depth=math.inf, name="Exact")

        winners = []
        for _ in range(3):  # repeat to verify determinism
            g = game.copy()
            while not g.is_game_over():
                result = ai.choose_move(g)
                g.make_move(result.best_move)
            winners.append(g.get_winner())

        # All runs should produce the same winner
        assert len(set(winners)) == 1
