"""Tests for the Stable Set Game engine."""

import pytest
import networkx as nx

from stable_set_game.engine import StableSetGame


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def path3() -> nx.Graph:
    """Path graph P_3: 0 -- 1 -- 2."""
    return nx.path_graph(3)


@pytest.fixture
def triangle() -> nx.Graph:
    """Complete graph K_3 (triangle): 0 -- 1 -- 2 -- 0."""
    return nx.complete_graph(3)


@pytest.fixture
def single_node() -> nx.Graph:
    """A graph with a single isolated vertex."""
    g = nx.Graph()
    g.add_node(0)
    return g


@pytest.fixture
def two_isolated() -> nx.Graph:
    """Two isolated vertices: 0  1 (no edges)."""
    g = nx.Graph()
    g.add_nodes_from([0, 1])
    return g


# ======================================================================
# Initialisation
# ======================================================================

class TestInit:
    """Tests for game initialisation."""

    def test_empty_graph_raises(self):
        """Creating a game on an empty graph should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            StableSetGame(nx.Graph())

    def test_initial_state(self, path3):
        """A fresh game starts with player 1, no selections, empty history."""
        game = StableSetGame(path3)
        assert game.current_player == 1
        assert game.selected == set()
        assert game.history == []

    def test_graph_is_copied(self, path3):
        """Mutating the original graph should not affect the game."""
        game = StableSetGame(path3)
        path3.add_node(99)
        assert 99 not in game.graph.nodes


# ======================================================================
# Legal moves
# ======================================================================

class TestLegalMoves:
    """Tests for legal_moves() and is_legal_move()."""

    def test_all_nodes_legal_at_start(self, path3):
        """At the start all vertices should be legal."""
        game = StableSetGame(path3)
        assert game.legal_moves() == [0, 1, 2]

    def test_selecting_middle_blocks_neighbours(self, path3):
        """Selecting vertex 1 in P_3 blocks 0 and 2 (its neighbours)."""
        game = StableSetGame(path3)
        game.make_move(1)
        assert game.legal_moves() == []

    def test_selecting_end_blocks_neighbour(self, path3):
        """Selecting vertex 0 in P_3 blocks only vertex 1."""
        game = StableSetGame(path3)
        game.make_move(0)
        assert game.legal_moves() == [2]

    def test_is_legal_move(self, path3):
        game = StableSetGame(path3)
        game.make_move(0)
        assert game.is_legal_move(2) is True
        assert game.is_legal_move(0) is False  # already selected
        assert game.is_legal_move(1) is False  # adjacent to 0

    def test_triangle_one_move_ends(self, triangle):
        """In K_3, any single selection blocks all others."""
        game = StableSetGame(triangle)
        game.make_move(0)
        assert game.legal_moves() == []

    def test_isolated_nodes_all_selectable(self, two_isolated):
        """Isolated vertices never block each other."""
        game = StableSetGame(two_isolated)
        game.make_move(0)
        assert game.legal_moves() == [1]


# ======================================================================
# Game over / winner / loser
# ======================================================================

class TestGameOver:
    """Tests for win/loss detection."""

    def test_not_over_at_start(self, path3):
        game = StableSetGame(path3)
        assert game.is_game_over() is False
        assert game.get_winner() is None
        assert game.get_loser() is None

    def test_single_node_game(self, single_node):
        """On a single-node graph, player 1 picks it, player 2 loses."""
        game = StableSetGame(single_node)
        game.make_move(0)
        assert game.is_game_over() is True
        assert game.get_winner() == 1
        assert game.get_loser() == 2

    def test_triangle_player1_wins(self, triangle):
        """In K_3, player 1 picks any vertex → player 2 has no moves → P1 wins."""
        game = StableSetGame(triangle)
        game.make_move(0)
        assert game.get_winner() == 1
        assert game.get_loser() == 2

    def test_path3_player2_wins(self, path3):
        """P_3: P1 picks 0, P2 picks 2 → no moves left → P2 wins."""
        game = StableSetGame(path3)
        game.make_move(0)  # player 1
        game.make_move(2)  # player 2
        assert game.is_game_over() is True
        assert game.get_winner() == 2
        assert game.get_loser() == 1

    def test_path3_player1_wins_by_centre(self, path3):
        """P_3: P1 picks 1 (centre) → blocks everything → P1 wins."""
        game = StableSetGame(path3)
        game.make_move(1)
        assert game.get_winner() == 1


# ======================================================================
# make_move validation
# ======================================================================

class TestMakeMove:
    """Tests for move validation."""

    def test_illegal_move_raises(self, path3):
        game = StableSetGame(path3)
        game.make_move(0)
        with pytest.raises(ValueError, match="not a legal move"):
            game.make_move(1)  # adjacent to 0

    def test_move_after_game_over_raises(self, triangle):
        game = StableSetGame(triangle)
        game.make_move(0)
        with pytest.raises(ValueError, match="already over"):
            game.make_move(1)

    def test_player_alternates(self, path3):
        game = StableSetGame(path3)
        assert game.current_player == 1
        game.make_move(0)
        assert game.current_player == 2
        game.make_move(2)
        assert game.current_player == 1


# ======================================================================
# Undo
# ======================================================================

class TestUndo:
    """Tests for undo_move()."""

    def test_undo_restores_state(self, path3):
        game = StableSetGame(path3)
        game.make_move(0)
        player, vertex = game.undo_move()
        assert player == 1
        assert vertex == 0
        assert game.current_player == 1
        assert game.selected == set()
        assert game.legal_moves() == [0, 1, 2]

    def test_undo_empty_raises(self, path3):
        game = StableSetGame(path3)
        with pytest.raises(ValueError, match="No moves to undo"):
            game.undo_move()

    def test_undo_multiple(self, path3):
        game = StableSetGame(path3)
        game.make_move(0)
        game.make_move(2)
        game.undo_move()
        assert game.current_player == 2
        assert game.legal_moves() == [2]
        game.undo_move()
        assert game.current_player == 1
        assert game.legal_moves() == [0, 1, 2]


# ======================================================================
# Copy
# ======================================================================

class TestCopy:
    """Tests for the copy() deep-copy method."""

    def test_copy_is_independent(self, path3):
        game = StableSetGame(path3)
        game.make_move(0)
        clone = game.copy()
        clone.make_move(2)
        # Original should be unaffected
        assert game.current_player == 2
        assert 2 not in game.selected


# ======================================================================
# Larger graph: path of length 5
# ======================================================================

class TestPath5:
    """Regression-style tests on a slightly larger graph P_5."""

    @pytest.fixture
    def path5(self) -> nx.Graph:
        return nx.path_graph(5)  # 0 -- 1 -- 2 -- 3 -- 4

    def test_full_game(self, path5):
        """Play out a complete game on P_5 and check the result."""
        game = StableSetGame(path5)
        game.make_move(0)  # P1 — blocks 1
        game.make_move(2)  # P2 — blocks 1, 3
        game.make_move(4)  # P1 — blocks 3
        # No legal moves remain
        assert game.is_game_over() is True
        assert game.get_winner() == 1
