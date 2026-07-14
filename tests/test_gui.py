"""Tests for the GUI module.

These tests verify the *logic* of the GameGUI class (node colours,
status text, AI turn triggering, undo, restart) **without** opening
a Matplotlib window.  We use the non-interactive 'Agg' backend.
"""

import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no window pops up
import matplotlib.pyplot as plt

import pytest
import networkx as nx

from stable_set_game.ai import AIPlayer
from stable_set_game.engine import StableSetGame
from stable_set_game.gui import (
    GameConfig,
    GameGUI,
    GameLauncher,
    _COLOURS,
    build_game_graph,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def path5_hvh() -> GameGUI:
    """Human vs Human GUI on P_5 (no AI)."""
    return GameGUI(nx.path_graph(5), mode="human_vs_human")


@pytest.fixture
def path5_hva() -> GameGUI:
    """Human vs AI GUI on P_5."""
    ai = AIPlayer(max_depth=4, name="TestAI")
    return GameGUI(nx.path_graph(5), mode="human_vs_ai", ai2=ai)


@pytest.fixture
def triangle_ava() -> GameGUI:
    """AI vs AI GUI on K_3."""
    ai1 = AIPlayer(max_depth=math.inf, name="A1")
    ai2 = AIPlayer(max_depth=math.inf, name="A2")
    return GameGUI(nx.complete_graph(3), mode="ai_vs_ai", ai1=ai1, ai2=ai2)


def _setup_fig(gui: GameGUI) -> None:
    """Create figure and axes without calling plt.show()."""
    gui.fig, gui.ax = plt.subplots(1, 1, figsize=(6, 4))


# ======================================================================
# Initialisation
# ======================================================================

class TestInit:

    def test_valid_modes(self):
        g = nx.path_graph(3)
        for mode in ("human_vs_human", "human_vs_ai", "ai_vs_ai"):
            gui = GameGUI(g, mode=mode)
            assert gui.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            GameGUI(nx.path_graph(3), mode="invalid")

    def test_default_ai_created_for_human_vs_ai(self):
        gui = GameGUI(nx.path_graph(3), mode="human_vs_ai")
        assert gui.ai2 is not None
        assert gui.ai1 is None

    def test_default_ais_created_for_ai_vs_ai(self):
        gui = GameGUI(nx.path_graph(3), mode="ai_vs_ai")
        assert gui.ai1 is not None
        assert gui.ai2 is not None

    def test_positions_computed(self):
        gui = GameGUI(nx.path_graph(4), mode="human_vs_human")
        assert len(gui.pos) == 4


# ======================================================================
# Graphical setup screen and graph construction
# ======================================================================

class TestGameSetup:

    @pytest.mark.parametrize(
        "graph_type,n,expected_nodes",
        [
            ("Path", 7, 7),
            ("Cycle", 7, 7),
            ("Complete", 5, 5),
            ("Star", 5, 6),
            ("Erdos-Renyi", 8, 8),
            ("3-Regular", 8, 8),
            ("Fork", 7, 8),
        ],
    )
    def test_builds_every_graph_type(self, graph_type, n, expected_nodes):
        graph, label, layout = build_game_graph(
            GameConfig(graph_type=graph_type, n=n)
        )
        assert graph.number_of_nodes() == expected_nodes
        assert graph_type.split("-")[0] in label
        assert layout in ("spring", "circular")

    def test_rejects_invalid_regular_graph(self):
        with pytest.raises(ValueError, match="even n"):
            build_game_graph(GameConfig(graph_type="3-Regular", n=7))

    def test_rejects_excessive_depth(self):
        with pytest.raises(ValueError, match="depth"):
            build_game_graph(GameConfig(depth=13))

    def test_launcher_default_configuration(self):
        launcher = GameLauncher()
        launcher._build_screen()
        config = launcher._selected_config()
        assert config == GameConfig()

    def test_visible_game_controls_created(self, path5_hvh):
        path5_hvh.fig = plt.figure(figsize=(8, 6))
        path5_hvh.ax = path5_hvh.fig.add_axes([0.05, 0.16, 0.90, 0.76])
        path5_hvh._create_controls()
        assert set(path5_hvh._buttons) == {"restart", "undo", "next", "quit"}
        assert path5_hvh._buttons["next"].ax.get_visible() is False


# ======================================================================
# Node colours
# ======================================================================

class TestNodeColours:

    def test_all_green_at_start(self, path5_hvh):
        colours = path5_hvh._node_colours()
        assert all(c == _COLOURS["available"] for c in colours)

    def test_selected_gets_player_colour(self, path5_hvh):
        path5_hvh.game.make_move(0)
        colours = path5_hvh._node_colours()
        # Node 0 is first in iteration → should be player1 colour
        nodes = list(path5_hvh.game.graph.nodes)
        idx0 = nodes.index(0)
        assert colours[idx0] == _COLOURS["player1"]

    def test_blocked_nodes_grey(self, path5_hvh):
        path5_hvh.game.make_move(2)  # blocks 1, 3
        colours = path5_hvh._node_colours()
        nodes = list(path5_hvh.game.graph.nodes)
        idx1 = nodes.index(1)
        idx3 = nodes.index(3)
        assert colours[idx1] == _COLOURS["blocked"]
        assert colours[idx3] == _COLOURS["blocked"]


# ======================================================================
# Status text
# ======================================================================

class TestStatusText:

    def test_shows_current_player(self, path5_hvh):
        text = path5_hvh._status_text()
        assert "Player 1" in text

    def test_shows_game_over(self, path5_hvh):
        # Play to completion
        path5_hvh.game.make_move(0)
        path5_hvh.game.make_move(2)
        path5_hvh.game.make_move(4)
        text = path5_hvh._status_text()
        assert "Game Over" in text
        assert "wins" in text

    def test_human_vs_ai_shows_ai_thinking(self, path5_hva):
        path5_hva.game.make_move(0)  # human move; now it's AI's turn
        text = path5_hva._status_text()
        assert "AI is thinking" in text


# ======================================================================
# is_human
# ======================================================================

class TestIsHuman:

    def test_human_vs_human(self, path5_hvh):
        assert path5_hvh._is_human(1) is True
        assert path5_hvh._is_human(2) is True

    def test_human_vs_ai(self, path5_hva):
        assert path5_hva._is_human(1) is True
        assert path5_hva._is_human(2) is False

    def test_human_vs_ai_human_is_player2(self):
        """When human_player=2, the human is P2 and the AI goes first."""
        gui = GameGUI(
            nx.path_graph(5), mode="human_vs_ai", human_player=2,
        )
        assert gui._is_human(1) is False
        assert gui._is_human(2) is True
        # AI should be in slot 1
        assert gui.ai1 is not None
        assert gui.ai2 is None

    def test_invalid_human_player_raises(self):
        with pytest.raises(ValueError, match="human_player"):
            GameGUI(nx.path_graph(3), mode="human_vs_ai", human_player=3)

    def test_ai_vs_ai(self, triangle_ava):
        assert triangle_ava._is_human(1) is False
        assert triangle_ava._is_human(2) is False


# ======================================================================
# AI turn logic
# ======================================================================

class TestAITurn:

    def test_ai_responds_in_human_vs_ai(self, path5_hva):
        """After human moves, AI should respond automatically."""
        _setup_fig(path5_hva)
        path5_hva.game.make_move(0)  # human (player 1)
        path5_hva._maybe_ai_turn()   # should trigger AI (player 2)
        # AI should have made a move; now it's player 1 again
        assert path5_hva.game.current_player == 1
        assert len(path5_hva.game.history) == 2

    def test_ai_does_not_act_in_human_vs_human(self, path5_hvh):
        _setup_fig(path5_hvh)
        path5_hvh._maybe_ai_turn()
        assert len(path5_hvh.game.history) == 0

    def test_ai_vs_ai_steps_once(self, triangle_ava):
        _setup_fig(triangle_ava)
        triangle_ava._maybe_ai_turn()
        assert len(triangle_ava.game.history) == 1


# ======================================================================
# Undo
# ======================================================================

class TestUndo:

    def test_undo_in_human_vs_human(self, path5_hvh):
        _setup_fig(path5_hvh)
        path5_hvh.game.make_move(0)
        path5_hvh._undo()
        assert len(path5_hvh.game.history) == 0
        assert path5_hvh.game.current_player == 1

    def test_undo_in_human_vs_ai(self, path5_hva):
        """Undo should revert both the AI's move and the human's."""
        _setup_fig(path5_hva)
        path5_hva.game.make_move(0)        # human
        path5_hva._maybe_ai_turn()          # AI
        assert len(path5_hva.game.history) == 2
        path5_hva._undo()
        assert len(path5_hva.game.history) == 0
        assert path5_hva.game.current_player == 1

    def test_undo_on_empty_does_nothing(self, path5_hvh):
        _setup_fig(path5_hvh)
        path5_hvh._undo()  # should not raise
        assert len(path5_hvh.game.history) == 0


# ======================================================================
# Restart
# ======================================================================

class TestRestart:

    def test_restart_resets_state(self, path5_hvh):
        _setup_fig(path5_hvh)
        path5_hvh.game.make_move(0)
        path5_hvh.game.make_move(2)
        path5_hvh._restart()
        assert path5_hvh.game.current_player == 1
        assert path5_hvh.game.selected == set()
        assert path5_hvh.game.history == []


# ======================================================================
# find_closest_node
# ======================================================================

class TestFindClosestNode:

    def test_finds_exact_position(self, path5_hvh):
        for node, (x, y) in path5_hvh.pos.items():
            found = path5_hvh._find_closest_node(x, y)
            assert found == node

    def test_returns_none_for_far_away(self, path5_hvh):
        found = path5_hvh._find_closest_node(999.0, 999.0)
        assert found is None


# ======================================================================
# Layout options
# ======================================================================

class TestLayout:

    def test_different_layouts(self):
        g = nx.path_graph(5)
        for layout in ("spring", "circular", "shell", "kamada_kawai"):
            gui = GameGUI(g, mode="human_vs_human", layout=layout)
            assert len(gui.pos) == 5


# ======================================================================
# Cleanup
# ======================================================================

@pytest.fixture(autouse=True)
def close_figures():
    """Close all Matplotlib figures after each test."""
    yield
    plt.close("all")
