"""
Simple interactive GUI for the Stable Set Game.

Built with Matplotlib so that no extra GUI toolkit (Tkinter, PyQt, etc.)
is required beyond the project's existing dependencies.

Supported modes
---------------
* **Human vs Human** — both players click vertices on the graph.
* **Human vs AI**    — the human clicks; the AI responds automatically.
* **AI vs AI**       — watch two AIs play each other step by step.

Usage
-----
::

    from stable_set_game.gui import GameGUI
    import networkx as nx

    # Human vs AI on a path graph
    GameGUI(nx.path_graph(7), mode="human_vs_ai").run()

    # Human vs Human on a cycle
    GameGUI(nx.cycle_graph(6), mode="human_vs_human").run()

    # AI vs AI (spectator mode)
    GameGUI(nx.path_graph(8), mode="ai_vs_ai").run()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button, RadioButtons, TextBox
import networkx as nx

from stable_set_game.ai import AIPlayer
from stable_set_game.engine import StableSetGame
from stable_set_game.evaluation import WeightedEvaluator


# ======================================================================
# Colour palette
# ======================================================================

_COLOURS = {
    "available": "#4CAF50",     # green — legal move
    "blocked": "#BDBDBD",       # grey  — blocked by adjacency
    "player1": "#1E88E5",       # blue  — selected by player 1
    "player2": "#E53935",       # red   — selected by player 2
    "edge": "#9E9E9E",          # grey  — edges
    "highlight": "#FFC107",     # amber — hover highlight
    "background": "#FAFAFA",    # off-white
    "navy": "#17324D",          # application header / primary action
    "panel": "#EEF3F7",         # setup panels
    "muted": "#5F6B76",         # secondary text
    "danger": "#B33A3A",        # quit / error
}

_PLAYER_LABELS = {1: "Player 1 (Blue)", 2: "Player 2 (Red)"}

_GRAPH_LABELS = (
    "Path",
    "Cycle",
    "Complete",
    "Star",
    "Erdos-Renyi",
    "3-Regular",
    "Fork",
)

_MODE_LABELS = {
    "Human vs Human": "human_vs_human",
    "Human vs AI": "human_vs_ai",
    "AI vs AI": "ai_vs_ai",
}


@dataclass(frozen=True)
class GameConfig:
    """Validated options selected on the graphical setup screen."""

    graph_type: str = "Path"
    n: int = 7
    mode: str = "human_vs_ai"
    human_player: int = 1
    depth: int = 6


def build_game_graph(config: GameConfig) -> Tuple[nx.Graph, str, str]:
    """Build the selected graph and return ``(graph, label, layout)``.

    Keeping graph construction outside the widgets makes the setup screen
    straightforward to test and prevents UI changes from touching the
    experiment or Sprague-Grundy computation modules.
    """
    from stable_set_game.generators import (
        complete_graph,
        cycle_graph,
        erdos_renyi_graph,
        fork_graph,
        path_graph,
        random_regular_graph,
        star_graph,
    )

    if config.graph_type not in _GRAPH_LABELS:
        raise ValueError(f"Unknown graph type: {config.graph_type}.")
    if config.mode not in _MODE_LABELS.values():
        raise ValueError(f"Unknown game mode: {config.mode}.")
    if config.human_player not in (1, 2):
        raise ValueError("Human player must be Player 1 or Player 2.")
    if not 1 <= config.depth <= 12:
        raise ValueError("AI search depth must be between 1 and 12.")
    if not 1 <= config.n <= 60:
        raise ValueError("Graph size must be between 1 and 60.")

    n = config.n
    if config.graph_type == "Path":
        if n < 2:
            raise ValueError("A path graph requires at least 2 vertices.")
        return path_graph(n), f"Path P{n}", "spring"
    if config.graph_type == "Cycle":
        if n < 3:
            raise ValueError("A cycle graph requires at least 3 vertices.")
        return cycle_graph(n), f"Cycle C{n}", "circular"
    if config.graph_type == "Complete":
        return complete_graph(n), f"Complete K{n}", "circular"
    if config.graph_type == "Star":
        return star_graph(n), f"Star S{n}", "spring"
    if config.graph_type == "Erdos-Renyi":
        return (
            erdos_renyi_graph(n, 0.4, seed=42),
            f"Erdos-Renyi G({n}, 0.4)",
            "spring",
        )
    if config.graph_type == "3-Regular":
        if n < 4 or n % 2:
            raise ValueError("A 3-regular graph requires an even n of at least 4.")
        return random_regular_graph(n, 3, seed=42), f"3-Regular ({n})", "spring"
    if n < 3:
        raise ValueError("A fork graph requires n of at least 3.")
    return fork_graph(n), f"Fork F{n}", "spring"


# ======================================================================
# GUI class
# ======================================================================

class GameGUI:
    """Interactive Matplotlib-based GUI for the Stable Set Game.

    Args:
        graph:      The graph to play on.
        mode:       One of ``"human_vs_human"``, ``"human_vs_ai"``,
                    ``"ai_vs_ai"``.
        ai1:        AI config for player 1 (used in ``ai_vs_ai`` mode).
        ai2:        AI config for player 2 (used in ``human_vs_ai`` and
                    ``ai_vs_ai`` modes).
        layout:     NetworkX layout function name (e.g. ``"spring"``,
                    ``"circular"``, ``"shell"``).  Defaults to
                    ``"spring"``.
        node_size:  Base size of drawn nodes.
        title:      Window title.
    """

    def __init__(
        self,
        graph: nx.Graph,
        mode: str = "human_vs_ai",
        ai1: Optional[AIPlayer] = None,
        ai2: Optional[AIPlayer] = None,
        layout: str = "spring",
        node_size: int = 600,
        title: str = "Stable Set Game",
        human_player: int = 1,
    ) -> None:
        if mode not in ("human_vs_human", "human_vs_ai", "ai_vs_ai"):
            raise ValueError(
                f"Unknown mode '{mode}'. Choose from: "
                "human_vs_human, human_vs_ai, ai_vs_ai."
            )
        if human_player not in (1, 2):
            raise ValueError(
                f"human_player must be 1 or 2, got {human_player}."
            )

        self.mode = mode
        self.game = StableSetGame(graph)
        self.node_size = node_size
        self.title = title
        # In human_vs_ai mode, which player is the human (1 or 2)
        self.human_player = human_player

        # AI players — fill in defaults where needed.
        # In human_vs_ai mode the AI occupies whichever slot the human
        # does NOT: if human is player 1, AI is ai2; vice versa.
        self.ai1: Optional[AIPlayer] = ai1
        self.ai2: Optional[AIPlayer] = ai2
        if mode == "human_vs_ai":
            if human_player == 1 and self.ai2 is None:
                self.ai2 = AIPlayer(max_depth=6, name="AI")
            if human_player == 2 and self.ai1 is None:
                self.ai1 = AIPlayer(max_depth=6, name="AI")
        if mode == "ai_vs_ai":
            if self.ai1 is None:
                self.ai1 = AIPlayer(max_depth=6, name="AI-1")
            if self.ai2 is None:
                self.ai2 = AIPlayer(max_depth=6, name="AI-2")

        # Compute a stable layout once
        self.pos: Dict = self._compute_layout(graph, layout)

        # Matplotlib state
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self._hover_node: Optional[object] = None
        self._message: str = ""
        self._buttons: Dict[str, Button] = {}
        self._status_artist = None

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_layout(graph: nx.Graph, name: str) -> Dict:
        """Compute node positions using the named layout algorithm."""
        layouts = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "shell": nx.shell_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "spectral": nx.spectral_layout,
        }
        fn = layouts.get(name, nx.spring_layout)
        return fn(graph)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _node_colours(self) -> List[str]:
        """Assign a colour to every node based on game state."""
        legal = set(self.game.legal_moves())
        colours = []
        for v in self.game.graph.nodes:
            if v in self.game.selected:
                # Find which player selected it
                for player, vertex in self.game.history:
                    if vertex == v:
                        colours.append(
                            _COLOURS["player1"] if player == 1
                            else _COLOURS["player2"]
                        )
                        break
            elif v in legal:
                colours.append(_COLOURS["available"])
            else:
                colours.append(_COLOURS["blocked"])
        return colours

    def _draw(self) -> None:
        """Redraw the entire game board."""
        self.ax.clear()
        self.ax.set_facecolor(_COLOURS["background"])

        colours = self._node_colours()
        legal = set(self.game.legal_moves())

        # Draw edges
        nx.draw_networkx_edges(
            self.game.graph, self.pos, ax=self.ax,
            edge_color=_COLOURS["edge"], width=1.5, alpha=0.6,
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.game.graph, self.pos, ax=self.ax,
            node_color=colours, node_size=self.node_size,
            edgecolors="black", linewidths=1.5,
        )

        # Hover highlight
        if self._hover_node is not None and self._hover_node in legal:
            nx.draw_networkx_nodes(
                self.game.graph, self.pos, ax=self.ax,
                nodelist=[self._hover_node],
                node_color=_COLOURS["highlight"],
                node_size=self.node_size + 100,
                edgecolors="black", linewidths=2.5,
            )

        # Labels
        nx.draw_networkx_labels(
            self.game.graph, self.pos, ax=self.ax,
            font_size=10, font_weight="bold",
        )

        # Status text. In the full window this lives in a dedicated status
        # line; tests and embedded uses fall back to the axes title.
        if self._status_artist is not None:
            self._status_artist.set_text(self._status_text())
        else:
            self.ax.set_title(self._status_text(), fontsize=13, pad=12)

        # Legend
        legend_handles = [
            mpatches.Patch(color=_COLOURS["player1"], label="Player 1"),
            mpatches.Patch(color=_COLOURS["player2"], label="Player 2"),
            mpatches.Patch(color=_COLOURS["available"], label="Available"),
            mpatches.Patch(color=_COLOURS["blocked"], label="Blocked"),
        ]
        self.ax.legend(
            handles=legend_handles, loc="upper left",
            fontsize=8, framealpha=0.8,
        )

        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def _status_text(self) -> str:
        """Build the title/status line."""
        if self._message:
            return self._message
        if self.game.is_game_over():
            winner = self.game.get_winner()
            return (
                f"Game Over!  {_PLAYER_LABELS[winner]} wins "
                f"in {len(self.game.history)} moves.  "
                "Use Restart to play again."
            )
        player = self.game.current_player
        label = _PLAYER_LABELS[player]
        is_human = self._is_human(player)
        action = "Click a green vertex" if is_human else "AI is thinking..."
        return f"{label}'s turn — {action}"

    def _is_human(self, player: int) -> bool:
        """Return True if the given player is human in the current mode."""
        if self.mode == "human_vs_human":
            return True
        if self.mode == "human_vs_ai":
            return player == self.human_player
        return False  # ai_vs_ai

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _find_closest_node(self, x: float, y: float) -> Optional[object]:
        """Return the node closest to (x, y), or None if too far away."""
        best_node = None
        best_dist = float("inf")
        # Threshold in data coordinates — generous for clicks
        threshold = 0.08

        for node, (nx_, ny) in self.pos.items():
            dist = ((x - nx_) ** 2 + (y - ny) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node if best_dist < threshold else None

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse click: human makes a move."""
        if event.inaxes != self.ax:
            return
        if self.game.is_game_over():
            return
        if not self._is_human(self.game.current_player):
            return

        node = self._find_closest_node(event.xdata, event.ydata)
        if node is not None and self.game.is_legal_move(node):
            self.game.make_move(node)
            self._hover_node = None
            self._draw()
            # After human moves, trigger AI if needed
            self._maybe_ai_turn()

    def _on_motion(self, event: MouseEvent) -> None:
        """Handle mouse motion: update hover highlight."""
        if event.inaxes != self.ax:
            if self._hover_node is not None:
                self._hover_node = None
                self._draw()
            return

        node = self._find_closest_node(event.xdata, event.ydata)
        if node != self._hover_node:
            self._hover_node = node
            self._draw()

    def _on_key(self, event) -> None:
        """Handle keyboard input: R = restart, Q = quit, U = undo."""
        if event.key == "r":
            self._restart()
        elif event.key == "q":
            plt.close(self.fig)
        elif event.key == "u":
            self._undo()
        elif event.key == " " and self.mode == "ai_vs_ai":
            # Space to step in AI vs AI mode
            self._maybe_ai_turn()

    def _quit(self, _event=None) -> None:
        """Close the game window."""
        if self.fig is not None:
            plt.close(self.fig)

    def _next_ai(self, _event=None) -> None:
        """Advance one move in AI-vs-AI spectator mode."""
        if self.mode == "ai_vs_ai":
            self._maybe_ai_turn()

    def _restart(self, _event=None) -> None:
        """Reset the game to its initial state."""
        self.game = StableSetGame(self.game.graph)
        self._hover_node = None
        self._message = ""
        self._draw()
        self._maybe_ai_turn()

    def _undo(self, _event=None) -> None:
        """Undo the last move (or last two in human_vs_ai mode).

        In human_vs_ai mode we want to revert both moves so the human
        is once again facing the same decision point.
        """
        if not self.game.history:
            return

        if (
            self.mode == "human_vs_ai"
            and self.human_player == 2
            and len(self.game.history) < 2
        ):
            # The AI's opening move alone cannot be undone without leaving
            # the game on an AI turn that will not automatically restart.
            return

        if self.mode == "human_vs_ai":
            # Roll back only as far as the most recent human decision point.
            self.game.undo_move()
            while self.game.history and self.game.current_player != self.human_player:
                self.game.undo_move()
        else:
            self.game.undo_move()

        self._message = ""
        self._draw()

    # ------------------------------------------------------------------
    # AI turn logic
    # ------------------------------------------------------------------

    def _get_ai(self, player: int) -> Optional[AIPlayer]:
        """Return the AI for the given player number, or None."""
        if self.mode == "human_vs_human":
            return None
        if self.mode == "human_vs_ai":
            # The AI is whichever slot the human does NOT occupy
            if player == self.human_player:
                return None
            return self.ai1 if player == 1 else self.ai2
        # ai_vs_ai
        return self.ai1 if player == 1 else self.ai2

    def _maybe_ai_turn(self) -> None:
        """If the current player is an AI, make its move."""
        if self.game.is_game_over():
            return

        ai = self._get_ai(self.game.current_player)
        if ai is None:
            return

        if self.mode == "ai_vs_ai":
            # Single step per space press — let user watch
            self._message = f"{_PLAYER_LABELS[self.game.current_player]}: AI is thinking..."
            self._draw()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            result = ai.choose_move(self.game)
            self.game.make_move(result.best_move)
            self._message = ""
            self._draw()
            return

        # human_vs_ai: AI responds immediately
        self._message = f"{_PLAYER_LABELS[self.game.current_player]}: AI is thinking..."
        self._draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        result = ai.choose_move(self.game)
        self.game.make_move(result.best_move)
        self._message = ""
        self._draw()

    def _create_controls(self) -> None:
        """Create visible game controls below the board."""
        specs = [
            ("restart", "Restart", 0.16, _COLOURS["panel"], self._restart),
            ("undo", "Undo", 0.32, _COLOURS["panel"], self._undo),
            ("next", "Next AI Move", 0.48, "#DCEAF7", self._next_ai),
            ("quit", "Quit", 0.68, "#F5DADA", self._quit),
        ]
        for key, label, left, colour, callback in specs:
            width = 0.16 if key == "next" else 0.12
            button_ax = self.fig.add_axes([left, 0.055, width, 0.06])
            button = Button(
                button_ax,
                label,
                color=colour,
                hovercolor="#C7D9E8" if key != "quit" else "#E8B8B8",
            )
            button.label.set_fontsize(10)
            button.on_clicked(callback)
            self._buttons[key] = button

        # The Next button is only meaningful in spectator mode.
        self._buttons["next"].ax.set_visible(self.mode == "ai_vs_ai")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        """Create the game window without starting the blocking event loop."""
        self.fig = plt.figure(figsize=(10, 7), facecolor=_COLOURS["background"])
        self.ax = self.fig.add_axes([0.05, 0.16, 0.90, 0.69])
        self.fig.canvas.manager.set_window_title(self.title)
        self.fig.text(
            0.05, 0.965, self.title,
            fontsize=16, fontweight="bold", color=_COLOURS["navy"], va="top",
        )
        mode_label = next(
            label for label, value in _MODE_LABELS.items() if value == self.mode
        )
        self.fig.text(
            0.95, 0.965, mode_label,
            fontsize=10, color=_COLOURS["muted"], ha="right", va="top",
        )
        self._status_artist = self.fig.text(
            0.50, 0.89, self._status_text(),
            fontsize=12, color=_COLOURS["navy"], ha="center", va="center",
        )
        self._create_controls()

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()

    def run(self) -> None:
        """Open the GUI window and start the interactive game loop."""
        self._build_window()

        # Kick off AI if it goes first
        if self.mode == "ai_vs_ai":
            # In AI vs AI mode, user presses Space to step
            pass
        else:
            self._maybe_ai_turn()

        plt.show()


# ======================================================================
# Graphical launcher (python -m stable_set_game)
# ======================================================================

class GameLauncher:
    """Matplotlib setup screen used instead of command-line prompts."""

    def __init__(self) -> None:
        self.config: Optional[GameConfig] = None
        self.fig: Optional[plt.Figure] = None
        self.graph_radio: Optional[RadioButtons] = None
        self.mode_radio: Optional[RadioButtons] = None
        self.side_radio: Optional[RadioButtons] = None
        self.n_box: Optional[TextBox] = None
        self.depth_box: Optional[TextBox] = None
        self.status_artist = None
        self._buttons: Dict[str, Button] = {}

    def _build_screen(self) -> None:
        self.fig = plt.figure(figsize=(10, 7), facecolor=_COLOURS["background"])
        self.fig.canvas.manager.set_window_title("Stable Set Game - New Game")

        self.fig.text(
            0.07, 0.94, "Stable Set Game",
            fontsize=24, fontweight="bold", color=_COLOURS["navy"], va="top",
        )
        self.fig.text(
            0.07, 0.885,
            "Configure a graph and choose how you would like to play.",
            fontsize=11, color=_COLOURS["muted"], va="top",
        )

        self.fig.text(0.07, 0.80, "1. Graph", fontsize=13, fontweight="bold")
        graph_ax = self.fig.add_axes([0.07, 0.32, 0.25, 0.45], facecolor=_COLOURS["panel"])
        self.graph_radio = RadioButtons(graph_ax, _GRAPH_LABELS, active=0)
        for label in self.graph_radio.labels:
            label.set_fontsize(10)

        self.fig.text(0.38, 0.80, "2. Game mode", fontsize=13, fontweight="bold")
        mode_ax = self.fig.add_axes([0.38, 0.57, 0.25, 0.20], facecolor=_COLOURS["panel"])
        self.mode_radio = RadioButtons(mode_ax, tuple(_MODE_LABELS), active=1)
        for label in self.mode_radio.labels:
            label.set_fontsize(10)

        self.fig.text(0.38, 0.50, "Human side (Human vs AI)", fontsize=10)
        side_ax = self.fig.add_axes([0.38, 0.34, 0.25, 0.14], facecolor=_COLOURS["panel"])
        self.side_radio = RadioButtons(side_ax, ("Player 1 - first", "Player 2 - second"), active=0)
        for label in self.side_radio.labels:
            label.set_fontsize(9)

        self.fig.text(0.70, 0.80, "3. Parameters", fontsize=13, fontweight="bold")
        n_ax = self.fig.add_axes([0.70, 0.66, 0.22, 0.06])
        self.n_box = TextBox(n_ax, "Graph size n  ", initial="7")
        depth_ax = self.fig.add_axes([0.70, 0.55, 0.22, 0.06])
        self.depth_box = TextBox(depth_ax, "AI depth  ", initial="6")

        self.fig.text(
            0.70, 0.46,
            "Limits: n = 1-60, depth = 1-12\n"
            "3-Regular graphs require even n >= 4.\n"
            "Random graphs use p=0.4 and seed=42.",
            fontsize=9, color=_COLOURS["muted"], linespacing=1.5,
        )

        start_ax = self.fig.add_axes([0.70, 0.30, 0.15, 0.075])
        start = Button(start_ax, "Start Game", color="#D6E8F5", hovercolor="#BFD8EA")
        start.label.set_fontweight("bold")
        start.on_clicked(self._on_start)
        self._buttons["start"] = start

        quit_ax = self.fig.add_axes([0.87, 0.30, 0.08, 0.075])
        quit_button = Button(quit_ax, "Quit", color="#F5DADA", hovercolor="#E8B8B8")
        quit_button.on_clicked(self._on_quit)
        self._buttons["quit"] = quit_button

        self.status_artist = self.fig.text(
            0.07, 0.17,
            "Green vertices are legal moves. The player with no legal move loses.",
            fontsize=10, color=_COLOURS["muted"],
        )
        self.fig.text(
            0.07, 0.08,
            "Controls in game: click a green vertex, or use the visible Restart, Undo, Next and Quit buttons.",
            fontsize=9, color=_COLOURS["muted"],
        )

    def _selected_config(self) -> GameConfig:
        try:
            n = int(self.n_box.text.strip())
            depth = int(self.depth_box.text.strip())
        except ValueError as exc:
            raise ValueError("Graph size and AI depth must be whole numbers.") from exc

        mode = _MODE_LABELS[self.mode_radio.value_selected]
        human_player = 1 if self.side_radio.value_selected.startswith("Player 1") else 2
        return GameConfig(
            graph_type=self.graph_radio.value_selected,
            n=n,
            mode=mode,
            human_player=human_player,
            depth=depth,
        )

    def _on_start(self, _event) -> None:
        try:
            config = self._selected_config()
            build_game_graph(config)  # validate before closing the setup screen
        except ValueError as exc:
            self.status_artist.set_text(str(exc))
            self.status_artist.set_color(_COLOURS["danger"])
            self.fig.canvas.draw_idle()
            return

        self.config = config
        plt.close(self.fig)

    def _on_quit(self, _event) -> None:
        self.config = None
        plt.close(self.fig)

    def run(self) -> Optional[GameConfig]:
        self._build_screen()
        plt.show()
        return self.config


def main() -> None:
    """Launch the graphical setup screen, then start the selected game."""
    config = GameLauncher().run()
    if config is None:
        return

    graph, graph_name, layout = build_game_graph(config)
    ai1 = AIPlayer(max_depth=config.depth, name="AI-1")
    ai2 = AIPlayer(max_depth=config.depth, name="AI-2")
    gui = GameGUI(
        graph,
        mode=config.mode,
        ai1=ai1,
        ai2=ai2,
        layout=layout,
        human_player=config.human_player,
        title=f"Stable Set Game - {graph_name}",
    )
    gui.run()


if __name__ == "__main__":
    main()
