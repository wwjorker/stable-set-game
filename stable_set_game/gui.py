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
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backend_bases import MouseEvent
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
}

_PLAYER_LABELS = {1: "Player 1 (Blue)", 2: "Player 2 (Red)"}


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

        # Status text
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
        if self.game.is_game_over():
            winner = self.game.get_winner()
            return (
                f"Game Over!  {_PLAYER_LABELS[winner]} wins "
                f"in {len(self.game.history)} moves.  "
                f"(Press R to restart, Q to quit)"
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

    def _restart(self) -> None:
        """Reset the game to its initial state."""
        self.game = StableSetGame(self.game.graph)
        self._hover_node = None
        self._draw()
        self._maybe_ai_turn()

    def _undo(self) -> None:
        """Undo the last move (or last two in human_vs_ai mode).

        In human_vs_ai mode we want to revert both moves so the human
        is once again facing the same decision point.
        """
        if not self.game.history:
            return

        if self.mode == "human_vs_ai" and len(self.game.history) >= 2:
            # Undo both the AI's move and the human's previous move
            self.game.undo_move()
            self.game.undo_move()
        else:
            self.game.undo_move()

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
            result = ai.choose_move(self.game)
            self.game.make_move(result.best_move)
            self._draw()
            return

        # human_vs_ai: AI responds immediately
        result = ai.choose_move(self.game)
        self.game.make_move(result.best_move)
        self._draw()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the GUI window and start the interactive game loop."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title(self.title)
        self.fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()

        # Kick off AI if it goes first
        if self.mode == "ai_vs_ai":
            # In AI vs AI mode, user presses Space to step
            pass
        else:
            self._maybe_ai_turn()

        plt.show()


# ======================================================================
# Convenience launcher (python -m stable_set_game.gui)
# ======================================================================

def main() -> None:
    """Command-line entry point with a simple menu."""
    import sys

    print("=" * 50)
    print("  Stable Set Game")
    print("=" * 50)
    print()
    print("Graph types:")
    print("  1. Path graph P_n")
    print("  2. Cycle graph C_n")
    print("  3. Complete graph K_n")
    print("  4. Star graph S_n")
    print("  5. Random (Erdos-Renyi) G(n, 0.4)")
    print("  6. Random d-regular (d=3)")
    print()

    try:
        choice = input("Choose graph type [1-6, default=1]: ").strip() or "1"
        n = int(input("Number of vertices n [default=7]: ").strip() or "7")
    except (ValueError, EOFError):
        choice, n = "1", 7

    from stable_set_game.generators import (
        path_graph, cycle_graph, complete_graph, star_graph,
        erdos_renyi_graph, random_regular_graph,
    )

    # For random_regular: ensure n*d is even by making n even when d=3
    def _build_regular():
        n_fixed = n if (n * 3) % 2 == 0 else n + 1
        return random_regular_graph(max(n_fixed, 4), 3, seed=42)

    graph_builders = {
        "1": ("Path", lambda: path_graph(max(n, 2))),
        "2": ("Cycle", lambda: cycle_graph(max(n, 3))),
        "3": ("Complete", lambda: complete_graph(max(n, 1))),
        "4": ("Star", lambda: star_graph(max(n, 1))),
        "5": ("Erdos-Renyi", lambda: erdos_renyi_graph(max(n, 2), 0.4, seed=42)),
        "6": ("3-Regular", _build_regular),
    }

    graph_name, builder = graph_builders.get(choice, graph_builders["1"])
    graph = builder()

    print()
    print("Game modes:")
    print("  1. Human vs Human")
    print("  2. Human vs AI")
    print("  3. AI vs AI (press Space to step)")
    print()

    try:
        mode_choice = input("Choose mode [1-3, default=2]: ").strip() or "2"
    except EOFError:
        mode_choice = "2"

    modes = {"1": "human_vs_human", "2": "human_vs_ai", "3": "ai_vs_ai"}
    mode = modes.get(mode_choice, "human_vs_ai")

    # In human-vs-AI mode, let the user choose whether to go first.
    # This matters a lot: on e.g. P_4 or C_4 the first player is in
    # a losing position, so choosing to go second is the winning choice.
    human_player = 1
    if mode == "human_vs_ai":
        try:
            side_choice = input(
                "Go first (Player 1) or second (Player 2)? [1/2, default=1]: "
            ).strip() or "1"
            human_player = 2 if side_choice == "2" else 1
        except EOFError:
            human_player = 1

    try:
        depth_str = input("AI search depth [default=6]: ").strip() or "6"
        depth = int(depth_str)
    except (ValueError, EOFError):
        depth = 6

    print()
    print(f"Starting: {graph_name} graph (n={n}), mode={mode}, depth={depth}")
    print("Controls: click=select vertex, R=restart, U=undo, Q=quit")
    if mode == "ai_vs_ai":
        print("          Space=next AI move")
    print()

    ai1 = AIPlayer(max_depth=depth, name="AI-1")
    ai2 = AIPlayer(max_depth=depth, name="AI-2")

    layout = "circular" if choice in ("2", "3") else "spring"
    gui = GameGUI(
        graph, mode=mode, ai1=ai1, ai2=ai2,
        layout=layout,
        human_player=human_player,
        title=f"Stable Set Game — {graph_name}(n={n})",
    )
    gui.run()


if __name__ == "__main__":
    main()
