"""
Self-play framework for the Stable Set Game.

Lets different AI configurations compete against each other on various
graphs and collects detailed match statistics.  Designed for the
dissertation's experimental evaluation: "which evaluation-function
weights / search depths perform best on which graph families?"

Key components
--------------
* ``MatchResult`` — outcome of a single game between two AI players.
* ``play_match``  — run one game and return a ``MatchResult``.
* ``play_series`` — run multiple games (optionally swapping who goes
  first) on a single graph and return a ``SeriesResult``.
* ``run_tournament`` — round-robin tournament across multiple AI
  configs and graphs, returning a ``TournamentResult`` with
  aggregated win rates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from stable_set_game.ai import AIPlayer, SearchResult
from stable_set_game.engine import StableSetGame


# ======================================================================
# Single-match result
# ======================================================================

@dataclass
class MatchResult:
    """Outcome of a single game.

    Attributes:
        player1_name:     Name of the AI that went first.
        player2_name:     Name of the AI that went second.
        winner_name:      Name of the winning AI.
        loser_name:       Name of the losing AI.
        num_moves:        Total number of moves played.
        move_history:     List of (player_number, vertex) tuples.
        total_nodes:      Total game-tree nodes explored by both AIs.
        elapsed_seconds:  Wall-clock time for the match.
        graph_info:       Short description of the graph used.
    """
    player1_name: str = ""
    player2_name: str = ""
    winner_name: str = ""
    loser_name: str = ""
    num_moves: int = 0
    move_history: List[Tuple[int, object]] = field(default_factory=list)
    total_nodes: int = 0
    elapsed_seconds: float = 0.0
    graph_info: str = ""


def play_match(
    player1: AIPlayer,
    player2: AIPlayer,
    graph: nx.Graph,
    graph_info: str = "",
) -> MatchResult:
    """Play a single game between two AI players.

    Args:
        player1: AI that moves first.
        player2: AI that moves second.
        graph:   The graph to play on (not modified).
        graph_info: Optional label describing the graph (for logging).

    Returns:
        A ``MatchResult`` with full game details.
    """
    game = StableSetGame(graph)
    players = {1: player1, 2: player2}
    total_nodes = 0

    start = time.perf_counter()

    while not game.is_game_over():
        current_ai = players[game.current_player]
        result: SearchResult = current_ai.choose_move(game)
        total_nodes += result.nodes_explored
        game.make_move(result.best_move)

    elapsed = time.perf_counter() - start
    winner_num = game.get_winner()
    loser_num = game.get_loser()

    return MatchResult(
        player1_name=player1.name,
        player2_name=player2.name,
        winner_name=players[winner_num].name,
        loser_name=players[loser_num].name,
        num_moves=len(game.history),
        move_history=list(game.history),
        total_nodes=total_nodes,
        elapsed_seconds=elapsed,
        graph_info=graph_info,
    )


# ======================================================================
# Series result (multiple games on one graph)
# ======================================================================

@dataclass
class SeriesResult:
    """Aggregated results from a series of games between two AIs.

    Attributes:
        player1_name: Name of one AI.
        player2_name: Name of the other AI.
        wins:         Mapping from player name → number of wins.
        total_games:  Total number of games played.
        matches:      List of individual ``MatchResult``s.
        graph_info:   Description of the graph used.
    """
    player1_name: str = ""
    player2_name: str = ""
    wins: Dict[str, int] = field(default_factory=dict)
    total_games: int = 0
    matches: List[MatchResult] = field(default_factory=list)
    graph_info: str = ""

    def win_rate(self, player_name: str) -> float:
        """Return the win rate (0.0–1.0) for the given player.

        Args:
            player_name: Must match one of the player names.

        Raises:
            ValueError: If the player name is not found.
        """
        if player_name not in self.wins:
            raise ValueError(
                f"Unknown player '{player_name}'. "
                f"Known: {list(self.wins.keys())}"
            )
        if self.total_games == 0:
            return 0.0
        return self.wins[player_name] / self.total_games


def play_series(
    player1: AIPlayer,
    player2: AIPlayer,
    graph: nx.Graph,
    num_games: int = 10,
    swap_sides: bool = True,
    graph_info: str = "",
) -> SeriesResult:
    """Play a series of games between two AIs on the same graph.

    Args:
        player1:    First AI configuration.
        player2:    Second AI configuration.
        graph:      The graph to play on.
        num_games:  Total number of games to play.
        swap_sides: If True, players alternate who goes first each game
                    to reduce first-mover bias.
        graph_info: Optional description of the graph.

    Returns:
        A ``SeriesResult`` with win counts and individual match results.
    """
    wins: Dict[str, int] = {player1.name: 0, player2.name: 0}
    matches: List[MatchResult] = []

    for i in range(num_games):
        # Swap who goes first every other game
        if swap_sides and i % 2 == 1:
            first, second = player2, player1
        else:
            first, second = player1, player2

        result = play_match(first, second, graph, graph_info=graph_info)
        matches.append(result)
        wins[result.winner_name] += 1

    return SeriesResult(
        player1_name=player1.name,
        player2_name=player2.name,
        wins=wins,
        total_games=num_games,
        matches=matches,
        graph_info=graph_info,
    )


# ======================================================================
# Tournament result (round-robin across configs and graphs)
# ======================================================================

@dataclass
class TournamentResult:
    """Results from a round-robin tournament.

    Attributes:
        series_results: All series results, keyed by
                        ``(player1_name, player2_name, graph_info)``.
        leaderboard:    Mapping from player name → total wins across
                        all series.
        total_games:    Total number of individual games played.
    """
    series_results: Dict[Tuple[str, str, str], SeriesResult] = field(
        default_factory=dict
    )
    leaderboard: Dict[str, int] = field(default_factory=dict)
    total_games: int = 0

    def games_played_by(self, player_name: str) -> int:
        """Number of games the given player actually participated in."""
        return sum(
            sr.total_games
            for sr in self.series_results.values()
            if sr.player1_name == player_name or sr.player2_name == player_name
        )

    def win_rate(self, player_name: str) -> float:
        """Win rate for a player across games they participated in.

        Note: divides by the player's own games_played (not the total
        tournament games), so in a round-robin with k players the
        maximum possible rate is 1.0, not 1/k.
        """
        if player_name not in self.leaderboard:
            raise ValueError(f"Unknown player '{player_name}'.")
        played = self.games_played_by(player_name)
        if played == 0:
            return 0.0
        return self.leaderboard[player_name] / played

    def summary(self) -> str:
        """Return a human-readable summary table of the tournament."""
        lines = ["=" * 60, "Tournament Summary", "=" * 60]

        # Sort leaderboard by wins descending
        sorted_players = sorted(
            self.leaderboard.items(), key=lambda x: x[1], reverse=True
        )

        lines.append(f"{'Player':<25} {'Wins':>6} {'Games':>6} {'Win%':>8}")
        lines.append("-" * 50)
        for name, wins in sorted_players:
            rate = self.win_rate(name) * 100
            player_games = self.games_played_by(name)
            lines.append(f"{name:<25} {wins:>6} {player_games:>6} {rate:>7.1f}%")

        lines.append("-" * 50)
        lines.append(f"Total games played: {self.total_games}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_tournament(
    players: Sequence[AIPlayer],
    graphs: Sequence[Tuple[nx.Graph, str]],
    games_per_series: int = 10,
    swap_sides: bool = True,
) -> TournamentResult:
    """Run a round-robin tournament: every pair of AIs plays on every graph.

    Args:
        players:          List of AI configurations to compete.
        graphs:           List of (graph, description) pairs.
        games_per_series: Number of games per (player_pair, graph) series.
        swap_sides:       Whether to alternate first-mover each game.

    Returns:
        A ``TournamentResult`` with aggregated statistics.

    Raises:
        ValueError: If fewer than 2 players are provided.
        ValueError: If any two players share the same name.
    """
    if len(players) < 2:
        raise ValueError("Tournament requires at least 2 players.")

    names = [p.name for p in players]
    if len(set(names)) != len(names):
        raise ValueError(
            "All players must have unique names. "
            f"Got: {names}"
        )

    leaderboard: Dict[str, int] = {p.name: 0 for p in players}
    series_results: Dict[Tuple[str, str, str], SeriesResult] = {}
    total_games = 0

    for i, p1 in enumerate(players):
        for p2 in players[i + 1:]:
            for graph, graph_info in graphs:
                series = play_series(
                    p1, p2, graph,
                    num_games=games_per_series,
                    swap_sides=swap_sides,
                    graph_info=graph_info,
                )
                key = (p1.name, p2.name, graph_info)
                series_results[key] = series
                total_games += series.total_games

                for name, wins in series.wins.items():
                    leaderboard[name] += wins

    return TournamentResult(
        series_results=series_results,
        leaderboard=leaderboard,
        total_games=total_games,
    )
