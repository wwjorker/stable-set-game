"""Tests for the self-play framework."""

import math

import pytest
import networkx as nx

from stable_set_game.ai import AIPlayer
from stable_set_game.evaluation import WeightedEvaluator
from stable_set_game.self_play import (
    MatchResult,
    TournamentResult,
    play_match,
    play_series,
    run_tournament,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def ai_deep() -> AIPlayer:
    return AIPlayer(max_depth=math.inf, name="Deep")


@pytest.fixture
def ai_shallow() -> AIPlayer:
    return AIPlayer(max_depth=2, name="Shallow")


@pytest.fixture
def ai_move_count() -> AIPlayer:
    return AIPlayer(
        max_depth=3,
        evaluator=WeightedEvaluator(weights={"move_count": 1.0}),
        name="MoveCount",
    )


@pytest.fixture
def ai_parity() -> AIPlayer:
    return AIPlayer(
        max_depth=3,
        evaluator=WeightedEvaluator(weights={"move_parity": 1.0}),
        name="Parity",
    )


@pytest.fixture
def small_path() -> nx.Graph:
    return nx.path_graph(5)


@pytest.fixture
def triangle() -> nx.Graph:
    return nx.complete_graph(3)


# ======================================================================
# play_match
# ======================================================================

class TestPlayMatch:

    def test_returns_match_result(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path, graph_info="P5")
        assert isinstance(result, MatchResult)

    def test_winner_is_one_of_the_players(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path)
        assert result.winner_name in [ai_deep.name, ai_shallow.name]

    def test_loser_is_the_other(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path)
        assert result.winner_name != result.loser_name
        assert result.loser_name in [ai_deep.name, ai_shallow.name]

    def test_move_history_not_empty(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path)
        assert result.num_moves >= 1
        assert len(result.move_history) == result.num_moves

    def test_nodes_explored_positive(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path)
        assert result.total_nodes > 0

    def test_elapsed_time_non_negative(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path)
        assert result.elapsed_seconds >= 0.0

    def test_graph_info_passed_through(self, ai_deep, ai_shallow, small_path):
        result = play_match(ai_deep, ai_shallow, small_path, graph_info="P5")
        assert result.graph_info == "P5"

    def test_triangle_player1_always_wins(self, ai_deep, ai_shallow, triangle):
        """On K_3 the first player always wins (one move ends the game)."""
        result = play_match(ai_deep, ai_shallow, triangle)
        assert result.winner_name == ai_deep.name
        assert result.num_moves == 1

    def test_graph_not_modified(self, ai_deep, ai_shallow, small_path):
        """The original graph must not be mutated."""
        nodes_before = set(small_path.nodes)
        edges_before = set(small_path.edges)
        play_match(ai_deep, ai_shallow, small_path)
        assert set(small_path.nodes) == nodes_before
        assert set(small_path.edges) == edges_before


# ======================================================================
# play_series
# ======================================================================

class TestPlaySeries:

    def test_correct_game_count(self, ai_deep, ai_shallow, small_path):
        result = play_series(ai_deep, ai_shallow, small_path, num_games=6)
        assert result.total_games == 6
        assert len(result.matches) == 6

    def test_wins_sum_to_total(self, ai_deep, ai_shallow, small_path):
        result = play_series(ai_deep, ai_shallow, small_path, num_games=4)
        total_wins = sum(result.wins.values())
        assert total_wins == result.total_games

    def test_win_rate_bounds(self, ai_deep, ai_shallow, small_path):
        result = play_series(ai_deep, ai_shallow, small_path, num_games=4)
        for name in [ai_deep.name, ai_shallow.name]:
            rate = result.win_rate(name)
            assert 0.0 <= rate <= 1.0

    def test_win_rate_unknown_player_raises(self, ai_deep, ai_shallow, small_path):
        result = play_series(ai_deep, ai_shallow, small_path, num_games=2)
        with pytest.raises(ValueError, match="Unknown player"):
            result.win_rate("NonExistent")

    def test_swap_sides(self, ai_deep, ai_shallow, small_path):
        """With swap_sides=True, players alternate who goes first."""
        result = play_series(
            ai_deep, ai_shallow, small_path,
            num_games=4, swap_sides=True,
        )
        # Game 0: Deep first, Game 1: Shallow first,
        # Game 2: Deep first, Game 3: Shallow first
        assert result.matches[0].player1_name == "Deep"
        assert result.matches[1].player1_name == "Shallow"
        assert result.matches[2].player1_name == "Deep"
        assert result.matches[3].player1_name == "Shallow"

    def test_no_swap(self, ai_deep, ai_shallow, small_path):
        """With swap_sides=False, player1 always goes first."""
        result = play_series(
            ai_deep, ai_shallow, small_path,
            num_games=3, swap_sides=False,
        )
        for match in result.matches:
            assert match.player1_name == "Deep"

    def test_graph_info_propagated(self, ai_deep, ai_shallow, small_path):
        result = play_series(
            ai_deep, ai_shallow, small_path,
            num_games=2, graph_info="P5",
        )
        assert result.graph_info == "P5"
        for match in result.matches:
            assert match.graph_info == "P5"


# ======================================================================
# run_tournament
# ======================================================================

class TestRunTournament:

    def test_basic_tournament(self, ai_deep, ai_shallow, small_path, triangle):
        """Two players, two graphs, 2 games each → 8 total games."""
        result = run_tournament(
            players=[ai_deep, ai_shallow],
            graphs=[(small_path, "P5"), (triangle, "K3")],
            games_per_series=2,
        )
        assert isinstance(result, TournamentResult)
        # 1 pair × 2 graphs × 2 games = 4 total
        assert result.total_games == 4
        assert sum(result.leaderboard.values()) == 4

    def test_three_players_round_robin(
        self, ai_deep, ai_shallow, ai_move_count, small_path
    ):
        """Three players → C(3,2) = 3 pairings."""
        result = run_tournament(
            players=[ai_deep, ai_shallow, ai_move_count],
            graphs=[(small_path, "P5")],
            games_per_series=2,
        )
        # 3 pairings × 1 graph × 2 games = 6 total
        assert result.total_games == 6
        assert len(result.series_results) == 3

    def test_leaderboard_has_all_players(
        self, ai_deep, ai_shallow, ai_move_count, small_path
    ):
        result = run_tournament(
            players=[ai_deep, ai_shallow, ai_move_count],
            graphs=[(small_path, "P5")],
            games_per_series=2,
        )
        assert set(result.leaderboard.keys()) == {"Deep", "Shallow", "MoveCount"}

    def test_win_rate(self, ai_deep, ai_shallow, small_path):
        result = run_tournament(
            players=[ai_deep, ai_shallow],
            graphs=[(small_path, "P5")],
            games_per_series=4,
        )
        for name in ["Deep", "Shallow"]:
            rate = result.win_rate(name)
            assert 0.0 <= rate <= 1.0

    def test_win_rate_unknown_raises(self, ai_deep, ai_shallow, small_path):
        result = run_tournament(
            players=[ai_deep, ai_shallow],
            graphs=[(small_path, "P5")],
            games_per_series=2,
        )
        with pytest.raises(ValueError, match="Unknown player"):
            result.win_rate("Ghost")

    def test_summary_string(self, ai_deep, ai_shallow, small_path):
        result = run_tournament(
            players=[ai_deep, ai_shallow],
            graphs=[(small_path, "P5")],
            games_per_series=2,
        )
        summary = result.summary()
        assert "Tournament Summary" in summary
        assert "Deep" in summary
        assert "Shallow" in summary

    def test_fewer_than_two_players_raises(self, small_path):
        ai = AIPlayer(name="Solo")
        with pytest.raises(ValueError, match="at least 2"):
            run_tournament(
                players=[ai],
                graphs=[(small_path, "P5")],
            )

    def test_duplicate_names_raises(self, small_path):
        a1 = AIPlayer(name="Same")
        a2 = AIPlayer(name="Same")
        with pytest.raises(ValueError, match="unique names"):
            run_tournament(
                players=[a1, a2],
                graphs=[(small_path, "P5")],
            )


# ======================================================================
# Determinism: exact AIs on the same graph produce consistent results
# ======================================================================

class TestDeterminism:

    def test_same_result_twice(self, small_path):
        """Two identical tournaments must produce the same leaderboard."""
        ai_a = AIPlayer(max_depth=math.inf, name="A")
        ai_b = AIPlayer(max_depth=math.inf, name="B")

        r1 = run_tournament(
            players=[ai_a, ai_b],
            graphs=[(small_path, "P5")],
            games_per_series=4,
            swap_sides=True,
        )
        r2 = run_tournament(
            players=[ai_a, ai_b],
            graphs=[(small_path, "P5")],
            games_per_series=4,
            swap_sides=True,
        )
        assert r1.leaderboard == r2.leaderboard


# ======================================================================
# Integration: different configs produce different outcomes
# ======================================================================

class TestDifferentConfigs:

    def test_depth_advantage(self):
        """A deeper-searching AI should win at least as often against
        a very shallow AI on a non-trivial graph."""
        deep = AIPlayer(max_depth=8, name="Deep8")
        shallow = AIPlayer(max_depth=1, name="Shallow1")
        graph = nx.path_graph(7)

        result = play_series(
            deep, shallow, graph,
            num_games=4, swap_sides=True, graph_info="P7",
        )
        # Deep should dominate or at least not lose all
        assert result.wins["Deep8"] >= result.wins["Shallow1"]
