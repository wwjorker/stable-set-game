"""Tests for evaluation functions."""

import math

import pytest
import networkx as nx

from stable_set_game.engine import StableSetGame
from stable_set_game.evaluation import (
    SCORE_LOSS,
    WeightedEvaluator,
    feature_move_difference,
    feature_blocked_ratio,
    feature_degree_sum,
    feature_move_parity,
    feature_component_count,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def path5_game() -> StableSetGame:
    """Fresh game on P_5: 0 -- 1 -- 2 -- 3 -- 4."""
    return StableSetGame(nx.path_graph(5))


@pytest.fixture
def triangle_game() -> StableSetGame:
    """Fresh game on K_3."""
    return StableSetGame(nx.complete_graph(3))


# ======================================================================
# feature_move_difference
# ======================================================================

class TestFeatureMoveDifference:

    def test_initial_path5(self, path5_game):
        """All 5 vertices are legal at the start."""
        assert feature_move_difference(path5_game) == 5.0

    def test_after_one_move(self, path5_game):
        path5_game.make_move(2)  # blocks 1 and 3
        # Legal: 0, 4
        assert feature_move_difference(path5_game) == 2.0

    def test_no_moves(self, triangle_game):
        triangle_game.make_move(0)
        assert feature_move_difference(triangle_game) == 0.0


# ======================================================================
# feature_blocked_ratio
# ======================================================================

class TestFeatureBlockedRatio:

    def test_no_blocked_at_start(self, path5_game):
        """At the start nothing is blocked → ratio = 0 → return -0.0."""
        val = feature_blocked_ratio(path5_game)
        assert val == pytest.approx(0.0)

    def test_partially_blocked(self, path5_game):
        path5_game.make_move(2)  # selected: {2}, blocked neighbours: {1, 3}
        # Free vertices (not selected): 0, 1, 3, 4 → 4 total
        # Legal among free: 0, 4 → 2
        # blocked_fraction = 1 - 2/4 = 0.5 → return -0.5
        val = feature_blocked_ratio(path5_game)
        assert val == pytest.approx(-0.5)


# ======================================================================
# feature_degree_sum
# ======================================================================

class TestFeatureDegreeSum:

    def test_initial_path5(self, path5_game):
        # Degrees in P_5: 0→1, 1→2, 2→2, 3→2, 4→1 → sum = 8
        assert feature_degree_sum(path5_game) == 8.0

    def test_after_move(self, path5_game):
        path5_game.make_move(2)
        # Legal moves: [0, 4], degrees: 1+1 = 2
        assert feature_degree_sum(path5_game) == 2.0


# ======================================================================
# feature_move_parity
# ======================================================================

class TestFeatureMoveParity:

    def test_odd_moves(self, path5_game):
        """5 legal moves → odd → +1.0."""
        assert feature_move_parity(path5_game) == 1.0

    def test_even_moves(self, path5_game):
        path5_game.make_move(2)  # 2 legal moves left → even
        assert feature_move_parity(path5_game) == -1.0


# ======================================================================
# feature_component_count
# ======================================================================

class TestFeatureComponentCount:

    def test_initial_path5(self, path5_game):
        """P_5 is one connected component at the start."""
        assert feature_component_count(path5_game) == 1.0

    def test_after_middle_move_splits_path(self, path5_game):
        """After selecting vertex 2 on P_5, the available set is {0, 4}
        (vertex 2 is selected; vertices 1 and 3 are blocked neighbours).
        Those two vertices form two isolated components.
        """
        path5_game.make_move(2)
        assert feature_component_count(path5_game) == 2.0

    def test_terminal_returns_zero(self, triangle_game):
        """No legal moves → 0 components."""
        triangle_game.make_move(0)
        assert feature_component_count(triangle_game) == 0.0

    def test_disconnected_graph(self):
        """Two disjoint edges → 2 components initially."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (2, 3)])
        game = StableSetGame(g)
        assert feature_component_count(game) == 2.0

    def test_star_collapses_to_one(self):
        """Before any move, a star S_3 (centre 0, leaves 1..3) is one
        component.  After selecting a leaf, the remaining legal moves are
        the other two leaves — the centre is blocked — giving 2 isolated
        components.
        """
        g = nx.star_graph(3)
        game = StableSetGame(g)
        assert feature_component_count(game) == 1.0
        game.make_move(1)  # selects a leaf, blocks the centre
        assert feature_component_count(game) == 2.0


# ======================================================================
# WeightedEvaluator
# ======================================================================

class TestWeightedEvaluator:

    def test_terminal_loss(self, triangle_game):
        """When current player has no moves, evaluator returns -inf."""
        triangle_game.make_move(0)
        evaluator = WeightedEvaluator()
        assert evaluator(triangle_game) == SCORE_LOSS

    def test_custom_weights(self, path5_game):
        """Score should equal the weighted sum of active features."""
        evaluator = WeightedEvaluator(weights={
            "move_count": 2.0,
            "move_parity": 3.0,
        })
        expected = 2.0 * 5.0 + 3.0 * 1.0  # 10 + 3 = 13
        assert evaluator(path5_game) == pytest.approx(expected)

    def test_single_feature(self, path5_game):
        evaluator = WeightedEvaluator(weights={"move_count": 1.0})
        assert evaluator(path5_game) == pytest.approx(5.0)

    def test_unknown_feature_raises(self, path5_game):
        """Referencing a non-existent feature should raise KeyError."""
        evaluator = WeightedEvaluator(weights={"nonexistent": 1.0})
        with pytest.raises(KeyError):
            evaluator(path5_game)

    def test_default_weights_work(self, path5_game):
        """Default evaluator should not crash."""
        evaluator = WeightedEvaluator()
        score = evaluator(path5_game)
        assert isinstance(score, float)
        assert math.isfinite(score)
