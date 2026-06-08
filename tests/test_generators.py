"""Tests for graph generators."""

import pytest
import networkx as nx

from stable_set_game.generators import (
    path_graph,
    cycle_graph,
    erdos_renyi_graph,
    random_regular_graph,
    complete_graph,
    star_graph,
    fork_graph,
)


# ======================================================================
# Path graph
# ======================================================================

class TestPathGraph:

    def test_basic_properties(self):
        g = path_graph(5)
        assert g.number_of_nodes() == 5
        assert g.number_of_edges() == 4

    def test_is_connected(self):
        assert nx.is_connected(path_graph(4))

    def test_endpoints_have_degree_1(self):
        g = path_graph(6)
        assert g.degree(0) == 1
        assert g.degree(5) == 1

    def test_interior_nodes_have_degree_2(self):
        g = path_graph(5)
        for v in range(1, 4):
            assert g.degree(v) == 2

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            path_graph(1)
        with pytest.raises(ValueError):
            path_graph(0)


# ======================================================================
# Cycle graph
# ======================================================================

class TestCycleGraph:

    def test_basic_properties(self):
        g = cycle_graph(5)
        assert g.number_of_nodes() == 5
        assert g.number_of_edges() == 5

    def test_all_degree_2(self):
        g = cycle_graph(6)
        for v in g.nodes:
            assert g.degree(v) == 2

    def test_is_connected(self):
        assert nx.is_connected(cycle_graph(7))

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            cycle_graph(2)


# ======================================================================
# Erdős–Rényi G(n, p)
# ======================================================================

class TestErdosRenyi:

    def test_node_count(self):
        g = erdos_renyi_graph(10, 0.5, seed=42)
        assert g.number_of_nodes() == 10

    def test_p_zero_gives_no_edges(self):
        g = erdos_renyi_graph(10, 0.0, seed=42)
        assert g.number_of_edges() == 0

    def test_p_one_gives_complete(self):
        g = erdos_renyi_graph(5, 1.0, seed=42)
        assert g.number_of_edges() == 10  # C(5,2) = 10

    def test_seed_reproducibility(self):
        g1 = erdos_renyi_graph(20, 0.3, seed=123)
        g2 = erdos_renyi_graph(20, 0.3, seed=123)
        assert set(g1.edges) == set(g2.edges)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            erdos_renyi_graph(5, -0.1)
        with pytest.raises(ValueError):
            erdos_renyi_graph(5, 1.5)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            erdos_renyi_graph(0, 0.5)


# ======================================================================
# Random regular graph
# ======================================================================

class TestRandomRegular:

    def test_all_same_degree(self):
        g = random_regular_graph(10, 3, seed=42)
        for v in g.nodes:
            assert g.degree(v) == 3

    def test_node_count(self):
        g = random_regular_graph(8, 2, seed=42)
        assert g.number_of_nodes() == 8

    def test_seed_reproducibility(self):
        g1 = random_regular_graph(10, 4, seed=99)
        g2 = random_regular_graph(10, 4, seed=99)
        assert set(g1.edges) == set(g2.edges)

    def test_odd_product_raises(self):
        """n * d must be even."""
        with pytest.raises(ValueError):
            random_regular_graph(5, 3)  # 5 * 3 = 15 (odd)

    def test_degree_too_large_raises(self):
        with pytest.raises(ValueError):
            random_regular_graph(5, 5)

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError):
            random_regular_graph(4, -1)

    def test_n_too_small_raises(self):
        with pytest.raises(ValueError):
            random_regular_graph(1, 0)


# ======================================================================
# Complete graph
# ======================================================================

class TestCompleteGraph:

    def test_edge_count(self):
        g = complete_graph(5)
        assert g.number_of_edges() == 10

    def test_all_pairs_adjacent(self):
        g = complete_graph(4)
        for u in g.nodes:
            for v in g.nodes:
                if u != v:
                    assert g.has_edge(u, v)

    def test_single_node(self):
        g = complete_graph(1)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            complete_graph(0)


# ======================================================================
# Star graph
# ======================================================================

class TestStarGraph:

    def test_node_count(self):
        g = star_graph(4)
        assert g.number_of_nodes() == 5  # 4 leaves + 1 centre

    def test_centre_degree(self):
        g = star_graph(5)
        assert g.degree(0) == 5  # centre connected to all leaves

    def test_leaf_degree(self):
        g = star_graph(3)
        for v in range(1, 4):
            assert g.degree(v) == 1

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            star_graph(0)


# ======================================================================
# Fork graph
# ======================================================================

class TestForkGraph:

    def test_node_count(self):
        g = fork_graph(5)
        assert g.number_of_nodes() == 6  # n + 1

    def test_edge_count(self):
        g = fork_graph(5)
        assert g.number_of_edges() == 5  # (n-1) path edges + 1 fork edge

    def test_fork_edge_exists(self):
        """Extra vertex n must be attached to vertex n-2."""
        n = 7
        g = fork_graph(n)
        assert g.has_edge(n - 2, n)

    def test_fork_point_has_degree_3(self):
        """Vertex n-2 is the fork point (branching vertex)."""
        n = 6
        g = fork_graph(n)
        assert g.degree(n - 2) == 3

    def test_extra_vertex_is_leaf(self):
        """The attached vertex n has degree 1."""
        n = 8
        g = fork_graph(n)
        assert g.degree(n) == 1

    def test_path_endpoint_zero_has_degree_1(self):
        g = fork_graph(6)
        assert g.degree(0) == 1

    def test_last_path_vertex_has_degree_1(self):
        """Vertex n-1 is adjacent only to n-2."""
        n = 6
        g = fork_graph(n)
        assert g.degree(n - 1) == 1

    def test_is_connected(self):
        assert nx.is_connected(fork_graph(5))

    def test_f3_structure(self):
        """F_3: path 0-1-2, extra vertex 3 attached to 1 — a star S_3."""
        g = fork_graph(3)
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 3
        assert g.degree(1) == 3

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            fork_graph(2)
        with pytest.raises(ValueError):
            fork_graph(0)


# ======================================================================
# Integration: generators + engine
# ======================================================================

class TestGeneratorsWithEngine:
    """Verify that generated graphs work correctly with the game engine."""

    def test_play_on_cycle(self):
        """Play a game on C_4 and verify it terminates."""
        from stable_set_game.engine import StableSetGame

        g = cycle_graph(4)  # 0 -- 1 -- 2 -- 3 -- 0
        game = StableSetGame(g)
        game.make_move(0)  # blocks 1 and 3
        assert game.legal_moves() == [2]
        game.make_move(2)  # blocks 1 and 3 (already blocked)
        assert game.is_game_over() is True
        assert game.get_winner() == 2

    def test_play_on_star(self):
        """On a star, picking the centre blocks all leaves."""
        from stable_set_game.engine import StableSetGame

        g = star_graph(4)
        game = StableSetGame(g)
        game.make_move(0)  # centre — blocks all leaves
        assert game.is_game_over() is True
        assert game.get_winner() == 1

    def test_play_on_complete(self):
        """On K_5, one move ends the game."""
        from stable_set_game.engine import StableSetGame

        g = complete_graph(5)
        game = StableSetGame(g)
        game.make_move(2)
        assert game.is_game_over() is True
