"""Tests for the fast recurrence-based fork Grundy computation.

The recurrence in ``experiments/fork_grundy_fast.py`` must reproduce the
brute-force game-tree results exactly.  We validate against an independent
brute-force solver (``experiments.fork_grundy.grundy_value``) for F_3 … F_22
(kept under ~2 s) and check that reported winning first moves really do lead
to Grundy-0 positions.
"""

import sys
from pathlib import Path

# Make the project root importable (so ``experiments`` is a package).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fork_grundy import grundy_value  # brute force (independent)
from experiments.fork_grundy_fast import (
    compute_fork_grundy,
    compute_path_grundy,
    fork_grundy_value,
    winning_first_moves,
    find_eventual_period,
)
from stable_set_game.generators import fork_graph


# Precompute once for the whole module (cheap).
_P = compute_path_grundy(60)
_FORK = compute_fork_grundy(60, _P)


# ======================================================================
# Path recurrence
# ======================================================================

class TestPathGrundy:

    def test_small_values(self):
        # G(P_0..P_5) = 0,1,1,2,0,3 (standard Node Kayles / Dawson sequence).
        assert _P[0:6] == [0, 1, 1, 2, 0, 3]

    def test_known_zeros(self):
        zeros = {m for m in range(35) if _P[m] == 0}
        for z in (0, 4, 8, 14, 20, 24, 28, 34):
            assert z in zeros


# ======================================================================
# CRITICAL: recurrence vs brute force
# ======================================================================

class TestForkRecurrenceMatchesBruteForce:

    def test_f3_to_f22_match_bruteforce(self):
        for k in range(3, 23):
            brute = grundy_value(fork_graph(k))
            rec = fork_grundy_value(k, _P, _FORK)
            assert rec == brute, (
                f"F_{k}: recurrence={rec} but brute force={brute}"
            )

    def test_winner_classification(self):
        # The seven P2 (G=0) positions among F_3..F_22, from brute force.
        p2 = {k for k in range(3, 23)
              if fork_grundy_value(k, _P, _FORK) == 0}
        assert p2 == {5, 6, 10, 14, 15, 19, 20}


# ======================================================================
# Winning first moves
# ======================================================================

class TestWinningMoves:

    def test_winning_moves_lead_to_p_position(self):
        """Each reported winning move must yield a Grundy-0 position.

        We verify independently: delete the move's closed neighbourhood and
        brute-force the Grundy value of the remaining induced subgraph.
        """
        for k in [3, 7, 9, 11, 12, 16, 18, 22]:
            g = fork_grundy_value(k, _P, _FORK)
            if g == 0:
                continue  # P2 wins → no winning first move
            winners = winning_first_moves(k, _P, _FORK)
            assert winners, f"F_{k} is a P1 win but no winning move found"

            graph = fork_graph(k)
            for v in winners:
                closed = {v} | set(graph.neighbors(v))
                remaining = [u for u in graph.nodes if u not in closed]
                sub = graph.subgraph(remaining).copy()
                gg = 0 if sub.number_of_nodes() == 0 else grundy_value(sub)
                assert gg == 0, (
                    f"F_{k}: move {v} claimed winning but leaves Grundy {gg}"
                )

    def test_p2_positions_have_no_winning_move(self):
        for k in (5, 6, 10, 14):
            assert fork_grundy_value(k, _P, _FORK) == 0
            assert winning_first_moves(k, _P, _FORK) == []

    def test_claw_f3_all_moves_win(self):
        # F_3 is the claw K_{1,3}: every first move reaches a P-position.
        assert fork_grundy_value(3, _P, _FORK) == 1
        assert winning_first_moves(3, _P, _FORK) == [0, 1, 2, 3]


# ======================================================================
# Periodicity helper
# ======================================================================

class TestPeriodHelper:

    def test_detects_simple_period(self):
        seq = [9, 9, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        result = find_eventual_period(seq, max_period=10, min_repeats=3)
        assert result is not None
        period, _start = result
        assert period == 3

    def test_returns_none_for_aperiodic(self):
        seq = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]
        assert find_eventual_period(seq, max_period=4, min_repeats=4) is None
