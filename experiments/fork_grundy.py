"""
Exhaustive Sprague–Grundy computation for fork graphs.

A fork graph F_n is the path P_n with one extra vertex (labelled n) attached
to vertex n-2.  This script computes the exact Grundy value g(F_n) for
n = 3 … 30 by exploring the *entire* game tree via memoised recursion on
game states (no AI / minimax involved).

Definitions
-----------
A game state in the Stable Set Game is the set of vertices already chosen
(the current independent set).  From a state S, the legal moves are the
vertices v such that S ∪ {v} is still an independent set.  A position is
terminal when no such v exists; its Grundy value is 0 (P-position: the
player to move loses under normal play).

The Grundy value of a non-terminal state is the **mex** (minimum
excludant) of the Grundy values of its successors:

        g(S) = mex { g(S ∪ {v}) : v is a legal move from S }

Because the game is impartial with no draws, g(F_n) > 0 means the first
player wins (N-position) and g(F_n) = 0 means the second player wins
(P-position).

Output
------
Prints a table of g(F_n) for n = 3 … 30, the first-player winner, and any
periodic patterns detected in the sequence.  Results are also saved to
``experiments/results/fork_grundy.csv`` and ``fork_grundy.png``.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, FrozenSet, List

# Allow running as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — save PNGs only
import matplotlib.pyplot as plt
import networkx as nx

from stable_set_game.generators import fork_graph


# Output directory (shared with the other experiments).
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Core Grundy computation
# ======================================================================

def _legal_moves(
    graph: nx.Graph,
    selected: FrozenSet[int],
) -> list[int]:
    """Return all vertices that can still be chosen (independent of *selected*)."""
    blocked = set(selected)
    for v in selected:
        blocked.update(graph.neighbors(v))
    return [v for v in graph.nodes if v not in blocked]


def _grundy(
    graph: nx.Graph,
    selected: FrozenSet[int],
    memo: Dict[FrozenSet[int], int],
) -> int:
    """Exact Grundy value of the position with *selected* chosen so far.

    Uses recursion + memoisation; the state space is at most 2^|V|, and
    only independent sets are ever visited in practice.
    """
    if selected in memo:
        return memo[selected]

    moves = _legal_moves(graph, selected)
    if not moves:
        memo[selected] = 0
        return 0

    successor_values = set()
    for v in moves:
        successor_values.add(_grundy(graph, selected | {v}, memo))

    # mex: smallest non-negative integer not in the set
    g = 0
    while g in successor_values:
        g += 1

    memo[selected] = g
    return g


def grundy_value(graph: nx.Graph) -> int:
    """Grundy value of the empty-selection starting position."""
    memo: Dict[FrozenSet[int], int] = {}
    return _grundy(graph, frozenset(), memo)


# ======================================================================
# Pattern analysis
# ======================================================================

def longest_periodic_suffix(
    seq: list[int],
    min_period: int = 2,
    max_period: int = 10,
) -> tuple[int, int] | None:
    """Find the shortest period *p* whose repeating block covers the longest
    suffix of *seq*.

    Returns ``(period, num_blocks)`` — the period length and how many full
    repetitions of the block are visible at the tail — or ``None`` if no
    non-trivial period fits at least two full blocks.

    Rationale: the fork-graph Grundy sequence may only become periodic
    after some "transient" at the start, so we only require the tail to
    be periodic, not the whole sequence.  We skip period 1 by default
    because any two consecutive equal values would match trivially.
    """
    best: tuple[int, int] | None = None
    for p in range(min_period, max_period + 1):
        # How many full repeating p-blocks fit at the end of seq?
        block = seq[-p:]
        k = 1
        while (k + 1) * p <= len(seq) and seq[-(k + 1) * p:-k * p] == block:
            k += 1
        if k >= 2:  # at least two matching blocks visible
            if best is None or k > best[1]:
                best = (p, k)
    return best


# ======================================================================
# Output: CSV + chart
# ======================================================================

def _save_csv(rows: List[Dict]) -> None:
    """Write the per-graph Grundy results to fork_grundy.csv."""
    csv_path = RESULTS_DIR / "fork_grundy.csv"
    fieldnames = ["n", "num_vertices", "num_edges", "grundy_value", "winner"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def _chart_fork_grundy(rows: List[Dict]) -> None:
    """Bar chart of Grundy values for F_n, colour-coded by winner.

    Blue bars = first player wins (Grundy > 0, N-position);
    red bars  = second player wins (Grundy = 0, P-position).  Because
    P-positions have Grundy value 0 their bars have zero height, so we
    also place a small red marker on the axis to keep them visible.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ns = [r["n"] for r in rows]
    grundy = [r["grundy_value"] for r in rows]
    colours = ["#1E88E5" if g > 0 else "#E53935" for g in grundy]

    bars = ax.bar(ns, grundy, color=colours, edgecolor="black")

    # Annotate each bar with its Grundy value.
    for bar, g in zip(bars, grundy):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(g), ha="center", fontsize=8)

    # Keep zero-height (P2-win) bars visible with a red marker at y=0.
    p2_ns = [n for n, g in zip(ns, grundy) if g == 0]
    if p2_ns:
        ax.scatter(p2_ns, [0] * len(p2_ns), color="#E53935",
                   marker="v", s=40, zorder=3)

    ax.set_xlabel("n  (fork graph F_n)")
    ax.set_ylabel("Grundy value  g(F_n)")
    ax.set_title("Sprague–Grundy values of fork graphs F_n")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#1E88E5", label="P1 wins (g > 0, N-position)"),
        Patch(color="#E53935", label="P2 wins (g = 0, P-position)"),
    ]
    ax.legend(handles=legend, loc="upper right")

    fig.tight_layout()
    png_path = RESULTS_DIR / "fork_grundy.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Driver
# ======================================================================

MAX_N = 30
PER_GRAPH_TIME_BUDGET = 60.0  # seconds — abort if one graph exceeds this


def main() -> None:
    print("=" * 68)
    print(f"Sprague–Grundy values for fork graphs F_3 … F_{MAX_N}")
    print(f"(aborts early if any single graph exceeds "
          f"{PER_GRAPH_TIME_BUDGET:.0f} s)")
    print("=" * 68)
    print()
    print(f"{'n':>3} {'|V|':>4} {'|E|':>4} {'g(F_n)':>8} {'winner':>8} {'states':>12} {'time(s)':>9}")
    print("-" * 58)

    values: list[int] = []
    rows: List[Dict] = []
    last_n = 2
    for n in range(3, MAX_N + 1):
        g = fork_graph(n)
        memo: Dict[FrozenSet[int], int] = {}
        t0 = time.perf_counter()
        _grundy(g, frozenset(), memo)
        elapsed = time.perf_counter() - t0
        gv = memo[frozenset()]
        values.append(gv)
        last_n = n
        winner = "P1 (N)" if gv > 0 else "P2 (P)"
        rows.append({
            "n": n,
            "num_vertices": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "grundy_value": gv,
            "winner": "P1" if gv > 0 else "P2",
        })
        print(
            f"{n:>3} {g.number_of_nodes():>4} {g.number_of_edges():>4} "
            f"{gv:>8} {winner:>8} {len(memo):>12} {elapsed:>9.3f}",
            flush=True,
        )
        if elapsed > PER_GRAPH_TIME_BUDGET:
            print(f"\n[stopping: F_{n} took {elapsed:.1f} s, "
                  f"exceeds {PER_GRAPH_TIME_BUDGET:.0f} s budget]")
            break

    # Persist results to CSV + PNG (matching the other experiments).
    _save_csv(rows)
    _chart_fork_grundy(rows)

    print()
    print(f"Grundy sequence (n = 3 … {last_n}):")
    print("  " + ", ".join(str(v) for v in values))
    print()

    # Winner pattern (N vs P positions)
    winner_seq = ["N" if v > 0 else "P" for v in values]
    print("Winner pattern (N = first-player win, P = second-player win):")
    print("  " + " ".join(winner_seq))
    print()

    # Try to spot a periodic tail in the Grundy sequence and the winner
    # sequence.  (A leading transient is allowed — we only require the
    # last k*p values to repeat.)
    period_vals = longest_periodic_suffix(values)
    period_winner = longest_periodic_suffix(
        [1 if v > 0 else 0 for v in values]
    )
    if period_vals is not None:
        p, k = period_vals
        tail = values[-p:]
        print(f"Grundy values show a periodic tail of period {p} "
              f"covering the last {k} blocks ({k * p} of {len(values)} "
              f"values): repeating block {tail}")
    else:
        print("No non-trivial periodic tail in the Grundy sequence "
              "(checked up to period 10).")

    if period_winner is not None:
        p, k = period_winner
        tail = winner_seq[-p:]
        print(f"Winner pattern shows a periodic tail of period {p} "
              f"covering the last {k} blocks ({k * p} of "
              f"{len(winner_seq)} values): repeating block {tail}")
    else:
        print("No non-trivial periodic tail in the N/P winner pattern "
              "(checked up to period 10).")

    print()
    print("Zero (P-)positions — second player wins on F_n for:")
    zeros = [n for n, v in zip(range(3, last_n + 1), values) if v == 0]
    print("  " + (", ".join(f"F_{n}" for n in zeros) if zeros else "(none)"))

    # The early values suggest a period-5 winner pattern from n = 11;
    # check whether it persists over the computed range.
    if len(values) >= 10:
        winner_bits = [1 if v > 0 else 0 for v in values]
        tail_start = 8  # corresponds to n = 11 (index 8 in n=3.. seq)
        if tail_start < len(winner_bits):
            tail = winner_bits[tail_start:]
            period5_ok = all(
                tail[i] == tail[i % 5] for i in range(len(tail))
            )
            last_n_covered = 3 + tail_start + len(tail) - 1
            status = "HOLDS" if period5_ok else "BREAKS"
            print(
                f"\nPeriod-5 winner pattern [N,N,N,P,P] starting at n=11 "
                f"through n={last_n_covered}: {status}."
            )


if __name__ == "__main__":
    main()
