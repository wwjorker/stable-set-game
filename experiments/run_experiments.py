"""
Dissertation experiments for the Stable Set Game.

Three experiments:

Experiment 1 — Graph Selection
    For each combination of n in {10, 15, 20} and p in
    {0.1, 0.2, 0.3, 0.4, 0.5}, play 100 AI-vs-AI games on freshly
    sampled Erdős–Rényi graphs.  Report average game length and
    player-1 win rate.

Experiment 2 — Self-play Parameter Tuning
    Five AI configurations with different evaluation-function weights
    compete in a round-robin tournament (50 games per pair) on a
    single G(15, 0.3) graph.  Rank by overall win rate.

Experiment 3 — Theory vs AI
    Use exact minimax on path graphs P_3 … P_12 and cycle graphs
    C_3 … C_12 to determine the winner under optimal play.  Compare
    with the Grundy-theoretic prediction (computed independently via
    recursive Sprague–Grundy analysis on the engine).

All raw data are saved to CSV in ``experiments/results/``; each
experiment also produces a Matplotlib chart saved as a PNG.
"""

from __future__ import annotations

import csv
import math
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
import numpy as np

from stable_set_game.ai import AIPlayer
from stable_set_game.engine import StableSetGame
from stable_set_game.evaluation import WeightedEvaluator
from stable_set_game.generators import (
    cycle_graph,
    erdos_renyi_graph,
    path_graph,
)
from stable_set_game.self_play import play_match, run_tournament


# ======================================================================
# Output paths
# ======================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Experiment 1: Graph selection (ER density sweep)
# ======================================================================

def experiment_1_graph_selection(
    n_values: List[int] = (10, 15, 20),
    p_values: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    games_per_cell: int = 100,
    ai_depth: int = 3,
) -> List[Dict]:
    """Sweep (n, p) space on Erdős–Rényi graphs.

    For each (n, p) cell we play ``games_per_cell`` AI-vs-AI games,
    each on a freshly sampled G(n, p).  Returns a list of dicts
    with one entry per cell.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Graph Selection on Erdos-Renyi graphs")
    print("=" * 60)

    ai1 = AIPlayer(max_depth=ai_depth, name="AI-1")
    ai2 = AIPlayer(max_depth=ai_depth, name="AI-2")

    rows: List[Dict] = []

    for n in n_values:
        for p in p_values:
            t0 = time.perf_counter()
            p1_wins = 0
            total_moves = 0

            for g_idx in range(games_per_cell):
                graph = erdos_renyi_graph(n, p, seed=g_idx)
                # Guard against empty graphs (p=0 with no edges still valid)
                result = play_match(ai1, ai2, graph, graph_info=f"ER(n={n},p={p})")
                total_moves += result.num_moves
                if result.winner_name == ai1.name:
                    p1_wins += 1

            elapsed = time.perf_counter() - t0
            row = {
                "n": n,
                "p": p,
                "games": games_per_cell,
                "p1_win_rate": p1_wins / games_per_cell,
                "avg_game_length": total_moves / games_per_cell,
                "elapsed_seconds": round(elapsed, 2),
            }
            rows.append(row)
            print(
                f"  n={n:<3} p={p:.2f}  "
                f"P1 win rate={row['p1_win_rate']:.2%}  "
                f"avg length={row['avg_game_length']:.2f}  "
                f"[{elapsed:.1f}s]"
            )

    # Save CSV
    csv_path = RESULTS_DIR / "exp1_graph_selection.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # Chart
    _chart_experiment_1(rows, n_values, p_values)
    return rows


def _chart_experiment_1(rows: List[Dict], n_values, p_values) -> None:
    """Two-panel chart: P1 win rate and average game length vs p, per n."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for n in n_values:
        xs = [r["p"] for r in rows if r["n"] == n]
        win_rates = [r["p1_win_rate"] for r in rows if r["n"] == n]
        lengths = [r["avg_game_length"] for r in rows if r["n"] == n]

        axes[0].plot(xs, win_rates, marker="o", linewidth=2, label=f"n={n}")
        axes[1].plot(xs, lengths, marker="s", linewidth=2, label=f"n={n}")

    axes[0].axhline(0.5, linestyle="--", color="grey", alpha=0.7,
                    label="50% (fair)")
    axes[0].set_xlabel("Edge probability p")
    axes[0].set_ylabel("Player 1 win rate")
    axes[0].set_title("Exp 1: Player 1 win rate on G(n, p)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Edge probability p")
    axes[1].set_ylabel("Average game length (moves)")
    axes[1].set_title("Exp 1: Average game length on G(n, p)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    png_path = RESULTS_DIR / "exp1_graph_selection.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Experiment 2: Self-play parameter tuning
# ======================================================================

def experiment_2_parameter_tuning(
    n: int = 15,
    p: float = 0.3,
    games_per_pair: int = 50,
    ai_depth: int = 3,
) -> Dict:
    """Round-robin tournament of 5 evaluation configs on G(n, p).

    Each of the ``games_per_pair`` games uses a freshly sampled
    G(n, p) with a different seed, so that the ranking reflects the
    configs' performance across the **distribution** rather than on
    a single (deterministic) sample.  With deterministic AIs on a
    fixed graph the outcome is fully determined by who goes first,
    which leaves every pair tied at 50/50 and is uninformative.
    """
    print("\n" + "=" * 60)
    print(f"Experiment 2: Self-play Parameter Tuning on G({n}, {p})")
    print("=" * 60)

    configs = [
        ("Move-Count", {"move_count": 1.0}),
        ("Parity-Heavy", {"move_parity": 3.0, "move_count": 1.0}),
        ("Degree-Aware", {"degree_sum": 1.0, "move_count": 0.5}),
        ("Blocker", {"blocked_ratio": 2.0, "move_count": 1.0}),
        ("Balanced",
         {"move_count": 1.0, "move_parity": 1.0,
          "degree_sum": 0.3, "blocked_ratio": 0.5}),
    ]

    players = [
        AIPlayer(
            max_depth=ai_depth,
            evaluator=WeightedEvaluator(weights=dict(w)),
            name=name,
        )
        for name, w in configs
    ]

    print(f"  Sampling a fresh G({n}, {p}) for each game "
          f"({games_per_pair} seeds per pair)")

    # We do a manual round-robin so we can vary the graph each game.
    leaderboard: Dict[str, int] = {p.name: 0 for p in players}
    games_played: Dict[str, int] = {p.name: 0 for p in players}

    t0 = time.perf_counter()
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            p1, p2 = players[i], players[j]
            for k in range(games_per_pair):
                graph = erdos_renyi_graph(n, p, seed=k)
                # Alternate first mover to remove first-player advantage bias.
                if k % 2 == 0:
                    first, second = p1, p2
                else:
                    first, second = p2, p1
                result = play_match(first, second, graph,
                                    graph_info=f"ER(n={n},p={p},seed={k})")
                leaderboard[result.winner_name] += 1
                games_played[p1.name] += 1
                games_played[p2.name] += 1
    elapsed = time.perf_counter() - t0
    total_games = sum(games_played.values()) // 2
    print(f"  Tournament finished in {elapsed:.1f}s, "
          f"{total_games} games played")

    # Build ranking rows
    rows: List[Dict] = []
    for name, wins in leaderboard.items():
        played = games_played[name]
        rows.append({
            "name": name,
            "wins": wins,
            "games_played": played,
            "win_rate": wins / played if played else 0.0,
            "weights": str(dict(dict(configs)[name])),
        })
    rows.sort(key=lambda r: r["win_rate"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank

    # Print a small leaderboard
    print()
    print(f"  {'Rank':<5} {'Name':<15} {'Wins':>6} {'Games':>6} {'Win%':>7}")
    print("  " + "-" * 48)
    for r in rows:
        print(f"  {r['rank']:<5} {r['name']:<15} {r['wins']:>6} "
              f"{r['games_played']:>6} {r['win_rate'] * 100:>6.1f}%")

    # Save CSV
    csv_path = RESULTS_DIR / "exp2_parameter_tuning.csv"
    fieldnames = ["rank", "name", "wins", "games_played", "win_rate", "weights"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Saved: {csv_path}")

    # Chart
    _chart_experiment_2(rows)
    return {"rows": rows, "tournament": result}


def _chart_experiment_2(rows: List[Dict]) -> None:
    """Horizontal bar chart of win rates."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names = [r["name"] for r in rows]
    rates = [r["win_rate"] for r in rows]
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    bars = ax.barh(names, rates, color=colours, edgecolor="black")
    ax.axvline(0.5, linestyle="--", color="grey", alpha=0.7,
               label="50% (random play)")
    ax.set_xlabel("Win rate")
    ax.set_title("Exp 2: Evaluation-function configuration ranking "
                 "(round-robin on G(15, 0.3))")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    # Annotate with percentage
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center")

    fig.tight_layout()
    png_path = RESULTS_DIR / "exp2_parameter_tuning.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Experiment 3: Theory vs AI (Grundy values on P_n and C_n)
# ======================================================================

def _grundy_value(graph: nx.Graph, selected: FrozenSet = frozenset()) -> int:
    """Compute the Sprague–Grundy value of a Node-Kayles position.

    Uses memoisation keyed on the selected frozenset.  The recursion:

        G(state) = mex { G(state ∪ {v}) : v is a legal move from state }

    and G = 0 if there are no legal moves (terminal losing position).
    """
    # Memo is populated per call via a helper
    memo: Dict[FrozenSet, int] = {}

    def _legal(state: FrozenSet) -> List:
        blocked = set()
        for v in state:
            blocked.add(v)
            blocked.update(graph.neighbors(v))
        return [v for v in graph.nodes if v not in blocked]

    def _g(state: FrozenSet) -> int:
        if state in memo:
            return memo[state]
        moves = _legal(state)
        if not moves:
            memo[state] = 0
            return 0
        children = {_g(state | {v}) for v in moves}
        # mex = minimum excludant
        m = 0
        while m in children:
            m += 1
        memo[state] = m
        return m

    return _g(frozenset(selected))


def experiment_3_theory_vs_ai(
    max_n: int = 12,
    min_n: int = 3,
) -> List[Dict]:
    """Compare AI's minimax winner with Sprague–Grundy prediction."""
    print("\n" + "=" * 60)
    print(f"Experiment 3: Theory vs AI on P_n, C_n (n={min_n}..{max_n})")
    print("=" * 60)

    ai = AIPlayer(max_depth=math.inf, name="Exact")

    rows: List[Dict] = []

    for family, builder in [("P", path_graph), ("C", cycle_graph)]:
        for n in range(min_n, max_n + 1):
            # C_n requires n >= 3, P_n requires n >= 2
            if family == "C" and n < 3:
                continue
            graph = builder(n)

            # Theoretical prediction via Grundy value
            g_value = _grundy_value(graph)
            theory_winner = 1 if g_value != 0 else 2

            # AI's empirical winner: P1 wins iff root score is +inf
            game = StableSetGame(graph)
            result = ai.choose_move(game)
            ai_winner = 1 if result.score == math.inf else 2

            # Simulate the full game length under optimal play
            game_sim = StableSetGame(graph)
            while not game_sim.is_game_over():
                r = ai.choose_move(game_sim)
                game_sim.make_move(r.best_move)
            game_length = len(game_sim.history)

            match = theory_winner == ai_winner
            rows.append({
                "graph": f"{family}_{n}",
                "family": family,
                "n": n,
                "grundy_value": g_value,
                "theory_winner": theory_winner,
                "ai_winner": ai_winner,
                "match": int(match),
                "game_length": game_length,
            })
            print(
                f"  {family}_{n:<2}  G={g_value:<2}  "
                f"theory=P{theory_winner}  AI=P{ai_winner}  "
                f"{'MATCH' if match else 'MISMATCH!'}  "
                f"length={game_length}"
            )

    # Save CSV
    csv_path = RESULTS_DIR / "exp3_theory_vs_ai.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # Chart
    _chart_experiment_3(rows)

    # Sanity summary
    total = len(rows)
    matches = sum(r["match"] for r in rows)
    print(f"\n  Theory-AI agreement: {matches}/{total} = {matches/total:.1%}")

    return rows


def _chart_experiment_3(rows: List[Dict]) -> None:
    """Grid of Grundy values and winners for P_n and C_n."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, family, title in [
        (axes[0], "P", "Path graphs P_n"),
        (axes[1], "C", "Cycle graphs C_n"),
    ]:
        fam_rows = [r for r in rows if r["family"] == family]
        ns = [r["n"] for r in fam_rows]
        grundy = [r["grundy_value"] for r in fam_rows]
        winners = [r["ai_winner"] for r in fam_rows]

        # Bar colour = winner (blue=P1, red=P2)
        colours = ["#1E88E5" if w == 1 else "#E53935" for w in winners]
        bars = ax.bar(ns, grundy, color=colours, edgecolor="black")

        for bar, g in zip(bars, grundy):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(g), ha="center", fontsize=9)

        ax.set_xlabel("n")
        ax.set_ylabel("Grundy value")
        ax.set_title(f"Exp 3: {title} — Grundy & winner")
        ax.set_xticks(ns)
        ax.grid(axis="y", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color="#1E88E5", label="P1 wins (G > 0)"),
            Patch(color="#E53935", label="P2 wins (G = 0)"),
        ]
        ax.legend(handles=legend, loc="upper left")

    fig.tight_layout()
    png_path = RESULTS_DIR / "exp3_theory_vs_ai.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Experiment 4: Stable set size vs maximum independent set
# ======================================================================

def _max_independent_set_size(graph: nx.Graph) -> int:
    """Exact maximum independent set size.

    Uses the duality |MIS(G)| = |ω(complement(G))|: a maximum clique in
    the complement graph is exactly a maximum independent set in *G*.
    NetworkX's ``find_cliques`` enumerates all maximal cliques via
    Bron–Kerbosch; taking the longest gives the maximum.  Exponential
    in the worst case, but comfortable for n ≤ ~25.
    """
    if graph.number_of_nodes() == 0:
        return 0
    if graph.number_of_edges() == 0:
        return graph.number_of_nodes()
    complement = nx.complement(graph)
    return max(len(c) for c in nx.find_cliques(complement))


def experiment_4_stable_set_size(
    n_values: List[int] = (10, 15, 20),
    p_values: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    games_per_cell: int = 100,
    ai_depth: int = 3,
) -> List[Dict]:
    """Compare the played stable-set size to the graph's MIS.

    For each (n, p) cell we sample ``games_per_cell`` fresh G(n, p)
    graphs (seeds 0 … games_per_cell-1), play one AI-vs-AI game on each,
    record the size of the resulting stable set (= number of chosen
    vertices), compute the exact MIS via complement-clique enumeration,
    and report the ratio game_size / mis_size in [0, 1].

    A ratio of 1.0 means optimal play terminated on a *maximum*
    independent set; lower ratios indicate sub-maximal terminal sets —
    which are still maximal (no vertex can be added) but smaller than
    the global MIS.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Stable Set Size vs Maximum Independent Set")
    print("=" * 60)

    ai1 = AIPlayer(max_depth=ai_depth, name="AI-1")
    ai2 = AIPlayer(max_depth=ai_depth, name="AI-2")

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for n in n_values:
        for p in p_values:
            t0 = time.perf_counter()
            ratios: List[float] = []
            game_sizes: List[int] = []
            mis_sizes: List[int] = []

            for g_idx in range(games_per_cell):
                graph = erdos_renyi_graph(n, p, seed=g_idx)
                result = play_match(
                    ai1, ai2, graph, graph_info=f"ER(n={n},p={p})"
                )

                game_size = result.num_moves  # each move adds 1 vertex
                mis_size = _max_independent_set_size(graph)
                ratio = game_size / mis_size if mis_size > 0 else 0.0

                ratios.append(ratio)
                game_sizes.append(game_size)
                mis_sizes.append(mis_size)

                detail_rows.append({
                    "n": n,
                    "p": p,
                    "seed": g_idx,
                    "game_stable_set_size": game_size,
                    "mis_size": mis_size,
                    "ratio": round(ratio, 4),
                })

            elapsed = time.perf_counter() - t0
            mean_ratio = float(np.mean(ratios))
            std_ratio = float(np.std(ratios))
            summary_rows.append({
                "n": n,
                "p": p,
                "games": games_per_cell,
                "mean_game_size": round(float(np.mean(game_sizes)), 3),
                "mean_mis_size": round(float(np.mean(mis_sizes)), 3),
                "mean_ratio": round(mean_ratio, 4),
                "std_ratio": round(std_ratio, 4),
                "min_ratio": round(min(ratios), 4),
                "max_ratio": round(max(ratios), 4),
                "elapsed_seconds": round(elapsed, 2),
            })
            print(
                f"  n={n:<3} p={p:.2f}  "
                f"mean game={np.mean(game_sizes):.2f}  "
                f"mean MIS={np.mean(mis_sizes):.2f}  "
                f"ratio={mean_ratio:.3f} ± {std_ratio:.3f}  "
                f"[{elapsed:.1f}s]"
            )

    # Save CSVs — summary per cell and per-game detail
    summary_path = RESULTS_DIR / "exp4_stable_set_size_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")

    detail_path = RESULTS_DIR / "exp4_stable_set_size_detail.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Saved: {detail_path}")

    _chart_experiment_4(summary_rows, n_values, p_values)
    return summary_rows


def _chart_experiment_4(rows: List[Dict], n_values, p_values) -> None:
    """Plot mean stable-set / MIS ratio vs p, one curve per n, with
    a ±1σ shaded band."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    for n in n_values:
        cell_rows = [r for r in rows if r["n"] == n]
        xs = [r["p"] for r in cell_rows]
        means = [r["mean_ratio"] for r in cell_rows]
        stds = [r["std_ratio"] for r in cell_rows]
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        line, = ax.plot(xs, means, marker="o", linewidth=2, label=f"n={n}")
        ax.fill_between(
            xs,
            np.clip(means_arr - stds_arr, 0, 1),
            np.clip(means_arr + stds_arr, 0, 1),
            alpha=0.15, color=line.get_color(),
        )

    ax.axhline(1.0, linestyle="--", color="grey", alpha=0.7,
               label="1.0 (optimal = MIS)")
    ax.set_xlabel("Edge probability p")
    ax.set_ylabel("Mean ratio  (game stable set size) / (MIS size)")
    ax.set_title(
        "Exp 4: How close does AI-vs-AI play get to the maximum "
        "independent set?"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    png_path = RESULTS_DIR / "exp4_stable_set_size.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    total_t0 = time.perf_counter()
    print(f"Results directory: {RESULTS_DIR}")

    experiment_1_graph_selection()
    experiment_2_parameter_tuning()
    experiment_3_theory_vs_ai()
    experiment_4_stable_set_size()

    print("\n" + "=" * 60)
    print(f"All experiments completed in "
          f"{time.perf_counter() - total_t0:.1f}s")
    print(f"Results saved under: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
