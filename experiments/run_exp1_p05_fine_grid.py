"""Supplementary Experiment 1 sweep for the dense ``p = 0.5`` case.

This script preserves the original Experiment 1 outputs.  It repeats the
published settings for every integer graph size from 10 through 20, and saves
both game-level observations and an aggregated length distribution under new
filenames in ``experiments/results``.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_set_game.ai import AIPlayer
from stable_set_game.generators import erdos_renyi_graph
from stable_set_game.self_play import play_match


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV = RESULTS_DIR / "exp1_p05_fine_grid_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "exp1_p05_fine_grid_summary.csv"
FIGURE_PNG = RESULTS_DIR / "exp1_p05_fine_grid.png"


def run_fine_grid(
    n_values=range(10, 21),
    p: float = 0.5,
    games_per_n: int = 100,
    ai_depth: int = 3,
) -> tuple[List[Dict], List[Dict]]:
    """Run the supplementary sweep using the original Experiment 1 setup."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ai1 = AIPlayer(max_depth=ai_depth, name="AI-1")
    ai2 = AIPlayer(max_depth=ai_depth, name="AI-2")
    raw_rows: List[Dict] = []
    summaries: List[Dict] = []

    for n in n_values:
        lengths: List[int] = []
        p1_wins = 0

        for seed in range(games_per_n):
            graph = erdos_renyi_graph(n, p, seed=seed)
            result = play_match(
                ai1,
                ai2,
                graph,
                graph_info=f"ER(n={n},p={p},seed={seed})",
            )
            p1_win = result.winner_name == ai1.name
            lengths.append(result.num_moves)
            p1_wins += int(p1_win)
            raw_rows.append(
                {
                    "n": n,
                    "p": p,
                    "seed": seed,
                    "ai_depth": ai_depth,
                    "num_moves": result.num_moves,
                    "winner": 1 if p1_win else 2,
                    "nodes_explored": result.total_nodes,
                    "elapsed_seconds": round(result.elapsed_seconds, 6),
                }
            )

        counts = Counter(lengths)
        summary = {
            "n": n,
            "p": p,
            "games": games_per_n,
            "ai_depth": ai_depth,
            "avg_game_length": mean(lengths),
            "min_game_length": min(lengths),
            "max_game_length": max(lengths),
            "p1_wins": p1_wins,
            "p1_win_rate": p1_wins / games_per_n,
            "odd_length_games": sum(c for length, c in counts.items() if length % 2),
            "even_length_games": sum(c for length, c in counts.items() if not length % 2),
            "length_counts": counts,
        }
        summaries.append(summary)
        print(
            f"n={n:2d}: mean={summary['avg_game_length']:.2f}, "
            f"P1={summary['p1_win_rate']:.0%}, "
            f"lengths={dict(sorted(counts.items()))}"
        )

    _write_csvs(raw_rows, summaries)
    _make_figure(summaries)
    return raw_rows, summaries


def _write_csvs(raw_rows: List[Dict], summaries: List[Dict]) -> None:
    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    observed_lengths = sorted(
        {length for row in summaries for length in row["length_counts"]}
    )
    fieldnames = [
        "n",
        "p",
        "games",
        "ai_depth",
        "avg_game_length",
        "min_game_length",
        "max_game_length",
        "p1_wins",
        "p1_win_rate",
        "odd_length_games",
        "even_length_games",
        *[f"length_{length}_count" for length in observed_lengths],
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary[key] for key in fieldnames if key in summary}
            for length in observed_lengths:
                row[f"length_{length}_count"] = summary["length_counts"].get(length, 0)
            writer.writerow(row)


def _make_figure(summaries: List[Dict]) -> None:
    ns = [row["n"] for row in summaries]
    means = [row["avg_game_length"] for row in summaries]
    win_rates = [row["p1_win_rate"] for row in summaries]
    observed_lengths = sorted(
        {length for row in summaries for length in row["length_counts"]}
    )

    fig = plt.figure(figsize=(10, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.2))
    length_axis = fig.add_subplot(grid[0, 0])
    win_axis = fig.add_subplot(grid[0, 1])
    distribution_axis = fig.add_subplot(grid[1, :])

    length_axis.plot(ns, means, marker="o", color="#2563EB", linewidth=2)
    length_axis.set_title("Average game length")
    length_axis.set_xlabel("Number of vertices, n")
    length_axis.set_ylabel("Moves")
    length_axis.set_xticks(ns)
    length_axis.grid(alpha=0.25)

    win_axis.plot(ns, win_rates, marker="o", color="#DC2626", linewidth=2)
    win_axis.axhline(0.5, color="#64748B", linestyle="--", linewidth=1)
    win_axis.set_title("First-player win rate")
    win_axis.set_xlabel("Number of vertices, n")
    win_axis.set_ylabel("Win rate")
    win_axis.set_ylim(0, 1)
    win_axis.set_xticks(ns)
    win_axis.grid(alpha=0.25)

    bottoms = [0.0] * len(ns)
    colours = plt.cm.viridis(
        [i / max(1, len(observed_lengths) - 1) for i in range(len(observed_lengths))]
    )
    for length, colour in zip(observed_lengths, colours):
        proportions = [row["length_counts"].get(length, 0) / row["games"] for row in summaries]
        distribution_axis.bar(
            ns, proportions, bottom=bottoms, label=str(length), color=colour
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, proportions)]
    distribution_axis.set_title("Game-length distribution")
    distribution_axis.set_xlabel("Number of vertices, n")
    distribution_axis.set_ylabel("Proportion of games")
    distribution_axis.set_xticks(ns)
    distribution_axis.set_ylim(0, 1)
    distribution_axis.legend(title="Moves", ncols=len(observed_lengths), fontsize=8)

    fig.suptitle("Supplementary Experiment 1: Erdős–Rényi G(n, 0.5), 100 games per n")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_fine_grid()
