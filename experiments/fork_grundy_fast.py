"""
Fast recurrence-based Sprague–Grundy computation for fork graphs.

This replaces the brute-force game-tree enumeration in ``fork_grundy.py``
(exponential, only reaches ~F_30) with a polynomial recurrence built on the
Sprague–Grundy decomposition of Node Kayles.

Background
----------
``fork_graph(k)`` is the path P_k on vertices 0 … k-1 plus one extra leaf
(vertex k) attached to vertex k-2.  Using the analysis labelling:

    prong1 = k-1, prong2 = k   (two leaf tips on the branch)
    branch = k-2               (degree-3 fork point)
    tail   = k-3, k-4, …, 0    (the path hanging off the branch)

We parametrise a fork by its **tail length** ``t`` = number of path vertices
below the branch, so ``F_k = Fork_{t = k-2}``.  ``Fork_t`` has ``t + 3``
vertices (branch + 2 prongs + a tail path of t vertices).

Key insight (Node Kayles = closed-neighbourhood deletion)
---------------------------------------------------------
Selecting a vertex v deletes its closed neighbourhood N[v] = {v} ∪ neighbours.
The remaining graph splits into connected components whose Grundy values XOR
together.  So we only need Grundy values of the pieces produced by each move:
isolated vertices (G = 1), paths (precomputed), and smaller fork remnants
(already computed).

Path recurrence
---------------
    G(P_m) = mex over i=0..m-1 of { G(P_{i-1}) XOR G(P_{m-i-2}) }
with G(P_m) = 0 for m <= 0.

Fork recurrence (tail length t)
-------------------------------
Label the tail a_1 (adjacent to branch) … a_t (far endpoint).  Each move and
the Grundy value of the resulting position:

    select prong1 / prong2 : 1 XOR P(t)          (other prong isolated + tail path)
    select branch          : P(t-1)              (both prongs + a_1 deleted)
    select a_1             : P(t-2)              (branch deleted ⇒ 2 isolated prongs cancel)
    select a_i (2≤i≤t-1)   : Fork(i-2) XOR P(t-i-1)   (smaller fork above + path below)
    select a_t (endpoint)  : Fork(t-2)           (only when t ≥ 2)

    Fork(t) = mex over all of the above.

Base cases fall out naturally: Fork_0 is the claw-less stub P_3 (G = 2),
Fork_1 = F_3 (G = 1).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — save PNGs only
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Path Grundy values (Node Kayles on P_m — the octal game 0.137)
# ======================================================================

def compute_path_grundy(max_m: int) -> List[int]:
    """Return ``P`` where ``P[m] = G(P_m)`` for m = 0 … max_m.

    Selecting vertex i (0-indexed) in P_m deletes i-1, i, i+1, leaving two
    sub-paths with i-1 and m-i-2 vertices respectively.
    """
    P = [0] * (max_m + 1)  # P[0] = 0 (empty graph)
    for m in range(1, max_m + 1):
        reach = set()
        for i in range(m):
            left = i - 1 if i >= 1 else 0      # vertices 0 … i-2  (i-1 of them)
            right = m - i - 2                  # vertices i+2 … m-1
            if right < 0:
                right = 0
            reach.add(P[left] ^ P[right])
        g = 0
        while g in reach:
            g += 1
        P[m] = g
    return P


# ======================================================================
# Fork Grundy values via the decomposition recurrence
# ======================================================================

def _pget(P: List[int], m: int) -> int:
    """Path Grundy with the convention G(P_m) = 0 for m <= 0."""
    return P[m] if m > 0 else 0


def _fork_move_values(t: int, P: List[int], Fork: List[int]) -> List[int]:
    """Grundy values of every position reachable by one move on Fork_t.

    ``Fork`` must already hold the values Fork[0 … t-1].
    """
    vals: List[int] = []
    # Prong tips: other prong becomes isolated, the tail stays a path P_t.
    vals.append(1 ^ _pget(P, t))      # prong1
    vals.append(1 ^ _pget(P, t))      # prong2
    # Branch: deletes both prongs and a_1, leaving P_{t-1}.
    vals.append(_pget(P, t - 1))
    if t >= 1:
        # a_1 (adjacent to branch): branch deleted ⇒ both prongs isolated
        # (their two 1s cancel under XOR), leaving P_{t-2}.
        vals.append(_pget(P, t - 2))
    # Middle tail vertices a_i, 2 ≤ i ≤ t-1: smaller fork above + path below.
    for i in range(2, t):
        vals.append(Fork[i - 2] ^ _pget(P, t - i - 1))
    if t >= 2:
        # Far endpoint a_t: leaves Fork_{t-2}.
        vals.append(Fork[t - 2])
    return vals


def compute_fork_grundy(max_t: int, P: List[int]) -> List[int]:
    """Return ``Fork`` where ``Fork[t] = G(Fork_t)`` for t = 0 … max_t.

    Requires ``len(P) > max_t``.
    """
    Fork = [0] * (max_t + 1)
    for t in range(max_t + 1):
        reach = set(_fork_move_values(t, P, Fork))
        g = 0
        while g in reach:
            g += 1
        Fork[t] = g
    return Fork


def fork_grundy_value(k: int, P: List[int], Fork: List[int]) -> int:
    """Grundy value of the actual fork graph F_k (k >= 3)."""
    return Fork[k - 2]


def winning_first_moves(k: int, P: List[int], Fork: List[int]) -> List[int]:
    """Vertices of F_k whose selection moves to a Grundy-0 (P-)position.

    These are exactly the winning first moves for player 1 (only meaningful
    when G(F_k) > 0).  Vertex labels match ``fork_graph(k)``:
    branch = k-2, prongs = k-1 and k, tail a_i = (k-2) - i (so a_1 = k-3,
    a_t = vertex 0).
    """
    t = k - 2
    moves: List[Tuple[int, int]] = []
    moves.append((k - 1, 1 ^ _pget(P, t)))        # prong1
    moves.append((k, 1 ^ _pget(P, t)))            # prong2
    moves.append((k - 2, _pget(P, t - 1)))        # branch
    if t >= 1:
        moves.append((k - 3, _pget(P, t - 2)))    # a_1
    for i in range(2, t):
        moves.append(((k - 2) - i, Fork[i - 2] ^ _pget(P, t - i - 1)))  # a_i
    if t >= 2:
        moves.append((0, Fork[t - 2]))            # a_t (endpoint)
    return sorted({v for v, val in moves if val == 0})


# ======================================================================
# Periodicity detection
# ======================================================================

def find_eventual_period(
    seq: List[int],
    max_period: int = 80,
    min_repeats: int = 4,
) -> Tuple[int, int] | None:
    """Detect an eventually-periodic tail.

    Returns ``(period, preperiod_start_index)`` for the smallest period whose
    repeating block covers at least ``min_repeats`` full repetitions at the
    end of ``seq``, or ``None`` if no such period is found.
    """
    n = len(seq)
    best: Tuple[int, int] | None = None
    best_matched = -1
    for p in range(1, max_period + 1):
        # Count how far back from the end the p-periodicity holds.
        matched = 0
        i = n - 1
        while i - p >= 0 and seq[i] == seq[i - p]:
            matched += 1
            i -= 1
        if matched >= min_repeats * p:
            # Prefer the period covering the longest tail (the fundamental
            # period covers more than a spuriously short one); break ties
            # toward the smaller period.
            start = max(n - matched - p, 0)
            if matched > best_matched:
                best_matched = matched
                best = (p, start)
    return best


# ======================================================================
# Validation against brute force
# ======================================================================

def validate_against_csv(P: List[int], Fork: List[int]) -> bool:
    """Compare recurrence values to the brute-force fork_grundy.csv.

    Returns True if every k = 3 … 30 matches exactly, else prints the
    mismatches and returns False.
    """
    csv_path = RESULTS_DIR / "fork_grundy.csv"
    if not csv_path.exists():
        print(f"[validation] {csv_path} not found — skipping CSV check.")
        return True

    brute: Dict[int, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            brute[int(row["n"])] = int(row["grundy_value"])

    mismatches = []
    for k in sorted(brute):
        rec = fork_grundy_value(k, P, Fork)
        if rec != brute[k]:
            mismatches.append((k, brute[k], rec))

    if mismatches:
        print("[validation] MISMATCH between recurrence and brute force:")
        for k, b, r in mismatches:
            print(f"    F_{k}: brute={b}  recurrence={r}")
        return False

    print(f"[validation] PASS — recurrence matches brute force for all "
          f"{len(brute)} graphs F_3 … F_{max(brute)}.")
    return True


# ======================================================================
# Output: CSV + chart
# ======================================================================

def _save_csv(rows: List[Dict]) -> None:
    csv_path = RESULTS_DIR / "fork_grundy_extended.csv"
    fieldnames = ["k", "num_vertices", "num_edges", "grundy_value",
                  "winner", "winning_first_moves"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def _chart_extended(ks: List[int], grundy: List[int]) -> None:
    """Two panels: Grundy values (scatter) and a winner strip chart.

    With up to ~500 values a bar chart is unreadable, so the bottom panel is
    a colour strip (blue = P1 wins, red = P2 wins).
    """
    winner_bits = [1 if g > 0 else 0 for g in grundy]  # 1 = P1, 0 = P2
    p2_ks = [k for k, g in zip(ks, grundy) if g == 0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 6), height_ratios=[3, 1], sharex=True
    )

    # --- Top: Grundy values, coloured by winner -----------------------
    colours = ["#1E88E5" if g > 0 else "#E53935" for g in grundy]
    ax1.scatter(ks, grundy, c=colours, s=12, zorder=3)
    # Make the rare P2 wins (g = 0) visible with red markers on the axis.
    if p2_ks:
        ax1.scatter(p2_ks, [0] * len(p2_ks), color="#E53935",
                    marker="v", s=45, zorder=4)
    ax1.set_ylabel("Grundy value g(F_k)")
    ax1.set_title("Sprague–Grundy values of fork graphs F_k (recurrence, k = 3 … "
                  f"{ks[-1]})")
    ax1.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color="#1E88E5", label="P1 wins (g > 0)"),
        Patch(color="#E53935", label="P2 wins (g = 0)"),
    ], loc="upper right")

    # --- Bottom: winner strip -----------------------------------------
    cmap = ListedColormap(["#E53935", "#1E88E5"])  # 0→red(P2), 1→blue(P1)
    arr = np.array(winner_bits).reshape(1, -1)
    ax2.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1,
               extent=[ks[0] - 0.5, ks[-1] + 0.5, 0, 1],
               interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xlabel("k  (fork graph F_k)")
    ax2.set_title("Winner strip (blue = P1, red = P2)")

    fig.tight_layout()
    png_path = RESULTS_DIR / "fork_grundy_extended.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ======================================================================
# Driver
# ======================================================================

MAX_K = 500


def main() -> None:
    max_t = MAX_K - 2
    max_m = max(1000, max_t)

    print("=" * 70)
    print(f"Fast recurrence Grundy computation for fork graphs F_3 … F_{MAX_K}")
    print("=" * 70)

    # --- Step 1: path Grundy values + sanity check --------------------
    P = compute_path_grundy(max_m)
    path_zeros = [m for m in range(35) if P[m] == 0]
    print(f"\nPath Grundy zeros (m ≤ 34): {path_zeros}")
    expected = [0, 4, 8, 14, 20, 24, 28, 34]
    assert all(z in path_zeros for z in expected), (
        f"Path recurrence wrong: expected zeros {expected}, got {path_zeros}"
    )
    print("Path recurrence sanity check: OK")

    # --- Step 2: fork Grundy values -----------------------------------
    Fork = compute_fork_grundy(max_t, P)

    # --- Step 3: CRITICAL validation against brute force --------------
    print()
    if not validate_against_csv(P, Fork):
        print("\nABORTING: recurrence disagrees with brute force. "
              "The decomposition is wrong — debug before trusting results.")
        sys.exit(1)

    # --- Build rows + winning moves (Step 6) --------------------------
    rows: List[Dict] = []
    ks: List[int] = []
    grundy: List[int] = []
    for k in range(3, MAX_K + 1):
        g = fork_grundy_value(k, P, Fork)
        winners = winning_first_moves(k, P, Fork) if g > 0 else []
        rows.append({
            "k": k,
            "num_vertices": k + 1,
            "num_edges": k,
            "grundy_value": g,
            "winner": "P1" if g > 0 else "P2",
            "winning_first_moves": ";".join(str(v) for v in winners),
        })
        ks.append(k)
        grundy.append(g)

    # --- Step 5: save outputs -----------------------------------------
    print()
    _save_csv(rows)
    _chart_extended(ks, grundy)

    # --- Step 4: report -----------------------------------------------
    p2_wins = [k for k, g in zip(ks, grundy) if g == 0]
    print("\n" + "-" * 70)
    print("FINDINGS")
    print("-" * 70)
    print(f"Player 2 (G = 0) wins on F_k for k = "
          f"{p2_wins if p2_wins else '(none)'}")
    after_20 = [k for k in p2_wins if k > 20]
    if after_20:
        print(f"P2 wins DO recur after k = 20, at: {after_20}")
    else:
        print(f"P2 NEVER wins again after k = 20 "
              f"(last P2 win: F_{max(p2_wins)}). "
              f"From F_21 to F_{MAX_K}, player 1 always wins.")

    grundy_period = find_eventual_period(grundy)
    winner_period = find_eventual_period([1 if g > 0 else 0 for g in grundy])
    print()
    if grundy_period:
        p, start = grundy_period
        print(f"Grundy sequence becomes periodic: period {p}, "
              f"from about k = {ks[start]}.")
    else:
        print("No eventual period detected in the Grundy sequence "
              f"(checked up to period 80).")
    if winner_period:
        p, start = winner_period
        print(f"Winner (N/P) pattern becomes periodic: period {p}, "
              f"from about k = {ks[start]}.")
    else:
        print("No eventual period detected in the winner pattern.")

    # A couple of concrete winning-move examples for the supervisor.
    print("\nExample winning first moves (vertex labels in fork_graph(k)):")
    for k in [7, 9, 11, 16, 100, 250]:
        if k <= MAX_K:
            g = fork_grundy_value(k, P, Fork)
            if g > 0:
                wm = winning_first_moves(k, P, Fork)
                print(f"  F_{k}: g={g}, winning move(s) = {wm}")


if __name__ == "__main__":
    main()
