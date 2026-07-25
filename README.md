# Stable Set Game (Node Kayles)

An MSc dissertation project that studies the combinatorial game **Stable Set
Game** — also known as **Node Kayles** — through a clean game engine, a
configurable minimax AI, an interactive GUI, and a suite of computational
experiments linking empirical play to Sprague–Grundy theory.

## The game

Two players alternate turns on an undirected graph. On each turn the current
player selects a vertex that is **not adjacent to any previously selected
vertex**, so the set of chosen vertices always forms an *independent (stable)
set*. The player who cannot make a legal move **loses** (normal play
convention). The game is *impartial*, so it can be analysed with
Sprague–Grundy theory.

## Project structure

```
stable_set_game/        Core library
├── engine.py           Game rules and state (StableSetGame)
├── evaluation.py       Parameterised heuristic evaluation (WeightedEvaluator)
├── ai.py               Minimax + alpha-beta AI player (AIPlayer)
├── generators.py       Graph factories (path, cycle, random, fork, star, ...)
├── self_play.py        Match / series / tournament framework
├── desktop_gui.py      Default PySide6 desktop application
├── gui.py              Legacy Matplotlib GUI fallback
└── __main__.py         Entry point for `python -m stable_set_game`

experiments/            Dissertation experiments
├── run_experiments.py  Experiments 1–4 (data + charts to results/)
├── fork_grundy.py      Brute-force Grundy analysis of fork graphs F_n (≤ F_30)
├── fork_grundy_fast.py Recurrence-based Grundy values for F_n (tested to F_2000)
└── results/            Generated CSV data and PNG charts

tests/                  Unit tests (pytest), including both GUI front ends
requirements.txt        Python dependencies
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `networkx`, `matplotlib`, `numpy`, `pytest`, `PySide6`.

## Usage

### Play the game (GUI)

```bash
python -m stable_set_game
```

The PySide6 desktop application provides a graphical setup page, live graph
preview, three game modes (Human vs Human, Human vs AI, AI vs AI), a game
status panel, move history, non-blocking AI search, undo/restart controls and
step-by-step AI-vs-AI play. Click a green vertex to make a legal move.
Erdős–Rényi graphs expose a configurable edge probability `p` in `[0, 1]`;
the fixed demonstration seed keeps a selected `(n, p)` reproducible.

The earlier lightweight Matplotlib interface is retained as a fallback:

```bash
python -m stable_set_game.gui
```

### Run the experiments

```bash
python experiments/run_experiments.py     # Experiments 1–4 → experiments/results/
python experiments/fork_grundy.py         # Exact Grundy values for fork graphs
```

### Run the tests

```bash
pytest
```

## The experiments

| # | Question | Method |
|---|----------|--------|
| 1 | How does graph density affect first-player win rate and game length? | AI-vs-AI on Erdős–Rényi `G(n, p)` across a sweep of `n` and `p`. |
| 2 | Which evaluation-function weights play best? | Round-robin comparison of 7 hand-set weight configurations on random graphs. |
| 3 | Does the AI agree with theory? | Exact minimax winner vs Sprague–Grundy prediction on paths `P_n` and cycles `C_n`. |
| 4 | How close does depth-3 play get to the maximum independent set? | Compare the played stable-set size to the exact MIS. |
| Fork | What are the Grundy values of the fork graphs `F_n`, and are they periodic? | Brute force (`fork_grundy.py`, ≤ F_30) and a fast Sprague–Grundy recurrence (`fork_grundy_fast.py`, tested to F_2000). The recurrence matches brute force on F_3–F_30 and computationally reproduces a period-34 pattern from F_313 through F_2000. A subsequent literature comparison identified [Songsuwan's Theorem 3.2](https://doi.org/10.48550/arXiv.2512.24221) for an isomorphic graph family, establishing indefinite period-34 behaviour after the index shift `k = s + 2`. |

Experiment 3 is a **small-instance correctness check**: agreement between the
AI's exact solver and the independently computed Grundy values validates the
engine and full-depth search on P_3–P_12 and C_3–C_12.

## Using the library directly

```python
import networkx as nx
from stable_set_game.engine import StableSetGame
from stable_set_game.ai import AIPlayer

game = StableSetGame(nx.path_graph(7))
ai = AIPlayer(max_depth=10)
while not game.is_game_over():
    result = ai.choose_move(game)
    game.make_move(result.best_move)
print("Winner: Player", game.get_winner())
```

## Known limitations

- The AI's negamax search has **no transposition table**, so exact solving
  (`max_depth=inf`) is only practical up to roughly `n = 12`.
- Experiment results depend on search depth and the evaluation weights; the
  experiments use a shallow depth (3) for tractability, so rankings reflect
  heuristic play rather than perfect play.

## License

Academic project — not licensed for redistribution.
