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
├── gui.py              Interactive Matplotlib GUI (human/AI play)
└── __main__.py         Entry point for `python -m stable_set_game`

experiments/            Dissertation experiments
├── run_experiments.py  Experiments 1–4 (data + charts to results/)
├── fork_grundy.py      Brute-force Grundy analysis of fork graphs F_n (≤ F_30)
├── fork_grundy_fast.py Recurrence-based Grundy values for F_n (to F_500+)
└── results/            Generated CSV data and PNG charts

tests/                  Unit tests (pytest) covering every module
requirements.txt        Python dependencies
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `networkx`, `matplotlib`, `numpy`, `pytest`.

## Usage

### Play the game (GUI)

```bash
python -m stable_set_game
```

A menu lets you choose the graph type, size, game mode (Human vs Human,
Human vs AI, AI vs AI) and AI search depth. In the window: **click** a green
vertex to play, **R** to restart, **U** to undo, **Q** to quit (and **Space**
to step in AI-vs-AI mode).

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
| 2 | Which evaluation-function weights play best? | Round-robin tournament of 5 weight configurations on random graphs. |
| 3 | Does the AI agree with theory? | Exact minimax winner vs Sprague–Grundy prediction on paths `P_n` and cycles `C_n`. |
| 4 | How close does optimal play get to the maximum independent set? | Compare the played stable-set size to the exact MIS. |
| Fork | What are the Grundy values of the fork graphs `F_n`, and are they periodic? | Brute force (`fork_grundy.py`, ≤ F_30) and a fast Sprague–Grundy recurrence (`fork_grundy_fast.py`, to F_500+). The recurrence is validated to match brute force exactly on F_3–F_30, then shows the sequence is **eventually periodic with period 34** (so P2 wins recur indefinitely, e.g. F_34, F_39, …). |

Experiment 3 is the **correctness anchor**: agreement between the AI's exact
solver and the independently computed Grundy values validates the engine and
search.

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
