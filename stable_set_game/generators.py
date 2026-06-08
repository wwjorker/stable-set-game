"""
Graph generators for the Stable Set Game.

Provides factory functions that produce NetworkX graphs commonly used in
combinatorial game theory research:

- Path graph  P_n
- Cycle graph C_n
- Erdős–Rényi random graph  G(n, p)
- Random regular graph  G(n, d)
"""

from __future__ import annotations

from typing import Optional

import networkx as nx


def path_graph(n: int) -> nx.Graph:
    """Return the path graph P_n with *n* vertices labelled 0 … n-1.

    A path graph is a tree with exactly two leaves (degree-1 vertices)
    and n-1 edges forming a single chain.

    Args:
        n: Number of vertices (must be >= 2).

    Raises:
        ValueError: If n < 2.
    """
    if n < 2:
        raise ValueError(f"Path graph requires n >= 2, got {n}.")
    return nx.path_graph(n)


def cycle_graph(n: int) -> nx.Graph:
    """Return the cycle graph C_n with *n* vertices labelled 0 … n-1.

    A cycle graph is a 2-regular graph (every vertex has degree 2)
    with exactly n edges.

    Args:
        n: Number of vertices (must be >= 3).

    Raises:
        ValueError: If n < 3.
    """
    if n < 3:
        raise ValueError(f"Cycle graph requires n >= 3, got {n}.")
    return nx.cycle_graph(n)


def erdos_renyi_graph(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """Return an Erdős–Rényi random graph G(n, p).

    Each possible edge is included independently with probability *p*.

    Args:
        n: Number of vertices (must be >= 1).
        p: Edge probability, must be in [0, 1].
        seed: Optional random seed for reproducibility.

    Raises:
        ValueError: If n < 1 or p is outside [0, 1].
    """
    if n < 1:
        raise ValueError(f"Erdős–Rényi graph requires n >= 1, got {n}.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Edge probability p must be in [0, 1], got {p}.")
    return nx.erdos_renyi_graph(n, p, seed=seed)


def random_regular_graph(n: int, d: int, seed: Optional[int] = None) -> nx.Graph:
    """Return a random d-regular graph on *n* vertices.

    Every vertex has exactly degree *d*.  Note that n*d must be even
    (a basic requirement of graph theory).

    Args:
        n: Number of vertices (must be >= 2).
        d: Degree of each vertex (must satisfy 0 <= d < n and n*d even).
        seed: Optional random seed for reproducibility.

    Raises:
        ValueError: If n < 2, d < 0, d >= n, or n*d is odd.
    """
    if n < 2:
        raise ValueError(f"Random regular graph requires n >= 2, got {n}.")
    if d < 0 or d >= n:
        raise ValueError(f"Degree d must satisfy 0 <= d < n, got d={d}, n={n}.")
    if (n * d) % 2 != 0:
        raise ValueError(
            f"n * d must be even for a regular graph, got n={n}, d={d} "
            f"(product={n * d})."
        )
    return nx.random_regular_graph(d, n, seed=seed)


def complete_graph(n: int) -> nx.Graph:
    """Return the complete graph K_n.

    Useful as a trivial test case: in K_n the maximum independent set
    has size 1, so the game always ends after a single move.

    Args:
        n: Number of vertices (must be >= 1).

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"Complete graph requires n >= 1, got {n}.")
    return nx.complete_graph(n)


def fork_graph(n: int) -> nx.Graph:
    """Return the fork graph F_n.

    F_n is the path P_n (vertices 0 … n-1) with one extra vertex (labelled
    n) attached to the path vertex n-2.  The result has n+1 vertices and
    n edges.  Vertex n-2 becomes a degree-3 "fork point":

        0 — 1 — 2 — … — (n-3) — (n-2) — (n-1)
                                   |
                                   n

    Example: F_5 is the path 0-1-2-3-4 with vertex 5 attached to vertex 3.

    Args:
        n: Length of the underlying path (must be >= 3 so that the fork
           point n-2 is an internal vertex, not an endpoint).

    Raises:
        ValueError: If n < 3.
    """
    if n < 3:
        raise ValueError(f"Fork graph requires n >= 3, got {n}.")
    g = nx.path_graph(n)
    g.add_edge(n - 2, n)
    return g


def star_graph(n: int) -> nx.Graph:
    """Return the star graph S_n with *n* leaves and 1 centre vertex.

    The resulting graph has n+1 vertices: the centre is vertex 0,
    and the leaves are vertices 1 … n.

    Args:
        n: Number of leaves (must be >= 1).

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"Star graph requires n >= 1, got {n}.")
    return nx.star_graph(n)
