"""
generate_homophilic_SB_graph_vectorized.py — vectorized stochastic block model graph generator.

Replaces the O(N²) Python loop of generate_homophilic_SB_graph.py with a numpy-based
implementation (~50–100× faster for N=1000). Used by sbm_simulation.py and matched_simulation.py.

The API is identical to generate_homophilic_SB_graph.py, but the random draws are not:
this version uses numpy's default_rng, so the same seed gives a different graph.
generate_homophilic_SB_graph.py is kept unchanged because the Fig 2 cartoon depends on it.
"""

import numpy as np
import networkx as nx


def homophilic_stochastic_block_graph_assym(N, minority_fraction, h_ina, h_inb, h_out, seed=None):
    """Return homophilic random graph using a 2-block stochastic block model.

    Parameters
    ----------
    N : int
        Number of nodes.
    minority_fraction : float
        Fraction of nodes belonging to the minority group.
    h_ina : float
        Within-group edge probability for minority (a–a), in [0, 1].
    h_inb : float
        Within-group edge probability for majority (b–b), in [0, 1].
    h_out : float
        Between-group edge probability for a–b and b–a, in [0, 1].
    seed : int or None
        Seed for NumPy RNG.

    Returns
    -------
    G : nx.Graph
    minority_nodes : list
    """
    rng = np.random.default_rng(seed)

    minority_count = int(round(minority_fraction * N))
    # minority[i] == True  →  node i is in the minority group
    minority = np.zeros(N, dtype=bool)
    minority[:minority_count] = True
    rng.shuffle(minority)

    # Build probability matrix for the upper triangle
    # Start with all cross-group probability
    prob = np.full((N, N), h_out, dtype=np.float64)
    min_idx = np.where(minority)[0]
    maj_idx = np.where(~minority)[0]
    prob[np.ix_(min_idx, min_idx)] = h_ina
    prob[np.ix_(maj_idx, maj_idx)] = h_inb

    # Draw and threshold — upper triangle only (undirected, no self-loops)
    draws = rng.random((N, N))
    upper = np.triu(draws < prob, k=1)
    rows, cols = np.where(upper)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for n in range(N):
        G.nodes[n]['color'] = '#f2d138' if minority[n] else '#19bdff'
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))

    return G, list(min_idx)
