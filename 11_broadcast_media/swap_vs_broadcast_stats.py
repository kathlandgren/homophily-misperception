"""
swap_vs_broadcast_stats.py — statistics on the promoted (swapped-in) opponent nodes, for the
swap-vs-broadcast comparison paragraph after the broadcast figure (fig:broadcast_reach).

Networks and swaps follow the swap simulations (06_swap_simulations) exactly (src/swap_simulation.py: same generator, seeds
0..n-1, promote the highest-degree current supporter, demote the lowest-degree current opponent).

For each network, after `swaps` swaps:
  edge_share       fraction of all edges with at least one promoted endpoint
  endpoint_share   sum of promoted degrees / sum of all degrees
  degree_ratio     mean promoted degree / mean network degree
  to_orig_supp     fraction of promoted nodes' connections that go to originally-supporter nodes
  supp_reached     fraction of current supporters with at least one promoted neighbour
  share_among      mean, over those supporters, of (promoted neighbours / degree)
  unsw_*           the same two quantities for original supporters that were never swapped
Values are averaged over networks.

Runs locally in a few minutes; no precomputed data needed.

Usage:
    python 11_broadcast_media/swap_vs_broadcast_stats.py --m 10 --swaps 122 [--n 100]
"""

import argparse
import importlib.util
import sys

import numpy as np
from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

homophilic_barabasi_albert_graph_assym = _load(
    "generate_homophilic_graph_asymmetric", "src/generate_homophilic_graph_asymmetric.py").homophilic_barabasi_albert_graph_assym


def one(seed, m, swaps, h=0.75, f_s=2 / 3, N=1000):
    G, minority = homophilic_barabasi_albert_graph_assym(N=N, m=m, minority_fraction=1 - f_s,
                                                         h_ab=1 - h, h_ba=1 - h, seed=seed)
    e = np.array(list(G.edges()), dtype=np.int32)
    adj = np.zeros((N, N), dtype=bool)
    adj[e[:, 0], e[:, 1]] = adj[e[:, 1], e[:, 0]] = True
    deg = adj.sum(1).astype(float)
    is_opp = np.zeros(N, bool); is_opp[minority] = True
    orig_supp = ~is_opp.copy()
    promoted = np.zeros(N, bool)
    for _ in range(swaps):
        s = np.where(~is_opp)[0]; p = s[np.argmax(deg[s])]
        o = np.where(is_opp)[0];  d = o[np.argmin(deg[o])]
        is_opp[p] = True; is_opp[d] = False; promoted[p] = True
    is_supp = ~is_opp
    n_prom_nbrs = adj[:, promoted].sum(1)
    reached = is_supp & (n_prom_nbrs > 0)
    unsw = orig_supp & is_supp                      # original supporters still labelled supporter
    reached_u = unsw & (n_prom_nbrs > 0)
    return dict(
        edge_share=(promoted[e[:, 0]] | promoted[e[:, 1]]).mean(),
        endpoint_share=deg[promoted].sum() / deg.sum(),
        degree_ratio=deg[promoted].mean() / deg.mean(),
        to_orig_supp=adj[promoted][:, orig_supp].sum() / deg[promoted].sum(),
        supp_reached=reached.sum() / is_supp.sum(),
        share_among=(n_prom_nbrs[reached] / deg[reached]).mean(),
        unsw_supp_reached=reached_u.sum() / unsw.sum(),
        unsw_share_among=(n_prom_nbrs[reached_u] / deg[reached_u]).mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--swaps', type=int, nargs='+', default=[122])
    ap.add_argument('--n', type=int, default=100)
    a = ap.parse_args()
    for sw in a.swaps:
        rows = [one(seed, a.m, sw) for seed in range(a.n)]
        print(f'm={a.m} swaps={sw} ({sw / 10:.1f}%) n={a.n}: ' +
              ', '.join(f'{k}={np.mean([r[k] for r in rows]):.3f}' for k in rows[0]))


if __name__ == '__main__':
    main()
