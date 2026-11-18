"""
swap_simulation.py — media-bias swap simulation core (06_swap_simulations).

For each (h_ss, h_oo) condition, generates PA networks and applies incremental
opinion swaps (promote highest-degree supporter → opponent, demote lowest-degree
opponent → supporter). Records β_all and β_peers at target swap counts.

Swap convention (matches opinion_functions.py:244):
  - promote: highest-degree current supporter → becomes opponent
  - demote:  lowest-degree current opponent   → becomes supporter
  - Network topology never changes; only group-membership labels change

Hard cap: `swap_counts` must already be trimmed to ≤ N_minority = round(f_o * N).
PA guarantee: degree ≥ m ≥ 2 for all nodes, so supp_nbrs / degree is always defined.
"""

import importlib.util
import pathlib
import sys

import numpy as np

# Load sibling modules from src/ by file path (same pattern as opinion_functions.py);
# registered in sys.modules so ProcessPoolExecutor workers can unpickle their functions.
_src = pathlib.Path(__file__).parent

def _load_src(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _src / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_perception = _load_src("perception")
perceived_fraction = _perception.perceived_fraction
homophilic_barabasi_albert_graph_assym  = _load_src("generate_homophilic_graph_asymmetric").homophilic_barabasi_albert_graph_assym


def _record_betas(meas_idx, is_supp, is_opp, degree, supp_nbrs,
                  promoted_mask, demoted_mask, f_s, f_o,
                  S1_all, S2_all, S1_peers, S2_peers, narcissistic=True):
    """Accumulate β_all and β_peers at one measurement point."""
    raw_perc = perceived_fraction(supp_nbrs, degree, is_supp, narcissistic)

    # β for all nodes
    b_s   = f_s - raw_perc[is_supp].mean()
    b_o   = f_s - raw_perc[is_opp].mean()
    b_all = f_s * b_s + f_o * b_o
    v = np.array([b_s, b_o, b_all])
    S1_all[meas_idx] += v
    S2_all[meas_idx] += v * v

    # β for unswapped nodes only (original label preserved)
    # unswapped_supp = original supporters still labeled supporter
    # unswapped_opp  = original opponents  still labeled opponent
    unswapped = ~(promoted_mask | demoted_mask)
    us = unswapped & is_supp
    uo = unswapped & is_opp

    if us.any() and uo.any():
        b_s_p   = f_s - raw_perc[us].mean()
        b_o_p   = f_s - raw_perc[uo].mean()
        b_all_p = f_s * b_s_p + f_o * b_o_p
    else:
        b_s_p = b_s; b_o_p = b_o; b_all_p = b_all

    vp = np.array([b_s_p, b_o_p, b_all_p])
    S1_peers[meas_idx] += vp
    S2_peers[meas_idx] += vp * vp


def run_one_condition(hss, hoo, m, f_s, N, num_sim, swap_counts, seed_base=0, narcissistic=True):
    """
    Run num_sim PA simulations at (h_ss=hss, h_oo=hoo).

    Parameters
    ----------
    swap_counts : sorted list[int]
        Integer swap counts where β is measured.  Must already be capped at
        N_minority = round((1-f_s)*N).
    narcissistic : bool
        True: agents include their own opinion (Eq. 1); False: neighbors only.

    Returns
    -------
    dict with keys 'all' and 'peers', each containing:
        beta_s, beta_o, beta_overall, var_beta_s, var_beta_o, var_beta_overall
        — each a list of length len(swap_counts)
    """
    f_o    = 1.0 - f_s
    n_meas = len(swap_counts)

    S1_all   = np.zeros((n_meas, 3))
    S2_all   = np.zeros((n_meas, 3))
    S1_peers = np.zeros((n_meas, 3))
    S2_peers = np.zeros((n_meas, 3))

    max_swap = swap_counts[-1] if swap_counts else 0

    for sim_idx in range(num_sim):
        seed = seed_base * 10_000 + sim_idx
        G, minority_nodes = homophilic_barabasi_albert_graph_assym(
            N=N, m=m, minority_fraction=f_o,
            h_ab=1.0 - hoo, h_ba=1.0 - hss,
            seed=seed,
        )

        # Build adjacency matrix once (topology never changes)
        edge_arr = np.array(list(G.edges()), dtype=np.int32)
        adj = np.zeros((N, N), dtype=bool)
        if len(edge_arr) > 0:
            adj[edge_arr[:, 0], edge_arr[:, 1]] = True
            adj[edge_arr[:, 1], edge_arr[:, 0]] = True

        # Initial group membership
        is_opp  = np.zeros(N, dtype=bool)
        is_opp[minority_nodes] = True
        is_supp = ~is_opp

        # Fixed degree; supporter-neighbor count updated incrementally
        degree    = adj.sum(axis=1).astype(float)
        supp_nbrs = adj[:, is_supp].sum(axis=1).astype(float)

        # Track relabeled nodes for β_peers
        promoted_mask = np.zeros(N, dtype=bool)   # orig supp → now opp
        demoted_mask  = np.zeros(N, dtype=bool)   # orig opp  → now supp

        meas_ptr     = 0   # next index into swap_counts to record
        current_swap = 0   # swaps applied so far

        # Record at swap_count == 0 before any swap
        if meas_ptr < n_meas and swap_counts[meas_ptr] == 0:
            _record_betas(meas_ptr, is_supp, is_opp, degree, supp_nbrs,
                          promoted_mask, demoted_mask, f_s, f_o,
                          S1_all, S2_all, S1_peers, S2_peers, narcissistic)
            meas_ptr += 1

        while meas_ptr < n_meas and current_swap < max_swap:
            current_swap += 1

            # Select nodes to swap
            supp_arr = np.where(is_supp)[0]
            promote  = supp_arr[np.argmax(degree[supp_arr])]

            opp_arr = np.where(is_opp)[0]
            demote  = opp_arr[np.argmin(degree[opp_arr])]

            # Update labels
            is_supp[promote] = False; is_opp[promote] = True
            is_supp[demote]  = True;  is_opp[demote]  = False
            promoted_mask[promote] = True
            demoted_mask[demote]   = True

            # Incremental supp_nbrs update: O(N) instead of O(N²)
            supp_nbrs -= adj[:, promote].astype(float)
            supp_nbrs += adj[:, demote].astype(float)

            # Record if this is a target measurement
            if meas_ptr < n_meas and current_swap == swap_counts[meas_ptr]:
                _record_betas(meas_ptr, is_supp, is_opp, degree, supp_nbrs,
                              promoted_mask, demoted_mask, f_s, f_o,
                              S1_all, S2_all, S1_peers, S2_peers, narcissistic)
                meas_ptr += 1

    def _to_dict(S1, S2):
        mean = S1 / num_sim
        var  = S2 / num_sim - mean ** 2
        return {
            'beta_s':           mean[:, 0],
            'beta_o':           mean[:, 1],
            'beta_overall':     mean[:, 2],
            'var_beta_s':       var[:, 0],
            'var_beta_o':       var[:, 1],
            'var_beta_overall': var[:, 2],
        }

    return {
        'all':   _to_dict(S1_all,   S2_all),
        'peers': _to_dict(S1_peers, S2_peers),
    }
