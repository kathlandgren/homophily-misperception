"""
pa_simulation.py — preferential attachment finite-N simulation core (08_finite_n_pa).

Runs num_sim Barabasi-Albert (preferential attachment) networks at a single
(h_ss, h_oo) grid point and computes mean/variance of beta_s, beta_o,
beta_overall for all rescaling variants simultaneously.

Parameter mapping to the BA generator:
    h_ab = 1 - hoo   (cross-group prob from opponents/minority)
    h_ba = 1 - hss   (cross-group prob from supporters/majority)
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
rescale_bayes      = _perception.rescale_bayes


# ── Variant key helpers ────────────────────────────────────────────────────────

def _peragent_key(delta, gamma):
    return f'peragent_d{delta:.2f}_g{gamma:.2f}'

def _groupmean_key(delta, gamma):
    return f'groupmean_d{delta:.2f}_g{gamma:.2f}'

def all_variant_keys(delta_vals, gamma_vals):
    keys = ['none']
    for d in delta_vals:
        for g in gamma_vals:
            keys.append(_peragent_key(d, g))
            keys.append(_groupmean_key(d, g))
    return keys


# ── Core simulation ────────────────────────────────────────────────────────────

def run_one_gridpoint(hss, hoo, m, f_s, N, num_sim,
                      delta_vals, gamma_vals, seed_base=0, narcissistic=True):
    """
    Run num_sim BA simulations at (h_ss=hss, h_oo=hoo).

    hss = h_ss = supporter in-group attachment probability
    hoo = h_oo = opponent in-group attachment probability
    narcissistic = True: agents include their own opinion (Eq. 1); False: neighbors only

    Mapping to generator:
        h_ab = 1 - hoo  (cross-group prob from minority/opponents)
        h_ba = 1 - hss  (cross-group prob from majority/supporters)

    Returns
    -------
    dict keyed by variant_key -> dict with keys:
        beta_s, beta_o, beta_overall,
        var_beta_s, var_beta_o, var_beta_overall
    """
    f_o = 1.0 - f_s
    variants = all_variant_keys(delta_vals, gamma_vals)

    # Accumulators: sum and sum-of-squares for online mean/variance
    S1 = {v: np.zeros(3) for v in variants}   # [beta_s, beta_o, beta_overall]
    S2 = {v: np.zeros(3) for v in variants}

    for sim_idx in range(num_sim):
        seed = seed_base * 10_000 + sim_idx
        G, minority_nodes = homophilic_barabasi_albert_graph_assym(
            N=N, m=m, minority_fraction=f_o,
            h_ab=1 - hoo, h_ba=1 - hss,
            seed=seed,
        )

        # Build numpy adjacency via vectorized edge indexing
        edge_arr = np.array(list(G.edges()), dtype=np.int32)
        adj = np.zeros((N, N), dtype=bool)
        if len(edge_arr) > 0:
            adj[edge_arr[:, 0], edge_arr[:, 1]] = True
            adj[edge_arr[:, 1], edge_arr[:, 0]] = True

        # Node membership: minority=opponents, majority=supporters
        is_opp = np.zeros(N, dtype=bool)
        is_opp[minority_nodes] = True
        is_supp = ~is_opp

        # Raw per-node perceived supporter fraction
        # Isolated nodes (degree=0) use own opinion, matching opinion_functions.py line 117
        # (PA guarantees degree >= m, so this branch is never reached in practice)
        degree = adj.sum(axis=1).astype(float)
        supp_nbrs = (adj[:, is_supp]).sum(axis=1).astype(float)
        raw_perc = perceived_fraction(supp_nbrs, degree, is_supp, narcissistic)

        # Compute beta values for each variant
        for v in variants:
            if v == 'none':
                perc = raw_perc
                b_s  = f_s - perc[is_supp].mean()
                b_o  = f_s - perc[is_opp].mean()
            elif v.startswith('peragent'):
                _, d_str, g_str = v.split('_')
                d = float(d_str[1:]); g = float(g_str[1:])
                perc = rescale_bayes(raw_perc, d, g)
                b_s  = f_s - perc[is_supp].mean()
                b_o  = f_s - perc[is_opp].mean()
            else:  # groupmean
                _, d_str, g_str = v.split('_')
                d = float(d_str[1:]); g = float(g_str[1:])
                mu_s = raw_perc[is_supp].mean()
                mu_o = raw_perc[is_opp].mean()
                b_s  = f_s - rescale_bayes(mu_s, d, g)
                b_o  = f_s - rescale_bayes(mu_o, d, g)

            b_all = f_s * b_s + f_o * b_o
            vals = np.array([b_s, b_o, b_all])
            S1[v] += vals
            S2[v] += vals ** 2

    # Convert to mean / variance
    result = {}
    for v in variants:
        mean = S1[v] / num_sim
        var  = S2[v] / num_sim - mean ** 2
        result[v] = {
            'beta_s':           float(mean[0]),
            'beta_o':           float(mean[1]),
            'beta_overall':     float(mean[2]),
            'var_beta_s':       float(var[0]),
            'var_beta_o':       float(var[1]),
            'var_beta_overall': float(var[2]),
        }
    return result
