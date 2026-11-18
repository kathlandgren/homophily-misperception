"""
matched_simulation.py — degree-matched simulation core (10_degree_matched): preferential attachment and stochastic block model at degree-matched parameters.

For each (h_ss, h_oo) grid point, runs num_sim PA and num_sim SBM simulations
with the SBM parameterized to match the PA mean degree (2m).

Degree-matching formula:
    r_s = h_ss / (1 - h_ss)      (in-group / out-group ratio, supporters)
    r_o = h_oo / (1 - h_oo)      (in-group / out-group ratio, opponents)
    D   = f_s² r_s + 2 f_s f_o + f_o² r_o
    H_out   = 2m / (N · D)       (SBM between-group edge probability)
    H_in^s  = r_s · H_out        (SBM in-group prob, supporters)
    H_in^o  = r_o · H_out        (SBM in-group prob, opponents)

Isolated-node convention (narcissistic=False only, matching opinion_functions.py):
    A node with degree 0 perceives its own opinion: 1.0 if supporter, 0.0 if opponent.
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
homophilic_stochastic_block_graph_assym = _load_src("generate_homophilic_SB_graph_vectorized").homophilic_stochastic_block_graph_assym


def compute_sbm_params(hss, hoo, m, f_s, N):
    """Return (H_in_s, H_in_o, H_out) for the degree-matched SBM.

    At h=1 the ratio r=h/(1-h) diverges, but H_in = r*H_out has a finite
    limit: 2m/(N*f_s²) for supporters, 2m/(N*f_o²) for opponents.  We use
    a capped ratio (h→0.9999) to stay numerically stable while matching that
    limit to within floating-point precision.
    """
    hss_c = min(hss, 0.9999)
    hoo_c = min(hoo, 0.9999)
    r_s = hss_c / (1.0 - hss_c)
    r_o = hoo_c / (1.0 - hoo_c)
    f_o = 1.0 - f_s
    D = f_s**2 * r_s + 2.0 * f_s * f_o + f_o**2 * r_o
    H_out = (2.0 * m) / (N * D)
    return r_s * H_out, r_o * H_out, H_out


def _beta_from_graph(G, N, is_supp, is_opp, f_s, f_o, narcissistic=True):
    """Compute (beta_s, beta_o, beta_overall) for one simulated graph."""
    edge_arr = np.array(list(G.edges()), dtype=np.int32)
    adj = np.zeros((N, N), dtype=bool)
    if len(edge_arr) > 0:
        adj[edge_arr[:, 0], edge_arr[:, 1]] = True
        adj[edge_arr[:, 1], edge_arr[:, 0]] = True

    degree = adj.sum(axis=1).astype(float)
    supp_nbrs = adj[:, is_supp].sum(axis=1).astype(float)
    # Isolated nodes use own opinion when narcissistic=False (see perception.py)
    raw_perc = perceived_fraction(supp_nbrs, degree, is_supp, narcissistic)

    b_s = f_s - raw_perc[is_supp].mean()
    b_o = f_s - raw_perc[is_opp].mean()
    return np.array([b_s, b_o, f_s * b_s + f_o * b_o])


def run_one_condition(hss, hoo, m, f_s, N, num_sim, seed_base=0, narcissistic=True):
    """
    Run num_sim PA and num_sim SBM simulations at degree-matched parameters.

    Parameters
    ----------
    hss : float   h_ss, supporter in-group attachment probability (PA grid)
    hoo : float   h_oo, opponent in-group attachment probability  (PA grid)
    m   : int     BA edges per new node
    f_s : float   true supporter fraction
    N   : int     network size
    num_sim : int simulations per model
    seed_base : int  base seed for reproducibility
    narcissistic : bool  True: agents include their own opinion (Eq. 1); False: neighbors only

    Returns
    -------
    dict with keys 'pa', 'sbm' (each a stats dict) and 'H_out' (float)
    """
    f_o = 1.0 - f_s
    H_in_s, H_in_o, H_out = compute_sbm_params(hss, hoo, m, f_s, N)

    S1_pa  = np.zeros(3); S2_pa  = np.zeros(3)
    S1_sbm = np.zeros(3); S2_sbm = np.zeros(3)

    for sim_idx in range(num_sim):
        seed = seed_base * 10_000 + sim_idx

        # ── PA ────────────────────────────────────────────────────────────────
        G_pa, min_pa = homophilic_barabasi_albert_graph_assym(
            N=N, m=m, minority_fraction=f_o,
            h_ab=1.0 - hoo, h_ba=1.0 - hss,
            seed=seed,
        )
        is_opp_pa = np.zeros(N, dtype=bool)
        is_opp_pa[min_pa] = True
        v_pa = _beta_from_graph(G_pa, N, ~is_opp_pa, is_opp_pa, f_s, f_o, narcissistic)
        S1_pa += v_pa; S2_pa += v_pa**2

        # ── SBM (offset seed to avoid correlation with PA) ────────────────────
        G_sbm, min_sbm = homophilic_stochastic_block_graph_assym(
            N=N, minority_fraction=f_o,
            h_ina=H_in_o, h_inb=H_in_s, h_out=H_out,
            seed=seed + 5_000_000,
        )
        is_opp_sbm = np.zeros(N, dtype=bool)
        is_opp_sbm[min_sbm] = True
        v_sbm = _beta_from_graph(G_sbm, N, ~is_opp_sbm, is_opp_sbm, f_s, f_o, narcissistic)
        S1_sbm += v_sbm; S2_sbm += v_sbm**2

    def _stats(S1, S2):
        mean = S1 / num_sim
        var  = S2 / num_sim - mean**2
        return {
            'beta_s':           float(mean[0]),
            'beta_o':           float(mean[1]),
            'beta_overall':     float(mean[2]),
            'var_beta_s':       float(var[0]),
            'var_beta_o':       float(var[1]),
            'var_beta_overall': float(var[2]),
        }

    return {
        'pa':   _stats(S1_pa,  S2_pa),
        'sbm':  _stats(S1_sbm, S2_sbm),
        'H_out': float(H_out),
    }
