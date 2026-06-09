import importlib.util
import numpy as np

from pyprojroot import here

def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fun      = _load("opinion_functions",                    "src/opinion_functions.py")
gen_sym  = _load("generate_homophilic_graph_symmetric",  "src/generate_homophilic_graph_symmetric.py")
gen_asym = _load("generate_homophilic_graph_asymmetric", "src/generate_homophilic_graph_asymmetric.py")

generate_homophilic_graph_asymmetric = gen_asym

def run_grid_simulation(
    N=1000,
    m=5,
    fs=2/3,                 # majority fraction (true)
    n_points=101,
    seedval=11,
    bayes=True,
    delta=2,
    gamma=0.5,
    numsim=1,               # <-- NEW: # Monte Carlo runs per grid point
):
    """
    Runs the simulation on a grid of (h_ss, h_oo) values and computes misperception.

    Misperception definition:
        mis_majority = f_s - mean(perceived_majority)
        mis_minority = f_s - mean(perceived_minority)
        mis_overall  = f_s - mean(perceived_overall)

    With numsim > 1, we return:
      - mean misperception across runs for each grid cell
      - variance across runs for each grid cell

    Returns dict with arrays + axes:
        hss_vals, hoo_vals,
        mis_majority, mis_minority, mis_overall,
        var_mis_majority, var_mis_minority, var_mis_overall
    """
    if numsim < 1:
        raise ValueError("numsim must be >= 1")

    # Axes in display space
    hss_vals = np.linspace(0, 1, n_points)  # x-axis: h_ss = 1 - h_so
    hoo_vals = np.linspace(0, 1, n_points)  # y-axis: h_oo = 1 - h_os

    # Mean misperception grids
    mis_majority = np.zeros((n_points, n_points), dtype=float)
    mis_minority = np.zeros((n_points, n_points), dtype=float)
    mis_overall  = np.zeros((n_points, n_points), dtype=float)

    # Variance grids (across runs at fixed (hss, hoo))
    var_mis_majority = np.zeros((n_points, n_points), dtype=float)
    var_mis_minority = np.zeros((n_points, n_points), dtype=float)
    var_mis_overall  = np.zeros((n_points, n_points), dtype=float)

    for i, hoo in enumerate(hoo_vals):
        hos = 1.0 - hoo
        for j, hss in enumerate(hss_vals):
            hso = 1.0 - hss
            
            # Collect misperception across numsim runs
            maj_runs = np.empty(numsim, dtype=float)
            min_runs = np.empty(numsim, dtype=float)
            all_runs = np.empty(numsim, dtype=float)

            for r in range(numsim):
                # IMPORTANT:
                # If your graph generator is deterministic given `seed`,
                # you must vary the seed per run to actually get different networks.
                run_seed = seedval + r

                G, minority_nodes = generate_homophilic_graph_asymmetric.homophilic_barabasi_albert_graph_assym(
                    N=N, m=m, minority_fraction=1 - fs,
                    h_ab=hos, h_ba=hso,
                    seed=run_seed
                )

                true_opinion, perceived_opinion = fun.generate_perceived_opinion(
                    G, minority_nodes, {},
                    narcissistic=True,
                    weigh_connected=False,
                    bayes=bayes,
                    delta=delta,
                    gamma=gamma,
                )

                minority_nodes = list(minority_nodes)
                minority_set = set(minority_nodes)

                minority_op = [perceived_opinion[idx] for idx in minority_nodes]
                majority_nodes = [u for u in range(len(perceived_opinion)) if u not in minority_set]
                majority_op = [perceived_opinion[idx] for idx in majority_nodes]

                mm = float(np.mean(majority_op))
                mn = float(np.mean(minority_op))
                mo = (1 - fs) * mm + fs * mn

                maj_runs[r] = fs - mm
                min_runs[r] = fs - mn
                all_runs[r] = fs - mo

            # Store mean and variance across runs for this grid cell
            mis_majority[i, j] = float(np.mean(maj_runs))
            mis_minority[i, j] = float(np.mean(min_runs))
            mis_overall[i, j]  = float(np.mean(all_runs))

            # ddof=1 gives sample variance when numsim>1; 0 when numsim==1
            ddof = 1 if numsim > 1 else 0
            var_mis_majority[i, j] = float(np.var(maj_runs, ddof=ddof))
            var_mis_minority[i, j] = float(np.var(min_runs, ddof=ddof))
            var_mis_overall[i, j]  = float(np.var(all_runs, ddof=ddof))

    return {
        "hss_vals": hss_vals,
        "hoo_vals": hoo_vals,
        "mis_majority": mis_majority,
        "mis_minority": mis_minority,
        "mis_overall": mis_overall,
        "var_mis_majority": var_mis_majority,
        "var_mis_minority": var_mis_minority,
        "var_mis_overall": var_mis_overall,
    }


def save_npz(out_path, arrays, meta):
    """
    Save arrays + metadata to a single compressed .npz file.
    Note: metadata values should be scalar-ish (numbers/strings/bools).
    """
    payload = {**arrays, **{f"meta_{k}": v for k, v in meta.items()}}
    np.savez_compressed(out_path, **payload)

