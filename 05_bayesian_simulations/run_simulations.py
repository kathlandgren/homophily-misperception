import itertools
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
sim_fun  = _load("simulation_fun", "05_bayesian_simulations/simulation_functions.py")

# Fixed settings
N        = 1000
fs       = 2 / 3
n_points = 101
seedval  = 11
bayes    = True

# Parameter grid
m_vals      = [2, 5]
delta_vals  = [0.5, 1, 2]
gamma_vals  = [0.25, 0.5, 0.75, 1]
numsim_vals = [100, 1000]

output_dir = here() / "05_bayesian_simulations" / "output"
output_dir.mkdir(exist_ok=True)

for m, delta, gamma, numsim in itertools.product(m_vals, delta_vals, gamma_vals, numsim_vals):
    print(f"\nRunning: m={m}, delta={delta}, gamma={gamma}, numsim={numsim}")

    results = sim_fun.run_grid_simulation(
        N=N, m=m, fs=fs, n_points=n_points,
        seedval=seedval, bayes=bayes,
        delta=delta, gamma=gamma, numsim=numsim,
    )

    meta = {
        "N": N, "m": m, "fs": fs, "n_points": n_points,
        "seedval": seedval, "numsim": numsim, "bayes": bayes,
        "delta": delta, "gamma": gamma,
        "var_mis_overall_mean": float(np.mean(results["var_mis_overall"])),
        "var_mis_overall_max":  float(np.max(results["var_mis_overall"])),
        "format": "npz_compressed",
    }
    filename = (
        f"misperception_grid"
        f"_N{N}_m{m}_fs{fs:.4f}_n{n_points}"
        f"_seed{seedval}_numsim{numsim}"
        f"_bayes{int(bayes)}_delta{delta}_gamma{gamma}.npz"
    )
    out_file = output_dir / filename
    sim_fun.save_npz(str(out_file), results, meta)
    print(f"  Saved: {out_file.name}")
