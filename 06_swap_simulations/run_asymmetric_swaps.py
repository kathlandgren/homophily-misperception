"""
run_asymmetric_swaps.py — production parallel script for asymmetric homophily swap simulations.

Runs all 7 (h_ss, h_oo) conditions × 2 m values × 3 sim counts = 42 combinations.
Sim counts [10, 100, 1000] let you pull intermediate results early without waiting for the full run.

Usage:
    python run_asymmetric_swaps.py                      # defaults
    python run_asymmetric_swaps.py --outdir /path/out --workers 50
"""

import argparse
import itertools
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

from pyprojroot import here

_spec = importlib.util.spec_from_file_location("opinion_functions", here() / "src" / "opinion_functions.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fun = _mod


# ── Parameter grid ────────────────────────────────────────────────────────────
# Each tuple is (h_ss, h_oo): within-support and within-oppose homophily
CONDITIONS = [
    (0.25, 0.25),  # sym_25
    (0.50, 0.50),  # sym_50
    (0.75, 0.75),  # sym_75  (matches existing symmetric h=0.75 baseline)
    (0.25, 0.75),  # asym: low support, high oppose
    (0.75, 0.25),  # asym: high support, low oppose
    (0.50, 0.75),  # asym: moderate support, high oppose
    (0.75, 0.50),  # asym: high support, moderate oppose
]

M_VEC             = [2, 5]
NUM_AGENTS_VEC    = [100, 1000]   # 10_000 dropped — symmetric runs showed those never completed
NUM_SIM_VEC       = [10, 100, 1000]  # quick check → preliminary figures → final
MINORITY_FRACTION = 1 / 3

_DEFAULT_OUTDIR = str(here() / "06_swap_simulations" / "asymmetric_output")


def main(outdir: str, workers: int, sim_count: int | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)

    param_grid = [
        (h_ss, h_oo, m, num_agents, num_sim,
         int(MINORITY_FRACTION * num_agents), MINORITY_FRACTION, 100, outdir)
        for (h_ss, h_oo), m, num_agents, num_sim in itertools.product(CONDITIONS, M_VEC, NUM_AGENTS_VEC, NUM_SIM_VEC)
    ]

    if sim_count is not None:
        param_grid = [p for p in param_grid if p[4] == sim_count]  # index 4 = num_sim

    print(f"[INFO] Submitting {len(param_grid)} jobs with {workers} workers → {outdir}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(fun.run_simulation_wrapper_with_swaps_asymmetric, params)
            for params in param_grid
        ]

    for future in futures:
        future.result()

    print(f"[DONE] All jobs complete. Results in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetric homophily swap simulations")
    parser.add_argument("--outdir",  default=_DEFAULT_OUTDIR,
                        help="Output directory for pkl files")
    parser.add_argument("--workers", type=int, default=50,
                        help="Number of parallel workers")
    parser.add_argument("--sim-count", type=int, default=None,
                        help="Only run tasks with this num_sim value (10, 100, or 1000). "
                             "Omit to run all.")
    args = parser.parse_args()
    main(args.outdir, args.workers, args.sim_count)
