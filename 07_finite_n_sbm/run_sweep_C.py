"""
run_sweep_C.py — SBM finite-N simulation sweep.

Runs num_sim SBM networks at every point of an n_points × n_points
(H_in^s, H_in^o) grid for a single (H_out, f_s, N) combination,
parallelized across grid points via ProcessPoolExecutor.

Produces one .npz file per rescaling variant in outdir/.

Usage (cluster array job, see submit_sweep_C.sh):
    python 07_finite_n_sbm/run_sweep_C.py --hout 0.3 --fs 0.6667 --N 1000

Usage (local test):
    python 07_finite_n_sbm/run_sweep_C.py --hout 0.3 --fs 0.6667 --N 100 \
        --numsim 5 --n-points 5 --workers 2

Add --exclude-self to write the neighbors-only twin to 07_finite_n_sbm/output_exclude_self/
(needed by 12_include_self_comparison).
"""

import argparse
import importlib.util
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # registered so ProcessPoolExecutor workers can unpickle
    spec.loader.exec_module(mod)
    return mod

sim = _load("sbm_simulation", "src/sbm_simulation.py")
run_one_gridpoint, all_variant_keys = sim.run_one_gridpoint, sim.all_variant_keys

# ── Rescaling parameter sets ───────────────────────────────────────────────────
DELTA_VALS = [0.5, 1.0, 2.0]
GAMMA_VALS = [0.25, 0.50, 0.75]


# ── Worker function (must be top-level for pickle) ─────────────────────────────

def _worker(args):
    i, j, hins, hino, h_out, f_s, N, num_sim, seed_base, narcissistic = args
    result = run_one_gridpoint(
        hins=hins, hino=hino, h_out=h_out, f_s=f_s,
        N=N, num_sim=num_sim,
        delta_vals=DELTA_VALS, gamma_vals=GAMMA_VALS,
        seed_base=seed_base, narcissistic=narcissistic,
    )
    return i, j, result


# ── Main logic ─────────────────────────────────────────────────────────────────

def run_sweep(h_out, f_s, N, num_sim, n_points, workers, outdir, narcissistic=True):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hins_vals = np.linspace(0, 1, n_points)   # H_in^s (supporter in-group)
    hino_vals = np.linspace(0, 1, n_points)   # H_in^o (opponent in-group)
    variants  = all_variant_keys(DELTA_VALS, GAMMA_VALS)

    # Pre-allocate result arrays: shape (n_points, n_points) per metric per variant
    metrics = ['beta_s', 'beta_o', 'beta_overall',
               'var_beta_s', 'var_beta_o', 'var_beta_overall']
    arrays = {
        v: {m: np.full((n_points, n_points), np.nan) for m in metrics}
        for v in variants
    }

    # Build task list
    tasks = []
    for i, hino in enumerate(hino_vals):
        for j, hins in enumerate(hins_vals):
            seed_base = i * n_points + j
            tasks.append((i, j, hins, hino, h_out, f_s, N, num_sim, seed_base, narcissistic))

    n_total = len(tasks)
    print(f"Finite-N stochastic block model: h_out={h_out}, f_s={f_s:.4f}, N={N}, "
          f"num_sim={num_sim}, grid={n_points}×{n_points}, workers={workers}, "
          f"narcissistic={narcissistic}")
    print(f"  {n_total} grid points × {len(variants)} variants = "
          f"{n_total * len(variants)} result sets")

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            i, j, result = fut.result()
            for v in variants:
                for m in metrics:
                    arrays[v][m][i, j] = result[v][m]
            n_done += 1
            if n_done % max(1, n_total // 10) == 0:
                print(f"  {n_done}/{n_total} grid points done", flush=True)

    # Save one .npz per variant
    tag = f'hout{h_out:.2f}_fs{f_s:.4f}_N{N}'
    saved = 0
    for v in variants:
        fname = outdir / f'sweep_C_{tag}_{v}.npz'
        np.savez_compressed(
            fname,
            hins_vals=hins_vals, hino_vals=hino_vals,
            **arrays[v],
            h_out=h_out, f_s=f_s, N=N, num_sim=num_sim,
            variant=v, narcissistic=narcissistic,
        )
        saved += 1
    print(f"  Saved {saved} .npz files to {outdir}/")


def main():
    parser = argparse.ArgumentParser(description='Finite-N stochastic block model simulations')
    parser.add_argument('--hout',     type=float, required=True,
                        help='H_out value (cross-group edge probability)')
    parser.add_argument('--fs',       type=float, required=True,
                        help='Supporter fraction f_s')
    parser.add_argument('--N',        type=int, required=True,
                        help='Network size')
    parser.add_argument('--numsim',   type=int, default=200,
                        help='Simulations per grid point (default: 200)')
    parser.add_argument('--n-points', type=int, default=21,
                        help='Grid resolution per axis (default: 21)')
    parser.add_argument('--workers',  type=int, default=8,
                        help='ProcessPoolExecutor workers (default: 8)')
    parser.add_argument('--outdir',  default=None,
                        help='Output directory (default: 07_finite_n_sbm/output/, or output_exclude_self/ with --exclude-self)')
    parser.add_argument('--exclude-self', action='store_true',
                        help='Agents count only their neighbors (narcissistic=False); used for the SI include-self comparison')
    args = parser.parse_args()
    outdir = args.outdir or here() / '07_finite_n_sbm' / ('output_exclude_self' if args.exclude_self else 'output')

    run_sweep(
        h_out=args.hout, f_s=args.fs, N=args.N,
        num_sim=args.numsim, n_points=args.n_points,
        workers=args.workers, outdir=outdir,
        narcissistic=not args.exclude_self,
    )


if __name__ == '__main__':
    main()
