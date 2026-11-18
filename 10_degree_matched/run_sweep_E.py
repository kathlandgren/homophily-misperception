"""
run_sweep_E.py — degree-matched preferential attachment and stochastic block model simulations.

Runs num_sim PA and SBM simulations at all 121 points of the
(h_ss, h_oo) ∈ linspace(0, 1, 11)² grid for a single (m, f_s, N) combination,
parallelized across grid points via ProcessPoolExecutor.

SBM is parameterized to match the PA mean degree (2m):
    H_out = 2m / (N·D),  D = f_s²·r_s + 2·f_s·f_o + f_o²·r_o,
    H_in^s = r_s·H_out,  H_in^o = r_o·H_out
    where r_s = h_ss/(1-h_ss), r_o = h_oo/(1-h_oo).

Produces one .npz per (m, f_s, N) task with 11×11 result arrays for both models.

Usage (cluster array job, see submit_sweep_E.sh):
    python 10_degree_matched/run_sweep_E.py --m 2 --fs 0.6667 --N 1000

Usage (exact broadcast-model baselines on the 3×3 grid, see submit_sweep_E_G3x3.sh):
    python 10_degree_matched/run_sweep_E.py --m 2 --fs 0.6667 --N 1000 \
        --h-vals 0.25,0.5,0.75 --outdir 10_degree_matched/output_G3x3

Usage (local test):
    python 10_degree_matched/run_sweep_E.py --m 2 --fs 0.5 --N 100 --numsim 5 --workers 4

Add --exclude-self to write the neighbors-only twin to 10_degree_matched/output_exclude_self/
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

sim = _load("matched_simulation", "src/matched_simulation.py")
run_one_condition = sim.run_one_condition

HSS_VALS = [round(v, 1) for v in np.linspace(0.0, 1.0, 11)]
HOO_VALS = [round(v, 1) for v in np.linspace(0.0, 1.0, 11)]


def _worker(args):
    i, j, hss, hoo, m, f_s, N, num_sim, seed_base, narcissistic = args
    result = run_one_condition(
        hss=hss, hoo=hoo, m=m, f_s=f_s, N=N,
        num_sim=num_sim, seed_base=seed_base, narcissistic=narcissistic,
    )
    return i, j, result


def run_sweep(m, f_s, N, num_sim, workers, outdir, h_vals=None, narcissistic=True):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hss_vals = np.array(h_vals if h_vals is not None else HSS_VALS)
    hoo_vals = np.array(h_vals if h_vals is not None else HOO_VALS)
    n = len(hss_vals)

    metrics = ['beta_s', 'beta_o', 'beta_overall',
               'var_beta_s', 'var_beta_o', 'var_beta_overall']
    arrays = {
        f'{model}_{met}': np.full((n, n), np.nan)
        for model in ('pa', 'sbm') for met in metrics
    }
    matched_H_out = np.full((n, n), np.nan)

    tasks = []
    for i, hoo in enumerate(hoo_vals):
        for j, hss in enumerate(hss_vals):
            tasks.append((i, j, hss, hoo, m, f_s, N, num_sim, i * n + j, narcissistic))

    n_total = len(tasks)
    print(f"Degree-matched: m={m}, f_s={f_s:.4f}, N={N}, num_sim={num_sim}, "
          f"grid={n}×{n}, workers={workers}, narcissistic={narcissistic}")

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            i, j, result = fut.result()
            for model in ('pa', 'sbm'):
                for met in metrics:
                    arrays[f'{model}_{met}'][i, j] = result[model][met]
            matched_H_out[i, j] = result['H_out']
            n_done += 1
            print(f"  {n_done}/{n_total} grid points done", flush=True)

    tag = f'm{m}_fs{f_s:.4f}_N{N}'
    fname = outdir / f'sweep_E_{tag}.npz'
    np.savez_compressed(
        fname,
        hss_vals=hss_vals, hoo_vals=hoo_vals,
        matched_H_out=matched_H_out,
        **arrays,
        m=m, f_s=f_s, N=N, num_sim=num_sim, narcissistic=narcissistic,
    )
    print(f"  Saved {fname}")
    print(f"  H_out range: {matched_H_out.min():.5f} – {matched_H_out.max():.5f}")


def main():
    parser = argparse.ArgumentParser(description='Degree-matched preferential attachment and stochastic block model simulations')
    parser.add_argument('--m',       type=int,   required=True,
                        help='BA edges per new node')
    parser.add_argument('--fs',      type=float, required=True,
                        help='Supporter fraction f_s')
    parser.add_argument('--N',       type=int,   required=True,
                        help='Network size')
    parser.add_argument('--numsim',  type=int,   default=200,
                        help='Simulations per grid point per model (default: 200)')
    parser.add_argument('--workers', type=int,   default=32,
                        help='ProcessPoolExecutor workers (default: 32)')
    parser.add_argument('--outdir',  default=None,
                        help='Output directory (default: 10_degree_matched/output/, or output_exclude_self/ with --exclude-self)')
    parser.add_argument('--exclude-self', action='store_true',
                        help='Agents count only their neighbors (narcissistic=False); used for the SI include-self comparison')
    parser.add_argument('--h-vals',  default=None,
                        help='Comma-separated h grid used for both h_ss and h_oo '
                             '(default: 0, 0.1, ..., 1). E.g. 0.25,0.5,0.75 for the broadcast-model baselines.')
    args = parser.parse_args()
    outdir = args.outdir or here() / '10_degree_matched' / ('output_exclude_self' if args.exclude_self else 'output')
    h_vals = [float(v) for v in args.h_vals.split(',')] if args.h_vals else None

    run_sweep(
        m=args.m, f_s=args.fs, N=args.N,
        num_sim=args.numsim, workers=args.workers, outdir=outdir,
        narcissistic=not args.exclude_self,
        h_vals=h_vals,
    )


if __name__ == '__main__':
    main()
