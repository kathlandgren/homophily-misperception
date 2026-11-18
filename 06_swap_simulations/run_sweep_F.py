"""
run_sweep_F.py — media-bias swap simulations.

For a single (m, f_s) combination, runs num_sim PA simulations at every point of
the 6×6 (h_ss, h_oo) ∈ {0.25,0.375,0.5,0.625,0.75,0.875}² grid, measuring β_all
and β_peers at ~96 swap-fraction points (0–20% in ~0.25% steps then 20%–minority
cap in 1% steps).

Swap cap: stops at N_minority = round((1−f_s)×N). Beyond that, all original
opponents have been promoted and the simulation would be cycling within formerly-
majority nodes.

Produces one .npz per (m, f_s, N) task.

Usage (cluster array job, see submit_sweep_F.sh):
    python 06_swap_simulations/run_sweep_F.py --m 10 --fs 0.6667 --N 1000

Usage (local test):
    python 06_swap_simulations/run_sweep_F.py --m 2 --fs 0.6667 --N 100 --numsim 5 --workers 4

Add --exclude-self to write the neighbors-only twin to 06_swap_simulations/output_exclude_self/
(needed by 12_include_self_comparison and the Fig 5A check).
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

sim = _load("swap_simulation", "src/swap_simulation.py")
run_one_condition = sim.run_one_condition

HSS_VALS = [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
HOO_VALS = [0.25, 0.375, 0.50, 0.625, 0.75, 0.875]


def compute_swap_counts(f_s, N):
    """Return sorted integer swap counts, capped at N_minority = round((1-f_s)*N)."""
    n_min = round((1.0 - f_s) * N)
    # 0–20% range at ~0.25% steps (2.5 swaps per step for N=1000)
    step = 0.0025 * N
    coarse = sorted(set(round(step * k) for k in range(int(0.20 * N / step) + 2)))
    coarse = [c for c in coarse if c <= round(0.20 * N)]
    # 21–35% range at 1% steps
    pct_step = max(1, round(0.01 * N))
    fine = list(range(round(0.21 * N) // pct_step * pct_step,
                      round(0.35 * N) + 1, pct_step))
    all_pts = sorted(set(coarse + fine))
    return [p for p in all_pts if p <= n_min]


def _worker(args):
    i, j, hss, hoo, m, f_s, N, num_sim, swap_counts, seed_base, narcissistic = args
    result = run_one_condition(
        hss=hss, hoo=hoo, m=m, f_s=f_s, N=N,
        num_sim=num_sim, swap_counts=swap_counts, seed_base=seed_base,
        narcissistic=narcissistic,
    )
    return i, j, result


def run_sweep(m, f_s, N, num_sim, workers, outdir, narcissistic=True):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hss_vals   = np.array(HSS_VALS)
    hoo_vals   = np.array(HOO_VALS)
    nh         = len(hss_vals)
    swap_counts = compute_swap_counts(f_s, N)
    swap_fracs  = np.array(swap_counts) / N
    n_meas      = len(swap_counts)

    print(f"Swap simulations: m={m}, f_s={f_s:.4f}, N={N}, num_sim={num_sim}, "
          f"grid={nh}×{nh}, swap_meas={n_meas} (max={swap_counts[-1]} swaps), "
          f"workers={workers}, narcissistic={narcissistic}")

    metrics = ['beta_s', 'beta_o', 'beta_overall',
               'var_beta_s', 'var_beta_o', 'var_beta_overall']
    arrays = {
        f'{kind}_{met}': np.full((nh, nh, n_meas), np.nan)
        for kind in ('all', 'peers') for met in metrics
    }

    tasks = [
        (i, j, hss, hoo, m, f_s, N, num_sim, swap_counts, i * nh + j, narcissistic)
        for i, hoo in enumerate(hoo_vals)
        for j, hss in enumerate(hss_vals)
    ]

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            i, j, result = fut.result()
            for kind in ('all', 'peers'):
                for met in metrics:
                    arrays[f'{kind}_{met}'][i, j, :] = result[kind][met]
            n_done += 1
            print(f"  {n_done}/{len(tasks)} conditions done", flush=True)

    tag  = f'm{m}_fs{f_s:.4f}_N{N}'
    fname = outdir / f'sweep_F_{tag}.npz'
    np.savez_compressed(
        fname,
        hss_vals=hss_vals, hoo_vals=hoo_vals,
        swap_counts=np.array(swap_counts), swap_fracs=swap_fracs,
        **arrays,
        m=m, f_s=f_s, N=N, num_sim=num_sim, narcissistic=narcissistic,
    )
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description='Media-bias swap simulations')
    parser.add_argument('--m',       type=int,   required=True)
    parser.add_argument('--fs',      type=float, required=True)
    parser.add_argument('--N',       type=int,   default=1000)
    parser.add_argument('--numsim',  type=int,   default=200)
    parser.add_argument('--workers', type=int,   default=36)
    parser.add_argument('--outdir',  default=None,
                        help='Output directory (default: 06_swap_simulations/output/, or output_exclude_self/ with --exclude-self)')
    parser.add_argument('--exclude-self', action='store_true',
                        help='Agents count only their neighbors (narcissistic=False); used for the SI include-self comparison')
    args = parser.parse_args()
    outdir = args.outdir or here() / '06_swap_simulations' / ('output_exclude_self' if args.exclude_self else 'output')

    run_sweep(
        m=args.m, f_s=args.fs, N=args.N,
        num_sim=args.numsim, workers=args.workers, outdir=outdir,
        narcissistic=not args.exclude_self,
    )


if __name__ == '__main__':
    main()
