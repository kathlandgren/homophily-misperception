"""
run_sweep_B.py — preferential attachment mean-field misperception sweep over f_s.

Computes the large-N preferential attachment analytical mean field for every combination of:
  - h_ss, h_oo in [0, 1] on an n_points × n_points grid
  - f_s in {0.5, 2/3, 0.8}  (3 values)

Uses the cross-group constraint hso = 1 - hss, hos = 1 - hoo (Karimi et al. 2018).
Produces 3 compressed .npz files (one per f_s).

Runs locally in about a minute.

Usage:
    python 09_mean_field_sensitivity/run_sweep_B.py
    python 09_mean_field_sensitivity/run_sweep_B.py --outdir path/to/output --n-points 101
"""

import argparse
import importlib.util
import pathlib

import numpy as np
from pyprojroot import here

# ── Import 03_analytical/src/functions.py ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location("fun", here() / "03_analytical" / "src" / "functions.py")
fun   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fun)

# ── Parameter grid ─────────────────────────────────────────────────────────────
FS_VALUES = [0.5, 2 / 3, 0.8]
N         = 1000   # large-N limit; only the ratio Ns/No = fs/fo matters


def compute_sweep_B(n_points: int) -> dict:
    """
    Return a dict keyed by f_s → arrays dict.

    Per-point call chain (from 03_analytical/make_figure_main.ipynb):
        hso = 1 - hss,  hos = 1 - hoo   (Karimi constraint)
        C        = compute_C(fo, fs, hoo, hos, hso, hss)
        p_ss/oo/os/so computed from C
        beta_s   = compute_misperception(fs, compute_supporter_perception(...))
        beta_o   = compute_misperception(fs, compute_opponent_perception(...))
        beta_overall = compute_overall_misperception(fs, beta_o, beta_s)
    """
    hss_vals = np.linspace(0, 1, n_points)
    hoo_vals = np.linspace(0, 1, n_points)

    results = {}
    for f_s in FS_VALUES:
        f_o = 1.0 - f_s
        Ns  = round(f_s * N)
        No  = N - Ns

        beta_s_grid       = np.zeros((n_points, n_points))
        beta_o_grid       = np.zeros((n_points, n_points))
        beta_overall_grid = np.zeros((n_points, n_points))

        for i, hoo in enumerate(hoo_vals):
            for j, hss in enumerate(hss_vals):
                hso = 1.0 - hss
                hos = 1.0 - hoo

                C    = fun.compute_C(f_o, f_s, hoo, hos, hso, hss)
                p_oo = fun.compute_p_oo(hoo, hos, C)
                p_ss = fun.compute_p_ss(hss, hso, C)
                p_os = fun.compute_p_os(hos, hoo, C)
                p_so = fun.compute_p_so(hso, hss, C)

                b_o = fun.compute_misperception(
                    f_s, fun.compute_opponent_perception(No, Ns, p_os, p_so, p_oo))
                b_s = fun.compute_misperception(
                    f_s, fun.compute_supporter_perception(No, Ns, p_os, p_so, p_ss))
                b_all = fun.compute_overall_misperception(f_s, b_o, b_s)

                beta_s_grid[i, j]       = b_s
                beta_o_grid[i, j]       = b_o
                beta_overall_grid[i, j] = b_all

        results[float(f_s)] = {
            "hss_vals":    hss_vals,
            "hoo_vals":    hoo_vals,
            "beta_s":      beta_s_grid,
            "beta_o":      beta_o_grid,
            "beta_overall": beta_overall_grid,
            "f_s":         float(f_s),
        }
        print(f"  f_s={f_s:.4f} done")

    return results


def save_results(results: dict, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for f_s, arrays in results.items():
        fname = outdir / f"sweep_B_fs{f_s:.4f}.npz"
        np.savez_compressed(fname, **arrays)
        print(f"  saved {fname.name}")


def main(outdir: pathlib.Path, n_points: int) -> None:
    print(f"Computing preferential attachment mean field: {len(FS_VALUES)} f_s values × {n_points}×{n_points} grid …")
    results = compute_sweep_B(n_points)
    save_results(results, outdir)
    print(f"Done. {len(results)} files written to {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preferential attachment mean-field sweep over f_s")
    parser.add_argument("--outdir",   default=None,
                        help="Output directory for .npz files (default: 09_mean_field_sensitivity/output/)")
    parser.add_argument("--n-points", type=int, default=101,
                        help="Grid resolution along each axis (default: 101)")
    args = parser.parse_args()
    main(pathlib.Path(args.outdir or here() / "09_mean_field_sensitivity" / "output"), args.n_points)
