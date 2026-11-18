"""
run_sweep_A.py — stochastic block model mean-field misperception sweep over H_out and f_s.

Computes Theorem 1 (large-N SBM mean field) analytically for every combination of:
  - H_in^s, H_in^o in [0, 1] on an n_points × n_points grid
  - H_out in {0, 0.1, ..., 1.0}  (11 values)
  - f_s   in {0.5, 2/3, 0.8}     (3 values)

Produces 33 compressed .npz files (one per (H_out, f_s) pair), each containing
the full 2-D grids of beta_s, beta_o, and beta_overall.

Runs locally in seconds.

Usage:
    python 09_mean_field_sensitivity/run_sweep_A.py
    python 09_mean_field_sensitivity/run_sweep_A.py --outdir path/to/output --n-points 101
"""

import argparse
import pathlib

import numpy as np
from pyprojroot import here

# ── Parameter grid ─────────────────────────────────────────────────────────────
H_OUT_VALUES = np.round(np.linspace(0, 1, 11), 10)   # {0, 0.1, ..., 1.0}
FS_VALUES    = [0.5, 2 / 3, 0.8]
EPS          = 1e-10   # denominator clipping (matches 03_analytical/make_figure_main.ipynb)


def compute_sweep_A(n_points: int) -> dict:
    """
    Return a dict keyed by (H_out, f_s) → arrays dict.

    Formulas (from 03_analytical/make_figure_main.ipynb, G1_orig / G2_orig):
        x_s        = f_s * H_in_s / (f_s * H_in_s + f_o * H_out)
        x_o        = f_s * H_out  / (f_s * H_out  + f_o * H_in_o)
        beta_s     = f_s - x_s
        beta_o     = f_s - x_o
        beta_overall = f_s * beta_s + f_o * beta_o
    Denominators are clipped to EPS to avoid division by zero.
    """
    hins_vals = np.linspace(0, 1, n_points)   # x-axis: H_in^s (supporter in-group prob)
    hino_vals = np.linspace(0, 1, n_points)   # y-axis: H_in^o (opponent in-group prob)

    # 2-D broadcasting grids: shape (n_points, n_points)
    # Row index → hino (y), column index → hins (x), matching imshow with origin="lower"
    HINS, HINO = np.meshgrid(hins_vals, hino_vals)

    results = {}
    for h_out in H_OUT_VALUES:
        for f_s in FS_VALUES:
            f_o = 1.0 - f_s

            denom_s = np.maximum(f_s * HINS + f_o * h_out, EPS)
            denom_o = np.maximum(f_s * h_out + f_o * HINO, EPS)

            beta_s       = f_s - (f_s * HINS) / denom_s
            beta_o       = f_s - (f_s * h_out) / denom_o
            beta_overall = f_s * beta_s + f_o * beta_o

            results[(float(h_out), float(f_s))] = {
                "hins_vals":    hins_vals,
                "hino_vals":    hino_vals,
                "beta_s":       beta_s,
                "beta_o":       beta_o,
                "beta_overall": beta_overall,
                "H_out":        float(h_out),
                "f_s":          float(f_s),
            }
    return results


def save_results(results: dict, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for (h_out, f_s), arrays in results.items():
        fname = outdir / f"sweep_A_hout{h_out:.2f}_fs{f_s:.4f}.npz"
        np.savez_compressed(fname, **arrays)
        print(f"  saved {fname.name}")


def main(outdir: pathlib.Path, n_points: int) -> None:
    print(f"Computing stochastic block model mean field: {len(H_OUT_VALUES)} H_out × {len(FS_VALUES)} f_s "
          f"× {n_points}×{n_points} grid …")
    results = compute_sweep_A(n_points)
    save_results(results, outdir)
    print(f"Done. {len(results)} files written to {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic block model mean-field sweep over H_out and f_s")
    parser.add_argument("--outdir",   default=None,
                        help="Output directory for .npz files (default: 09_mean_field_sensitivity/output/)")
    parser.add_argument("--n-points", type=int, default=101,
                        help="Grid resolution along each axis (default: 101)")
    args = parser.parse_args()
    main(pathlib.Path(args.outdir or here() / "09_mean_field_sensitivity" / "output"), args.n_points)
