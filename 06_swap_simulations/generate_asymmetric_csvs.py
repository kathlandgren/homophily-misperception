"""
Generate per-condition summary CSVs from asymmetric swap pkl files.

Reads sim_1000 _maj and _min pkl files from asymmetric_output/ (or --pkl-dir),
computes per-swap-step mean and std statistics, and writes one CSV per
(condition, m) to swapped_data_v2/ (or --out-dir).

Usage:
    python generate_asymmetric_csvs.py
    python generate_asymmetric_csvs.py --pkl-dir /path/to/pkls --out-dir /path/to/csvs
"""

import argparse
import csv
import glob
import itertools
import os
import pickle

import numpy as np

CONDITIONS = [
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
    (0.25, 0.75),
    (0.75, 0.25),
    (0.50, 0.75),
    (0.75, 0.50),
]
M_VEC             = [2, 5]
NUM_AGENTS        = 1000
MINORITY_FRACTION = 1 / 3


def h_to_str(h):
    """Convert homophily float to filename string, e.g. 0.75 -> '0p75'."""
    return str(h).replace(".", "p")


def find_pkl(pkl_dir, h_ss, h_oo, m, num_agents, suffix, prefer_sim=1000):
    """Return path to the preferred pkl file; fall back to highest sim count found."""
    hss = h_to_str(h_ss)
    hoo = h_to_str(h_oo)
    pattern = os.path.join(
        pkl_dir,
        f"hss_{hss}_hoo_{hoo}_m_{m}_num_agents_{num_agents}_sim_*_{suffix}.pkl",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    preferred = [p for p in matches if f"_sim_{prefer_sim}_" in p]
    return preferred[0] if preferred else matches[-1]


def process_condition(pkl_dir, h_ss, h_oo, m, prefer_sim=1000):
    maj_path = find_pkl(pkl_dir, h_ss, h_oo, m, NUM_AGENTS, "maj", prefer_sim=prefer_sim)
    min_path = find_pkl(pkl_dir, h_ss, h_oo, m, NUM_AGENTS, "min", prefer_sim=prefer_sim)

    if maj_path is None:
        print(f"  MISSING maj pkl: hss={h_ss}, hoo={h_oo}, m={m}")
        return None
    if min_path is None:
        print(f"  MISSING min pkl: hss={h_ss}, hoo={h_oo}, m={m}")
        return None

    print(f"  maj: {os.path.basename(maj_path)}")
    print(f"  min: {os.path.basename(min_path)}")

    with open(maj_path, "rb") as f:
        maj = pickle.load(f)   # shape: (num_sim, num_swaps+1, num_maj_agents)
    with open(min_path, "rb") as f:
        minn = pickle.load(f)  # shape: (num_sim, num_swaps+1, num_min_agents)

    num_maj   = int(NUM_AGENTS * (1 - MINORITY_FRACTION))
    num_min   = NUM_AGENTS - num_maj
    num_swaps = maj.shape[1] - 1

    rows = []
    for k in range(num_swaps + 1):
        maj_k = maj[:, k, :]    # (num_sim, num_maj)
        min_k = minn[:, k, :]   # (num_sim, num_min)

        # Per-simulation means (shape: num_sim)
        sim_mean_maj = np.nanmean(maj_k, axis=1)
        sim_mean_min = np.nanmean(min_k, axis=1)
        sim_mean_all = (sim_mean_maj * num_maj + sim_mean_min * num_min) / NUM_AGENTS

        mean_maj = sim_mean_maj.mean()
        mean_min = sim_mean_min.mean()
        mean_all = sim_mean_all.mean()

        rows.append({
            "swap_count":                    k,
            "mean_opinion_percent":          mean_all,
            "mean_majority_opinion_percent": mean_maj,
            "mean_minority_opinion_percent": mean_min,
            "mean_misp":                     (1 - MINORITY_FRACTION) * 100 - mean_all,
            "std_opinion_percent":           sim_mean_all.std(),
            "std_majority_opinion_percent":  sim_mean_maj.std(),
            "std_minority_opinion_percent":  sim_mean_min.std(),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl-dir", default="asymmetric_output",
        help="Directory containing *_maj.pkl and *_min.pkl files",
    )
    parser.add_argument(
        "--out-dir", default="swapped_data_v2",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--sim-count", type=int, default=1000,
        help="Preferred simulation count to load (falls back to highest available)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for (h_ss, h_oo), m in itertools.product(CONDITIONS, M_VEC):
        print(f"\nhss={h_ss}, hoo={h_oo}, m={m}")
        df = process_condition(args.pkl_dir, h_ss, h_oo, m, prefer_sim=args.sim_count)
        if df is None:
            continue

        hss_label = str(int(h_ss * 100))
        hoo_label = str(int(h_oo * 100))
        out_path  = os.path.join(
            args.out_dir, f"swap_sim_hss_{hss_label}_hoo_{hoo_label}_m{m}.csv"
        )
        fieldnames = list(df[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(df)
        print(f"  -> {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
