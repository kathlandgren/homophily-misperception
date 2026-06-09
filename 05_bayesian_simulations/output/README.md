# Bayesian rescaling simulation outputs

This directory contains precomputed simulation results (`.npz` files) used by `make_supplementary.ipynb` to generate Figures S2 and S3.

## What is here

121 compressed NumPy archives produced by the Bayesian rescaling simulation pipeline, run on a compute cluster (Stanford Sherlock). Each file covers one combination of:

- **N** ∈ {100, 500} — network size
- **m** ∈ {2, 5, 10} — edges added per node in the BA model
- **delta** ∈ {0.5, 1, 2} — Bayesian prior strength
- **gamma** ∈ {0.25, 0.5, 0.75, 1} — perception uncertainty

Each file stores a grid of misperception values over (h_ss, h_oo) ∈ [0,1]². The notebook's `load_and_average` function finds all files matching a given (N, m, delta, gamma) and averages across seeds, preferring higher-resolution runs.

## To regenerate from scratch

Run `run_simulations.py` as an array job on a cluster, then place the resulting `.npz` files here. The naming convention is:

```
misperception_grid_N{N}_m{m}_fs{fs:.4f}_n{n_points}_seed{seed}_numsim{numsim}_bayes1_delta{delta}_gamma{gamma}.npz
```
