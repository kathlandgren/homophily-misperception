# Finite-N preferential attachment simulations

Simulated preferential attachment networks at N ∈ {100, 500, 1000} and m ∈ {2, 5, 10}, with and without Bayesian rescaling (per-agent and group-mean). This folder also holds the figure scripts that combine these simulations with the stochastic block model ones in `07_finite_n_sbm/`.

## Figures and tables

| Output | Script |
|---|---|
| Figs S4–S5 | `make_N_convergence_4panel.py` (S4 from `07_finite_n_sbm/`, S5 from this folder) |
| Figs S11–S12 | `make_bayesian_si.py` |
| Figs S13–S16 | `make_rescaling_order_figs.py` (S13–S14 from `07_finite_n_sbm/`, S15–S16 from this folder) |

## Run order

1. `submit_sweep_D.sh` (and `07_finite_n_sbm/submit_sweep_C.sh`) on a SLURM cluster (27 tasks each, up to 8 h) → `output/`. Repeat with `EXCLUDE_SELF=1` for `12_include_self_comparison/`.
2. The scripts above (seconds each).
