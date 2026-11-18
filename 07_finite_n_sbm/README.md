# Finite-N stochastic block model simulations

Simulated stochastic block networks at N ∈ {100, 500, 1000}, with and without Bayesian rescaling (per-agent and group-mean), for comparison with the mean-field results of Theorem 1.

## Figures and tables

The figures built from this sweep are made by the scripts in `08_finite_n_pa/`, which plot both models together:

| Output | Script |
|---|---|
| Fig S4 | `08_finite_n_pa/make_N_convergence_4panel.py` |
| Figs S13–S14 | `08_finite_n_pa/make_rescaling_order_figs.py` |

## Run order

1. `submit_sweep_C.sh` on a SLURM cluster (27 tasks, up to 2 h each) → `output/`. Repeat with `EXCLUDE_SELF=1` for `12_include_self_comparison/`.
2. The scripts in `08_finite_n_pa/`.
