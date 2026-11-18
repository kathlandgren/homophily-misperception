# Degree-matched comparison of the two models

Preferential attachment networks and stochastic block networks parameterized to the same mean degree (2m), simulated side by side at N ∈ {100, 500, 1000}. A second, smaller run on the 3×3 grid h ∈ {0.25, 0.5, 0.75} supplies the exact peer baselines for the broadcast model in `11_broadcast_media/`.

## Figures and tables

| Output | Script |
|---|---|
| Figs S9–S10 | `make_sweepE_figs.py` |

## Run order

1. `submit_sweep_E.sh` (27 tasks, up to 8 h each) → `output/` and `submit_sweep_E_G3x3.sh` (9 tasks) → `output_G3x3/`, on a SLURM cluster. Repeat `submit_sweep_E.sh` with `EXCLUDE_SELF=1` for `12_include_self_comparison/`.
2. `make_sweepE_figs.py` (seconds).
