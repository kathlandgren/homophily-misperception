# Effect of agents including their own opinion

Compares every simulation sweep run with agents including their own opinion (Eq. 1, the manuscript's formulation) against the same sweep run with agents counting only their neighbours.

## Figures and tables

| Output | Script |
|---|---|
| Figs S2–S3, Table S2 | `quantify_narcissistic_effect.py` |
| Per-condition summary | `summary_stats/narcissistic_effect_all.csv` (written by the same script) |

## Run order

1. The simulations in `06_swap_simulations/`, `07_finite_n_sbm/`, `08_finite_n_pa/` and `10_degree_matched/`, each run twice: once as is (→ `output/`) and once with `EXCLUDE_SELF=1` (→ `output_exclude_self/`).
2. `quantify_narcissistic_effect.py` (under a minute).
