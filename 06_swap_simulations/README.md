# Media bias as node swaps

Media bias is modelled by swapping the opinions of the best-connected supporters with the least-connected opponents, on preferential attachment networks (N = 1000, 200 simulations per condition).

## Figures and tables

| Output | Script |
|---|---|
| Fig 5 panel A | `make_fig5A.py` |
| Fig 5 panel B | `generate_opinion_histograms.ipynb` (runs without precomputed data) |
| Fig S17 | `generate_fig_asymmetric_swaps_full.py` |
| Fig S18 | `make_fig_sweep_F_diagram.py` |
| Table S4 | `quantify_ordering_window.py --latex` |
| Tables S5–S18 | `generate_subset_tables.py` |

## Run order

1. `submit_sweep_F.sh` on a SLURM cluster (6 tasks, up to 8 h each) → `output/`. Repeat with `EXCLUDE_SELF=1` for `12_include_self_comparison/`.
2. Any of the scripts above (seconds each). Then compile `figures/combine_swap_fig_narcTrue.tex` for the assembled Fig 5.
