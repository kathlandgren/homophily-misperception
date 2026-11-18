# Broadcast node model of media bias

Media bias as a broadcast source that every agent attends to with attention share λ and that reports a stance μ, applied analytically to the peer-network misperception from the degree-matched (`10_degree_matched/`) and swap (`06_swap_simulations/`) simulations. Pure numpy; runs locally.

## Figures and tables

| Output | Script |
|---|---|
| Fig S19 | `generate_fig_sweep_G_broadcast_illustrative.py` |
| Figs S20–S21 | `generate_fig_sweep_G_diagrams.py` |
| Swap-vs-broadcast statistics (SI Section S13) | `swap_vs_broadcast_stats.py` (no precomputed data needed) |

## Run order

1. The swap simulations (`06_swap_simulations/`) and the degree-matched simulations (`10_degree_matched/`, including `submit_sweep_E_G3x3.sh`).
2. `run_sweep_G.py` (seconds) → `output/`. It prints how many peer baselines it had to interpolate; this should be 0 once `10_degree_matched/output_G3x3/` exists.
3. The figure scripts above.
