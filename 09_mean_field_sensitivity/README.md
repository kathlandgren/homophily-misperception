# Sensitivity to H_out and f_s in the mean field

Large-N mean-field misperception for the stochastic block model across cross-group connection probabilities H_out and supporter fractions f_s (`run_sweep_A.py`), and for the preferential attachment model across f_s (`run_sweep_B.py`).

## Figures and tables

| Output | Script |
|---|---|
| Figs S6–S8 | `make_sweepAB_figs.py` |
| Table S3 | `make_area_fraction_table.py` |

## Run order

1. `run_sweep_A.py` and `run_sweep_B.py` (locally, about a minute) → `output/`.
2. The scripts above.
