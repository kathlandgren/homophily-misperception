# Code for: "Can homophily explain public underestimation of climate policy support?"

**Ekaterina Landgren, Shriya Nagpal, Joshua Garland, Yaw Acquah, Matthew G. Burgess**

This repository contains replication code for the network simulations and figures in the manuscript studying how network homophily produces systematic opinion misperception.

## Dependencies

```bash
pip install -r requirements.txt
```

## Empirical data

The empirical analyses (Figure 1, Figure S1) use survey data from:

> Sparkman, Gregg, Nathan Geiger, and Elke U. Weber. "Americans experience a false social reality by underestimating popular climate policy support by nearly half." *Nature Communications* 13.1 (2022): 4779.

The raw data files with survey participant data are not tracked in this repository. See `01_empirical/data/README.md`.


## Repository structure

```
src/                           Shared simulation library (network generation, opinion dynamics)
01_empirical/                  Empirical pattern analysis (Figs 1, S1)
02_network_illustrations/      Conceptual diagrams for Fig 2
03_analytical/                 Closed-form mathematical results (Figs 3, 4 panels F–H)
04_homophily_simulations/      Baseline homophily-only simulations
05_bayesian_simulations/       Bayesian perception distortion (Figs 4, S2–S3)
06_swap_simulations/           Robustness checks with node swaps (Figs 5, S4–S5, Tables S1–S7)
figures/                       All figure outputs; .tex files for assembled multi-panel figures
```

## Quick start

These notebooks require no additional data and can be run right away:

- `03_analytical/make_figure_main.ipynb` — Fig 3
- `03_analytical/make_figure_bayesian.ipynb` — Fig 4 panels F–H
- `02_network_illustrations/create_cartoon.ipynb` — Fig 2 (left)
- `02_network_illustrations/create_misperception_example.ipynb` — Fig 2 (right)
- `05_bayesian_simulations/make_scurve_figure.ipynb` — Fig 4 panel A
- `05_bayesian_simulations/make_histogram_panel.ipynb` — Fig 4 panels B–E
- `05_bayesian_simulations/make_supplementary.ipynb` — Figs S2–S3 (precomputed data included)
- `06_swap_simulations/generate_swap_figure.ipynb` — Fig 5 panel A (precomputed data included)
- `06_swap_simulations/generate_opinion_histograms.ipynb` — Fig 5 panel B
- `06_swap_simulations/generate_asymmetric_figures.ipynb` — Figs S4–S5 (precomputed data included)

These notebooks need external data before they can be run:

- `01_empirical/*.ipynb` — requires `participant_data.csv` (Sparkman et al. 2022; see `01_empirical/data/README.md`)
- `04_homophily_simulations/compute_summary_statistics.ipynb` — run `04_homophily_simulations/run_simulations.py` first
- `06_swap_simulations/compute_summary_statistics.ipynb` — run `06_swap_simulations/run_simulations.py` first

## Reproducing the figures

Run the steps below in order. Simulation scripts (`run_simulations.py`) are computationally expensive and use `ProcessPoolExecutor`; notebooks that only visualize precomputed results can be run independently once their simulation data exists.

| Figure | Script / Notebook | Output file |
|--------|-------------------|-------------|
| Fig 1 | `01_empirical/carbon_tax_histograms.ipynb` | `figures/carbon_tax_fig1_combined.pdf` |
| Fig 2 (left) | `02_network_illustrations/create_cartoon.ipynb` | `figures/figure_1_cartoon.pdf` |
| Fig 2 (right) | `02_network_illustrations/create_misperception_example.ipynb` | `figures/figure_1_example.pdf` |
| Fig 3 | `03_analytical/make_figure_main.ipynb` | `figures/figure_2.pdf` |
| Fig 4 (panel A) | `05_bayesian_simulations/make_scurve_figure.ipynb` | `figures/rescale_bayes_variants.pdf` |
| Fig 4 (panels B–E) | `05_bayesian_simulations/make_histogram_panel.ipynb` | `figures/figure_histogram_example_grouped.pdf` |
| Fig 4 (panels F–H) | `03_analytical/make_figure_bayesian.ipynb` | `figures/figure_bayesian_rescaling_SBM_BA.pdf` |
| Fig 4 (assembled) | compile `figures/combine_bayesian_fig.tex` | `figures/combine_bayesian_fig.pdf` |
| Fig 5 (panel A) | precomputed — already in `figures/` | `figures/single_panel_high_homophily_without_legend.pdf` |
| Fig 5 (panel B) | `05_bayesian_simulations/make_histogram_panel.ipynb` | (same as Fig 4 B–E) |
| Fig 5 (assembled) | compile `figures/combine_swap_fig.tex` | `figures/combine_swap_fig.pdf` |
| Fig S1 | `01_empirical/correlation_heatmap.ipynb` | `figures/correlation_matrix.pdf` |
| Fig S2–S3 | `05_bayesian_simulations/make_supplementary.ipynb` (loads precomputed data from `05_bayesian_simulations/output/`) | `figures/supp_bayesian_N_vs_delta.pdf`, `figures/supp_bayesian_m_vs_delta.pdf` |
| Fig S4–S5 | `06_swap_simulations/generate_asymmetric_figures.ipynb` | `figures/figure_asymmetric_swaps_full.pdf`, `figures/figure_asymmetric_swaps_diagram.pdf` |
| Tables S1–S7 | `06_swap_simulations/compute_summary_statistics.ipynb` | CSV data in `06_swap_simulations/summary_stats/` |

To compile the assembled figures, run `pdflatex` from inside `figures/`:
```bash
cd figures && pdflatex combine_bayesian_fig.tex && pdflatex combine_swap_fig.tex
```

## Simulation pipeline

For figures that require precomputed simulation data:

```bash
# Baseline homophily-only (outputs .pkl to 04_homophily_simulations/output/)
python 04_homophily_simulations/run_simulations.py

# Node-swap robustness (outputs .pkl to 06_swap_simulations/output/)
python 06_swap_simulations/run_simulations.py

# Asymmetric swaps — designed for HPC cluster
python 06_swap_simulations/run_asymmetric_swaps.py

# Bayesian rescaling (outputs .npz to 05_bayesian_simulations/output/)
python 05_bayesian_simulations/run_simulations.py
```

The scripts sweep homophily ∈ {0, 0.25, 0.5, 0.75, 1}, m ∈ {2, 5}, num_agents ∈ {1000, 10000}.


## Attribution for `src/`

The network generators in `src/` (`generate_homophilic_graph_symmetric.py`, `generate_homophilic_graph_asymmetric.py`, `generate_homophilic_SB_graph.py`) are based on code originally written by Fariba Karimi (2016) for the preferential attachment model with homophily described in:

> Karimi, Fariba, Mathieu Génois, Claudia Wagner, Philipp Singer, and Markus Strohmaier. "Homophily influences ranking of minorities in social networks." *Scientific Reports* 8.1 (2018): 11077.

The code has been substantially modified for this project (asymmetric homophily, stochastic block model variant, opinion assessment and misperception computation in `opinion_functions.py`).
