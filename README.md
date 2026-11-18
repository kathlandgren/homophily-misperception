# Replication code: "Can homophily explain public underestimation of climate policy support?"

**Ekaterina Landgren, Shriya Nagpal, Joshua Garland, Yaw Acquah, Matthew G. Burgess**

Preprint: [arXiv:2606.15106](https://arxiv.org/abs/2606.15106)

This repository contains replication code for the network simulations, analytical results and figures in the manuscript studying how network homophily produces systematic opinion misperception.

## Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `matplotlib`, `numpy`, `networkx`, `pandas`, `seaborn`, `scipy`, `pyprojroot`. The Fig 2 cartoon is rendered with LaTeX (`text.usetex`), and the assembled Figs 4 and 5 are compiled with `pdflatex`, so a LaTeX installation is needed for those.

All paths are resolved with `pyprojroot.here()` from the `.here` file at the repository root, so run scripts and notebooks from anywhere inside the repository.

## Empirical data

The empirical analyses (Figure 1, Figure S1) use survey data from:

> Sparkman, Gregg, Nathan Geiger, and Elke U. Weber. "Americans experience a false social reality by underestimating popular climate policy support by nearly half." *Nature Communications* 13.1 (2022): 4779.

The raw data files with survey participant data are not tracked in this repository. See `01_empirical/data/README.md`.

## Quick start: what runs immediately

These require no precomputed data and run on a laptop in seconds to minutes:

- `02_network_illustrations/create_cartoon.ipynb` — Fig 2 (left)
- `02_network_illustrations/create_misperception_example.ipynb` — Fig 2 (right)
- `03_analytical/make_figure_main.ipynb` — Fig 3
- `03_analytical/make_figure_bayesian.ipynb` — Fig 4 panels C–H
- `05_bayesian_simulations/make_scurve_figure.ipynb` — Fig 4 panel A
- `05_bayesian_simulations/make_histogram_panel.ipynb` — Fig 4 panel B
- `06_swap_simulations/generate_opinion_histograms.ipynb` — Fig 5 panel B
- `09_mean_field_sensitivity/` — Figs S6–S8, Table S3 (run `run_sweep_A.py` and `run_sweep_B.py` first; about a minute)
- `11_broadcast_media/swap_vs_broadcast_stats.py` — statistics quoted in SI Section S13

These need external data first (see the note at the top of each):

- `01_empirical/*.ipynb` — requires `participant_data.csv` (Sparkman et al. 2022; see `01_empirical/data/README.md`)

These need simulation outputs that are run on a compute cluster as SLURM array jobs (`submit_*.sh`); the outputs are not tracked in git. Each `output/README.md` lists the files, the grid and the runtime.

- `06_swap_simulations/` (media-bias swaps) — Fig 5 panel A, Figs S17–S18, Tables S4–S18
- `07_finite_n_sbm/` and `08_finite_n_pa/` (finite-N simulations of both models) — Figs S4–S5, S11–S16
- `10_degree_matched/` (degree-matched comparison) — Figs S9–S10
- `11_broadcast_media/` (broadcast node model; runs locally once 06 and 10 have outputs) — Figs S19–S21
- `12_include_self_comparison/` (needs 06, 07, 08 and 10, each also run with `EXCLUDE_SELF=1`) — Figs S2–S3, Table S2

## Reproducing the figures

Figures are written to `figures/` and LaTeX tables to `tables/`, under the file names used in the manuscript. Scripts that need simulation outputs say so in the note at the top of each.

| Figure | Script / Notebook | Output file |
|--------|-------------------|-------------|
| Fig 1 | `01_empirical/carbon_tax_histograms.ipynb` | `figures/carbon_tax_fig1_combined.pdf` |
| Fig 2 (left) | `02_network_illustrations/create_cartoon.ipynb` | `figures/figure_1_cartoon.pdf` |
| Fig 2 (right) | `02_network_illustrations/create_misperception_example.ipynb` | `figures/figure_1_example.pdf` |
| Fig 3 | `03_analytical/make_figure_main.ipynb` | `figures/figure_2.pdf` |
| Fig 4 (panel A) | `05_bayesian_simulations/make_scurve_figure.ipynb` | `figures/rescale_bayes_variants.pdf` |
| Fig 4 (panel B) | `05_bayesian_simulations/make_histogram_panel.ipynb` | `figures/figure_histogram_example_grouped_bayes_narcTrue.pdf` |
| Fig 4 (panels C–H) | `03_analytical/make_figure_bayesian.ipynb` | `figures/figure_bayesian_rescaling_SBM_BA.pdf` |
| Fig 4 (assembled) | compile `figures/combine_bayesian_fig_narcTrue.tex` | `figures/combine_bayesian_fig_narcTrue.pdf` |
| Fig 5 (panel A) | `06_swap_simulations/make_fig5A.py` | `figures/single_panel_high_homophily_without_legend_narcTrue.pdf` |
| Fig 5 (panel B) | `06_swap_simulations/generate_opinion_histograms.ipynb` | `figures/figure_histogram_example_grouped_swap_narcTrue.pdf` |
| Fig 5 (assembled) | compile `figures/combine_swap_fig_narcTrue.tex` | `figures/combine_swap_fig_narcTrue.pdf` |
| Fig S1 | `01_empirical/correlation_heatmap.ipynb` | `figures/correlation_matrix.pdf` |
| Figs S2–S3, Table S2 | `12_include_self_comparison/quantify_narcissistic_effect.py` | `figures/fig_si_narcissistic_{comparison,diff}.pdf`, `tables/narcissistic_effect_headline.tex` |
| Figs S4–S5 | `08_finite_n_pa/make_N_convergence_4panel.py` | `figures/fig_sweep_{C,D}_N_convergence_4panel.pdf` |
| Figs S6–S7 | `09_mean_field_sensitivity/make_sweepAB_figs.py` | `figures/fig_sweep_A_hout_selected.pdf`, `figures/fig_sweep_A_fs_variation.pdf` |
| Fig S8 | `09_mean_field_sensitivity/make_sweepAB_figs.py` | `figures/fig_sweep_B_fs_variation.pdf` |
| Table S3 | `09_mean_field_sensitivity/make_area_fraction_table.py` | `tables/area_fraction_table.tex` |
| Figs S9–S10 | `10_degree_matched/make_sweepE_figs.py` | `figures/fig_sweep_E_pa_vs_sbm.pdf`, `figures/fig_sweep_E_grid_diff.pdf` |
| Figs S11–S12 | `08_finite_n_pa/make_bayesian_si.py` | `figures/supp_bayesian_N_vs_delta.pdf`, `figures/supp_bayesian_m_vs_delta.pdf` |
| Figs S13–S16 | `08_finite_n_pa/make_rescaling_order_figs.py` | `figures/fig_sweep_{C,D}_rescaling_{comparison,diff}.pdf` |
| Fig S17 | `06_swap_simulations/generate_fig_asymmetric_swaps_full.py` | `figures/figure_asymmetric_swaps_full.pdf` |
| Fig S18 | `06_swap_simulations/make_fig_sweep_F_diagram.py` | `figures/fig_sweep_F_diagram.pdf` |
| Table S4 | `06_swap_simulations/quantify_ordering_window.py --latex` | `tables/ordering_window_table.tex` |
| Tables S5–S18 | `06_swap_simulations/generate_subset_tables.py` | `tables/si_tables.tex` |
| Fig S19 | `11_broadcast_media/generate_fig_sweep_G_broadcast_illustrative.py` | `figures/fig_sweep_G_broadcast_illustrative.pdf` |
| Figs S20–S21 | `11_broadcast_media/generate_fig_sweep_G_diagrams.py` | `figures/fig_sweep_G_diagram_{pa,sbm}.pdf` |

To compile the assembled figures, run `pdflatex` from inside `figures/`:
```bash
cd figures && pdflatex combine_bayesian_fig_narcTrue.tex && pdflatex combine_swap_fig_narcTrue.tex
```

## Repository structure

```
src/                           Shared library: network generators, opinion assessment, sweep simulation cores, SI plot style
01_empirical/                  Empirical pattern analysis (Fig 1, Fig S1)
02_network_illustrations/      Conceptual diagrams for Fig 2
03_analytical/                 Closed-form mathematical results (Fig 3, Fig 4 panels C–H)
04_homophily_simulations/      Baseline homophily-only simulations
05_bayesian_simulations/       Bayesian perception distortion example (Fig 4 panels A–B)
06_swap_simulations/           Media bias as node swaps (Fig 5, Figs S17–S18, Tables S4–S18)
07_finite_n_sbm/               Finite-N stochastic block model simulations
08_finite_n_pa/                Finite-N preferential attachment simulations; figures for 07 and 08 (Figs S4–S5, S11–S16)
09_mean_field_sensitivity/     Sensitivity to H_out and f_s in the mean field (Figs S6–S8, Table S3)
10_degree_matched/             Degree-matched comparison of the two models (Figs S9–S10)
11_broadcast_media/            Broadcast node model of media bias (Figs S19–S21)
12_include_self_comparison/    Effect of agents including their own opinion (Figs S2–S3, Table S2)
figures/                       All figure outputs; .tex files for assembled multi-panel figures
tables/                        LaTeX tables for the SI
```

## Attribution for code in `src/`

The network generators in `src/` (`generate_homophilic_graph_symmetric.py`, `generate_homophilic_graph_asymmetric.py`, `generate_homophilic_SB_graph.py`, `generate_homophilic_SB_graph_vectorized.py`) are based on code originally written by Fariba Karimi (2016) for the preferential attachment model with homophily described in:

> Karimi, Fariba, Mathieu Génois, Claudia Wagner, Philipp Singer, and Markus Strohmaier. "Homophily influences ranking of minorities in social networks." *Scientific Reports* 8.1 (2018): 11077.

The code has been substantially modified for this project (asymmetric homophily, stochastic block model variant, opinion assessment and misperception computation).
