"""
make_bayesian_si.py — SI figures supp_bayesian_N_vs_delta.pdf and supp_bayesian_m_vs_delta.pdf
(preferential attachment networks with per-agent Bayesian rescaling), built from the finite-N preferential attachment simulations.

The `peragent_d{delta}_g{gamma}` variant of those simulations is the computation of the first-submission
Bayesian pipeline (each agent's perceived fraction passed through rescale_bayes, then averaged
by group), with agents including their own opinion and the correct group weights
(beta_overall = f_s*beta_s + f_o*beta_o; the first-submission pipeline had them reversed).
Grid is 21x21; 200 sims per grid point.

  Fig S-N : rows N in {100, 500}, cols delta in {0.5, 1, 2}; m = 5, gamma = 0.5
  Fig S-m : rows m in {2, 5, 10}, cols delta in {0.5, 1, 2}; N = 500, gamma = 0.5
Both use f_s = 2/3 and only N <= 500, so they do not depend on the N = 1000 tasks.

Data required: 08_finite_n_pa/output/ (submit_sweep_D.sh).

Usage:
    python 08_finite_n_pa/make_bayesian_si.py                 # manuscript figures -> figures/
    python 08_finite_n_pa/make_bayesian_si.py --exclude-self  # same figures from output_exclude_self/
                                                              # (check; written as *_exclude_self.pdf)
"""

import argparse
import importlib.util
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

S = _load("plot_style", "src/plot_style.py")

DIRS = {True:  here() / '08_finite_n_pa' / 'output',
        False: here() / '08_finite_n_pa' / 'output_exclude_self'}
OUT_DIR = here() / 'figures'

GAMMA, FS_TAG = 0.5, '0.6667'
DELTAS = [0.5, 1, 2]
CONTOUR_LEVELS = [0.1, 0.2, 0.25, 0.30]
MYBLACK = '#222222'
TITLE_SIZE, LABEL_SIZE = 9, 8
plt.rc('font', size=8)
plt.rc('axes', titlesize=8, labelsize=9)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)


def load(narc, N, m, delta):
    f = DIRS[narc] / f'sweep_D_m{m}_fs{FS_TAG}_N{N}_peragent_d{delta:.2f}_g{GAMMA:.2f}.npz'
    d = np.load(f)
    return d['hss_vals'], d['hoo_vals'], d['beta_overall']      # beta_overall is [i_hoo, i_hss]


def customize_axes(ax):
    for sp in ax.spines.values():
        sp.set_color(MYBLACK)
    ax.tick_params(colors=MYBLACK)


def grid_figure(data, rows, row_title, title, out_path):
    """rows: values of the row parameter; row_title(rv) -> row title text; columns = delta."""
    S.apply_rc()
    vmax = max(np.abs(v[2]).max() for v in data.values())
    fig, axes, cax = S.heatmap_grid(len(rows), len(DELTAS), suptitle=True, row_titles=True, sup_lines=2)
    im_last = None
    for r, rv in enumerate(rows):
        S.row_title(fig, axes[r], row_title(rv))
        for c, delta in enumerate(DELTAS):
            ax = axes[r, c]
            hss, hoo, mis = data[(rv, delta)]
            im_last = ax.imshow(mis, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, aspect='equal')
            S.style_heatmap(ax, r'$h_{ss}$', r'$h_{oo}$' if c == 0 else None)
            S.panel_title(ax, 'ABCDEFGHI'[r * len(DELTAS) + c], rf'Prior $\delta={delta}$')
            X, Y = np.meshgrid(hss, hoo)
            cs = ax.contour(X, Y, mis, levels=CONTOUR_LEVELS, colors=[MYBLACK], linewidths=0.8)
            ax.clabel(cs, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[MYBLACK])
            cz = ax.contour(X, Y, mis, levels=[0], colors=[MYBLACK], linewidths=0.8, linestyles='dotted')
            ax.clabel(cz, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[MYBLACK])
    S.colorbar(fig, im_last, cax, r'Overall misperception $\beta_{\mathrm{overall}}$')
    S.suptitle(fig, title, axes)
    S.save(fig, out_path)


def main(narc):
    suffix = '' if narc else '_exclude_self'
    Ns, m_fig1 = [100, 500], 5
    grid_figure({(N, dl): load(narc, N, m_fig1, dl) for N in Ns for dl in DELTAS}, Ns,
                lambda N: rf'Network size $N={N}$',
                'Overall misperception with Bayesian rescaling\non preferential attachment networks of varying size',
                OUT_DIR / f'supp_bayesian_N_vs_delta{suffix}.pdf')
    ms, N_fig2 = [2, 5, 10], 500
    grid_figure({(m, dl): load(narc, N_fig2, m, dl) for m in ms for dl in DELTAS}, ms,
                lambda m: rf'Density parameter $m={m}$',
                'Overall misperception with Bayesian rescaling\non preferential attachment networks of varying density',
                OUT_DIR / f'supp_bayesian_m_vs_delta{suffix}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-self', action='store_true')
    main(narc=not ap.parse_args().exclude_self)
