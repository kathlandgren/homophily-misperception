"""
make_fig5A.py — main-text Fig 5A (media bias and misperception) from the swap simulations.

Swap simulations, h_ss = h_oo = 0.75, m = 10, f_s = 2/3, N = 1000, 200 sims, beta over all nodes
(all_beta_*), drawn as lines. m = 10 matches the example network of Figs 4B and 5B.
The shaded band marks the swap percentages where beta_overall is in the empirical range.

Data required: 06_swap_simulations/output/sweep_F_m10_fs0.6667_N1000.npz (submit_sweep_F.sh).

Output: figures/single_panel_high_homophily_without_legend_narcTrue.pdf
        (assembled with panel B by figures/combine_swap_fig_narcTrue.tex)

Usage:
    python 06_swap_simulations/make_fig5A.py
    python 06_swap_simulations/make_fig5A.py --exclude-self   # same panel from output_exclude_self/
                                                              # (check; ..._without_legend_exclude_self.pdf)
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pyprojroot import here

# Fixed settings
H = 0.75
M = 10
BETA_LO, BETA_HI = 0.21, 0.30          # empirical band (as in the published panel)
X_MAX = 33.0                           # % of nodes swapped (common range)

DATA = {True:  lambda m: here() / '06_swap_simulations' / 'output' / f'sweep_F_m{m}_fs0.6667_N1000.npz',
        False: lambda m: here() / '06_swap_simulations' / 'output_exclude_self' / f'sweep_F_m{m}_fs0.6667_N1000.npz'}
FIG_DIR = here() / 'figures'

# ── style (from the first-submission generate_swap_figure.ipynb) ─────────────
COLORS, GREY = ['#19bdff', '#f2d138'], '#626262'
plt.rc('font', size=10)
plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)


# ── loading ───────────────────────────────────────────────────────────────────
def load_sweepF(path):
    d = np.load(path)
    hl = [round(float(v), 4) for v in d['hss_vals']]
    si = oi = hl.index(H)                                  # arrays are [i_hoo, i_hss, swap]
    return dict(x=d['swap_fracs'] * 100,
                overall=d['all_beta_overall'][oi, si], s=d['all_beta_s'][oi, si], o=d['all_beta_o'][oi, si])


# ── metrics ───────────────────────────────────────────────────────────────────
XG = np.round(np.arange(0, X_MAX + 1e-9, 0.1), 4)      # common 0.1 %-point grid


def on_grid(c, key):
    return np.interp(XG, c['x'], c[key])


def band_span(c):
    """(entry, exit) swap % where beta_overall is inside [LO, HI], on the common grid."""
    y = on_grid(c, 'overall')
    inside = (y >= BETA_LO) & (y <= BETA_HI)
    if not inside.any():
        return np.nan, np.nan
    return XG[np.argmax(inside)], XG[len(inside) - 1 - np.argmax(inside[::-1])]


# ── panel ─────────────────────────────────────────────────────────────────────
def draw_panel(c, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(3.42, 2.5), dpi=300)
    for key, col in (('overall', 'grey'), ('o', COLORS[1]), ('s', COLORS[0])):
        ax.plot(c['x'], c[key], color=col, lw=2.6, solid_capstyle='round')
    lo, hi = band_span(c)
    if not np.isnan(lo):
        ax.axvspan(lo, hi, color='#a83232', alpha=0.15, edgecolor=None)
        ax.text(lo + 0.2, -0.43, 'empirical range', rotation=90, va='bottom', fontsize=8, color='#a83232')
    ax.set_xlabel('Percent of nodes swapped')
    ax.set_ylabel(r'misperception $\beta$')
    ax.set_ylim(-0.45, 0.66)   # lowered from -0.33 so the 'empirical range' label fits
    ax.set_xticks([0, 10, 20, 30])
    ax.set_title('A. Media bias and misperception')
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_color('black')
    sns.despine(ax=ax)
    handles = [Line2D([0], [0], color=col, lw=2.6, label=lab)
               for lab, col in (('overall', GREY), ('support', COLORS[0]), ('oppose', COLORS[1]))]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-self', action='store_true')
    narc = not ap.parse_args().exclude_self
    c = load_sweepF(DATA[narc](M))
    lo, hi = band_span(c)
    print(f'beta_overall in [{BETA_LO}, {BETA_HI}] for {lo:.1f}%–{hi:.1f}% of nodes swapped')
    name = 'narcTrue' if narc else 'exclude_self'
    draw_panel(c, FIG_DIR / f'single_panel_high_homophily_without_legend_{name}.pdf')
