"""generate_fig_asymmetric_swaps_full.py — SI figure figure_asymmetric_swaps_full.pdf.

Misperception as a function of the fraction of swapped nodes, from the swap simulations.
Solid lines = all nodes; dashed = unswapped nodes only.
Layout: 3 x 3 grid of equal panels. Row 1: symmetric (0.25, 0.5, 0.75; panel C is the
main-text condition). Row 2: h_oo > h_ss. Row 3: h_ss > h_oo. The empty third slots in rows
2-3 hold the legends.
Light shading: beta_overall in the empirical band. Dark shading: joint window (in band AND
beta_o > beta_s), computed with quantify_ordering_window.analyse so it matches the SI table.
Panel titles 'A. h_ss=..., h_oo=...' (regular weight, centered, as in src/plot_style.py).
Uses m=5, f_s=2/3, N=1000.

Data required: 06_swap_simulations/output/ (submit_sweep_F.sh).

Usage:
    python 06_swap_simulations/generate_fig_asymmetric_swaps_full.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
import importlib.util
import os
import sys

from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = str(here() / '06_swap_simulations' / 'output')
OUT_PATH   = str(here() / 'figures' / 'figure_asymmetric_swaps_full.pdf')
analyse = _load("quantify_ordering_window", "06_swap_simulations/quantify_ordering_window.py").analyse

# ── Parameters ────────────────────────────────────────────────────────────
M_VAL            = 5
FS_VAL           = 0.6667
FS_STR           = '0.6667'
N                = 1000
BETA_EMPIRICAL_LO = 21.0
BETA_EMPIRICAL_HI = 30.0

# ── Style (matches original submission) ───────────────────────────────────
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
matplotlib.rcParams['font.family'] = 'Arial'
SMALL_SIZE  = 9
MEDIUM_SIZE = 9
plt.rc('font',   size=SMALL_SIZE)
plt.rc('axes',   titlesize=SMALL_SIZE)
plt.rc('axes',   labelsize=MEDIUM_SIZE)
plt.rc('xtick',  labelsize=SMALL_SIZE)
plt.rc('ytick',  labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)

COLOR_S   = '#19bdff'   # supporters (majority)
COLOR_O   = '#f2d138'   # opponents (minority)
COLOR_ALL = '#626262'   # overall
MYBLACK   = '#222222'

def style_ax(ax):
    ax.spines['bottom'].set_color(MYBLACK)
    ax.spines['left'].set_color(MYBLACK)
    ax.tick_params(axis='x', colors=MYBLACK)
    ax.tick_params(axis='y', colors=MYBLACK)
    sns.despine(ax=ax)

# ── Layout (same as original) ─────────────────────────────────────────────
LAYOUT = [  # (row, col, h_ss, h_oo)
    (0, 0, 0.25, 0.25), (0, 1, 0.50, 0.50), (0, 2, 0.75, 0.75),   # symmetric
    (1, 0, 0.25, 0.75), (1, 1, 0.50, 0.75),                       # h_oo > h_ss
    (2, 0, 0.75, 0.25), (2, 1, 0.75, 0.50),                       # h_ss > h_oo
]
BAND_COLOR = '#a83232'
ROW_LABELS = [
    'Symmetric homophily',
    'Higher opponent homophily ($h_{oo} > h_{ss}$)',
    'Higher supporter homophily ($h_{ss} > h_{oo}$)',
]
PANEL_LABELS = list('ABCDEFG')

# ── Load data ─────────────────────────────────────────────────────────────
npz_path = os.path.join(DATA_DIR, f'sweep_F_m{M_VAL}_fs{FS_STR}_N{N}.npz')
d = np.load(npz_path, allow_pickle=True)
hss_vals    = d['hss_vals']
hoo_vals    = d['hoo_vals']
swap_fracs  = d['swap_fracs']        # fraction 0–1
swap_pct    = swap_fracs * 100.0     # percent

def get_idx(arr, val):
    return int(np.where(np.isclose(arr, val))[0][0])

def get_series(hss, hoo):
    ih = get_idx(hss_vals, hss)
    io = get_idx(hoo_vals, hoo)
    # npz axes are [i_hoo, i_hss, :] (outer loop over hoo, inner over hss in run_sweep_F)
    return {
        'bs_all':   d['all_beta_s'][io, ih, :]       * 100,
        'bo_all':   d['all_beta_o'][io, ih, :]       * 100,
        'bov_all':  d['all_beta_overall'][io, ih, :] * 100,
        'bs_un':    d['peers_beta_s'][io, ih, :]         * 100,
        'bo_un':    d['peers_beta_o'][io, ih, :]         * 100,
        'bov_un':   d['peers_beta_overall'][io, ih, :]   * 100,
    }

# ── Plot ──────────────────────────────────────────────────────────────────
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(5.0, 5.6), dpi=300)

LW = 1.3

for idx, (row, col, hss, hoo) in enumerate(LAYOUT):
    ax  = axes[row, col]
    s   = get_series(hss, hoo)

    # Clip x to ≤33% (matches original xlim)
    mask = swap_pct <= 33.0
    x = swap_pct[mask]

    # Solid = all nodes
    ax.plot(x, s['bov_all'][mask], color=COLOR_ALL, lw=LW, ls='-')
    ax.plot(x, s['bs_all'][mask],  color=COLOR_S,   lw=LW, ls='-')
    ax.plot(x, s['bo_all'][mask],  color=COLOR_O,   lw=LW, ls='-')

    # Dashed = unswapped nodes only
    ax.plot(x, s['bov_un'][mask], color=COLOR_ALL, lw=LW, ls='--')
    ax.plot(x, s['bs_un'][mask],  color=COLOR_S,   lw=LW, ls='--')
    ax.plot(x, s['bo_un'][mask],  color=COLOR_O,   lw=LW, ls='--')

    # Light shading: x-range where beta_overall (all nodes) is in the empirical band
    bov = s['bov_all'][mask]
    inside = (bov >= BETA_EMPIRICAL_LO) & (bov <= BETA_EMPIRICAL_HI)
    if inside.any():
        x_entry = x[np.argmax(inside)]
        x_exit  = x[len(inside) - 1 - np.argmax(inside[::-1])]
        ax.axvspan(x_entry, x_exit, color=BAND_COLOR, alpha=0.12, lw=0, zorder=0)

    # Dark shading: joint window (in band AND beta_o > beta_s), same rule as the SI table
    r = analyse(x, s['bs_all'][mask] / 100, s['bo_all'][mask] / 100, bov / 100)
    if r['window'] is not None:
        w0, w1 = r['window']
        ax.axvspan(w0, max(w1, w0 + 0.4), color=BAND_COLOR, alpha=0.30, lw=0, zorder=0)

    ax.axhline(0, color=MYBLACK, lw=0.6, ls=':', zorder=1)

    ax.set_xlim(0, 33)
    ax.set_ylim(-0.33 * 100, 0.66 * 100)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel('% nodes swapped', size=SMALL_SIZE)
    if col == 0:
        ax.set_ylabel(r'misperception $\beta$ (pp)', size=SMALL_SIZE)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(f'{PANEL_LABELS[idx]}. $h_{{ss}}$={hss}, $h_{{oo}}$={hoo}', size=SMALL_SIZE - 1,
                 fontweight='normal', loc='center')
    style_ax(ax)

# ── Legends in the empty slots ───────────────────────────────────────────
from matplotlib.patches import Patch
for ax in (axes[1, 2], axes[2, 2]):
    ax.axis('off')
line_handles = [
    Line2D([0], [0], color=COLOR_ALL, lw=LW, ls='-',  label=r'$\beta_{\mathrm{overall}}$'),
    Line2D([0], [0], color=COLOR_S,   lw=LW, ls='-',  label=r'$\beta_s$ (supporters)'),
    Line2D([0], [0], color=COLOR_O,   lw=LW, ls='-',  label=r'$\beta_o$ (opponents)'),
    Line2D([0], [0], color=MYBLACK,   lw=LW, ls='-',  label='all nodes'),
    Line2D([0], [0], color=MYBLACK,   lw=LW, ls='--', label='unswapped nodes'),
]
shade_handles = [
    Patch(facecolor=BAND_COLOR, alpha=0.12, label=r'$\beta_{\mathrm{overall}}$ in empirical range'),
    Patch(facecolor=BAND_COLOR, alpha=0.42, label='in range and ' + r'$\beta_o > \beta_s$'),
]
axes[1, 2].legend(handles=line_handles, loc='center left', frameon=False,
                  fontsize=SMALL_SIZE - 1, labelspacing=0.35, borderaxespad=0)
axes[2, 2].legend(handles=shade_handles, loc='center left', frameon=False,
                  fontsize=SMALL_SIZE - 1, labelspacing=0.5, borderaxespad=0)

# ── Row labels ────────────────────────────────────────────────────────────
ROW_SPAN = [3, 2, 2]  # panels per row
plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
plt.subplots_adjust(hspace=1.0, wspace=0.12)
fig.canvas.draw()
for row, label in enumerate(ROW_LABELS):
    pos0 = axes[row, 0].get_position()
    pos1 = axes[row, ROW_SPAN[row] - 1].get_position()
    x_mid = (pos0.x0 + pos1.x1) / 2
    y_top = pos0.y1 + 0.045
    fig.text(x_mid, y_top, label, ha='center', va='bottom',
             fontsize=SMALL_SIZE, transform=fig.transFigure)

plt.savefig(OUT_PATH, format='pdf', bbox_inches='tight')
print(f'Saved {OUT_PATH}')
