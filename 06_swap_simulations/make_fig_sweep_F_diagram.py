"""make_fig_sweep_F_diagram.py — SI figure fig_sweep_F_diagram.pdf.

For every (h_ss, h_oo) on the swap-simulation grid, the minimum percentage of swapped nodes at which
beta_overall enters the empirical range (21–30 percentage points), for all nodes and for
unswapped nodes only; rows f_s ∈ {2/3, 0.8}, columns m ∈ {2, 5, 10}.

npz arrays are indexed [i_hoo, i_hss, i_swap].

Data required: 06_swap_simulations/output/ (submit_sweep_F.sh).

Usage:
    python 06_swap_simulations/make_fig_sweep_F_diagram.py
"""

import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pyprojroot import here

DATA_DIR = here() / '06_swap_simulations' / 'output'
OUT_PATH = here() / 'figures' / 'fig_sweep_F_diagram.pdf'

TITLE_SIZE = 9
LABEL_SIZE = 8
TICK_SIZE  = 8
ANNOT_SIZE = 7

plt.rc('font',   size=LABEL_SIZE)
plt.rc('axes',   titlesize=LABEL_SIZE)
plt.rc('axes',   labelsize=LABEL_SIZE)
plt.rc('xtick',  labelsize=TICK_SIZE)
plt.rc('ytick',  labelsize=TICK_SIZE)
plt.rc('legend', fontsize=LABEL_SIZE)
plt.rc('figure', titlesize=TITLE_SIZE)

# ── Load all .npz files: sweep_F_m{m}_fs{fs}_N{N}.npz ─────────────────────────
_PAT = re.compile(r'sweep_F_m(\d+)_fs([\d.]+)_N(\d+)\.npz')

rec_by_fs = {}  # f_s -> m -> data dict  (N is always 1000)
for fpath in sorted(DATA_DIR.glob('sweep_F_*.npz')):
    mo = _PAT.match(fpath.name)
    if not mo: continue
    m_val, f_s = int(mo[1]), float(mo[2])
    rec_by_fs.setdefault(f_s, {})[m_val] = dict(np.load(fpath))
m_vals = sorted({m for d in rec_by_fs.values() for m in d})

BETA_EMPIRICAL_LO = 21.0
BETA_EMPIRICAL_HI = 30.0

def empirical_entry(curve_pp, swap_fracs_pct):
    """Return (label_str, found) — minimum swap% to enter empirical range, or symbol."""
    inside = (curve_pp >= BETA_EMPIRICAL_LO) & (curve_pp <= BETA_EMPIRICAL_HI)
    if inside.any():
        return f'{swap_fracs_pct[np.argmax(inside)]:.0f}%', True
    if curve_pp[-1] < BETA_EMPIRICAL_LO:
        return '—', False
    return '×', False

COLOR_SUPP  = '#19bdff'
COLOR_OPP   = '#f2d138'
COLOR_PEERS = '#626262'
MYBLACK     = '#222222'

m0       = sorted(rec_by_fs[sorted(rec_by_fs.keys())[0]].keys())[0]
hss_grid = rec_by_fs[sorted(rec_by_fs.keys())[0]][m0]['hss_vals']
hoo_grid = rec_by_fs[sorted(rec_by_fs.keys())[0]][m0]['hoo_vals']
fs_list  = sorted(rec_by_fs.keys())
n_rows, n_cols = len(fs_list), len(m_vals)
LETTERS  = 'ABCDEF'

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 6.2), constrained_layout=False)
plt.subplots_adjust(hspace=0.50, wspace=0.15, top=0.90, bottom=0.27, left=0.08, right=0.97)
PAD = 0.0625

for row_idx, f_s in enumerate(fs_list):
    fs_str = '2/3' if abs(f_s - 0.6667) < 0.001 else f'{f_s}'
    for col_idx, m_val in enumerate(m_vals):
        ax  = axes[row_idx, col_idx]
        d   = rec_by_fs[f_s][m_val]
        sf_pct = d['swap_fracs'] * 100.0
        ax.add_patch(MplPolygon(
            [(hss_grid[0]-PAD,hss_grid[0]-PAD),(hss_grid[0]-PAD,hoo_grid[-1]+PAD),(hss_grid[-1]+PAD,hoo_grid[-1]+PAD)],
            closed=True, facecolor=COLOR_OPP, alpha=0.10, edgecolor='none', zorder=0))
        ax.add_patch(MplPolygon(
            [(hss_grid[0]-PAD,hss_grid[0]-PAD),(hss_grid[-1]+PAD,hss_grid[0]-PAD),(hss_grid[-1]+PAD,hoo_grid[-1]+PAD)],
            closed=True, facecolor=COLOR_SUPP, alpha=0.10, edgecolor='none', zorder=0))
        for si, hss in enumerate(hss_grid):
            for oi, hoo in enumerate(hoo_grid):
                ca = d['all_beta_overall'][oi, si, :] * 100
                cp = d['peers_beta_overall'][oi, si, :] * 100
                la, _ = empirical_entry(ca, sf_pct)
                lp, _ = empirical_entry(cp, sf_pct)
                ax.text(hss, hoo + 0.025, la, ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color=MYBLACK, zorder=4)
                ax.text(hss, hoo - 0.025, f'({lp})', ha='center', va='center',
                        fontsize=5.0, color=COLOR_PEERS, zorder=4)
        ax.set_xlim(hss_grid[0] - PAD*2, hss_grid[-1] + PAD*2)
        ax.set_ylim(hoo_grid[0] - PAD*2, hoo_grid[-1] + PAD*2)
        ax.set_xticks(hss_grid); ax.set_yticks(hoo_grid)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticklabels([f'{v:.3g}' for v in hss_grid],
                           rotation=45, ha='right', fontsize=ANNOT_SIZE)
        if row_idx == n_rows - 1:
            ax.set_xlabel(r'$h_{ss}$', fontsize=LABEL_SIZE, color=MYBLACK, labelpad=1)
        if col_idx == 0:
            ax.set_ylabel(r'$h_{oo}$', fontsize=LABEL_SIZE, color=MYBLACK)
            ax.set_yticklabels([f'{v:.3g}' for v in hoo_grid], fontsize=ANNOT_SIZE)
        else:
            ax.set_yticklabels([])
        letter = LETTERS[row_idx * n_cols + col_idx]
        ax.set_title(f'{letter}. $f_s={fs_str}$, $m={m_val}$',
                     fontsize=TITLE_SIZE, color=MYBLACK, fontweight='normal', pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MYBLACK, pad=0, length=2)

# ─── Callout ──────────────────────────────────────────────────────────────────
cax = fig.add_axes([0.09, 0.01, 0.88, 0.16])
cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis('off')

bx, by, bw, bh = 0.03, 0.26, 0.055, 0.46
cax.add_patch(MplPolygon([(bx,by),(bx,by+bh),(bx+bw,by+bh)],
    closed=True, facecolor=COLOR_OPP, alpha=0.20, edgecolor='none', zorder=2))
cax.add_patch(MplPolygon([(bx,by),(bx+bw,by),(bx+bw,by+bh)],
    closed=True, facecolor=COLOR_SUPP, alpha=0.20, edgecolor='none', zorder=2))
cax.add_patch(mpatches.FancyBboxPatch((bx,by), bw, bh, boxstyle='square,pad=0.0',
    facecolor='none', edgecolor=MYBLACK, linewidth=0.7, zorder=3))

cax.text(bx+bw/2, by+bh+0.06, r'$h_{oo} > h_{ss}$',
    ha='center', va='bottom', fontsize=4.5, color='#8a6a00', fontstyle='italic')
cax.text(bx+bw/2, by-0.06, r'$h_{ss} > h_{oo}$',
    ha='center', va='top', fontsize=4.5, color='#005580', fontstyle='italic')

cax.text(bx+bw/2, by+bh*0.70, '7%',
    ha='center', va='center', fontsize=7.5, fontweight='bold', color=MYBLACK, zorder=4)
cax.text(bx+bw/2, by+bh*0.30, '(12%)',
    ha='center', va='center', fontsize=5.5, color=COLOR_PEERS, zorder=4)

RX = bx + bw
LABEL_X = RX + 0.07

cax.text(LABEL_X, by+bh,
    'Minimum % minority nodes swapped to enter empirical misperception range for:',
    ha='left', va='top', fontsize=5.5, color=MYBLACK)

cax.annotate('All nodes',
    xy=(RX+0.005, by+bh*0.70), xytext=(LABEL_X, by+bh*0.70),
    fontsize=5.5, color=MYBLACK, va='center', ha='left',
    arrowprops=dict(arrowstyle='->', color=MYBLACK, lw=0.7))
cax.annotate('Unswapped nodes only',
    xy=(RX+0.005, by+bh*0.30), xytext=(LABEL_X, by+bh*0.30),
    fontsize=5.5, color=COLOR_PEERS, va='center', ha='left',
    arrowprops=dict(arrowstyle='->', color=COLOR_PEERS, lw=0.7))

SYM_X = 0.55
cax.text(SYM_X, by+bh*0.70, '—  never reaches empirical range',
    va='center', ha='left', fontsize=5.0, color='#555555')
cax.text(SYM_X, by+bh*0.30, '×  always above empirical range',
    va='center', ha='left', fontsize=5.0, color='#555555')

fig.text(0.5, 0.955,
    'Minimum swap percentage to reach the empirical misperception range\non preferential attachment networks',
    ha='center', va='center', fontsize=TITLE_SIZE, color=MYBLACK, linespacing=1.3)

plt.savefig(OUT_PATH, bbox_inches='tight')
print(f'Saved {OUT_PATH}')