"""generate_fig_sweep_G_diagrams.py — SI figures fig_sweep_G_diagram_pa.pdf and fig_sweep_G_diagram_sbm.pdf.

Matches the style of fig_sweep_F_diagram.pdf (06_swap_simulations/make_fig_sweep_F_diagram.py):
  - Bold black cell labels only (no grey beta^peer text per cell)
  - Callout box at bottom with arrows explaining the layout and symbols
  - All text in MYBLACK (no grey)
  - Model names spelled out (no abbreviations)

Data required: run 11_broadcast_media/run_sweep_G.py first (11_broadcast_media/output/).

Usage:
    python 11_broadcast_media/generate_fig_sweep_G_diagrams.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import re
from pyprojroot import here

DATA_DIR   = here() / '11_broadcast_media' / 'output'
FIG_DIR    = here() / 'figures'

BETA_EMPIRICAL_LO = 21.0
BETA_EMPIRICAL_HI = 30.0

COLOR_S  = '#19bdff'
COLOR_O  = '#f2d138'
MYBLACK  = '#222222'

M_VALS  = [2, 5, 10]
FS_VALS = [2/3, 0.8]
H_VALS  = [0.25, 0.5, 0.75]

TICK_SIZE  = 7
LABEL_SIZE = 8
TITLE_SIZE = 9   # matches fig_sweep_F_diagram
ANNOT_SIZE = 7

def fs_tag(fs):
    if abs(fs - 2/3)  < 1e-4: return '0667'
    if abs(fs - 0.8)  < 1e-4: return '0800'
    if abs(fs - 0.5)  < 1e-4: return '0500'
    return f'{fs:.4f}'.replace('.', '')

def h_tag(h): return str(int(round(h * 1000)))


def load_data():
    """Return data[m][fs][hss][hoo][model] = npz dict."""
    data = {}
    pat = re.compile(r'sweep_G_m(\d+)_fs(\w+)_hss(\d+)_hoo(\d+)_(\w+)_rescaled\.npz')
    for fpath in sorted(DATA_DIR.glob('sweep_G_*_rescaled.npz')):
        mo = pat.match(fpath.name)
        if not mo:
            continue
        m_val  = int(mo[1])
        fs_str = mo[2]
        hss    = int(mo[3]) / 1000
        hoo    = int(mo[4]) / 1000
        model  = mo[5]
        fs_val = next((v for v in [2/3, 0.8, 0.5] if fs_tag(v) == fs_str), None)
        if fs_val is None:
            continue
        d = data.setdefault(m_val, {}).setdefault(fs_val, {}).setdefault(hss, {}).setdefault(hoo, {})
        d[model] = dict(np.load(fpath, allow_pickle=True))
    return data


def lstar(bov_frac, fs, lo=BETA_EMPIRICAL_LO/100, hi=BETA_EMPIRICAL_HI/100):
    """Min symmetric attention share (lambda*) to enter [lo, hi] with mu=0 opposition broadcast."""
    if lo <= bov_frac <= hi:
        return '0', True
    if bov_frac > hi:
        return '×', False          # already above empirical range
    slope = fs - bov_frac              # (f_s - mu) with mu=0 → f_s; shift per unit lambda
    if slope <= 0:
        return '—', False          # can't help
    lam = (lo - bov_frac) / slope
    if lam < 0.005:
        return '0', True
    return f'{lam:.2f}', True


def make_diagram(model, data, out_path):
    fs_plot = [fs for fs in FS_VALS]
    n_rows, n_cols = len(fs_plot), len(M_VALS)
    PAD = 0.0625

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 5.5),
                             constrained_layout=False)
    plt.subplots_adjust(hspace=0.55, wspace=0.15, top=0.88, bottom=0.22,
                        left=0.08, right=0.97)

    for row_idx, fs in enumerate(fs_plot):
        for col_idx, m in enumerate(M_VALS):
            ax = axes[row_idx, col_idx]

            lim_lo = H_VALS[0]  - PAD
            lim_hi = H_VALS[-1] + PAD

            ax.add_patch(MplPolygon(
                [(lim_lo, lim_lo), (lim_lo, lim_hi), (lim_hi, lim_hi)],
                closed=True, facecolor=COLOR_O, alpha=0.10, edgecolor='none', zorder=0))
            ax.add_patch(MplPolygon(
                [(lim_lo, lim_lo), (lim_hi, lim_lo), (lim_hi, lim_hi)],
                closed=True, facecolor=COLOR_S, alpha=0.10, edgecolor='none', zorder=0))

            for hss in H_VALS:
                for hoo in H_VALS:
                    d   = data[m][fs][hss][hoo][model]
                    bov = fs * float(d['beta_s_peer']) + (1-fs) * float(d['beta_o_peer'])
                    lbl, _ = lstar(bov, fs)

                    # Single centred label — always black
                    ax.text(hss, hoo, lbl,
                            ha='center', va='center',
                            fontsize=ANNOT_SIZE, fontweight='bold',
                            color=MYBLACK, zorder=4)

            ax.set_xlim(lim_lo - 0.04, lim_hi + 0.04)
            ax.set_ylim(lim_lo - 0.04, lim_hi + 0.04)
            ax.set_xticks(H_VALS)
            ax.set_yticks(H_VALS)
            ax.set_aspect('equal', adjustable='box')

            if row_idx == n_rows - 1:
                ax.set_xticklabels([f'{v:.2g}' for v in H_VALS],
                                   rotation=45, ha='right', fontsize=TICK_SIZE)
                ax.set_xlabel(r'$h_{ss}$', fontsize=LABEL_SIZE, color=MYBLACK, labelpad=1)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_yticklabels([f'{v:.2g}' for v in H_VALS], fontsize=TICK_SIZE)
                ax.set_ylabel(r'$h_{oo}$', fontsize=LABEL_SIZE, color=MYBLACK)
            else:
                ax.set_yticklabels([])

            panel_idx = row_idx * n_cols + col_idx
            panel_letter = chr(ord('A') + panel_idx)
            fs_str = '2/3' if abs(fs - 2/3) < 0.001 else f'{fs}'
            ax.set_title(rf'{panel_letter}. $f_s={fs_str}$, $m={m}$',
                         fontsize=TITLE_SIZE, color=MYBLACK, pad=3, loc='center', fontweight='normal')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors=MYBLACK, pad=0, length=2)

    # ── Callout box ───────────────────────────────────────────────────────────
    cax = fig.add_axes([0.09, 0.005, 0.88, 0.14])
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis('off')

    bx, by, bw, bh = 0.03, 0.18, 0.055, 0.58
    cax.add_patch(MplPolygon(
        [(bx,by),(bx,by+bh),(bx+bw,by+bh)],
        closed=True, facecolor=COLOR_O, alpha=0.20, edgecolor='none', zorder=2))
    cax.add_patch(MplPolygon(
        [(bx,by),(bx+bw,by),(bx+bw,by+bh)],
        closed=True, facecolor=COLOR_S, alpha=0.20, edgecolor='none', zorder=2))
    cax.add_patch(mpatches.FancyBboxPatch(
        (bx,by), bw, bh, boxstyle='square,pad=0.0',
        facecolor='none', edgecolor=MYBLACK, linewidth=0.7, zorder=3))

    # Use the top-left cell value (h_ss=0.25, h_oo=0.75) as the example
    _d_ex = data[M_VALS[0]][fs_plot[0]][H_VALS[0]][H_VALS[-1]][model]
    _bov_ex = fs_plot[0] * float(_d_ex['beta_s_peer']) + (1 - fs_plot[0]) * float(_d_ex['beta_o_peer'])
    _ex_lbl, _ = lstar(_bov_ex, fs_plot[0])
    cax.text(bx+bw/2, by+bh*0.50, _ex_lbl,
             ha='center', va='center',
             fontsize=7.5, fontweight='bold', color=MYBLACK, zorder=4)
    cax.text(bx+bw/2, by+bh+0.08, r'$h_{oo} > h_{ss}$',
             ha='center', va='bottom', fontsize=4.5, color='#8a6a00', fontstyle='italic')
    cax.text(bx+bw/2, by-0.08, r'$h_{ss} > h_{oo}$',
             ha='center', va='top',    fontsize=4.5, color='#005580', fontstyle='italic')

    RX = bx + bw
    LABEL_X = RX + 0.07

    cax.text(LABEL_X, by+bh+0.02,
             r'Minimum symmetric attention share $\lambda^*$ to enter empirical misperception range',
             ha='left', va='top', fontsize=5.5, color=MYBLACK)
    cax.annotate('',
                 xy=(RX+0.005, by+bh*0.50), xytext=(LABEL_X, by+bh*0.50),
                 arrowprops=dict(arrowstyle='->', color=MYBLACK, lw=0.7))
    cax.text(LABEL_X + 0.005, by+bh*0.50,
             r'Minimum attention share $\lambda^*$',
             ha='left', va='center', fontsize=5.5, color=MYBLACK)

    SYM_X = 0.66   # moved right (was 0.56) so the longer 'attention share' label does not crowd the symbol key
    cax.text(SYM_X, by+bh*0.75,
             '—  never reaches empirical range',
             va='center', ha='left', fontsize=5.0, color=MYBLACK)
    cax.text(SYM_X, by+bh*0.50,
             '×  always above empirical range',
             va='center', ha='left', fontsize=5.0, color=MYBLACK)
    cax.text(SYM_X, by+bh*0.25,
             '0  already in empirical range',
             va='center', ha='left', fontsize=5.0, color=MYBLACK)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    model_name = ('preferential attachment networks' if model == 'pa'
                  else 'degree-matched stochastic block networks')
    fig.text(0.5, 0.955,
             r'Minimum broadcast attention share $\lambda^*$ to reach the empirical misperception range'
             f'\non {model_name}',
             ha='center', va='center', fontsize=TITLE_SIZE, color=MYBLACK, linespacing=1.3)

    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


data = load_data()
for model in ['pa', 'sbm']:
    out = FIG_DIR / f'fig_sweep_G_diagram_{model}.pdf'
    make_diagram(model, data, out)
