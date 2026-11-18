"""make_sweepAB_figs.py — SI figures for the mean-field sensitivity analyses.

Writes to figures/:
    fig_sweep_A_hout_selected.pdf   overall misperception on stochastic block networks, H_out ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
    fig_sweep_A_fs_variation.pdf    stochastic block networks, H_out = 0.3, f_s ∈ {0.5, 2/3, 0.8}
    fig_sweep_B_fs_variation.pdf    preferential attachment networks, f_s ∈ {0.5, 2/3, 0.8}

Data required: run_sweep_A.py and run_sweep_B.py first (09_mean_field_sensitivity/output/).

Usage:
    python 09_mean_field_sensitivity/make_sweepAB_figs.py
"""

import importlib.util
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

S = _load("plot_style", "src/plot_style.py")

DATA_DIR = here() / '09_mean_field_sensitivity' / 'output'
FIG_DIR  = here() / 'figures'
CMAP = 'RdBu_r'
BAND = (0.21, 0.30)

COL_KEYS = ['beta_overall', 'beta_s', 'beta_o']
COL_TITLES = [r"Overall misperception $\beta_{\mathrm{overall}}$",
              r"Supporters' misperception $\beta_{s}$",
              r"Opponents' misperception $\beta_{o}$"]
FS_LABEL = {0.5: '0.50', 2 / 3: '2/3', 0.8: '0.80'}


def fs_label(fs):
    k = min(FS_LABEL, key=lambda v: abs(v - fs))
    return rf'Supporter fraction $f_s = {FS_LABEL[k]}$'


def load_A():
    data = {}
    for f in sorted(DATA_DIR.glob('sweep_A_*.npz')):
        d = np.load(f)
        data[(round(float(d['H_out']), 2), round(float(d['f_s']), 4))] = d
    return data


def load_B():
    data = {}
    for f in sorted(DATA_DIR.glob('sweep_B_*.npz')):
        d = np.load(f)
        data[round(float(d['f_s']), 4)] = d
    return data


def zero_contour(ax, X, Y, arr):
    cs = ax.contour(X, Y, arr, levels=[0], colors=[S.BLACK], linewidths=1.2, linestyles='dotted')
    ax.clabel(cs, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[S.BLACK])


def fs_grid(get, fs_values, xkey, ykey, xlabel, ylabel, title, out):
    """3x3: rows = f_s, cols = overall / supporters / opponents (S7, S8)."""
    vmax = max(float(np.nanmax(np.abs(get(fs)[k]))) for fs in fs_values for k in COL_KEYS)
    fig, axes, cax = S.heatmap_grid(3, 3, suptitle=True, row_titles=True)
    im = None
    for r, fs in enumerate(fs_values):
        d = get(fs)
        X, Y = np.meshgrid(d[xkey], d[ykey])
        S.row_title(fig, axes[r], fs_label(fs))
        for c, (key, t) in enumerate(zip(COL_KEYS, COL_TITLES)):
            ax = axes[r, c]
            im = ax.imshow(d[key], extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                           vmin=-vmax, vmax=vmax, aspect='equal')
            zero_contour(ax, X, Y, d[key])
            S.style_heatmap(ax, xlabel=xlabel, ylabel=ylabel if c == 0 else None)
            S.panel_title(ax, 'ABCDEFGHI'[r * 3 + c], t)
    S.colorbar(fig, im, cax, r'Misperception $\beta$')
    S.suptitle(fig, title, axes)
    S.save(fig, out)


def hout_row(data, out):
    """S6: beta_overall for five H_out values at f_s = 2/3."""
    fs, hs = 0.6667, [0.1, 0.3, 0.5, 0.7, 1.0]
    vmax = max(float(np.nanmax(np.abs(data[(h, fs)]['beta_overall']))) for h in hs)
    # five panels at full Fig 3 size would exceed \linewidth; shrink panels, keep font scale
    fig, axes, cax = S.heatmap_grid(1, 5, suptitle=True, panel=1.29)
    im = None
    for c, h in enumerate(hs):
        d = data[(h, fs)]
        ax = axes[0, c]
        X, Y = np.meshgrid(d['hins_vals'], d['hino_vals'])
        im = ax.imshow(d['beta_overall'], extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                       vmin=-vmax, vmax=vmax, aspect='equal')
        zero_contour(ax, X, Y, d['beta_overall'])
        cs = ax.contour(X, Y, d['beta_overall'], levels=list(BAND), colors=[S.BLACK], linewidths=1.2)
        ax.clabel(cs, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[S.BLACK])
        S.style_heatmap(ax, xlabel=r'$H_{\mathrm{in}}^s$', ylabel=r'$H_{\mathrm{in}}^o$' if c == 0 else None)
        S.panel_title(ax, 'ABCDE'[c], rf'$H_{{\mathrm{{out}}}} = {h:.1f}$')
    S.colorbar(fig, im, cax, r'Overall misperception $\beta_{\mathrm{overall}}$')
    S.suptitle(fig, 'Overall misperception on stochastic block networks '
                    'across cross-group connection probabilities', axes)
    S.save(fig, out, panel_in=1.29)


if __name__ == '__main__':
    S.apply_rc()
    A = load_A()
    hout_row(A, FIG_DIR / 'fig_sweep_A_hout_selected.pdf')
    fsA = sorted({k[1] for k in A})
    fs_grid(lambda fs: A[(0.3, fs)], fsA, 'hins_vals', 'hino_vals',
            r'$H_{\mathrm{in}}^s$', r'$H_{\mathrm{in}}^o$',
            r'Misperception on stochastic block networks with homophily for a fixed $H_{\mathrm{out}}=0.3$',
            FIG_DIR / 'fig_sweep_A_fs_variation.pdf')
    B = load_B()
    fs_grid(lambda fs: B[fs], sorted(B), 'hss_vals', 'hoo_vals', r'$h_{ss}$', r'$h_{oo}$',
            'Misperception on preferential attachment networks with homophily',
            FIG_DIR / 'fig_sweep_B_fs_variation.pdf')
