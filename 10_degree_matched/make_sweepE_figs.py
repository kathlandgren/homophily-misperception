"""make_sweepE_figs.py — SI figures for the degree-matched comparison.

Writes to figures/:
    fig_sweep_E_pa_vs_sbm.pdf    preferential attachment vs degree-matched stochastic block networks
    fig_sweep_E_grid_diff.pdf    difference in overall misperception between the two models

Data required: 10_degree_matched/output/ (submit_sweep_E.sh).

Usage:
    python 10_degree_matched/make_sweepE_figs.py
"""

import importlib.util
import re
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

DATA_DIR = here() / '10_degree_matched' / 'output'
FIG_DIR  = here() / 'figures'
CMAP = 'RdBu_r'
PAT = re.compile(r'sweep_E_m(\d+)_fs([\d.]+)_N(\d+)\.npz')

METRICS = ['beta_overall', 'beta_s', 'beta_o']
METRIC_TITLES = [r"Overall misperception $\beta_{\mathrm{overall}}$",
                 r"Supporters' misperception $\beta_{s}$",
                 r"Opponents' misperception $\beta_{o}$"]
T_PA = (0, 0.25, 0.5, 0.75, 1.0)
T_SBM = (0, 0.3, 0.6, 0.9)


def edges(v):
    step = v[1] - v[0]
    return np.concatenate([[v[0] - step / 2], v + step / 2])


def load():
    rec = {}
    for f in sorted(DATA_DIR.glob('sweep_E_*.npz')):
        mo = PAT.match(f.name)
        if mo:
            rec[(int(mo[1]), round(float(mo[2]), 4), int(mo[3]))] = dict(np.load(f))
    return rec


def pa_vs_sbm(rec, m=5, fs=0.6667, N=1000):
    d = rec[(m, fs, N)]
    hs, ho = d['hss_vals'], d['hoo_vals']
    vmax = max(np.abs(d[f'pa_{k}']).max() for k in METRICS)
    vmax = max(vmax, max(np.abs(d[f'sbm_{k}'][:-1, :-1]).max() for k in METRICS))
    fig, axes, cax = S.heatmap_grid(2, 3, suptitle=True, row_titles=True)
    rows = [('pa', 'Preferential attachment networks'),
            ('sbm', 'Degree-matched stochastic block networks')]
    im = None
    for r, (mdl, lab) in enumerate(rows):
        S.row_title(fig, axes[r], lab)
        for c, (k, t) in enumerate(zip(METRICS, METRIC_TITLES)):
            ax = axes[r, c]
            arr = d[f'{mdl}_{k}']
            if mdl == 'sbm':        # h = 1 is degenerate under degree matching
                arr = arr[:-1, :-1]
                im = ax.pcolormesh(edges(hs[:-1]), edges(ho[:-1]), arr, cmap=CMAP,
                                   vmin=-vmax, vmax=vmax, rasterized=True)
                S.style_heatmap(ax, r'$h_{ss}$', r'$h_{oo}$' if c == 0 else None, ticks=T_SBM, lim=(0, 0.9))
            else:
                im = ax.pcolormesh(edges(hs), edges(ho), arr, cmap=CMAP, vmin=-vmax, vmax=vmax,
                                   rasterized=True)
                S.style_heatmap(ax, r'$h_{ss}$', r'$h_{oo}$' if c == 0 else None, ticks=T_PA)
            ax.set_aspect('equal')
            S.panel_title(ax, 'ABCDEF'[r * 3 + c], t)
    S.colorbar(fig, im, cax, r'Misperception $\beta$')
    S.suptitle(fig, 'Misperception in degree-matched preferential attachment and stochastic block networks',
               axes)
    S.save(fig, FIG_DIR / 'fig_sweep_E_pa_vs_sbm.pdf')


def grid_diff(rec, N=1000):
    ms = sorted({k[0] for k in rec})
    fss = sorted({k[1] for k in rec})
    diffs = {(m, fs): rec[(m, fs, N)]['pa_beta_overall'][:-1, :-1] - rec[(m, fs, N)]['sbm_beta_overall'][:-1, :-1]
             for m in ms for fs in fss}
    vmax = max(np.abs(v).max() for v in diffs.values())
    fs_txt = {0.5: '0.50', 0.6667: '2/3', 0.8: '0.80'}
    fig, axes, cax = S.heatmap_grid(len(ms), len(fss), suptitle=True, row_titles=True, sup_lines=2)
    im = None
    for r, m in enumerate(ms):
        S.row_title(fig, axes[r], rf'Density parameter $m = {m}$')
        for c, fs in enumerate(fss):
            ax = axes[r, c]
            d = rec[(m, fs, N)]
            im = ax.pcolormesh(edges(d['hss_vals'][:-1]), edges(d['hoo_vals'][:-1]), diffs[(m, fs)],
                               cmap=CMAP, vmin=-vmax, vmax=vmax, rasterized=True)
            ax.set_aspect('equal')
            S.style_heatmap(ax, r'$h_{ss}$', r'$h_{oo}$' if c == 0 else None, ticks=T_SBM, lim=(0, 0.9))
            S.panel_title(ax, 'ABCDEFGHI'[r * len(fss) + c], rf'Supporter fraction $f_s = {fs_txt[fs]}$')
    S.colorbar(fig, im, cax, r'Difference in overall misperception '
               r'$\beta_{\mathrm{overall}}^{\mathrm{PA}} - \beta_{\mathrm{overall}}^{\mathrm{SBM}}$')
    S.suptitle(fig, 'Difference in overall misperception between preferential attachment\n'
                    'and degree-matched stochastic block networks', axes)
    S.save(fig, FIG_DIR / 'fig_sweep_E_grid_diff.pdf')


if __name__ == '__main__':
    S.apply_rc()
    R = load()
    pa_vs_sbm(R)
    grid_diff(R)
