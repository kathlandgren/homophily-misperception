"""make_rescaling_order_figs.py — SI figures on the order of Bayesian rescaling, both network models.

Compares no rescaling, per-agent rescaling and group-mean rescaling. Writes to figures/:
    fig_sweep_C_rescaling_comparison.pdf, fig_sweep_C_rescaling_diff.pdf   stochastic block, H_out = 0.3
    fig_sweep_D_rescaling_comparison.pdf, fig_sweep_D_rescaling_diff.pdf   preferential attachment, m = 2
    both: f_s = 2/3, N = 1000, delta = 0.5, gamma = 0.5

Data required: 07_finite_n_sbm/output/ (submit_sweep_C.sh) and 08_finite_n_pa/output/ (submit_sweep_D.sh).

Usage:
    python 08_finite_n_pa/make_rescaling_order_figs.py
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
CMAP = 'RdBu_r'
DELTA, GAMMA = 0.50, 0.50

METRICS = ['beta_overall', 'beta_s', 'beta_o']
METRIC_TITLES = [r"Overall misperception $\beta_{\mathrm{overall}}$",
                 r"Supporters' misperception $\beta_{s}$",
                 r"Opponents' misperception $\beta_{o}$"]
VARIANTS = ['none', f'peragent_d{DELTA:.2f}_g{GAMMA:.2f}', f'groupmean_d{DELTA:.2f}_g{GAMMA:.2f}']
VARIANT_TITLES = ['No rescaling', 'Per-agent rescaling', 'Group-mean rescaling']

CONFIG = {
    'C': dict(dir=here() / '07_finite_n_sbm' / 'output',
              file=lambda v: f'sweep_C_hout0.30_fs0.6667_N1000_{v}.npz',
              x='hins_vals', y='hino_vals', xl=r'$H_{\mathrm{in}}^s$', yl=r'$H_{\mathrm{in}}^o$',
              model='stochastic block networks'),
    'D': dict(dir=here() / '08_finite_n_pa' / 'output',
              file=lambda v: f'sweep_D_m2_fs0.6667_N1000_{v}.npz',
              x='hss_vals', y='hoo_vals', xl=r'$h_{ss}$', yl=r'$h_{oo}$',
              model='preferential attachment networks'),
}


def make(sweep):
    cfg = CONFIG[sweep]
    out_dir = cfg['dir']
    rec = {v: dict(np.load(out_dir / cfg['file'](v))) for v in VARIANTS}

    # comparison: rows = metric, cols = rescaling variant
    vmax = max(np.abs(rec[v][k]).max() for v in VARIANTS for k in METRICS)
    fig, axes, cax = S.heatmap_grid(3, 3, suptitle=True, row_titles=True, sup_lines=2)
    im = None
    for r, (k, rt) in enumerate(zip(METRICS, METRIC_TITLES)):
        S.row_title(fig, axes[r], rt)
        for c, (v, vt) in enumerate(zip(VARIANTS, VARIANT_TITLES)):
            ax = axes[r, c]
            d = rec[v]
            X, Y = np.meshgrid(d[cfg['x']], d[cfg['y']])
            im = ax.imshow(d[k], extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                           vmin=-vmax, vmax=vmax, aspect='equal')
            cs = ax.contour(X, Y, d[k], levels=[0], colors=[S.BLACK], linewidths=1.2, linestyles='dotted')
            ax.clabel(cs, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[S.BLACK])
            S.style_heatmap(ax, cfg['xl'], cfg['yl'] if c == 0 else None)
            S.panel_title(ax, 'ABCDEFGHI'[r * 3 + c], vt)
    S.colorbar(fig, im, cax, r'Misperception $\beta$')
    S.suptitle(fig, f'Effect of the order of Bayesian rescaling on misperception\nin {cfg["model"]}', axes)
    S.save(fig, here() / 'figures' / f'fig_sweep_{sweep}_rescaling_comparison.pdf')

    # difference: per-agent minus group-mean
    diffs = [rec[VARIANTS[1]][k] - rec[VARIANTS[2]][k] for k in METRICS]
    vmax = max(np.abs(x).max() for x in diffs)
    fig, axes, cax = S.heatmap_grid(1, 3, suptitle=True, sup_lines=2)
    for c, (diff, t) in enumerate(zip(diffs, METRIC_TITLES)):
        ax = axes[0, c]
        im = ax.imshow(diff, extent=[0, 1, 0, 1], origin='lower', cmap=CMAP, vmin=-vmax, vmax=vmax,
                       aspect='equal')
        S.style_heatmap(ax, cfg['xl'], cfg['yl'] if c == 0 else None)
        S.panel_title(ax, 'ABC'[c], t)
    S.colorbar(fig, im, cax, r'Difference in misperception $\beta$ (per-agent minus group-mean rescaling)')
    S.suptitle(fig, 'Difference in misperception between per-agent and group-mean Bayesian rescaling\n'
                    f'in {cfg["model"]}', axes)
    S.save(fig, here() / 'figures' / f'fig_sweep_{sweep}_rescaling_diff.pdf')


if __name__ == '__main__':
    S.apply_rc()
    make('C')
    make('D')
