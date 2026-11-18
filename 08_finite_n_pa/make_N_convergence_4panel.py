"""
make_N_convergence_4panel.py — SI finite-N convergence figures for both network models.

Writes to figures/:
    fig_sweep_C_N_convergence_4panel.pdf   stochastic block networks (07_finite_n_sbm, H_out = 0.3)
    fig_sweep_D_N_convergence_4panel.pdf   preferential attachment networks (08_finite_n_pa, m = 2)

Each shows beta_overall at N = 100 / 500 / 1000 from the simulations (agents include their
own opinion), plus a fourth "Mean field" panel. The mean-field panel is the neighbours-only
analytical formula used for all mean-field figures in the manuscript:
  - stochastic block model: Theorem 1 (as in 09_mean_field_sensitivity/run_sweep_A.py)
  - preferential attachment: 03_analytical/src/functions.py (compute_C, compute_*_perception)

Data required: 07_finite_n_sbm/output/ (submit_sweep_C.sh) and 08_finite_n_pa/output/
(submit_sweep_D.sh). Any N whose output file is missing is skipped.

Usage:
    python 08_finite_n_pa/make_N_convergence_4panel.py            # both
    python 08_finite_n_pa/make_N_convergence_4panel.py --model pa  # one
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

fun = _load("functions", "03_analytical/src/functions.py")
S   = _load("plot_style", "src/plot_style.py")

CMAP, MYBLACK = 'RdBu_r', '#222222'
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, ANNOT_SIZE = 9, 8, 8, 7
plt.rc('font', size=LABEL_SIZE)
plt.rc('axes', titlesize=LABEL_SIZE, labelsize=LABEL_SIZE)
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)

N_VALS = [100, 500, 1000]
MF_POINTS = 201


# ── mean-field (neighbours-only) ───────────────────────────────────────────────
def mf_sbm(h_out, f_s, n=MF_POINTS, eps=1e-10):
    """Theorem 1, as in 09_mean_field_sensitivity/run_sweep_A.py::compute_sweep_A. Returns x (H_in^s), y (H_in^o), beta[y, x]."""
    f_o = 1 - f_s
    v = np.linspace(0, 1, n)
    HINS, HINO = np.meshgrid(v, v)
    beta_s = f_s - (f_s * HINS) / np.maximum(f_s * HINS + f_o * h_out, eps)
    beta_o = f_s - (f_s * h_out) / np.maximum(f_s * h_out + f_o * HINO, eps)
    return v, v, f_s * beta_s + f_o * beta_o


def mf_pa(f_s, n=MF_POINTS):
    """Preferential attachment mean field (neighbours only, narcissistic=False).
    Returns x (h_ss), y (h_oo), beta[y, x]."""
    f_o = 1 - f_s
    v = np.linspace(0, 1, n)
    out = np.full((n, n), np.nan)
    for i, hoo in enumerate(v):
        for j, hss in enumerate(v):
            hos, hso = 1 - hoo, 1 - hss
            try:
                C = fun.compute_C(f_o, f_s, hoo, hos, hso, hss)
                p_oo, p_ss = fun.compute_p_oo(hoo, hos, C), fun.compute_p_ss(hss, hso, C)
                p_os, p_so = fun.compute_p_os(hos, hoo, C), fun.compute_p_so(hso, hss, C)
                b_o = fun.compute_misperception(f_s, fun.compute_opponent_perception(f_o, f_s, p_os, p_so, p_oo))
                b_s = fun.compute_misperception(f_s, fun.compute_supporter_perception(f_o, f_s, p_os, p_so, p_ss))
                out[i, j] = fun.compute_overall_misperception(f_s, b_o, b_s)
            except (ZeroDivisionError, FloatingPointError, ValueError):
                pass
    return v, v, out


# ── figure ────────────────────────────────────────────────────────────────────
CONFIG = {
    'C': dict(file=lambda N: here() / '07_finite_n_sbm' / 'output' / f'sweep_C_hout0.30_fs0.6667_N{N}_none.npz',
              xkey='hins_vals', ykey='hino_vals',
              xlabel=r'$H_{\mathrm{in}}^s$', ylabel=r'$H_{\mathrm{in}}^o$',
              mf=lambda: mf_sbm(0.3, 2 / 3),
              title='Overall misperception in finite-size simulations of stochastic block networks',
              out=here() / 'figures' / 'fig_sweep_C_N_convergence_4panel.pdf'),
    'D': dict(file=lambda N: here() / '08_finite_n_pa' / 'output' / f'sweep_D_m2_fs0.6667_N{N}_none.npz',
              xkey='hss_vals', ykey='hoo_vals',
              xlabel=r'$h_{ss}$', ylabel=r'$h_{oo}$',
              mf=lambda: mf_pa(2 / 3),
              title='Overall misperception in finite-size simulations of preferential attachment networks',
              out=here() / 'figures' / 'fig_sweep_D_N_convergence_4panel.pdf'),
}


def style_ax(ax, bold=False):
    for sp in ax.spines.values():
        sp.set_color(MYBLACK)
        sp.set_linewidth(2.0 if bold else 0.8)
    ax.tick_params(colors=MYBLACK)


def make(sweep):
    cfg = CONFIG[sweep]
    panels = []
    for N in N_VALS:
        f = cfg['file'](N)
        if f.exists():
            d = np.load(f)
            panels.append((rf'$N={N}$', d[cfg['xkey']], d[cfg['ykey']], d['beta_overall'], False))
        else:
            print(f'  [{sweep}] N={N} missing ({f.name}) — panel skipped')
    x, y, mf = cfg['mf']()
    panels.append(('Mean field', x, y, mf, True))

    vmax = max(np.nanmax(np.abs(p[3])) for p in panels[:-1]) if len(panels) > 1 else np.nanmax(np.abs(mf))
    S.apply_rc()
    fig, axes, cax = S.heatmap_grid(1, len(panels), suptitle=True)
    im_ref = None
    for k, (lab, xv, yv, arr, is_mf) in enumerate(panels):
        ax = axes[0, k]
        im = ax.imshow(arr, extent=[0, 1, 0, 1], origin='lower', cmap=CMAP,
                       vmin=-vmax, vmax=vmax, aspect='equal')
        im_ref = im_ref or im
        X, Y = np.meshgrid(xv, yv)
        cs = ax.contour(X, Y, arr, levels=[0], colors=[MYBLACK], linewidths=1.2, linestyles='dotted')
        ax.clabel(cs, inline=True, fmt='%.2f', fontsize=S.ANNOT, colors=[MYBLACK])
        S.style_heatmap(ax, xlabel=cfg['xlabel'], ylabel=cfg['ylabel'] if k == 0 else None)
        S.panel_title(ax, 'ABCD'[k], lab)
        if is_mf:
            for sp in ax.spines.values():
                sp.set_linewidth(2.0)

    S.colorbar(fig, im_ref, cax, r'Overall misperception $\beta_{\mathrm{overall}}$')
    S.suptitle(fig, cfg['title'], axes)
    S.save(fig, cfg['out'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['sbm', 'pa'], nargs='*', default=['sbm', 'pa'])
    for model in ap.parse_args().model:
        make({'sbm': 'C', 'pa': 'D'}[model])
