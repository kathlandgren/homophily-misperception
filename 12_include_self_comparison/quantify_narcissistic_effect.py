"""
quantify_narcissistic_effect.py — how much does including the agent's own opinion matter?

Pairs every simulation output in which agents include their own opinion (narcissistic=True,
<folder>/output/) with its neighbours-only twin (narcissistic=False, <folder>/output_exclude_self/)
by filename, and reports Δ = (narc=True) − (narc=False), for 06_swap_simulations,
07_finite_n_sbm, 08_finite_n_pa and 10_degree_matched. (The `sweep` column of the CSV keeps
the letter used in the output file names: C = 07, D = 08, E = 10, F = 06.)

Outputs:
  12_include_self_comparison/summary_stats/narcissistic_effect_all.csv
                                     one row per condition (sweep × parameters × variant × model/kind)
  tables/narcissistic_effect_headline.tex
                                     SI table (three panels: preferential attachment, stochastic block, degree-matched)
  figures/fig_si_narcissistic_comparison.pdf
                                     preferential attachment beta_overall, both formulations, m in {2,5,10}, f_s=2/3, N=1000
  figures/fig_si_narcissistic_diff.pdf
                                     |Δ beta_overall| for the same conditions
(`--N 500` writes *_N500.pdf previews of the two figures.)

Data required: output/ and output_exclude_self/ of 06, 07, 08 and 10 (run each submit_sweep_*.sh
twice, the second time with EXCLUDE_SELF=1). Conditions whose include-self file is missing are
listed as 'pending'. The broadcast model (11) is not quantified separately: it is a linear
transform of the degree-matched and swap peer baselines, beta_new = beta_peer*(1-λ) + λ*(f_s-μ), so |Δbeta_new| ≤ |Δbeta_peer|.

Grids are indexed [i_hoo, i_hss] (C: [i_hino, i_hins]); F arrays are [i_hoo, i_hss, swap].

Usage:
    python 12_include_self_comparison/quantify_narcissistic_effect.py
"""

import argparse
import csv
import importlib.util
import re
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

OUT_CSV = here() / '12_include_self_comparison' / 'summary_stats'
OUT_TEX = here() / 'tables'
OUT_FIG = here() / 'figures'
for _d in (OUT_CSV, OUT_TEX):
    _d.mkdir(exist_ok=True)

BAND = (0.21, 0.30)
SIGN_TOL = 0.005          # cells with |beta| < tol under both formulations are not counted for sign agreement

FOLDERS = {'C': '07_finite_n_sbm', 'D': '08_finite_n_pa', 'E': '10_degree_matched', 'F': '06_swap_simulations'}
DIRS = {s: (here() / f / 'output', here() / f / 'output_exclude_self') for s, f in FOLDERS.items()}

PATTERNS = {
    'C': re.compile(r'sweep_C_hout(?P<hout>[\d.]+)_fs(?P<fs>[\d.]+)_N(?P<N>\d+)_(?P<variant>\S+)\.npz'),
    'D': re.compile(r'sweep_D_m(?P<m>\d+)_fs(?P<fs>[\d.]+)_N(?P<N>\d+)_(?P<variant>\S+)\.npz'),
    'E': re.compile(r'sweep_E_m(?P<m>\d+)_fs(?P<fs>[\d.]+)_N(?P<N>\d+)\.npz'),
    'F': re.compile(r'sweep_F_m(?P<m>\d+)_fs(?P<fs>[\d.]+)_N(?P<N>\d+)\.npz'),
}


# ── metrics ───────────────────────────────────────────────────────────────────
def grid_metrics(bF, bT):
    """bF, bT: dicts with beta_overall, beta_s, beta_o arrays (same shape)."""
    r = {}
    for k in ('beta_overall', 'beta_s', 'beta_o'):
        d = bT[k] - bF[k]
        r[f'max_abs_d_{k}'] = np.nanmax(np.abs(d))
        r[f'median_abs_d_{k}'] = np.nanmedian(np.abs(d))
        r[f'mean_d_{k}'] = np.nanmean(d)
    oF, oT = bF['beta_overall'], bT['beta_overall']
    ok = np.isfinite(oF) & np.isfinite(oT)
    informative = ok & ~((np.abs(oF) < SIGN_TOL) & (np.abs(oT) < SIGN_TOL))
    r['sign_agreement_beta_overall'] = np.mean(np.sign(oF[informative]) == np.sign(oT[informative]))
    r['area_pos_narcFalse'] = np.mean(oF[ok] > 0)
    r['area_pos_narcTrue'] = np.mean(oT[ok] > 0)
    r['band_frac_narcFalse'] = np.mean((oF[ok] >= BAND[0]) & (oF[ok] <= BAND[1]))
    r['band_frac_narcTrue'] = np.mean((oT[ok] >= BAND[0]) & (oT[ok] <= BAND[1]))
    return r


def band_entry_pct(curve, x_pct):
    inside = (curve >= BAND[0]) & (curve <= BAND[1])
    if inside[0]:
        return 0.0
    return x_pct[np.argmax(inside)] if inside.any() else np.nan


def swap_metrics(dF, dT, kind):
    x = dF['swap_fracs'] * 100
    r = grid_metrics({k: dF[f'{kind}_{k}'][:, :, 0] for k in ('beta_overall', 'beta_s', 'beta_o')},
                     {k: dT[f'{kind}_{k}'][:, :, 0] for k in ('beta_overall', 'beta_s', 'beta_o')})
    r = {f'{k}@0swaps': v for k, v in r.items()}
    full = {k: dT[f'{kind}_{k}'] - dF[f'{kind}_{k}'] for k in ('beta_overall',)}
    r['max_abs_d_beta_overall_allswaps'] = np.nanmax(np.abs(full['beta_overall']))
    eF = np.array([[band_entry_pct(dF[f'{kind}_beta_overall'][i, j], x) for j in range(dF['hss_vals'].size)]
                   for i in range(dF['hoo_vals'].size)])
    eT = np.array([[band_entry_pct(dT[f'{kind}_beta_overall'][i, j], x) for j in range(dT['hss_vals'].size)]
                   for i in range(dT['hoo_vals'].size)])
    both = np.isfinite(eF) & np.isfinite(eT)
    r['band_entry_shift_pct_mean'] = np.mean(eT[both] - eF[both]) if both.any() else np.nan
    r['band_entry_shift_pct_max_abs'] = np.max(np.abs(eT[both] - eF[both])) if both.any() else np.nan
    r['n_cells_reach_band_narcFalse'] = int(np.isfinite(eF).sum())
    r['n_cells_reach_band_narcTrue'] = int(np.isfinite(eT).sum())
    r['n_cells'] = int(eF.size)
    return r


# ── collect ───────────────────────────────────────────────────────────────────
def collect():
    rows = []
    for s in 'CDEF':
        dT_dir, dF_dir = DIRS[s]
        for fF in sorted(dF_dir.glob(f'sweep_{s}_*.npz')):
            mo = PATTERNS[s].match(fF.name)
            if not mo:
                continue
            base = dict(sweep=s, **mo.groupdict())
            fT = dT_dir / fF.name
            if not fT.exists():
                rows.append({**base, 'status': 'pending'})
                continue
            dF, dT = np.load(fF), np.load(fT)
            if s in 'CD':
                rows.append({**base, 'status': 'ok', **grid_metrics(
                    {k: dF[k] for k in ('beta_overall', 'beta_s', 'beta_o')},
                    {k: dT[k] for k in ('beta_overall', 'beta_s', 'beta_o')})})
            elif s == 'E':
                for model in ('pa', 'sbm'):
                    rows.append({**base, 'model': model, 'status': 'ok', **grid_metrics(
                        {k: dF[f'{model}_{k}'] for k in ('beta_overall', 'beta_s', 'beta_o')},
                        {k: dT[f'{model}_{k}'] for k in ('beta_overall', 'beta_s', 'beta_o')})})
            else:
                for kind in ('all', 'peers'):
                    rows.append({**base, 'kind': kind, 'status': 'ok', **swap_metrics(dF, dT, kind)})
    return rows


def write_csv(rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = OUT_CSV / 'narcissistic_effect_all.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    print(f'Saved {path}  ({len(rows)} rows, {sum(r["status"] == "pending" for r in rows)} pending)')


# ── headline table ────────────────────────────────────────────────────────────
def find(rows, **kw):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def write_table(rows):
    """One table, three panels (A: preferential attachment by m; B: stochastic block by N;
    C: degree-matched comparison). Preferential attachment N = 500 and the 0-swap grid of the swap simulations
    are left out: they repeat the N = 1000 / m = 5 rows (N = 500 agrees to two decimals)."""
    D = lambda m: find(rows, sweep='D', m=m, fs='0.6667', N=1000, variant='none')      # noqa: E731
    C = lambda N: find(rows, sweep='C', hout='0.30', fs='0.6667', N=N, variant='none')  # noqa: E731
    E = lambda mo: find(rows, sweep='E', m=5, fs='0.6667', N=1000, model=mo)            # noqa: E731
    panels = [
        (r'(A) Preferential attachment, $N=1000$', [(f'$m={m}$', D(m)) for m in (2, 5, 10)]),
        (r'(B) Stochastic block, $H_{\mathrm{out}}=0.3$', [(f'$N={N}$', C(N)) for N in (100, 500, 1000)]),
        (r'(C) Degree-matched networks, $m=5$, $N=1000$',
         [('Preferential attachment', E('pa')), ('Stochastic block', E('sbm'))]),
    ]
    lines = [
        '% narcissistic_effect_headline.tex — generated by 12_include_self_comparison/quantify_narcissistic_effect.py',
        '% Δ = (agent includes own opinion) − (neighbors only); f_s = 2/3; no rescaling.',
        '% Needs booktabs (\\toprule, \\cmidrule, \\addlinespace).',
        r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\caption{\textcolor{blue}{Effect of including the agent\textquotesingle s own opinion on misperception, '
        r'$\Delta = \beta(\text{own opinion included}) - \beta(\text{neighbors only})$, '
        r'for $f_s = 2/3$ without Bayesian rescaling (200 simulations per cell of a $21\times21$ homophily grid). '
        r'Each row summarizes the homophily grid of that model. '
        r'The last two columns give the percentage of the grid with $\beta_{\mathrm{overall}}>0$ and with '
        r'$\beta_{\mathrm{overall}}$ in the empirical range $[0.21, 0.30]$, '
        r'neighbors only $\to$ own opinion included. '
        r'Preferential attachment networks with $N=500$ give the same values as $N=1000$ to two decimals.}}',
        r'\label{tab:si-narcissistic-effect}',
        r'\begin{tabular}{lcccccc}', r'\toprule',
        r' & \multicolumn{2}{c}{$|\Delta\beta_{\mathrm{overall}}|$} & \multicolumn{2}{c}{mean $\Delta\beta_{\mathrm{group}}$}'
        r' & \multicolumn{2}{c}{Share of grid (\%)} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
        r' & max & median & supporters & opponents & $\beta_{\mathrm{overall}}>0$ & empirical range \\',
        r'\midrule',
    ]
    for i, (head, items) in enumerate(panels):
        if i:
            lines.append(r'\addlinespace')
        lines.append(f'\\multicolumn{{7}}{{l}}{{\\textit{{{head}}}}} \\\\')
        for lab, r in items:
            if r is None or r.get('status') != 'ok':
                lines.append(f'\\quad {lab} & \\multicolumn{{6}}{{c}}{{pending}} \\\\')
                continue
            lines.append(
                f'\\quad {lab} & {r["max_abs_d_beta_overall"]:.3f} & {r["median_abs_d_beta_overall"]:.3f} & '
                f'${r["mean_d_beta_s"]:+.3f}$ & ${r["mean_d_beta_o"]:+.3f}$ & '
                f'{100 * r["area_pos_narcFalse"]:.0f} $\\to$ {100 * r["area_pos_narcTrue"]:.0f} & '
                f'{100 * r["band_frac_narcFalse"]:.0f} $\\to$ {100 * r["band_frac_narcTrue"]:.0f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    path = OUT_TEX / 'narcissistic_effect_headline.tex'
    path.write_text('\n'.join(lines))
    print(f'Saved {path}')


# ── SI figures ────────────────────────────────────────────────────────────────


def si_figures(N):
    S.apply_rc()
    ms = [2, 5, 10]
    sfx = '' if N == 1000 else f'_N{N}'
    pair = {}
    for m in ms:
        name = f'sweep_D_m{m}_fs0.6667_N{N}_none.npz'
        fT, fF = DIRS['D'][0] / name, DIRS['D'][1] / name
        pair[m] = (np.load(fF), np.load(fT)) if fT.exists() and fF.exists() else None
    avail = [p for p in pair.values() if p]
    if not avail:
        print(f'  SI figures (N={N}): no data yet')
        return
    vmax = max(np.nanmax(np.abs(d['beta_overall'])) for p in avail for d in p)
    dmax = max(np.nanmax(np.abs(p[1]['beta_overall'] - p[0]['beta_overall'])) for p in avail)

    # comparison: rows = formulation, cols = m
    fig, axes, cax = S.heatmap_grid(2, 3, suptitle=True, row_titles=True, sup_lines=2)
    im = None
    for r, lab in enumerate(('Agents average over neighbors only',
                             "Agents include their own opinion")):
        S.row_title(fig, axes[r], lab)
        for c, m in enumerate(ms):
            ax = axes[r, c]
            S.style_heatmap(ax, xlabel=r'$h_{ss}$', ylabel=r'$h_{oo}$' if c == 0 else None)
            S.panel_title(ax, 'ABCDEF'[r * 3 + c], f'$m={m}$')
            if pair[m] is None:
                continue
            d = pair[m][r]
            im = ax.imshow(d['beta_overall'], extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, aspect='equal')
            X, Y = np.meshgrid(d['hss_vals'], d['hoo_vals'])
            cz = ax.contour(X, Y, d['beta_overall'], levels=[0], colors=S.BLACK, linewidths=1.2,
                            linestyles='dotted')
            ax.clabel(cz, fmt='%.2f', fontsize=S.ANNOT)
            cb_ = ax.contour(X, Y, d['beta_overall'], levels=list(BAND), colors=S.BLACK, linewidths=1.2)
            ax.clabel(cb_, fmt='%.2f', fontsize=S.ANNOT)
    if im is not None:
        S.colorbar(fig, im, cax, r'Overall misperception $\beta_{\mathrm{overall}}$')
    S.suptitle(fig, "The effect of including the agents' own opinion in their estimate\n"
                    "on preferential attachment networks", axes)
    S.save(fig, OUT_FIG / f'fig_si_narcissistic_comparison{sfx}.pdf')

    # |Δ|
    fig, axes, cax = S.heatmap_grid(1, 3, suptitle=True, sup_lines=2)
    im = None
    for c, m in enumerate(ms):
        ax = axes[0, c]
        S.style_heatmap(ax, xlabel=r'$h_{ss}$', ylabel=r'$h_{oo}$' if c == 0 else None)
        S.panel_title(ax, 'ABC'[c], f'$m={m}$')
        if pair[m] is None:
            continue
        dF, dT = pair[m]
        diff = np.abs(dT['beta_overall'] - dF['beta_overall'])
        im = ax.imshow(diff, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=dmax,
                       aspect='equal')
    if im is not None:
        S.colorbar(fig, im, cax, r'Absolute change in overall misperception $|\Delta\beta_{\mathrm{overall}}|$')
    S.suptitle(fig, "Change in overall misperception from including the agents' own opinion\n"
                    "on preferential attachment networks", axes)
    S.save(fig, OUT_FIG / f'fig_si_narcissistic_diff{sfx}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='*', default=[1000],
                    help='network size(s) for the two SI figures (1000 = manuscript; others are previews)')
    args = ap.parse_args()
    rows = collect()
    write_csv(rows)
    write_table(rows)
    for N in args.N:
        si_figures(N)
