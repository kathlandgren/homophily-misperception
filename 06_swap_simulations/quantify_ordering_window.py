"""quantify_ordering_window.py — over which swap range does the swap model both
(i) put beta_overall in the empirical band and (ii) keep opponents more wrong than
supporters (beta_o > beta_s)? Produces the SI ordering-window table (tab:ordering-window).

Reads the swap simulations (each agent's estimate includes its own opinion).
npz axes are [i_hoo, i_hss, i_swap].

For each (h_ss, h_oo) in the 7 SI conditions x m in {2,5,10} x f_s in {2/3, 0.8} reports:
  entry      first swap % with beta_overall in [LO, HI]
  exit       first swap % after entry with beta_overall > HI (None if it never leaves)
  cross      first swap % > 0 with beta_s >= beta_o (supporters overtake opponents)
  window     swap % range with BOTH conditions (contiguous from entry)
  gap@entry  beta_o - beta_s at entry, percentage points
Repeated with the 'peers_' (unswapped-only) arrays where available.

Data required: 06_swap_simulations/output/ (submit_sweep_F.sh).

Usage:
    python 06_swap_simulations/quantify_ordering_window.py            # print the full report
    python 06_swap_simulations/quantify_ordering_window.py --latex    # also write tables/ordering_window_table.tex
    python 06_swap_simulations/quantify_ordering_window.py --all-grid # every (h_ss, h_oo) on the sweep grid
"""

import argparse
import pathlib

import numpy as np
from pyprojroot import here

DATA_DIR = here() / '06_swap_simulations' / 'output'
LO, HI = 0.21, 0.30
CONDITIONS = [(0.25, 0.25), (0.50, 0.50), (0.75, 0.75),
              (0.25, 0.75), (0.75, 0.25), (0.50, 0.75), (0.75, 0.50)]
M_VALS = [2, 5, 10]
FS_VALS = ['0.6667', '0.8000']


def analyse(sf, bs, bo, bov):
    inb = (bov >= LO) & (bov <= HI)
    out = dict(entry=None, exit=None, cross=None, window=None, gap=None, max_bov=float(bov.max()))
    over = np.where((sf > 0) & (bs >= bo))[0]
    if len(over):
        out['cross'] = float(sf[over[0]])
    if not inb.any():
        return out
    i0 = int(np.argmax(inb))
    out['entry'] = float(sf[i0])
    out['gap'] = float(bo[i0] - bs[i0])
    above = np.where((np.arange(len(sf)) > i0) & (bov > HI))[0]
    if len(above):
        out['exit'] = float(sf[above[0]])
    both = inb & (bo > bs)
    if both[i0]:
        j = i0
        while j + 1 < len(sf) and both[j + 1]:
            j += 1
        out['window'] = (float(sf[i0]), float(sf[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all-grid', action='store_true', help='all (h_ss,h_oo) on the sweep grid')
    ap.add_argument('--latex', type=str, nargs='?', default=None,
                    const=str(here() / 'tables' / 'ordering_window_table.tex'),
                    help='write the SI table (default path: tables/ordering_window_table.tex)')
    a = ap.parse_args()
    rows = []
    for fs in FS_VALS:
        for m in M_VALS:
            d = np.load(DATA_DIR / f'sweep_F_m{m}_fs{fs}_N1000.npz', allow_pickle=True)
            hs, ho = list(np.round(d['hss_vals'], 3)), list(np.round(d['hoo_vals'], 3))
            sf = d['swap_fracs'] * 100
            conds = [(s, o) for o in ho for s in hs] if a.all_grid else CONDITIONS
            for hss, hoo in conds:
                si, oi = hs.index(round(hss, 3)), ho.index(round(hoo, 3))
                r_all = analyse(sf, d['all_beta_s'][oi, si], d['all_beta_o'][oi, si],
                                d['all_beta_overall'][oi, si])
                r_un = analyse(sf, d['peers_beta_s'][oi, si], d['peers_beta_o'][oi, si],
                               d['peers_beta_overall'][oi, si])
                rows.append((fs, m, hss, hoo, r_all, r_un))

    f = lambda x: '--' if x is None else f'{x:.1f}'
    w = lambda x: '--' if x is None else f'{x[0]:.0f}-{x[1]:.0f}'
    print(f'{"f_s":>6} {"m":>3} {"h_ss":>5} {"h_oo":>5} | {"entry":>5} {"exit":>5} {"cross":>5} '
          f'{"window":>7} {"gap":>5} {"maxB":>5} || unswapped: {"entry":>5} {"cross":>5} {"window":>7}')
    for fs, m, hss, hoo, r, u in rows:
        gap = '--' if r['gap'] is None else f'{100 * r["gap"]:.1f}'
        print(f'{fs:>6} {m:>3} {hss:>5} {hoo:>5} | {f(r["entry"]):>5} {f(r["exit"]):>5} '
              f'{f(r["cross"]):>5} {w(r["window"]):>7} {gap:>5} '
              f'{100 * r["max_bov"]:>5.1f} || {f(u["entry"]):>16} {f(u["cross"]):>5} {w(u["window"]):>7}')

    if a.latex:
        # Compact SI table: rows = (f_s, condition); per m: joint window and gap at band entry.
        def tw(x):
            if x is None:
                return '---'
            a0, a1 = round(x[0]), round(x[1])
            return f'{a0}' if a0 == a1 else f'{a0}--{a1}'
        lines = [r'\begin{table}[H]', r'\centering', r'\small',
                 r'\caption{\textcolor{blue}{Swap percentages over which the swap model reproduces both empirical features: '
                 r'overall misperception $\beta_{\mathrm{overall}}$ within the empirical range (21--30 percentage points) '
                 r'and opponents more wrong than supporters ($\beta_o > \beta_s$). '
                 r"``Window'' gives the first and last swap percentage at which both hold; "
                 r'--- indicates that no such range exists. '
                 r"``Gap'' is $\beta_o - \beta_s$ (percentage points) at the swap percentage where $\beta_{\mathrm{overall}}$ "
                 r'first enters the empirical range. $N = 1{,}000$, 200 simulations per condition; '
                 r"each agent's estimate includes its own opinion (Eq.~\eqref{eq:1}).}}",
                 r'\label{tab:ordering-window}',
                 r'\begin{tabular}{l rr rr rr}', r'\toprule',
                 r' & \multicolumn{2}{c}{$m=2$} & \multicolumn{2}{c}{$m=5$} & \multicolumn{2}{c}{$m=10$} \\',
                 r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
                 r'$(h_{ss}, h_{oo})$ & window (\%) & gap & window (\%) & gap & window (\%) & gap \\',
                 r'\midrule']
        by = {(fs, hss, hoo, m): r for fs, m, hss, hoo, r, u in rows}
        for fs in FS_VALS:
            fsl = r'\nicefrac{2}{3}' if fs == '0.6667' else '0.8'
            lines.append(r'\multicolumn{7}{l}{\textit{$f_s = ' + fsl + r'$}} \\')
            for hss, hoo in CONDITIONS:
                cells = []
                for m in M_VALS:
                    r = by[(fs, hss, hoo, m)]
                    gap = '---' if r['gap'] is None else f'${100 * r["gap"]:.1f}$'
                    cells += [tw(r['window']), gap]
                lines.append(f'\\quad ({hss:g}, {hoo:g}) & ' + ' & '.join(cells) + r' \\')
            if fs != FS_VALS[-1]:
                lines.append(r'\addlinespace')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        pathlib.Path(a.latex).parent.mkdir(exist_ok=True)
        pathlib.Path(a.latex).write_text('\n'.join(lines) + '\n')
        print('wrote', a.latex)


if __name__ == '__main__':
    main()
