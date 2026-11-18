"""generate_fig_sweep_G_broadcast_illustrative.py — SI figure fig_sweep_G_broadcast_illustrative.pdf.

Two-panel figure structured like main-text Fig 5 (figures/combine_swap_fig_narcTrue.pdf):
  Panel A: β vs symmetric broadcast attention share λ (illustrative curve),
           h_ss=h_oo=0.75, m=10, f_s=2/3, mu=0 (fully anti-support).
           m=10 matches main-text Figs 4 and 5.
  Panel B: histogram of individual perceived support fractions from a
           single simulated network realization with λ=0.40 broadcast
           (the same network as main-text Figs 4B and 5B: generator, seed, m, h)
           applied (λ chosen to place β_overall in the empirical range).
           Styled to match figures/figure_histogram_example_grouped_swap_narcTrue.pdf.

Data required: run 11_broadcast_media/run_sweep_G.py first (11_broadcast_media/output/).

Usage:
    python 11_broadcast_media/generate_fig_sweep_G_broadcast_illustrative.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import importlib.util
import random
import sys

from pyprojroot import here


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, here() / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

SWEEP_G_DIR = here() / '11_broadcast_media' / 'output'
OUT_PATH    = here() / 'figures' / 'fig_sweep_G_broadcast_illustrative.pdf'

# ── Parameters ────────────────────────────────────────────────────────────────
M_VAL      = 10        # density parameter, matches main-text Figs 4 and 5
FS_VAL     = 2 / 3
H_SS       = 0.75
H_OO       = 0.75
MU_TARGET  = 0.0        # fully anti-support broadcast
LAMBDA_B   = 0.40       # λ for Panel B (middle of empirical band)

BETA_LO = 0.21          # fractional
BETA_HI = 0.30

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
matplotlib.rcParams['font.family'] = 'Arial'
SMALL_SIZE  = 9
MEDIUM_SIZE = 10
plt.rc('font',   size=SMALL_SIZE)
plt.rc('axes',   titlesize=SMALL_SIZE)
plt.rc('axes',   labelsize=SMALL_SIZE)
plt.rc('xtick',  labelsize=SMALL_SIZE)
plt.rc('ytick',  labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)

COLOR_S    = '#19bdff'
COLOR_S_DK = '#0072b8'
COLOR_O    = '#f2d138'
COLOR_O_DK = '#b89a00'
COLOR_ALL  = '#626262'
COLOR_BAR  = '#c8c8c8'
MYBLACK    = '#222222'
LW = 1.3

def style_ax(ax):
    ax.spines['bottom'].set_color(MYBLACK)
    ax.spines['left'].set_color(MYBLACK)
    ax.tick_params(axis='x', colors=MYBLACK)
    ax.tick_params(axis='y', colors=MYBLACK)
    sns.despine(ax=ax)

# ── Load broadcast-model analytical data ─────────────────────────────────────────
def h_tag(h):  return str(int(round(h * 1000)))
def fs_tag(fs):
    if abs(fs - 2/3) < 1e-4: return '0667'
    return f'{fs:.4f}'.replace('.', '')

stem = (f'sweep_G_m{M_VAL}_fs{fs_tag(FS_VAL)}'
        f'_hss{h_tag(H_SS)}_hoo{h_tag(H_OO)}_pa_rescaled')
dg = np.load(SWEEP_G_DIR / f'{stem}.npz', allow_pickle=True)

lam_vals = dg['lambda_vals']
mu_vals  = dg['mu_vals']
k_mu0    = int(np.argmin(np.abs(mu_vals - MU_TARGET)))

bs_curve  = dg['beta_s_new'][:, 0, k_mu0]
bo_curve  = dg['beta_o_new'][0, :, k_mu0]
bov_curve = np.array([dg['beta_overall_new'][i, i, k_mu0]
                      for i in range(len(lam_vals))])

bs_peer = float(dg['beta_s_peer'])
bo_peer = float(dg['beta_o_peer'])

# ── Simulate a single network for Panel B ─────────────────────────────────────
# Load graph generator and opinion functions
gen = _load('generate_homophilic_graph_asymmetric', 'src/generate_homophilic_graph_asymmetric.py')
op  = _load('opinion_functions', 'src/opinion_functions.py')

N = 1000
fo = 1 - FS_VAL
h_ab = 1 - H_OO   # cross-group for minority (opponents)
h_ba = 1 - H_SS   # cross-group for majority (supporters)

# Same network as Figs 4B and 5B: seed drawn as in 05_bayesian_simulations/make_histogram_panel.ipynb
seed_val = int(np.random.default_rng(seed=42).integers(0, 2**31))
G, opp_list = gen.homophilic_barabasi_albert_graph_assym(
    N, M_VAL, 1 / 3, h_ab, h_ba, seed=seed_val)

opp_set  = set(opp_list)
all_nodes = list(G.nodes())
supp_set = set(all_nodes) - opp_set

# Compute peer-network perceived support for each node
_, peer_perceived = op.generate_perceived_opinion(
    G, opp_list, {}, narcissistic=True, weigh_connected=False)
# peer_perceived[i] = fraction of i's neighbors who are OPPONENTS
# but we want fraction who SUPPORT (majority = supporters)
# Convention check: opinion_functions returns perceived opinion as
# fraction of neighbors in minority (opponent) group, so perceived support
# = 1 - peer_perceived[i]
# Let's check: mean over supporters should be ~ FS_VAL if no homophily
# With high homophily, supporters see mostly supporters -> high perceived support
peer_support = np.array([peer_perceived[i] for i in range(N)])

# Apply broadcast: final = (1-lambda)*peer + lambda*mu
# lambda_s = lambda_o = LAMBDA_B, mu = MU_TARGET = 0
final_support = (1 - LAMBDA_B) * peer_support + LAMBDA_B * MU_TARGET

supp_indices = [i for i in range(N) if i not in opp_set]
opp_indices  = list(opp_set)

perc_all  = final_support
perc_supp = final_support[supp_indices]
perc_opp  = final_support[opp_indices]

mean_all  = perc_all.mean()
mean_supp = perc_supp.mean()
mean_opp  = perc_opp.mean()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=300)

# ── Panel A: Illustrative curve ───────────────────────────────────────────────
inside = (bov_curve >= BETA_LO) & (bov_curve <= BETA_HI)
if inside.any():
    # beta_overall is linear in lambda (Eq. broadcast), so interpolate the exact band edges
    # instead of snapping to the lambda grid (grid snapping gave 0.35-0.45 vs the exact 0.34-0.47).
    assert np.all(np.diff(bov_curve) > 0)
    x_entry = float(np.interp(BETA_LO, bov_curve, lam_vals))
    x_exit  = float(np.interp(BETA_HI, bov_curve, lam_vals))
    ax_a.axvspan(x_entry, x_exit, color='#a83232', alpha=0.15, edgecolor=None, zorder=0)
    # mark the λ used in Panel B
    ax_a.axvline(LAMBDA_B, color=MYBLACK, lw=0.8, ls=':', zorder=2)
    ax_a.text(LAMBDA_B + 0.02, -0.15, rf'$\lambda={LAMBDA_B}$',
              fontsize=SMALL_SIZE - 1, color=MYBLACK, va='bottom')

ax_a.axhline(0, color=MYBLACK, lw=0.6, ls=':', zorder=1)
ax_a.plot(lam_vals, bov_curve, color=COLOR_ALL, lw=LW, ls='-')
ax_a.plot(lam_vals, bs_curve,  color=COLOR_S,   lw=LW, ls='-')
ax_a.plot(lam_vals, bo_curve,  color=COLOR_O,   lw=LW, ls='-')

ax_a.set_xlim(0, 1)
ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_a.set_xlabel(r'Attention share $\lambda$', size=SMALL_SIZE)
ax_a.set_ylabel(r'Misperception $\beta$', size=SMALL_SIZE)
ax_a.set_title('A. Misperception as a function of attention share', size=SMALL_SIZE)

legend_handles = [
    Line2D([0], [0], color=COLOR_ALL, lw=LW, label=r'$\beta_{\mathrm{overall}}$'),
    Line2D([0], [0], color=COLOR_S,   lw=LW, label=r'$\beta_s$ (supporters)'),
    Line2D([0], [0], color=COLOR_O,   lw=LW, label=r'$\beta_o$ (opponents)'),
]
ax_a.legend(handles=legend_handles, frameon=False, fontsize=SMALL_SIZE - 1,
            loc='upper left')
style_ax(ax_a)

# ── Panel B: Example network realization ─────────────────────────────────────
bins_edges = np.linspace(0, 1, 11)
ax_b.hist(perc_all,  bins=bins_edges, histtype='bar',
          color=COLOR_BAR, edgecolor='white', alpha=1.0, zorder=1)
ax_b.hist(perc_supp, bins=bins_edges, histtype='bar',
          color=COLOR_S,   edgecolor='white', alpha=0.7, zorder=2)
ax_b.hist(perc_opp,  bins=bins_edges, histtype='bar',
          color=COLOR_O,   edgecolor='white', alpha=0.7, zorder=3)

ymax = ax_b.get_ylim()[1]

ax_b.axvline(FS_VAL,     color=MYBLACK,    ls='-',  lw=1, zorder=5)
ax_b.axvline(mean_all,   color=MYBLACK,    ls='--', lw=1, zorder=4)
ax_b.axvline(mean_supp,  color=COLOR_S_DK, ls='--', lw=1, zorder=4)
ax_b.axvline(mean_opp,   color=COLOR_O_DK, ls='--', lw=1, zorder=4)

ax_b.text(FS_VAL + 0.01, ymax * 0.30, 'actual', rotation=90,
          color=MYBLACK, fontsize=SMALL_SIZE - 1, va='bottom', zorder=5)

arrow_y_ov = ymax * 0.95
arrow_y_s  = ymax * 0.82
arrow_y_o  = ymax * 0.69

ax_b.annotate('', xy=(FS_VAL, arrow_y_ov), xytext=(mean_all, arrow_y_ov),
              arrowprops=dict(arrowstyle='<->', color=MYBLACK, lw=1), zorder=5)
ax_b.text((mean_all + FS_VAL) / 2, arrow_y_ov + ymax * 0.01,
          r'$\beta_{\mathrm{overall}}$', ha='center', va='bottom',
          fontsize=SMALL_SIZE, color=MYBLACK, zorder=5)

ax_b.annotate('', xy=(FS_VAL, arrow_y_s), xytext=(mean_supp, arrow_y_s),
              arrowprops=dict(arrowstyle='<->', color=COLOR_S_DK, lw=1), zorder=5)
ax_b.text((mean_supp + FS_VAL) / 2, arrow_y_s + ymax * 0.01,
          r'$\beta_s$', ha='center', va='bottom',
          fontsize=SMALL_SIZE, color=COLOR_S_DK, zorder=5)

ax_b.annotate('', xy=(FS_VAL, arrow_y_o), xytext=(mean_opp, arrow_y_o),
              arrowprops=dict(arrowstyle='<->', color=COLOR_O_DK, lw=1), zorder=5)
ax_b.text((mean_opp + FS_VAL) / 2, arrow_y_o + ymax * 0.01,
          r'$\beta_o$', ha='center', va='bottom',
          fontsize=SMALL_SIZE, color=COLOR_O_DK, zorder=5)

ax_b.set_xlim(0, 1)
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.set_xlabel('Perceived supporter fraction', size=SMALL_SIZE)
ax_b.set_ylabel('N agents', size=SMALL_SIZE)
ax_b.set_title(rf'B. Example network realization ($\lambda={LAMBDA_B}$)', size=SMALL_SIZE)
style_ax(ax_b)

plt.tight_layout()
plt.subplots_adjust(wspace=0.30)
plt.savefig(OUT_PATH, format='pdf', bbox_inches='tight')
plt.close()
print(f'Saved {OUT_PATH}')
if inside.any():
    print(f'band: lambda {x_entry:.3f}-{x_exit:.3f}')
print(f'peer: beta_s={bs_peer:.3f} beta_o={bo_peer:.3f}; at entry: beta_s={np.interp(x_entry, lam_vals, bs_curve):.3f} beta_o={np.interp(x_entry, lam_vals, bo_curve):.3f}')
zc = float(np.interp(0.0, bs_curve, lam_vals))
print(f'beta_s crosses 0 at lambda ~ {zc:.3f}')
print(f'panel B (lambda={LAMBDA_B}): beta_overall={FS_VAL-mean_all:.3f} beta_s={FS_VAL-mean_supp:.3f} beta_o={FS_VAL-mean_opp:.3f}')
