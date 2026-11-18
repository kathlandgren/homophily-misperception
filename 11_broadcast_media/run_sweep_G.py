"""
run_sweep_G.py — broadcast node model of media bias (pure numpy, no simulation).

For each peer-network baseline condition (h_ss, h_oo, m, f_s, model), applies:

  Network-edge (alpha = 1/(2m+2), narcissistic=True):  beta_g^new = beta_g^peer*(1 - lam_g*alpha) + lam_g*alpha*(f_s - mu)
  Rescaled:      beta_g^new = beta_g^peer*(1 - lam_g)       + lam_g*(f_s - mu)

Data sources (full 3x3 (h_ss,h_oo) grid):
  PA (f_s in {2/3, 0.8}): swap simulations (06), peers_beta_s/o at swap_count=0
  PA (f_s = 0.5):         degree-matched simulations (10), pa_beta_s/o
  SBM (all f_s):          degree-matched simulations (10), sbm_beta_s/o  (exact 3x3 run if present, else bilinear interp. of 11x11)

Data required: 10_degree_matched/output/ (submit_sweep_E.sh), 10_degree_matched/output_G3x3/
(submit_sweep_E_G3x3.sh) and 06_swap_simulations/output/ (submit_sweep_F.sh). Runs locally in seconds.

Output: 11_broadcast_media/output/sweep_G_m{m}_fs{fstag}_hss{htag}_hoo{htag}_{model}[_rescaled].npz

Usage:
    python 11_broadcast_media/run_sweep_G.py
"""
import numpy as np
from pyprojroot import here

SWEEP_E_DIR = here() / '10_degree_matched' / 'output'
SWEEP_F_DIR = here() / '06_swap_simulations' / 'output'
# Exact degree-matched baselines on the broadcast-model grid (submit_sweep_E_G3x3.sh). If absent, the 11x11
# degree-matched grid is bilinearly interpolated at 0.25/0.75 (and the run prints a warning).
SWEEP_E_G3X3_DIR = here() / '10_degree_matched' / 'output_G3x3'
OUT_DIR     = here() / '11_broadcast_media' / 'output'
OUT_DIR.mkdir(exist_ok=True)

M_VALS   = [2, 5, 10]
FS_VALS  = [0.5, 2/3, 0.8]
H_VALS   = [0.25, 0.5, 0.75]
N_FIXED  = 1000

LAMBDA_VALS  = np.linspace(0, 1, 21)
MU_VALS_BASE = np.linspace(0, 1, 21)


def load_sweep_F_baselines(m_val, fs_val):
    """Return dict keyed by (h_ss, h_oo) -> (beta_s, beta_o) from the swap simulations at swap=0."""
    tag  = f'{fs_val:.4f}'
    path = SWEEP_F_DIR / f'sweep_F_m{m_val}_fs{tag}_N1000.npz'
    if not path.exists():
        return {}
    d        = np.load(path)
    hss_list = list(d['hss_vals'])
    result   = {}
    for hss in H_VALS:
        if hss not in hss_list:
            continue
        si = hss_list.index(hss)
        for hoo in H_VALS:
            if hoo not in hss_list:
                continue
            oi = hss_list.index(hoo)
            result[(hss, hoo)] = (
                float(d['peers_beta_s'][oi, si, 0]),
                float(d['peers_beta_o'][oi, si, 0]),
                'sweepF_exact',
            )
    return result


def _bilinear(grid_h, arr_hoo_hss, hss, hoo):
    """Bilinear interpolation on a regular grid; arr is indexed [i_hoo, i_hss]."""
    g = np.asarray(grid_h, dtype=float)
    def bracket(x):
        j = int(np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2))
        t = (x - g[j]) / (g[j + 1] - g[j])
        return j, t
    js, ts = bracket(hss)
    jo, to = bracket(hoo)
    a = arr_hoo_hss
    return float((1 - to) * ((1 - ts) * a[jo, js] + ts * a[jo, js + 1])
                 + to * ((1 - ts) * a[jo + 1, js] + ts * a[jo + 1, js + 1]))


def load_sweep_E_baselines(m_val, fs_val, model):
    """Return dict keyed by (h_ss, h_oo) -> (beta_s, beta_o, source) from the degree-matched simulations.

    Prefers the exact 3x3 G-grid run; otherwise uses exact 11x11 grid points where they
    exist and bilinear interpolation elsewhere. Arrays are indexed [i_hoo, i_hss].
    """
    tag    = f'{fs_val:.4f}'
    key_s  = f'{model}_beta_s'
    key_o  = f'{model}_beta_o'
    result = {}

    exact = SWEEP_E_G3X3_DIR / f'sweep_E_m{m_val}_fs{tag}_N{N_FIXED}.npz'
    if exact.exists():
        d = np.load(exact)
        hl = [round(float(v), 4) for v in d['hss_vals']]
        for hss in H_VALS:
            for hoo in H_VALS:
                if round(hss, 4) in hl and round(hoo, 4) in hl:
                    si, oi = hl.index(round(hss, 4)), hl.index(round(hoo, 4))
                    result[(hss, hoo)] = (float(d[key_s][oi, si]), float(d[key_o][oi, si]),
                                          'sweepE_G3x3_exact')
        if len(result) == len(H_VALS) ** 2:
            return result

    path = SWEEP_E_DIR / f'sweep_E_m{m_val}_fs{tag}_N{N_FIXED}.npz'
    if not path.exists():
        return result
    d    = np.load(path)
    grid = [round(float(v), 4) for v in d['hss_vals']]
    for hss in H_VALS:
        for hoo in H_VALS:
            if (hss, hoo) in result:
                continue
            if round(hss, 4) in grid and round(hoo, 4) in grid:
                si, oi = grid.index(round(hss, 4)), grid.index(round(hoo, 4))
                result[(hss, hoo)] = (float(d[key_s][oi, si]), float(d[key_o][oi, si]),
                                      'sweepE_11x11_exact')
            else:
                result[(hss, hoo)] = (_bilinear(grid, d[key_s], hss, hoo),
                                      _bilinear(grid, d[key_o], hss, hoo),
                                      'sweepE_11x11_interpolated')
    return result


def apply_transform(beta_s_p, beta_o_p, fs_val, m_val, lam_grid, mu_grid):
    """Network-edge model. Returns arrays shape (n_lam_s, n_lam_o, n_mu)."""
    alpha = 1.0 / (2 * m_val + 2)   # narcissistic=True: self + ~2m neighbours + broadcast
    lam_s = lam_grid[:, None, None]
    lam_o = lam_grid[None, :, None]
    mu    = mu_grid[None, None, :]
    bs    = beta_s_p * (1 - lam_s * alpha) + lam_s * alpha * (fs_val - mu)
    bo    = beta_o_p * (1 - lam_o * alpha) + lam_o * alpha * (fs_val - mu)
    return bs, bo, fs_val * bs + (1 - fs_val) * bo


def apply_transform_rescaled(beta_s_p, beta_o_p, fs_val, lam_grid, mu_grid):
    """Rescaled model (no degree). Returns arrays shape (n_lam_s, n_lam_o, n_mu)."""
    lam_s = lam_grid[:, None, None]
    lam_o = lam_grid[None, :, None]
    mu    = mu_grid[None, None, :]
    bs    = beta_s_p * (1 - lam_s) + lam_s * (fs_val - mu)
    bo    = beta_o_p * (1 - lam_o) + lam_o * (fs_val - mu)
    return bs, bo, fs_val * bs + (1 - fs_val) * bo


def fs_tag(fs_val):
    if abs(fs_val - 0.5) < 1e-4: return '0500'
    if abs(fs_val - 2/3) < 1e-4: return '0667'
    if abs(fs_val - 0.8) < 1e-4: return '0800'
    return f'{fs_val:.4f}'.replace('.', '')

def h_tag(h):
    return str(int(round(h * 1000)))


n_interp = 0
for m_val in M_VALS:
    for fs_val in FS_VALS:
        mu_grid = np.unique(np.append(MU_VALS_BASE, fs_val))

        use_sweep_F  = abs(fs_val - 0.5) > 1e-4
        pa_baselines = (load_sweep_F_baselines(m_val, fs_val)
                        if use_sweep_F
                        else load_sweep_E_baselines(m_val, fs_val, 'pa'))
        sbm_baselines = load_sweep_E_baselines(m_val, fs_val, 'sbm')

        for (hss, hoo), model, baselines in [
            *[((hss, hoo), 'pa',  pa_baselines)  for hss in H_VALS for hoo in H_VALS],
            *[((hss, hoo), 'sbm', sbm_baselines) for hss in H_VALS for hoo in H_VALS],
        ]:
            key = (hss, hoo)
            if key not in baselines:
                print(f'  SKIP {model} m={m_val} fs={fs_val:.4f} hss={hss} hoo={hoo}')
                continue

            beta_s_p, beta_o_p, source = baselines[key]
            n_interp += source.endswith('interpolated')
            stem = f'sweep_G_m{m_val}_fs{fs_tag(fs_val)}_hss{h_tag(hss)}_hoo{h_tag(hoo)}_{model}'

            bs, bo, bov = apply_transform(beta_s_p, beta_o_p, fs_val, m_val,
                                          LAMBDA_VALS, mu_grid)
            np.savez(OUT_DIR / f'{stem}.npz',
                     lambda_vals=LAMBDA_VALS, mu_vals=mu_grid,
                     h_ss=hss, h_oo=hoo, m=m_val, f_s=fs_val, model=model,
                     beta_s_peer=beta_s_p, beta_o_peer=beta_o_p, baseline_source=source,
                     beta_s_new=bs, beta_o_new=bo, beta_overall_new=bov)

            bs_r, bo_r, bov_r = apply_transform_rescaled(beta_s_p, beta_o_p, fs_val,
                                                          LAMBDA_VALS, mu_grid)
            np.savez(OUT_DIR / f'{stem}_rescaled.npz',
                     lambda_vals=LAMBDA_VALS, mu_vals=mu_grid,
                     h_ss=hss, h_oo=hoo, m=m_val, f_s=fs_val, model=model,
                     beta_s_peer=beta_s_p, beta_o_peer=beta_o_p, baseline_source=source,
                     beta_s_new=bs_r, beta_o_new=bo_r, beta_overall_new=bov_r)

        print(f'  m={m_val} fs={fs_val:.4f} done')

print(f'Done. {n_interp} baselines bilinearly interpolated from the 11x11 degree-matched grid'
      + (' (WARNING: run 10_degree_matched/submit_sweep_E_G3x3.sh for exact values).' if n_interp else '.'))
