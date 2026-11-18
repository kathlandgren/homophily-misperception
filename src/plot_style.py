"""plot_style.py — shared look for the SI figures, matched to main-text Figs 3 and 4.

Rules:
  * no bold anywhere; panel titles "A. Title", centered, regular weight
  * suptitle: what is shown and on which model, no parameter values
  * panel/row titles show only what varies
  * heatmap panels have the same physical size on the page as main Fig 3
    (1.478 in native panel in a figure placed so that 6.46 in -> 0.67\\linewidth)

Usage:
    S = _load("plot_style", "src/plot_style.py")   # importlib helper, see any make_*.py
    S.apply_rc()
    fig, axes, cax = S.heatmap_grid(nrows, ncols, suptitle=True, row_titles=True)
    S.panel_title(axes[0, 0], 'A', r'Overall misperception $\\beta_{\\mathrm{overall}}$')
    S.row_title(fig, axes[0], 'Supporter fraction $f_s=0.50$')
    S.suptitle(fig, 'Misperception on ...')
    S.save(fig, path)          # tight bbox; prints suggested \\includegraphics width
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

BLACK = '#222222'

# font sizes (main Fig 3, 03_analytical/make_figure_main.ipynb)
TITLE = 8        # panel titles
LABEL = 9        # axis labels
TICK = 8
ROW = 9          # row titles (Fig 3 uses its row titles at this level)
SUP = 10         # figure suptitle (one step above row titles)
ANNOT = 7        # contour labels, in-panel annotations

# geometry in inches (measured from figure_2.pdf: panel 1.478 in, pitch 2.051 in)
PANEL = 1.478
HGAP = 0.573     # pitch - panel
LEFT = 0.62      # y tick labels + y label
RIGHT = 0.30
ABOVE_PANEL = 0.24   # panel title
ROW_BLOCK = 0.26     # row title above the panel titles
BELOW_PANEL = 0.50   # x tick labels + x label
SUP_BLOCK = 0.36
CBAR_GAP = 0.10
CBAR_H = 0.11
CBAR_BELOW = 0.42    # colorbar tick labels + label

# reference for the tex width: figure_2.pdf is 6.46 in wide at 0.67\linewidth
REF_W, REF_TEX = 6.46, 0.67


def apply_rc():
    mpl.rcdefaults()
    plt.rc('font', size=TICK)
    plt.rc('axes', titlesize=TITLE, labelsize=LABEL, edgecolor=BLACK, labelcolor=BLACK,
           titleweight='normal', titlecolor=BLACK)
    plt.rc('xtick', labelsize=TICK, color=BLACK)
    plt.rc('ytick', labelsize=TICK, color=BLACK)
    plt.rc('legend', fontsize=ANNOT + 1, frameon=False)
    plt.rc('figure', titlesize=SUP, titleweight='normal')
    plt.rc('text', color=BLACK)
    plt.rc('pdf', fonttype=42)
    plt.rc('savefig', dpi=300)


def heatmap_grid(nrows, ncols, suptitle=True, row_titles=False, cbar=True,
                 panel=PANEL, hgap=HGAP, below=BELOW_PANEL, sup_lines=1):
    """Figure with square panels at absolute positions. Returns fig, axes (2-D), cax."""
    row_h = (ROW_BLOCK if row_titles else 0) + ABOVE_PANEL + panel + below
    top = (SUP_BLOCK + 0.17 * (sup_lines - 1)) if suptitle else 0.05
    bottom = (CBAR_GAP + CBAR_H + CBAR_BELOW) if cbar else 0.05
    W = LEFT + ncols * panel + (ncols - 1) * hgap + RIGHT
    H = top + nrows * row_h + bottom
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        y_top = H - top - r * row_h - (ROW_BLOCK if row_titles else 0) - ABOVE_PANEL
        for c in range(ncols):
            x = LEFT + c * (panel + hgap)
            axes[r, c] = fig.add_axes([x / W, (y_top - panel) / H, panel / W, panel / H])
    cax = None
    if cbar:
        span = ncols * panel + (ncols - 1) * hgap
        cw = min(span * 0.8, 5.2)
        x0 = LEFT + (span - cw) / 2
        y0 = CBAR_BELOW
        cax = fig.add_axes([x0 / W, y0 / H, cw / W, CBAR_H / H])
    fig._si_geom = dict(W=W, H=H, top=top, row_h=row_h, row_titles=row_titles, panel=panel)
    return fig, axes, cax


def panel_title(ax, letter, text='', **kw):
    s = f'{letter}. {text}' if text else f'{letter}.'
    ax.set_title(s, fontsize=TITLE, fontweight='normal', color=BLACK, loc='center', pad=4, **kw)


def row_title(fig, axes_row, text, pad_in=0.24):
    """Centered over a row of axes, just above the panel titles."""
    ax0, ax1 = axes_row[0], axes_row[-1]
    x = (ax0.get_position().x0 + ax1.get_position().x1) / 2
    H = fig.get_size_inches()[1]
    y = ax0.get_position().y1 + (pad_in + 0.06) / H
    fig.text(x, y, text, ha='center', va='bottom', fontsize=ROW, fontweight='normal', color=BLACK)


def suptitle(fig, text, axes=None):
    """Centered over the panels (not over the y labels). Use '\\n' to break a long title
    (and pass sup_lines=2 to heatmap_grid)."""
    if axes is not None:
        a = np.atleast_1d(axes).ravel()
        x = (min(ax.get_position().x0 for ax in a) + max(ax.get_position().x1 for ax in a)) / 2
    else:
        x = 0.5
    H = fig.get_size_inches()[1]
    fig.text(x, 1 - 0.12 / H, text, ha='center', va='top', fontsize=SUP, fontweight='normal',
             color=BLACK, linespacing=1.3)


def style_heatmap(ax, xlabel=None, ylabel=None, ticks=(0, 0.25, 0.5, 0.75, 1.0), lim=(0, 1),
                  fmt='{:.2f}'):
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks])
    ax.set_yticklabels([fmt.format(t) for t in ticks])
    if xlabel: ax.set_xlabel(xlabel, labelpad=2)
    if ylabel: ax.set_ylabel(ylabel, labelpad=2)
    for sp in ax.spines.values():
        sp.set_color(BLACK); sp.set_linewidth(0.8)
    ax.tick_params(colors=BLACK, length=3, pad=2)


def colorbar(fig, im, cax, label, ticks=None):
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=LABEL, color=BLACK, labelpad=3)
    cb.ax.tick_params(labelsize=TICK, colors=BLACK, length=3, pad=2)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.outline.set_edgecolor(BLACK)
    return cb


def tex_width(width_in, panel_in=PANEL):
    """\\includegraphics width (fraction of \\linewidth) that keeps main-Fig-3 font scale
    (and therefore main-Fig-3 panel size when panel_in == PANEL)."""
    return REF_TEX * width_in / REF_W


def save(fig, path, panel_in=PANEL):
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03)
    # report the tight width -> suggested tex width
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    bb = fig.get_tightbbox(FigureCanvasAgg(fig).get_renderer())
    w = bb.width + 0.06
    print(f'Saved {path}  ({w:.2f} in wide -> width={tex_width(w, panel_in):.2f}\\linewidth)')
    plt.close(fig)
