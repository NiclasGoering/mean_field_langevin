"""
SLIM plotting helper — drop-in module (no install needed)

Usage
-----
Place this file in a folder inside your repo, e.g. `my_project/plot_utils/slim_plotters.py`,
then import:

    from plot_utils.slim_plotters import (
        apply, color, PlotStyle, SciencePlotter, use_cycle, use_overrides
    )

    apply(theme="light")  # set rcParams + color cycle once per session
    plotter = SciencePlotter()
    fig = plotter.plot_series([
        (x1, y1, PlotStyle(color=color("blue"), linestyle='-', linewidth=2.0, marker='o', label='NN N=400')),
        (x2, y2, PlotStyle(color=color("purple"), linestyle='--', linewidth=2.0, marker='s', label='NTK')),
    ], xlabel=r'Training Set Size $m$', ylabel=r'Generalization error $\\mathcal{E}_{\\mathrm{gen}}$', legend_loc='lower left')
    fig.savefig('plot.png', dpi=300, bbox_inches='tight')

Everything lives in this single file: colors, theme apply, context managers, and a thin plotter.
"""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

__all__ = [
    "COLORS", "CYCLE", "color", "apply", "use_cycle", "use_overrides",
    "PlotStyle", "SciencePlotter",
]

# ---------------------------------------------------------------------------
# Named colors + default cycle (edit these once to set your look)
# ---------------------------------------------------------------------------
COLORS = {
    "purple":   "#8710FF",
    "blue":     "#01B5EB",
    "green":    "#96FFC0",
    "orange":   "#FFB463",
    "red":      "#d1001d",
    "greyblack":"#404040",
}

CYCLE = [
    COLORS["blue"], COLORS["red"], COLORS["purple"],
    COLORS["orange"], COLORS["green"], COLORS["greyblack"],
]


def color(name: str) -> str:
    return COLORS[name]


# ---------------------------------------------------------------------------
# One-liner theme apply (rcParams + color cycle)
# ---------------------------------------------------------------------------

def apply(*, theme: str = "light", serif: bool = True) -> None:
    """Apply consistent rcParams and set the global color cycle.

    Parameters
    ----------
    theme : str
        "light" or "dark".
    serif : bool
        Use Computer-Modern-like serif if True; system sans-serif if False.
    """
    plt.style.use("default")

    base = {
        # figure & save
        "figure.figsize": (3.3, 2.5),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # fonts & math
        "text.usetex": False,
        "axes.formatter.use_mathtext": True,
        "mathtext.fontset": "cm",
        # axes & spines
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.right": True,
        # lines & legend
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "legend.frameon": False,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.5,
        # grid (you can turn on per-axes via ax.grid(True))
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
    }

    if serif:
        base.update({
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        })
    else:
        base.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        })

    if theme == "dark":
        base.update({
            "figure.facecolor": "#111111",
            "axes.facecolor":   "#111111",
            "axes.edgecolor":   "#E0E0E0",
            "text.color":       "#E0E0E0",
            "axes.labelcolor":  "#E0E0E0",
            "xtick.color":      "#CCCCCC",
            "ytick.color":      "#CCCCCC",
            "grid.color":       "#E0E0E0",
        })
    else:
        base.update({
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor":   "#FFFFFF",
            "axes.edgecolor":   "#000000",
            "text.color":       "#000000",
            "axes.labelcolor":  "#000000",
            "xtick.color":      "#000000",
            "ytick.color":      "#000000",
            "grid.color":       "#000000",
        })

    mpl.rcParams.update(base)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=CYCLE)


@contextmanager
def use_cycle(colors: List[str]):
    """Temporarily override the color cycle within a `with` block."""
    old = mpl.rcParams.get("axes.prop_cycle")
    try:
        mpl.rcParams["axes.prop_cycle"] = cycler(color=list(colors))
        yield
    finally:
        mpl.rcParams["axes.prop_cycle"] = old


@contextmanager
def use_overrides(**rc):
    """Temporarily override any rcParams within a `with` block."""
    old = {k: mpl.rcParams.get(k) for k in rc}
    try:
        mpl.rcParams.update(rc)
        yield
    finally:
        for k, v in old.items():
            mpl.rcParams[k] = v


# ---------------------------------------------------------------------------
# Minimal, reusable plotter with good log-axis behavior
# ---------------------------------------------------------------------------

@dataclass
class PlotStyle:
    color: str
    linestyle: str
    linewidth: float
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: float = 1.0
    label: Optional[str] = None


class SciencePlotter:
    """Thin convenience layer to draw multiple (x,y) series with consistent style.
    Keeps your preferred log-axis ticks and legend defaults.
    """

    @staticmethod
    def configure_log_axes(ax: plt.Axes, *, xlog: bool = True, ylog: bool = True) -> None:
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.minorticks_on()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
            axis.set_major_locator(mpl.ticker.LogLocator(numticks=15))
            axis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(2, 10), numticks=15))
            axis.set_tick_params(which='both', direction='in')

    def plot_series(
        self,
        series: List[Tuple[np.ndarray, np.ndarray, PlotStyle]],
        *,
        figsize: Tuple[float, float] = (3.3, 2.5),
        xlabel: str = "",
        ylabel: str = "",
        legend_loc: str = 'upper right',
        legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
        legend_fontsize: Optional[float] = None,
        xlog: bool = True,
        ylog: bool = True,
        grid: bool = False,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        for x, y, st in series:
            ax.plot(
                x, y,
                color=st.color,
                linestyle=st.linestyle,
                linewidth=st.linewidth,
                marker=st.marker,
                markersize=st.markersize,
                alpha=st.alpha,
                label=st.label,
            )

        self.configure_log_axes(ax, xlog=xlog, ylog=ylog)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        if grid:
            ax.grid(True, which='major', linestyle='--', alpha=0.4)

        leg_kw = {
            'frameon': False,
            'loc': legend_loc,
            'handlelength': 1.0,
            'handletextpad': 0.5,
        }
        if legend_bbox_to_anchor is not None:
            leg_kw['bbox_to_anchor'] = legend_bbox_to_anchor
        if legend_fontsize is not None:
            leg_kw['fontsize'] = legend_fontsize
        if any(st.label for _, _, st in series):
            ax.legend(**leg_kw)

        plt.tight_layout()
        return fig
