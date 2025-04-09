# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
# ruff: noqa: E731
"""Functions for drawing contours of Dirichlet distributions."""

# Author: Thomas Boggs
# Modified by: Amanjit Singh Kainth

from functools import partial
from typing import Any, Optional
from collections.abc import Sequence
from pathlib import Path
from os import PathLike
import math
from math import sqrt, ceil, floor
import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp, random as jr, vmap
from jaxtyping import Array, PRNGKeyArray, Float

import math_utils as mu
import costs

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rcParams, ticker as mtick, tri
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.transforms import Bbox, Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection, LineCollection

import matplotlib.colors as mpc
import matplotlib.axes as mpa
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

import warnings

plt.set_loglevel("warning")
ColorMapLike = Optional[str | mpc.Colormap]
OptionalPath = Optional[str | bytes | PathLike]

get_cmap = mpl.colormaps.get_cmap
DEFAULT_CMAP = "viridis"
NUM_LEVELS = 8
IN_COLOR, OUT_COLOR, MAP_COLOR = "#1A254B", "#A7BED3", "#F2545B"
PATH_COLOR = [0.5, 0.5, 1, 0.1]


def custom_warning(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = custom_warning


def warn(msg):
    warnings.warn(msg, category=RuntimeWarning)


# simplex plotting
_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = np.array([_corners[np.roll(range(3), -i)[1:]] for i in range(3)])
_rng = np.random.default_rng(seed=0)


SPINE_COLOR = "black"


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.487 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 100
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + str(fig_height)
            + "so will reduce to"
            + str(MAX_HEIGHT_INCHES)
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES
    # , '\usepackage{amsmath, amsfonts}',
    params = {
        # "backend": "Qt5Agg",
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{amssymb,amsthm,amscd,empheq,amsmath}",
                r"\def\grad{\bm{\nabla}}",
                r"\def\primal{\mathcal{X}}",
                r"\def\dual{\mathcal{Y}}",
            ]
        ),
        "axes.labelsize": 12,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 14,
        "figure.titlesize": 16,
        "font.size": 10,  # was 10
        "legend.fontsize": 12,  # was 10
        "legend.title_fontsize": 14,  # was 10
        "legend.shadow": False,
        "legend.fancybox": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "font.serif": "cm",
        "mathtext.fontset": "cm",
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2,
        "lines.markersize": 5,
    }

    rcParams.update(params)


def format_axes(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    leg_loc=None,
    leg_title=None,
    leg_kwargs=None,
    nbins=None,
    grid=None,
    visible_spines: Optional[list[str] | str] = "lb",
    xlabel_kwargs=None,
    ylabel_kwargs=None,
    spine_placement: Optional[tuple[str, float] | str] = None,
):
    if visible_spines is None or visible_spines == "none":
        visible_spines = []
    elif visible_spines == "all":
        visible_spines = ["left", "bottom", "right", "top"]
    elif isinstance(visible_spines, str):
        visible_spines = [
            dict(l="left", b="bottom", r="right", t="top")[s]
            for s in set(visible_spines).intersection({"l", "b", "r", "t"})
        ]

    spine_placement = ("axes", 0) if spine_placement is None else spine_placement
    for loc, spine in ax.spines.items():
        if loc in visible_spines:
            if loc == "left" or loc == "bottom":
                spine.set_position(spine_placement)
            else:
                spine.set_position(("axes", 1))
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(0.7)
        else:
            spine.set_visible(False)  # don't draw spine

    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)

    ax.set(xlim=xlim, ylim=ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        if xlabel_kwargs is None:
            xlabel_kwargs = {}
        ax.set_xlabel(xlabel, labelpad=0.4, **xlabel_kwargs)
    if ylabel is not None:
        if ylabel_kwargs is None:
            ylabel_kwargs = {}
        ax.set_ylabel(ylabel, labelpad=3.0, **ylabel_kwargs)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if leg_loc is not None:
        if leg_kwargs is None:
            leg_kwargs = {}
        ax.legend(loc=leg_loc, title=leg_title, **leg_kwargs)
    if nbins is not None:
        ax.locator_params(axis="both", nbins=nbins, tight=True)
        # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nbins))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=nbins))
    if grid is not None:
        ax.grid(grid, lw=0.3)
    return ax


# From: https://stackoverflow.com/a/53586826
def multiple_formatter(denominator=2, number=np.pi, latex=r"\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return rf"${latex}$"
            elif num == -1:
                return rf"$-{latex}$"
            else:
                return rf"${num}{latex}$"
        else:
            if num == 1:
                return rf"$\frac{{{latex}}}{{{den}}}$"
            elif num == -1:
                return rf"$-\frac{{{latex}}}{{{den}}}$"
            else:
                return rf"$\frac{{{num}{latex}}}{{{den}}}$"

    return _multiple_formatter


# From: https://stackoverflow.com/a/53586826
class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex=r"\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return mtick.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return mtick.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )

    def set_xaxis(self, ax):
        ax.xaxis.set(major_locator=self.locator(), major_formatter=self.formatter())

    def set_yaxis(self, ax):
        ax.yaxis.set(major_locator=self.locator(), major_formatter=self.formatter())

    def set_axis(self, ax):
        self.set_xaxis(ax)
        self.set_yaxis(ax)


@partial(vmap, in_axes=(None, 0))
def tri_area(xy, pair):
    """Computes the area of a triangle.

    Arguments:

        `xy`: A length-2 sequence containing the x and y value.

        `pair`: A length-2x2 sequence of the two other points.
    """
    return 0.5 * jnp.linalg.norm(jnp.cross(*(pair - xy)))


@partial(vmap, in_axes=0)
def xy2bc(xy, tol=np.finfo(float).eps):
    """Converts 2D Cartesian coordinates to barycentric.

    Arguments:

        `xy`: A length-2 sequence containing the x and y value.
    """
    coords = tri_area(xy, _pairs) / _AREA
    return jnp.clip(coords, tol, 1.0 - tol)


@partial(jnp.vectorize, signature="(3)->(2)")
def bc2xy(bc):
    """Converts barycentric to 2D Cartesian coordinates.

    Arguments:

        `bc`: A length-3 sequence containing the barycentric coordinates.
    """
    return bc.dot(_corners)


def simplex_ticks(ax, start, stop, tick, n, offset=(0.0, 0.0), **kwargs):
    r = np.linspace(0, 1, n + 1)
    x = start[0] * (1 - r) + stop[0] * r
    x = np.vstack((x, x + tick[0]))
    y = start[1] * (1 - r) + stop[1] * r
    y = np.vstack((y, y + tick[1]))
    ax.plot(x, y, "k", lw=1)

    # add tick labels
    for xx, yy, rr in zip(x[1], y[1], r):
        ax.text(xx + offset[0], yy + offset[1], "{:.2}".format(rr), **kwargs)


def draw_arrow(ax, start, stop, offset=(0.0, 0.0), **kwargs):
    """Plot orientation arrows for simplex coordinates."""
    del offset
    ax.arrow(start[0], start[1], stop[0] - start[0], stop[1] - start[1], **kwargs)


def label_simplex(ax, p="p"):
    """Plots ticks on all sides."""
    left, right, top = _corners
    # define vectors for ticks
    # n, tick_size = 5, 0.02
    # bottom_tick = tick_size * (right - top) / n
    # right_tick = tick_size * (top - left) / n
    # left_tick = tick_size * (left - right) / n

    # kwargs = {"fontsize": 8}
    # simplex_ticks(ax, top, left, left_tick, n, offset=(-0.12, 0.0), **kwargs)
    # simplex_ticks(ax, left, right, bottom_tick, n, offset=(0, -0.06), **kwargs)
    # simplex_ticks(ax, right, top, right_tick, n, offset=(0.03, 0.0), **kwargs)

    kwargs = {
        "color": "black",
        "length_includes_head": True,
        "alpha": 0.5,
        "head_width": 0.025,
        "head_length": 0.05,
    }
    r3 = top[1] * 2
    yint = 0.1
    draw_arrow(ax, (0.4, r3 * 0.4 + yint), (0.05, r3 * 0.05 + yint), **kwargs)
    draw_arrow(ax, left + [0.2, -0.05], right + [-0.2, -0.05], **kwargs)
    draw_arrow(
        ax, (0.95, -r3 * 0.95 + r3 + yint), (0.6, -r3 * 0.6 + r3 + yint), **kwargs
    )

    kwargs = {"fontsize": 12, "ha": "center"}
    ax.text(0.1, 3**0.5 / 4, rf"${p}_0$", rotation=60, **kwargs)
    ax.text(0.5, -0.12, rf"${p}_1$", rotation=0, **kwargs)
    ax.text(0.9, 3**0.5 / 4, rf"${p}_2$", rotation=-60, **kwargs)


def add_colorbar(mappable, ax, **kwargs):
    """Add a colorbar to the plot."""
    cbf = mtick.ScalarFormatter()
    cbf.set_powerlimits((-1, 2))
    cb = plt.colorbar(
        mappable,
        ax=ax,
        # orientation="horizontal",
        location="bottom",
        format=cbf,
        ticks=mtick.MaxNLocator(nbins=5),
        fraction=0.046,
        pad=0.04,
        **kwargs,
    )
    return cb


def full_extent(fig, ax, w_pad=0.0, h_pad=0.0, pad=None):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles. From: https://stackoverflow.com/a/26432947."""
    if pad is not None:
        w_pad = pad
        h_pad = pad
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    axison = ax.axison
    if not axison:
        ax.set_axis_on()
    fig.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    if not axison:
        ax.set_axis_off()
    return bbox.expanded(1.0 + w_pad, 1.0 + h_pad)


def locate_colorbar_axis(fig):
    """Locate the colorbar axes in the figure."""
    # find the colorbar axes
    cax = None
    for ax in fig.axes[::-1]:
        if ax.get_label() == "<colorbar>":
            cax = ax
            break
    return cax


def get_extent(fig, ax, **kwargs):
    """Get the extent of an axes ax belonging to figure fig."""
    extent = full_extent(fig, ax, **kwargs).transformed(fig.dpi_scale_trans.inverted())
    return extent


def save_axis(fig, ax, filename, **kwargs):
    """Save the axes ax of figure fig to file named filename."""
    extent = get_extent(fig, ax, **kwargs)
    fig.savefig(filename, bbox_inches=extent)


def save_axes(fig, axes, filename, cbar=True, **kwargs):
    """Save a row/column/matrix of axes to file named filename."""
    extent = Bbox.union([get_extent(fig, ax, **kwargs) for ax in axes.flatten()])
    if cbar:
        cax = locate_colorbar_axis(fig)
        if cax is not None:
            cax_extent = get_extent(fig, cax, **kwargs)
            extent = Bbox.union([extent, cax_extent])
    fig.savefig(filename, bbox_inches=extent)


def make_figure(
    nrows: int,
    ncols: int,
    size: float | tuple[float, float] | None = None,
    size_full: bool = False,
    squeeze: bool = False,
    constrained_layout: bool = True,
):
    if isinstance(size, (int, float)):
        size = (size, size)

    if size is not None:
        if size_full:
            figsize = size
        else:
            figsize = (size[0] * ncols, size[1] * nrows)
    else:
        figsize = None

    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)
    subfigs = subfigs.squeeze(axis=1)
    axes = []
    for _, subfig in enumerate(subfigs):
        axes.append(
            subfig.subplots(
                nrows=1,
                ncols=ncols,
                sharex=True,
                sharey=True,
                # subplot_kw=dict(box_aspect=ncols / nrows),
                squeeze=False,
            )
        )
    axes = np.vstack(axes)
    if squeeze:
        axes = axes.squeeze()
    return fig, subfigs, axes


def simplex_grid(subdiv=8):
    """Create a 2-simplex grid."""
    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    p = xy2bc(jnp.c_[trimesh.x, trimesh.y])
    return p


def draw_simplex_contours(
    ax,
    func,
    cbar=True,
    coords=True,
    border=True,
    subdiv=8,
    nan_fill=False,
    debug=False,
    **kwargs,
):
    """Draws pdf contours over an equilateral triangle (2-simplex).

    Arguments:

        `func`: A function whose contours to draw.  Must take a single argument.

        `border` (bool): If True, the simplex border is drawn.

        `nlevels` (int): Number of contours to draw.

        `subdiv` (int): Number of recursive mesh subdivisions to create.

        kwargs: Keyword args passed on to `plt.triplot`.
    """
    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    p = xy2bc(jnp.c_[trimesh.x, trimesh.y])
    pvals = func(p)
    if debug and jnp.isnan(pvals).any().item():
        warn("Some values are NaN.")
        print(pvals[jnp.isnan(pvals)])
        print(p[jnp.isnan(pvals)])
    if debug and not jnp.isfinite(pvals).all().item():
        idx = ~jnp.isfinite(pvals)
        warn("Some values are not finite.")
        print(pvals[idx])
        print(p[idx])
    if nan_fill:
        pvals = jnp.nan_to_num(pvals)
    filled = kwargs.pop("filled", False)
    if filled:
        cs = ax.tricontourf(trimesh, pvals, **kwargs)
    else:
        cs = ax.tricontour(trimesh, pvals, **kwargs)
    # cs._A = []
    if cbar:
        add_colorbar(cs, ax)
    if border is True:
        ax.triplot(_triangle, c="black", linewidth=1)
    if coords:
        label_simplex(ax)
    ax.axis("off")
    return ax


def setup_simplex(ax, p="p"):
    ax.triplot(_triangle, c="black", linewidth=1)
    if p:
        label_simplex(ax, p=p)
    ax.axis("off")


def plot_simplex(ax, X, *args, scatter=True, barycentric=True, **kwargs):
    """Plots a set of points in the simplex.

    Arguments:

        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.

        `barycentric` (bool): Indicates if `X` is in barycentric coords.

        kwargs: Keyword args passed on to `plt.plot`.
    """
    if barycentric:
        X = X.dot(_corners)
    lines = plot_2dpc(ax, X, *args, scatter=scatter, **kwargs)
    ax.axis("off")
    return lines


def get_marginal_axes(ax):
    if ax.child_axes and len(ax.child_axes) == 2:
        ax_x, ax_y = ax.child_axes
    else:
        ax_x = ax.inset_axes([0, 0.99, 1, 0.2], sharex=ax)
        ax_y = ax.inset_axes([0.99, 0, 0.2, 1], sharey=ax)
    return ax_x, ax_y


def square_grid(lb=-5.0, ub=5.0, n=100):
    """Create a 2d square grid"""
    x = jnp.linspace(lb, ub, n)
    y = jnp.linspace(lb, ub, n)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.c_[X.ravel(), Y.ravel()]
    return x, y, X, Y, Z


def dynamic_lims(X, xlim=None, ylim=None, bw=(0, 0), cut: float = 3):
    """Compute dynamic axis limits for contour plots etc."""
    if xlim is None:
        xlim = (-math.inf, math.inf)
    if ylim is None:
        ylim = (-math.inf, math.inf)
    minx, miny = jnp.min(X, axis=0)
    maxx, maxy = jnp.max(X, axis=0)
    nxlim = (
        max(floor(minx - bw[0] * cut), xlim[0]),
        min(ceil(maxx + bw[0] * cut), xlim[1]),
    )
    nylim = (
        max(floor(miny - bw[1] * cut), ylim[0]),
        min(ceil(maxy + bw[1] * cut), ylim[1]),
    )
    return nxlim, nylim


def draw_scatter(
    ax,
    X,
    density=None,
    kde=False,
    kde_kw=None,
    sample=True,
    sample_indices=None,
    scatter_kw=None,
    format_kw=None,
    is_simplex=False,
    xlim=None,
    ylim=None,
    cut=3,
    coverage=1.0,
    marginals=True,
    **contour_kw,
):
    """Plot KDE contours of a function over a square grid."""
    assert X.ndim == 2, "X must be 2d"
    # don't do anything for 1d distributions
    if X.shape[1] == 1:
        return X

    if X.shape[1] > 2:
        X_pca, _ = mu.pca(X, 2)
        draw_scatter(
            ax,
            X_pca,
            density=None,
            kde=kde,
            kde_kw=kde_kw,
            sample=sample,
            sample_indices=sample_indices,
            scatter_kw=scatter_kw,
            format_kw=format_kw,
            is_simplex=False,
            xlim=None,
            ylim=None,
            cut=cut,
            coverage=coverage,
            marginals=marginals,
            **contour_kw,
        )

    # for scatter plots of samples
    if scatter_kw is None:
        scatter_kw = {}
    scatter_kw["scatter"] = True
    scatter_kw["alpha"] = scatter_kw.get("alpha", 0.2) if sample else 0

    if ylim is None:
        ylim = xlim

    if sample_indices is None:
        plot_X = X
    else:
        plot_X = X[np.asarray(sample_indices)]

    if is_simplex:
        if density is not None:
            draw_simplex_contours(
                ax,
                lambda p: density(mu.p_to_s(p)),
                cbar=False,
                coords=False,
                border=False,
                subdiv=8,
                **contour_kw,
            )
        if sample:
            plot_simplex(
                ax,
                mu.s_to_p(plot_X),
                barycentric=True,
                **scatter_kw,
            )
    else:
        if kde or density is not None:
            if density is None or xlim is None:
                if kde_kw is None:
                    kde_kw = {}
                dist = mu.KDE(X, **kde_kw)
                xlim, ylim = dynamic_lims(X, bw=dist.bandwidth, cut=cut)
                if density is None:
                    density = vmap(dist.prob)
            draw_2dcontours(
                ax,
                density,
                coverage=coverage,
                xlim=xlim,
                ylim=ylim,
                marginals=marginals,
                **contour_kw,
            )
        if sample:
            plot_2dpc(ax, plot_X, **scatter_kw)
        if format_kw is not None:
            format_axes(
                ax,
                **dict(
                    format_kw,
                    **{
                        "xlim": xlim,
                        "ylim": ylim,
                        "xticks": xlim,
                        "yticks": ylim,
                        "visible_spines": "all",
                        "spine_placement": ("axes", 0),
                    },
                ),
            )
    return X


def draw_2dcontours(
    ax,
    func,
    coverage=1.0,
    marginals=True,
    cbar=False,
    lb=-5.0,
    ub=5.0,
    xlim=None,
    ylim=None,
    nan_fill=False,
    debug=False,
    gridsize=100,
    **kwargs,
):
    """Plot 2d contours of a function over a square grid."""
    filled = kwargs.pop("filled", False)
    if xlim is None:
        xlim = (lb, ub)
    if ylim is None:
        ylim = xlim

    x = jnp.linspace(xlim[0], xlim[1], gridsize)
    y = jnp.linspace(ylim[0], ylim[1], gridsize)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.c_[X.ravel(), Y.ravel()]
    Z = func(grid)
    if 0 < coverage < 1:
        vmin = jnp.nanpercentile(Z, 100 * (1 - coverage))
        vmax = jnp.nanpercentile(Z, 100 * coverage)
        # Z = jnp.clip(Z, vmin, vmax)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        levels = kwargs.pop("levels", 20 if filled else 12)
        kwargs["levels"] = np.linspace(vmin, vmax, levels)

    if debug and not jnp.isfinite(Z).all().item():
        idx = ~jnp.isfinite(Z)
        warn("Some values are not finite.")
        print(Z[idx])
        print(grid[idx])
    if nan_fill:
        Z = jnp.nan_to_num(Z, nan=0.0, posinf=0.0)
    Z = Z.reshape(X.shape)
    if jnp.isnan(Z).any():
        warn(f"{func} contains NaNs on grid {xlim} x {ylim}")
    # pop color of marginals before calling contour
    color = kwargs.pop("color", get_cmap(kwargs.get("cmap", DEFAULT_CMAP))(0.6))
    if filled:
        cs = ax.contourf(X, Y, Z, **kwargs)
    else:
        cs = ax.contour(X, Y, Z, **kwargs)
    retval = (cs,)
    if cbar:
        retval = retval + (add_colorbar(cs, ax),)
    # plot marginals
    if marginals:
        fargs = dict(color=color, alpha=0.4)
        # print(f"{jnp.isnan(Z).any()=}")
        # print(f"{jnp.where(jnp.isnan(Z))=}")
        # create axis for the marginals over grid
        ax_x, ax_y = get_marginal_axes(ax)
        retval = retval + (ax_x.fill_between(x, jnp.nansum(Z, axis=0), **fargs),)
        retval = retval + (ax_y.fill_betweenx(y, jnp.nansum(Z, axis=-1), **fargs),)
        ax_x.axis("off")
        ax_y.axis("off")
    return retval


def draw_1ddensity(
    ax,
    func,
    lb=-5.0,
    ub=5.0,
    xlim=None,
    debug=False,
    gridsize=10000,
    **kwargs,
):
    """Plot 1d density of a function over a grid."""
    filled = kwargs.pop("filled", False)
    if xlim is None:
        xlim = (lb, ub)

    x = jnp.linspace(xlim[0], xlim[1], gridsize)
    y = func(x)
    if debug and not jnp.isfinite(y).all().item():
        idx = ~jnp.isfinite(y)
        warn("Some values are not finite.")
        print(y[idx])
        print(x[idx])
    fargs = dict(
        # linewidth=kwargs.pop("linewidth", 1),
        color=kwargs.get("color", get_cmap(kwargs.get("cmap", DEFAULT_CMAP))(0.6)),
        # facecolor=kwargs.pop("c", "C0"),
        # edgecolor=None,
        alpha=0.4,
    )
    if jnp.isnan(y).any():
        warn(f"{func} contains NaNs on grid {xlim}")
    ax.plot(x, y, **kwargs)
    if filled:
        ax.fill_between(x, y, **fargs)
    return


def plot_2dpc(ax, X, *args, scatter=True, **kwargs):
    """Plot pointcloud in 2d."""
    if scatter:
        # contour plots have zorder=1, and scatter plots even
        # if draw after contours will be behind them
        kwargs["zorder"] = kwargs.get("zorder", 2)
        return ax.scatter(X[:, 0], X[:, 1], *args, **kwargs)
    else:
        return ax.plot(X[:, 0], X[:, 1], *args, **kwargs)


def convolve(signal, kernel, stride: int | None = None):
    kernel_size = len(kernel)
    if stride is None:
        stride = kernel_size
    output_size = (len(signal) - kernel_size) // stride + 1
    output = np.zeros(output_size)

    for i in range(output_size):
        start = i * stride
        end = start + kernel_size
        output[i] = np.sum(signal[start:end] * kernel)
    return output


def add_arrow(
    line,
    sample_indices: Sequence[int] | None = None,
    indices: Sequence[int] | None = None,
    positions: Sequence[float] | Sequence[tuple[float, float]] | None = None,
    num_arrows: int | None = None,
    direction="right",
    size=15,
    color=None,
):
    """https://stackoverflow.com/a/34018322
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    assert (
        positions is not None
        or num_arrows is not None
        or indices is not None
        or sample_indices is not None
    )
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if sample_indices is not None:
        sample_indices = np.asarray(sample_indices)  #  pyright: ignore
        near_indices = convolve(sample_indices, np.asarray([0.25, 0.75]), 1)
        positions = tuple(xdata[int(idx)] for idx in near_indices)
        return add_arrow(
            line, positions=positions, direction=direction, size=size, color=color
        )
    elif indices is not None:
        indices = np.asarray(indices, dtype=int)  #  pyright: ignore
    elif num_arrows is not None:
        num_points = len(xdata)
        window_size = max(1, num_points // num_arrows - 1)
        positions = convolve(xdata, np.ones(window_size) / window_size)
        return add_arrow(
            line, positions=positions, direction=direction, size=size, color=color
        )
    else:
        assert positions is not None
        indices = []
        for position in positions:
            if isinstance(position, tuple):
                x, *_ = position
            else:
                x = position
            # find closest index
            indices.append(np.argmin(np.absolute(xdata - x)))
        indices = np.asarray(indices, dtype=int)  #  pyright: ignore

    assert indices is not None

    for end_ind in indices:
        if direction == "right":
            start_ind = end_ind - 1
        else:
            start_ind = end_ind + 1

        line.axes.annotate(
            "",
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            size=size,
        )


def compute_lims(
    X: Float[Array, "N 2"],
    margin=0.05,
    x_margin: Optional[float] = None,
    y_margin: Optional[float] = None,
    ax: Optional[mpa.Axes] = None,
):
    if x_margin is None:
        x_margin = margin
    if y_margin is None:
        y_margin = margin
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    x_eps = x_margin * (x_max - x_min)
    y_eps = y_margin * (y_max - y_min)
    x_lim = (x_min - x_eps, x_max + x_eps)
    y_lim = (y_min - y_eps, y_max + y_eps)
    if ax is not None:
        ax.set(xlim=x_lim, ylim=y_lim)
    return x_lim, y_lim


def color_prop_cycle(cmap, n_colors, ax=None, start=0.0, end=1.0):
    colormap = get_cmap(cmap)
    if cmap == "gist_rainbow":
        colormap = colormap[::-1]
    colors = colormap(np.linspace(start, end, n_colors))
    cyc = cycler("color", colors)
    if ax is not None:
        ax.set_prop_cycle(cyc)
    return cyc


def linear_cmap(colors, nodes=None, name="mycmap"):
    """Create a colormap from a list of colors.
    https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#directly-creating-a-segmented-colormap-from-a-list
    """
    if nodes is None:
        params = colors
    else:
        assert len(nodes) == len(colors)
        params = list(zip(nodes, colors))
    return LinearSegmentedColormap.from_list(name, params)


def unicmap(color, alpha=0.0):
    return linear_cmap([to_rgba(color, alpha), color])


def nested_cmaps(cmap, n_cmaps):
    """Return n_cmaps cmaps from cmap.
    Each cmap runs from white to a color from cmap.
    """
    colors = cmap(np.linspace(0, 1, n_cmaps))
    return [LinearSegmentedColormap.from_list("cmap", ["white", c]) for c in colors]


def create_legend_handles(labels, colors=None, cmaps=None, bar=True, **kwargs):
    """Create manual handles for a legend."""
    if colors is None and cmaps is None:
        raise ValueError("Either colors or cmaps must be specified.")
    if colors is not None:
        return [
            mpatches.Patch(color=color, label=label, **kwargs)
            if bar
            else mlines.Line2D([], [], color=color, label=label, **kwargs)
            for label, color in zip(labels, colors)
        ]
    elif cmaps is not None:
        colors = [get_cmap(cmap)(0.0) for cmap in cmaps]
        return create_legend_handles(labels, colors=colors, bar=bar, **kwargs)


def visualize_transport(
    trainer, state, n: Optional[int] = None, density: bool = False, fname=None
):
    """Visualize Bregman transport for one step."""
    filled = False
    nlevels = 20 if filled else 12

    bregman = trainer.bregman
    simplex = bregman.domain == "simplex"
    # assert state.jko_step == 1
    source, target, *_ = trainer.sample_trajectory(n, jko_state=state)
    assert source.shape == target.shape
    source_dual = bregman.to_dual(source)
    target_dual = bregman.to_dual(target)
    assert source_dual.shape == source.shape
    do_pca = source.shape[-1] > 2
    if do_pca:
        # visualize PCA in higher dimensions
        source, source_dual, *_ = mu.mirror_pca(bregman, source, n_components=2)
        target, target_dual, *_ = mu.mirror_pca(bregman, target, n_components=2)

    scatter_kw = dict(alpha=0.3, ms=3, marker="o")
    primal_range = bregman.primal_range
    dual_range = bregman.dual_range
    if simplex:
        pplot_fn = partial(plot_simplex, barycentric=True, **scatter_kw)
        contour_fn = partial(
            draw_simplex_contours,
            cbar=False,
            coords=False,
            border=False,
            subdiv=8,
            levels=nlevels,
            filled=filled,
            nan_fill=False,
            debug=False,
        )
        source = mu.s_to_p(source)
        target = mu.s_to_p(target)
    else:
        pplot_fn = partial(plot_2dpc, **scatter_kw)
        contour_fn = partial(
            draw_2dcontours,
            marginals=True,
            cbar=False,
            lb=primal_range[0],
            ub=primal_range[1],
            levels=nlevels,
            filled=filled,
            debug=False,
        )
    # dual coordinates scatter plot
    dplot_fn = partial(plot_2dpc, **scatter_kw)

    latexify()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    pformat_kw = dict(
        # xlabel=r"$\theta_1$",
        # ylabel=r"$\theta_2$",
        xticks=primal_range,
        yticks=primal_range,
        xlim=primal_range,
        ylim=primal_range,
        nbins=2,
        visible_spines="all",
        # xlabel_kwargs=dict(x=0.9),
        # ylabel_kwargs=dict(y=0.9),
        spine_placement=("axes", 0),
    )
    dformat_kw = dict(
        # xlabel=r"$\theta_1$",
        # ylabel=r"$\theta_2$",
        # xticks=dual_range,
        # yticks=dual_range,
        xlim=dual_range,
        ylim=dual_range,
        nbins=3,
        visible_spines="all",
        # xlabel_kwargs=dict(x=0.9),
        # ylabel_kwargs=dict(y=0.9),
        spine_placement=("axes", 0),
    )

    # primal coordinates
    pplot_fn(axes[0, 0], source, c="C0")
    axes[0, 0].set_title(r"$\mu$")
    pplot_fn(axes[0, 1], target, c="C1")
    axes[0, 1].set_title(r"$(\grad\Omega^{-1}\circ T)_{\#}\mu$")
    if not simplex:
        for ax in axes[0, :]:
            format_axes(ax, **pformat_kw)

    # if simplex:
    # plot_fn = partial(plot_simplex, barycentric=True, **scatter_kw)
    #     plot_fn(axes[0, 0], mu.s_to_p(source), c="C0")
    #     axes[0, 0].set_title(r"$\mu$")
    #     plot_fn(axes[0, 1], mu.s_to_p(target), c="C1")
    #     axes[0, 1].set_title(r"$(\grad\Omega^{-1}\circ T)_{\#}\mu$")
    # else:
    #     plot_2dpc(axes[0, 0], source, c="C0", **scatter_kw)
    #     format_axes(axes[0, 0], title=r"$\mu$", **pformat_kw)
    #     plot_2dpc(axes[0, 1], target, c="C1", **scatter_kw)
    #     format_axes(
    #         axes[0, 1], title=r"$(\grad\Omega^{-1}\circ T)_{\#}\mu$", **pformat_kw
    #     )

    # dual coordinates
    dplot_fn(axes[1, 0], source_dual, c="C0")
    format_axes(axes[1, 0], title=r"$(\grad\Omega)_{\#}\mu$", **dformat_kw)
    dplot_fn(axes[1, 1], target_dual, c="C1", **scatter_kw)
    format_axes(axes[1, 1], title=r"$T_{\#}\mu$", **dformat_kw)

    plt.tight_layout()
    if fname is not None:
        plt.savefig(trainer.save_path / fname)
    else:
        plt.show()


def animate_jko(
    trainer,
    n: Optional[int] = None,
    metrics: Optional[Array] = None,
    dual: bool = False,
    kde: bool = False,
    filled: bool = False,
    sample: bool = False,
    palette: Any = "blend:C0,C1",
    paper: bool = True,
    paper_steps: int = 3,
):
    """Plot the trajectory of particles on the simplex or 2d space."""
    if math.prod(trainer.event_shape) == 1:
        # 1-dimensional distribution
        return animate_1d_jko(trainer, n, metrics, dual, kde, filled, sample, palette)
    # plotting contours
    # filled = False
    nlevels = 20 if filled else 12
    coverage = 1.0

    dt = trainer.delta
    bregman = trainer.bregman
    pi2_ticks = Multiple(2, np.pi, r"\pi")
    primal = trainer.sample_trajectory(n)
    # dual = bregman.to_dual(primal)
    assert primal.ndim == 3, "primal must be a 3d array (batch trajectory)"

    latexify()
    subplot_kw = dict(box_aspect=1)
    nrows = 2 if dual else 1
    if paper:
        # init, target and intermediate densities
        ncols = paper_steps + 2
    else:
        ncols = 3
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(3 * ncols, 3 * nrows),
        subplot_kw=subplot_kw,
        layout="constrained",
        sharey="row",
        sharex="row",
    )
    if dual:
        if paper:
            pax, dax = axes[:, 1:-1]
        else:
            pax, dax = axes[:, 1]
        mirrored = bregman.to_dual(primal)
    else:
        if paper:
            pax = axes[1:-1]
        else:
            pax = axes[1]
        dax = None
        mirrored = primal

    # plot initial and target density on the side
    if dual:
        if paper:
            it_axes = axes[:, np.r_[0, -1]]
        else:
            it_axes = axes[:, 0::2]
    else:
        if paper:
            it_axes = axes[np.r_[0, -1]]
        else:
            it_axes = axes[0::2]
    visualize_distributions(
        trainer, axes=it_axes, dual=dual, filled=filled, sample=sample
    )

    do_pca = primal.shape[-1] > 2
    simplex = bregman.domain == "simplex" and not do_pca
    primal_range = bregman.primal_range
    dual_range = bregman.dual_range
    if simplex:
        plot_fn = partial(plot_simplex, barycentric=True)
        contour_fn = partial(
            draw_simplex_contours,
            cbar=False,
            coords=False,
            border=False,
            subdiv=8,
            levels=nlevels,
            filled=filled,
            nan_fill=True,
            debug=False,
        )
        primal = mu.s_to_p(primal)
    else:
        plot_fn = plot_2dpc
        contour_fn = partial(
            draw_scatter,
            coverage=coverage,
            marginals=True,
            cbar=False,
            xlim=bregman.soft_pr if not do_pca else None,
            levels=nlevels,
            filled=filled,
            debug=False,
        )

    # colormap to interpolate between for trajectory particles
    cmap = sns.color_palette(palette, as_cmap=True)
    n_steps = primal.shape[0]
    colors = cmap(np.linspace(0, 1, n_steps))
    # kdeplot_kw = dict(fill=False, linewidth=3.0, levels=10, alpha=0.7)
    scatter_kw = dict(alpha=0.1 if sample else 0, ms=3, marker="o")
    pformat_kw = dict(
        xticks=primal_range if not do_pca else None,
        yticks=primal_range if not do_pca else None,
        xlim=primal_range if not do_pca else None,
        ylim=primal_range if not do_pca else None,
        # nbins=2,
        visible_spines="all",
        spine_placement=("axes", 0),
    )
    dformat_kw = dict(
        xticks=dual_range if not do_pca else None,
        yticks=dual_range if not do_pca else None,
        xlim=dual_range if not do_pca else None,
        ylim=dual_range if not do_pca else None,
        # nbins=5,
        visible_spines="all",
        spine_placement=("axes", 0),
    )

    # format axes once
    # if not simplex:
    #     pax.autoscale(enable=False)
    #     format_axes(pax, title=r"$\rho_t$", **pformat_kw)
    # else:
    #     pax.set_title(r"$\rho_t$")

    # if dual and dax is not None:
    #     dax.autoscale(enable=False)
    #     format_axes(dax, title=r"$(\grad\Omega)_{\#}\rho_t$", **dformat_kw)

    def pax_clear(pax: mpa.Axes):
        if simplex:
            pax.clear()
        else:
            pax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            for artist in pax.lines + pax.collections:
                artist.remove()
            for ax in pax.get_children():
                if isinstance(ax, mpa.Axes):
                    ax.clear()

    def dax_clear(dax: Optional[mpa.Axes]):
        if dax is not None:
            dax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            for artist in dax.lines + dax.collections:
                artist.remove()
            for ax in dax.get_children():
                if isinstance(ax, mpa.Axes):
                    ax.clear()

    def init_func(pax: mpa.Axes, dax: Optional[mpa.Axes]):
        if "arctan" in bregman.name:
            pi2_ticks.set_axis(pax)
        if not simplex:
            format_axes(pax, **pformat_kw)
        pax.autoscale(enable=False)
        if dual and dax is not None:
            if bregman.name == "riemanntanh":
                pi2_ticks.set_axis(dax)
            format_axes(dax, **dformat_kw)
            dax.autoscale(enable=False)
        return

    def update(pax: mpa.Axes, dax: Optional[mpa.Axes], i: int):
        primal_batch = primal.at[i].get()
        c = tuple(colors[i])
        cur_cmap = sns.light_palette(c, as_cmap=True)
        pax.clear()  # clear previous frame
        if kde:
            if simplex:
                contour_fn(
                    pax,
                    vmap(mu.KDE(primal_batch).prob),
                    # _kde_density(primal_batch),
                    # _mirror_kde_density(bregman, primal_batch, mu.p_to_s),
                    # _mirror_kde_density(costs.SimplexILR(), primal_batch, mu.p_to_s),
                    cmap=cur_cmap,
                )
            else:
                primal_batch = contour_fn(pax, primal_batch, cmap=cur_cmap)
        elif do_pca:
            primal_batch = mu.pca(primal_batch, 2)
        artists = plot_fn(pax, primal_batch, c=c, **scatter_kw)
        if metrics is None:
            title = rf"$\rho_t\ (t = {i * dt:.2f})$"
        else:
            title = (
                r"$\mathrm{{KL}}(\rho_{{t}} \| \rho_{{\infty}})"
                + rf"= {metrics[i]:.3f}\ "
                + rf"(t = {i * dt:.2f})$"
            )
        pax.set_title(title, color=c)
        # if not simplex:
        #     format_axes(pax, **pformat_kw)
        if dual and dax is not None:
            dual_batch = mirrored.at[i].get()
            dax.clear()  # clear previous frame
            if kde:
                dual_batch = draw_scatter(
                    dax,
                    dual_batch,
                    coverage=coverage,
                    cmap=cur_cmap,
                    marginals=True,
                    cbar=False,
                    xlim=bregman.soft_dr if not do_pca else None,
                    levels=nlevels,
                    filled=filled,
                    debug=False,
                )
            elif do_pca:
                dual_batch = mu.pca(dual_batch, 2)
            plot_2dpc(dax, dual_batch, c=c, **scatter_kw)
            dax.set_title(r"$(\grad\Omega)_{\#}\rho_t$")
            # format_axes(dax, title=r"$(\grad\Omega)_{\#}\rho_t$", **dformat_kw)
        init_func(pax, dax)
        return artists

    if paper:
        n = n_steps
        k = paper_steps
        indices = list(reversed(range(n - 1, n // k - 1, -(n // k))))
        indices = indices[-k:]
        for i in range(len(pax)):
            if dax is not None:
                update(pax[i], dax[i], indices[i])
            else:
                update(pax[i], None, indices[i])
        # fig.tight_layout()

        def save_fn(fname, **kwargs):
            del kwargs
            save_name = trainer.save_path / fname
            save_name = save_name.with_suffix(save_name.suffix + ".pdf")
            fig.savefig(save_name)
    else:
        ani = FuncAnimation(
            fig=fig,
            func=update,
            frames=list(range(0, n_steps, n_steps // 10)),  # pyright: ignore
            init_func=init_func,  # pyright: ignore
            interval=10,
            repeat=True,
        )

        def save_fn(fname, dpi=200, fps=2.5):
            save_name = (trainer.save_path / fname).with_suffix(".gif")
            ani.save(save_name, writer=PillowWriter(fps=fps), dpi=dpi)

    return save_fn


def animate_1d_jko(
    trainer,
    n: Optional[int] = None,
    metrics: Optional[Array] = None,
    dual: bool = False,
    kde: bool = False,
    filled: bool = False,
    sample: bool = False,
    palette: Any = "blend:C0,C1",
):
    del metrics, sample, kde
    dt = trainer.delta
    bregman = trainer.bregman
    pi2_ticks = Multiple(2, np.pi, r"\pi")
    primal = trainer.sample_trajectory(n)
    # dual = bregman.to_dual(primal)
    assert primal.ndim == 3, "primal must be a 3d array (batch trajectory)"
    assert primal.shape[-1] == 1, "primal must be a 1d distribution sample"
    num_steps = primal.shape[0]
    simplex = bregman.domain == "simplex"

    latexify()
    nrows = 2 if dual else 1
    fig = plt.figure(figsize=plt.figaspect(nrows))
    # fig = plt.figure(figsize=(5, 5 * nrows))
    gs = GridSpec(nrows, 1, figure=fig)
    gs.update(hspace=-0.1)
    pax = fig.add_subplot(gs[0], projection="3d")
    if dual:
        dax = fig.add_subplot(gs[1], projection="3d")
        mirrored = jnp.squeeze(bregman.to_dual(primal), -1)

    primal = jnp.squeeze(primal, -1)

    colors = sns.color_palette(palette, as_cmap=True)(np.linspace(0, 1, num_steps))
    low, high = bregman.soft_pr
    px = np.linspace(low, high, 1000)
    low, high = bregman.soft_dr
    dx = np.linspace(low, high, 1000)

    ipx = px.reshape(-1, 1)
    pax.plot(px, 0, trainer.init_distribution.prob(ipx), c=colors[0])
    pax.plot(px, num_steps, trainer.targ_distribution.prob(ipx), c=colors[-1])
    pax.autoscale(axis="z", tight=True)
    pax.autoscale(enable=False, axis="z")

    if dual:
        init = bregman.dual_dist(trainer.init_distribution)
        targ = bregman.dual_dist(trainer.targ_distribution)
        idx = dx.reshape(-1, 1)
        dax.plot(dx, 0, init.prob(idx), c=colors[0])
        dax.plot(dx, num_steps, targ.prob(idx), c=colors[-1])
        dax.autoscale(axis="z", tight=True)
        dax.autoscale(enable=False, axis="z")

    for t in range(num_steps):
        c = colors[t]
        prob_px = mu.KDE(primal[t]).prob(px)
        pax.plot(px, t, prob_px, c=c)
        if filled:
            pax.add_collection3d(
                pax.fill_between(px, prob_px, color=c, alpha=0.5), zs=t, zdir="y"
            )
        if dual:
            prob_dx = mu.KDE(mirrored[t]).prob(dx)
            dax.plot(dx, t, prob_dx, c=c)
            if filled:
                dax.add_collection3d(
                    dax.fill_between(dx, prob_dx, color=c, alpha=0.5), zs=t, zdir="y"
                )
    axes = [
        pax,
    ]
    if dual:
        axes.append(dax)

    label_kw = dict(labelpad=-15)

    for ax in axes:
        ax.grid(False)
        ax.set_ylabel("$t$", **label_kw)
        ax.set_yticks([0, num_steps - 1, num_steps])
        ax.set_yticklabels(["$0$", f"${num_steps - 1}$", r"$\infty$"])
        ax.set_zlim(bottom=0)
        # ax.locator_params(axis="z", nbins=2)
        ax.set_zticks([])
        ax.view_init(elev=30, azim=-45, roll=0)
        # ax.set_box_aspect(None, zoom=0.85)

    if "arctan" in bregman.name:
        pi2_ticks.set_xaxis(pax)
    pax.set_xticks(bregman.primal_range)
    pax.set_xlabel("$x$", **label_kw)
    pax.set_zlabel(r"$p(x,t)$", **label_kw)
    pax.tick_params(direction="in", length=0, width=0, pad=-5)
    if dual:
        if bregman.name == "riemanntanh":
            pi2_ticks.set_xaxis(dax)
        dax.set_xticks(bregman.dual_range)
        dax.set_xlabel("$y$", **label_kw)
        dax.set_zlabel(r"$p(y,t)$", **label_kw)
        dax.tick_params(direction="in", length=0, width=0, pad=-5)

    # fig.subplots_adjust(left=-0.11)  # plot outside the normal area
    fig.subplots_adjust(left=0, right=0.95, bottom=0, top=1)
    # fig.tight_layout()

    def save_fn(fname, dpi=200):
        save_name = (trainer.save_path / fname).with_suffix(".png")
        fig.savefig(save_name, dpi=dpi)

    return save_fn


def visualize_1d_distributions(
    trainer, axes=None, dual=True, filled=False, sample=False, fname=None
):
    """Visualize initial/target distributions."""
    bregman = trainer.bregman
    primal_range = bregman.primal_range
    dual_range = bregman.dual_range
    eps = bregman.eps
    pi2_ticks = Multiple(2, np.pi, r"\pi")
    event_shape = trainer.event_shape

    # initial and target distributions
    init_primal = trainer.init_distribution
    targ_primal = trainer.targ_distribution
    ipd = vmap(init_primal.prob)
    idd = vmap(bregman.dual_dist(init_primal).prob)
    tpd = vmap(targ_primal.prob)
    tdd = vmap(bregman.dual_dist(targ_primal).prob)

    fargs = dict(alpha=0.5)

    pformat_kw = dict(
        xticks=primal_range,
        xlim=primal_range,
        visible_spines="all",
        spine_placement=("axes", 0),
    )
    dformat_kw = dict(
        xticks=dual_range,
        xlim=dual_range,
        visible_spines="all",
        spine_placement=("axes", 0),
    )

    def format_1d(ax, format_kw: dict):
        format_axes(ax, **format_kw)
        ax.set_ylim(bottom=0)
        ax.locator_params(axis="y", nbins=5)

    def format_primal(ax, **kwargs):
        if "arctan" in bregman.name:
            pi2_ticks.set_xaxis(ax)
        kwargs.update(pformat_kw)
        format_1d(ax, kwargs)

    def format_dual(ax, **kwargs):
        if bregman.name == "riemanntanh":
            pi2_ticks.set_xaxis(ax)
        kwargs.update(dformat_kw)
        format_1d(ax, kwargs)

    latexify()
    nrows = 2 if dual else 1
    if axes is None:
        new_fig = True
        _, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            squeeze=False,
            figsize=(6, 3 * nrows),
            sharey="row",
        )
    else:
        new_fig = False
        assert axes.size in (2, 4)
        dual = axes.size == 4
        if dual:
            axes = axes.reshape((2, 2))
        else:
            axes = axes.reshape((1, 2))

    x = jnp.linspace(primal_range[0] + eps, primal_range[1] - eps, 1000)
    x = jnp.reshape(x, (-1, 1))  # make it 2d as (n_samples, 1)
    axes[0, 0].plot(x.squeeze(), ipd(x), c="C0")
    axes[0, 1].plot(x.squeeze(), tpd(x), c="C1")
    if filled:
        axes[0, 0].fill_between(x.squeeze(), ipd(x), color="C0", **fargs)
        axes[0, 1].fill_between(x.squeeze(), tpd(x), color="C1", **fargs)
    format_primal(axes[0, 0])
    format_primal(axes[0, 1])
    if dual:
        y = jnp.linspace(dual_range[0] + eps, dual_range[1] - eps, 1000)
        y = jnp.reshape(y, (-1, 1))  # make it 2d as (n_samples, 1)
        axes[1, 0].plot(y.squeeze(), idd(y), c="C0")
        axes[1, 1].plot(y.squeeze(), tdd(y), c="C1")
        if filled:
            axes[1, 0].fill_between(y.squeeze(), idd(y), color="C0", **fargs)
            axes[1, 1].fill_between(y.squeeze(), tdd(y), color="C1", **fargs)
        format_dual(axes[1, 0], title=r"$\rho_0$")
        format_dual(axes[1, 1], title=r"$\rho_{\infty}$")
    else:
        axes[0, 0].set_title(r"$\rho_0$")
        axes[0, 1].set_title(r"$\rho_{\infty}$")

    if new_fig:
        plt.tight_layout()
    if fname is not None:
        plt.savefig(trainer.save_path / fname)


def visualize_distributions(
    trainer, axes=None, dual=True, filled=False, sample=False, fname=None
):
    """Visualize initial/target distributions."""
    bregman = trainer.bregman
    simplex = bregman.domain == "simplex"
    pi2_ticks = Multiple(2, np.pi, r"\pi")

    # filled = False
    nlevels = 20 if filled else 12
    if math.prod(trainer.event_shape) == 1:
        # 1-dimensional distribution
        return visualize_1d_distributions(trainer, axes, dual, filled, sample, fname)
    (dim,) = trainer.event_shape  # assume event_shape is one-dimensional

    do_pca = dim > 2
    simplex = simplex and not do_pca

    scatter_kw = dict(alpha=0.1 if sample else 0, ms=3, marker="o")
    primal_range = bregman.primal_range
    dual_range = bregman.dual_range

    if simplex:
        pplot_fn = partial(plot_simplex, barycentric=True, **scatter_kw)
        pcontour_fn = partial(
            draw_simplex_contours,
            cbar=False,
            coords=False,
            border=False,
            subdiv=8,
            levels=nlevels,
            filled=filled,
            nan_fill=True,
            debug=False,
        )
        # density_transform = mu.p_to_s
    else:
        pplot_fn = partial(plot_2dpc, **scatter_kw)
        pcontour_fn = partial(
            draw_scatter,
            xlim=bregman.soft_pr if not do_pca else None,
            ylim=None,
            marginals=True,
            cbar=False,
            levels=nlevels,
            filled=filled,
            debug=False,
        )
        # density_transform = None

    # dual coordinates scatter plot
    dplot_fn = partial(plot_2dpc, **scatter_kw)
    dcontour_fn = partial(
        draw_scatter,
        xlim=bregman.soft_dr if not do_pca else None,
        ylim=None,
        marginals=True,
        cbar=False,
        levels=nlevels,
        filled=filled,
        debug=False,
    )
    pformat_kw = dict(
        xticks=primal_range if not do_pca else None,
        yticks=primal_range if not do_pca else None,
        xlim=primal_range if not do_pca else None,
        ylim=primal_range if not do_pca else None,
        # nbins=2,
        visible_spines="all",
        spine_placement=("axes", 0),
    )
    dformat_kw = dict(
        xticks=dual_range if not do_pca else None,
        yticks=dual_range if not do_pca else None,
        xlim=dual_range if not do_pca else None,
        ylim=dual_range if not do_pca else None,
        # nbins=3,
        visible_spines="all",
        spine_placement=("axes", 0),
    )

    # sample points from distributions
    X_init = trainer.init_distribution.sample(
        seed=trainer.key, sample_shape=(trainer.n_eval_samples,)
    )
    X_targ = trainer.targ_distribution.sample(
        seed=trainer.key, sample_shape=(trainer.n_eval_samples,)
    )
    Y_init = bregman.to_dual(X_init)
    Y_targ = bregman.to_dual(X_targ)
    # derive the dual coordinates
    if do_pca:
        init_primal_density = None
        init_dual_density = None
        targ_primal_density = None
        targ_dual_density = None
    else:
        if simplex:
            init_primal_density = mu.OverparamSimplex(trainer.init_distribution).prob
            targ_primal_density = mu.OverparamSimplex(trainer.targ_distribution).prob
        else:
            init_primal_density = vmap(trainer.init_distribution.prob)
            targ_primal_density = vmap(trainer.targ_distribution.prob)

        init_dual_density = vmap(bregman.dual_dist(trainer.init_distribution).prob)
        targ_dual_density = vmap(bregman.dual_dist(trainer.targ_distribution).prob)

    if simplex:
        # put in barycentric coordinates
        X_init = mu.s_to_p(X_init)
        X_targ = mu.s_to_p(X_targ)

    latexify()
    nrows = 2 if dual else 1
    if axes is None:
        new_fig = True
        _, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            squeeze=False,
            figsize=(6, 3 * nrows),
            sharey="row",
            sharex="row",
        )
    else:
        new_fig = False
        assert axes.size in (2, 4)
        dual = axes.size == 4
        if dual:
            axes = axes.reshape((2, 2))
        else:
            axes = axes.reshape((1, 2))

    if simplex:
        pcontour_fn(axes[0, 0], init_primal_density, cmap="Blues")
        pcontour_fn(axes[0, 1], targ_primal_density, cmap="Oranges")
    else:
        X_init = pcontour_fn(axes[0, 0], X_init, init_primal_density, cmap="Blues")
        X_targ = pcontour_fn(axes[0, 1], X_targ, targ_primal_density, cmap="Oranges")
    pplot_fn(axes[0, 0], X_init, c="C0")
    pplot_fn(axes[0, 1], X_targ, c="C1")
    if "arctan" in bregman.name:
        pi2_ticks.set_axis(axes[0, 0])
        pi2_ticks.set_axis(axes[0, 1])
    if not simplex:
        format_axes(axes[0, 0], **pformat_kw)
        format_axes(axes[0, 1], **pformat_kw)

    if dual:
        Y_init = dcontour_fn(axes[1, 0], Y_init, init_dual_density, cmap="Blues")
        Y_targ = dcontour_fn(axes[1, 1], Y_targ, targ_dual_density, cmap="Oranges")
        dplot_fn(axes[1, 0], Y_init, c="C0")
        dplot_fn(axes[1, 1], Y_targ, c="C1")
        if bregman.name == "riemanntanh":
            pi2_ticks.set_axis(axes[1, 0])
            pi2_ticks.set_axis(axes[1, 1])
        format_axes(axes[1, 0], title=r"$\rho_0$", **dformat_kw)
        format_axes(axes[1, 1], title=r"$\rho_{\infty}$", **dformat_kw)
    else:
        axes[0, 0].set_title(r"$\rho_0$")
        axes[0, 1].set_title(r"$\rho_{\infty}$")

    if new_fig:
        plt.tight_layout()
    if fname is not None:
        plt.savefig(trainer.save_path / fname)


def plot_kelly_portfolios(
    deltas,
    weights,
    ax: Optional[mpa.Axes] = None,
    fname: OptionalPath = None,
    **format_kw,
):
    """Plot the Kelly GOP weights, one for each lambda in deltas.

    Args:
        deltas: array of deltas
        weights: array of weights
        format_kw: kwargs for format_axes
    """
    assert deltas.ndim == 1
    assert weights.ndim == 2
    assert weights.shape[0] == len(deltas)
    D = weights.shape[1]
    # max 10 distinct colors
    cmap = get_cmap("tab10")
    cum_weights = jnp.cumsum(weights, axis=1)
    latexify()
    # handles = []
    new_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    last = 0
    for i in range(D):
        ax.plot(deltas, cum_weights[:, i], c=cmap(i))
        ax.fill_between(deltas, last, cum_weights[:, i], color=cmap(i), alpha=0.5)
        last = cum_weights[:, i]
        # handles.append(mpatches.Patch(color=cmap(i), label=f"Stock {i+1}"))
    # ax.legend(handles=handles)
    format_kw["title"] = format_kw.get("title", "Allocation of Assets")
    ax.set_xscale("log")
    format_axes(
        ax,
        xlabel=r"$\delta$",
        ylabel=r"$\mathbf{w}$",
        yticks=np.linspace(0, 1, 5),
        xlim=(deltas.min(), deltas.max()),
        ylim=(0, 1),
        spine_placement=("axes", 0),
        **format_kw,
    )
    if new_fig:
        plt.tight_layout()
    if fname is not None:
        fname = Path(fname)
        plt.savefig(fname.parent / (fname.name + ".pdf"))


def plot_distribution(
    dist: mu.dx.Distribution,
    seed: int | Array,
    kde: bool = False,
    ax: Optional[mpa.Axes] = None,
    fname: OptionalPath = None,
    **kwargs,
):
    """Plot distribution density.
    kwargs passed on to draw_scatter
    """
    n = kwargs.pop("n", 10000)
    X, lp_X = dist.sample_and_log_prob(seed=seed, sample_shape=(n,))
    p_X = jnp.exp(lp_X)
    kde_kw = dict(weights=p_X)

    if ax is None:
        new_fig = True
        _, ax = plt.subplots(figsize=(3, 3))
    else:
        new_fig = False

    if kde:
        dist = mu.KDE(X, **kde_kw)
    draw_scatter(ax, X, vmap(dist.prob), kde=kde, kde_kw=kde_kw, **kwargs)
    if new_fig:
        plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def plot_transport(
    batch: dict[str, Any],
    num_points: Optional[int] = None,
    forward: bool = True,
    overlap: bool = True,
    rng: Optional[PRNGKeyArray] = None,
    source_label: str = "source",
    target_label: str = "target",
    map_label: str = "OT",
    marker_size: float = 5,
    margin: float = 0.1,
):
    """Plot samples from the source and target measures.

    If source samples mapped by the fitted map are provided in ``batch``,
    the function plots these predictions as well.
    """
    rng = jax.random.key(0) if rng is None else rng
    if num_points is None:
        subsample = jnp.arange(len(batch["source"]))
    else:
        # only unique if plotting transport map
        replace = "mapped_source" not in batch
        subsample = jr.choice(
            rng, a=len(batch["source"]), shape=(num_points,), replace=replace
        )
    source = batch["source"][subsample]
    target = batch["target"][subsample]

    if overlap:
        fig, ax = plt.subplots(figsize=(4, 4))
        sax = tax = ax
    else:
        fig, (sax, tax) = plt.subplots(
            1,
            2,
            figsize=(8, 4),
            gridspec_kw={"wspace": 0, "hspace": 0},
        )

    if forward:
        label_transport = rf"{map_label}({source_label})"
        source_color, target_color = "#1A254B", "#A7BED3"
    else:
        label_transport = rf"{map_label}({target_label})"
        source_color, target_color = "#A7BED3", "#1A254B"
    push_color = "#F2545B"

    scatter_kwargs = dict(s=marker_size, alpha=0.5)
    sax.scatter(
        source[:, 0],
        source[:, 1],
        label=source_label,
        color=source_color,
        **scatter_kwargs,
    )
    tax.scatter(
        target[:, 0],
        target[:, 1],
        label=target_label,
        color=target_color,
        **scatter_kwargs,
    )
    plot_push = "push" in batch
    if plot_push:
        push = batch["push"][subsample]
        ax = tax if forward else sax
        ax.autoscale(enable=False)
        ax.scatter(
            push[:, 0],
            push[:, 1],
            label=label_transport,
            color=push_color,
            **scatter_kwargs,
        )
        # plot OT map if overlapping
        if overlap:
            initial = source if forward else target
            U, V = (push - initial).T
            tax.quiver(
                initial[:, 0],
                initial[:, 1],
                U,
                V,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                headwidth=1,
                headlength=0,
                headaxislength=0,
                color=PATH_COLOR,
            )
            sax.set_axis_off()
    legend_ncol = 2 + (1 if plot_push else 0)
    if overlap:
        # ax.set_title("Source and Target")
        ax.legend(
            **{
                "ncol": legend_ncol,
                "loc": "upper center",
                "bbox_to_anchor": (0.5, -0.05),
            }
        )
        compute_lims(jnp.concat((source, target), axis=0), margin=margin, ax=ax)
    else:
        sax.set_title("Source")
        tax.set_title("Target")
        sax.set_axis_off()
        tax.set_axis_off()
        axison = sax.axison and tax.axison
        fig.legend(
            **{
                "ncol": legend_ncol,
                "loc": "outside lower center",
                "bbox_to_anchor": (0.5, -0.05),
                # "loc": "upper center",
                # "bbox_to_anchor": (0.5, 0.05),
            }
        )
        compute_lims(source, margin=margin, ax=sax)
        compute_lims(target, margin=margin, ax=tax)

    fig.tight_layout()
    return fig


class BOTVisualizer(object):
    def __init__(
        self,
        bregman: costs.AbstractBregman,
        data: dict[str, Any],
        primal: bool = True,
        marker_size: float = 7,
    ):
        self.bregman = bregman
        self.data = data
        self.primal = primal
        self.size = marker_size

    @property
    def ddi(self) -> bool:
        return self.data.get("ddi", True)

    def fetch(self, key, primal: Optional[bool] = None):
        if primal is None:
            space = "primal" if self.primal else "dual"
        else:
            space = "primal" if primal else "dual"
        return self.data[key].get(space)

    def source(self, primal: Optional[bool] = None):
        return self.fetch("source", primal)

    def target(self, primal: Optional[bool] = None):
        return self.fetch("target", primal)

    def push(self, primal: Optional[bool] = None):
        return self.fetch("push", primal)

    def pull(self, primal: Optional[bool] = None):
        return self.fetch("pull", primal)

    def map_label(self, forward: bool):
        if self.ddi:
            label = r"Df" if forward else r"Df^*"
        else:
            label = r"Dh" if forward else r"Dh^*"
        return rf"$({label})_{{\#}}$"

    @staticmethod
    def interpolate(
        A: Float[Array, "N D"], B: Float[Array, "N D"], num_interp: int
    ) -> Float[Array, "{num_interp} N D"]:
        """Interpolate between A and B.
        Precondition: A and B have the same shape.
        """
        T = mu.linear_interpolation(num_interp)
        C = jnp.stack((A, B))
        return jnp.einsum("n2,2ND->nND", T, C)

    def plot_samples(self, forward: bool = True):
        if self.ddi:
            kw = dict(
                source_label=r"$\mu_0^{\primal}$", target_label=r"$\mu_1^{\dual}$"
            )
        else:
            kw = dict(
                source_label=r"$\mu_0^{\dual}$", target_label=r"$\mu_1^{\primal}$"
            )
        primal = self.ddi
        source = self.source(primal=primal)
        target = self.target(primal=not primal)
        kw["map_label"] = self.map_label(forward)
        kw["marker_size"] = self.size
        batch = dict(source=source, target=target)
        push = self.push if forward else self.pull
        primal = forward != self.ddi
        batch["push"] = push(primal=primal)
        fig = plot_transport(batch, forward=forward, overlap=False, **kw)
        return fig

    def plot_interpolation(
        self,
        forward: bool = True,
        num_interp: int = 20,
        ax: Optional[mpa.Axes] = None,
        margin: float = 0.1,
        paths: bool = True,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure

        # source and target in fixed coordinates (primal or dual)
        source = self.source()
        target = self.target()

        power = 3
        num_t = num_interp**power
        step_t = (num_t - 1) // (num_interp - 1)

        # push_t shape (num_t, num_samples, 2)
        inputs = self.source if forward else self.target
        pushed = self.push if forward else self.pull
        primal = forward != self.ddi
        A, B = inputs(primal=primal), pushed(primal=primal)
        C = self.interpolate(A, B, num_t)
        if primal != self.primal:
            if self.primal:
                C = self.bregman.to_primal(C)
            else:
                C = self.bregman.to_dual(C)
        push_t = C
        if forward:
            source_color, target_color = IN_COLOR, OUT_COLOR
        else:
            source_color, target_color = OUT_COLOR, IN_COLOR

        s = self.size
        # ax.autoscale(enable=False)
        if paths:
            ax.scatter(push_t[-1, :, 0], push_t[-1, :, 1], s=s, color=MAP_COLOR)
            lines = LineCollection(jnp.unstack(push_t, axis=1), colors=PATH_COLOR)
            ax.add_collection(lines)
        else:
            # cmap = unicmap(MAP_COLOR, 0.0)
            in_color = to_rgba(IN_COLOR, 0.1)
            out_color = to_rgba(OUT_COLOR, 0.1)
            cmap = sns.blend_palette([in_color, out_color], as_cmap=True)
            colors = cmap(np.linspace(0, 1, num_interp))
            for t_idx, t in enumerate(range(0, num_t, step_t)):
                ax.scatter(push_t[t, :, 0], push_t[t, :, 1], s=s, color=colors[t_idx])
        # plot source and target on top
        ax.scatter(source[:, 0], source[:, 1], s=s, color=source_color)
        ax.scatter(target[:, 0], target[:, 1], s=s, color=target_color)
        inputs = jnp.concat((source, target), axis=0)
        compute_lims(inputs, margin=margin, ax=ax)
        ax.set_axis_off()

        return fig


def plot_series(
    returns: Optional[pd.DataFrame] = None,
    log_returns: Optional[pd.DataFrame] = None,
    w: Optional[pd.DataFrame | pd.Series] = None,
    cmap="tab20",
    n_colors=20,
    height=6,
    width=10,
    ax=None,
    title="Historical Compounded Cumulative Returns",
    legend=True,
):
    r"""
    Create a chart with the compounded cumulative of the portfolios.

    Parameters
    ----------
    returns : DataFrame of shape (n_samples, n_assets)
        Assets returns DataFrame, where n_samples is the number of
        observations and n_assets is the number of assets.
    w : DataFrame or Series of shape (n_assets, 1)
        Portfolio weights, where n_assets is the number of assets.
    cmap : cmap, optional
        Colorscale used to plot each portfolio compounded cumulative return.
        The default is 'tab20'.
    n_colors : int, optional
        Number of distinct colors per color cycle. If there are more assets
        than n_colors, the chart is going to start to repeat the color cycle.
        The default is 20.
    height : float, optional
        Height of the image in inches. The default is 6.
    width : float, optional
        Width of the image in inches. The default is 10.
    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib axis
        Returns the Axes object with the plot for further tweaking.

    Example
    -------
    ::

        ax = rp.plot_series(returns=Y, w=ws, cmap="tab20", height=6, width=10, ax=None)

    .. image:: images/Port_Series.png

    """

    assert returns is not None or log_returns is not None
    if log_returns is not None:
        returns = np.expm1(log_returns).dropna()  # pyright: ignore

    # number of samples, number of assets
    N, D = returns.shape  # pyright: ignore

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if w is None:
        # uniform weights
        w = pd.Series(np.ones(D) / D, index=returns.columns.copy())

    if not isinstance(w, pd.DataFrame):
        if isinstance(w, pd.Series):
            w_ = w.to_frame()
        else:
            raise ValueError("w must be a DataFrame or Series.")
    else:
        w_ = w.copy()

    if returns.columns.tolist() != w_.index.tolist():
        if returns.columns.tolist() == w_.index.tolist():
            w_ = w_.T
        else:
            raise ValueError("returns and w must have the same assets.")

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()

    ax.grid(linestyle=":")
    if title:
        ax.set_title(title)

    labels = w_.columns.tolist()
    index = returns.index.tolist()

    colormap = get_cmap(cmap)
    colormap = colormap(np.linspace(0, 1, int(n_colors)))

    if cmap == "gist_rainbow":
        colormap = colormap[::-1]

    cycle = plt.cycler("color", colormap)
    ax.set_prop_cycle(cycle)

    for i in range(len(labels)):
        a = np.array(returns, ndmin=2) @ np.array(w_[labels[i]], ndmin=2).T
        prices = 1 + np.insert(a, 0, 0, axis=0)
        prices = np.cumprod(prices, axis=0)
        prices = np.ravel(prices).tolist()
        del prices[0]

        ax.plot(index, prices, "-", label=f"{labels[i]:.0e}")

    # ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=None, minticks=5, maxticks=10))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # ax.yaxis.set_major_formatter("{x:3.2f}")
    format_returns_axis(ax)
    if legend:
        ax.legend(title=r"$\varepsilon$", loc="center left", bbox_to_anchor=(1, 0.5))

    return ax


def format_xaxis_date(ax, fmt="%Y-%m", minticks=5, maxticks=10):
    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator(tz=None, minticks=minticks, maxticks=maxticks)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def format_returns_axis(ax, **kwargs):
    """Format returns axis for plotting data."""
    format_xaxis_date(ax, *kwargs)
    ax.yaxis.set_major_formatter("{x:3.2f}")


if __name__ == "__main__":
    from subprocess import run

    def make_visualizer(dset: str, cost: costs.AbstractBregman, ddi: bool):
        prefix = "ddi" if ddi else "pdi"
        fname = Path("neural").joinpath(f"{prefix}_{dset}_{cost.name}_summary.npy")
        if not fname.exists():
            raise ValueError
        data = jnp.load(fname, allow_pickle=True).item()
        return BOTVisualizer(cost, data)

    def plot(visualizer: BOTVisualizer, forward: bool, **kw):
        # forward = visualizer.ddi
        visualizer.plot_interpolation(forward, **kw)
        return visualizer.plot_samples(forward)

    labels = dict()
    cost_fns = []
    cost_fns.append(costs.Euclidean())
    labels[cost_fns[-1].name] = r"\frac{1}{2 }(x^i)^2"
    cost_fns.append(costs.ExtendedKL(a=1.0).dualized())
    labels[cost_fns[-1].name] = r"\exp(x^i)"
    cost_fns.append(costs.ExtendedKL(a=-1.0).dualized())
    labels[cost_fns[-1].name] = r"\exp(-x^i)"
    cost_fns.append(costs.HNNTanh(beta=1.0).dualized())
    labels[cost_fns[-1].name] = r"\frac{1}{2} \log(1 + \exp(2x^i))"

    def name(
        cost: Optional[costs.AbstractBregman],
        dset: str,
        ddi: Optional[bool],
        forward: bool,
        paths: bool,
        pdf: bool = False,
    ):
        info = (
            (("ddi" if ddi else "pdi") if ddi is not None else "mixed")
            + "_"
            + ("forward" if forward else "inverse")
        )
        suffix = "pdf" if pdf else "png"
        if cost is None:
            fname = f"{dset}_{info}_{'paths' if paths else 'samples'}"
        else:
            fname = f"{dset}_{info}_{cost.name}"
        return Path("neural").joinpath(f"{fname}.{suffix}")

    def plot_all(dset: str, ddi: bool, forward: bool, paths: bool, pdf: bool = False):
        visualizers = []
        for cost_fn in cost_fns:
            visualizers.append(make_visualizer(dset, cost_fn, ddi=ddi))
        latexify()
        fig, axes = plt.subplots(
            1,
            len(cost_fns),
            figsize=(4 * len(cost_fns), 4),
        )
        for ax, visualizer in zip(axes, visualizers):
            label = labels[visualizer.bregman.name]
            title = rf"$\Omega_i(x^i) = {label}$"
            sfig = plot(visualizer, forward=forward, ax=ax, paths=paths)
            # sfig.legends[0].set_title(title)
            # sfig.suptitle(title)
            fname = name(visualizer.bregman, dset, ddi, forward, paths, pdf=pdf)
            sfig.savefig(fname, bbox_inches="tight")
            plt.close(sfig)
            ax.set_title(title)

        if ddi:
            if forward:
                label = r"$\mu_t^{\primal} = (D\Omega^* \circ ((1-t)D\Omega + tDf))_{\#} \mu_0^{\primal}$"
            else:
                label = (
                    r"$\mu_t^{\primal} = (tD\Omega^* + (1-t)Df^*)_{\#} \mu_1^{\dual}$"
                )
        else:
            if forward:
                label = r"$\mu_t^{\primal} = ((1-t)D\Omega + tDh)_{\#} \mu_0^{\dual}$"
            else:
                label = r"$\mu_t^{\primal} = (D\Omega^* \circ (tD\Omega + (1-t)Dh^*))_{\#} \mu_1^{\primal}$"
        if paths:
            handles = create_legend_handles(
                [label], colors=[PATH_COLOR], bar=False, lw=3, alpha=0.5
            )
            fig.legend(
                handles,
                [label],
                loc="outside lower center",
                fontsize=24,
                frameon=False,
                bbox_to_anchor=(0.5, -0.15),
            )
        fig.tight_layout()
        fname = name(None, dset, ddi, forward, paths, pdf=pdf)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        # plt.show()

    def merge(f1, f2, out, pdf=False, n=5):
        if pdf:
            # return run(["pdfunite", f1, f2, out], capture_output=True, text=True)
            commands = [
                # ["pdfunite", f1, f2, out],
                # ["pdfxup", "-x", "1", "-y", str(n), "-l", "0", "-o", out, out],
                ["pdfjam", "--nup", "1x2", "--outfile", out, "--", f1, f2],
                ["pdfcrop.sh", out, out],
            ]
            for command in commands:
                run(command, capture_output=True, text=True)
            # return run(["pdfunite", f1, f2, out], capture_output=True, text=True)
        else:
            return run(
                ["convert", "-append", f1, f2, out], capture_output=True, text=True
            )

    PDF = True
    COMBINE = False

    DSETS = ["demo", "rout_2"]

    for dset in DSETS:
        for ddi in [True, False]:
            for forward in [True, False]:
                for paths in [True, False]:
                    plot_all(dset, ddi, forward, paths, pdf=PDF)

    if COMBINE:
        for dset in DSETS:
            for forward in [True, False]:
                for paths in [True, False]:
                    f1 = name(None, dset, True, forward, paths, pdf=PDF)
                    f2 = name(None, dset, False, forward, paths, pdf=PDF)
                    out = name(None, dset, None, forward, paths, pdf=PDF)
                    merge(f1, f2, out, pdf=PDF)
                for cost in cost_fns:
                    f1 = name(cost, dset, True, forward, paths, pdf=PDF)
                    f2 = name(cost, dset, False, forward, paths, pdf=PDF)
                    out = name(cost, dset, None, forward, paths, pdf=PDF)
                    merge(f1, f2, out, pdf=PDF)
