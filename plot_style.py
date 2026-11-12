r"""Style shared by every figure: palette, fonts, sizes, layout helpers, and saving.

Defines the color palette, matplotlib rcParams, marker and bar styles, and panel
layout helpers used across the plotting scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# step number of the formula (1 to 7) -> label under the axis of a figure,
# short enough to be set horizontally (mathtext for the pulses; EA idling, PM
# idling, and FL idling are the error-accumulation idling, the parity-mapping
# idling, and the feedback latency, as in the caption of the figure of the cycle)
STEP_LABELS = dict(enumerate(
    ["EA\nidling", r"$\hat X_{\pi/2}$", "PM\nidling", r"$\hat X_{-\pi/2}$",
     "meas.", "FL\nidling", r"$\hat X_{\pi}$"], start=1))
# the group of a bar chart by mechanism that holds the small mechanisms together
OTHER_MECHANISMS = "other mechanisms"
# Names of the groups of a bar chart by mechanism, under its axis in two rows:
# a short name of the mechanism, and under it the mode that a group of
# mechanisms belongs to, with a thin line that spans the group. The legends
# keep the full names. Mode -> its first and last mechanism; the distances of
# the line and of the top of the name of the mode below the axis (pt)
MECHANISM_TICK_LABELS = {
    "all mechanisms": "total", "ancilla relaxation": "relax.", "ancilla excitation": "excit.",
    "ancilla dephasing": "deph.", "photon loss": "loss", "photon gain": "gain",
    OTHER_MECHANISMS: "other"}
MECHANISM_MODES = {"ancilla": ("ancilla relaxation", "ancilla dephasing"),
                   "resonator": ("photon loss", "photon gain")}
MODE_LINE_OFFSET, MODE_TEXT_OFFSET = 13.5, 15.0
# Every color of the manuscript, name -> hex. This dict is the only place
# where a color value is written: the constants below and the TikZ colors of
# ``figs/style/palette.tex`` are defined from it. The meaning of a color
# depends on the category of the panel:
#   diagrams: neglected process = blue, failure state and the transitions into
#     it = rosewood, leakage subspace = parchment, everything kept by the model = ink;
#   logical manifestation: leakage = parchment, X_L = citron, Y_L = olive,
#     Z_L = forest;
#   mechanism: see ``MECHANISM_COLORS`` and ``TERM_COLORS``;
#   parameter set: see ``POINT_STYLES``;
#   grays: guides and quantities without a color of their own.
PALETTE = {
    "ink": "#000000",
    "blue": "#4477AA",
    "light_blue": "#A3C1E0",
    "indigo": "#332288",
    "green": "#1B7837",
    "light_green": "#8FD694",
    "orange": "#EE7733",
    "yellow": "#E3B505",
    "rose": "#CC6677",
    "cyan": "#66CCEE",
    "red": "#CC3311",
    "mauve": "#B48DA8",
    # the logical manifestations: colors sampled from Claude Monet, Woman with
    # a Parasol (1875), from the pale sky down to the dark grass
    "parchment": "#E9DDBA",
    "citron": "#BCB536",
    "olive": "#717814",
    "forest": "#1F4429",
    # same painting, TikZ trees: umber = actual tree
    "umber": "#85600E",
    # TikZ figures: rosewood = failure state and the transitions into it; in
    # Fig. 1, clay (and a light tint of it) and rosewood shade failure by step
    # (darker = later step); oxblood is not used there
    "oxblood": "#632C26",
    "rosewood": "#8D4F46",
    "clay": "#B37066",
    # the mechanisms: colors sampled from Claude Monet, Poppy Field (1873);
    # poppy and earth tones for the ancilla mechanisms, the blues of the sky
    # for the resonator mechanisms, cream and straw for the mechanisms of the
    # analytical formula that the budget merges
    "poppy": "#9E4122",
    "terracotta": "#C97E57",
    "dark_olive": "#34321D",
    "steel_blue": "#4D6D81",
    "haze_blue": "#90B9CD",
    "cream": "#DED9C5",
    "straw": "#B9B65E",
    # the discrepancies of the accuracy sweep: palette PGL of chencrafts
    # (color_palettes["PGL"]); blues for the two leakage discrepancies, reds
    # for the two logical-error discrepancies, the dark shade for the first
    # of each pair, and moss for the discrepancy of the error trajectories
    "midnight": "#0C2E6D",
    "powder_blue": "#91ADC2",
    "raspberry": "#B63566",
    "blush": "#E9C2C3",
    "moss": "#AEB358",
    "gray_pale": "#F2F2F2",
    "gray_10": "#E6E6E6",
    "gray_25": "#BFBFBF",
    "gray_45": "#8C8C8C",
    "gray_65": "#595959",
    "gray_85": "#262626",
}
# the palette as TikZ colors, read by the style of the TikZ figures
TIKZ_PALETTE = Path(__file__).resolve().parent / "figs" / "style" / "palette.tex"
# categorical colors in fixed order, for quantities that have no color of
# their own (the discrepancies)
COLORS = [PALETTE[name] for name in
          ("blue", "orange", "cyan", "yellow", "rose", "green", "indigo")]
# black of the totals, of the actual QEC dynamics, and of the reference parameters
EMPHASIS = PALETTE["ink"]
# the leakage part of a budget entry has the color of the leakage subspace of the
# tree figures (aebLeak of figs/style/aeb-figure-style.tex)
LEAKAGE = PALETTE["parchment"]
# The parameter sets that the figures of the optimization mark: label (the
# wording of the note) to color, marker, and whether the marker or bar is
# filled. The reference parameters (Ofek et al.) are black and open, so that
# they are never taken for a minimum. Every set is a circle: the color tells the
# sets apart, and the shape of a marker is left to the models
# (``MODEL_MARKERS``).
POINT_STYLES = {
    "Ofek et al.": dict(color=EMPHASIS, marker="o", filled=False),
    r"$\Delta>0$ minimum": dict(color=PALETTE["poppy"], marker="o", filled=True),
    r"$\Delta<0$ minimum": dict(color=PALETTE["steel_blue"], marker="o", filled=True),
}
# The two numerically computed models of one QEC cycle, by the shape of their
# marker in every figure: label (the wording of the note) to marker, whether
# the marker has a face, and its size in units of the marker size. Both are
# as wide as the upright cross "+" (0.85 of the marker size): a small open
# diamond (its path is sqrt(2) times its size wide) shows where its value lies
# more precisely than a circle, and where the two models agree, the cross lies
# over the diamond and joins its corners. The analytical formula is drawn as
# curves or bars, except where its single terms are compared with numerical
# values drawn as bars (``formula-terms.py``): there it is an oblique cross of
# the same width.
MODEL_MARKERS = {
    "actual QEC dynamics": dict(marker="D", open_face=True, scale=0.85 / np.sqrt(2)),
    "failure Markov chain": dict(marker="+", open_face=False, scale=0.85),
    "analytical formula": dict(marker="x", open_face=False, scale=0.85),
}
# Text sizes in pt, on the page: one size for every text of a figure (labels,
# ticks, legends, panel tags), and one smaller size, permitted only for the
# numbers written inside a plot
TEXT_SIZE = 8
SMALL_TEXT_SIZE = 7
# Line widths in pt: three for every line of every figure, the first two
# those of the TikZ figures (0.8 pt and 0.4 pt) and of the rules of the
# tables. The frame of the axes and its ticks, and every outline and marker
# edge; the grid and the guides; a curve. Every line of the data figures has
# one of these widths; the white separators and the hatchings of the bars
# (``PART_SEPARATOR_WIDTH``, ``HATCH_WIDTH``) belong to the fill of a bar and
# have their own. Then the marker size and its edge.
FRAME_WIDTH = 0.8
THIN_WIDTH = 0.4
CURVE_WIDTH = 1.2
MARKER_SIZE = 4.5
MARKER_EDGE_WIDTH = FRAME_WIDTH
# Widths of a figure in inch: the column width and the text width of the
# manuscript (REVTeX 4.2, two columns: 246 pt and 510 pt), so that a figure
# drawn at one of them is included without scaling and its text keeps its size
POINTS_PER_INCH = 72.27
COLUMN_WIDTH = 246.0 / POINTS_PER_INCH
TEXT_WIDTH = 510.0 / POINTS_PER_INCH
# Shapes of the axes box of a panel, height over width of the plotting area
# (without tick labels, axis labels, or color bar). Every panel of the
# manuscript has one of them, so that the panels look alike across figures:
# regular (2:3), flat (1:2), and thin (1:3)
ASPECT_REGULAR, ASPECT_FLAT, ASPECT_THIN = 2 / 3, 1 / 2, 1 / 3
# largest change of a width by the rounding of a figure to whole pixels (inch;
# one pixel at 100 dpi), within which ``save_figure`` restores the width
WIDTH_TOLERANCE = 0.01
# Math packages of the document (``macros.tex``), in its order. No font
# package: the text and the math of a figure are set by LaTeX in Computer
# Modern, the default of REVTeX, which is the font of the page
# Sub- and superscripts of the 8 pt text are set at 7 pt, as in the TikZ
# figures (``figs/style/figure-fonts.tex``): at the default 6 pt the records,
# the labels of the discrepancies, and the indices are too small to read.
TEX_PREAMBLE = "\n".join([
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{bm}",
    r"\DeclareMathSizes{8}{8}{7}{6}",
])
# horizontal line of the failure rate at the reference parameters of either
# ancilla: the color of the reference set, dashed to tell it from the best run
REFERENCE_LINE = dict(color=EMPHASIS, lw=CURVE_WIDTH, ls="--")
# vertical line at a bound of the search range of a parameter: thin and solid
# in a neutral gray, so that it is taken neither for a curve nor for the
# dashed reference value
BOUND_LINE = dict(color=PALETTE["gray_45"], lw=THIN_WIDTH, ls="-")
# grays of quantities that belong to no parameter set: a neutral one, and a
# light and a dark one for bars and for curves that are told apart by gray
NEUTRAL, LIGHT_GRAY, DARK_GRAY = PALETTE["gray_45"], PALETTE["gray_25"], PALETTE["gray_65"]
# vertical line at a marked value of the swept parameter (a reference scale, the
# value at an optimum): thin, dashed, and neutral gray, behind the data, so that
# it is taken neither for a curve nor for the solid bound of a search range
MARKED_VALUE_LINE = dict(color=NEUTRAL, lw=THIN_WIDTH, ls="--", zorder=-1)


# A color map in the style of the palette ("chalk"). Measured in the
# perceptual color space OKLCH (lightness L from 0 to 1, chroma C, hue h), the
# colors of PALETTE that the figures use never reach black or white (L from
# 0.31 for dark_olive to 0.90 for parchment), keep a moderate chroma (median
# 0.09; viridis reaches 0.19), and have midnight as their darkest blue.
# chalk_colormap turns a color map of matplotlib into this style: it keeps
# the hue of each color, maps its lightness linearly onto CHALK_LIGHTNESS,
# scales its chroma by CHALK_CHROMA_SCALE and caps it at CHALK_CHROMA_CAP,
# and turns the hue of the dark end by CHALK_DARK_HUE_TURN_DEG (the violet of
# viridis toward midnight), less and less toward the light end.
CHALK_LIGHTNESS = (0.30, 0.88)
CHALK_CHROMA_SCALE, CHALK_CHROMA_CAP = 0.75, 0.11
CHALK_DARK_HUE_TURN_DEG = -45.0


def _to_oklab(rgb: np.ndarray) -> np.ndarray:
    r"""Convert sRGB colors (rows, components in [0, 1]) to OKLab (rows L, a, b)."""
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    lms = np.cbrt(linear @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                     [0.2119034982, 0.6806995451, 0.1073969566],
                                     [0.0883024619, 0.2817188376, 0.6299787005]]).T)
    return lms @ np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                           [1.9779984951, -2.4285922050, 0.4505937099],
                           [0.0259040371, 0.7827717662, -0.8086757660]]).T


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    r"""Convert OKLab colors (rows L, a, b) to sRGB, not clipped to [0, 1]."""
    lms = (lab @ np.array([[1.0, 0.3963377774, 0.2158037573],
                           [1.0, -0.1055613458, -0.0638541728],
                           [1.0, -0.0894841775, -1.2914855480]]).T) ** 3
    linear = lms @ np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                             [-1.2684380046, 2.6097574011, -0.3413193965],
                             [-0.0041960863, -0.7034186147, 1.7076147010]]).T
    return np.where(linear > 0.0031308, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055,
                    12.92 * linear)


def chalk_colormap(name: str = "viridis", samples: int = 256):
    r"""A color map of matplotlib in the style of the palette (lighter darks, softer chroma).

    Parameters
    ----------
    name : str
        Name of the color map of matplotlib to convert; its hues are kept.
    samples : int
        Number of colors of the returned map.

    Returns
    -------
    matplotlib.colors.ListedColormap
        The converted map, named ``chalk_<name>``: lightness rises linearly
        over ``CHALK_LIGHTNESS`` as in the original map, the chroma is
        scaled and capped (``CHALK_CHROMA_SCALE``, ``CHALK_CHROMA_CAP``) and
        kept inside the colors that sRGB can show, and the hue of the dark
        end is turned by ``CHALK_DARK_HUE_TURN_DEG``.
    """
    import matplotlib
    from matplotlib.colors import ListedColormap

    position = np.linspace(0.0, 1.0, samples)
    lab = _to_oklab(matplotlib.colormaps[name](position)[:, :3])
    lightness, chroma = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    hue = np.arctan2(lab[:, 2], lab[:, 1]) + np.radians(CHALK_DARK_HUE_TURN_DEG) * (1 - position) ** 2
    low, high = CHALK_LIGHTNESS
    lightness = low + (lightness - lightness[0]) / (lightness[-1] - lightness[0]) * (high - low)
    chroma = np.minimum(CHALK_CHROMA_SCALE * chroma, CHALK_CHROMA_CAP)

    def shown(c):
        # whether each color of chroma c lies inside the colors of sRGB
        rgb = _from_oklab(np.stack([lightness, c * np.cos(hue), c * np.sin(hue)], axis=1))
        return np.all((rgb >= 0.0) & (rgb <= 1.0), axis=1)

    # largest chroma that sRGB can show at each lightness and hue, by bisection
    inside, outside = np.zeros(samples), np.full(samples, 0.5)
    for _ in range(40):
        middle = (inside + outside) / 2
        ok = shown(middle)
        inside, outside = np.where(ok, middle, inside), np.where(ok, outside, middle)
    chroma = np.minimum(chroma, 0.98 * inside)
    rgb = _from_oklab(np.stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=1))
    return ListedColormap(np.clip(rgb, 0.0, 1.0), name=f"chalk_{name}")


def write_tikz_palette(path: Path = TIKZ_PALETTE) -> None:
    r"""Write ``PALETTE`` as a LaTeX file of color definitions for the TikZ figures.

    Parameters
    ----------
    path : pathlib.Path
        Output file. Each entry ``some_name`` of ``PALETTE`` becomes the color
        ``aebPalSomeName``.

    Returns
    -------
    None
    """
    lines = [
        "% palette.tex",
        "% GENERATED from plot_style.PALETTE (plot_style.py) by",
        "% `python plot_style.py`. Do not edit by hand: change PALETTE and run it again.",
        "% One color per entry of the palette; the role colors of the TikZ figures",
        "% (aebNeglected, aebFail, ...) are defined from these in aeb-figure-style.tex.",
    ]
    for name, value in PALETTE.items():
        # some_name -> aebPalSomeName; xcolor reads the hex digits without the "#"
        tex_name = "aebPal" + "".join(part.capitalize() for part in name.split("_"))
        lines.append(f"\\definecolor{{{tex_name}}}{{HTML}}{{{value.lstrip('#').upper()}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def plain_text(text: str) -> str:
    r"""A label written in plain characters, as the active text renderer needs it.

    Parameters
    ----------
    text : str
        Label as it is to be read, for example ``"leakage  12%"``.

    Returns
    -------
    str
        The same label if matplotlib renders the text itself. If LaTeX renders
        it (``text.usetex``), the percent sign is escaped, since it would
        start a comment, and a run of spaces is kept, since LaTeX would reduce
        it to one space.
    """
    import matplotlib

    if not matplotlib.rcParams["text.usetex"]:
        return text
    return text.replace("%", r"\%").replace("  ", "~~")


def use_manuscript_style() -> None:
    r"""Set the fonts, text sizes, and line widths of the figures drawn afterwards.

    The text and the math of a figure are set by LaTeX with ``TEX_PREAMBLE``,
    at ``TEXT_SIZE`` points. The values are written to ``matplotlib.rcParams``:
    a package imported by the sweep scripts rebinds ``pyplot.rcParams``, and
    the axes read the former.

    Returns
    -------
    None
    """
    import matplotlib

    matplotlib.rcParams.update({
        # font of the document; the serif family makes LaTeX use its roman
        # font, Computer Modern, and fonts are embedded as outlines
        # (Type 1 from LaTeX, TrueType otherwise), never as bitmaps
        "text.usetex": True, "text.latex.preamble": TEX_PREAMBLE,
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
        # one text size
        "font.size": TEXT_SIZE, "axes.labelsize": TEXT_SIZE, "axes.titlesize": TEXT_SIZE,
        "xtick.labelsize": TEXT_SIZE, "ytick.labelsize": TEXT_SIZE,
        "legend.fontsize": TEXT_SIZE, "legend.title_fontsize": TEXT_SIZE,
        "figure.titlesize": TEXT_SIZE,
        # frame of the axes and its ticks
        "axes.linewidth": FRAME_WIDTH,
        "xtick.major.width": FRAME_WIDTH, "ytick.major.width": FRAME_WIDTH,
        "xtick.minor.width": FRAME_WIDTH, "ytick.minor.width": FRAME_WIDTH,
        # outline of a patch (a bar, a handle of a legend) that sets no width
        "patch.linewidth": FRAME_WIDTH,
        # grid and guides, behind the data
        "grid.linewidth": THIN_WIDTH, "grid.color": LIGHT_GRAY, "axes.axisbelow": True,
        # curves and markers
        "lines.linewidth": CURVE_WIDTH, "lines.markersize": MARKER_SIZE,
        "lines.markeredgewidth": MARKER_EDGE_WIDTH,
        "legend.frameon": False})


def panel_tag(ax, text: str, pad: float | None = None) -> None:
    r"""Panel label above the upper left corner of the axes, as in the figures of the note.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    text : str
        Label, for example ``"(a)"``.
    pad : float, optional
        Distance of the label from the upper edge of the axes (pt), for the
        panels of a row of which one has an axis on its upper edge: the same
        value puts their labels at the same height. By default the label sits
        directly above the axes, or above an axis on the upper edge.

    Returns
    -------
    None
    """
    if pad is None:
        ax.set_title(text, loc="left", fontsize=TEXT_SIZE)
    else:
        # a given height: the title is not moved above an axis on the upper edge
        ax.set_title(text, loc="left", fontsize=TEXT_SIZE, pad=pad, y=1.0)


def point_marker_style(name: str, size: float = 6, open_face: bool | None = None) -> dict:
    r"""Keyword arguments that draw the marker of a parameter set without a line.

    Parameters
    ----------
    name : str
        Key of ``POINT_STYLES``.
    size : float
        Marker size (pt).
    open_face : bool, optional
        Whether the marker is open (white face, colored edge). By default the
        value of ``POINT_STYLES``: the minima filled, Ofek et al. open.

    Returns
    -------
    dict
        Arguments of ``ax.plot`` or ``matplotlib.lines.Line2D``: a filled
        marker has a white edge, so that it stays visible on a curve of its
        own color; an open one has the color of the set as its edge.
    """
    style = POINT_STYLES[name]
    if open_face is None:
        open_face = not style["filled"]
    if open_face:
        return dict(ls="none", marker=style["marker"], ms=size, markerfacecolor="white",
                    markeredgecolor=style["color"], markeredgewidth=MARKER_EDGE_WIDTH)
    return dict(ls="none", marker=style["marker"], ms=size, color=style["color"],
                markeredgecolor="white", markeredgewidth=MARKER_EDGE_WIDTH)


def model_marker_style(name: str, color: str, size: float = MARKER_SIZE) -> dict:
    r"""Keyword arguments that draw the marker of a numerically computed model.

    Parameters
    ----------
    name : str
        Key of ``MODEL_MARKERS``.
    color : str
        Color of the quantity that the marker shows.
    size : float
        Marker size (pt).

    Returns
    -------
    dict
        Arguments of ``ax.plot`` or ``matplotlib.lines.Line2D``, without a
        line: a small open diamond for the actual QEC dynamics and a cross for
        the failure Markov chain, both as wide as the cross. The cross is drawn
        above the diamond, so that both stay visible where the two models
        agree.
    """
    style = MODEL_MARKERS[name]
    if style["open_face"]:
        return dict(ls="none", marker=style["marker"], ms=style["scale"] * size,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=MARKER_EDGE_WIDTH, zorder=4)
    return dict(ls="none", marker=style["marker"], ms=style["scale"] * size, color=color,
                markeredgewidth=MARKER_EDGE_WIDTH, zorder=5)


def save_figure(fig, path: Path, tight: bool = True) -> None:
    r"""Save a figure as PDF (for the manuscript) and as PNG at 150 dpi (for a quick look).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure; it is closed afterwards.
    path : pathlib.Path
        Output file without extension.
    tight : bool
        Whether the saved area is cropped to what is drawn, which changes the
        size of the figure. With ``False`` the file has exactly the size of
        the figure (``figsize``), as needed for a figure that is included
        without scaling; a width that the backend of a window has rounded to
        whole pixels is set back to ``COLUMN_WIDTH`` or ``TEXT_WIDTH``.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    if not tight:
        # The backend of a window rounds the size of a new figure to whole
        # pixels (3.40 inch for a column of 3.404 inch), and it is active
        # whenever a package imported after ``matplotlib.use("Agg")`` selects
        # the backend again. A width within ``WIDTH_TOLERANCE`` of a width of
        # the manuscript is that width.
        width, height = fig.get_size_inches()
        for exact in (COLUMN_WIDTH, TEXT_WIDTH):
            if abs(width - exact) < WIDTH_TOLERANCE:
                fig.set_size_inches(exact, height, forward=False)
    # the cropping is asked for only when wanted: without it the canvas is kept
    crop = dict(bbox_inches="tight") if tight else {}
    for extension, dpi in ((".pdf", 300), (".png", 150)):
        fig.savefig(path.with_name(path.name + extension), dpi=dpi, **crop)
    plt.close(fig)
    print(f"wrote {path}.pdf and .png")


# The line plots (a quantity against a parameter) share the helpers below: one
# legend style, filled markers for curves that are not a model, axes and
# legends placed in inch on a canvas of the final size, the canvas of a figure
# with one panel, and the grid of a logarithmic axis.
# legend without a frame, with the same handle length and spacings everywhere
LEGEND_STYLE = dict(frameon=False, handlelength=2.0, handletextpad=0.5, columnspacing=1.2,
                    labelspacing=0.35, borderpad=0.0, borderaxespad=0.5)
# Filled markers, in fixed order, for curves that are not a model and are told
# apart by color (and, if they must be told apart without it, by shape). The
# diamond and the cross are absent: they mean the two numerically computed
# models (``MODEL_MARKERS``).
SERIES_MARKERS = ["o", "s", "^", "v", "p"]
# size of these markers (pt): a filled marker looks larger than the open
# marker of a model of the same size, so it is drawn smaller than ``MARKER_SIZE``
SERIES_MARKER_SIZE = 3.8
# width of a dotted guide that is read by its color: at ``THIN_WIDTH`` the
# dots are too small to show a color, so a colored dotted guide must be wider
# to keep its color, and it stays lighter than a curve
GUIDE_COLORED_WIDTH = FRAME_WIDTH
# Margins of the axes box of a figure with one panel at the width of a
# column (inch; left, bottom, right, top), which hold the tick labels and the
# axis labels. The height of the canvas follows from the shape of the box.
SINGLE_PANEL_MARGINS = (0.50, 0.37, 0.05, 0.06)


def axes_in_inches(fig, left: float, bottom: float, width: float, height: float):
    r"""Axes at a position given in inch from the lower left corner of the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure without a layout engine, drawn at its final size.
    left, bottom : float
        Lower left corner of the axes box (inch).
    width, height : float
        Size of the axes box (inch).

    Returns
    -------
    matplotlib.axes.Axes
        The new axes.
    """
    figure_width, figure_height = fig.get_size_inches()
    return fig.add_axes((left / figure_width, bottom / figure_height,
                         width / figure_width, height / figure_height))


def legend_in_inches(fig, handles: list, left: float, top: float, columns: int = 1,
                     title: str | None = None):
    r"""Legend of the figure whose upper left corner is at a position given in inch.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure drawn at its final size.
    handles : list
        Legend handles.
    left, top : float
        Upper left corner of the legend, from the lower left corner of the
        figure (inch).
    columns : int
        Number of columns; the entries fill one column after the other.
    title : str, optional
        Title of the legend, set flush left.

    Returns
    -------
    matplotlib.legend.Legend
        The legend, in ``LEGEND_STYLE``.
    """
    figure_width, figure_height = fig.get_size_inches()
    # the corner of the legend itself is anchored: no pad between the two
    style = LEGEND_STYLE | dict(borderaxespad=0.0)
    return fig.legend(handles=handles, ncol=columns, loc="upper left", title=title,
                      alignment="left",
                      bbox_to_anchor=(left / figure_width, top / figure_height), **style)


def single_panel(aspect: float = None):
    r"""Figure with one panel at the width of a column, with the shared axes box.

    Parameters
    ----------
    aspect : float, optional
        Shape of the axes box, height over width: ``ASPECT_REGULAR`` (the
        default) or ``ASPECT_FLAT``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Canvas of ``COLUMN_WIDTH``, as high as the margins and the box need.
    ax : matplotlib.axes.Axes
        Axes inside the margins ``SINGLE_PANEL_MARGINS``.
    """
    import matplotlib.pyplot as plt

    if aspect is None:
        aspect = ASPECT_REGULAR
    # the width of the box is set by the column and the margins, its height by the shape
    left, bottom, right, top = SINGLE_PANEL_MARGINS
    width = COLUMN_WIDTH - left - right
    height = bottom + aspect * width + top
    fig = plt.figure(figsize=(COLUMN_WIDTH, height))
    ax = axes_in_inches(fig, left, bottom, width, aspect * width)
    return fig, ax


def log_grid(ax, axis: str = "both") -> None:
    r"""Dotted grid at the major ticks of a logarithmic axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    axis : str
        ``"x"``, ``"y"``, or ``"both"``: the logarithmic axes of the panel. A
        linear axis of a line plot has no grid.

    Returns
    -------
    None
    """
    ax.grid(True, which="major", axis=axis, linestyle=":", linewidth=THIN_WIDTH,
            color=LIGHT_GRAY)
    ax.set_axisbelow(True)


def series_marker_style(marker: str, color: str, filled: bool = True) -> dict:
    r"""Keyword arguments that draw the marker of a curve that is not a model.

    Parameters
    ----------
    marker : str
        Entry of ``SERIES_MARKERS``.
    color : str
        Color of the curve.
    filled : bool
        Whether the marker is filled; an open one has a white face.

    Returns
    -------
    dict
        Arguments of ``ax.plot`` or ``matplotlib.lines.Line2D``, without the
        line: the marker in the color of the curve, at ``SERIES_MARKER_SIZE``.
    """
    return dict(marker=marker, ms=SERIES_MARKER_SIZE, markeredgecolor=color,
                markeredgewidth=MARKER_EDGE_WIDTH,
                markerfacecolor=color if filled else "white")


# The two figures that compare the models of one QEC cycle step by step
# (``error-budget.py``: actual QEC dynamics next to the failure Markov chain;
# ``cycle-mechanism-budget.py``: failure Markov chain next to the analytical
# formula) share one design, so that the failure Markov chain is the same bar
# in both: a black frame around its split by logical manifestation.
# logical manifestation (key of the JSON records) -> label and color, from
# the top of a stacked bar down: the pale sky of Woman with a Parasol above,
# the dark grass below
MANIFESTATION_STYLES = {
    "leakage": ("leakage", LEAKAGE),
    "X": (r"$X_{\mathrm{L}}$", PALETTE["citron"]),
    "Y": (r"$Y_{\mathrm{L}}$", PALETTE["olive"]),
    "Z": (r"$Z_{\mathrm{L}}$", PALETTE["forest"]),
}
# mechanism of the analytical formula -> color, in the order in which the
# figures list the mechanisms (``cycle-mechanism-budget.py`` puts ancilla
# dephasing first): poppy and earth tones for the ancilla, blues
# for the resonator, and gray for the mechanisms that are merged. Within a
# mode the downward process (relaxation, loss) is dark and the upward one
# (excitation, gain) light; dephasing, which moves no energy, stands apart
MECHANISM_COLORS = {
    "ancilla relaxation": PALETTE["poppy"],
    "ancilla excitation": PALETTE["terracotta"],
    "ancilla dephasing": PALETTE["dark_olive"],
    "photon loss": PALETTE["steel_blue"],
    "photon gain": PALETTE["haze_blue"],
    OTHER_MECHANISMS: PALETTE["gray_25"],
}
# the mechanisms with a group of bars of their own in a bar chart by
# mechanism, in the order of MECHANISM_COLORS; the others are small and share
# the group OTHER_MECHANISMS
SEPARATE_MECHANISMS = tuple(name for name in MECHANISM_COLORS if name != OTHER_MECHANISMS)
# the groups of a bar chart by mechanism from the left: the sum of all, the
# separate mechanisms, and the others together
MECHANISM_GROUPS = ("all mechanisms", *SEPARATE_MECHANISMS, OTHER_MECHANISMS)
# term of the analytical formula -> color, in the figure that shows every term
# (``formula-terms.py``): the mechanisms above in their order, then the two
# that the figures of the budget merge into the other mechanisms
TERM_COLORS = {name: color for name, color in MECHANISM_COLORS.items()
               if name != OTHER_MECHANISMS} | {
    "coherent error": PALETTE["cream"],
    "readout assignment error": PALETTE["straw"],
}
# face of the bar of the actual QEC dynamics, which has no split
ACTUAL_FACE = PALETTE["gray_10"]
# One bar geometry for every bar chart, in pt on the page: the width of a bar,
# and the smallest gap between the bars of one group (the models or parameter
# sets of one step or mechanism); ``group_offsets`` shares the rest of a slot
BAR_WIDTH_PT = 6.0
BAR_GAP_PT = 1.0
# gap between the bars of one group in slots of equal width, a little wider
# than ``BAR_GAP_PT`` so that the bars of a pair read as two
BAR_GROUP_GAP_PT = 2.0
CHAIN_LABEL = "failure Markov chain"
# Line widths of the bar charts in pt: the black frame of the failure Markov
# chain (the frame width, the thick line of the TikZ figures), the white line between the parts
# of a stacked bar, and a line of a hatching
CHAIN_FRAME_WIDTH = FRAME_WIDTH
PART_SEPARATOR_WIDTH = 0.25
# smallest height (pt) of the two parts on either side of a white separator
SEPARATED_PART_PT = 1.0
HATCH_WIDTH = 0.7
# Hatchings of the bar charts, coarse enough to stay open in print: white
# diagonal lines for the second of two or three bars that share a color, and
# white crossed lines for the third
DIAGONAL_HATCH, CROSSED_HATCH = "////", "xxxx"
# The legend of every bar chart: no frame, one handle size (in units of the
# text size) that is large enough to show a hatching, and one spacing
BAR_LEGEND = dict(frameon=False, fontsize=TEXT_SIZE, handlelength=1.5, handleheight=0.8,
                  handletextpad=0.5, labelspacing=0.3, columnspacing=1.2, borderpad=0.0,
                  borderaxespad=0.5)
# A percentage written on a difference that is smaller than this part of the
# largest difference of its panel is set in gray, the others in black, so
# that the eye goes to the differences that matter
MINOR_DIFFERENCE = 0.25
# Geometry in pt of the two figures that compare two models step by step: the
# canvas has the width of a column, and the margins and the shapes of the two
# axes boxes (``ASPECT_FLAT`` above, ``ASPECT_THIN`` below) are fixed, so that
# both figures have the same axes boxes whatever their tick labels are. The margins hold: left, the label of one line and the
# tick labels, as in ``SINGLE_PANEL_MARGINS``; bottom, the names of the steps
# in up to two lines; top and between the axes, a panel tag.
PAIR_MARGINS = dict(left=36.0, right=2.5, top=13.5, bottom=27.0, between=20.0)
PAIR_AXES_WIDTH = COLUMN_WIDTH * POINTS_PER_INCH - PAIR_MARGINS["left"] - PAIR_MARGINS["right"]
PAIR_AXES_HEIGHTS = (ASPECT_FLAT * PAIR_AXES_WIDTH, ASPECT_THIN * PAIR_AXES_WIDTH)


# faces lighter than this relative luminance (haze blue, citron, straw, cream,
# parchment, the light grays) carry a dark hatching, since a white one
# vanishes on them
LIGHT_FACE_LUMINANCE = 0.4


def hatch_style(pattern: str | None, face: str | None = None) -> dict:
    r"""Keyword arguments of a bar or of a legend patch with a hatching.

    Parameters
    ----------
    pattern : str or None
        ``DIAGONAL_HATCH``, ``CROSSED_HATCH``, or None for a bar without a
        hatching.
    face : str, optional
        Face color of the bar. A face lighter than ``LIGHT_FACE_LUMINANCE``
        gets a hatching in ``DARK_GRAY``; any other face, or none given, a
        white one.

    Returns
    -------
    dict
        Arguments of ``ax.bar`` or ``matplotlib.patches.Patch``: lines of the
        width ``HATCH_WIDTH`` over the face color, the same in a bar and in its
        legend patch; empty for None.
    """
    from matplotlib.colors import to_rgb

    if pattern is None:
        return {}
    line = "white"
    if face is not None:
        # relative luminance of the face (sRGB, as in WCAG)
        channels = np.array(to_rgb(face))
        linear = np.where(channels <= 0.04045, channels / 12.92,
                          ((channels + 0.055) / 1.055) ** 2.4)
        if float(linear @ np.array([0.2126, 0.7152, 0.0722])) > LIGHT_FACE_LUMINANCE:
            line = DARK_GRAY
    return dict(hatch=pattern, edgecolor=line, hatch_linewidth=HATCH_WIDTH)


def points_to_data(ax, length: float) -> float:
    r"""A horizontal length on the page, in data units of the horizontal axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with a linear horizontal axis whose position and limits are
        final (placed in inch, not by a layout engine).
    length : float
        Length (pt).

    Returns
    -------
    float
        The same length in data units, so that a bar of ``BAR_WIDTH_PT`` has
        that width on the page in every figure.
    """
    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    left, right = ax.get_xlim()
    return length * (right - left) / width_pt


def group_offsets(ax, count: int, pitch: float | None = None) -> tuple[float, np.ndarray]:
    r"""Width and positions of the bars of one group, in data units.

    The bars are ``BAR_WIDTH_PT`` wide in every figure. For groups placed in
    slots of equal width (``pitch``), the bars of a group are
    ``BAR_GROUP_GAP_PT`` apart; where that would leave less than a bar width
    between two groups (many bars per slot), they are ``BAR_GAP_PT`` apart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes as in :func:`points_to_data`.
    count : int
        Number of bars in a group.
    pitch : float, optional
        Distance between the centers of two neighboring groups (data units).
        Without it the bars of a group are ``BAR_GAP_PT`` apart, for a group
        that is part of a larger group (the two parity error spaces of one
        mechanism in ``formula-terms.py``).

    Returns
    -------
    width : float
        Width of a bar, ``BAR_WIDTH_PT``.
    offsets : np.ndarray
        Centers of the bars relative to the center of the group.
    """
    width = points_to_data(ax, BAR_WIDTH_PT)
    if pitch is None:
        return width, (np.arange(count) - (count - 1) / 2) * (width + points_to_data(ax, BAR_GAP_PT))
    # the wider gap, unless the groups would then come closer than a bar width
    gap = points_to_data(ax, BAR_GROUP_GAP_PT)
    if pitch - count * width - (count - 1) * gap < width:
        gap = points_to_data(ax, BAR_GAP_PT)
    return width, (np.arange(count) - (count - 1) / 2) * (width + gap)


def value_grid(ax, axis: str = "y") -> None:
    r"""Dotted grid lines of the value axis of a bar chart, behind the bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    axis : str
        The axis of the values: ``"y"`` for vertical bars, ``"x"`` for
        horizontal bars. The axis of the categories has no grid.

    Returns
    -------
    None
    """
    ax.grid(True, axis=axis, linestyle=":", linewidth=THIN_WIDTH, color=LIGHT_GRAY)
    ax.set_axisbelow(True)


def bar_marker_style(name: str, size: float = MARKER_SIZE) -> dict:
    r"""Keyword arguments of the marker of a numerically computed model on a bar.

    Parameters
    ----------
    name : str
        Key of ``MODEL_MARKERS``.
    size : float
        Marker size (pt); ``NARROW_BAR_MARKER_SIZE`` for a narrow bar.

    Returns
    -------
    dict
        The arguments of :func:`model_marker_style` in black.
    """
    return model_marker_style(name, EMPHASIS, size)


def percent_text(number: str) -> str:
    r"""A signed percentage written inside a plot, set as math.

    Parameters
    ----------
    number : str
        The number with its sign, without the percent sign, for example
        ``"+6"`` or ``"-0.8"``.

    Returns
    -------
    str
        The number and the percent sign in math mode, so that the sign is a
        plus or a minus sign of the math font, as in the text.
    """
    return rf"${number}\%$"


def share_labels(shares: dict) -> dict:
    r"""Legend labels of the parts of a stacked bar, with their shares in a second column.

    Parameters
    ----------
    shares : dict
        Name of a part to its share of the total, in percent.

    Returns
    -------
    dict
        Name of a part to its legend label: the name, two spaces, and the
        share as an integer percentage. If LaTeX renders the text, each name
        is set in a box of the width of the longest name and each share flush
        right in a box of the width of the longest share, so that the names
        and the shares of the legend form two columns.
    """
    import matplotlib

    texts = {label: f"{share:.0f}%" for label, share in shares.items()}
    if not matplotlib.rcParams["text.usetex"]:
        return {label: f"{label}  {text}" for label, text in texts.items()}

    def widest(entries) -> str:
        r"""An empty box of the width of the widest of some texts.

        Parameters
        ----------
        entries : iterable of str
            Texts in LaTeX.

        Returns
        -------
        str
            A horizontal phantom of a one-column table of the texts.
        """
        rows = r"\\".join(entries)
        return rf"\hphantom{{\begin{{tabular}}{{@{{}}l@{{}}}}{rows}\end{{tabular}}}}"

    escaped = {label: text.replace("%", r"\%") for label, text in texts.items()}
    name_box, share_box = widest(shares), widest(escaped.values())
    # the line is started first: a box of zero width alone would be a line of its own
    return {label: (rf"\leavevmode\rlap{{{label}}}{name_box}~~{share_box}"
                    rf"\llap{{{escaped[label]}}}")
            for label in shares}


def set_canvas(fig, width: float, height: float) -> None:
    r"""Give a figure exactly the size at which it is included in the manuscript.

    The backend of a window rounds the size of a new figure to whole pixels.
    :func:`save_figure` sets the width back; this function also sets the
    height, for the figures that must have the same height whatever the
    backend is (:func:`step_pair_figure`).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure, before anything is laid out in it.
    width, height : float
        Size of the canvas (inch).

    Returns
    -------
    None
    """
    fig.set_size_inches(width, height, forward=False)


def step_pair_figure():
    r"""Figure of the two panels that compare two models step by step.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the width of a column, with the fixed geometry of
        ``PAIR_MARGINS`` and ``PAIR_AXES_HEIGHTS``; to be saved with
        ``save_figure(fig, path, tight=False)``.
    axes : tuple of matplotlib.axes.Axes
        The upper axes (:func:`entries_by_step`) and the lower axes
        (:func:`difference_by_step`), which share the horizontal axis; only
        the lower one names the steps.
    """
    import matplotlib.pyplot as plt

    margins, (upper_height, lower_height) = PAIR_MARGINS, PAIR_AXES_HEIGHTS
    width = COLUMN_WIDTH * POINTS_PER_INCH
    height = (margins["top"] + upper_height + margins["between"] + lower_height
              + margins["bottom"])
    fig = plt.figure(figsize=(COLUMN_WIDTH, height / POINTS_PER_INCH))
    set_canvas(fig, COLUMN_WIDTH, height / POINTS_PER_INCH)
    # rectangles of the axes in parts of the canvas, from the lengths in pt
    axes_width = (width - margins["left"] - margins["right"]) / width
    lower = fig.add_axes([margins["left"] / width, margins["bottom"] / height,
                          axes_width, lower_height / height])
    upper = fig.add_axes(
        [margins["left"] / width, (margins["bottom"] + lower_height + margins["between"]) / height,
         axes_width, upper_height / height], sharex=lower)
    upper.tick_params(labelbottom=False)
    return fig, (upper, lower)


def stacked_step_bars(ax, offset: float, width: float, parts: dict, colors: dict,
                      frame: bool) -> list:
    r"""Draw one stacked bar per step and return the legend handles of its parts.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes; the steps sit at 1, 2, ... Its vertical limits are
        final, since a line between two parts is drawn only where both are
        tall enough on the page.
    offset : float
        Horizontal shift of the bars from the positions of the steps.
    width : float
        Width of a bar (data units).
    parts : dict
        Label of a part to its values at the steps, from the top of the bar
        down: the order of the legend.
    colors : dict
        Label of a part to its color.
    frame : bool
        Whether a black frame is drawn around each bar (the failure Markov
        chain).

    Returns
    -------
    list
        One ``matplotlib.patches.Patch`` per part. If the bar has more than
        one part, the label ends with the part of the total over all steps.
    """
    from matplotlib.patches import Patch

    values = {label: np.asarray(v, dtype=float) for label, v in parts.items()}
    positions = np.arange(1, len(next(iter(values.values()))) + 1) + offset
    total = sum(v.sum() for v in values.values())
    bottom = np.zeros(len(positions))
    handles = []
    # the shares of the parts in the legend, as a column next to the names
    texts = share_labels({label: 100 * v.sum() / total for label, v in values.items()})
    # height on the page (pt) of one data unit of the vertical axis
    low, high = ax.get_ylim()
    pt_per_unit = ax.get_position().height * ax.figure.get_figheight() * 72.0 / (high - low)
    # stacked from the last part up, so that the bar and the legend read in
    # the same order from the top down
    below = None
    for label, v in reversed(values.items()):
        ax.bar(positions, v, width, bottom=bottom, color=colors[label], linewidth=0)
        # a thin white line separates two parts where both are taller than
        # the line, so that a tiny part does not show as a white streak
        if below is not None:
            for x, top, a, b in zip(positions, bottom, below, v):
                if min(a, b) * pt_per_unit > SEPARATED_PART_PT:
                    ax.plot([x - width / 2, x + width / 2], [top, top], color="white",
                            lw=PART_SEPARATOR_WIDTH, solid_capstyle="butt")
        below = v
        bottom += v
    for label in values:
        text = label if len(values) == 1 else texts[label]
        # a bar without a split keeps its thin outline in the legend
        edge = dict(edgecolor=DARK_GRAY, linewidth=THIN_WIDTH) if len(values) == 1 else {}
        handles.append(Patch(facecolor=colors[label], label=text, **edge))
    if frame:
        ax.bar(positions, bottom, width, facecolor="none", edgecolor=EMPHASIS,
               linewidth=CHAIN_FRAME_WIDTH)
    else:
        # the same outline in a thin line, so that a pale bar keeps its edge
        ax.bar(positions, bottom, width, facecolor="none", edgecolor=DARK_GRAY,
               linewidth=THIN_WIDTH)
    return handles


def step_axis(ax) -> None:
    r"""Name the steps on the horizontal axis of a panel with one position per step.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes; the steps sit at 1, 2, ...

    Returns
    -------
    None
    """
    numbers = sorted(STEP_LABELS)
    # horizontal names, centered under their bars
    ax.set_xticks(numbers, [STEP_LABELS[n] for n in numbers])
    ax.set_xlim(0.4, len(numbers) + 0.6)
    ax.tick_params(axis="x", length=0)


def mechanism_axis(ax, positions, names: list, half_width: float = 0.4) -> None:
    r"""Name the groups of a bar chart by mechanism in two rows under its axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    positions : array_like
        Horizontal position of the center of each group.
    names : list of str
        Name of each group: ``"all mechanisms"``, the keys of
        ``MECHANISM_COLORS``, or ``OTHER_MECHANISMS``; for example
        ``list(MECHANISM_GROUPS)``.
    half_width : float
        Half the width of a group in data units, to which the line under
        the mechanisms of one mode extends beyond its outer groups.

    Returns
    -------
    None
        First row: the short names of ``MECHANISM_TICK_LABELS``. Second row:
        the modes of ``MECHANISM_MODES``, each under a thin line that spans
        its mechanisms, drawn outside the axes.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy

    positions = list(positions)
    ax.set_xticks(positions, [MECHANISM_TICK_LABELS[name] for name in names])
    ax.tick_params(axis="x", length=0)
    # horizontal position in data units, vertical in pt below the lower edge
    below = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-MODE_LINE_OFFSET,
                        units="points")
    for mode, (first, last) in MECHANISM_MODES.items():
        left, right = positions[names.index(first)], positions[names.index(last)]
        ax.add_line(Line2D([left - half_width, right + half_width], [0, 0], transform=below,
                           clip_on=False, color=NEUTRAL, lw=THIN_WIDTH))
        ax.annotate(mode, ((left + right) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -MODE_TEXT_OFFSET), textcoords="offset points",
                    ha="center", va="top", annotation_clip=False)


def mechanism_group_values(total: float, by_mechanism: dict) -> list:
    r"""Heights of the groups of a bar chart by mechanism, in the order of ``MECHANISM_GROUPS``.

    Parameters
    ----------
    total : float
        The value of the group of all mechanisms, e.g. the failure rate of
        the analytical formula.
    by_mechanism : dict
        Mechanism name to its contribution.

    Returns
    -------
    list of float
        ``total``, the contribution of each of ``SEPARATE_MECHANISMS``, and
        the sum of the contributions of the other mechanisms.
    """
    values = [total]
    values += [by_mechanism[name] for name in SEPARATE_MECHANISMS]
    # the small mechanisms share one group
    values.append(sum(value for name, value in by_mechanism.items()
                      if name not in SEPARATE_MECHANISMS))
    return values


def entries_by_step(ax, left: dict, right: dict) -> list:
    r"""Panel of the fidelity loss at each step in two models, as two bars per step.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    left, right : dict
        The model of the left and of the right bar: ``parts`` (label to the
        fidelity losses at the steps), ``colors`` (label to color), and
        ``frame`` (whether the bar has the black frame of the failure Markov
        chain).

    Returns
    -------
    list
        Legend handles of the left bar, then of the right bar; the caller
        composes the legend.
    """
    models = [{label: 1e2 * np.asarray(v, dtype=float) for label, v in model["parts"].items()}
              for model in (left, right)]
    # the axes are final before the bars are drawn (``stacked_step_bars``): the
    # axis ends at the next integer above the tallest bar, so that the figures
    # of this design that show the same cycle have the same axis
    step_axis(ax)
    ax.set_ylim(0, np.ceil(max(np.max(sum(parts.values())) for parts in models)))
    width, offsets = group_offsets(ax, 2, pitch=1.0)
    handles = []
    for model, parts, offset in zip((left, right), models, offsets):
        handles.append(stacked_step_bars(ax, offset, width, parts, model["colors"],
                                         model["frame"]))
    ax.set_ylabel(r"fidelity loss $\Delta F(n)$ ($10^{-2}$)")
    value_grid(ax)
    return handles[0] + handles[1]


def difference_by_step(ax, difference: np.ndarray, reference: np.ndarray, label: str) -> None:
    r"""Panel of the difference of the two bars of :func:`entries_by_step` at each step.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    difference : np.ndarray
        Right bar minus left bar at the steps.
    reference : np.ndarray
        Left bar at the steps; the number on each bar is the difference
        relative to it.
    label : str
        Label of the vertical axis, without the unit.

    Returns
    -------
    None
    """
    difference, reference = np.asarray(difference, dtype=float), np.asarray(reference, dtype=float)
    positions = np.arange(1, len(difference) + 1)
    step_axis(ax)
    # one bar under the pair of bars of the step above it, as wide as the pair
    width, offsets = group_offsets(ax, 2, pitch=1.0)
    ax.bar(positions, 1e4 * difference, offsets[1] - offsets[0] + width, color=NEUTRAL)
    ax.axhline(0.0, color=EMPHASIS, linewidth=FRAME_WIDTH)
    largest = np.max(np.abs(difference))
    for x, d, r in zip(positions, difference, reference):
        relative = 100 * d / r
        # a difference below half a percent of the entry is written as zero
        text = "0" if abs(relative) < 0.5 else percent_text(f"{relative:+.0f}")
        color = NEUTRAL if abs(d) < MINOR_DIFFERENCE * largest else EMPHASIS
        ax.annotate(text, (x, max(1e4 * d, 0.0)), xytext=(0, 2.0), textcoords="offset points",
                    ha="center", va="bottom", fontsize=SMALL_TEXT_SIZE, color=color)
    ax.set_ylabel(label + " ($10^{-4}$)")
    ax.margins(y=0.25)
    value_grid(ax)


def chain_model(by_manifestation: dict) -> dict:
    r"""The failure Markov chain as a model of :func:`entries_by_step`.

    Parameters
    ----------
    by_manifestation : dict
        Key of ``MANIFESTATION_STYLES`` to the step contributions of the
        failure Markov chain.

    Returns
    -------
    dict
        ``parts``, ``colors``, and ``frame`` of the framed bar.
    """
    parts = {label: by_manifestation[key] for key, (label, _) in MANIFESTATION_STYLES.items()}
    colors = {label: color for label, color in MANIFESTATION_STYLES.values()}
    return dict(parts=parts, colors=colors, frame=True)


def chain_handle():
    r"""Legend handle of the failure Markov chain: the black frame of its bars.

    Returns
    -------
    matplotlib.patches.Patch
        An empty framed patch with the label of the chain.
    """
    from matplotlib.patches import Patch

    return Patch(facecolor="none", edgecolor=EMPHASIS, linewidth=CHAIN_FRAME_WIDTH,
                 label=CHAIN_LABEL)


if __name__ == "__main__":
    # the TikZ figures read the palette from the file written here
    write_tikz_palette()
