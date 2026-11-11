r"""Figures of the parameter optimization of the cat-code example, from stored data.

Reads ``parameter-landscape.json``, ``budget-at-three-points.json``, and
``parameter-tradeoff.json`` and draws ``figs/parameter-optimization.pdf`` and
``figs/parameter-tradeoff.pdf``.

Usage: ``python optimization-figure.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# headless backend: the script runs without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

import plot_style  # noqa: E402

FOLDER = Path(__file__).resolve().parent
FIGURE = FOLDER / "figs" / "parameter-optimization"
TRADEOFF_FIGURE = FOLDER / "figs" / "parameter-tradeoff"

# Layout (inch) of the figure with the map and the bars, on a canvas of one
# column: the map above the bars. The two axes share their left edge, and the
# bars end under the tick labels of the color bar of the map, so that the two
# panels look equally wide and the names of the mechanisms have room.
# color bar of panel (a): distance from the map, width, and width of its tick
# labels
COLOR_BAR_GAP, COLOR_BAR_WIDTH, COLOR_BAR_TICKS = 0.06, 0.09, 0.22
# width of the map, and the width of the bars, which end under the tick labels
# of the color bar
MAP_WIDTH = 2.25
BARS_WIDTH = MAP_WIDTH + COLOR_BAR_GAP + COLOR_BAR_WIDTH + COLOR_BAR_TICKS
# panel -> left edge, lower edge, width, height of its axes: the map of the
# shape ASPECT_REGULAR, the bars of the shape ASPECT_FLAT
PANELS = {"a": (0.50, 2.41, MAP_WIDTH, plot_style.ASPECT_REGULAR * MAP_WIDTH),
          "b": (0.50, 0.42, BARS_WIDTH, plot_style.ASPECT_FLAT * BARS_WIDTH)}
# height of the canvas: the map, and above it the axis of the Josephson energy
# and the label of panel (a)
HEIGHT = PANELS["a"][1] + PANELS["a"][3] + 0.55
# Panel (a) has no legend: the resonance lines are named along them, and the
# parameter sets are named by the legend of panel (b), inside its axes (on the
# map a minimum is told by its side of the transitions, and the caption names
# the circles).
# height of the label of panel (a) above its axes (pt): above the axis of the
# Josephson energy on its upper edge
UPPER_TAG_PAD = 27.0
# The figure of the trade-off has the canvas of every figure with one panel
# (``plot_style.single_panel``); all its labels are inside the axes.

# ---- parameter sets: key of the data files -> style ----
OFEK, POS, NEG = "Ofek et al.", "Delta>0 minimum", "Delta<0 minimum"
# keys of plot_style.POINT_STYLES of the three sets that every figure marks
STYLE_KEYS = {OFEK: "Ofek et al.", POS: r"$\Delta>0$ minimum", NEG: r"$\Delta<0$ minimum"}
# names of the three computations in the data files, and the labels of the two
# numerically computed ones in plot_style.MODEL_MARKERS
FORMULA, CHAIN, ACTUAL = "analytical_formula", "failure_Markov_chain", "actual_QEC_dynamics"
MODEL_LABELS = {ACTUAL: "actual QEC dynamics", CHAIN: "failure Markov chain"}

# ---- panel (a) ----
# light gray is a low failure rate
GRAYS = LinearSegmentedColormap.from_list(
    "rate", [plot_style.PALETTE[name] for name in ("gray_pale", "gray_45", "gray_85")])
# Upper end of the gray scale (1/ms): the failure rate at
# the parameters of Ofek et al. lies inside it; the few pairs above it, in the
# corner of the Delta>0 side farthest from the transitions, are drawn in the
# darkest gray
MAP_MAXIMUM_PER_MS = 3.4
# level lines: rise of the failure rate above the lowest one of the map, all
# below the upper end of the gray scale
LEVELS = (0.02, 0.05, 0.10, 0.20, 0.30)
# the largest rise whose line is labeled on the Delta>0 side and on the
# Delta<0 side (the lines above it run along the blank band, where a label
# would cover them)
LARGEST_LABELED_ABOVE, LARGEST_LABELED_BELOW = 0.30, 0.20
# placing of the labels of the level lines (GHz): smallest distance between
# two labels, shifts of the resonator frequency that are tried in turn to
# reach it, and distance kept from the frame
LABEL_DISTANCE_GHZ, LABEL_SHIFTS_GHZ, LABEL_MARGIN_GHZ = 0.75, (0.0, -0.6, 0.6, -1.2), 0.35
# detuning (GHz) that the center of a label keeps from the edge of the blank
# band: the band runs at 45 degrees, so the label must clear it by its half
# width plus its half height, both in GHz
LABEL_BAND_MARGIN_GHZ = 0.7
# color of the level lines and of their labels on each side of the
# transitions (sign of Delta): dark on the light Delta<0 side, light on the
# dark Delta>0 side, so that a label needs no patch behind it
LEVEL_COLORS = {-1: plot_style.PALETTE["gray_85"], 1: plot_style.PALETTE["gray_10"]}
# the two transitions of the ancilla, drawn in the blank band: thin black lines
RESONANCE = dict(color=plot_style.EMPHASIS, lw=plot_style.FRAME_WIDTH)
# each resonance line is named in the blank band, along the line: the 0-1 line
# above it, the 1-2 line below it, both starting at this ancilla frequency
# (GHz), and this far from the line (pt)
RESONANCE_LABEL_START_GHZ, RESONANCE_LABEL_GAP_PT = 6.0, 1.5
# marker size of a parameter set on the map (pt)
SET_MARKER_SIZE = 6.0

# ---- trade-off ----
# black and two grays with their own line styles for the three contributions;
# the sum has the color of the Delta<0 minimum
STEP_STYLES = {
    "all seven steps": dict(color=plot_style.POINT_STYLES[STYLE_KEYS[NEG]]["color"], ls="-",
                            lw=plot_style.CURVE_WIDTH),
    "idle": dict(color=plot_style.EMPHASIS, ls="-", lw=plot_style.CURVE_WIDTH),
    "parity-mapping idling": dict(color=plot_style.DARK_GRAY, ls="-.", lw=plot_style.CURVE_WIDTH),
    "other five steps": dict(color=plot_style.NEUTRAL, ls=":", lw=plot_style.CURVE_WIDTH),
}
# Each curve is named beside it, in its color, instead of in a legend: key of
# the data -> name, position of the left end of the name (dispersive shift in
# MHz, failure rate in 1/ms), chosen where no curve, marker, or guide line lies.
STEP_NAMES = {
    "all seven steps": ("total", (2.08, 3.22)),
    "idle": ("EA idling", (0.52, 1.65)),
    "parity-mapping idling": ("PM idling", (0.74, 0.85)),
    "other five steps": ("other five steps", (1.77, 0.53)),
}
# The curves end as far beyond the last numerically computed point as they
# start before the first one, so that the panel is balanced; the axis leaves
# this margin (MHz) on both sides of the curves.
CURVE_MARGIN_MHZ = 0.065
# upper end of the failure-rate axis (1/ms): room above the curves for the
# legend of the models
TRADEOFF_TOP_PER_MS = 3.75
# The legend of the models sits under the upper frame, centered above the
# minimum of the sum. Its handles are shorter than in a legend of curves alone, so that the
# line of the formula is not much wider than a marker.
MODEL_LEGEND_STYLE = plot_style.LEGEND_STYLE | dict(handlelength=1.5, handletextpad=0.4)
# a thin gray line that sets the sum apart in panel (b) (dotted)
GUIDE = dict(color=plot_style.NEUTRAL, lw=plot_style.THIN_WIDTH)
# gap, in axes fraction, between the top of that line and the legend above it
LEGEND_GAP = 0.03


def set_marker(label: str, size: float = SET_MARKER_SIZE) -> dict:
    r"""Marker of a parameter set in panel (a).

    Parameters
    ----------
    label : str
        ``OFEK``, ``POS``, or ``NEG``.
    size : float
        Marker size (pt).

    Returns
    -------
    dict
        Arguments of ``ax.plot``: the marker of the set in every figure, a
        filled circle for the two minima and an open one for Ofek et al.
    """
    return plot_style.point_marker_style(STYLE_KEYS[label], size=size)


def bar_style(label: str) -> dict:
    r"""Appearance of the bars of one parameter set in panel (b).

    Parameters
    ----------
    label : str
        ``OFEK``, ``POS``, or ``NEG``.

    Returns
    -------
    dict
        Arguments of ``Axes.bar``: the color of the set as face, or, for the
        parameters of Ofek et al., a white face with an edge of that color.
    """
    style = plot_style.POINT_STYLES[STYLE_KEYS[label]]
    if style["filled"]:
        return dict(color=style["color"])
    return dict(facecolor="white", edgecolor=style["color"], linewidth=plot_style.FRAME_WIDTH)


def model_handles(on_bars: bool = False) -> list:
    r"""Legend handles of the two numerically computed models.

    Parameters
    ----------
    on_bars : bool
        Whether the markers lie on bars (panel (b)): black markers of
        ``plot_style.bar_marker_style``. Otherwise the gray markers of a panel
        in which the color of a marker is that of its quantity (the trade-off).

    Returns
    -------
    list
        One ``Line2D`` per entry of ``MODEL_LABELS``.
    """
    if on_bars:
        return [Line2D([], [], label=label, **plot_style.bar_marker_style(label))
                for label in MODEL_LABELS.values()]
    return [Line2D([], [], label=label,
                   **plot_style.model_marker_style(label, plot_style.NEUTRAL))
            for label in MODEL_LABELS.values()]


def plot_map(ax, color_axes, record: dict) -> list:
    r"""Panel (a): failure rate minimized over the coupling and the EA idling duration, over the two frequencies.

    Parameters
    ----------
    ax, color_axes : matplotlib.axes.Axes
        Axes of the map and of its color bar.
    record : dict
        Content of ``parameter-landscape.json``.

    Returns
    -------
    None
    """
    grid, sets = record["map"], record["parameter_sets"]
    omega_01, omega_r = np.array(grid["omega_01_GHz"]), np.array(grid["omega_r_GHz"])
    rate = np.array([[np.nan if v is None else v for v in row]
                     for row in grid["failure_rate_per_ms"]])
    lowest = grid["lowest_failure_rate_per_ms"]
    band = record["settings"]["band_GHz"]
    # The pairs that are left out (None) lie in the band; the band is drawn as
    # an exact white strip over the map, so that its edges are straight lines
    # and not the steps of the grid. Under the strip, each blank cell takes the
    # value of the nearest computed cell: only the slivers of these cells that
    # stick out of the band are seen, less than half a cell wide.
    from scipy.ndimage import distance_transform_edt

    blank = np.isnan(rate)
    nearest = distance_transform_edt(blank, return_distances=False, return_indices=True)
    filled = rate[tuple(nearest)]
    mesh = ax.pcolormesh(omega_01, omega_r, filled, cmap=GRAYS, shading="nearest", vmin=lowest,
                         vmax=MAP_MAXIMUM_PER_MS, rasterized=True)
    fine = np.linspace(omega_01[0] - 0.1, omega_01[-1] + 0.1, 400)
    omega_12 = np.interp(fine, omega_01, grid["omega_12_GHz"])
    ax.fill_between(fine, np.minimum(fine, omega_12) - band, np.maximum(fine, omega_12) + band,
                    facecolor="white", edgecolor="none", zorder=1.5)
    # lines of equal failure rate, labeled by the rise above the lowest rate
    # of the map; each side of the transitions is contoured by itself, so that
    # no line is interpolated across the blank band
    side = np.sign(omega_r[:, None] - omega_01[None, :])
    negative = sets[NEG]
    for sign in (-1, 1):
        on_side = np.where(side == sign, rate, np.nan)
        lines = ax.contour(omega_01, omega_r, on_side, levels=[lowest * (1 + v) for v in LEVELS],
                           colors=LEVEL_COLORS[sign], linewidths=plot_style.FRAME_WIDTH)
        # one label per line, where the lines are far apart: on the Delta<0
        # side where a line crosses the resonator frequency of the minimum at
        # a lower ancilla transition (or a frequency LABEL_SHIFTS_GHZ away
        # from it, if the label would touch the one before); on the Delta>0
        # side at the point of a line that is farthest from the two markers
        # there and not at the edge of the blank band
        places = []
        for value, segments in zip(LEVELS, lines.allsegs):
            vertices = np.concatenate(segments) if len(segments) else np.empty((0, 2))
            vertices = vertices[
                (vertices[:, 0] > omega_01[0] + LABEL_MARGIN_GHZ)
                & (vertices[:, 0] < omega_01[-1] - LABEL_MARGIN_GHZ)
                & (vertices[:, 1] > omega_r[0] + LABEL_MARGIN_GHZ)
                & (vertices[:, 1] < omega_r[-1] - LABEL_MARGIN_GHZ)]
            # on both sides, a label keeps its distance from the blank band
            detuning = np.minimum(
                np.abs(vertices[:, 1] - vertices[:, 0]),
                np.abs(vertices[:, 1] - np.interp(vertices[:, 0], omega_01, grid["omega_12_GHz"])))
            vertices = vertices[detuning > band + LABEL_BAND_MARGIN_GHZ]
            if sign == -1:
                vertices = vertices[vertices[:, 0] < negative["omega_01_GHz"]]
                if len(vertices) == 0 or value > LARGEST_LABELED_BELOW:
                    continue
                for shift in LABEL_SHIFTS_GHZ:
                    place = vertices[np.argmin(np.abs(
                        vertices[:, 1] - negative["free_parameters"]["E_osc_GHz"] - shift))]
                    if all(np.hypot(*(place - other)) > LABEL_DISTANCE_GHZ for other in places):
                        break
                places.append(place)
            elif value <= LARGEST_LABELED_ABOVE:
                if len(vertices) == 0:
                    continue
                markers = np.array([[sets[label]["omega_01_GHz"],
                                     sets[label]["free_parameters"]["E_osc_GHz"]]
                                    for label in (POS, OFEK)])
                distance = np.min(np.hypot(*(vertices[:, None, :] - markers[None]).transpose(2, 0, 1)),
                                  axis=1)
                places.append(vertices[np.argmax(distance)])
        places = [tuple(place) for place in places]
        # the sign and the percent sign are set in math mode
        labels = ax.clabel(lines, fmt={level: rf"$+{100 * v:.0f}\%$"
                                       for level, v in zip(lines.levels, LEVELS)},
                           fontsize=plot_style.SMALL_TEXT_SIZE, inline_spacing=2, manual=places)
        # upright labels, readable where a line runs steeply
        for label in labels:
            label.set_rotation(0)
    # the two transitions that the blank band surrounds
    ax.plot(omega_01, omega_01, ls="--", **RESONANCE)
    ax.plot(omega_01, grid["omega_12_GHz"], ls=":", **RESONANCE)
    # the three parameter sets, named by the legend of panel (b)
    for label in (NEG, POS, OFEK):
        point = sets[label]
        ax.plot(point["omega_01_GHz"], point["free_parameters"]["E_osc_GHz"], zorder=5,
                **set_marker(label))
    half = (omega_01[1] - omega_01[0]) / 2
    ax.set_xlim(omega_01[0] - half, omega_01[-1] + half)
    half = (omega_r[1] - omega_r[0]) / 2
    ax.set_ylim(omega_r[0] - half, omega_r[-1] + half)
    ax.set_xlabel(r"ancilla transition frequency $\omega_{\mathrm{a}}/2\pi$ (GHz)")
    ax.set_ylabel(r"resonator frequency $\omega_{\mathrm{r}}/2\pi$ (GHz)")
    # the Josephson energy that gives each transition frequency, on the top axis
    energies = np.array(grid["EJ_GHz"])
    top = ax.secondary_xaxis("top", functions=(
        lambda w: np.interp(w, omega_01, energies), lambda e: np.interp(e, energies, omega_01)))
    top.set_xticks([t for t in range(20, 70, 10) if energies[0] <= t <= energies[-1]])
    top.set_xlabel(r"Josephson energy $E_J/h$ (GHz)")
    # a triangle on the color bar only if some pair lies above the gray scale
    above = np.nanmax(rate) > MAP_MAXIMUM_PER_MS
    bar = ax.figure.colorbar(mesh, cax=color_axes, extend="max" if above else "neither")
    # the caption says over what the rate is minimized
    bar.set_label(r"failure rate (ms$^{-1}$)")
    # the blank band has no legend entry: the caption names it
    label_resonances(ax, omega_01, np.array(grid["omega_12_GHz"]))


def label_resonances(ax, omega_01, omega_12) -> None:
    r"""Name the two resonance lines of panel (a) along them, in the blank band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the map, with its final limits and position on the canvas.
    omega_01, omega_12 : numpy.ndarray
        The two transition frequencies of the ancilla (GHz) on the grid.

    Returns
    -------
    None
    """
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D, ScaledTranslation

    x = RESONANCE_LABEL_START_GHZ
    lines = ((omega_01, r"$\omega_{\mathrm{r}}=\omega_{\mathrm{a}}$", 1),
             (omega_12, r"$\omega_{\mathrm{r}}=\omega_{12}$", -1))
    for frequency, text, side in lines:
        # angle of the line on the page, from two nearby points of it
        y0, y1 = np.interp([x, x + 0.1], omega_01, frequency)
        (u0, v0), (u1, v1) = ax.transData.transform([(x, y0), (x + 0.1, y1)])
        angle = np.degrees(np.arctan2(v1 - v0, u1 - u0))
        # The label is drawn as the outline of its glyphs (in pt): matplotlib
        # 3.10 sets rotated LaTeX text at an angle that is not a multiple of
        # 90 degrees with its subscripts on the baseline. The baseline is
        # moved off the line, above it (side 1) by the gap plus the depth of
        # the subscripts, or below it by the gap plus the height of the text.
        glyphs = TextPath((0, 0), text, size=plot_style.TEXT_SIZE, usetex=True)
        extent = glyphs.get_extents()
        baseline = (RESONANCE_LABEL_GAP_PT - extent.y0 if side == 1
                    else -RESONANCE_LABEL_GAP_PT - extent.y1)
        # pt -> inch on the page, rotated along the line, placed at the point
        # (x, y0) of the line
        place = (Affine2D().translate(0, baseline).rotate_deg(angle).scale(1 / 72)
                 + ax.figure.dpi_scale_trans + ScaledTranslation(x, y0, ax.transData))
        ax.add_patch(PathPatch(glyphs, transform=place, facecolor=plot_style.EMPHASIS,
                               edgecolor="none", zorder=4))


def plot_mechanisms(ax, record: dict) -> None:
    r"""Panel (b): contributions of the mechanisms at the three parameter sets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    record : dict
        Content of ``budget-at-three-points.json``.

    Returns
    -------
    None
        The legend is drawn inside the axes: the bars of the three sets, which
        also names the circles of panel (a), and the markers of the two
        numerically computed models.
    """
    sets = record["parameter_sets"]
    labels = (OFEK, POS, NEG)
    # groups from the left: the sum, set apart, then the mechanisms; the axis
    # is final before the bars, whose width is set on the page
    # the groups of plot_style.MECHANISM_GROUPS: the mechanisms of the formula
    # with a group of their own (the ancilla, then the resonator); the others
    # (the coherent error of the conditional pulse, the readout assignment
    # error, chi_1') are small at every set and share one group
    x = np.arange(len(plot_style.MECHANISM_GROUPS), dtype=float)
    x[1:] += 0.25
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    width, offsets = plot_style.group_offsets(ax, len(labels), pitch=1.0)
    for offset, label in zip(offsets, labels):
        point = sets[label]
        values = plot_style.mechanism_group_values(point["failure_rate_per_ms"][FORMULA],
                                                   point["rate_by_mechanism_per_ms"])
        ax.bar(x + offset, values, width, **bar_style(label))
        # the two numerically computed failure rates on the bar of the sum
        for model, model_label in MODEL_LABELS.items():
            ax.plot(x[0] + offset, point["failure_rate_per_ms"][model],
                    **plot_style.bar_marker_style(model_label))
    ax.axvline((x[0] + x[1]) / 2, ls=":", **GUIDE)
    plot_style.mechanism_axis(ax, x, list(plot_style.MECHANISM_GROUPS))
    ax.set_ylim(0, 1.08 * max(sets[label]["failure_rate_per_ms"][FORMULA] for label in labels))
    # a contribution to a failure rate is a failure rate; the caption names it
    ax.set_ylabel(r"failure rate (ms$^{-1}$)")
    plot_style.value_grid(ax, axis="y")
    names = {OFEK: "Ofek et al.", POS: r"$\Delta>0$ minimum", NEG: r"$\Delta<0$ minimum"}
    # the EA idling duration of each set is in the table of parameter sets
    handles = [Patch(label=names[label], **bar_style(label)) for label in labels]
    # the upper right of the panel is free: the contributions of the mechanisms are low
    ax.legend(handles=handles + model_handles(on_bars=True), loc="upper right",
              **plot_style.LEGEND_STYLE)


def plot_tradeoff(ax, record: dict) -> None:
    r"""Trade-off: contributions to the failure rate against the dispersive shift.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes, at their final size on the canvas.
    record : dict
        Content of ``parameter-tradeoff.json``.

    Returns
    -------
    None
        Every label is inside the axes: each curve is named beside it, a legend
        says which model the curves and the two markers are.
    """
    formula, points = record["formula"], record["numerical"]
    chi = np.array(formula["chi_1_MHz"])
    chi_points = [point["chi_1_MHz"] for point in points]
    # the curves end as far after the last point as they start before the first
    shown = chi <= chi_points[-1] + (chi_points[0] - chi[0])
    for name, style in STEP_STYLES.items():
        values = np.array(formula[name])
        ax.plot(chi[shown], values[shown], **style)
        # the two numerically computed models, in the color of the quantity
        for model, model_label in MODEL_LABELS.items():
            ax.plot(chi_points, [point["contributions"][model][name] for point in points],
                    **plot_style.model_marker_style(model_label, style["color"]))
        text, place = STEP_NAMES[name]
        ax.text(*place, text, color=style["color"], ha="left", va="center")
    ax.set_xlim(chi[shown][0] - CURVE_MARGIN_MHZ, chi[shown][-1] + CURVE_MARGIN_MHZ)
    ax.set_ylim(0, TRADEOFF_TOP_PER_MS)
    from matplotlib.transforms import blended_transform_factory

    # which model a curve or a marker is, said once, under the upper frame and
    # above the minimum of the sum
    handles = [Line2D([], [], color=plot_style.NEUTRAL, lw=plot_style.CURVE_WIDTH,
                      label="AEB by mechanism"), *model_handles()]
    total = np.array(formula["all seven steps"])
    above_minimum = blended_transform_factory(ax.transData, ax.transAxes)
    legend = ax.legend(handles=handles, loc="upper center",
                       bbox_to_anchor=(chi[np.argmin(total)], 1.0),
                       bbox_transform=above_minimum, **MODEL_LEGEND_STYLE)
    # the dispersive shift of the Delta<0 minimum, a thin dashed gray line behind
    # the data that stops under the legend, which is centered on nearly the same shift
    ax.figure.draw_without_rendering()
    legend_bottom = legend.get_window_extent().transformed(ax.transAxes.inverted()).y0
    ax.axvline(record["numbers"]["chi_1_MHz_at_the_minimum"], ymax=legend_bottom - LEGEND_GAP,
               **plot_style.MARKED_VALUE_LINE)
    # the caption says that the coupling is varied, and that the curves are
    # contributions to the failure rate
    ax.set_xlabel(r"dispersive shift $|\bar\chi_1|/2\pi$ (MHz)")
    ax.set_ylabel(r"failure rate (ms$^{-1}$)")
    # a tick every half failure per ms
    ax.yaxis.set_major_locator(MultipleLocator(0.5))


def main() -> None:
    r"""Read the three data files and draw the two figures.

    Returns
    -------
    None
    """
    landscape = json.loads((FOLDER / "parameter-landscape.json").read_text())
    budget = json.loads((FOLDER / "budget-at-three-points.json").read_text())
    tradeoff = json.loads((FOLDER / "parameter-tradeoff.json").read_text())

    plot_style.use_manuscript_style()
    FIGURE.parent.mkdir(exist_ok=True)

    # ---- the map and the bars, at the width of one column ----
    fig = plt.figure(figsize=(plot_style.COLUMN_WIDTH, HEIGHT))
    ax_a, ax_b = (plot_style.axes_in_inches(fig, *PANELS[tag]) for tag in "ab")
    # the color bar of the map beside it, with the height of the map
    left, bottom, width, height = PANELS["a"]
    color_a = plot_style.axes_in_inches(fig, left + width + COLOR_BAR_GAP, bottom,
                                        COLOR_BAR_WIDTH, height)
    plot_map(ax_a, color_a, landscape)
    plot_mechanisms(ax_b, budget)
    # the label of panel (a) above the axis of the Josephson energy
    plot_style.panel_tag(ax_a, "(a)", pad=UPPER_TAG_PAD)
    plot_style.panel_tag(ax_b, "(b)")
    # the canvas has the width of one column and is saved as it is
    plot_style.save_figure(fig, FIGURE, tight=False)

    # ---- the trade-off, at the width of one column ----
    fig, ax = plot_style.single_panel()
    plot_tradeoff(ax, tradeoff)
    plot_style.save_figure(fig, TRADEOFF_FIGURE, tight=False)


if __name__ == "__main__":
    main()
