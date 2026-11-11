r"""Figure of the magnitudes of the neglected processes at each step, from stored data.

Reads ``magnitudes.json`` at incoherent-error scale 1 and draws the step-by-step
heatmap ``figs/magnitudes-by-step.pdf``.

Usage: ``python magnitudes-figure.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# headless backend: the script runs without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

import plot_style  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "magnitudes.json"
SCALE = "1"
# output file without extension (plot_style.save_figure adds .pdf and .png)
FIGURE = HERE / "figs" / "magnitudes-by-step"

# Key of magnitudes_by_step -> label of its row, from the top in the order of
# the reductions that neglect the processes, as the rows of Table I
# (tab:regime): unexpected measurement outcome; creation and return of
# error--leakage coherence; leakage and seepage; coherence between error
# spaces; neglected error-space transitions; coherence of logical errors;
# stochastic logical errors. The JSON key err_traj is the code name of the
# neglected error-space transitions.
ROWS = {
    "unexp": r"$\epsilon^{\mathrm{unexp}}_n$",
    "leak_coh": r"$\epsilon^{\mathrm{leak\text{-}coh}}_n$",
    "seep_coh": r"$\epsilon^{\mathrm{seep\text{-}coh}}_n$",
    "leak": r"$\epsilon^{\mathrm{leak}}_n$",
    "seep": r"$\epsilon^{\mathrm{seep}}_n$",
    "err_coh": r"$\epsilon^{\mathrm{err\text{-}coh}}_n$",
    "err_traj": r"$\epsilon^{\mathrm{err\text{-}trans}}_n$",
    "logi_coh": r"$\epsilon^{\mathrm{logi\text{-}coh}}_n$",
    "logi_err": r"$\epsilon^{\mathrm{logi\text{-}err}}_n$",
}
# Magnitudes that enter no bound of Appendix B, drawn as white cells. The
# leakage bounds pair a creation at step i with a return at a later step j,
# so a return at the first step and a creation at the last step enter none;
# the bound on the neglected error-space transitions sums over the steps
# after the error-accumulation step, and the bound on the unexpected
# measurement outcome uses its magnitude at the measurement step only.
RETURNS = ("seep", "seep_coh")
CREATIONS = ("leak", "leak_coh")
AFTER_ERROR_ACCUMULATION = ("err_traj",)
AT_MEASUREMENT = ("unexp",)
# color map of the magnitude, from small to large, turned into the style of
# the palette by plot_style.chalk_colormap
COLOR_MAP = "viridis"
# whether the color scale is logarithmic (else linear, from zero)
LOG_SCALE = True
# linear scale: its upper end is the largest magnitude rounded up to a
# multiple of this step
LINEAR_STEP = 0.05
# decades between the labeled ticks of the color bar
TICK_DECADES = 1
# Layout (inch) on a canvas of the width of a column: left edge of the map
# (holds the row labels), height of one row, gap between map and color bar,
# width of the color bar, room right of it (holds its tick labels), and the margins
# under the map (holds the step names) and above it
MAP_LEFT, ROW_HEIGHT = 0.62, 0.25
BAR_GAP, BAR_WIDTH, BAR_ROOM = 0.10, 0.10, 0.36
BOTTOM_MARGIN, TOP_MARGIN = 0.38, 0.08
MAP_WIDTH = plot_style.COLUMN_WIDTH - MAP_LEFT - BAR_GAP - BAR_WIDTH - BAR_ROOM
MAP_HEIGHT = ROW_HEIGHT * len(ROWS)
FIGURE_HEIGHT = BOTTOM_MARGIN + MAP_HEIGHT + TOP_MARGIN


def main() -> None:
    r"""Read the magnitudes at the incoherent-error scale 1 and draw the map.

    Returns
    -------
    None
    """
    data = json.loads(DATA.read_text())
    magnitudes = data["scales"][SCALE]["magnitudes_by_step"]
    # step numbers (from 1) -> columns (from 0)
    error_accumulation = [n - 1 for n in data["error_accumulation_steps"]]
    measurement = [n - 1 for n in data["measurement_steps"]]
    values = np.array([magnitudes[key] for key in ROWS], dtype=float)
    # magnitudes that enter no bound: masked, and drawn white
    shown = np.ma.array(values)
    row = {key: list(ROWS).index(key) for key in ROWS}
    others = [n for n in range(values.shape[1]) if n not in measurement]
    for keys, columns in ((RETURNS, [0]), (CREATIONS, [-1]),
                          (AFTER_ERROR_ACCUMULATION, error_accumulation),
                          (AT_MEASUREMENT, others)):
        for key in keys:
            shown[row[key], columns] = np.ma.masked
    # a zero magnitude has no logarithm; on either scale it takes the lowest
    # color
    plot_style.use_manuscript_style()
    fig = plt.figure(figsize=(plot_style.COLUMN_WIDTH, FIGURE_HEIGHT))
    ax = plot_style.axes_in_inches(fig, MAP_LEFT, BOTTOM_MARGIN, MAP_WIDTH, MAP_HEIGHT)
    color_axes = plot_style.axes_in_inches(fig, MAP_LEFT + MAP_WIDTH + BAR_GAP, BOTTOM_MARGIN,
                                           BAR_WIDTH, MAP_HEIGHT)
    cmap = plot_style.chalk_colormap(COLOR_MAP)
    cmap.set_bad("white")
    steps = np.arange(1, len(plot_style.STEP_LABELS) + 1)
    rows = np.arange(len(ROWS))
    if LOG_SCALE:
        # the lower end at the smallest nonzero magnitude drawn, the upper end
        # at the whole decade above the largest
        smallest = shown[shown > 0].min()
        highest = np.ceil(np.log10(shown.max()))
        norm = LogNorm(vmin=smallest, vmax=10.0**highest)
        # a zero below the lower end, so that LogNorm draws it in the lowest color
        shown = np.ma.where(shown > 0, shown, smallest / 10.0)
    else:
        norm = Normalize(vmin=0.0, vmax=LINEAR_STEP * np.ceil(shown.max() / LINEAR_STEP))
    mesh = ax.pcolormesh(steps, rows, shown, cmap=cmap, shading="nearest", norm=norm)
    plot_style.step_axis(ax)
    ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
    ax.set_yticks(rows, list(ROWS.values()))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(rows[-1] + 0.5, rows[0] - 0.5)
    if LOG_SCALE:
        # no triangle on the color bar: only the zeros lie below it, and the
        # caption says that they take the lowest color
        bar = fig.colorbar(mesh, cax=color_axes)
        bar.set_ticks(10.0 ** np.arange(highest, np.log10(smallest), -TICK_DECADES))
        bar.minorticks_off()
    else:
        fig.colorbar(mesh, cax=color_axes)
    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    main()
