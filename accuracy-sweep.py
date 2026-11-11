r"""Accuracy of the error budget by step against the incoherent-error scale.

Evaluates the process fidelities of the chain of models, the seven
discrepancies, and the logical-error and leakage probabilities across the
incoherent-error scale :math:`\xi` at the parameters of Ofek et al. [Nature
536, 441 (2016)]. Writes ``accuracy-sweep.json`` and
``figs/accuracy-sweep.pdf``.

Usage: ``python accuracy-sweep.py`` (or ``--plot-only``, ``--update``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain
import plot_style

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "accuracy-sweep.json"
# output file without extension (plot_style.save_figure adds .pdf and .png)
FIGURE = HERE / "figs" / "accuracy-sweep"

# incoherent-error scales; 1 is the parameter set of Ofek et al.
SCALES = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]
# scale at which no decoherence acts: every rate is 1e-4 of its value
NO_INCOHERENT_ERROR_SCALE = 1e-4
# the exponents are fitted for the scales in this range
EXPONENT_RANGE = (1 / 16, 1.0)
# quality factors divided by the scale; ancilla dephasing rate and the two
# readout assignment errors multiplied by it
DIVIDED_BY_SCALE = device_model.DIVIDED_BY_INCOHERENT_ERROR_SCALE
MULTIPLIED_BY_SCALE = device_model.MULTIPLIED_BY_INCOHERENT_ERROR_SCALE
# worker processes for the construction of the maps
NUM_CPUS = 1

# symbol and color of each discrepancy, in the order of the chain; the keys
# are those of model_chain.REDUCTIONS. The five that are drawn take the five
# colors of the palette PGL: reds for the two leakage discrepancies, blues for
# the two logical-error discrepancies, moss for the error-space transitions.
# delta_unexp and delta_err-coh vanish at every scale and are not drawn (the
# caption names them); their colors are placeholders
DISCREPANCIES = {
    "unexp": (r"$\delta_{\mathrm{unexp}}$", plot_style.COLORS[6]),
    "leak_coh": (r"$\delta_{\mathrm{leak\text{-}coh}}$", plot_style.PALETTE["raspberry"]),
    "leak": (r"$\delta_{\mathrm{leak}}$", plot_style.PALETTE["blush"]),
    "err_coh": (r"$\delta_{\mathrm{err\text{-}coh}}$", plot_style.COLORS[5]),
    "err_traj": (r"$\delta_{\mathrm{err\text{-}trans}}$", plot_style.PALETTE["moss"]),
    "logi_coh": (r"$\delta_{\mathrm{logi\text{-}coh}}$", plot_style.PALETTE["midnight"]),
    "logi_err": (r"$\delta_{\mathrm{logi\text{-}err}}$", plot_style.PALETTE["powder_blue"]),
}
# a discrepancy below this value at every scale is rounding and is not drawn
PLOT_FLOOR = 1e-12
# Layout of the figure (inch), on a canvas of the width of a column: left
# edge and width of both axes; the margin below panel (b), the gap between the
# panels, and the margin above panel (a); lower edge and height of panel (a),
# of the shape ASPECT_REGULAR, and of panel (b), of the shape ASPECT_THIN;
# height of the canvas
AXES_LEFT, AXES_WIDTH = 0.62, plot_style.COLUMN_WIDTH - 0.62 - 0.05
BOTTOM_MARGIN, PANEL_GAP, TOP_MARGIN = 0.45, 0.28, 0.19
PANEL_B = (BOTTOM_MARGIN, plot_style.ASPECT_THIN * AXES_WIDTH)
PANEL_A = (sum(PANEL_B) + PANEL_GAP, plot_style.ASPECT_REGULAR * AXES_WIDTH)
FIGURE_HEIGHT = sum(PANEL_A) + TOP_MARGIN
# Slope guides of panel (a): discrepancy -> exponent of the scale, label, the
# factor by which the guide lies below the discrepancy at the middle of
# EXPONENT_RANGE (geometric mean of its ends), and the scale and the side
# ("above" or "below" the guide) of the label, where it touches no curve, and the
# gap in points between the guide and the label. Each guide spans
# EXPONENT_RANGE, the scales over which the exponents are fitted, just under
# its discrepancy (err-trans nearly linear, leak nearly quadratic). The factor
# of err-trans keeps its guide under leak-coh (it still crosses leak, which
# rises faster); that of leak keeps its guide under logi-coh and above logi-err.
SLOPE_GUIDES = {"err_traj": (1, r"$\propto\xi$", 0.6, 0.09, "below", 3),
                "leak": (2, r"$\propto\xi^2$", 0.2, 0.1, "below", 6)}
GUIDE_RANGE = (1 / 32, 8)   # scales spanned by the slope guides
# thin gray lines of guides, not data: dashed for the slopes and, with
# plot_style.MARKED_VALUE_LINE, for the scale of Ofek et al., which lies on a
# dotted grid line of the axis
SLOPE_LINE = dict(color=plot_style.NEUTRAL, lw=plot_style.THIN_WIDTH, ls="--", zorder=-1)

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def evaluate(scale: float) -> dict:
    r"""Process fidelities of the chain of models at one incoherent-error scale.

    Parameters
    ----------
    scale : float
        Incoherent-error scale :math:`\xi`.

    Returns
    -------
    dict
        The scale, the output of ``model_chain.chain_fidelities``, the
        output of ``model_chain.photon_loss_trajectories``
        (``photon_loss_trajectories``), and the computation time (s).
    """
    t0 = time.time()
    graph, _, _ = device_model.build_actual_tree(
        device_model.incoherent_error_scaled_parameters(scale), n_cycles=1, num_cpus=NUM_CPUS)
    row: dict = {"scale": scale} | model_chain.chain_fidelities(graph)
    row["photon_loss_trajectories"] = model_chain.photon_loss_trajectories(graph)
    row["seconds"] = time.time() - t0
    return row


def quoted_numbers(rows: list[dict]) -> dict:
    r"""Summary metrics extracted from the incoherent-error scale sweep.

    Parameters
    ----------
    rows : list of dict
        Outputs of :func:`evaluate`, in increasing order of the scale.

    Returns
    -------
    dict
        ``difference_over_infidelity``: for each scale,
        :math:`(F_{\mathbb T}-F_{\mathbb M})/(1-F_{\mathbb T})`, with the
        entries ``at_scale_1`` and ``at_smallest_scale``;
        ``manifestation_at_scale_1``: for the logical X, Y, Z errors and
        leakage, the probability over one cycle in the actual QEC dynamics
        and in the failure Markov chain and their relative difference
        (failure Markov chain over actual QEC dynamics, minus one), and
        ``largest_relative_difference``; ``largest_discrepancy_at_scale_1``;
        ``logi_coh_on_photon_loss_trajectories``: the part of
        :math:`\delta_{\mathrm{logi\text{-}coh}}` (signed, ``F_G`` minus
        ``F_twirled``) on the error trajectories with the photon loss of
        the error-accumulation idling divided by the whole, for each scale,
        at :math:`\xi=1`, and at the smallest :math:`\xi`, and the exponent of
        :math:`\delta_{\mathrm{logi\text{-}coh}}` between the two smallest
        scales.
    """
    relative = {row["scale"]: (row["fidelity"]["F_T"] - row["fidelity"]["F_M"])
                / (1.0 - row["fidelity"]["F_T"]) for row in rows}
    reference = next(row for row in rows if row["scale"] == 1.0)
    manifestation = {}
    # entry 0 of a manifestation list is the process fidelity; skip it
    for position, name in enumerate(model_chain.MANIFESTATIONS[1:], start=1):
        actual = reference["manifestation"]["F_T"][position]
        markov = reference["manifestation"]["F_M"][position]
        manifestation[name] = dict(actual_QEC_dynamics=actual, failure_Markov_chain=markov,
                                   relative_difference=markov / actual - 1.0)
    infidelity = {row["scale"]: 1.0 - row["fidelity"]["F_T"] for row in rows}
    smallest = min(infidelity)
    # exponents: straight line through the logarithms, for the scales in EXPONENT_RANGE
    in_range = sorted(scale for scale in infidelity
                      if EXPONENT_RANGE[0] <= scale <= EXPONENT_RANGE[1])

    def exponent(values: dict) -> float:
        r"""Slope of the logarithm of ``values`` against that of the scale.

        Parameters
        ----------
        values : dict
            Scale to a positive value.

        Returns
        -------
        float
            The slope of the least-squares line.
        """
        return float(np.polyfit(np.log(in_range), np.log([values[s] for s in in_range]), 1)[0])

    names = [name for name in reference["difference"]
             if all(abs(row["difference"][name]) > PLOT_FLOOR for row in rows)]
    # where delta_logi-coh sits: the share on the error trajectories that start
    # with a photon loss, and its exponent where the decoherence is weakest
    loss_share = {row["scale"]: row["photon_loss_trajectories"]["logi_coh"]
                  / row["difference"]["logi_coh"] for row in rows}
    two_smallest = sorted(loss_share)[:2]
    logi_coh = {row["scale"]: abs(row["difference"]["logi_coh"]) for row in rows}
    return dict(
        discrepancy_over_infidelity=dict(
            at_scale_1={name: abs(reference["difference"][name]) / infidelity[1.0]
                        for name in reference["difference"]},
            at_smallest_scale={
                name: abs(next(r for r in rows if r["scale"] == smallest)["difference"][name])
                / infidelity[smallest] for name in reference["difference"]},
            all_in_the_same_direction_at_scale_1=all(
                value > -PLOT_FLOOR for value in reference["difference"].values()),
            sum_at_scale_1=sum(reference["difference"].values()) / infidelity[1.0],
        ),
        exponent_for_scales_up_to_1=dict(
            scales=in_range,
            infidelity=exponent(infidelity),
            discrepancies={name: exponent({row["scale"]: abs(row["difference"][name])
                                           for row in rows}) for name in names},
        ),
        difference_over_infidelity=dict(
            by_scale={str(scale): value for scale, value in relative.items()},
            at_scale_1=relative[1.0],
            at_smallest_scale=relative[min(relative)],
            smallest_scale=min(relative),
        ),
        manifestation_at_scale_1=manifestation | dict(
            largest_relative_difference=max(
                abs(entry["relative_difference"]) for entry in manifestation.values())),
        largest_discrepancy_at_scale_1=max(
            reference["difference"], key=lambda name: abs(reference["difference"][name])),
        logi_coh_on_photon_loss_trajectories=dict(
            share_by_scale={str(scale): value for scale, value in loss_share.items()},
            share_at_scale_1=loss_share[1.0],
            share_at_smallest_scale=loss_share[two_smallest[0]],
            smallest_scale=two_smallest[0],
            exponent_between_the_two_smallest_scales=float(
                np.log(logi_coh[two_smallest[1]] / logi_coh[two_smallest[0]])
                / np.log(two_smallest[1] / two_smallest[0])),
            scales_of_that_exponent=two_smallest,
        ),
    )


def plot(rows: list[dict]) -> None:
    r"""Draw the discrepancies against the incoherent-error scale.

    Parameters
    ----------
    rows : list of dict
        Outputs of :func:`evaluate`, in increasing order of the scale.

    Returns
    -------
    None
    """
    import matplotlib

    # headless backend: the script also runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_style.use_manuscript_style()
    scales = np.array([row["scale"] for row in rows])
    infidelity = np.array([1.0 - row["fidelity"]["F_T"] for row in rows])
    total = np.array([row["fidelity"]["F_T"] - row["fidelity"]["F_M"] for row in rows])
    # two panels with the same left and right edge, the legend below them
    fig = plt.figure(figsize=(plot_style.COLUMN_WIDTH, FIGURE_HEIGHT))
    axes = [plot_style.axes_in_inches(fig, AXES_LEFT, bottom, AXES_WIDTH, height)
            for bottom, height in (PANEL_A, PANEL_B)]
    axes[0].sharex(axes[1])
    axes[0].tick_params(labelbottom=False)

    def signed_curve(ax, values: np.ndarray, marker: str, color: str, lw: float,
                     zorder: float = 2) -> Line2D:
        r"""Curve of ``|values|`` with a marker at every scale, and its legend handle.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes, with logarithmic scales.
        values : np.ndarray
            Values at ``scales``; a filled marker is a positive value, an open
            one a negative value.
        marker : str
            Entry of ``plot_style.SERIES_MARKERS``.
        color : str
            Color of the curve and of its markers.
        lw : float
            Line width (pt).
        zorder : float
            Drawing order of the curve and of its markers.

        Returns
        -------
        matplotlib.lines.Line2D
            Handle without a label: the line with the filled marker.
        """
        ax.loglog(scales, np.abs(values), "-", color=color, lw=lw, zorder=zorder)
        for chosen, filled in ((values >= 0, True), (values < 0, False)):
            ax.loglog(scales[chosen], np.abs(values[chosen]), ls="none", zorder=zorder,
                      **plot_style.series_marker_style(marker, color, filled))
        return Line2D([], [], color=color, lw=lw,
                      **plot_style.series_marker_style(marker, color))

    # (a) the infidelity of the actual QEC dynamics, the difference between
    # the two models, and the discrepancy of each reduction
    ax = axes[0]
    # the infidelity in gray; the difference and the discrepancies share one
    # marker shape (they are all differences of fidelities) and are told
    # apart by their colors, the difference in black and above every other
    # curve (above the diamonds of the infidelity too)
    black, gray = plot_style.EMPHASIS, plot_style.NEUTRAL
    circle = plot_style.SERIES_MARKERS[0]
    actual = plot_style.model_marker_style("actual QEC dynamics", gray)
    dashed = dict(color=gray, lw=plot_style.CURVE_WIDTH, ls="--")
    ax.loglog(scales, infidelity, **dashed)
    ax.loglog(scales, infidelity, **actual)
    handles = [Line2D([], [], **dashed | actual | dict(ls="--")),
               signed_curve(ax, total, circle, black, plot_style.CURVE_WIDTH, zorder=6)]
    handles[0].set_label(r"$1-F_{\mathbb{T}}$")
    handles[1].set_label(r"$F_{\mathbb{T}}-F_{\mathbb{M}}$")
    for name, (symbol, color) in DISCREPANCIES.items():
        values = np.array([row["difference"][name] for row in rows])
        # a discrepancy that vanishes at every scale cannot be drawn on this axis
        if np.max(np.abs(values)) < PLOT_FLOOR:
            continue
        handles.append(signed_curve(ax, values, circle, color, plot_style.CURVE_WIDTH))
        handles[-1].set_label(symbol)
    # one name for all of them: each is a difference of two process fidelities,
    # the infidelity the one from the fidelity 1 of the ideal protocol
    ax.set_ylabel("fidelity difference")
    # the two power laws that the caption names, each under its discrepancy
    ends = np.array(GUIDE_RANGE)
    middle = float(np.sqrt(ends[0] * ends[1]))
    for name, (power, label, factor, at, side, gap) in SLOPE_GUIDES.items():
        values = np.abs([row["difference"][name] for row in rows])
        anchor = factor * np.exp(np.interp(np.log(middle), np.log(scales), np.log(values)))
        ax.loglog(ends, anchor * (ends / middle) ** power, **SLOPE_LINE)
        above = side == "above"
        ax.annotate(label, (at, anchor * (at / middle) ** power), xytext=(0, gap if above else -gap),
                    textcoords="offset points", ha="center", va="bottom" if above else "top",
                    fontsize=plot_style.SMALL_TEXT_SIZE, color=plot_style.NEUTRAL)
    plot_style.log_grid(ax)
    plot_style.panel_tag(ax, "(a)")

    # (b) the difference relative to the infidelity, in percent
    ax = axes[1]
    relative = 100 * total / infidelity
    ax.semilogx(scales, relative, "-", color=black,
                lw=plot_style.CURVE_WIDTH, **plot_style.series_marker_style(circle, black))
    # the highest point lies as far (inch) from the upper edge as the first and
    # last points lie from the left and right edges
    side = ax.margins()[0] / (1 + 2 * ax.margins()[0]) * AXES_WIDTH
    ax.set_ylim(0, relative.max() / (1 - side / PANEL_B[1]))
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel(plot_style.plain_text(
        r"$\dfrac{F_{\mathbb{T}}-F_{\mathbb{M}}}{1-F_{\mathbb{T}}}$ (%)"))
    ax.set_xlabel(r"incoherent-error scale $\xi$")
    # only the scale is on a logarithmic axis
    plot_style.log_grid(ax, axis="x")
    plot_style.panel_tag(ax, "(b)")
    fig.align_ylabels(axes)
    # the scale 1, the parameters of Ofek et al., in both panels, behind the data
    for ax in axes:
        ax.axvline(1.0, **plot_style.MARKED_VALUE_LINE)

    # The legend in the two empty corners of panel (a), on white so that the
    # line of the scale 1 does not run through the text: the two fidelity
    # differences above the data in the upper left, the five discrepancies
    # below the data in the lower right
    on_white = plot_style.LEGEND_STYLE | dict(
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
        borderpad=0.2, borderaxespad=0.2)
    upper = axes[0].legend(handles=handles[:2], loc="upper left", **on_white)
    axes[0].add_artist(upper)
    # Two one-column legends whose rows align: err-trans on top of the two
    # logical-error discrepancies in the lower-right corner, and to its left
    # the two leakage discrepancies, separated by the column spacing.
    # Separate legends leave the corner above the leakage pair free, so its
    # white frame hides no data
    right = axes[0].legend(handles=handles[4:], loc="lower right", **on_white)
    axes[0].add_artist(right)
    fig.draw_without_rendering()
    to_figure = fig.transFigure.inverted()
    box = right.get_frame().get_window_extent().transformed(to_figure)
    # the column spacing less the two inner paddings of the frames, in
    # figure fractions
    gap = ((on_white["columnspacing"] - 2 * on_white["borderpad"])
           * plot_style.TEXT_SIZE / 72 / fig.get_figwidth())

    def row_heights(legend) -> np.ndarray:
        r"""Heights of the centers of the legend lines, in figure fractions, top first.

        Parameters
        ----------
        legend : matplotlib.legend.Legend
            A drawn legend.

        Returns
        -------
        np.ndarray
            One height per entry.
        """
        return np.array([handle.get_window_extent().transformed(to_figure).get_points()[:, 1].mean()
                         for handle in legend.legend_handles])

    def left_legend(spacing: float, shift: float):
        r"""Legend of the two leakage discrepancies left of ``right``.

        Parameters
        ----------
        spacing : float
            Label spacing (font sizes).
        shift : float
            Height of its lower edge above that of ``right`` (figure fraction).

        Returns
        -------
        matplotlib.legend.Legend
            The legend, added to panel (a).
        """
        return axes[0].legend(handles=handles[2:4], loc="lower right",
                              bbox_to_anchor=(box.x0 - gap, box.y0 + shift),
                              bbox_transform=fig.transFigure,
                              **on_white | dict(borderaxespad=0.0, labelspacing=spacing))

    # The labels of the right legend have descenders (g, j) and those of the
    # left one do not, so equal label spacings leave the rows apart. The
    # spacing and the shift that put the two lines of the left legend on the
    # two lower rows of the right one follow from the drawn legends.
    left = left_legend(on_white["labelspacing"], 0.0)
    fig.draw_without_rendering()
    em = plot_style.TEXT_SIZE / 72 / fig.get_figheight()
    target, current = row_heights(right)[1:], row_heights(left)
    spacing = on_white["labelspacing"] + (np.diff(-target)[0] - np.diff(-current)[0]) / em
    left.remove()
    left = left_legend(spacing, 0.0)
    fig.draw_without_rendering()
    left.remove()
    left_legend(spacing, target[-1] - row_heights(left)[-1])
    FIGURE.parent.mkdir(exist_ok=True)
    # the canvas has the width of a column and is saved as it is
    plot_style.save_figure(fig, FIGURE, tight=False)


def main() -> None:
    r"""Evaluate every incoherent-error scale, then write the JSON record and the figure.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from accuracy-sweep.json")
    parser.add_argument("--update", action="store_true",
                        help="add the row without decoherence and recompute the numbers "
                             "from the rows of accuracy-sweep.json")
    args = parser.parse_args()

    if args.update:
        record = json.loads(RESULTS_JSON.read_text())
        record["without_decoherence"] = evaluate(NO_INCOHERENT_ERROR_SCALE)
        record["numbers"] = quoted_numbers(record["rows"])
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["numbers"], indent=1))
        print(json.dumps(record["without_decoherence"]["difference"], indent=1))
        print(f"wrote {RESULTS_JSON}")
        return

    if args.plot_only:
        rows = json.loads(RESULTS_JSON.read_text())["rows"]
    else:
        rows = []
        for scale in sorted(SCALES):
            row = evaluate(scale)
            rows.append(row)
            print(f"xi = {scale:g}: differences {row['difference']}, "
                  f"{row['seconds']:.0f} s", flush=True)
        _, reference, _ = device_model.exact_cat_parameters()
        record = {
            "description": "Chain-of-models process fidelities, discrepancies, and logical error probabilities of one cat-code QEC cycle across the incoherent-error scale at the parameters of Ofek et al.",
            "divided_by_scale": DIVIDED_BY_SCALE,
            "multiplied_by_scale": MULTIPLIED_BY_SCALE,
            "reference_values": {key: reference[key]
                                 for key in DIVIDED_BY_SCALE + MULTIPLIED_BY_SCALE},
            "chain": model_chain.CHAIN,
            "reductions": model_chain.REDUCTIONS,
            "manifestations": model_chain.MANIFESTATIONS,
            "numbers": quoted_numbers(rows),
            "without_decoherence": evaluate(NO_INCOHERENT_ERROR_SCALE),
            "rows": rows,
            "environment": device_model.environment_record(NUM_CPUS),
        }
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["numbers"], indent=1))
        print(f"wrote {RESULTS_JSON}")
    plot(rows)


if __name__ == "__main__":
    main()
