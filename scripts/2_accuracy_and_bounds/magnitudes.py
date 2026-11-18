r"""Magnitudes of the seven neglected processes, accuracy bounds of the AEB, and step-by-step heatmap.

Evaluates the step-by-step diamond-norm magnitudes of the neglected processes
at incoherent-error scales :math:`\xi = 1` and :math:`\xi = 1/32` for the
cat-code cycle of Ofek et al., compares the resulting discrepancy bounds
against the exact discrepancies in ``data/accuracy-sweep.json``, writes
``data/magnitudes.json``, and draws ``figs/magnitudes-by-step.pdf``.

Usage:
  ``python scripts/2_accuracy_and_bounds/magnitudes.py --plot-only``
  ``python scripts/2_accuracy_and_bounds/magnitudes.py --scale 1``
  ``python scripts/2_accuracy_and_bounds/magnitudes.py --scale 1/32``
  ``python scripts/2_accuracy_and_bounds/magnitudes.py --combine``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

# headless backend: the script runs without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

import chencrafts.settings as cc_settings  # noqa: E402
from qec_aeb.magnitudes import (  # noqa: E402
    MAGNITUDES,
    SOLVER,
    SOLVER_EPS,
    SOLVER_MAX_ITERS,
    logical_step_magnitudes,
    magnitudes_by_step,
    physical_step_magnitudes,
    square_root_relations,
    step_groups,
    table_combinations,
)
from cat_code import device_model, plot_style  # noqa: E402

RESULTS_JSON = device_model.DATA_DIR / "magnitudes.json"
PARTS = device_model.DATA_DIR / "magnitudes-parts"
ACCURACY_SWEEP_JSON = device_model.DATA_DIR / "accuracy-sweep.json"
FIGURE = device_model.FIGS_DIR / "magnitudes-by-step"

# incoherent-error scales: label in the JSON file -> scale xi
SCALES = {"1": 1.0, "1/32": 1 / 32}
# number of steps of the cycle, the error-accumulation step, and the
# measurement step (steps counted from 1; one of each per QEC cycle)
N_STEPS = device_model.STEPS_PER_CYCLE
ERROR_ACCUMULATION_STEPS = [1]
MEASUREMENT_STEPS = [5]
LOGICAL_DIMENSION = 2
NUMBER_OF_SYNDROMES = 2
NUM_CPUS = 1
# a combination at most this part of the overall failure probability counts
# as much smaller than it (the reading of "<<" in the key "holds")
MUCH_SMALLER = 0.1
# a discrepancy below this value is rounding; no ratio is formed with it
DISCREPANCY_FLOOR = 1e-12

# ---- Figure constants ----
PLOT_SCALE = "1"
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
RETURNS = ("seep", "seep_coh")
CREATIONS = ("leak", "leak_coh")
AFTER_ERROR_ACCUMULATION = ("err_traj",)
AT_MEASUREMENT = ("unexp",)
COLOR_MAP = "viridis"
LOG_SCALE = True
LINEAR_STEP = 0.05
TICK_DECADES = 1
MAP_LEFT, ROW_HEIGHT = 0.62, 0.25
BAR_GAP, BAR_WIDTH, BAR_ROOM = 0.10, 0.10, 0.36
BOTTOM_MARGIN, TOP_MARGIN = 0.38, 0.08
MAP_WIDTH = plot_style.COLUMN_WIDTH - MAP_LEFT - BAR_GAP - BAR_WIDTH - BAR_ROOM
MAP_HEIGHT = ROW_HEIGHT * len(ROWS)
FIGURE_HEIGHT = BOTTOM_MARGIN + MAP_HEIGHT + TOP_MARGIN

cc_settings.PROGRESSBAR_DISABLED = True


def compute_scale(label: str) -> dict:
    r"""Every magnitude at one incoherent-error scale.

    Parameters
    ----------
    label : str
        Key of :data:`SCALES`.

    Returns
    -------
    dict
    """
    t0 = time.time()
    graph, _, _ = device_model.build_actual_tree(
        device_model.incoherent_error_scaled_parameters(SCALES[label]),
        n_cycles=1,
        num_cpus=NUM_CPUS,
    )
    seconds_tree = time.time() - t0
    print(f"scale {label}: tree in {seconds_tree:.0f} s", flush=True)
    physical = [
        physical_step_magnitudes(
            group,
            step_names=device_model.STEP_NAMES,
            error_accumulation_steps=ERROR_ACCUMULATION_STEPS,
        )
        for group in step_groups(graph)
    ]
    logical_tree = graph.logical_evolution_tree()
    logical = logical_step_magnitudes(logical_tree)
    return {
        "scale": SCALES[label],
        "label": label,
        "physical": physical,
        "logical": logical,
        "F_M": float(logical_tree.intermediate_success_prob_by_step()[-1]),
        "F_T": float(graph.process_fidelity_by_step()[-1]),
        "seconds_tree": seconds_tree,
        "seconds": time.time() - t0,
        "settings": dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            solver=SOLVER,
            solver_eps=SOLVER_EPS,
            solver_max_iters=SOLVER_MAX_ITERS,
        ),
        "environment": device_model.environment_record(NUM_CPUS),
    }


def part_path(label: str) -> Path:
    r"""Part file of one incoherent-error scale."""
    return PARTS / f"scale-{label.replace('/', '_')}.json"


def compare(combinations: dict, failure: float, difference: dict) -> dict:
    r"""Each combination against the overall failure probability, each bound against its discrepancy."""
    result = {}
    for name, row in combinations.items():
        discrepancy = abs(difference[name])
        resolved = discrepancy > DISCREPANCY_FLOOR
        conditions = {
            key: {
                "value": value,
                "over_failure": value / failure,
                "holds": value / failure <= MUCH_SMALLER,
                "over_discrepancy": value / discrepancy if resolved else None,
            }
            for key, value in row["conditions"].items()
        }
        result[name] = row | {
            "conditions": conditions,
            "discrepancy": discrepancy,
            "discrepancy_over_failure": discrepancy / failure,
            "bound_over_discrepancy": row["bound"] / discrepancy if resolved else None,
            "bound_over_failure": row["bound"] / failure,
        }
    return result


def quoted_numbers(scales: dict) -> dict:
    r"""Summary metrics of the bounds and conditions at each incoherent-error scale."""
    numbers = {}
    for label, entry in scales.items():
        ratios = {
            name: row["bound_over_discrepancy"]
            for name, row in entry["comparison"].items()
            if row["bound_over_discrepancy"] is not None
        }
        conditions = {
            f"{name}: {key}": condition["over_failure"]
            for name, row in entry["comparison"].items()
            for key, condition in row["conditions"].items()
        }
        numbers[label] = {
            "one_minus_F_M": entry["one_minus_F_M"],
            "largest_bound_over_discrepancy": max(ratios.items(), key=lambda item: item[1]),
            "smallest_bound_over_discrepancy": min(ratios.items(), key=lambda item: item[1]),
            "conditions_that_hold": {
                key: value for key, value in conditions.items() if value <= MUCH_SMALLER
            },
            "conditions_that_fail": {
                key: value for key, value in conditions.items() if value > MUCH_SMALLER
            },
            "largest_relative_gap_of_the_coherence_magnitudes": entry["largest_relative_gap"],
        }
    return numbers


def largest_relative_gap(part: dict) -> float:
    r"""Largest relative gap between the certified bounds of the coherence magnitudes."""
    records = [
        row[name]
        for row in part["physical"]
        for name in ("leak_coh", "seep_coh", "err_coh")
        if row[name] is not None
    ] + [row["logi_coh"] for row in part["logical"]]
    return max(
        (record["upper"] - record["lower"]) / record["upper"]
        for record in records
        if record["upper"] > 0
    )


def combine() -> None:
    r"""Read the part files and ``data/accuracy-sweep.json`` and write ``data/magnitudes.json``."""
    sweep = json.loads(ACCURACY_SWEEP_JSON.read_text())
    scales = {}
    for label, scale in SCALES.items():
        part = json.loads(part_path(label).read_text())
        row = next(row for row in sweep["rows"] if abs(row["scale"] - scale) < 1e-12)
        for model in ("F_M", "F_T"):
            if abs(part[model] - row["fidelity"][model]) > 1e-9:
                raise RuntimeError(
                    f"scale {label}: {model} {part[model]} differs from "
                    f"accuracy-sweep.json {row['fidelity'][model]}"
                )
        failure = 1 - row["fidelity"]["F_M"]
        eps = magnitudes_by_step(part, n_steps=N_STEPS)
        combinations = table_combinations(
            eps,
            n_steps=N_STEPS,
            error_accumulation_steps=ERROR_ACCUMULATION_STEPS,
            measurement_steps=MEASUREMENT_STEPS,
        )
        scales[label] = {
            "scale": scale,
            "one_minus_F_M": failure,
            "F_T_minus_F_M": row["fidelity"]["F_T"] - row["fidelity"]["F_M"],
            "magnitudes_by_step": eps,
            "comparison": compare(combinations, failure, row["difference"]),
            "square_root_relations": square_root_relations(
                eps,
                n_steps=N_STEPS,
                error_accumulation_steps=ERROR_ACCUMULATION_STEPS,
                logical_dimension=LOGICAL_DIMENSION,
                number_of_syndromes=NUMBER_OF_SYNDROMES,
            ),
            "largest_relative_gap": largest_relative_gap(part),
            "computation": part,
        }
    record = {
        "description": "Step-by-step diamond-norm magnitudes of the neglected processes and discrepancy bounds for one cat-code QEC cycle at incoherent-error scales 1 and 1/32.",
        "much_smaller": MUCH_SMALLER,
        "discrepancy_floor": DISCREPANCY_FLOOR,
        "magnitudes": {
            name: {"kind": kind, "symbol": symbol}
            for name, (kind, symbol) in MAGNITUDES.items()
        },
        "steps": device_model.STEP_NAMES,
        "error_accumulation_steps": ERROR_ACCUMULATION_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "solver": {"name": SOLVER, "eps": SOLVER_EPS, "max_iters": SOLVER_MAX_ITERS},
        "numbers": quoted_numbers(scales),
        "scales": scales,
    }
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")


def plot() -> None:
    r"""Read ``data/magnitudes.json`` at incoherent-error scale 1 and draw ``figs/magnitudes-by-step.pdf``."""
    data = json.loads(RESULTS_JSON.read_text())
    magnitudes = data["scales"][PLOT_SCALE]["magnitudes_by_step"]
    error_accumulation = [n - 1 for n in data["error_accumulation_steps"]]
    measurement = [n - 1 for n in data["measurement_steps"]]
    values = np.array([magnitudes[key] for key in ROWS], dtype=float)
    shown = np.ma.array(values)
    row = {key: list(ROWS).index(key) for key in ROWS}
    others = [n for n in range(values.shape[1]) if n not in measurement]
    for keys, columns in (
        (RETURNS, [0]),
        (CREATIONS, [-1]),
        (AFTER_ERROR_ACCUMULATION, error_accumulation),
        (AT_MEASUREMENT, others),
    ):
        for key in keys:
            shown[row[key], columns] = np.ma.masked
    plot_style.use_manuscript_style()
    fig = plt.figure(figsize=(plot_style.COLUMN_WIDTH, FIGURE_HEIGHT))
    ax = plot_style.axes_in_inches(fig, MAP_LEFT, BOTTOM_MARGIN, MAP_WIDTH, MAP_HEIGHT)
    color_axes = plot_style.axes_in_inches(
        fig, MAP_LEFT + MAP_WIDTH + BAR_GAP, BOTTOM_MARGIN, BAR_WIDTH, MAP_HEIGHT
    )
    cmap = plot_style.chalk_colormap(COLOR_MAP)
    cmap.set_bad("white")
    steps = np.arange(1, len(plot_style.STEP_LABELS) + 1)
    rows = np.arange(len(ROWS))
    if LOG_SCALE:
        smallest = shown[shown > 0].min()
        highest = np.ceil(np.log10(shown.max()))
        norm = LogNorm(vmin=smallest, vmax=10.0**highest)
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
        bar = fig.colorbar(mesh, cax=color_axes)
        bar.set_ticks(10.0 ** np.arange(highest, np.log10(smallest), -TICK_DECADES))
        bar.minorticks_off()
    else:
        fig.colorbar(mesh, cax=color_axes)
    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", choices=list(SCALES), help="compute one incoherent-error scale")
    parser.add_argument("--combine", action="store_true", help="write magnitudes.json from the parts")
    parser.add_argument("--plot-only", action="store_true", help="draw figs/magnitudes-by-step.pdf from stored data")
    arguments = parser.parse_args()
    if arguments.plot_only:
        plot()
    elif arguments.combine:
        combine()
        plot()
    elif arguments.scale:
        PARTS.mkdir(exist_ok=True)
        part = compute_scale(arguments.scale)
        part_path(arguments.scale).write_text(json.dumps(part, indent=1))
        print(f"wrote {part_path(arguments.scale)} in {part['seconds']:.0f} s")
    else:
        plot()
