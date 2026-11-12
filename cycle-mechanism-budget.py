r"""Analytical formula of the cat-code protocol, by step and by mechanism, against the numerical computation.

Evaluates the analytical error budget by mechanism and by step alongside the
step-by-step fidelity losses of the failure Markov chain and actual QEC
dynamics at the parameters of Ofek et al. Writes
``cycle-mechanism-budget.json`` and ``figs/cycle-mechanism-budget.pdf``.

Usage: ``python cycle-mechanism-budget.py`` (or ``--plot-only``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# headless backend: the script runs without a display
matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import qec_aeb as bsq  # noqa: E402
import chencrafts.settings as cc_settings  # noqa: E402

# importing device_model sets the symmetric orthonormalization of the error
# states before any tree is built
import device_model  # noqa: E402
import plot_style  # noqa: E402
from budget_tools import STEP_KEYS, formula_terms  # noqa: E402

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "cycle-mechanism-budget.json"
# a mechanism below this part of the formula is drawn with the other small ones
MERGE_BELOW = 0.02
# output of the figure, without extension
FIGURE = HERE / "figs" / "cycle-mechanism-budget"
# the four entries that the text calls dominant
DOMINANT_STEPS = (1, 3, 5, 6)
# worker processes for the construction of the dynamical maps
NUM_CPUS = 1


def compute() -> dict:
    r"""Evaluate the formula and the two numerical computations of one cycle.

    Returns
    -------
    dict
        The record that is written to ``cycle-mechanism-budget.json``.
    """
    sweep, simulation_parameters, parameters = device_model.build_sweep()
    ingredients = sweep.full_dict()

    # analytical formula: every term, then the sums by step and by mechanism
    terms = formula_terms(ingredients)
    steps = sorted(STEP_KEYS)
    formula_by_step = np.array([sum(t["value"] for t in terms if t["step"] == n) for n in steps])
    formula_total = float(device_model.formula_infidelity(ingredients))
    mechanisms = list(dict.fromkeys(t["mechanism"] for t in terms))
    by_mechanism = {
        m: dict(
            total=sum(t["value"] for t in terms if t["mechanism"] == m),
            by_step=[sum(t["value"] for t in terms if t["mechanism"] == m and t["step"] == n)
                     for n in steps])
        for m in mechanisms}
    # the mechanisms in the order of their contributions, largest first
    order = sorted(mechanisms, key=lambda m: -by_mechanism[m]["total"])
    p_even, p_odd = (float(p) for p in bsq.cat_branch_probabilities(**ingredients))

    # numerical computation: the actual evolution tree of one cycle and the
    # failure Markov chain that follows from its logical evolution tree
    graph = device_model.build_tree_from_sweep(sweep, simulation_parameters, num_cpus=NUM_CPUS)
    numerical = device_model.one_cycle_fidelity_losses(graph)
    actual, markov = numerical[device_model.ACTUAL], numerical[device_model.CHAIN]
    F_actual = np.asarray(actual["after_each_step"], float)
    F_markov = np.asarray(markov["after_each_step"], float)
    loss_actual = np.asarray(actual["by_step"], float)
    loss_markov = np.asarray(markov["by_step"], float)
    infidelity_markov, infidelity_actual = markov["infidelity"], actual["infidelity"]

    # summary metrics
    dominant = [n - 1 for n in DOMINANT_STEPS]
    excitation = by_mechanism["ancilla excitation"]
    quoted = dict(
        formula_over_markov_chain_minus_1_cycle=formula_total / infidelity_markov - 1,
        formula_over_actual_QEC_dynamics_minus_1_cycle=formula_total / infidelity_actual - 1,
        dominant_steps=[STEP_KEYS[n] for n in DOMINANT_STEPS],
        formula_over_markov_chain_minus_1_dominant_steps={
            STEP_KEYS[n]: float(formula_by_step[n - 1] / loss_markov[n - 1] - 1)
            for n in DOMINANT_STEPS},
        largest_deviation_of_a_dominant_step=float(np.max(np.abs(
            formula_by_step[dominant] / loss_markov[dominant] - 1))),
        formula_over_markov_chain_minus_1_sum_of_dominant_steps=float(
            formula_by_step[dominant].sum() / loss_markov[dominant].sum() - 1),
        dominant_steps_over_cycle_infidelity=dict(
            failure_Markov_chain=float(loss_markov[dominant].sum() / infidelity_markov),
            analytical_formula=float(formula_by_step[dominant].sum() / formula_total)),
        mechanisms_largest_first=order,
        mechanism_over_formula_infidelity={
            m: by_mechanism[m]["total"] / formula_total for m in order},
        ancilla_excitation_over_photon_loss=(
            excitation["total"] / by_mechanism["photon loss"]["total"]),
        ancilla_relaxation_over_ancilla_excitation=(
            by_mechanism["ancilla relaxation"]["total"] / excitation["total"]),
        idle_part_of_ancilla_excitation=excitation["by_step"][0] / excitation["total"],
    )

    return dict(
        description="Analytical error budget by mechanism and by step compared with the failure Markov chain and actual QEC dynamics for one cat-code QEC cycle at the parameters of Ofek et al.",
        settings=dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
            orthonormalization=bsq.QEC_graph.StateNode.ORTHOGONALIZE_METHOD,
        ),
        environment=device_model.environment_record(NUM_CPUS),
        parameters={k: float(v) for k, v in parameters.items()},
        t_cycle=float(sweep["t_cycle"]),
        error_space_probabilities_formula=dict(even_parity=p_even, odd_parity=p_odd),
        steps=[STEP_KEYS[n] for n in steps],
        formula=dict(
            terms=terms,
            by_step=formula_by_step.tolist(),
            by_mechanism=by_mechanism,
            cycle_infidelity=formula_total),
        failure_Markov_chain=dict(
            success_probability_after_each_step=F_markov.tolist(),
            fidelity_loss_by_step=loss_markov.tolist(),
            cycle_infidelity=infidelity_markov),
        actual_QEC_dynamics=dict(
            process_fidelity_after_each_step=F_actual.tolist(),
            fidelity_loss_by_step=loss_actual.tolist(),
            cycle_infidelity=infidelity_actual),
        formula_over_markov_chain_minus_1_by_step=(formula_by_step / loss_markov - 1).tolist(),
        quoted=quoted,
    )


def plot(record: dict) -> None:
    r"""Draw the two panels from the record, in the design of ``error-budget.py``.

    (a) Two bars per step: the failure Markov chain, in the black frame of
    the figure of ``error-budget.py``, and the analytical formula, split by
    mechanism. (b) The difference of the two bars.

    Parameters
    ----------
    record : dict
        Output of :func:`compute`, or the content of the JSON file.

    Returns
    -------
    None
    """
    plot_style.use_manuscript_style()
    formula = record["formula"]
    chain_loss = np.array(record["failure_Markov_chain"]["fidelity_loss_by_step"])
    # the chain is the black frame of the figure of error-budget.py, here
    # without its split by logical manifestation
    chain = dict(parts={plot_style.CHAIN_LABEL: chain_loss},
                 colors={plot_style.CHAIN_LABEL: "white"}, frame=True)
    # the mechanisms above MERGE_BELOW of the formula: ancilla dephasing on top
    # of the bars and the legend, then the order of plot_style.MECHANISM_COLORS;
    # the others are added up
    part = record["quoted"]["mechanism_over_formula_infidelity"]
    by_step = {m: np.array(v["by_step"], dtype=float)
               for m, v in formula["by_mechanism"].items()}
    order = sorted(plot_style.MECHANISM_COLORS, key=lambda m: m != "ancilla dephasing")
    parts = {m: by_step[m] for m in order
             if m in by_step and part[m] >= MERGE_BELOW}
    parts[plot_style.OTHER_MECHANISMS] = sum(v for m, v in by_step.items() if m not in parts)
    formula_model = dict(parts=parts, colors=plot_style.MECHANISM_COLORS, frame=False)

    # the canvas and the axes boxes of the figure of error-budget.py
    fig, axes = plot_style.step_pair_figure()

    # (a) the chain keeps the place next to the model it is compared with
    ax = axes[0]
    handles = plot_style.entries_by_step(ax, chain, formula_model)
    title = Patch(facecolor="none", edgecolor="none", label="AEB:")
    ax.legend(handles=[plot_style.chain_handle(), title] + handles[1:],
              loc="upper right", **plot_style.BAR_LEGEND)
    plot_style.panel_tag(ax, "(a)")

    # (b) formula minus chain, in units of 1e-4; the number on a bar is the
    # difference relative to the chain
    ax = axes[1]
    plot_style.difference_by_step(
        ax, np.array(formula["by_step"]) - chain_loss, chain_loss,
        "deviation")
    plot_style.panel_tag(ax, "(b)")
    # the labels of the two vertical axes on one line
    fig.align_ylabels(axes)

    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from the JSON file")
    if parser.parse_args().plot_only:
        record = json.loads(RESULTS_JSON.read_text())
    else:
        record = compute()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["quoted"], indent=1))
        print(f"wrote {RESULTS_JSON}")
    plot(record)
