r"""Error budget by step of the cat-code protocol, and its accuracy at every step.

Evaluates the step-by-step fidelity losses of the actual evolution tree, the
logical evolution tree, and the failure Markov chain (split by logical
manifestation) at the parameters of Ofek et al. Writes ``error-budget.json``
and ``figs/error-budget.pdf``.

Usage: ``python error-budget.py`` (or ``--plot-only``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain
import plot_style

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "error-budget.json"
# output file without extension (plot_style.save_figure adds .pdf and .png)
FIGURE = HERE / "figs" / "error-budget"

# worker processes for the construction of the maps
NUM_CPUS = 1

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def compute() -> dict:
    r"""Fidelity losses of the three models and the split of the step contributions by logical manifestation.

    Returns
    -------
    dict
        The record that is written to ``error-budget.json``: ``steps`` (the
        names of the seven steps); ``F_by_step`` (the three process
        fidelities before the cycle and after each step);
        ``fidelity_loss_by_step`` (their decrease at each step);
        ``budget_entry_by_manifestation`` (the fidelity loss of the failure
        Markov chain at each step, split into leakage and the logical X, Y, Z
        errors); ``entry_over_infidelity`` (each term as a part of
        the overall failure probability :math:`1-F_{\mathbb M}`); ``loss_difference_M_minus_T`` (panel (b));
        ``one_cycle`` (the three process fidelities after the cycle, their
        differences, and the differences relative to the infidelity
        :math:`1-F_{\mathbb T}`); and the parameters and the environment of
        the run.
    """
    graph, sweep, parameters = device_model.build_actual_tree(n_cycles=1, num_cpus=NUM_CPUS)
    # the three models after every step, their losses, and the split of the
    # step contributions by logical manifestation
    budget = model_chain.losses_by_step(graph)
    fidelity, loss = budget["fidelity"], budget["loss"]
    split = budget["chain_by_manifestation"]
    by_manifestation = {key: values.tolist() for key, values in split.items()}

    final = {name: float(values[-1]) for name, values in fidelity.items()}
    infidelity = 1.0 - final["F_T"]
    differences = {
        "F_T_minus_F_M": final["F_T"] - final["F_M"],
        "F_T_minus_F_G": final["F_T"] - final["F_G"],
        "F_G_minus_F_M": final["F_G"] - final["F_M"],
    }
    return dict(
        description="Step-by-step fidelity losses of the actual evolution tree, logical evolution tree, and failure Markov chain (split by logical manifestation) at the parameters of Ofek et al.",
        steps=model_chain.STEP_NAMES,
        F_by_step={name: values.tolist() for name, values in fidelity.items()},
        fidelity_loss_by_step={name: values.tolist() for name, values in loss.items()},
        budget_entry_by_manifestation=by_manifestation,
        entry_over_infidelity=(loss["F_M"] / (1.0 - final["F_M"])).tolist(),
        leakage_over_infidelity=float(split["leakage"].sum() / (1.0 - final["F_M"])),
        loss_difference_M_minus_T=(loss["F_M"] - loss["F_T"]).tolist(),
        one_cycle=dict(
            **final,
            infidelity_actual_QEC_dynamics=infidelity,
            infidelity_failure_Markov_chain=1.0 - final["F_M"],
            **differences,
            **{f"{name}_over_infidelity": value / infidelity
               for name, value in differences.items()},
            t_cycle_ns=float(sweep["t_cycle"]),
            failure_rate_per_ms_actual_QEC_dynamics=float(
                device_model.failure_rate_per_ms(infidelity, float(sweep["t_cycle"]))),
        ),
        parameters={key: float(value) if isinstance(value, (int, float)) else value
                    for key, value in parameters.items()},
        settings=dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
        ),
        environment=device_model.environment_record(NUM_CPUS),
    )


def plot(record: dict) -> None:
    r"""Draw the two-panel figure from the record of ``error-budget.json``.

    Parameters
    ----------
    record : dict
        Output of :func:`compute`.

    Returns
    -------
    None
    """
    import matplotlib

    # headless backend: the script also runs without a display
    matplotlib.use("Agg")

    plot_style.use_manuscript_style()
    loss = {name: np.array(values) for name, values in record["fidelity_loss_by_step"].items()}

    # the canvas and the axes boxes of the two figures of this design
    fig, axes = plot_style.step_pair_figure()

    # (a) two bars per step: the actual QEC dynamics, and the failure Markov
    # chain, whose bar is split by logical manifestation (the error budget)
    ax = axes[0]
    actual = dict(parts={"actual QEC dynamics": loss["F_T"]},
                  colors={"actual QEC dynamics": plot_style.ACTUAL_FACE}, frame=False)
    chain = plot_style.chain_model(record["budget_entry_by_manifestation"])
    handles = plot_style.entries_by_step(ax, actual, chain)
    ax.legend(handles=handles[:1] + [plot_style.chain_handle()] + handles[1:],
              loc="upper right", **plot_style.BAR_LEGEND)
    plot_style.panel_tag(ax, "(a)")

    # (b) the difference of the two bars of (a), in units of 1e-4; the number
    # on a bar is the difference relative to the actual QEC dynamics
    ax = axes[1]
    plot_style.difference_by_step(
        ax, record["loss_difference_M_minus_T"], loss["F_T"],
        "deviation")
    plot_style.panel_tag(ax, "(b)")
    # the labels of the two vertical axes on one line
    fig.align_ylabels(axes)
    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from error-budget.json")
    args = parser.parse_args()
    if args.plot_only:
        record = json.loads(RESULTS_JSON.read_text())
    else:
        record = compute()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["one_cycle"], indent=1))
        print(f"wrote {RESULTS_JSON}")
    plot(record)
