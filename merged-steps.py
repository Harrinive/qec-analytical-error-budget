r"""Accuracy of the error budget by step when the parity mapping is one step.

Evaluates the chain of models across the incoherent-error scale for the
seven-step cat-code cycle and for the five-step cycle in which steps 2 to 4
(:math:`\hat X_{\pi/2}` pulse, parity-mapping idling, :math:`\hat X_{-\pi/2}`
pulse) are merged into a single parity-mapping step. Writes
``merged-steps.json``.

Usage: ``python merged-steps.py``.
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

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "merged-steps.json"
# the seven-step results that this script reproduces and compares with
ACCURACY_SWEEP_JSON = HERE / "accuracy-sweep.json"

# steps of the chencrafts builder that become one: X_pi/2 pulse,
# parity-mapping idling, X_-pi/2 pulse (steps 2 to 4)
MERGED_STEPS = ["qubit_gate_1", "parity_mapping", "qubit_gate_2"]
# plain names of the five steps, in the order of model_chain.STEP_NAMES
FIVE_STEP_NAMES = [model_chain.STEP_NAMES[0], "parity mapping", *model_chain.STEP_NAMES[4:]]
# step after which the error spaces of the two cycles must agree: the merged
# step in the five-step cycle, the X_-pi/2 pulse in the seven-step cycle
AFTER_MERGED = {"five_steps": 2, "seven_steps": 4}
# the two cycles, as keys of the result file
CYCLES = ("seven_steps", "five_steps")
# worker processes for the construction of the maps
NUM_CPUS = 1

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def merge_parity_mapping(builder) -> None:
    r"""Make steps 2 to 4 of the cycle one step of the tree builder.

    Parameters
    ----------
    builder : qec_aeb.QEC_graph.FullCatTreeBuilder
        Tree builder before its maps are built; changed in place.

    Returns
    -------
    None
    """
    builder.merge_steps(MERGED_STEPS)


def node_after(graph, steps: int):
    r"""The node of a tree after a number of steps without a measurement.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.
    steps : int
        Number of steps from the initial node; none of them may branch.

    Returns
    -------
    StateNode
        The only node after ``steps`` steps.
    """
    node = graph.nodes[0]
    for _ in range(steps):
        # before the measurement every node has one outgoing edge
        if len(node.out_edges) != 1:
            raise RuntimeError("the tree branches before the requested step")
        node = node.out_edges[0].final_state
    return node


def evaluate(scale: float) -> dict:
    r"""Chain of models of the seven-step and of the five-step cycle at one scale.

    Parameters
    ----------
    scale : float
        Incoherent-error scale :math:`\xi`.

    Returns
    -------
    dict
        ``scale``; for each of :data:`CYCLES`, the output of
        ``model_chain.chain_fidelities`` and ``by_step``;
        ``error_states_after_parity_mapping``; ``seconds``.
    """
    t0 = time.time()
    sweep, simulation_parameters, _ = device_model.build_sweep(
        device_model.incoherent_error_scaled_parameters(scale))
    # the same dressed sweep for both cycles; only the step table differs
    graphs = {
        "seven_steps": device_model.build_tree_from_sweep(
            sweep, simulation_parameters, num_cpus=NUM_CPUS),
        "five_steps": device_model.build_tree_from_sweep(
            sweep, simulation_parameters, num_cpus=NUM_CPUS,
            prepare_builder=merge_parity_mapping),
    }
    row: dict = {"scale": scale}
    for name, graph in graphs.items():
        row[name] = model_chain.chain_fidelities(graph) | {
            "by_step": {model: values.tolist() for model, values
                        in model_chain.fidelities_by_step(graph).items()}}
    # the error states after the merged step must be those after step 4
    encoders = {name: node_after(graphs[name], steps).ideal_encoders()
                for name, steps in AFTER_MERGED.items()}
    row["error_states_after_parity_mapping"] = max(
        float(np.abs((five - seven).full()).max())
        for five, seven in zip(encoders["five_steps"], encoders["seven_steps"], strict=True))
    row["seconds"] = time.time() - t0
    return row


def against_accuracy_sweep(rows: list[dict]) -> None:
    r"""Add to each row the largest difference from the seven-step results stored before.

    Parameters
    ----------
    rows : list of dict
        Outputs of :func:`evaluate`; each gets the entry
        ``seven_steps_against_accuracy_sweep`` (``None`` for a scale that
        ``accuracy-sweep.json`` does not hold).

    Returns
    -------
    None
    """
    stored = {row["scale"]: row["fidelity"]
              for row in json.loads(ACCURACY_SWEEP_JSON.read_text())["rows"]}
    for row in rows:
        reference = stored.get(row["scale"])
        row["seven_steps_against_accuracy_sweep"] = None if reference is None else max(
            abs(row["seven_steps"]["fidelity"][model] - reference[model])
            for model in model_chain.CHAIN)


def quoted_numbers(rows: list[dict]) -> dict:
    r"""Summary metrics comparing the seven-step and five-step cycles across scales.

    Parameters
    ----------
    rows : list of dict
        Outputs of :func:`evaluate`, in increasing order of the scale; they
        must include the scale 1.

    Returns
    -------
    dict
        For each of :data:`CYCLES`: ``deviation_over_infidelity``,
        :math:`(F_{\mathbb T}-F_{\mathbb M})/(1-F_{\mathbb T})`, and
        ``leak_coh_over_infidelity``,
        :math:`\delta_{\mathrm{leak\text{-}coh}}/(1-F_{\mathbb T})`, each by
        scale and at :math:`\xi=1` and at the smallest :math:`\xi`;
        ``leak_coh`` at both scales. ``change``: the fall of the deviation
        relative to the infidelity from seven to five steps (difference, and
        as a fraction of the seven-step value), at both scales and by scale;
        ``leak_coh_share_of_the_change``: the fall of
        :math:`\delta_{\mathrm{leak\text{-}coh}}` over the fall of
        :math:`F_{\mathbb T}-F_{\mathbb M}`, by scale and at both scales;
        ``discrepancy_change_over_infidelity``: the fall of each discrepancy
        relative to the infidelity at both scales. ``largest_F_T_difference``:
        the largest difference of :math:`F_{\mathbb T}` between the two
        cycles over the scales; ``largest_error_state_difference`` and
        ``largest_difference_from_accuracy_sweep``: the largest of the two
        checks of the rows.
    """
    by_scale = {row["scale"]: row for row in rows}
    smallest = min(by_scale)
    ends = {"at_scale_1": 1.0, "at_smallest_scale": smallest}

    def per_scale(function) -> dict:
        r"""A quantity at every scale and at the two quoted scales.

        Parameters
        ----------
        function : callable
            Maps a row to a float.

        Returns
        -------
        dict
            ``by_scale`` (scale as a string to the value), ``at_scale_1``,
            ``at_smallest_scale``.
        """
        values = {scale: function(row) for scale, row in by_scale.items()}
        return {"by_scale": {str(scale): value for scale, value in values.items()}} | {
            end: values[scale] for end, scale in ends.items()}

    def infidelity(row: dict) -> float:
        r"""One minus the process fidelity of the actual QEC dynamics (seven-step tree).

        Parameters
        ----------
        row : dict
            Output of :func:`evaluate`.

        Returns
        -------
        float
        """
        return 1.0 - row["seven_steps"]["fidelity"]["F_T"]

    def deviation(row: dict, cycle: str) -> float:
        r""":math:`F_{\mathbb T}-F_{\mathbb M}` of one cycle.

        Parameters
        ----------
        row : dict
            Output of :func:`evaluate`.
        cycle : str
            One of :data:`CYCLES`.

        Returns
        -------
        float
        """
        fidelity = row[cycle]["fidelity"]
        return fidelity["F_T"] - fidelity["F_M"]

    numbers: dict = {}
    for cycle in CYCLES:
        numbers[cycle] = dict(
            deviation_over_infidelity=per_scale(
                lambda row, c=cycle: deviation(row, c) / infidelity(row)),
            leak_coh_over_infidelity=per_scale(
                lambda row, c=cycle: row[c]["difference"]["leak_coh"] / infidelity(row)),
            leak_coh={end: by_scale[scale][cycle]["difference"]["leak_coh"]
                      for end, scale in ends.items()},
        )
    seven, five = "seven_steps", "five_steps"
    numbers["change"] = dict(
        deviation_over_infidelity_fall=per_scale(
            lambda row: (deviation(row, seven) - deviation(row, five)) / infidelity(row)),
        relative_fall_of_the_deviation=per_scale(
            lambda row: 1.0 - deviation(row, five) / deviation(row, seven)),
        leak_coh_share_of_the_change=per_scale(
            lambda row: (row[seven]["difference"]["leak_coh"]
                         - row[five]["difference"]["leak_coh"])
            / (deviation(row, seven) - deviation(row, five))),
        discrepancy_change_over_infidelity={
            end: {name: (by_scale[scale][seven]["difference"][name]
                         - by_scale[scale][five]["difference"][name])
                  / infidelity(by_scale[scale])
                  for name, _, _ in model_chain.REDUCTIONS}
            for end, scale in ends.items()},
    )
    numbers["checks"] = dict(
        largest_F_T_difference=max(
            abs(row[seven]["fidelity"]["F_T"] - row[five]["fidelity"]["F_T"]) for row in rows),
        largest_error_state_difference=max(
            row["error_states_after_parity_mapping"] for row in rows),
        largest_difference_from_accuracy_sweep=max(
            (row["seven_steps_against_accuracy_sweep"] for row in rows
             if row.get("seven_steps_against_accuracy_sweep") is not None),
            default=None),
    )
    return numbers


def main() -> None:
    r"""Evaluate both cycles at every scale of ``accuracy-sweep.json`` and write the JSON record.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    # the scales of the seven-step sweep, so that every row has its reference
    scales = sorted(row["scale"] for row in json.loads(ACCURACY_SWEEP_JSON.read_text())["rows"])
    rows = []
    for scale in scales:
        row = evaluate(scale)
        rows.append(row)
        print(f"xi = {scale:g}: deviation over infidelity "
              + ", ".join(f"{cycle} {(row[cycle]['fidelity']['F_T'] - row[cycle]['fidelity']['F_M']) / (1 - row[cycle]['fidelity']['F_T']):.4f}"
                          for cycle in CYCLES)
              + f"; leak_coh {row['seven_steps']['difference']['leak_coh']:.3e} -> "
              f"{row['five_steps']['difference']['leak_coh']:.3e}; {row['seconds']:.0f} s",
              flush=True)
    against_accuracy_sweep(rows)
    record = {
        "description": "Chain-of-models process fidelities and discrepancies across the incoherent-error scale for the seven-step cat-code cycle and the five-step cycle with steps 2 to 4 merged.",
        "merged_steps": MERGED_STEPS,
        "steps": {"seven_steps": model_chain.STEP_NAMES, "five_steps": FIVE_STEP_NAMES},
        "chain": model_chain.CHAIN,
        "reductions": model_chain.REDUCTIONS,
        "manifestations": model_chain.MANIFESTATIONS,
        "numbers": quoted_numbers(rows),
        "rows": rows,
        "environment": device_model.environment_record(NUM_CPUS),
    }
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["numbers"], indent=1))
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
