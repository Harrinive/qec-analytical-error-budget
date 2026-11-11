r"""Which pairs of steps carry the discrepancy :math:`\delta_{\mathrm{leak\text{-}coh}}`.

Decomposes the error--leakage coherence discrepancy
:math:`\delta_{\mathrm{leak\text{-}coh}}` into exact contributions from each
pair of steps :math:`(i, j)` (:math:`i < j`), with the full noise model and
without decoherence at the parameters of Ofek et al. Writes
``leak-coh-attribution.json``.

Usage: ``python leak-coh-attribution.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import chencrafts.settings as cc_settings
from qec_aeb.QEC_graph import process_matrix_diagonal

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain

RESULTS_JSON = Path(__file__).resolve().parent / "leak-coh-attribution.json"

# the two computations: name -> mechanisms kept (None: the full noise model)
MODELS = {"full_noise_model": None, "no_decoherence": []}
# the pair of the two X(+-pi/2) pulses
PULSE_PAIR = (2, 4)
# worker processes for the construction of the maps
NUM_CPUS = 1
POPULATIONS = (model_chain.WITHIN_ERROR, model_chain.WITHIN_LEAKAGE)
COHERENCE = (model_chain.COHERENCE,)

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def fidelity_keeping(graph, kept_after_step) -> float:
    r"""Process fidelity of the ELM with chosen components kept after each step.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle.
    kept_after_step : callable
        Step index to the tuple of components kept after that step, or to
        ``None`` for every component
        (``model_chain.elm_keeping_components``).

    Returns
    -------
    float
        The entry :math:`\mathcal X_{II}` of the process matrix of the ELM.
        The ELM of a single term of the expansion is not completely
        positive, and this entry can be negative.
    """
    return float(process_matrix_diagonal(
        model_chain.elm_keeping_components(graph, kept_after_step))[0])


def pair_contributions(graph) -> dict:
    r"""Contribution of every pair of steps to :math:`\delta_{\mathrm{leak\text{-}coh}}`.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle.

    Returns
    -------
    dict
        ``delta_leak_coh`` (the process fidelity with every component kept
        minus that with the error--leakage coherence removed after every
        step); ``pairs`` (one row for each pair :math:`i<j`: the two steps,
        their names, the contribution, and its share of the discrepancy);
        ``sum_over_pairs``; ``remainder`` (the discrepancy minus that sum:
        the terms in which coherence is created and returned more than
        once) and ``remainder_share``; ``created_at_step`` and
        ``returned_at_step`` (the contributions added over the other step of
        the pair, as shares of the discrepancy).
    """
    n_steps = device_model.STEPS_PER_CYCLE
    delta = (fidelity_keeping(graph, lambda step: None)
             - fidelity_keeping(graph, lambda step: POPULATIONS))
    pairs = []
    for created in range(1, n_steps):
        for returned in range(created + 1, n_steps + 1):
            # coherence after the steps created, ..., returned - 1, and the
            # components within the two subspaces after every other step
            value = fidelity_keeping(
                graph, lambda step, i=created, j=returned:
                COHERENCE if i <= step < j else POPULATIONS)
            pairs.append(dict(
                created_at_step=created, returned_at_step=returned,
                created_at=model_chain.STEP_NAMES[created - 1],
                returned_at=model_chain.STEP_NAMES[returned - 1],
                contribution=value, share=value / delta))
    total = sum(row["contribution"] for row in pairs)
    return dict(
        delta_leak_coh=delta,
        pairs=pairs,
        sum_over_pairs=total,
        remainder=delta - total,
        remainder_share=(delta - total) / delta,
        created_at_step={
            model_chain.STEP_NAMES[step - 1]: sum(
                row["share"] for row in pairs if row["created_at_step"] == step)
            for step in range(1, n_steps)},
        returned_at_step={
            model_chain.STEP_NAMES[step - 1]: sum(
                row["share"] for row in pairs if row["returned_at_step"] == step)
            for step in range(2, n_steps + 1)},
    )


if __name__ == "__main__":
    results = {}
    for name, kept in MODELS.items():
        sweep, simulation_parameters = model_chain.sweep_with_mechanisms(kept)
        graph = device_model.build_tree_from_sweep(
            sweep, simulation_parameters, num_cpus=NUM_CPUS)
        results[name] = pair_contributions(graph)
        # the infidelity of the same tree, for the size of the discrepancy
        results[name]["infidelity_actual_QEC_dynamics"] = (
            device_model.one_cycle_infidelities(graph)["actual_QEC_dynamics"])
        largest = sorted(results[name]["pairs"], key=lambda row: -abs(row["contribution"]))[:4]
        print(f"{name}: delta_leak_coh = {results[name]['delta_leak_coh']:.4e}, remainder "
              f"{results[name]['remainder_share']:+.2%}; largest pairs: "
              + "; ".join(f"({row['created_at_step']},{row['returned_at_step']}) "
                          f"{row['share']:+.1%}" for row in largest), flush=True)

    def pulse_pair(name: str) -> dict:
        r"""Row of the pair of the two X(+-pi/2) pulses in one computation.

        Parameters
        ----------
        name : str
            Key of :data:`MODELS`.

        Returns
        -------
        dict
            The row of ``pairs`` with the steps of :data:`PULSE_PAIR`.
        """
        return next(row for row in results[name]["pairs"]
                    if (row["created_at_step"], row["returned_at_step"]) == PULSE_PAIR)

    full, closed = results["full_noise_model"], results["no_decoherence"]
    record = dict(
        description="Exact pair-of-steps decomposition of delta_leak-coh for one cat-code QEC cycle at the parameters of Ofek et al., with the full noise model and without decoherence.",
        steps=model_chain.STEP_NAMES,
        numbers=dict(
            pulse_pair_steps=list(PULSE_PAIR),
            pulse_pair_share_full_noise_model=pulse_pair("full_noise_model")["share"],
            pulse_pair_share_no_decoherence=pulse_pair("no_decoherence")["share"],
            pulse_pair_contribution_full_noise_model=pulse_pair("full_noise_model")["contribution"],
            pulse_pair_contribution_no_decoherence=pulse_pair("no_decoherence")["contribution"],
            delta_leak_coh_full_noise_model=full["delta_leak_coh"],
            delta_leak_coh_no_decoherence=closed["delta_leak_coh"],
            delta_leak_coh_no_decoherence_over_full=(
                closed["delta_leak_coh"] / full["delta_leak_coh"]),
        ),
        **results,
        settings=dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
        ),
        environment=device_model.environment_record(NUM_CPUS),
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["numbers"], indent=1))
    print(f"wrote {RESULTS_JSON}")
