r"""Process fidelity of the actual QEC dynamics over several QEC cycles, at two parameter sets.

Evaluates the multi-cycle process fidelity :math:`F_{\mathbb T}`, fits its
exponential decay rate, and tracks the second-cycle return of late first-cycle
errors at the parameters of Ofek et al. and at the optimized parameters. Writes
``multi-cycle-fidelity.json``.

Usage: ``python multi-cycle-fidelity.py`` (or ``--update``, ``--add-late-error-return``).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import chencrafts.settings as cc_settings
import chencrafts.toolbox as tb
from qec_aeb.QEC_graph import process_matrix_diagonal

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain
from device_model import OFEK, OPTIMIZED

RESULTS_JSON = Path(__file__).resolve().parent / "multi-cycle-fidelity.json"

# |beta|^2 at which the window of each tree ends: the overlap of two adjacent
# coherent states is then 1/e, and the codeword |0_L> is 85 % vacuum
DISPLACEMENT_SQUARED_AT_WINDOW_END = 1.0
# offset of the fit: process fidelity of the completely depolarizing map on a
# qubit, 1/d^2 with d = 2
FIT_OFFSET = 0.25
# amplitude of the fit: the process fidelity of the encoded state is 1 at
# time zero
FIT_AMPLITUDE = 1.0 - FIT_OFFSET

# The return of late errors (key ``late_error_return``). A tree of two QEC
# cycles: the first is the one-cycle protocol, the second holds the parity
# measurement that records a photon lost late in the first
LATE_ERROR_CYCLES = 2
# steps after which the components of the state at the end of the first cycle
# are read: the measurement of the second cycle, and the end of that cycle
READ_STEPS = {
    "after_measurement_of_cycle_2":
        device_model.STEPS_PER_CYCLE + model_chain.STEP_NAMES.index("measurement") + 1,
    "end_of_cycle_2": LATE_ERROR_CYCLES * device_model.STEPS_PER_CYCLE,
}
# the three components of the state at the end of the first cycle; they add
# up to the state
FIRST_CYCLE_COMPONENTS = {
    "error_spaces": model_chain.WITHIN_ERROR,
    "leakage": model_chain.WITHIN_LEAKAGE,
    "error_leakage_coherence": model_chain.COHERENCE,
}
# build methods of the tree builder for the two steps after the parity mapping
# whose photon loss is switched off in the comparison
LATE_STEP_BUILDS = ("build_qubit_measurement_process", "build_feedback_latency_process")
# the photon-loss terms of the same two steps in formula-terms.json
FORMULA_TERMS_JSON = Path(__file__).resolve().parent / "formula-terms.json"
FORMULA_LATE_STEPS = ("measurement", "feedback_latency")

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def window(parameter_overrides: dict) -> dict:
    r"""Cycles of one parameter set before the coherent states overlap strongly.

    The cat state displacement shrinks as
    :math:`|\beta|^2(t)=|\beta|^2(0)\,e^{-\kappa t}`, for the state with and
    without a photon loss alike, so :math:`|\beta|^2` reaches
    ``DISPLACEMENT_SQUARED_AT_WINDOW_END`` at
    :math:`t_{\mathrm{end}}=\ln[|\beta|^2(0)/|\beta|^2_{\mathrm{end}}]/\kappa`.

    Parameters
    ----------
    parameter_overrides : dict
        Parameters that replace their reference values, an entry of
        ``device_model.parameter_sets``.

    Returns
    -------
    dict
        ``photon_lifetime_us`` (:math:`1/\kappa`, with the dressed photon-loss
        rate of the parameter set), ``t_end_us`` (:math:`t_{\mathrm{end}}`),
        and ``n_cycles``, the number of whole QEC cycles that end before it.
    """
    # the dressed sweep holds the photon-loss rate (1/ns) and the cycle
    # duration (ns); no dynamical map is built
    sweep, _, parameters = device_model.build_sweep(parameter_overrides)
    kappa = float(sweep["jump_a"])
    # "disp" is the cat state displacement at time zero
    t_end_ns = np.log(float(parameters["disp"]) ** 2 / DISPLACEMENT_SQUARED_AT_WINDOW_END) / kappa
    return dict(photon_lifetime_us=1 / kappa / 1e3, t_end_us=float(t_end_ns / 1e3),
                n_cycles=int(t_end_ns // float(sweep["t_cycle"])))


def cycle_end_fidelities(parameter_overrides: dict, n_cycles: int) -> dict:
    r"""Process fidelity of the actual QEC dynamics at the end of every QEC cycle.

    Parameters
    ----------
    parameter_overrides : dict
        Parameters that replace their reference values, an entry of
        ``device_model.parameter_sets``.
    n_cycles : int
        Number of QEC cycles in the tree.

    Returns
    -------
    dict
        ``kappa_a_phi`` (ancilla dephasing rate used, ns^-1), ``t_ns`` and
        ``F_T`` (arrays of length ``n_cycles + 1``: the times of the cycle
        ends, starting with 0, and the process fidelity there, starting with
        1 for the encoded state).
    """
    graph, sweep, parameters = device_model.build_actual_tree(
        parameter_overrides, n_cycles=n_cycles)
    t_ns = float(sweep["t_cycle"]) * np.arange(n_cycles + 1)
    # F_T after every step, summed over the measurement records; within a
    # cycle the loss is concentrated on a few steps, so only the cycle ends
    # are kept
    F_by_step = np.asarray(graph.process_fidelity_by_step(), float)
    F_T = F_by_step[::device_model.STEPS_PER_CYCLE]
    # the tree holds every node of all its cycles; free it before the next set
    del graph, sweep
    gc.collect()
    return dict(kappa_a_phi=float(parameters["kappa_a_phi"]), t_ns=t_ns, F_T=F_T)


def fit_exponential_decay(t_ns: np.ndarray, F: np.ndarray) -> dict:
    r"""Fit :math:`F(t)=1/4+(3/4)\,e^{-t/t_{\mathrm L}}` to the cycle ends.

    The offset is held at ``FIT_OFFSET`` and the amplitude at
    ``FIT_AMPLITUDE``, so that the fit is 1 at :math:`t=0`; only
    :math:`t_{\mathrm L}` is fitted.

    Parameters
    ----------
    t_ns : np.ndarray
        Times of the cycle ends (ns), including 0.
    F : np.ndarray
        Process fidelity at those times.

    Returns
    -------
    dict
        ``offset``, ``amplitude``, ``t_L_us`` (:math:`t_{\mathrm L}`) and its
        one-standard-error uncertainty ``t_L_err_us``,
        ``fitted_rate_per_ms`` (:math:`-\mathrm dF/\mathrm dt` of the fit at
        :math:`t=0`, which is :math:`3/(4t_{\mathrm L})`) and its uncertainty
        ``fitted_rate_err_per_ms``, and ``max_residual``, the largest
        distance of a point from the fit.
    """
    # the fit works in microseconds, where the decay rate is of order 1e-2
    t_us = np.asarray(t_ns, float) / 1e3
    F = np.asarray(F, float)
    # amplitude exp(-rate t) + offset, fitted to the points themselves
    amplitude, rate, offset, (_, rate_err, _) = tb.decay_rate(
        t_us, F, extract_envelope=False, return_error=True,
        offset=FIT_OFFSET, amplitude=FIT_AMPLITUDE)
    residual = F - (offset + amplitude * np.exp(-rate * t_us))
    fitted_rate = float(device_model.failure_rate_per_ms(amplitude, 1e3 / rate))
    return dict(offset=float(offset), amplitude=float(amplitude), t_L_us=float(1 / rate),
                # t_L = 1/rate, so its relative uncertainty is that of the rate
                t_L_err_us=float(rate_err / rate ** 2),
                fitted_rate_per_ms=fitted_rate,
                # the fitted rate is proportional to the rate of the fit
                fitted_rate_err_per_ms=float(fitted_rate * rate_err / rate),
                max_residual=float(np.abs(residual).max()))


def set_record(parameter_overrides: dict) -> dict:
    r"""Everything the comparison needs at one parameter set.

    Parameters
    ----------
    parameter_overrides : dict
        See :func:`window`.

    Returns
    -------
    dict
        ``parameter_overrides``, ``kappa_a_phi``, ``window`` (output of
        :func:`window`), ``t_cycle_us``; ``t_us`` and ``F_T``, the times of
        the cycle ends and the process fidelity there, starting with the
        encoded state; ``failure_rate_per_ms``,
        :math:`(1-F_{\mathbb T})/t_{\mathrm{cyc}}` after the first cycle; and
        ``fit``, the output of :func:`fit_exponential_decay`.
    """
    cycles = window(parameter_overrides)
    result = cycle_end_fidelities(parameter_overrides, cycles["n_cycles"])
    t_ns, F_T = result["t_ns"], result["F_T"]
    return dict(parameter_overrides=parameter_overrides, kappa_a_phi=result["kappa_a_phi"],
                window=cycles, t_cycle_us=float(t_ns[1] / 1e3),
                t_us=(t_ns / 1e3).tolist(), F_T=F_T.tolist(),
                failure_rate_per_ms=float(
                    device_model.failure_rate_per_ms(1 - F_T[1], t_ns[1])),
                fit=fit_exponential_decay(t_ns, F_T))


def compare(records: dict) -> dict:
    r"""Reduction of the two rates from the parameters of Ofek et al. to the optimized ones.

    A reduction is one minus the ratio of the rate at the optimized
    parameters to the rate at the parameters of Ofek et al.

    Parameters
    ----------
    records : dict
        Label (``OFEK``, ``OPTIMIZED``) to the output of :func:`set_record`.

    Returns
    -------
    dict
        ``one_cycle`` (the single-cycle failure rate) and ``fitted`` (the
        fitted multi-cycle failure rate), each with the ``rate_per_ms`` of
        both sets and the ``reduction``.
    """
    rates = dict(
        one_cycle={label: record["failure_rate_per_ms"] for label, record in records.items()},
        fitted={label: record["fit"]["fitted_rate_per_ms"] for label, record in records.items()},
    )
    return {name: dict(rate_per_ms=rate, reduction=float(1 - rate[OPTIMIZED] / rate[OFEK]))
            for name, rate in rates.items()}


def single_cycle_against_fitted(records: dict) -> dict:
    r"""The single-cycle failure rate against the fitted multi-cycle failure rate.

    Parameters
    ----------
    records : dict
        Label to the output of :func:`set_record`.

    Returns
    -------
    dict
        For each set: ``one_cycle_over_fitted_minus_1`` and its inverse
        direction ``fitted_over_one_cycle_minus_1`` (negative when the fitted
        rate is the lower one), and the decay of
        :math:`F_{\mathbb T}-1/4` per cycle, divided by the duration of the
        cycle (1/ms): ``decay_per_cycle_rate_per_ms`` for every cycle of the
        window, ``first_over_second`` and ``first_over_later_maximum``; the decay
        of the first cycle exceeds that of every later cycle if the last of
        them is larger than 1.
    """
    result = {}
    for label, record in records.items():
        F = np.asarray(record["F_T"], float) - FIT_OFFSET
        decay = (1.0 - F[1:] / F[:-1]) / record["t_cycle_us"] * 1e3
        result[label] = dict(
            one_cycle_over_fitted_minus_1=float(
                record["failure_rate_per_ms"] / record["fit"]["fitted_rate_per_ms"] - 1.0),
            # the direction in which the manuscript quotes the comparison
            fitted_over_one_cycle_minus_1=float(
                record["fit"]["fitted_rate_per_ms"] / record["failure_rate_per_ms"] - 1.0),
            decay_per_cycle_rate_per_ms=decay.tolist(),
            first_over_second=float(decay[0] / decay[1]),
            first_over_later_maximum=float(decay[0] / decay[1:].max()),
        )
    return result


def without_late_photon_loss(builder) -> None:
    r"""Remove photon loss from the measurement and the feedback-latency idling of a tree builder.

    Used as ``prepare_builder`` of ``device_model.build_tree_from_sweep``,
    before any dynamical map is built. The dynamical maps of the two steps
    after the parity mapping are built without the photon-loss operators of
    the master equation (:math:`\hat a`, and :math:`\hat a` together with an
    ancilla jump, ``jump_a`` and ``jump_a_<ij>``); every other step, the
    ideal operators, and the other jumps are unchanged.

    Parameters
    ----------
    builder : bsq.QEC_graph.FullCatTreeBuilder
        Tree builder with no dynamical map built
        (``device_model.configure_tree_builder``).

    Returns
    -------
    None
        The two build methods of ``builder`` are replaced in place.
    """
    every_operator = list(builder.c_ops)
    # "jump_a" is photon loss and "jump_a_<ij>" photon loss with an ancilla
    # jump; "jump_adag..." (photon gain, resonator dephasing) stays
    kept = [operator for name, operator in builder.collapse_operators.items()
            if not (name == "jump_a" or name.startswith("jump_a_"))]

    def with_kept_operators(build):
        r"""Wrap a build method so that it sees only the kept collapse operators.

        Parameters
        ----------
        build : callable
            Bound build method of the builder.

        Returns
        -------
        callable
            The method with the collapse operators swapped during the call.
        """
        def wrapped(*args, **kwargs):
            builder.c_ops = kept
            try:
                return build(*args, **kwargs)
            finally:
                # the other steps are built with every jump
                builder.c_ops = every_operator
        return wrapped

    for name in LATE_STEP_BUILDS:
        setattr(builder, name, with_kept_operators(getattr(builder, name)))


def nodes_after_step(graph, step: int) -> list:
    r"""Nodes of the expected measurement records after a given step.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.
    step : int
        Number :math:`n\geq1` of steps.

    Returns
    -------
    list
        The nodes after step :math:`n`; terminated nodes (unexpected records)
        are left out.
    """
    return [edge.final_state for index, _, edges in graph.iter_steps() if index == step
            for edge in edges]


def population_outside_error_spaces(graph, step: int) -> float:
    r"""Population outside the joint error space of the record after a step.

    For the maximally mixed logical state, the trace of
    :math:`\hat Q\hat\rho\hat Q` summed over the expected measurement
    records, with :math:`\hat Q=\hat 1-\hat P` and :math:`\hat P` the
    projector onto the joint error space of the record.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.
    step : int
        Number of steps.

    Returns
    -------
    float
        The population.
    """
    total = 0.0
    for node in nodes_after_step(graph, step):
        process = node.encoded_process.full()
        dim = int(round(np.sqrt(process.shape[0])))
        # columns 0 and 3 are the images of |0><0| and |1><1| (column stacking)
        state = (0.5 * (process[:, 0] + process[:, 3])).reshape(dim, dim, order="F")
        outside = np.eye(dim) - node.joint_error_space_projector().full()
        total += float(np.real(np.trace(outside @ state)))
    return total


def components_after_first_cycle(graph) -> dict:
    r"""Where each component of the state at the end of the first cycle is read later.

    At the end of the first QEC cycle the state of every measurement record
    splits into the component within the joint error space of the record,
    the component within the leakage subspace, and the error--leakage
    coherence (``model_chain.WITHIN_ERROR``, ``WITHIN_LEAKAGE``,
    ``COHERENCE``). Each component is propagated alone by the actual QEC
    dynamics of the second cycle and read in the error spaces of the
    records after a later step, with the steps after it taken as ideal: the
    process fidelity after :math:`n` steps (App. A) of that component. The
    one-cycle bookkeeping :math:`F_{\mathbb T}(N)` counts only the first
    component; the second is the population it counts as lost.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of two QEC cycles.

    Returns
    -------
    dict
        For each reading step (``READ_STEPS``) and each component
        (``FIRST_CYCLE_COMPONENTS``): ``F`` (process fidelity), ``X``, ``Y``,
        ``Z`` (probabilities of the logical errors: the diagonal of the
        process matrix in the logical Pauli basis), and ``population`` (their
        sum, the population of the component inside the error spaces of the
        records for the maximally mixed logical state); and
        ``sum_minus_F_T``, the three process fidelities summed minus the
        process fidelity of the tree after that step (a check, zero up to
        rounding).
    """
    F_by_step = np.asarray(graph.process_fidelity_by_step(), float)
    result = {}
    for read_name, read_step in READ_STEPS.items():
        entry = result[read_name] = {}
        for name, component in FIRST_CYCLE_COMPONENTS.items():
            # only this component is kept at the end of the first cycle;
            # every component is kept after the other steps
            elm = model_chain.elm_keeping_components(
                graph,
                lambda step, kept=component: (kept,) if step == device_model.STEPS_PER_CYCLE else None,
                read_after_step=read_step)
            diagonal = process_matrix_diagonal(elm)
            entry[name] = dict(zip(("F", "X", "Y", "Z"), map(float, diagonal)),
                               population=float(diagonal.sum()))
        entry["sum_minus_F_T"] = float(
            sum(entry[name]["F"] for name in FIRST_CYCLE_COMPONENTS) - F_by_step[read_step])
    return result


def late_error_tree_record(parameter_overrides: dict, late_photon_loss: bool) -> dict:
    r"""The two-cycle tree of one parameter set, read for the return of late errors.

    Parameters
    ----------
    parameter_overrides : dict
        An entry of ``device_model.parameter_sets``.
    late_photon_loss : bool
        False removes photon loss from the measurement and the
        feedback-latency idling (:func:`without_late_photon_loss`).

    Returns
    -------
    dict
        ``t_cycle_us``; ``F_T_by_step`` (process fidelity after every step of
        the two cycles, starting with the encoded state);
        ``one_cycle_infidelity`` (:math:`1-F_{\mathbb T}(N)` after the first
        cycle); ``outside_error_spaces_after_first_cycle``
        (:func:`population_outside_error_spaces`); and ``components``
        (:func:`components_after_first_cycle`).
    """
    sweep, simulation_parameters, _ = device_model.build_sweep(parameter_overrides)
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=LATE_ERROR_CYCLES, num_cpus=1,
        prepare_builder=None if late_photon_loss else without_late_photon_loss)
    F_by_step = np.asarray(graph.process_fidelity_by_step(), float)
    record = dict(
        t_cycle_us=float(sweep["t_cycle"]) / 1e3,
        F_T_by_step=F_by_step.tolist(),
        one_cycle_infidelity=float(1 - F_by_step[device_model.STEPS_PER_CYCLE]),
        outside_error_spaces_after_first_cycle=population_outside_error_spaces(
            graph, device_model.STEPS_PER_CYCLE),
        components=components_after_first_cycle(graph),
    )
    # the tree holds every node of both cycles; free it before the next one
    del graph, sweep
    gc.collect()
    return record


def late_error_numbers(full: dict, switched_off: dict, stored_set: dict) -> dict:
    r"""The return of the population lost at the end of the first cycle, as parts of the infidelity.

    Parameters
    ----------
    full : dict
        :func:`late_error_tree_record` with the full noise model.
    switched_off : dict
        The same without photon loss in the measurement and the
        feedback-latency idling.
    stored_set : dict
        The entry of the parameter set under ``parameter_sets`` of
        ``multi-cycle-fidelity.json`` (single-cycle rate and fit).

    Returns
    -------
    dict
        See the description of the key ``late_error_return`` in
        :func:`add_late_error_return`.
    """
    infidelity = full["one_cycle_infidelity"]
    at_measurement = full["components"]["after_measurement_of_cycle_2"]["leakage"]
    at_end = full["components"]["end_of_cycle_2"]["leakage"]
    off_at_measurement = switched_off["components"]["after_measurement_of_cycle_2"]["leakage"]
    late_infidelity = infidelity - switched_off["one_cycle_infidelity"]
    late_return = at_measurement["F"] - off_at_measurement["F"]
    t_cycle_us = full["t_cycle_us"]
    # loss of F - 1/4 in the first and second cycle, relative to its value
    # before the cycle, per ms; the second also with the return counted as lost
    F = np.asarray(full["F_T_by_step"], float)
    first, second = F[0], F[device_model.STEPS_PER_CYCLE]
    third = F[LATE_ERROR_CYCLES * device_model.STEPS_PER_CYCLE]
    decay_first = (first - second) / (first - FIT_OFFSET)
    decay_second = (second - third) / (second - FIT_OFFSET)
    decay_second_lost = (second - third + at_end["F"]) / (second - FIT_OFFSET)
    fitted_rate = stored_set["fit"]["fitted_rate_per_ms"]
    net_rate = float(device_model.failure_rate_per_ms(infidelity - at_end["F"], 1e3 * t_cycle_us))
    return dict(
        one_cycle_infidelity=infidelity,
        outside_error_spaces_after_first_cycle_over_infidelity=(
            full["outside_error_spaces_after_first_cycle"] / infidelity),
        returned_fidelity_after_measurement_of_cycle_2=at_measurement["F"],
        returned_fidelity_over_infidelity=at_measurement["F"] / infidelity,
        returned_population_after_measurement_of_cycle_2=at_measurement["population"],
        returned_population_over_infidelity=at_measurement["population"] / infidelity,
        returned_fidelity_at_end_of_cycle_2_over_infidelity=at_end["F"] / infidelity,
        late_photon_loss=dict(
            infidelity=late_infidelity,
            infidelity_over_infidelity=late_infidelity / infidelity,
            returned_fidelity=late_return,
            returned_fidelity_over_infidelity=late_return / infidelity,
            returned_fidelity_over_its_infidelity=late_return / late_infidelity,
            share_of_the_returned_fidelity=late_return / at_measurement["F"],
        ),
        rates=dict(
            single_cycle_rate_per_ms=float(
                device_model.failure_rate_per_ms(infidelity, 1e3 * t_cycle_us)),
            stored_single_cycle_rate_per_ms=stored_set["failure_rate_per_ms"],
            fitted_rate_per_ms=fitted_rate,
            one_cycle_over_fitted_minus_1=stored_set["failure_rate_per_ms"] / fitted_rate - 1,
            net_of_return_rate_per_ms=net_rate,
            net_of_return_over_fitted_minus_1=net_rate / fitted_rate - 1,
        ),
        decay_of_the_first_two_cycles=dict(
            first_rate_per_ms=decay_first / t_cycle_us * 1e3,
            second_rate_per_ms=decay_second / t_cycle_us * 1e3,
            second_with_return_counted_as_lost_rate_per_ms=decay_second_lost / t_cycle_us * 1e3,
            first_over_second=decay_first / decay_second,
            first_over_second_with_return_counted_as_lost=decay_first / decay_second_lost,
        ),
        F_T_after_two_cycles_minus_stored=float(third - stored_set["F_T"][2]),
    )


def add_late_error_return() -> dict:
    r"""Compute the key ``late_error_return`` and add it to ``multi-cycle-fidelity.json``.

    Builds two-cycle evolution trees with the full noise model and without
    photon loss during the measurement and feedback-latency idling, and
    tracks the second-cycle return of population that lay outside the joint
    error space at the end of the first cycle.

    Returns
    -------
    dict
        The updated content of ``multi-cycle-fidelity.json``.
    """
    stored = json.loads(RESULTS_JSON.read_text())
    sets = {}
    for label, parameter_overrides in device_model.parameter_sets().items():
        full = late_error_tree_record(parameter_overrides, late_photon_loss=True)
        switched_off = late_error_tree_record(parameter_overrides, late_photon_loss=False)
        sets[label] = dict(full_noise_model=full, without_late_photon_loss=switched_off,
                           numbers=late_error_numbers(full, switched_off,
                                                      stored["parameter_sets"][label]))
        print(label, json.dumps(sets[label]["numbers"], indent=1))
    # the photon-loss terms of the two late steps, numerically computed
    # mechanism by mechanism in formula-terms.py
    terms = json.loads(FORMULA_TERMS_JSON.read_text())["cycle_contributions"]
    late_terms = {entry["step"]: entry for entry in terms
                  if entry["term"] == "photon_loss" and entry["step"] in FORMULA_LATE_STEPS}
    ofek_infidelity = sets[OFEK]["numbers"]["one_cycle_infidelity"]
    stored["late_error_return"] = dict(
        description="Second-cycle return of first-cycle late photon-loss errors from two-cycle evolution trees with deferred recovery.",
        settings=dict(n_cycles=LATE_ERROR_CYCLES, read_steps=READ_STEPS,
                      components={name: list(component)
                                  for name, component in FIRST_CYCLE_COMPONENTS.items()},
                      switched_off_builds=list(LATE_STEP_BUILDS)),
        environment=device_model.environment_record(worker_processes=1),
        parameter_sets=sets,
        quoted_for_comparison=dict(
            source="formula-terms.json, cycle_contributions, term photon_loss (Ofek et al.)",
            numerical={step: entry["numerical"] for step, entry in late_terms.items()},
            formula={step: entry["formula"] for step, entry in late_terms.items()},
            numerical_sum_over_infidelity=sum(
                entry["numerical"] for entry in late_terms.values()) / ofek_infidelity,
            formula_sum_over_infidelity=sum(
                entry["formula"] for entry in late_terms.values()) / ofek_infidelity,
        ),
    )
    RESULTS_JSON.write_text(json.dumps(stored, indent=1))
    print(f"wrote late_error_return to {RESULTS_JSON}")
    return stored


def compute_record() -> dict:
    r"""Build the two trees, fit, compare, and write the data file.

    Returns
    -------
    dict
        The content of ``multi-cycle-fidelity.json``: ``settings``,
        ``parameter_sets`` (label to the output of :func:`set_record`), and
        ``comparison`` (output of :func:`compare`).
    """
    records = {}
    for label, parameter_overrides in device_model.parameter_sets().items():
        record = records[label] = set_record(parameter_overrides)
        fit, cycles = record["fit"], record["window"]
        print(f"{label}: t_cyc = {record['t_cycle_us']:.2f} us, |beta|^2 = "
              f"{DISPLACEMENT_SQUARED_AT_WINDOW_END} at {cycles['t_end_us']:.0f} us, {cycles['n_cycles']} cycles "
              f"({record['t_us'][-1]:.1f} us), F_T after the first and the last cycle: "
              f"{record['F_T'][1]:.5f}, {record['F_T'][-1]:.5f}; fit t_L = {fit['t_L_us']:.0f} "
              f"+- {fit['t_L_err_us']:.0f} us, fitted rate {fit['fitted_rate_per_ms']:.3f} "
              f"+- {fit['fitted_rate_err_per_ms']:.3f} /ms, "
              f"largest residual {fit['max_residual']:.1e}")
    record = dict(
        settings=dict(
            displacement_squared_at_window_end=DISPLACEMENT_SQUARED_AT_WINDOW_END,
            fit_offset=FIT_OFFSET,
            fit_amplitude=FIT_AMPLITUDE,
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
        ),
        parameter_sets=records,
        comparison=compare(records),
        single_cycle_against_fitted=single_cycle_against_fitted(records),
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["comparison"], indent=1))
    print(f"wrote {RESULTS_JSON}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help="recompute the fits and the comparisons from the stored process "
                             "fidelities; no trees")
    parser.add_argument("--add-late-error-return", action="store_true",
                        help="compute only the key late_error_return (four trees of two "
                             "cycles) and add it to multi-cycle-fidelity.json")
    arguments = parser.parse_args()
    if arguments.add_late_error_return:
        add_late_error_return()
    elif arguments.update:
        stored = json.loads(RESULTS_JSON.read_text())
        # the fits use only the stored times (us) and process fidelities
        for stored_set in stored["parameter_sets"].values():
            stored_set["fit"] = fit_exponential_decay(
                1e3 * np.asarray(stored_set["t_us"]), stored_set["F_T"])
        stored["settings"]["fit_amplitude"] = FIT_AMPLITUDE
        stored["comparison"] = compare(stored["parameter_sets"])
        stored["single_cycle_against_fitted"] = single_cycle_against_fitted(
            stored["parameter_sets"])
        RESULTS_JSON.write_text(json.dumps(stored, indent=1))
        print(json.dumps({key: stored_set["fit"]
                          for key, stored_set in stored["parameter_sets"].items()}, indent=1))
        print(json.dumps(stored["comparison"], indent=1))
        print(json.dumps(stored["single_cycle_against_fitted"], indent=1))
    else:
        compute_record()
        # the return of late errors reads the fits that the full run has
        # just written
        add_late_error_return()
