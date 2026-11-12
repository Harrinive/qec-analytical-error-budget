r"""Magnitudes of the seven neglected processes and accuracy bounds of the AEB.

Evaluates the step-by-step diamond-norm magnitudes of the neglected processes
at incoherent-error scales :math:`\xi = 1` and :math:`\xi = 1/32` for the
cat-code cycle of Ofek et al., and compares the resulting discrepancy bounds
against the exact discrepancies in ``accuracy-sweep.json``. Writes
``magnitudes.json``.

Usage: ``python magnitudes.py --scale 1``, ``python magnitudes.py --scale 1/32``, then ``python magnitudes.py --combine``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
import chencrafts.settings as cc_settings
from qec_aeb.QEC_graph import component_projector_map, process_matrix_diagonal

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain
from step_magnitudes import (
    absent_transition_magnitude, leakage_operator, most_seeping_state, restricted_spectrum,
    step_groups, unexpected_outcome_magnitude)

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "magnitudes.json"
PARTS = HERE / "magnitudes-parts"
ACCURACY_SWEEP_JSON = HERE / "accuracy-sweep.json"

# incoherent-error scales: label in the JSON file -> scale xi
SCALES = {"1": 1.0, "1/32": 1 / 32}
# number of steps of the cycle, the error-accumulation step, and the
# measurement step (steps counted from 1; one of each per QEC cycle)
N_STEPS = device_model.STEPS_PER_CYCLE
ERROR_ACCUMULATION_STEPS = [1]
MEASUREMENT_STEPS = [5]
# dimension of the logical subspace
LOGICAL_DIMENSION = 2
# number of syndromes |S| (even and odd photon-number parity)
NUMBER_OF_SYNDROMES = 2
# error space after the error-accumulation idling that holds no error (the
# ideal operators of the idling are: no jump, one photon loss)
NO_ERROR_SPACE = 0
# method of chencrafts.cqed.dnorm_hp for the coherence magnitudes, the
# relative gap between its bounds at which it stops, and its largest number
# of iterations (a map that has not converged keeps its certified bounds)
SOLVER, SOLVER_EPS, SOLVER_MAX_ITERS = "FIXED_POINT", 1e-5, 20000
# worker processes for the construction of the maps
NUM_CPUS = 1
# a combination at most this part of the overall failure probability counts
# as much smaller than it (the reading of "<<" in the key "holds")
MUCH_SMALLER = 0.1
# a discrepancy below this value is rounding; no ratio is formed with it
DISCREPANCY_FLOOR = 1e-12
# the magnitudes in the order of Appendix B: key -> (kind, symbol)
MAGNITUDES = {
    "unexp": ("population", "epsilon^unexp_n"),
    "leak": ("population", "epsilon^leak_n"),
    "seep": ("population", "epsilon^seep_n"),
    "leak_coh": ("coherence", "epsilon^leak-coh_n"),
    "seep_coh": ("coherence", "epsilon^seep-coh_n"),
    "err_traj": ("population", "epsilon^err-traj_n"),
    "err_coh": ("coherence", "epsilon^err-coh_n"),
    "logi_coh": ("coherence", "epsilon^logi-coh_n"),
    "logi_err": ("population", "epsilon^logi-err_n"),
}

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


# ------------------------------------------------------------- the maps ----
def error_space_isometry(node) -> qt.Qobj:
    r"""Isometry from the abstract joint error space into the simulated space.

    Its columns are the ideal error states of the node, error space by error
    space, so an operator :math:`\hat X` of the joint error space is
    :math:`\hat W\hat X\hat W^\dagger` on the simulated space.

    Parameters
    ----------
    node : StateNode
        Evolved node of the actual evolution tree.

    Returns
    -------
    qutip.Qobj
        :math:`\hat W`, of dims ``[simulated, [2 x number of error spaces]]``.
    """
    encoders = node.ideal_encoders()
    columns = np.column_stack([encoder.full() for encoder in encoders])
    return qt.Qobj(columns, dims=[encoders[0].dims[0], [columns.shape[1]]])


def coherence_creation_maps(group: dict) -> list[qt.Qobj]:
    r"""Maps that create error--leakage coherence at one step, one per expected outcome.

    :math:`\mathcal E^{\mathrm{leak\text{-}coh}\leftarrow\mathrm{err}}_{n,\mathbf r}`,
    written from the joint error space before the step (through
    :func:`error_space_isometry`) to the simulated space.

    Parameters
    ----------
    group : dict
        Entry of ``step_magnitudes.step_groups``.

    Returns
    -------
    list of qutip.Qobj
        One superoperator for each expected outcome of the step.
    """
    embed = qt.to_super(error_space_isometry(group["node"]))
    return [component_projector_map(edge.final_state.ideal_encoders(), model_chain.COHERENCE)
            * edge.physical_map() * embed for edge in group["edges"]]


def coherence_return_maps(group: dict) -> list[qt.Qobj]:
    r"""Maps that return error--leakage coherence at one step, one per expected outcome.

    :math:`\mathcal E^{\mathrm{err}\leftarrow\mathrm{leak\text{-}coh}}_{n,\mathbf r}`,
    written from the simulated space to the joint error space after the
    step. Reading the image in the joint error space through the adjoint of
    the isometry keeps its trace norm.

    Parameters
    ----------
    group : dict
        Entry of ``step_magnitudes.step_groups``.

    Returns
    -------
    list of qutip.Qobj
        One superoperator for each expected outcome of the step.
    """
    before = component_projector_map(group["node"].ideal_encoders(), model_chain.COHERENCE)
    return [qt.to_super(error_space_isometry(edge.final_state).dag())
            * edge.physical_map() * before for edge in group["edges"]]


def error_coherence_maps(group: dict) -> list[qt.Qobj] | None:
    r"""Maps of the coherence between error spaces at one step, one per expected outcome.

    The creation :math:`\sum\mathcal E^{\vec{\mathbf e}\vec{\mathbf
    e}'\leftarrow\vec{\mathbf g}\vec{\mathbf g}}_{n,\mathbf r}` (from the
    components within the error spaces before the step to the coherence
    between the error spaces after it) plus the return
    :math:`\sum\mathcal E^{\vec{\mathbf e}\vec{\mathbf e}\leftarrow\vec{\mathbf
    g}\vec{\mathbf g}'}_{n,\mathbf r}` (the reverse), written from the joint
    error space before the step to that after it.

    Parameters
    ----------
    group : dict
        Entry of ``step_magnitudes.step_groups``.

    Returns
    -------
    list of qutip.Qobj or None
        One superoperator for each expected outcome; ``None`` when no
        measurement record before or after the step has two error spaces,
        so that the maps vanish.
    """
    node = group["node"]
    embed = qt.to_super(error_space_isometry(node))
    # component_projector_map gives the integer 0 for the coherence of a
    # single error space
    within_before = component_projector_map(node.ideal_encoders(), "LD")
    between_before = component_projector_map(node.ideal_encoders(), "LC")
    maps = []
    for edge in group["edges"]:
        encoders = edge.final_state.ideal_encoders()
        within_after = component_projector_map(encoders, "LD")
        between_after = component_projector_map(encoders, "LC")
        step = edge.physical_map()
        parts = [after * step * before for after, before in
                 ((between_after, within_before), (within_after, between_before))
                 if not isinstance(after, int) and not isinstance(before, int)]
        if parts:
            maps.append(qt.to_super(error_space_isometry(edge.final_state).dag())
                        * sum(parts) * embed)
    return maps or None


def coherent_parts(selms: list[qt.Qobj]) -> list[qt.Qobj]:
    r"""Coherent parts of SELMs: each SELM minus its Pauli twirl.

    The Pauli twirl keeps the diagonal of the process matrix in the logical
    Pauli basis; the coherent part holds its off-diagonal entries, the
    coherence of logical errors.

    Parameters
    ----------
    selms : list of qutip.Qobj
        Step-wise effective logical maps (qutip ``super`` representation).

    Returns
    -------
    list of qutip.Qobj
        :math:`\mathcal C(\tilde{\mathcal E})` for each.
    """
    return [selm - cqed.Pauli_twirl(selm) for selm in selms]


# ------------------------------------------------------- the magnitudes ----
def coherence_magnitude(maps: list[qt.Qobj] | None) -> dict:
    r"""Diamond norm of maps kept apart, with certified bounds (``chencrafts.cqed.dnorm_hp``).

    Parameters
    ----------
    maps : list of qutip.Qobj or None
        Hermiticity-preserving maps with the same initial space, one for
        each outcome of a step; ``None`` for vanishing maps.

    Returns
    -------
    dict
        ``upper`` and ``lower`` (certified bounds on the norm of the direct
        sum), ``value`` (the optimal value of the solver), ``status``,
        ``choi_dim``, and ``seconds``; with more than one map also
        ``by_outcome`` (the bounds for each map alone),
        ``max_over_outcomes`` and ``sum_over_outcomes`` (of the upper
        bounds; they enclose the norm of the direct sum).
    """
    if maps is None:
        return {"upper": 0.0, "lower": 0.0, "value": 0.0, "status": "no such map",
                "choi_dim": 0, "seconds": 0.0}
    t0 = time.time()
    result = cqed.dnorm_hp(maps, solver=SOLVER, eps=SOLVER_EPS, max_iters=SOLVER_MAX_ITERS)
    record = {key: result[key] for key in ("upper", "lower", "value", "status", "choi_dim")}
    if len(maps) > 1:
        alone = [cqed.dnorm_hp(each, solver=SOLVER, eps=SOLVER_EPS, max_iters=SOLVER_MAX_ITERS)
                 for each in maps]
        record["by_outcome"] = [{"upper": each["upper"], "lower": each["lower"]}
                                for each in alone]
        record["max_over_outcomes"] = max(each["upper"] for each in alone)
        record["sum_over_outcomes"] = sum(each["upper"] for each in alone)
    record["seconds"] = time.time() - t0
    print(f"    diamond norm: {record['lower']:.6e} to {record['upper']:.6e}, Choi {record['choi_dim']}, "
          f"{record['status']}, {record['seconds']:.0f} s", flush=True)
    return record


def correctable_error_probability(group: dict) -> float:
    r"""Largest probability of a correctable error at an error-accumulation step.

    :math:`p^{\mathrm{corr}}_n` of the square-root relation of
    :math:`\epsilon^{\mathrm{err\text{-}coh}}_n` at an error-accumulation
    step: the largest probability, over the states of the error space before
    the step, of reaching an error space other than that of no error.

    Parameters
    ----------
    group : dict
        Entry of ``step_magnitudes.step_groups`` of an error-accumulation
        step (one error space before the step).

    Returns
    -------
    float
    """
    selms = [selm for edge in group["edges"]
             for final, selm in enumerate(edge.effective_logical_process(repr="super")[:, 0])
             if final != NO_ERROR_SPACE]
    return model_chain.completely_positive_dnorm(sum(selms))


def physical_step_magnitudes(group: dict) -> dict:
    r"""Magnitudes of one step for one measurement record before it, on the simulated space.

    Parameters
    ----------
    group : dict
        Entry of ``step_magnitudes.step_groups``.

    Returns
    -------
    dict
        ``step``, ``step_name``, ``record``; the largest probabilities
        ``unexp`` (``None`` without an unexpected outcome), ``leak``,
        ``seep`` (``None`` at the first step, which no step precedes), and
        ``err_traj``; the coherence magnitudes ``leak_coh``, ``seep_coh``
        (``None`` at the first step), and ``err_coh``
        (:func:`coherence_magnitude`); at an error-accumulation step
        ``p_corr``.
    """
    step = group["step"]
    print(f"  step {step}, record {group['record']}", flush=True)
    values, _ = restricted_spectrum(leakage_operator(group), group["node"])
    record = {
        "step": step,
        "step_name": model_chain.STEP_NAMES[step - 1],
        "record": group["record"],
        "unexp": unexpected_outcome_magnitude(group),
        # a largest eigenvalue can come out at -1e-17
        "leak": float(max(values[-1], 0.0)),
        "seep": most_seeping_state(group)[0] if step > 1 else None,
        "err_traj": absent_transition_magnitude(group),
        "leak_coh": coherence_magnitude(coherence_creation_maps(group)),
        "seep_coh": coherence_magnitude(coherence_return_maps(group)) if step > 1 else None,
        "err_coh": coherence_magnitude(error_coherence_maps(group)),
    }
    if step in ERROR_ACCUMULATION_STEPS:
        record["p_corr"] = correctable_error_probability(group)
    return record


def logical_step_magnitudes(logical) -> list[dict]:
    r"""Magnitudes of the logical errors for every state of the logical evolution tree.

    Parameters
    ----------
    logical : LogicalEvolutionTree
        Logical evolution tree of one QEC cycle.

    Returns
    -------
    list of dict
        One entry for each state before a step: ``step``, ``record``,
        ``edges`` (the names of the SELMs that leave the state),
        ``logi_err`` (the sum of their logical-error probabilities, the
        diagonal entries of the process matrices other than the identity),
        and ``logi_coh`` (:func:`coherence_magnitude` of their coherent
        parts, kept apart).
    """
    rows = []
    for step, node, edges in logical.iter_steps():
        selms = [edge.real_map for edge in edges]
        print(f"  logical step {step}, {[str(edge.name) for edge in edges]}", flush=True)
        rows.append({
            "step": step,
            "record": [int(outcome) for outcome in node.meas_record],
            "edges": [str(edge.name) for edge in edges],
            "logi_err": float(sum(process_matrix_diagonal(selm)[1:].sum() for selm in selms)),
            "logi_coh": coherence_magnitude(coherent_parts(selms)),
        })
    return rows


def compute_scale(label: str) -> dict:
    r"""Every magnitude at one incoherent-error scale.

    Parameters
    ----------
    label : str
        Key of :data:`SCALES`.

    Returns
    -------
    dict
        ``scale``, ``label``, ``physical`` (:func:`physical_step_magnitudes`
        for every step and record), ``logical``
        (:func:`logical_step_magnitudes`), ``F_M`` and ``F_T`` (the process
        fidelities of the failure Markov chain and of the actual QEC
        dynamics of this tree, compared with ``accuracy-sweep.json`` by
        :func:`combine`), ``seconds_tree``, ``seconds``, and the
        ``environment`` of the run.
    """
    t0 = time.time()
    graph, _, _ = device_model.build_actual_tree(
        device_model.incoherent_error_scaled_parameters(SCALES[label]), n_cycles=1,
        num_cpus=NUM_CPUS)
    seconds_tree = time.time() - t0
    print(f"scale {label}: tree in {seconds_tree:.0f} s", flush=True)
    physical = [physical_step_magnitudes(group) for group in step_groups(graph)]
    logical_tree = graph.logical_evolution_tree()
    logical = logical_step_magnitudes(logical_tree)
    return {
        "scale": SCALES[label], "label": label,
        "physical": physical, "logical": logical,
        "F_M": float(logical_tree.intermediate_success_prob_by_step()[-1]),
        "F_T": float(graph.process_fidelity_by_step()[-1]),
        "seconds_tree": seconds_tree, "seconds": time.time() - t0,
        "settings": dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            solver=SOLVER, solver_eps=SOLVER_EPS, solver_max_iters=SOLVER_MAX_ITERS),
        "environment": device_model.environment_record(NUM_CPUS),
    }


def part_path(label: str) -> Path:
    r"""Part file of one incoherent-error scale.

    Parameters
    ----------
    label : str
        Key of :data:`SCALES`.

    Returns
    -------
    Path
        ``magnitudes-parts/scale-<label>.json``, with ``/`` written ``_``.
    """
    return PARTS / f"scale-{label.replace('/', '_')}.json"


# ------------------------------------------------------ the combinations ----
def magnitudes_by_step(part: dict) -> dict:
    r"""Magnitude of each process at each step: the largest over the records before it.

    For the coherent processes the upper bound of
    the certified upper bound is used, and the lower bound is kept beside it.

    Parameters
    ----------
    part : dict
        Output of :func:`compute_scale`.

    Returns
    -------
    dict
        Key of :data:`MAGNITUDES` -> list of :data:`N_STEPS` values (step 1
        first; 0 where the process does not exist at the step), and, for the
        coherence magnitudes, the key ``<name>_lower`` with the lower bounds;
        ``p_corr`` at the error-accumulation steps.
    """
    result = {}
    sources = {name: part["physical"] for name in MAGNITUDES if name not in ("logi_coh", "logi_err")}
    sources |= {"logi_coh": part["logical"], "logi_err": part["logical"]}
    for name, (kind, _) in MAGNITUDES.items():
        for bound in (("upper", "lower") if kind == "coherence" else (None,)):
            values = [0.0] * N_STEPS
            for row in sources[name]:
                entry = row[name]
                if entry is None:
                    continue
                value = entry[bound] if bound else entry
                values[row["step"] - 1] = max(values[row["step"] - 1], value)
            result[name if bound in (None, "upper") else f"{name}_lower"] = values
    result["p_corr"] = {row["step"]: row["p_corr"] for row in part["physical"] if "p_corr" in row}
    return result


def pair_sum(earlier: list[float], later: list[float]) -> float:
    r"""Sum over the pairs of steps :math:`i<j` of ``earlier[i] * later[j]``.

    Parameters
    ----------
    earlier, later : list of float
        A magnitude at each step; the first factor comes from the earlier
        step.

    Returns
    -------
    float
    """
    return float(sum(earlier[i] * later[j] for i in range(N_STEPS) for j in range(i + 1, N_STEPS)))


def table_combinations(eps: dict) -> dict:
    r"""The combinations of Table I and the bounds of Appendix B on the discrepancies.

    Parameters
    ----------
    eps : dict
        Output of :func:`magnitudes_by_step`.

    Returns
    -------
    dict
        Discrepancy key (``model_chain.REDUCTIONS``) -> ``conditions`` (name
        -> value of each combination of its row of Table I), ``bound`` (the
        bound of Appendix B), ``bound_formula``, and for some rows further
        forms of the bound.
    """
    steps = range(N_STEPS)
    leak, seep = eps["leak"], eps["seep"]
    (n_star,) = MEASUREMENT_STEPS
    # the accumulated leakage magnitude before the measurement step
    accumulated_leakage = float(sum(np.sqrt(leak[m]) for m in range(n_star - 1)) ** 2)
    sum_unexp = float(eps["unexp"][n_star - 1])
    leak_seep = pair_sum(leak, seep)
    # the bound of Proposition prop:bound-leakcoh: the earlier bracket at
    # step i, the later bracket at step j
    created = [eps["leak_coh"][i] + 2 * np.sqrt(seep[i]) * sum(leak[:i]) for i in steps]
    returned = [eps["seep_coh"][j] + np.sqrt(leak[j]) * sum(seep[j + 1:]) for j in steps]
    mu = float(sum(seep[1:]) + sum(leak[:-1]))
    leak_coh_pair = pair_sum(eps["leak_coh"], eps["seep_coh"])
    # the neglected error-space transitions, steps 2 to n_m (none at the
    # error-accumulation step, none after the measurement step)
    err_traj = float(sum(eps["err_traj"][n - 1] for n in range(1, n_star + 1)
                         if n not in ERROR_ACCUMULATION_STEPS))
    return {
        "unexp": {
            "conditions": {"sum_eps_unexp": sum_unexp, "eps_leak_before_n_star": accumulated_leakage},
            "bound": float((np.sqrt(sum_unexp) + np.sqrt(accumulated_leakage)) ** 2),
            "bound_formula": "(sqrt(eps_unexp_{n_m}) + sqrt(eps_leak-before_{n_m}))^2",
            "bound_linear": 2 * sum_unexp + 2 * accumulated_leakage,
        },
        "leak_coh": {
            "conditions": {"sum_eps_leakcoh_i_eps_seepcoh_j": leak_coh_pair,
                           "N_sum_eps_leak_i_eps_seep_j": N_STEPS * leak_seep},
            "bound": pair_sum(created, returned),
            "bound_formula": "sum_{i<j} [eps_leak-coh_i + 2 sqrt(eps_seep_i) sum_{h<i} eps_leak_h]"
                             " [eps_seep-coh_j + sqrt(eps_leak_j) sum_{m>j} eps_seep_m]",
            "bound_table_form": leak_coh_pair + (2 + mu) * N_STEPS * leak_seep,
            "mu": mu,
        },
        "leak": {"conditions": {"sum_eps_leak_i_eps_seep_j": leak_seep},
                 "bound": leak_seep, "bound_formula": "sum_{i<j} eps_leak_i eps_seep_j"},
        "err_coh": {"conditions": {"sum_eps_errcoh_i_eps_errcoh_j":
                                   pair_sum(eps["err_coh"], eps["err_coh"])},
                    "bound": pair_sum(eps["err_coh"], eps["err_coh"]),
                    "bound_formula": "sum_{i<j} eps_err-coh_i eps_err-coh_j"},
        "err_traj": {"conditions": {"sum_eps_errtraj_n": err_traj},
                     "bound": err_traj, "bound_formula": "sum_n eps_err-traj_n, n not EA"},
        "logi_coh": {"conditions": {"sum_eps_logicoh_i_eps_logicoh_j":
                                    pair_sum(eps["logi_coh"], eps["logi_coh"])},
                     "bound": pair_sum(eps["logi_coh"], eps["logi_coh"]),
                     "bound_formula": "sum_{i<j} eps_logi-coh_i eps_logi-coh_j"},
        "logi_err": {"conditions": {"sum_eps_logierr_i_eps_logierr_j":
                                    pair_sum(eps["logi_err"], eps["logi_err"])},
                     "bound": pair_sum(eps["logi_err"], eps["logi_err"]),
                     "bound_formula": "sum_{i<j} eps_logi-err_i eps_logi-err_j",
                     "sum_eps_logierr_n": float(sum(eps["logi_err"]))},
    }


def square_root_relations(eps: dict) -> dict:
    r"""The square-root relations of Appendix B, evaluated as a check of the magnitudes.

    Each relation bounds a coherence magnitude by the population magnitude
    of the same step; the certified lower bound of the magnitude is tested,
    so a relation that fails is violated by the computed magnitudes.

    Parameters
    ----------
    eps : dict
        Output of :func:`magnitudes_by_step`.

    Returns
    -------
    dict
        Relation -> ``formula``, ``steps`` (the steps it covers), ``ratio``
        (lower bound of the magnitude over the right-hand side, at each of
        these steps; ``None`` where both vanish), and ``holds``.
    """
    def check(name: str, right: list[float], steps: list[int], formula: str) -> dict:
        r"""One relation at the given steps.

        Parameters
        ----------
        name : str
            Key of the coherence magnitude.
        right : list of float
            Right-hand side at each step (step 1 first).
        steps : list of int
            Steps at which the relation applies (counted from 1).
        formula : str
            The relation.

        Returns
        -------
        dict
        """
        left = eps[f"{name}_lower"]
        # a zero right-hand side with a nonzero magnitude violates the relation
        ratios = {n: (left[n - 1] / right[n - 1] if right[n - 1] > 0 else
                      (None if left[n - 1] == 0 else "nonzero over zero")) for n in steps}
        return {"formula": formula, "steps": steps, "ratio": ratios,
                "holds": all(ratio is None or (not isinstance(ratio, str) and ratio <= 1)
                             for ratio in ratios.values())}

    every, after_first = list(range(1, N_STEPS + 1)), list(range(2, N_STEPS + 1))
    not_accumulation = [n for n in every if n not in ERROR_ACCUMULATION_STEPS]
    d, S = LOGICAL_DIMENSION, NUMBER_OF_SYNDROMES
    p_corr = [eps["p_corr"].get(n, 0.0) for n in every]
    return {
        "leak_coh": check("leak_coh", [2 * np.sqrt(x) for x in eps["leak"]], every,
                          "eps_leak-coh_n <= 2 sqrt(eps_leak_n)"),
        "seep_coh": check("seep_coh", [np.sqrt(x) for x in eps["seep"]], after_first,
                          "eps_seep-coh_n <= sqrt(eps_seep_n)"),
        "err_coh": check("err_coh",
                         [2 * (1 + np.sqrt(S - 1)) * np.sqrt(x) + (S + 1 - 2 / S) * x
                          for x in eps["err_traj"]],
                         not_accumulation,
                         "eps_err-coh_n <= 2 (1 + sqrt(|S| - 1)) sqrt(eps_err-traj_n) "
                         "+ (|S| + 1 - 2/|S|) eps_err-traj_n"),
        "err_coh_accumulation": check(
            "err_coh", [2 * np.sqrt(x) + 2 * (1 - 1 / S) * x for x in p_corr],
            ERROR_ACCUMULATION_STEPS,
            "eps_err-coh_n <= 2 sqrt(p_corr_n) + 2 (1 - 1/|S|) p_corr_n"),
        "logi_coh": check("logi_coh", [2 * np.sqrt(d * x) + (d + 1) * x for x in eps["logi_err"]],
                          every, "eps_logi-coh_n <= 2 sqrt(d eps_logi-err_n) + (d+1) eps_logi-err_n"),
    }


def compare(combinations: dict, failure: float, difference: dict) -> dict:
    r"""Each combination against the overall failure probability, each bound against its discrepancy.

    Parameters
    ----------
    combinations : dict
        Output of :func:`table_combinations`.
    failure : float
        :math:`1-F_{\mathbb M}`.
    difference : dict
        Signed difference of each reduction (``accuracy-sweep.json``).

    Returns
    -------
    dict
        Discrepancy key -> the entries of ``combinations`` with
        ``discrepancy`` (its absolute value), ``bound_over_discrepancy``,
        and for each condition its ``value``, ``over_failure`` (the value
        over :math:`1-F_{\mathbb M}`), ``holds`` (``over_failure`` at most
        :data:`MUCH_SMALLER`), and ``over_discrepancy``. A ratio with a
        discrepancy below :data:`DISCREPANCY_FLOOR` is ``None``.
    """
    result = {}
    for name, row in combinations.items():
        discrepancy = abs(difference[name])
        resolved = discrepancy > DISCREPANCY_FLOOR
        conditions = {
            key: {"value": value, "over_failure": value / failure,
                  "holds": value / failure <= MUCH_SMALLER,
                  "over_discrepancy": value / discrepancy if resolved else None}
            for key, value in row["conditions"].items()}
        result[name] = row | {
            "conditions": conditions,
            "discrepancy": discrepancy,
            "discrepancy_over_failure": discrepancy / failure,
            "bound_over_discrepancy": row["bound"] / discrepancy if resolved else None,
            "bound_over_failure": row["bound"] / failure,
        }
    return result


def quoted_numbers(scales: dict) -> dict:
    r"""Summary metrics of the bounds and conditions at each incoherent-error scale.

    Parameters
    ----------
    scales : dict
        Label -> the entry of :func:`combine` for that scale.

    Returns
    -------
    dict
        At each scale: the overall failure probability, the largest and the
        smallest ratio of a bound to its discrepancy (with the discrepancy),
        the conditions that hold and those that fail with their ratio to the
        overall failure probability, and the largest relative gap of the
        diamond norms of the coherence magnitudes.
    """
    numbers = {}
    for label, entry in scales.items():
        ratios = {name: row["bound_over_discrepancy"] for name, row in entry["comparison"].items()
                  if row["bound_over_discrepancy"] is not None}
        conditions = {f"{name}: {key}": condition["over_failure"]
                      for name, row in entry["comparison"].items()
                      for key, condition in row["conditions"].items()}
        numbers[label] = {
            "one_minus_F_M": entry["one_minus_F_M"],
            "largest_bound_over_discrepancy": max(ratios.items(), key=lambda item: item[1]),
            "smallest_bound_over_discrepancy": min(ratios.items(), key=lambda item: item[1]),
            "conditions_that_hold": {key: value for key, value in conditions.items()
                                     if value <= MUCH_SMALLER},
            "conditions_that_fail": {key: value for key, value in conditions.items()
                                     if value > MUCH_SMALLER},
            "largest_relative_gap_of_the_coherence_magnitudes": entry["largest_relative_gap"],
        }
    return numbers


def largest_relative_gap(part: dict) -> float:
    r"""Largest relative gap between the certified bounds of the coherence magnitudes.

    Parameters
    ----------
    part : dict
        Output of :func:`compute_scale`.

    Returns
    -------
    float
        :math:`\max(\mathrm{upper}-\mathrm{lower})/\mathrm{upper}` over the
        coherence magnitudes with a nonzero upper bound.
    """
    records = [row[name] for row in part["physical"] for name in ("leak_coh", "seep_coh", "err_coh")
               if row[name] is not None] + [row["logi_coh"] for row in part["logical"]]
    return max((record["upper"] - record["lower"]) / record["upper"]
               for record in records if record["upper"] > 0)


def combine() -> None:
    r"""Read the part files and ``accuracy-sweep.json`` and write ``magnitudes.json``.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the process fidelities of a tree differ from those of
        ``accuracy-sweep.json`` at the same scale by more than 1e-9.
    """
    sweep = json.loads(ACCURACY_SWEEP_JSON.read_text())
    scales = {}
    for label, scale in SCALES.items():
        part = json.loads(part_path(label).read_text())
        row = next(row for row in sweep["rows"] if abs(row["scale"] - scale) < 1e-12)
        # the discrepancies come from another run of the same tree
        for model in ("F_M", "F_T"):
            if abs(part[model] - row["fidelity"][model]) > 1e-9:
                raise RuntimeError(f"scale {label}: {model} {part[model]} differs from "
                                   f"accuracy-sweep.json {row['fidelity'][model]}")
        failure = 1 - row["fidelity"]["F_M"]
        eps = magnitudes_by_step(part)
        scales[label] = {
            "scale": scale,
            "one_minus_F_M": failure,
            "F_T_minus_F_M": row["fidelity"]["F_T"] - row["fidelity"]["F_M"],
            "magnitudes_by_step": eps,
            "comparison": compare(table_combinations(eps), failure, row["difference"]),
            "square_root_relations": square_root_relations(eps),
            "largest_relative_gap": largest_relative_gap(part),
            "computation": part,
        }
    record = {
        "description": "Step-by-step diamond-norm magnitudes of the neglected processes and discrepancy bounds for one cat-code QEC cycle at incoherent-error scales 1 and 1/32.",
        "much_smaller": MUCH_SMALLER,
        "discrepancy_floor": DISCREPANCY_FLOOR,
        "magnitudes": {name: {"kind": kind, "symbol": symbol}
                       for name, (kind, symbol) in MAGNITUDES.items()},
        "steps": model_chain.STEP_NAMES,
        "error_accumulation_steps": ERROR_ACCUMULATION_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "solver": {"name": SOLVER, "eps": SOLVER_EPS, "max_iters": SOLVER_MAX_ITERS},
        "numbers": quoted_numbers(scales),
        "scales": scales,
    }
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", choices=list(SCALES), help="compute one incoherent-error scale")
    parser.add_argument("--combine", action="store_true", help="write magnitudes.json from the parts")
    arguments = parser.parse_args()
    if arguments.combine:
        combine()
    elif arguments.scale:
        PARTS.mkdir(exist_ok=True)
        part = compute_scale(arguments.scale)
        part_path(arguments.scale).write_text(json.dumps(part, indent=1))
        print(f"wrote {part_path(arguments.scale)} in {part['seconds']:.0f} s")
    else:
        parser.error("give --scale or --combine")
