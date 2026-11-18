r"""Step-by-step diamond-norm magnitudes of the neglected processes and discrepancy bounds.

Computes the exact diamond-norm magnitudes of the population-moving neglected
processes (leakage, seepage, neglected error-space transitions, unexpected
measurement outcomes), the certified diamond-norm bounds of the coherence
processes (error--leakage coherence creation and return, coherence between error
spaces, coherence of logical errors), the discrepancy bounds of Appendix B, and
the square-root relations on any traversed :class:`~qec_aeb.EvolutionTree`.
"""

from __future__ import annotations

__all__ = [
    "MAGNITUDES",
    "SOLVER",
    "SOLVER_EPS",
    "SOLVER_MAX_ITERS",
    "NO_ERROR_SPACE",
    "adjoint_image",
    "leakage_projector",
    "restricted_spectrum",
    "step_groups",
    "leakage_operator",
    "most_seeping_state",
    "absent_transition_magnitude",
    "unexpected_outcome_magnitude",
    "error_space_isometry",
    "coherence_creation_maps",
    "coherence_return_maps",
    "error_coherence_maps",
    "coherent_parts",
    "coherence_magnitude",
    "correctable_error_probability",
    "physical_step_magnitudes",
    "logical_step_magnitudes",
    "magnitudes_by_step",
    "pair_sum",
    "table_combinations",
    "square_root_relations",
]

import time
from typing import Iterable, Sequence

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
from .graph import process_matrix_diagonal
from .reduction_chain import COHERENCE, WITHIN_ERROR, completely_positive_dnorm
from .utils import component_projector_map

# method of chencrafts.cqed.dnorm_hp for the coherence magnitudes, the
# relative gap between its bounds at which it stops, and its largest number
# of iterations (a map that has not converged keeps its certified bounds)
SOLVER, SOLVER_EPS, SOLVER_MAX_ITERS = "FIXED_POINT", 1e-5, 20000
# error space after the error-accumulation step that holds no error
NO_ERROR_SPACE = 0

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


# ------------------------------------------------ population magnitudes ----
def adjoint_image(superop: qt.Qobj, operator: qt.Qobj) -> qt.Qobj:
    r"""Image of an operator under the adjoint of a map, :math:`\mathcal E^\dagger(\hat O)`.

    For a projector :math:`\hat P`, :math:`\langle\psi|\mathcal E^\dagger(\hat
    P)|\psi\rangle` is the probability that the map takes
    :math:`|\psi\rangle` into the range of :math:`\hat P`.

    Parameters
    ----------
    superop : qutip.Qobj
        The map :math:`\mathcal E` (qutip ``super`` representation).
    operator : qutip.Qobj
        The operator :math:`\hat O` on the space after the map.

    Returns
    -------
    qutip.Qobj
        The operator :math:`\mathcal E^\dagger(\hat O)`.
    """
    return cqed.superop_evolve(superop.dag(), operator)


def leakage_projector(node) -> qt.Qobj:
    r"""Projector onto the leakage subspace for the measurement record of a node.

    Parameters
    ----------
    node : StateNode
        Evolved node of the actual evolution tree.

    Returns
    -------
    qutip.Qobj
        :math:`\hat 1-\hat P_{\mathrm{err}}`.
    """
    error = node.joint_error_space_projector()
    return qt.qeye(error.dims[0]) - error


def restricted_spectrum(operator: qt.Qobj, node) -> tuple[np.ndarray, np.ndarray]:
    r"""Eigenvalues and eigenvectors of a Hermitian operator within the joint error space.

    The operator is diagonalized in the basis of the ideal error states of
    the node, so every eigenvector lies in the joint error space.

    Parameters
    ----------
    operator : qutip.Qobj
        Hermitian up to rounding.
    node : StateNode
        Evolved node of the actual evolution tree.

    Returns
    -------
    values : np.ndarray
        Eigenvalues in increasing order.
    vectors : np.ndarray
        Eigenvectors on the simulated space, as columns.
    """
    basis = np.column_stack([encoder.full() for encoder in node.ideal_encoders()])
    restricted = basis.conj().T @ operator.full() @ basis
    values, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    return values, basis @ vectors


def step_groups(graph) -> list[dict]:
    r"""The steps of the tree, one entry for each measurement record before a step.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.

    Returns
    -------
    list of dict
        ``step`` (1 to N), ``record`` (the measurement record before the
        step), ``node`` (the node that represents the state before the
        step), ``edges`` (the edges of the outcomes that are not
        unexpected), and ``unexpected_edges`` (the edges of the unexpected
        outcomes).
    """
    groups = []
    # iter_steps leaves the edges into terminated nodes out
    for step, node, edges in graph.iter_steps():
        groups.append({
            "step": step,
            "record": [int(outcome) for outcome in node.meas_record],
            "node": node,
            "edges": edges,
            "unexpected_edges": [edge for edge in node.out_edges
                                 if edge.final_state.terminated],
        })
    return groups


def leakage_operator(group: dict) -> qt.Qobj:
    r"""Operator whose expectation value is the leakage probability of one step.

    :math:`\sum_{\mathbf r}\mathcal E_{n,\mathbf r}^\dagger(\hat
    P^{\mathrm{leak}}_{\mathbf r})`, added over the outcomes that are not
    unexpected, with the leakage subspace of the record after the step.
    Within the joint error space before the step, its largest eigenvalue is
    the leakage magnitude.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    qutip.Qobj
        The operator on the simulated space before the step.
    """
    return sum(adjoint_image(edge.physical_map(), leakage_projector(edge.final_state))
               for edge in group["edges"])


def most_seeping_state(group: dict) -> tuple[float, qt.Qobj]:
    r"""Seepage magnitude of one step and the state of the leakage subspace that gives it.

    The seepage operator is :math:`\hat P_{\mathrm{leak}}\sum_{\mathbf r}
    \mathcal E_{n,\mathbf r}^\dagger(\hat P^{\mathrm{err}}_{\mathbf r})\hat
    P_{\mathrm{leak}}`, added over the outcomes that are not unexpected, with
    the joint error space of the record after the step and the leakage
    subspace before it. Its expectation value in a state of the leakage
    subspace is the probability that the step returns the state to the
    joint error space, and its largest eigenvalue is the seepage magnitude.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    magnitude : float
        The largest seepage probability over the states of the leakage
        subspace before the step.
    state : qutip.Qobj
        Ket of the leakage subspace with that seepage probability.
    """
    leak = leakage_projector(group["node"])
    returned = sum(adjoint_image(edge.physical_map(), edge.final_state.joint_error_space_projector())
                   for edge in group["edges"])
    restricted = (leak * returned * leak).full()
    values, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
    state = qt.Qobj(vectors[:, -1], dims=[leak.dims[0], [1] * len(leak.dims[0])])
    # a largest eigenvalue can come out at -1e-17
    return float(max(values[-1], 0.0)), state


def absent_transition_magnitude(group: dict) -> float:
    r"""Magnitude of the error-space transitions that the ideal protocol does not have.

    From an error space before the step, every step-wise effective logical
    map into an error space that the ideal evolution tree does not produce
    from it is such a transition. The magnitude is the diamond norm of all
    of them with the outcomes of the step kept apart. The maps are
    completely positive, so this is the largest eigenvalue of
    :math:`\sum\Phi^\dagger(\hat I)`, added over the outcomes of the step
    and the error spaces after it: the largest probability, over the logical
    states of one error space, that the step makes one of these transitions.
    It is maximized over the error spaces before the step. At an
    error-accumulation step (``edge.error_accumulation`` True), the ideal
    evolution tree contains its transitions, and the magnitude is zero there
    by definition.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    float
        The magnitude for the measurement record before the step.
    """
    per_initial = {}
    for edge in group["edges"]:
        if getattr(edge, "error_accumulation", False):
            continue
        # for each final error space, the error space before the step that
        # the ideal evolution tree produces it from
        produced_from = edge.init_error_space_by_final
        selm = edge.effective_logical_process(repr="super")
        n_final, n_initial = selm.shape
        for initial in range(n_initial):
            for final in range(n_final):
                if final < len(produced_from) and produced_from[final] == initial:
                    continue
                image = adjoint_image(selm[final, initial], qt.qeye(2)).full()
                per_initial[initial] = per_initial.get(initial, 0.0) + image
    # the largest eigenvalue of the Hermitian part; it can come out at -1e-17
    return max((max(float(np.linalg.eigvalsh(0.5 * (image + image.conj().T))[-1]), 0.0)
                for image in per_initial.values()), default=0.0)


def unexpected_outcome_magnitude(group: dict) -> float | None:
    r"""Largest probability of the unexpected measurement outcomes of one step.

    The magnitude keeps the unexpected outcomes apart; their projected
    dynamical maps are completely positive, so it is the diamond norm of
    their sum, the largest eigenvalue of :math:`\sum\Phi^\dagger(\hat I)`.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    float or None
        The largest probability of the unexpected outcomes over the states
        of the joint error space before the step; ``None`` for a step
        without an unexpected outcome.
    """
    if not group["unexpected_edges"]:
        return None
    onto_error = component_projector_map(
        group["node"].ideal_encoders(), WITHIN_ERROR)
    return completely_positive_dnorm(
        sum(edge.physical_map() for edge in group["unexpected_edges"]) * onto_error)


# ------------------------------------------------- coherence magnitudes ----
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
        Entry of :func:`step_groups`.

    Returns
    -------
    list of qutip.Qobj
        One superoperator for each expected outcome of the step.
    """
    embed = qt.to_super(error_space_isometry(group["node"]))
    return [component_projector_map(edge.final_state.ideal_encoders(), COHERENCE)
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
        Entry of :func:`step_groups`.

    Returns
    -------
    list of qutip.Qobj
        One superoperator for each expected outcome of the step.
    """
    before = component_projector_map(group["node"].ideal_encoders(), COHERENCE)
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
        Entry of :func:`step_groups`.

    Returns
    -------
    list of qutip.Qobj or None
        One superoperator for each expected outcome; ``None`` when no
        measurement record before or after the step has two error spaces,
        so that the maps vanish.
    """
    node = group["node"]
    embed = qt.to_super(error_space_isometry(node))
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


def coherence_magnitude(
    maps: list[qt.Qobj] | None,
    solver: str = SOLVER,
    eps: float = SOLVER_EPS,
    max_iters: int = SOLVER_MAX_ITERS,
) -> dict:
    r"""Diamond norm of maps kept apart, with certified bounds (``chencrafts.cqed.dnorm_hp``).

    Parameters
    ----------
    maps : list of qutip.Qobj or None
        Hermiticity-preserving maps with the same initial space, one for
        each outcome of a step; ``None`` for vanishing maps.
    solver : str
        Solver of ``chencrafts.cqed.dnorm_hp``.
    eps : float
        Relative gap between bounds at which the solver stops.
    max_iters : int
        Maximum solver iterations.

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
    result = cqed.dnorm_hp(maps, solver=solver, eps=eps, max_iters=max_iters)
    record = {key: result[key] for key in ("upper", "lower", "value", "status", "choi_dim")}
    if len(maps) > 1:
        alone = [cqed.dnorm_hp(each, solver=solver, eps=eps, max_iters=max_iters)
                 for each in maps]
        record["by_outcome"] = [{"upper": each["upper"], "lower": each["lower"]}
                                for each in alone]
        record["max_over_outcomes"] = max(each["upper"] for each in alone)
        record["sum_over_outcomes"] = sum(each["upper"] for each in alone)
    record["seconds"] = time.time() - t0
    print(f"    diamond norm: {record['lower']:.6e} to {record['upper']:.6e}, Choi {record['choi_dim']}, "
          f"{record['status']}, {record['seconds']:.0f} s", flush=True)
    return record


def correctable_error_probability(group: dict, no_error_space: int = NO_ERROR_SPACE) -> float:
    r"""Largest probability of a correctable error at an error-accumulation step.

    :math:`p^{\mathrm{corr}}_n` of the square-root relation of
    :math:`\epsilon^{\mathrm{err\text{-}coh}}_n` at an error-accumulation
    step: the largest probability, over the states of the error space before
    the step, of reaching an error space other than that of no error.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups` of an error-accumulation step (one
        error space before the step).
    no_error_space : int
        Index of the error space after the step that holds no error.

    Returns
    -------
    float
    """
    selms = [selm for edge in group["edges"]
             for final, selm in enumerate(edge.effective_logical_process(repr="super")[:, 0])
             if final != no_error_space]
    return completely_positive_dnorm(sum(selms))


def physical_step_magnitudes(
    group: dict,
    step_names: Sequence[str] | None = None,
    error_accumulation_steps: Iterable[int] = (1,),
) -> dict:
    r"""Magnitudes of one step for one measurement record before it, on the simulated space.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.
    step_names : sequence of str or None
        Plain names of the steps (step 1 first); if ``None``, uses the edge
        name of the first outgoing edge.
    error_accumulation_steps : iterable of int
        Steps (counted from 1) at which ``p_corr`` is recorded.

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
    step_name = (step_names[step - 1] if step_names is not None
                 else str(group["edges"][0].name))
    record = {
        "step": step,
        "step_name": step_name,
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
    if step in error_accumulation_steps:
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


# ----------------------------------------- combinations & Appendix B bounds ----
def magnitudes_by_step(part: dict, n_steps: int = 7) -> dict:
    r"""Magnitude of each process at each step: the largest over the records before it.

    For the coherent processes the certified upper bound is used, and the
    lower bound is kept beside it.

    Parameters
    ----------
    part : dict
        Holds ``physical`` (list of :func:`physical_step_magnitudes` dicts)
        and ``logical`` (list of :func:`logical_step_magnitudes` dicts).
    n_steps : int
        Number of steps in the cycle.

    Returns
    -------
    dict
        Key of :data:`MAGNITUDES` -> list of ``n_steps`` values (step 1
        first; 0 where the process does not exist at the step), and, for the
        coherence magnitudes, the key ``<name>_lower`` with the lower bounds;
        ``p_corr`` at the error-accumulation steps.
    """
    result = {}
    sources = {name: part["physical"] for name in MAGNITUDES if name not in ("logi_coh", "logi_err")}
    sources |= {"logi_coh": part["logical"], "logi_err": part["logical"]}
    for name, (kind, _) in MAGNITUDES.items():
        for bound in (("upper", "lower") if kind == "coherence" else (None,)):
            values = [0.0] * n_steps
            for row in sources[name]:
                entry = row[name]
                if entry is None:
                    continue
                value = entry[bound] if bound else entry
                values[row["step"] - 1] = max(values[row["step"] - 1], value)
            result[name if bound in (None, "upper") else f"{name}_lower"] = values
    result["p_corr"] = {row["step"]: row["p_corr"] for row in part["physical"] if "p_corr" in row}
    return result


def pair_sum(earlier: Sequence[float], later: Sequence[float]) -> float:
    r"""Sum over the pairs of steps :math:`i<j` of ``earlier[i] * later[j]``.

    Parameters
    ----------
    earlier, later : sequence of float
        A magnitude at each step; the first factor comes from the earlier
        step.

    Returns
    -------
    float
    """
    n_steps = len(earlier)
    return float(sum(earlier[i] * later[j] for i in range(n_steps) for j in range(i + 1, n_steps)))


def table_combinations(
    eps: dict,
    n_steps: int = 7,
    error_accumulation_steps: Sequence[int] = (1,),
    measurement_steps: Sequence[int] = (5,),
) -> dict:
    r"""The combinations of Table I and the bounds of Appendix B on the discrepancies.

    Parameters
    ----------
    eps : dict
        Output of :func:`magnitudes_by_step`.
    n_steps : int
        Number of steps in the cycle.
    error_accumulation_steps : sequence of int
        Error-accumulation step indices (counted from 1).
    measurement_steps : sequence of int
        Measurement step indices (counted from 1).

    Returns
    -------
    dict
        Discrepancy key (``qec_aeb.REDUCTIONS``) -> ``conditions`` (name
        -> value of each combination of its row of Table I), ``bound`` (the
        bound of Appendix B), ``bound_formula``, and for some rows further
        forms of the bound.
    """
    steps = range(n_steps)
    leak, seep = eps["leak"], eps["seep"]
    (n_star,) = measurement_steps
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
                         if n not in error_accumulation_steps))
    return {
        "unexp": {
            "conditions": {"sum_eps_unexp": sum_unexp, "eps_leak_before_n_star": accumulated_leakage},
            "bound": float((np.sqrt(sum_unexp) + np.sqrt(accumulated_leakage)) ** 2),
            "bound_formula": "(sqrt(eps_unexp_{n_m}) + sqrt(eps_leak-before_{n_m}))^2",
            "bound_linear": 2 * sum_unexp + 2 * accumulated_leakage,
        },
        "leak_coh": {
            "conditions": {"sum_eps_leakcoh_i_eps_seepcoh_j": leak_coh_pair,
                           "N_sum_eps_leak_i_eps_seep_j": n_steps * leak_seep},
            "bound": pair_sum(created, returned),
            "bound_formula": "sum_{i<j} [eps_leak-coh_i + 2 sqrt(eps_seep_i) sum_{h<i} eps_leak_h]"
                             " [eps_seep-coh_j + sqrt(eps_leak_j) sum_{m>j} eps_seep_m]",
            "bound_table_form": leak_coh_pair + (2 + mu) * n_steps * leak_seep,
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


def square_root_relations(
    eps: dict,
    n_steps: int = 7,
    error_accumulation_steps: Sequence[int] = (1,),
    logical_dimension: int = 2,
    number_of_syndromes: int = 2,
) -> dict:
    r"""The square-root relations of Appendix B, evaluated as a check of the magnitudes.

    Each relation bounds a coherence magnitude by the population magnitude
    of the same step; the certified lower bound of the magnitude is tested,
    so a relation that fails is violated by the computed magnitudes.

    Parameters
    ----------
    eps : dict
        Output of :func:`magnitudes_by_step`.
    n_steps : int
        Number of steps in the cycle.
    error_accumulation_steps : sequence of int
        Error-accumulation step indices (counted from 1).
    logical_dimension : int
        Dimension :math:`d` of the logical subspace.
    number_of_syndromes : int
        Number of syndromes :math:`|S|`.

    Returns
    -------
    dict
        Relation -> ``formula``, ``steps`` (the steps it covers), ``ratio``
        (lower bound of the magnitude over the right-hand side, at each of
        these steps; ``None`` where both vanish), and ``holds``.
    """
    def check(name: str, right: list[float], steps: list[int], formula: str) -> dict:
        left = eps[f"{name}_lower"]
        ratios = {n: (left[n - 1] / right[n - 1] if right[n - 1] > 0 else
                      (None if left[n - 1] == 0 else "nonzero over zero")) for n in steps}
        return {"formula": formula, "steps": steps, "ratio": ratios,
                "holds": all(ratio is None or (not isinstance(ratio, str) and ratio <= 1)
                             for ratio in ratios.values())}

    every, after_first = list(range(1, n_steps + 1)), list(range(2, n_steps + 1))
    not_accumulation = [n for n in every if n not in error_accumulation_steps]
    d, S = logical_dimension, number_of_syndromes
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
            list(error_accumulation_steps),
            "eps_err-coh_n <= 2 sqrt(p_corr_n) + 2 (1 - 1/|S|) p_corr_n"),
        "logi_coh": check("logi_coh", [2 * np.sqrt(d * x) + (d + 1) * x for x in eps["logi_err"]],
                          every, "eps_logi-coh_n <= 2 sqrt(d eps_logi-err_n) + (d+1) eps_logi-err_n"),
    }
