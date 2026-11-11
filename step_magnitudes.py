r"""Step-by-step magnitudes of the neglected processes that move population.

Computes the exact diamond-norm magnitudes of leakage, seepage, neglected
error-space transitions, and unexpected measurement outcomes on a traversed
``qec_aeb.QEC_graph.EvolutionTree`` via largest eigenvalues of adjoint projector
images.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
from qec_aeb.QEC_graph import component_projector_map

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model  # noqa: F401
import model_chain


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
        ``step`` (1 to 7), ``record`` (the measurement record before the
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
    It is maximized over the error spaces before the step. The idle is the
    error-accumulation step: the ideal evolution tree contains its
    transitions, and the magnitude is zero there by definition.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    float
        The magnitude for the measurement record before the step.
    """
    # for each error space before the step, the operator on its logical
    # states whose expectation value is the probability of an absent transition
    per_initial = {}
    for edge in group["edges"]:
        if model_chain.slot_of(edge) == model_chain.SLOTS[0]:
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
        group["node"].ideal_encoders(), model_chain.WITHIN_ERROR)
    return model_chain.completely_positive_dnorm(
        sum(edge.physical_map() for edge in group["unexpected_edges"]) * onto_error)
