r"""Process fidelities of the chain of eight models between the actual QEC dynamics and the failure Markov chain.

Evaluates the eight intermediate models from the actual evolution tree
(``F_T``) to the failure Markov chain (``F_M``), the seven discrepancies
between consecutive models, and the breakdown by logical manifestation on any
traversed :class:`~qec_aeb.EvolutionTree`.
"""

from __future__ import annotations

__all__ = [
    "CHAIN",
    "REDUCTIONS",
    "MANIFESTATIONS",
    "MANIFESTATION_COLUMNS",
    "WITHIN_ERROR",
    "WITHIN_LEAKAGE",
    "COHERENCE",
    "KEPT_COMPONENTS",
    "CP_ATOL",
    "completely_positive_dnorm",
    "manifestation_probabilities",
    "markov_manifestation_probabilities",
    "fidelities_by_step",
    "losses_by_step",
    "unexpected_record_elm",
    "elm_keeping_components",
    "chain_fidelities",
]

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
from .graph import process_matrix_diagonal
from .node import StateEnsemble

# order of the models in the chain, from the actual QEC dynamics to the
# failure Markov chain
CHAIN = ["F_T", "F_cons", "F_pop", "F_err", "F_trans", "F_G", "F_twirled", "F_M"]
# discrepancy of each reduction: (name, model before, model after). The names
# are the labels of the discrepancies with the hyphen written as an underscore
REDUCTIONS = [
    ("unexp", "F_T", "F_cons"),
    ("leak_coh", "F_cons", "F_pop"),
    ("leak", "F_pop", "F_err"),
    ("err_coh", "F_err", "F_trans"),
    ("err_traj", "F_trans", "F_G"),
    ("logi_coh", "F_G", "F_twirled"),
    ("logi_err", "F_twirled", "F_M"),
]

# entries by which a model is resolved: the process fidelity, the three
# logical errors, and leakage
MANIFESTATIONS = ["I", "X", "Y", "Z", "leak"]
# logical manifestation of a failure, as a key of the result files, to its
# column in ``LogicalEvolutionTree.failure_prob_by_step_and_manifestation``
# (columns X, Y, Z, leakage); the order is that of the legends, the stacked
# bar from the top down
MANIFESTATION_COLUMNS = {"leakage": 3, "X": 0, "Y": 1, "Z": 2}

# Components of an operator on the simulated space, in the notation of
# ``component_projector_map``: within the joint error space, within the
# leakage subspace, and the error--leakage coherence between the two. The
# three projector maps add up to the identity map
WITHIN_ERROR, WITHIN_LEAKAGE, COHERENCE = ("L", "L"), ("p", "p"), ("L", "p")
# components kept after each step by the three models of the chain that are
# composed on the simulated space; None keeps every component
KEPT_COMPONENTS = {
    "F_cons": None,
    "F_pop": (WITHIN_ERROR, WITHIN_LEAKAGE),
    "F_err": (WITHIN_ERROR,),
}

# Tolerance of complete positivity for the diamond norm without a semidefinite
# program (``chencrafts.cqed.dnorm`` with ``cp_only``): a negative eigenvalue
# of the Choi matrix above ``-CP_ATOL`` counts as rounding. The integrated
# ancilla pulses have Choi eigenvalues down to -2.5e-11, so the default
# tolerance of qutip (1e-12) would refuse them
CP_ATOL = 1e-9


def completely_positive_dnorm(superop: qt.Qobj) -> float:
    r"""
    Diamond norm of a completely positive map, without a semidefinite program.

    For a completely positive map :math:`\Phi`,
    :math:`\lVert\Phi\rVert_\diamond=\lVert\Phi^\dagger(\hat I)\rVert_\infty`,
    the largest probability, over the states before the map, of the process
    that :math:`\Phi` describes. ``chencrafts.cqed.dnorm`` evaluates this
    formula when the Choi matrix of the map is positive within
    :data:`CP_ATOL`.

    Parameters
    ----------
    superop : qutip.Qobj
        Superoperator of the map, for example a dynamical map between two
        projector maps of the form :math:`\hat P\,\cdot\,\hat P`.

    Returns
    -------
    float
        The diamond norm. A map that is not completely positive within the
        tolerance raises instead of starting a semidefinite program on the
        whole simulated space.
    """
    value = cqed.dnorm(superop, cp_only=True, cp_atol=CP_ATOL)
    if np.isnan(value):
        raise RuntimeError(
            f"the map is not completely positive within {CP_ATOL}; its diamond "
            "norm needs a semidefinite program")
    return value


def manifestation_probabilities(elm: qt.Qobj) -> list[float]:
    r"""
    Process fidelity, logical-error probabilities, and leakage probability of an ELM.

    The first four entries are the diagonal of the process matrix of the ELM
    in the Pauli basis, :math:`\mathcal X_{II}` (the process fidelity),
    :math:`\mathcal X_{XX}`, :math:`\mathcal X_{YY}`, :math:`\mathcal X_{ZZ}`.
    The last is :math:`1-\mathrm{tr}\,\mathcal X`, the probability that the
    ELM does not retain for the maximally mixed logical state.

    Parameters
    ----------
    elm : qutip.Qobj
        Map on the logical qubit, as a superoperator.

    Returns
    -------
    list of float
        The five entries, in the order of :data:`MANIFESTATIONS`.
    """
    diagonal = process_matrix_diagonal(elm)
    return [*diagonal.tolist(), float(1.0 - diagonal.sum())]


def markov_manifestation_probabilities(tree) -> list[float]:
    r"""
    Final success probability of the failure Markov chain and its failure
    probability by logical manifestation, added over the steps.

    The failure Markov chain has no ELM. Its entries are the failure
    probabilities of the steps split by logical manifestation
    (``LogicalEvolutionTree.failure_prob_by_step_and_manifestation``), added
    over the steps; a failure is counted with the manifestation it has at the
    step at which it occurs.

    Parameters
    ----------
    tree : LogicalEvolutionTree
        Logical evolution tree of one QEC cycle.

    Returns
    -------
    list of float
        :math:`F_{\mathbb M}` and the probabilities of the logical X, Y, Z
        errors and of leakage, in the order of :data:`MANIFESTATIONS`. The
        five entries add up to one.
    """
    accumulated = tree.failure_prob_by_step_and_manifestation().sum(axis=0)
    return [float(tree.intermediate_success_prob_by_step()[-1]), *accumulated.tolist()]


def fidelities_by_step(graph) -> dict:
    r"""
    Process fidelities of the three models after every step.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.

    Returns
    -------
    dict
        ``F_T`` (actual evolution tree: the process fidelity of the ELM
        reached when all later operations are taken as ideal), ``F_G``
        (logical evolution tree: the same for the composed step-wise
        effective logical maps), and ``F_M`` (failure Markov chain: the
        total probability of the success states), each an array whose entry
        :math:`n` belongs to the state after :math:`n` steps; entry 0 is
        before the first step.
    """
    logical = graph.logical_evolution_tree()
    return {
        "F_T": np.asarray(graph.process_fidelity_by_step(), float),
        "F_G": np.asarray(logical.process_fidelity_by_step(), float),
        "F_M": np.asarray(logical.intermediate_success_prob_by_step(), float),
    }


def losses_by_step(graph, steps_per_cycle: int | None = 7) -> dict:
    r"""
    Fidelity loss of the three models at each step of one QEC cycle, and the
    split of the step contributions by logical manifestation.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle.
    steps_per_cycle : int or None
        Expected number of steps in the cycle (7 by default); ``None`` skips
        the step-count check.

    Returns
    -------
    dict
        ``fidelity``: the output of :func:`fidelities_by_step`; ``loss``: for
        each of its models, the decrease of the process fidelity at each
        step (array); ``chain_by_manifestation``: for each key of
        :data:`MANIFESTATION_COLUMNS`, the failure probability of the failure
        Markov chain at each step with that logical manifestation (array).

    Raises
    ------
    ValueError
        If ``steps_per_cycle`` is given and the tree does not hold that many steps.
    RuntimeError
        If the four parts of a step contribution do not add up to the entry.
    """
    fidelity = fidelities_by_step(graph)
    if steps_per_cycle is not None and len(fidelity["F_T"]) != steps_per_cycle + 1:
        raise ValueError("the tree does not hold exactly one QEC cycle")
    loss = {name: values[:-1] - values[1:] for name, values in fidelity.items()}
    # failure probability of the failure Markov chain at each step, by
    # logical manifestation; row 0 (before the first step) is zero
    split = graph.logical_evolution_tree().failure_prob_by_step_and_manifestation()[1:]
    # the four parts of a step must add up to its term
    if np.abs(split.sum(axis=1) - loss["F_M"]).max() > 1e-12:
        raise RuntimeError("the split by logical manifestation does not add up to the entries")
    return dict(
        fidelity=fidelity, loss=loss,
        chain_by_manifestation={key: split[:, column]
                                for key, column in MANIFESTATION_COLUMNS.items()})


def unexpected_record_elm(graph) -> qt.Qobj:
    r"""
    ELM of the unexpected full measurement records.

    This is :math:`\tilde{\mathcal A}_{\mathbb T}-\tilde{\mathcal
    A}_{\mathrm{cons}}`. The protocol applies no operation after an unexpected
    outcome (the record ends in a terminated node), so the real process on
    encoded initial states (``StateNode.real_process_on_encoded``) gives the
    final state of the record, which is decoded in the initial logical
    subspace without a recovery.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.

    Returns
    -------
    qutip.Qobj
        The map on the logical qubit, as a superoperator; zero when every
        unexpected record ends outside the logical subspace.
    """
    # decode in the initial logical subspace (where the ancilla is in |0>)
    logical_decoder = qt.to_super(graph.nodes[0].ideal_encoders()[0].dag())
    # an edge at which a record becomes unexpected ends in a terminated node
    first_unexpected = [edge for edge in graph.edges
                        if edge.final_state.terminated and not edge.init_state.terminated]
    return sum(
        (logical_decoder * edge.final_state.real_process_on_encoded
         for edge in first_unexpected),
        0 * qt.to_super(qt.qeye(2)),
    )


def elm_keeping_components(graph, kept_after_step, read_after_step: int | None = None) -> qt.Qobj:
    r"""
    ELM in which chosen components of the state are kept after each step.

    Delegates to :meth:`~qec_aeb.EvolutionTree.physical_elm`: starts
    from ``root.real_process_on_encoded`` (:math:`\mathcal S_0`), walks the
    tree breadth-first with ``graph.iter_steps()``, projects after step
    :math:`n` onto ``kept_after_step(n)``, and decodes at the target step.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.
    kept_after_step : callable
        Maps the step index :math:`n\geq1` to a tuple of components, or to
        ``None`` for every component.
    read_after_step : int or None
        Step :math:`m` after which the ELM is read, in the error spaces of
        the nodes after step :math:`m` (the later steps are taken as ideal,
        as in the process fidelity after :math:`m` steps); ``None`` reads it
        after the last step of the tree, at its final nodes.

    Returns
    -------
    qutip.Qobj
        The ELM, as a superoperator on the logical qubit.
    """
    return graph.physical_elm(
        kept_after_step=kept_after_step,
        read_after_step=read_after_step,
    )


def chain_fidelities(graph) -> dict:
    r"""
    Process fidelities of the eight models of the chain and their differences.

    The physical-space models (``F_cons``, ``F_pop``, ``F_err``) come from
    :meth:`~qec_aeb.EvolutionTree.physical_elm` with the component
    projectors of :data:`KEPT_COMPONENTS` (and ``F_T`` adds
    :func:`unexpected_record_elm`). The error-space SELM models come from
    :meth:`~qec_aeb.EvolutionTree.logical_elm` (``transitions="all"``
    for ``F_trans``, ``transitions="intended"`` for ``F_G``, and
    ``transitions="intended", twirled=True`` for ``F_twirled``), and ``F_M``
    comes from the failure Markov chain.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree. Every final node is after the last
        step (a tree of complete QEC cycles).

    Returns
    -------
    dict
        ``fidelity``: the process fidelity of each model in :data:`CHAIN`;
        ``difference``: for each reduction in :data:`REDUCTIONS`, the signed
        difference (model before minus model after), the discrepancy of that
        reduction;
        ``manifestation``: for the actual QEC dynamics (``F_T``) and the
        failure Markov chain (``F_M``), the process fidelity and the
        probabilities of the logical X, Y, Z errors and of leakage, in the
        order of :data:`MANIFESTATIONS`;
        ``composition_check``: ``F_cons`` composed via ``physical_elm``
        minus ``F_cons`` from the trees.
    """
    by_step = fidelities_by_step(graph)
    logical = graph.logical_evolution_tree()
    n_steps = len(by_step["F_T"]) - 1
    # ELM of the actual evolution tree: every expected measurement record,
    # decoded from every error space of its final node
    a_t = StateEnsemble(graph.final_nodes()).effective_logical_process(
        repr="super", force_sum=True)
    unexpected = unexpected_record_elm(graph)
    # the three physical-space models (F_cons, F_pop, F_err)
    components = {
        name: float(process_matrix_diagonal(
            graph.physical_elm(lambda step, kept=kept: kept))[0])
        for name, kept in KEPT_COMPONENTS.items()}
    # A_trans: the SELMs composed along every sequence of error spaces
    a_trans = graph.logical_elm(transitions="all")
    fidelity = {
        "F_T": float(by_step["F_T"][n_steps]),
        "F_cons": float(by_step["F_T"][n_steps] - process_matrix_diagonal(unexpected)[0]),
        "F_pop": components["F_pop"],
        "F_err": components["F_err"],
        "F_trans": float(process_matrix_diagonal(a_trans)[0]),
        "F_G": float(by_step["F_G"][n_steps]),
        "F_twirled": float(logical.process_fidelity_twirled_by_step()[n_steps]),
        "F_M": float(by_step["F_M"][n_steps]),
    }
    # the composition on the simulated space must reproduce F_cons of the trees
    check = components["F_cons"] - fidelity["F_cons"]
    if abs(check) > 1e-9:
        raise RuntimeError(f"composition on the simulated space differs: {check}")
    manifestation = {
        "F_T": manifestation_probabilities(a_t),
        "F_M": markov_manifestation_probabilities(logical),
    }
    # the first entry of each must be the process fidelity found above
    for name, entries in manifestation.items():
        if abs(entries[0] - fidelity[name]) > 1e-9:
            raise RuntimeError(f"{name}: process-matrix entry {entries[0]} differs "
                               f"from the fidelity {fidelity[name]}")
    return {
        "fidelity": fidelity,
        "difference": {name: fidelity[before] - fidelity[after]
                       for name, before, after in REDUCTIONS},
        "manifestation": manifestation,
        "composition_check": check,
    }
