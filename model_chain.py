r"""Process fidelities of the chain of models between the actual QEC dynamics and the failure Markov chain.

Evaluates the eight intermediate models from the actual evolution tree
(``F_T``) to the failure Markov chain (``F_M``), the seven discrepancies
between consecutive models, and single-mechanism dressed sweeps on a traversed
``qec_aeb.QEC_graph`` tree.
"""
from __future__ import annotations

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
from qec_aeb.QEC_graph import (
    FullCatTreeBuilder,
    StateEnsemble,
    component_projector_map,
    process_matrix_diagonal,
)

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model

# Steps of one QEC cycle. STEP_NAMES: plain names, used as JSON entries and
# in printed tables. SLOTS: prefix of the edge names of each step in the tree
# of chencrafts (ID, G1, PM, G2, MS, FL, RS)
STEP_NAMES = ["idle", "X_pi/2 pulse", "parity-mapping idling", "X_-pi/2 pulse",
              "measurement", "feedback latency", "conditional X_pi pulse"]
SLOTS = list(FullCatTreeBuilder.STEP_EDGE_PREFIXES)

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
# error space after the error-accumulation idling that the ideal operator of
# one photon loss produces: the idling of chencrafts has the ideal operators
# (no jump, one photon loss), and its edges of the logical evolution tree end
# in the error space of the same index
PHOTON_LOSS_ERROR_SPACE = 1

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

# Decoherence mechanisms of the dynamics: name -> (sweep key of the rate,
# ancilla labels (i, j) of |i><j| or None). They are the Lindblad operators
# that the tree builder keeps and whose rate is nonzero at the parameters of
# Ofek et al.; resonator dephasing (jump_adag_a) has zero rate there
MECHANISMS = {
    "photon_loss": ("jump_a", None),
    "photon_gain": ("jump_adag", None),
    "ancilla_relaxation": ("jump_ij", (0, 1)),
    "ancilla_excitation": ("jump_ij", (1, 0)),
    "ancilla_dephasing": ("jump_ij", (1, 1)),
    "ancilla_jump_to_level_2": ("jump_ij", (2, 1)),
    "excited_ancilla_photon_loss": ("jump_a_ij", (1, 1)),
}
# every family of rates that the dressed sweep stores
RATE_KEYS = ["jump_a", "jump_adag", "jump_adag_a", "jump_ij", "jump_a_ij", "jump_adag_ij"]
# the readout assignment errors are parameters of the measurement, not rates
ASSIGNMENT = "readout_assignment_error"
ASSIGNMENT_KEYS = ["M_ge", "M_eg"]

# Tolerance of complete positivity for the diamond norm without a semidefinite
# program (``chencrafts.cqed.dnorm`` with ``cp_only``): a negative eigenvalue
# of the Choi matrix above ``-CP_ATOL`` counts as rounding. The integrated
# ancilla pulses have Choi eigenvalues down to -2.5e-11, so the default
# tolerance of qutip (1e-12) would refuse them
CP_ATOL = 1e-9


def sweep_with_mechanisms(kept: list[str] | None):
    r"""
    Dressed sweep at the parameters of Ofek et al. with chosen decoherence mechanisms.

    The sweep is that of ``device_model.build_sweep``, so the Hamiltonian,
    the pulse parameters, the durations, and the angles of the ancilla frame
    changes are those of the full model whatever is kept. The stored Lindblad
    rates of the mechanisms that are not kept are then set to zero; the tree
    builder reads every rate from the sweep (the collapse operators of the
    idles, those of the pulses, and the photon-loss rate of the ideal idle
    operators, so the ideal error states follow the photon loss that is
    kept). The readout assignment errors are parameters and are replaced
    before the sweep is built.

    Parameters
    ----------
    kept : list of str or None
        Names of the mechanisms that act (keys of :data:`MECHANISMS`, and
        :data:`ASSIGNMENT`); an empty list is the model without decoherence.
        ``None`` is the full model: every stored rate is left as it is,
        including those of Lindblad operators that are not in
        :data:`MECHANISMS`.

    Returns
    -------
    sweep : cqed.FlexibleSweep
        Sweep with the chosen rates.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.
    """
    if kept is None:
        sweep, simulation_parameters, _ = device_model.build_sweep()
        return sweep, simulation_parameters
    unknown = sorted(set(kept) - set(MECHANISMS) - {ASSIGNMENT})
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}")
    # without the assignment errors the measurement reports the ancilla level
    overrides = {} if ASSIGNMENT in kept else {key: 0.0 for key in ASSIGNMENT_KEYS}
    sweep, simulation_parameters, _ = device_model.build_sweep(overrides)
    # every stored rate is set to zero except those of the kept mechanisms
    for key in RATE_KEYS:
        rate = sweep.sweep[key]
        multiplier = np.zeros_like(np.asarray(rate, dtype=float))
        for name, (rate_key, labels) in MECHANISMS.items():
            if rate_key != key or name not in kept:
                continue
            if labels is None:
                multiplier[...] = 1.0
            else:
                # the last two axes are the ancilla labels (i, j) of |i><j|
                multiplier[..., labels[0], labels[1]] = 1.0
        scaled = rate.copy()
        scaled[...] = np.asarray(rate) * multiplier
        sweep.sweep.store_data(**{key: scaled})
    return sweep, simulation_parameters


def slot_of(edge) -> str:
    r"""
    Step of the cycle to which an edge belongs, from the prefix of its name.

    Parameters
    ----------
    edge : EvolutionEdge
        Edge of an actual or logical evolution tree; its name looks like
        ``MS (0.4)``.

    Returns
    -------
    str
        One of :data:`SLOTS`.
    """
    name = str(edge.name)
    for slot in SLOTS:
        if name.startswith(slot):
            return slot
    raise ValueError(f"unrecognized edge name: {name}")


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


def losses_by_step(graph) -> dict:
    r"""
    Fidelity loss of the three models at each step of one QEC cycle, and the
    split of the step contributions by logical manifestation.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle.

    Returns
    -------
    dict
        ``fidelity``: the output of :func:`fidelities_by_step`; ``loss``: for
        each of its models, the decrease of the process fidelity at each of
        the seven steps (array); ``chain_by_manifestation``: for each key of
        :data:`MANIFESTATION_COLUMNS`, the failure probability of the failure
        Markov chain at each step with that logical manifestation (array).

    Raises
    ------
    ValueError
        If the tree does not hold exactly one QEC cycle.
    RuntimeError
        If the four parts of a step contribution do not add
        up to the entry.
    """
    fidelity = fidelities_by_step(graph)
    if len(fidelity["F_T"]) != device_model.STEPS_PER_CYCLE + 1:
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
    outcome (the record ends in a terminated node), so the dynamical maps
    composed up to that outcome (``StateNode.full_process``) give the final
    state of the record, which is read in the logical subspace without a
    recovery.

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
    # the isometry from the logical qubit onto the logical subspace
    logical = graph.nodes[0].ideal_encoders()[0]
    # an edge at which a record becomes unexpected ends in a terminated node
    first_unexpected = [edge for edge in graph.edges
                        if edge.final_state.terminated and not edge.init_state.terminated]
    return sum(
        (cqed.superop_between_spaces(edge.final_state.full_process(), logical)
         for edge in first_unexpected),
        0 * qt.to_super(qt.qeye(2)),
    )


def elm_keeping_components(graph, kept_after_step, read_after_step: int | None = None) -> qt.Qobj:
    r"""
    ELM in which chosen components of the state are kept after each step.

    The dynamical maps are composed on the simulated space for every
    expected measurement record. After step :math:`n` the
    state is projected onto the components that ``kept_after_step`` names
    for that step: the component within the joint error space, the component
    within the leakage subspace, and the error--leakage coherence
    (:data:`WITHIN_ERROR`, :data:`WITHIN_LEAKAGE`, :data:`COHERENCE`), with
    the error spaces of the measurement record after the step. The composed
    map of each record is read in each of its final error spaces: it encodes
    the logical qubit before the cycle and decodes it with the ideal error
    states of that error space.

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
    root = graph.nodes[0]
    # the isometry from the logical qubit onto the logical subspace before the cycle
    initial = root.ideal_encoders()[0]
    # None stands for the identity map before the first step
    composed = {id(root): None}
    # nodes at which the ELM is read when it is read before the last step
    read_nodes = []
    for step, node, edges in graph.iter_steps():
        # the steps after the reading step do not enter the ELM
        if read_after_step is not None and step > read_after_step:
            break
        if step == read_after_step:
            read_nodes.extend(edge.final_state for edge in edges)
        components = kept_after_step(step)
        for edge in edges:
            final = edge.final_state
            kept_step = edge.physical_map()
            # projector map onto the kept components after the step
            if components is not None:
                encoders = final.ideal_encoders()
                kept_step = sum(
                    component_projector_map(encoders, component) for component in components
                ) * kept_step
            before = composed[id(node)]
            composed[id(final)] = kept_step if before is None else kept_step * before
    leaves = graph.final_nodes() if read_after_step is None else read_nodes
    return sum(
        cqed.superop_between_spaces(composed[id(leaf)], initial, encoder)
        for leaf in leaves for encoder in leaf.ideal_encoders())


def chain_fidelities(graph) -> dict:
    r"""
    Process fidelities of the eight models of the chain and their differences.

    ``F_T``, ``F_G``, ``F_twirled``, and ``F_M`` come from the trees of
    ``chencrafts``. ``F_cons`` is ``F_T`` minus the process fidelity of the
    ELM of the unexpected records. ``F_pop`` and ``F_err`` compose the
    dynamical maps on the simulated space (:func:`elm_keeping_components`).
    ``F_trans`` composes the step-wise effective logical maps along every
    sequence of error spaces (``EvolutionTree.traverse_and_approx``), ``F_G``
    along the sequences of the ideal evolution tree only, and ``F_twirled``
    composes their Pauli twirls along the same sequences
    (``LogicalEvolutionTree.process_fidelity_twirled_by_step``).

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
        ``composition_check``: ``F_cons`` composed on the simulated space
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
    # the three models composed on the simulated space
    components = {
        name: float(process_matrix_diagonal(
            elm_keeping_components(graph, lambda step, kept=kept: kept))[0])
        for name, kept in KEPT_COMPONENTS.items()}
    # A_trans: the SELMs composed along every sequence of error spaces, added
    # over the error spaces of the final nodes
    graph.traverse_and_approx()
    a_trans = StateEnsemble(graph.final_nodes()).accum_logical_process(
        repr="super", force_sum=True)
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


def photon_loss_trajectories(graph) -> dict:
    r"""
    Process fidelities of the logical evolution tree and of its Pauli-twirled
    version on the error trajectories that start with the photon loss of the
    error-accumulation idling.

    These are the trajectories whose error history holds one photon loss
    during the error-accumulation idling, the first step of the cycle. Their
    process fidelities are the sums of :math:`\mathrm{tr}(S)/4` over the
    final states reached through the state of one photon loss after that
    step (``process_fidelity_twirled_by_step`` with ``through_node``).

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle.

    Returns
    -------
    dict
        ``F_G`` and ``F_twirled`` on these trajectories, and ``logi_coh``,
        their difference: the part of :math:`\delta_{\mathrm{logi\text{-}coh}}`
        (signed, ``F_G`` minus ``F_twirled``) that these trajectories carry.
    """
    logical = graph.logical_evolution_tree()
    root = next(node for node in logical.nodes if node.step_index == 0)
    # the edge of the error-accumulation idling into the error space of one
    # photon loss; its name ends in "error space <before> to <after>"
    loss_edges = [edge for edge in root.out_edges
                  if slot_of(edge) == SLOTS[0]
                  and int(str(edge.name).rsplit(" to ", 1)[1]) == PHOTON_LOSS_ERROR_SPACE]
    if len(loss_edges) != 1:
        raise RuntimeError(f"expected one photon-loss state after the first step, "
                           f"found {len(loss_edges)}")
    loss_state = loss_edges[0].final_state
    n_steps = max(node.step_index for node in logical.nodes)
    every_step = range(1, n_steps + 1)
    f_g = float(logical.process_fidelity_twirled_by_step(
        every_step, through_node=loss_state)[n_steps])
    f_twirled = float(logical.process_fidelity_twirled_by_step(
        through_node=loss_state)[n_steps])
    return {"F_G": f_g, "F_twirled": f_twirled, "logi_coh": f_g - f_twirled}
