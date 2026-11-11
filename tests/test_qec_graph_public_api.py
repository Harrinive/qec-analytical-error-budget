r"""Tests of the public functions of ``qec_aeb.QEC_graph`` that read a built tree.

The functions return the projector onto the joint error space, the projector
maps onto the components of an operator, the nodes and edges of a tree in the
order of the steps, the process fidelity after each step, and the step-wise
success probability. Each test compares one function with the same expression
written out by hand, on a two-step tree small enough to check by hand.
"""

import numpy as np
import pytest
import qutip as qt

import qec_aeb.QEC_graph as QEC_graph
from qec_aeb.QEC_graph import utils
from qec_aeb.QEC_graph.edge import MeasurementEdge, PropagatorEdge
from qec_aeb.QEC_graph.graph import EvolutionTree
from qec_aeb.QEC_graph.node import StateNode

DIM = 5


def _encoders(num_error_spaces: int, seed: int = 0) -> np.ndarray:
    r"""Isometric encoders of orthogonal two-dimensional error spaces.

    Parameters
    ----------
    num_error_spaces : int
        Number of error spaces; ``2 * num_error_spaces <= DIM``.
    seed : int
        Seed of the random unitary whose columns span the error spaces.

    Returns
    -------
    np.ndarray
        Object array of Qobj encoders of dims ``[[DIM], [2]]``.
    """
    # the columns of a random unitary (from a QR decomposition) are
    # orthonormal, so each pair of columns spans one error space that is
    # orthogonal to the others
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM)))
    encoders = np.ndarray(num_error_spaces, dtype=qt.Qobj)
    for a in range(num_error_spaces):
        encoders[a] = qt.Qobj(q[:, 2 * a: 2 * a + 2], dims=[[DIM], [2]])
    return encoders


def _small_tree(loss: float = 0.1) -> EvolutionTree:
    r"""A traversed two-step tree: a lossy step, then a two-outcome measurement.

    The logical states are the levels 0 and 1 of a ``DIM``-level system. Step
    1 moves the population of level 1 to level 2, which lies in the leakage
    subspace, with probability ``loss``; its ideal operator is the identity.
    Step 2 measures whether the system is in levels 0-1 (outcome 0) or
    elsewhere (outcome 1). No correctable error can produce outcome 1, so
    the measurement record with outcome 1 is unexpected and its node is
    terminated.

    Parameters
    ----------
    loss : float
        Probability that level 1 moves to level 2 in step 1.

    Returns
    -------
    EvolutionTree
        Tree with nodes ``[initial, after step 1, outcome 0, outcome 1]``.
    """
    eye = qt.qeye(DIM)
    # step 1: amplitude damping from level 1 to level 2
    keep = qt.Qobj(np.diag([1, np.sqrt(1 - loss), 1, 1, 1]))
    jump = np.sqrt(loss) * qt.basis(DIM, 2) * qt.basis(DIM, 1).dag()
    step_1 = qt.kraus_to_super([keep, jump])
    # step 2: projective measurement of "inside levels 0-1" or not
    inside = qt.basis(DIM, 0).proj() + qt.basis(DIM, 1).proj()
    outside = eye - inside

    # the initial node holds the encoded logical qubit; the others are filled by the traversal
    tree = EvolutionTree()
    nodes = [StateNode.initial_note(
        (1 / np.sqrt(2), 1 / np.sqrt(2)), qt.basis(DIM, 0), qt.basis(DIM, 1),
    )] + [StateNode() for _ in range(3)]
    for node in nodes:
        tree.add_node(node)
    tree.add_edge_connect(PropagatorEdge("step 1", step_1, [eye]), nodes[0], nodes[1])
    tree.add_edge_connect(
        MeasurementEdge("measure", 0, qt.sprepost(inside, inside), [inside]), nodes[1], nodes[2])
    tree.add_edge_connect(
        MeasurementEdge("measure", 1, qt.sprepost(outside, outside), [outside]), nodes[1], nodes[3])
    tree.traverse(2)
    return tree


# projectors and projector maps ------------------------------------------------
def test_joint_error_space_projector_is_the_sum_of_the_projectors_of_the_error_spaces():
    """A: P_err = sum_a E_a E_a^dag."""
    encoders = _encoders(2)
    projector = QEC_graph.joint_error_space_projector(encoders)
    expected = sum(enc * enc.dag() for enc in encoders)
    assert (projector - expected).norm() == 0
    # orthogonal error spaces: a projector of rank 4, equal to the projector onto the span
    assert (projector * projector - projector).norm() < 1e-12
    assert projector.tr() == pytest.approx(4)
    columns = np.column_stack([enc.full() for enc in encoders])
    basis, _ = np.linalg.qr(columns)
    assert np.allclose(projector.full(), basis @ basis.conj().T, atol=1e-12)


def test_component_projector_maps_resolve_the_identity_and_match_the_kron_form():
    """A, C: the projector maps of the joint error space, the leakage subspace and the error-leakage coherence add up to the identity map."""
    encoders = _encoders(2, seed=1)
    p_err = QEC_graph.joint_error_space_projector(encoders).full()
    p_leak = np.eye(DIM) - p_err

    err = QEC_graph.component_projector_map(encoders, ("L", "L"))
    leak = QEC_graph.component_projector_map(encoders, ("p", "p"))
    coherence = QEC_graph.component_projector_map(encoders, ("L", "p"))
    # column stacking: vec(A rho B) = (B^T kron A) vec(rho)
    assert np.allclose(err.full(), np.kron(p_err.T, p_err), atol=1e-12)
    assert np.allclose(leak.full(), np.kron(p_leak.T, p_leak), atol=1e-12)
    assert np.allclose(
        coherence.full(), np.kron(p_leak.T, p_err) + np.kron(p_err.T, p_leak), atol=1e-12)
    assert np.allclose((err + leak + coherence).full(), np.eye(DIM ** 2), atol=1e-12)
    # the default component is the one within the joint error space
    assert (QEC_graph.component_projector_map(encoders) - err).norm() == 0

    # within the joint error space: the individual error spaces plus the coherence between them
    within = QEC_graph.component_projector_map(encoders, "LD")
    between = QEC_graph.component_projector_map(encoders, "LC")
    assert np.allclose((within + between).full(), err.full(), atol=1e-12)
    single = QEC_graph.component_projector_map(encoders, (1, 1))
    p_1 = (encoders[1] * encoders[1].dag()).full()
    assert np.allclose(single.full(), np.kron(p_1.T, p_1), atol=1e-12)


def test_node_joint_error_space_projector_uses_the_ideal_encoders_of_the_node():
    """A: the projector of the node after step 1 is the projector onto levels 0 and 1."""
    tree = _small_tree()
    node = tree.nodes[1]
    projector = node.joint_error_space_projector()
    expected = qt.basis(DIM, 0).proj() + qt.basis(DIM, 1).proj()
    assert (projector - expected).norm() < 1e-12
    # a node that was never evolved has no ideal logical states, and the method raises AttributeError
    with pytest.raises(AttributeError):
        StateNode().joint_error_space_projector()


# process fidelity after each step ---------------------------------------------
def test_process_fidelity_by_step_of_a_lossy_step():
    """E: the lossy step has a known process fidelity, which the measurement of the expected outcome keeps."""
    loss = 0.1
    tree = _small_tree(loss)
    fidelity = tree.process_fidelity_by_step()
    assert isinstance(fidelity, np.ndarray) and fidelity.dtype == float
    # Kraus operator diag(1, sqrt(1-loss)) on the logical qubit: |tr K|^2 / 4
    expected = (1 + np.sqrt(1 - loss)) ** 2 / 4
    # the fidelity is tr(S)/d^2, linear in the map, so it is exact up to rounding
    assert fidelity == pytest.approx([1.0, expected, expected], abs=1e-12)


# iteration over the nodes and edges -------------------------------------------
def _reference_iteration(graph):
    r"""Breadth-first iteration over a tree, written out independently of ``iter_steps``.

    Parameters
    ----------
    graph : EvolutionTree
        The tree to iterate over.

    Yields
    ------
    (step, node, edges)
        The index of a step, a node that represents a state before that
        step, and the edges of the step that start from the node and do
        not end at a terminated node.
    """
    layer = [(graph.nodes[0], 0)]
    while layer:
        nxt = []
        for node, step in layer:
            edges = [e for e in node.out_edges if not e.final_state.terminated]
            if edges:
                yield step + 1, node, edges
            nxt.extend((e.final_state, step + 1) for e in edges)
        layer = nxt


def test_iter_steps_matches_an_independent_breadth_first_iteration():
    """E: the same (step, node, edges) as the independent iteration, in the same order; terminated nodes are skipped."""
    tree = _small_tree()
    assert tree.nodes[3].terminated and not tree.nodes[2].terminated
    walked = list(tree.iter_steps())
    assert walked == list(_reference_iteration(tree))
    assert [(step, node.index, [e.index for e in edges]) for step, node, edges in walked] \
        == [(1, 0, [0]), (2, 1, [1])]
    # with the terminated node kept, the second step has both outcomes
    everything = list(tree.iter_steps(skip_terminated=False))
    assert [(step, node.index, [e.index for e in edges]) for step, node, edges in everything] \
        == [(1, 0, [0]), (2, 1, [1, 2])]


def test_final_nodes_have_no_edge_to_a_node_that_is_not_terminated():
    """E: nodes that are not terminated and have no edge to a node that is not terminated."""
    tree = _small_tree()
    expected = [n for n in tree.nodes if not n.terminated and not [
        e for e in n.out_edges if not e.final_state.terminated]]
    assert tree.final_nodes() == expected == [tree.nodes[2]]
    assert tree.final_nodes(skip_terminated=False) == [tree.nodes[2], tree.nodes[3]]


def test_iter_steps_and_final_nodes_work_on_the_logical_evolution_tree():
    """D: ``iter_steps`` and ``final_nodes`` also accept a logical evolution tree."""
    logical = _small_tree().logical_evolution_tree()
    assert [step for step, _, _ in logical.iter_steps()] == [1, 2]
    assert [node.step_index for node in logical.final_nodes()] == [2]


# step-wise success probability ------------------------------------------------
def test_step_wise_success_probability_equals_trace_over_four():
    """A: g = chi_II / 4 = tr(S) / 4."""
    rng = np.random.default_rng(3)
    kraus = [qt.Qobj(0.5 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))) for _ in range(2)]
    selm = qt.kraus_to_super(kraus)
    g = QEC_graph.step_wise_success_probability(selm)
    assert g == pytest.approx(np.real(np.trace(selm.full())) / 4, abs=1e-14)
    assert g == pytest.approx(sum(abs(k.tr()) ** 2 for k in kraus) / 4, abs=1e-14)
    assert QEC_graph.step_wise_success_probability(qt.to_super(qt.qeye(2))) == pytest.approx(1.0)


def test_process_matrix_diagonal_of_a_pauli_channel():
    """A: a mixture of Pauli operations has its probabilities on the diagonal, in the order I, X, Y, Z."""
    probabilities = [0.6, 0.1, 0.05, 0.2]
    paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    channel = sum(p * qt.to_super(op) for p, op in zip(probabilities, paulis))
    assert QEC_graph.process_matrix_diagonal(channel) == pytest.approx(probabilities, abs=1e-14)


def test_failure_prob_by_step_and_manifestation_of_a_lossy_step():
    """E: the lossy step fails by a logical Z error and by leakage with known probabilities; the measurement does not fail."""
    loss = 0.1
    logical = _small_tree(loss).logical_evolution_tree()
    failure_prob = logical.failure_prob_by_step_and_manifestation()
    # Kraus operator diag(1, sqrt(1-loss)) = a I + b Z on the logical qubit:
    # a Z error with probability b^2, and the lost population loss/2 of the
    # maximally mixed logical state as leakage
    b = (1 - np.sqrt(1 - loss)) / 2
    assert failure_prob == pytest.approx(
        np.array([[0, 0, 0, 0], [0, 0, b ** 2, loss / 2], [0, 0, 0, 0]]), abs=1e-12)
    # the four manifestations of a step add up to its failure probability
    success = logical.intermediate_success_prob_by_step()
    assert failure_prob.sum(axis=1)[1:] == pytest.approx(success[:-1] - success[1:], abs=1e-12)


# process fidelity of the twirled logical evolution tree -----------------------
def _branching_rotation_tree(jump: float, theta: float) -> EvolutionTree:
    r"""A traversed two-step tree with two error spaces and a logical rotation on each branch.

    The physical space has four levels; the logical states are the levels 0
    and 1, and the levels 2 and 3 form the error space of a jump. Step 1 is
    an error-accumulation step: with probability ``jump`` the jump
    :math:`\hat J=|2\rangle\langle0|+|3\rangle\langle1|` occurs together
    with the logical rotation :math:`\hat R=\exp(-\mathrm i\theta\hat X/2)`,
    and otherwise nothing happens; its ideal operators are the identity and
    :math:`\hat J`. Step 2 applies :math:`\hat R` within each error space,
    with the identity as its ideal operator.

    Parameters
    ----------
    jump : float
        Probability of the jump in step 1.
    theta : float
        Rotation angle of :math:`\hat R`.

    Returns
    -------
    EvolutionTree
        Tree with nodes ``[initial, after step 1, after step 2]``.
    """
    dim = 4
    jump_op = qt.basis(dim, 2) * qt.basis(dim, 0).dag() + qt.basis(dim, 3) * qt.basis(dim, 1).dag()
    rotation_2 = (-0.5j * theta * qt.sigmax()).expm().full()
    # R on the levels 0-1 in step 1; R on the levels 0-1 and on the levels 2-3 in step 2
    rotation_low = qt.Qobj(np.block([[rotation_2, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]]))
    rotation_both = qt.Qobj(np.kron(np.eye(2), rotation_2))
    step_1 = qt.kraus_to_super([np.sqrt(1 - jump) * qt.qeye(dim), np.sqrt(jump) * jump_op * rotation_low])
    tree = EvolutionTree()
    nodes = [StateNode.initial_note(
        (1 / np.sqrt(2), 1 / np.sqrt(2)), qt.basis(dim, 0), qt.basis(dim, 1),
    )] + [StateNode() for _ in range(2)]
    for node in nodes:
        tree.add_node(node)
    tree.add_edge_connect(
        PropagatorEdge("step 1", step_1, [qt.qeye(dim), jump_op], error_accumulation=True),
        nodes[0], nodes[1])
    tree.add_edge_connect(
        PropagatorEdge("step 2", qt.to_super(rotation_both), [qt.qeye(dim)]), nodes[1], nodes[2])
    tree.traverse(2)
    return tree


def test_process_fidelity_twirled_by_step_of_logical_rotations():
    """E: rotations about X give closed forms for the tree with every map, with none, and with one map Pauli-twirled, on the whole tree and on each branch."""
    jump, theta = 0.2, 0.3
    logical = _branching_rotation_tree(jump, theta).logical_evolution_tree()
    # no-jump branch: (1 - jump) id, then R; jump branch: jump R, then R
    c_1, c_2 = np.cos(theta / 2) ** 2, np.cos(theta) ** 2
    # the Pauli twirl of R is the Pauli map with the probabilities cos^2 and sin^2 of theta/2
    p, q = np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2
    untwirled = [1.0, (1 - jump) + jump * c_1, (1 - jump) * c_1 + jump * c_2]
    twirled = [1.0, (1 - jump) + jump * p, (1 - jump) * p + jump * (p ** 2 + q ** 2)]
    assert logical.process_fidelity_twirled_by_step((1, 2)) == pytest.approx(untwirled, abs=1e-12)
    assert logical.process_fidelity_by_step() == pytest.approx(untwirled, abs=1e-12)
    assert logical.process_fidelity_twirled_by_step() == pytest.approx(twirled, abs=1e-12)
    # one rotation composed with Pauli maps: the off-diagonal entries do not contribute
    assert logical.process_fidelity_twirled_by_step((2,)) == pytest.approx(twirled, abs=1e-12)
    # each branch alone; the root is the logical qubit, so it sums the whole tree
    jump_child = next(edge.final_state for edge in logical.nodes[0].out_edges
                      if edge.name.endswith("error space 0 to 1"))
    assert logical.process_fidelity_twirled_by_step((1, 2), through_node=jump_child) \
        == pytest.approx([0.0, jump * c_1, jump * c_2], abs=1e-12)
    assert logical.process_fidelity_twirled_by_step(through_node=jump_child) \
        == pytest.approx([0.0, jump * p, jump * (p ** 2 + q ** 2)], abs=1e-12)
    assert logical.process_fidelity_twirled_by_step(through_node=logical.nodes[0]) \
        == pytest.approx(twirled, abs=1e-12)


# solver arguments of the diamond-norm methods of an edge ----------------------
def test_edge_process_dnorm_agrees_between_solvers():
    """E: the default solver of qutip and the solver SCS give the same diamond norm against the scaled identity map."""
    pytest.importorskip("cvxpy")
    loss = 0.1
    edge = _small_tree(loss).edges[0]
    assert edge.process_dnorm(solver="SCS")[0, 0] == pytest.approx(
        edge.process_dnorm()[0, 0], abs=1e-4)


def test_process_block_dnorm_of_a_completely_positive_map():
    """E: the projected dynamical map within the joint error space has diamond norm 1, by the default, with a named solver, and with ``cp_only``."""
    pytest.importorskip("cvxpy")
    tree = _small_tree(0.1)
    edge = tree.edges[0]
    # completely positive, not trace preserving: ||E^dag(1)|| = 1 (level 0 is untouched)
    assert edge.process_block_dnorm(("L", "L"), ("L", "L")) == pytest.approx(1.0, abs=1e-12)
    assert edge.process_block_dnorm(("L", "L"), ("L", "L"), solver="SCS") \
        == pytest.approx(1.0, abs=2e-12)
    assert edge.process_block_dnorm(("L", "L"), ("L", "L"), cp_only=True, cp_atol=1e-9) \
        == pytest.approx(1.0, abs=2e-12)


# regression tests of corrected defects -----------------------------------------
def _row_stacking_choi(kraus: qt.Qobj) -> np.ndarray:
    r"""Choi matrix of the map :math:`\rho\mapsto K\rho K^\dagger` in the row-stacking convention, :math:`|K\rangle\rangle\langle\langle K|` with :math:`|K\rangle\rangle` the rows of :math:`K` one after another."""
    vec = kraus.full().reshape(-1)
    return np.outer(vec, vec.conj())


def test_choi_multiply_returns_the_convention_of_its_arguments():
    """Row-stacking Choi matrices of two maps give the row-stacking Choi matrix of the composed map; the same for column stacking."""
    rng = np.random.default_rng(11)
    k_1, k_2 = (qt.Qobj(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) for _ in range(2))
    # column stacking: the matrices of qutip
    col = utils.choi_multiply(qt.to_choi(qt.to_super(k_1)), qt.to_choi(qt.to_super(k_2)), convension='col')
    assert np.allclose(col.full(), qt.to_choi(qt.to_super(k_1 * k_2)).full(), atol=1e-12)
    # row stacking: written out from the Kraus operators, without qutip
    row_1 = utils.choi_conv_change(qt.to_choi(qt.to_super(k_1)))
    row_2 = utils.choi_conv_change(qt.to_choi(qt.to_super(k_2)))
    assert np.allclose(row_1.full(), _row_stacking_choi(k_1), atol=1e-12)
    row = utils.choi_multiply(row_1, row_2, convension='row')
    assert np.allclose(row.full(), _row_stacking_choi(k_1 * k_2), atol=1e-12)
    assert np.allclose(utils.choi_multiply(row_1, row_2).full(), row.full())


def test_clear_evolution_data_deletes_state_ideal_states_and_record():
    """After clearing, a node holds neither its state, nor its ideal logical states, nor its measurement record, nor its encoded process; an excluded node keeps them."""
    tree = _small_tree()
    tree.clear_evolution_data(exclude=[tree.nodes[0]])
    for node in tree.nodes[1:]:
        for name in (
            "state", "_raw_ideal_logical_states", "meas_record",
            "encoded_process",
        ):
            assert name not in vars(node)
    assert "encoded_process" in vars(tree.nodes[0])
    assert tree.nodes[0].probability == pytest.approx(1.0)
    assert tree.nodes[0].meas_record == []
    # a node that holds only a part of the data is cleared without an error
    partial = StateNode()
    partial.meas_record = [0]
    partial.clear_evolution_data()
    assert "meas_record" not in vars(partial)
    # the tree can be evolved again from the excluded initial node
    tree.traverse(2)
    assert tree.nodes[2].meas_record == [0]


def test_node_deepcopy_keeps_amplitudes_and_termination():
    """The copy of a node gives the fidelity and the ideal states of the node, and a copy of a terminated node is terminated."""
    loss = 0.1
    tree = _small_tree(loss)
    node, terminated = tree.nodes[2], tree.nodes[3]
    copied = node.deepcopy()
    assert copied.out_edges == [] and copied.in_edge is None
    assert copied.state is not node.state
    assert copied.probability == pytest.approx(1 - loss / 2, abs=1e-12)
    # state (|0> + sqrt(1-loss)|1>)/sqrt(2) against the ideal state (|0> + |1>)/sqrt(2)
    assert copied.fidelity == pytest.approx((1 + np.sqrt(1 - loss)) ** 2 / 4, abs=1e-12)
    assert copied.prob_amp_01 == pytest.approx((1 / np.sqrt(2), 1 / np.sqrt(2)))
    assert not copied.terminated
    assert terminated.deepcopy().terminated
    # the copy gives the effective logical map of the node
    for copied_map, node_map in zip(
        copied.effective_logical_process().ravel(),
        node.effective_logical_process().ravel(),
    ):
        assert np.allclose(copied_map.full(), node_map.full(), atol=1e-14)


def _leak_and_return_tree(theta: float) -> EvolutionTree:
    r"""A traversed two-step tree whose second step undoes the first: a rotation by ``theta`` between level 1 and level 2, which is outside the code space, and the inverse rotation. The ideal operator of each step is the identity."""
    generator = qt.basis(DIM, 2) * qt.basis(DIM, 1).dag()
    rotation = (theta * (generator - generator.dag())).expm()
    tree = EvolutionTree()
    nodes = [StateNode.initial_note(
        (1 / np.sqrt(2), 1 / np.sqrt(2)), qt.basis(DIM, 0), qt.basis(DIM, 1),
    )] + [StateNode() for _ in range(2)]
    for node in nodes:
        tree.add_node(node)
    tree.add_edge_connect(PropagatorEdge("out", qt.to_super(rotation), [qt.qeye(DIM)]), nodes[0], nodes[1])
    tree.add_edge_connect(PropagatorEdge("back", qt.to_super(rotation.dag()), [qt.qeye(DIM)]), nodes[1], nodes[2])
    tree.traverse(2)
    tree.traverse_and_approx()
    return tree


def test_process_dnorm_by_sum_takes_the_leakage_of_the_maps_it_sums():
    """With ``approx_proc``, the leakage probability is that of the product of the step-wise effective logical maps, K = diag(1, cos^2 theta)."""
    pytest.importorskip("cvxpy")
    theta = 0.4
    tree = _leak_and_return_tree(theta)
    ensemble = tree.traverse(2, evolve=False)
    # each step keeps the amplitude cos(theta) of logical 1; the two steps together are the identity
    kraus = qt.Qobj(np.diag([1.0, np.cos(theta) ** 2]))
    selm_product = qt.to_super(kraus)
    assert np.allclose(ensemble.accum_logical_process("super").full(), selm_product.full(), atol=1e-12)
    p = (1 + np.cos(theta) ** 4) / 2
    expected = (selm_product - p * qt.to_super(qt.qeye(2))).dnorm() + (1 - p)
    assert ensemble.process_dnorm_by_sum(approx_proc=True) == pytest.approx(expected, abs=1e-6)
    # the effective logical map of the two steps is the identity map: no leakage and no distance
    assert ensemble.process_dnorm_by_sum() == pytest.approx(0.0, abs=1e-6)


def test_process_dnorm_by_sum_is_an_upper_bound_of_the_diamond_norm():
    """||sum_i E_i - I|| <= sum_i ||E_i - p_i I|| + p_leak, for a lossy step and for the product of two leaking steps."""
    pytest.importorskip("cvxpy")
    ensemble = _small_tree(0.1).traverse(1, evolve=False)
    assert ensemble.process_dnorm() <= ensemble.process_dnorm_by_sum() + 1e-6
    ensemble = _leak_and_return_tree(0.4).traverse(2, evolve=False)
    norm = ensemble.process_dnorm(approx_proc=True)
    bound = ensemble.process_dnorm_by_sum(approx_proc=True)
    assert norm <= bound + 1e-6


def test_edge_fidelity_by_process_default_is_tr_s_over_d_squared():
    """The default call on the edge of a lossy step returns |tr K|^2 / 4 of its Kraus operator diag(1, sqrt(1-loss))."""
    loss = 0.1
    edge = _small_tree(loss).edges[0]
    expected = (1 + np.sqrt(1 - loss)) ** 2 / 4
    assert np.asarray(edge.fidelity_by_process()).ravel()[0] == pytest.approx(expected, abs=1e-12)
