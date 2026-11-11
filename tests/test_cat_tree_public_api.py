r"""Public access to the cat-code tree builder ``QEC_graph.FullCatTreeBuilder``.

A builder gives public access to its collapse operators with their names, to
the ideal operators and the measurement maps for durations and confusion
matrices other than its own, to its switches, and to the table of steps. These
tests check this access on a small model of the builder: a resonator with a
dispersively coupled ancilla, whose pulses are replaced by fixed unitaries
(``test_gate_open_system.py`` checks the pulse integration itself).

Each quantity is compared with the formula that defines it:

- collapse operators: the names of ``cat_recipe.dressed_jump_ops``, in the
  order of ``c_ops``;
- ideal operators: the formulas of ``test_cat_ideal_operators.py`` at the
  builder's durations;
- measurement maps:
  :math:`\mathcal M_j=\sum_iC_{ij}\mathcal P_ie^{\mathcal Lt}` for a confusion
  matrix :math:`C`, with :math:`\mathcal P_i` the projection of the ancilla on
  level i, :math:`\mathcal L` the Liouvillian of the idling, and :math:`t` the
  measurement duration;
- edge names: ``"<prefix> (<cycle>.<step>)"`` with the prefixes ID, G1, PM,
  G2, MS, FL, RS in the order of the protocol;
- merged steps (``merge_steps``): the dynamical map of the merged edge is
  the composition of the maps that the seven-step tree applies one after
  another, for every measurement record, and the error states after it are
  those after the last merged step.
"""

import numpy as np
import pytest
import qutip as qt
import scqubits as scq

from qec_aeb import cat_ideal, cat_real, cat_recipe
from qec_aeb.QEC_graph import (
    FullCatTreeBuilder, GATE_NU_HALF,
)
from chencrafts.cqed.qt_helper import oprt_in_basis
from tests.test_cat_ideal_operators import (
    _formula_half_pi, _formula_parity_mapping, _formula_reset,
)

RES_DIM = 5
N_REF = 2


class _Sweep(dict):
    r"""Stand-in of a single-point ``FlexibleSweep``: a dictionary of sweep
    data with the ``hilbertspace`` attribute that the builder passes to
    ``cat_real.qubit_gate``.
    """
    hilbertspace = None


def _sweep_data(qubit_dim: int) -> _Sweep:
    r"""Sweep data of a small resonator-ancilla model (durations in ns, coefficients in rad/ns, rates in 1/ns).

    Parameters
    ----------
    qubit_dim : int
        Number of ancilla levels.

    Returns
    -------
    _Sweep
        Dispersive coefficients, durations, assignment errors, and jump rates.
    """
    # rates of the ancilla jumps |i><j|: relaxation, excitation, dephasing,
    # and leakage from |1> to |2>
    jump_ij = np.zeros((qubit_dim, qubit_dim))
    jump_ij[0, 1], jump_ij[1, 0], jump_ij[1, 1] = 4e-5, 2e-6, 7e-5
    if qubit_dim > 2:
        jump_ij[2, 1] = 1e-6
    jump_a_ij = np.zeros((qubit_dim, qubit_dim))
    jump_a_ij[1, 1] = 3e-6
    return _Sweep(
        chi_sa=-1.1e-2, chi_prime_sa=4.0e-5, K_s=-2.0e-5, disp=np.sqrt(N_REF),
        tau_p_eff=12.0, T_W=2000.0, tau_m=400.0, tau_FD=300.0,
        M_ge=0.01, M_eg=0.03,
        jump_a=1e-6, jump_adag=1e-8, jump_adag_a=0.0,
        jump_ij=jump_ij, jump_a_ij=jump_a_ij,
        jump_adag_ij=np.zeros((qubit_dim, qubit_dim)),
    )


def _model_builder(qubit_dim: int) -> FullCatTreeBuilder:
    r"""A builder with the Hamiltonian and the collapse operators of a small dispersive model.

    The instance is created without ``__init__``, which needs a diagonalized
    circuit. It receives the attributes that ``_generate_cat_ingredients``
    would store: a static Hamiltonian
    :math:`\chi(\hat a^\dagger\hat a-n_{\mathrm{ref}})|1\rangle\langle1|` in the
    current frame, a frame Hamiltonian, and the collapse operators with
    their names.

    Parameters
    ----------
    qubit_dim : int
        Number of ancilla levels.

    Returns
    -------
    FullCatTreeBuilder
        Builder whose step maps are not built yet, with closed-system pulses
        (``gate_qubit_decoherence`` is False).
    """
    builder = object.__new__(FullCatTreeBuilder)
    builder.fsweep = _sweep_data(qubit_dim)
    builder.sim_para = dict(res_dim=RES_DIM, qubit_dim=qubit_dim, higher_order_decay=False)
    builder.new_recipe = True
    builder.res_dim, builder.qubit_dim = RES_DIM, qubit_dim
    builder.res_me_dim, builder.qubit_me_dim = RES_DIM, qubit_dim
    builder._res_mode_idx, builder._qubit_mode_idx = 0, 1
    builder._gate_axis = "x"
    builder.n_bar, builder.n_bar_ref = float(N_REF), N_REF
    builder.gate_qubit_decoherence = False
    # the reset is a pulse of finite duration (the class default makes the
    # reset ideal, with zero duration)
    builder.qubit_reset_is_ideal = False

    # Hamiltonians, diagonal in the product basis
    number = cat_ideal.res_number(RES_DIM, qubit_dim, 0)
    excited = cat_ideal.qubit_proj(RES_DIM, qubit_dim, 0, qubit_state=1)
    builder.static_hamiltonian = builder.fsweep["chi_sa"] * (number - N_REF) * excited
    builder.frame_hamiltonian = 2 * np.pi * (8.3 * number + 6.0 * excited)

    # collapse operators with their names, as cat_recipe.dressed_jump_ops
    # returns them
    jumps = cat_recipe.dressed_jump_ops(
        builder.fsweep, 0, RES_DIM, qubit_dim, qubit_dim, higher_order_decay=False)
    builder.c_ops, builder.c_op_names = list(jumps.values()), list(jumps.keys())
    # dressed states of the model: the product states themselves
    builder.partial_esys = (None, [
        cat_ideal.res_qubit_basis(RES_DIM, qubit_dim, (n, q))
        for n in range(RES_DIM) for q in range(qubit_dim)])
    builder.full_esys, builder.full_dressed_indices = None, None
    return builder


def _fixed_pulses(monkeypatch, qubit_dim: int) -> dict:
    r"""Replace the pulse integration by fixed lab-frame unitaries.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture that restores ``cat_real.qubit_gate`` after the test.
    qubit_dim : int
        Number of ancilla levels.

    Returns
    -------
    dict
        Rotation angle to the unitary that the replaced function returns.
    """
    rng = np.random.default_rng(7)
    unitaries = {}
    for angle in (np.pi / 2, -np.pi / 2, np.pi):
        generator = rng.normal(size=(RES_DIM * qubit_dim,) * 2)
        unitary = (-1j * qt.Qobj(generator + generator.T)).expm()
        unitary.dims = [[RES_DIM, qubit_dim]] * 2
        unitaries[angle] = unitary
    monkeypatch.setattr(
        cat_real, "qubit_gate",
        lambda *args, rotation_angle, **kwargs: unitaries[rotation_angle])
    return unitaries


# ---------------------------------------------------------------- step table --
@pytest.mark.unit
def test_step_table_of_the_cat_cycle():
    """The seven steps of the cycle and the prefixes of their edge names."""
    assert FullCatTreeBuilder.STEP_METHOD_NAMES == (
        "idle", "qubit_gate_1", "parity_mapping", "qubit_gate_2",
        "qubit_measurement", "feedback_latency", "qubit_reset")
    assert FullCatTreeBuilder.STEP_EDGE_PREFIXES == ("ID", "G1", "PM", "G2", "MS", "FL", "RS")
    assert FullCatTreeBuilder.steps_per_cycle == 7


@pytest.mark.unit
@pytest.mark.parametrize("with_check_point", [False, True])
def test_edge_names_follow_the_step_table(monkeypatch, with_check_point):
    """generate_tree names the edges '<prefix> (<cycle>.<step>)' in the order of the protocol."""
    qubit_dim = 3
    builder = _model_builder(qubit_dim)
    _fixed_pulses(monkeypatch, qubit_dim)
    builder.build_all_processes(num_cpus=1)
    logical_0, logical_1 = cat_ideal.cat_codewords(RES_DIM, qubit_dim, builder.fsweep["disp"])
    graph = builder.generate_tree(
        np.array([0, 1], dtype=complex), logical_0, logical_1,
        QEC_rounds=2, with_check_point=with_check_point)

    # the expected names, written out independently of the step table
    prefixes = ["ID", "G1", "PM", "G2", "MS", "FL", "RS"]
    if with_check_point:
        prefixes = [name for prefix in prefixes for name in (prefix, "CP")]
    expected = [f"{prefix} ({cycle}.{cycle * len(prefixes) + position})"
                for cycle in range(2) for position, prefix in enumerate(prefixes)]
    # every step adds one edge per active node; keep the order of first appearance
    names = list(dict.fromkeys(edge.name for edge in graph.edges))
    assert names == expected


# ------------------------------------------------------------- merged steps --
def _two_cycle_tree(builder: FullCatTreeBuilder, qubit_dim: int):
    r"""Traversed tree of two QEC cycles of a builder whose maps are built.

    Parameters
    ----------
    builder : FullCatTreeBuilder
        Builder after ``build_all_processes``.
    qubit_dim : int
        Number of ancilla levels.

    Returns
    -------
    EvolutionTree
        The tree, evolved through every step of the two cycles.
    """
    logical_0, logical_1 = cat_ideal.cat_codewords(RES_DIM, qubit_dim, builder.fsweep["disp"])
    graph = builder.generate_tree(
        np.array([0, 1], dtype=complex), logical_0, logical_1, QEC_rounds=2)
    graph.traverse(2 * builder.steps_per_cycle)
    return graph


@pytest.mark.unit
def test_merged_steps_compose_maps_and_ideal_operators(monkeypatch):
    r"""Steps 2 to 4 merged into one: the step table of the builder, the edge
    names, the composed dynamical map (which depends on the measurement record
    through gate 2), the error states after the merged step, and the final
    process fidelity agree with the seven-step tree."""
    qubit_dim = 3
    _fixed_pulses(monkeypatch, qubit_dim)
    seven = _model_builder(qubit_dim)
    seven.build_all_processes(num_cpus=1)
    five = _model_builder(qubit_dim)
    merged_name = five.merge_steps(["qubit_gate_1", "parity_mapping", "qubit_gate_2"])
    five.build_all_processes(num_cpus=1)

    # the instance has five steps; the class keeps its seven
    assert merged_name == "qubit_gate_1+parity_mapping+qubit_gate_2"
    assert five.STEP_EDGE_PREFIXES == ("ID", "G1+PM+G2", "MS", "FL", "RS")
    assert five.steps_per_cycle == 5
    assert FullCatTreeBuilder.steps_per_cycle == 7

    graph_7, graph_5 = _two_cycle_tree(seven, qubit_dim), _two_cycle_tree(five, qubit_dim)
    names = list(dict.fromkeys(edge.name for edge in graph_5.edges))
    assert names[:2] == ["ID (0.0)", "G1+PM+G2 (0.1)"] and names[6] == "G1+PM+G2 (1.6)"

    # the seven-step tree applies G1, PM, G2 one after another; their
    # composed maps, by the measurement record before G1
    expected = {}
    for edge in graph_7.edges:
        if not edge.name.startswith("G1"):
            continue
        composed, current = edge.physical_map(), edge
        for _ in range(2):
            current = current.final_state.out_edges[0]
            composed = current.physical_map() * composed
        expected[tuple(edge.init_state.meas_record)] = composed
    # records [], [0], [1]: gate 2 differs between the last two
    assert len(expected) == 3
    # every merged edge holds the composition of the three maps for its record
    merged_edges = [edge for edge in graph_5.edges if edge.name.startswith("G1+PM+G2")]
    assert len(merged_edges) == 3
    for edge in merged_edges:
        assert np.allclose(edge.physical_map().full(),
                           expected[tuple(edge.init_state.meas_record)].full(), atol=1e-12)

    # the error states after the merged step are those after step 4
    node_5, node_7 = graph_5.nodes[0], graph_7.nodes[0]
    for _ in range(2):
        node_5 = node_5.out_edges[0].final_state
    for _ in range(4):
        node_7 = node_7.out_edges[0].final_state
    for encoder_5, encoder_7 in zip(node_5.ideal_encoders(), node_7.ideal_encoders(), strict=True):
        assert np.allclose(encoder_5.full(), encoder_7.full(), atol=1e-10)

    # same dynamics and same final error spaces: the same process fidelity
    assert np.isclose(graph_5.process_fidelity_by_step()[-1],
                      graph_7.process_fidelity_by_step()[-1], atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("names", [
    ["qubit_gate_1"],
    ["qubit_gate_1", "qubit_gate_2"],
    ["parity_mapping", "qubit_gate_1"],
])
def test_merge_steps_refuses_runs_that_are_not_consecutive(names):
    """A merged step needs two or more consecutive steps, in the order of the protocol."""
    with pytest.raises(ValueError):
        _model_builder(3).merge_steps(names)


@pytest.mark.unit
def test_merged_step_refuses_a_measurement(monkeypatch):
    """A step with a measurement branches and cannot be part of a merged step."""
    qubit_dim = 3
    _fixed_pulses(monkeypatch, qubit_dim)
    builder = _model_builder(qubit_dim)
    builder.merge_steps(["qubit_gate_2", "qubit_measurement"])
    builder.build_all_processes(num_cpus=1)
    logical_0, logical_1 = cat_ideal.cat_codewords(RES_DIM, qubit_dim, builder.fsweep["disp"])
    with pytest.raises(ValueError):
        builder.generate_tree(np.array([0, 1], dtype=complex), logical_0, logical_1)


# ------------------------------------------------------------------ switches --
@pytest.mark.unit
def test_switches_are_read_from_sim_para(monkeypatch):
    """A switch that sim_para holds is set on the instance; a switch that sim_para does not hold keeps its class default."""
    # the construction of the ingredients needs a diagonalized circuit; skip it
    monkeypatch.setattr(FullCatTreeBuilder, "_find_sim_param", lambda self: None)
    monkeypatch.setattr(FullCatTreeBuilder, "_generate_cat_ingredients", lambda self: None)

    sim_para = dict(
        res_dim=RES_DIM, qubit_dim=3, gate_1_is_ideal=False, qubit_reset_is_ideal=False,
        use_frame_update=True, drive_operator="n_operator", gate_simulated_qubit_dim=4,
        gate_qubit_decoherence=False, not_a_switch=1,
    )
    builder = FullCatTreeBuilder(fsweep=None, sim_para=sim_para, new_recipe=True)
    for name in ("gate_1_is_ideal", "qubit_reset_is_ideal", "use_frame_update",
                 "drive_operator", "gate_simulated_qubit_dim", "gate_qubit_decoherence"):
        assert name in FullCatTreeBuilder.SWITCH_NAMES
        assert getattr(builder, name) == sim_para[name]
    assert not hasattr(builder, "not_a_switch")

    # absent switches: the class defaults, which the instance does not shadow
    empty = FullCatTreeBuilder(fsweep=None, sim_para=dict(res_dim=RES_DIM, qubit_dim=3))
    for name in FullCatTreeBuilder.SWITCH_NAMES:
        assert name not in vars(empty)
        assert getattr(empty, name) == getattr(FullCatTreeBuilder, name)
    assert FullCatTreeBuilder.gate_1_is_ideal is True
    assert FullCatTreeBuilder.use_frame_update is False

    # a later assignment still overrides sim_para
    builder.gate_1_is_ideal = True
    assert builder.gate_1_is_ideal is True


# -------------------------------------------------------- collapse operators --
@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [3, 4])
def test_collapse_operators_keep_their_names(qubit_dim):
    """The dictionary pairs c_ops with the names of cat_recipe.dressed_jump_ops, in the same order."""
    builder = _model_builder(qubit_dim)
    named = builder.collapse_operators
    assert list(named) == builder.c_op_names
    assert all(named[name] is op for name, op in zip(builder.c_op_names, builder.c_ops))
    assert list(named)[:3] == ["jump_a", "jump_adag", "jump_adag_a"]

    # each name identifies its operator: photon loss, ancilla relaxation,
    # and photon loss with the ancilla in |1>
    a = cat_ideal.res_destroy(RES_DIM, qubit_dim, 0)
    relaxation = cat_ideal.qubit_proj(RES_DIM, qubit_dim, 0, 0, 1)
    assert np.allclose(named["jump_a"].full(), np.sqrt(1e-6) * a.full())
    assert np.allclose(named["jump_01"].full(), np.sqrt(4e-5) * relaxation.full())
    assert np.allclose(
        named["jump_a_11"].full(),
        np.sqrt(3e-6) * (a * cat_ideal.qubit_proj(RES_DIM, qubit_dim, 0, 1, 1)).full())
    # an operator with a vanishing rate keeps its place as a zero matrix
    assert not np.any(named["jump_adag_a"].full())


@pytest.mark.unit
def test_collapse_operators_of_a_builder_without_stored_names():
    """A builder restored from a cache without c_op_names obtains the names from the sweep."""
    builder = _model_builder(3)
    expected = dict(builder.collapse_operators)
    del builder.c_op_names
    assert not hasattr(builder, "c_op_names") or builder.c_op_names is None
    restored = builder.collapse_operators
    assert list(restored) == list(expected)
    assert all(restored[name] is expected[name] for name in expected)

    # a number of names that differs from the number of operators raises an
    # error
    builder.c_ops = builder.c_ops[:-1]
    with pytest.raises(ValueError, match="collapse operators"):
        builder.collapse_operators

    # with new_recipe False, the collapse operators have no names
    builder.new_recipe = False
    with pytest.raises(ValueError, match="new_recipe"):
        builder.collapse_operators


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [3, 4])
def test_gate_collapse_operators_are_named_with_or_without_decoherence(qubit_dim):
    """The collapse operators of the pulses have names for either value of gate_qubit_decoherence; the list that the pulses use is None if the switch is False."""
    builder = _model_builder(qubit_dim)
    named = builder.gate_collapse_operators
    # only the ancilla jumps with a nonzero rate
    expected = ["jump_01", "jump_10", "jump_11", "jump_21"]
    assert list(named) == expected
    for name in expected:
        # the dressed states of the model are the product states
        assert np.allclose(named[name].full(), builder.collapse_operators[name].full())

    # gate_qubit_decoherence False: the pulses get no collapse operators,
    # and the named operators remain available
    assert builder.gate_qubit_decoherence is False
    assert builder._gate_collapse_operators is None


# ----------------------------------------------------------- ideal operators --
@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [3, 4])
def test_builder_ideal_operators_match_the_formulas(monkeypatch, qubit_dim):
    """The ideal operators of the builder equal the formulas of test_cat_ideal_operators at the builder's durations."""
    builder = _model_builder(qubit_dim)
    _fixed_pulses(monkeypatch, qubit_dim)
    builder.build_qubit_gate_process(num_cpus=1)
    builder.build_qubit_reset_process(num_cpus=1)
    builder.build_parity_mapping_process()
    ideal = builder.ideal_operators
    assert set(ideal) == {"qubit_gate_p", "qubit_gate_m", "qubit_reset", "parity_mapping"}

    sweep = builder.fsweep
    chi_1 = sweep["chi_sa"] + (2 * builder.n_bar - 1) * sweep["chi_prime_sa"]
    assert builder.average_pm_interaction == pytest.approx(chi_1, rel=1e-14)
    common = dict(chi_1=chi_1, K_s=sweep["K_s"], chi_prime=sweep["chi_prime_sa"], n_ref=N_REF)
    gate_time, reset_time = sweep["tau_p_eff"], 2 * sweep["tau_p_eff"]
    assert builder.step_durations["qubit_gate_1"] == gate_time
    assert builder.step_durations["qubit_reset"] == reset_time

    # pi/2 pulses: the formula of test_cat_ideal_operators
    for key, theta in (("qubit_gate_p", np.pi / 2), ("qubit_gate_m", -np.pi / 2)):
        formula = _formula_half_pi(
            RES_DIM, qubit_dim, theta, duration=gate_time, nu=GATE_NU_HALF, **common)
        assert len(ideal[key]) == 1
        assert np.allclose(ideal[key][0].full(), formula, atol=1e-12)
        assert np.allclose(
            builder.finite_duration_qubit_rotation(theta).full(), formula, atol=1e-12)

    # reset pulse
    assert np.allclose(
        ideal["qubit_reset"][0].full(),
        _formula_reset(RES_DIM, qubit_dim, duration=reset_time, **common),
        atol=1e-12)

    # parity-mapping idling: pi minus the angles of the two pulses
    angle = np.sign(chi_1) * np.pi - 2 * chi_1 * gate_time * GATE_NU_HALF
    assert builder.parity_mapping_angle == pytest.approx(angle, rel=1e-13)
    assert builder.parity_mapping_time == pytest.approx(angle / chi_1, rel=1e-13)
    pm_kwargs = dict(duration=builder.parity_mapping_time, K_s=sweep["K_s"],
                     chi_prime=sweep["chi_prime_sa"], n_ref=N_REF)
    assert np.allclose(
        ideal["parity_mapping"][0].full(),
        _formula_parity_mapping(RES_DIM, qubit_dim, builder.parity_mapping_angle, **pm_kwargs),
        atol=1e-12)

    # another duration: the builder's parameters with the given duration
    assert np.allclose(
        builder.finite_duration_qubit_reset(duration=31.0).full(),
        _formula_reset(RES_DIM, qubit_dim, duration=31.0, **common), atol=1e-12)
    assert np.allclose(
        builder.finite_duration_parity_mapping(angle=chi_1 * 50.0, duration=50.0).full(),
        _formula_parity_mapping(RES_DIM, qubit_dim, chi_1 * 50.0, 50.0, sweep["K_s"],
                                sweep["chi_prime_sa"], N_REF), atol=1e-12)


@pytest.mark.unit
def test_unimproved_switches_give_the_bare_operators(monkeypatch):
    """With use_improved_gate and use_improved_parity_mapping False, the ideal operators are the ancilla rotations alone and the controlled rotation of the resonator by pi per photon."""
    qubit_dim = 3
    builder = _model_builder(qubit_dim)
    _fixed_pulses(monkeypatch, qubit_dim)
    builder.use_improved_gate = False
    builder.use_improved_parity_mapping = False
    builder.build_qubit_gate_process(num_cpus=1)
    builder.build_qubit_reset_process(num_cpus=1)
    builder.build_parity_mapping_process()
    ideal = builder.ideal_operators

    def rotation(theta):
        return cat_ideal.qubit_rot_propagator(RES_DIM, qubit_dim, 0, angle=theta, axis="x").full()

    assert np.array_equal(ideal["qubit_gate_p"][0].full(), rotation(np.pi / 2))
    assert np.array_equal(ideal["qubit_gate_m"][0].full(), rotation(-np.pi / 2))
    assert np.array_equal(ideal["qubit_reset"][0].full(), rotation(np.pi))
    assert np.allclose(
        ideal["parity_mapping"][0].full(),
        _formula_parity_mapping(RES_DIM, qubit_dim, np.pi, 0.0, 0.0, 0.0, N_REF), atol=1e-12)


# -------------------------------------------------------------------- pulses --
@pytest.mark.unit
def test_pulse_maps_lab_frame_pulses_and_the_operator_option(monkeypatch):
    """The builder keeps the lab-frame unitaries of the pulses, the reset included, and returns a closed-system pulse as an operator on request."""
    qubit_dim = 3
    builder = _model_builder(qubit_dim)
    unitaries = _fixed_pulses(monkeypatch, qubit_dim)
    # frame changes of the protocol, with the phase of the Stark shift
    # (gate_frame_phase) in the sweep
    builder.use_frame_update = True
    builder.fsweep["gate_frame_phase"] = 0.21
    assert builder.gate_frame_phase == 0.21
    assert builder.lab_frame_pulses == {}
    builder.build_qubit_gate_process(num_cpus=1)
    builder.build_qubit_reset_process(num_cpus=1)

    lab = builder.lab_frame_pulses
    assert lab["qubit_gate_p"] is unitaries[np.pi / 2]
    assert lab["qubit_gate_m"] is unitaries[-np.pi / 2]
    assert lab["qubit_reset"] is unitaries[np.pi]

    # the operator of a pulse: frame change, pulse in the current frame, frame change
    gate_time = builder.step_durations["qubit_gate_1"]
    frame = builder.ancilla_frame_change(-builder.gate_frame_phase)
    assert np.allclose(np.diag(frame.full())[:qubit_dim],
                       np.exp(1j * 0.21 * np.arange(qubit_dim)))
    operator = builder.gate_map_in_current_frame(
        lab["qubit_gate_p"], gate_time, frame_change=frame, as_operator=True)
    current = builder.frame_transform(lab["qubit_gate_p"], total_time=gate_time, type="lab2current")
    assert operator.isoper
    assert np.allclose(operator.full(), (frame * current * frame).full(), atol=1e-14)
    # its superoperator is the step map
    maps = builder.step_maps
    assert set(maps) == {"qubit_gate_p", "qubit_gate_m", "qubit_reset"}
    assert np.allclose(qt.to_super(operator).full(), maps["qubit_gate_p"].full(), atol=1e-13)
    # the reset has no frame change
    reset = builder.gate_map_in_current_frame(
        lab["qubit_reset"], builder.step_durations["qubit_reset"], as_operator=True)
    assert np.allclose(qt.to_super(reset).full(), maps["qubit_reset"].full(), atol=1e-13)
    # by default, the method returns the superoperator
    assert builder.gate_map_in_current_frame(lab["qubit_reset"], 24.0).issuper

    # a dynamical map with jumps is not an operator
    with pytest.raises(ValueError, match="as_operator"):
        builder.gate_map_in_current_frame(maps["qubit_reset"], 24.0, as_operator=True)


# --------------------------------------------------------------- measurement --
@pytest.mark.unit
def test_assignment_confusion_matrix():
    """Rows are measured states and sum to one; only the outcomes 0 and 1 are confused."""
    matrix = cat_real.assignment_confusion_matrix(4, 0.01, 0.03)
    expected = np.eye(4)
    expected[:2, :2] = [[0.99, 0.01], [0.03, 0.97]]
    assert np.array_equal(matrix, expected)
    assert np.allclose(matrix.sum(axis=1), 1)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [3, 4])
def test_measurement_maps_for_any_confusion_matrix(qubit_dim):
    """M_j = sum_i C_ij P_i exp(L t), for the confusion matrix of the sweep, for no assignment error, and for a given matrix."""
    builder = _model_builder(qubit_dim)
    assert builder.res_mode_idx == 0 and builder.qubit_mode_idx == 1
    assert builder.qubit_measurement_time == 400.0

    def reference(confusion, time):
        # the formula, from the Liouvillian and the projectors
        evolution = (qt.liouvillian(builder.static_hamiltonian, builder.c_ops) * time).expm()
        projections = [
            qt.to_super(cat_ideal.qubit_proj(RES_DIM, qubit_dim, 0, qubit_state=i))
            for i in range(qubit_dim)]
        return [sum(confusion[i, j] * projections[i] for i in range(qubit_dim)) * evolution
                for j in range(qubit_dim)]

    def assert_same(maps, expected):
        assert len(maps) == qubit_dim
        for a, b in zip(maps, expected):
            assert np.allclose(a.full(), b.full(), atol=1e-13)

    # default: the assignment errors of the sweep, as the tree uses them
    sweep_matrix = builder.measurement_confusion_matrix
    assert sweep_matrix[0, 1] == 0.01 and sweep_matrix[1, 0] == 0.03
    assert_same(builder.measurement_maps(), reference(sweep_matrix, 400.0))
    builder.build_qubit_measurement_process()
    assert_same(builder.step_maps["qubit_measurement"], reference(sweep_matrix, 400.0))
    assert len(builder.ideal_operators["qubit_measurement"]) == qubit_dim

    # no assignment error, and a matrix and a duration given by the caller
    assert_same(builder.measurement_maps(confusion_matrix=None),
                reference(np.eye(qubit_dim), 400.0))
    other = np.eye(qubit_dim)
    other[:2, :2] = [[0.8, 0.2], [0.4, 0.6]]
    assert_same(builder.measurement_maps(other, time=150.0), reference(other, 150.0))
    assert_same(builder.measurement_maps(None, time=0.0), reference(np.eye(qubit_dim), 0.0))

    # the maps of all outcomes add up to a trace-preserving map
    trace = np.eye(RES_DIM * qubit_dim).reshape(-1, order="F")
    total = sum(builder.measurement_maps(other)).full()
    assert np.allclose(trace @ total, trace, atol=1e-10)

    with pytest.raises(ValueError, match="sweep"):
        builder.measurement_maps("none")


# -------------------------------------------------- dressed subspace operators --
@pytest.mark.unit
def test_dressed_subspace_operators_of_a_small_device():
    """Energies and drive matrix elements are the projections on the dressed states."""
    resonator = scq.Oscillator(E_osc=8.3, truncated_dim=6)
    transmon = scq.Transmon(EJ=20.0, EC=0.27, ng=0.3, ncut=12, truncated_dim=4)
    space = scq.HilbertSpace([resonator, transmon])
    space.add_interaction(
        g=0.12, op1=resonator.annihilation_operator, op2=transmon.n_operator, add_hc=True)
    space.generate_lookup()
    res_dim, qubit_dim = 4, 3
    operators = cat_real.dressed_subspace_operators(
        space, 0, 1, res_truncated_dim=res_dim, qubit_truncated_dim=qubit_dim,
        dressed_indices=space["dressed_indices"][0],
        eigensys=(space["evals"][0], space["evecs"][0]),
    )
    assert operators["drive_operator"] == "n_operator"
    assert operators["energies"].shape == (res_dim, qubit_dim)
    assert operators["dressed_states"].shape == (res_dim, qubit_dim)
    assert operators["drive_matrix_elements"].shape == (res_dim, qubit_dim, qubit_dim)

    h0, h1 = operators["static_hamiltonian"], operators["drive_hamiltonian"]
    assert np.allclose(h0.full(), 2 * np.pi * space.hamiltonian().full())
    for m in range(res_dim):
        states = operators["dressed_states"][m]
        # the dressed states are eigenstates: the projection of H is diagonal
        projected = qt.Qobj(oprt_in_basis(h0, states)).full()
        assert np.allclose(projected, np.diag(operators["energies"][m]), atol=1e-8)
        block = np.array([[h1.matrix_element(bra, ket) for ket in states]
                          for bra in states])
        assert np.allclose(operators["drive_matrix_elements"][m], block, atol=1e-12)
        # the charge operator is Hermitian and couples neighbouring levels
        assert np.allclose(block, block.conj().T, atol=1e-12)
        assert abs(block[0, 1]) > 10 * abs(block[0, 2])

    # the drive matrix elements are optional, and the drive operator can be
    # chosen by name
    without = cat_real.dressed_subspace_operators(
        space, 0, 1, res_truncated_dim=res_dim, qubit_truncated_dim=qubit_dim,
        dressed_indices=space["dressed_indices"][0],
        eigensys=(space["evals"][0], space["evecs"][0]),
        drive_operator="cos_phi_operator", drive_matrix_elements=False,
    )
    assert "drive_matrix_elements" not in without
    assert without["drive_operator"] == "cos_phi_operator"


# ---------------------------------------------------- parity-mapping duration --
@pytest.mark.unit
def test_parity_mapping_time_limited_to_the_idling_duration_is_a_float():
    """A parity-mapping idling longer than T_W is limited to T_W and returned as a float, also if the sweep holds T_W as an array."""
    builder = _model_builder(3)
    unlimited = builder.parity_mapping_time
    assert type(unlimited) is float and unlimited < builder.fsweep["T_W"]
    # T_W below the duration that the angle needs, held as a zero-dimensional array
    builder.fsweep["T_W"] = np.array(0.25 * unlimited)
    with pytest.warns(UserWarning, match="parity mapping time"):
        limited = builder.parity_mapping_time
    assert type(limited) is float
    assert limited == 0.25 * unlimited
