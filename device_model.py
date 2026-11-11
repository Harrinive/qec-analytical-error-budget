r"""System model of the cat-code example: parameters, dressed sweep, actual evolution tree.

Defines the transmon--resonator parameters of Ofek et al. [Nature 536, 441
(2016)], the :math:`1/f` dephasing model, the Hilbert-space truncation rule,
and the helpers that build the dressed sweep and the actual evolution tree.
Units: circuit energies in GHz, rates in ns^-1, times in ns, temperatures in K.
"""

from __future__ import annotations

import json
import os
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import qutip
import scipy
import scqubits as scq

import chencrafts
import qec_aeb as bsq
import chencrafts.cqed as cqed

# Even/odd error states are orthonormalized symmetrically (paper convention).
# The setting must be in place before any tree is built and must never be mixed
# with Gram-Schmidt, so it is made here, once, for every script of this folder
bsq.QEC_graph.StateNode.ORTHOGONALIZE_LOGICAL_STATES = True
bsq.QEC_graph.StateNode.ORTHOGONALIZE_METHOD = "symm"

# The parameter sets that the figures mark, by the wording of the manuscript:
# the device parameters, and the Delta<0 minimum of the analytical formula
OFEK, OPTIMIZED = "Ofek et al.", "optimized"
# free parameters of the optimized set (resonator frequency, coupling,
# Josephson energy, idle duration), written by the optimization
OPTIMIZED_PARAMETERS = Path(__file__).resolve().parent / "optimized-parameters.json"
# a rate in 1/ns times this factor is the rate in 1/ms
NS_PER_MS = 1e6
# names of the three computations of one cycle in the result files
FORMULA, CHAIN, ACTUAL = "analytical_formula", "failure_Markov_chain", "actual_QEC_dynamics"

# number of steps of one QEC cycle, read from the builder so that it cannot
# drift from the tree
STEPS_PER_CYCLE = bsq.QEC_graph.FullCatTreeBuilder.steps_per_cycle

# Dimensions of the DYNAMICS: resonator states and dressed ancilla levels that
# the master equation and the step maps keep (level 2 is the level that thermal
# excitation and the pulses leak into)
DYNAMICS_RESONATOR_DIMENSION = 10
DYNAMICS_ANCILLA_DIMENSION = 3
# Dimensions of the DIAGONALIZATION: transmon eigenstates of the joint space,
# and resonator states that it keeps beyond those of the dynamics. Never tie
# these to the dimensions of the dynamics: with three transmon levels the
# qubit-state-dependent resonator non-linearity has the wrong sign
DIAGONALIZATION_ANCILLA_DIMENSION = 10
EXTRA_RESONATOR_STATES = 4

# Inputs of the pulses that follow the ancilla: the transmon is driven through
# its charge operator by Gaussian pulses with a DRAG quadrature, only level 2
# shifts level 1 (three levels in the Stark-shift sum), and the parity mapping
# applies the ancilla frame changes (virtual Z gates)
BUDGET_PARAMETERS = {
    "drive_operator": "n_operator",
    "pulse_scheme": "drag",
    "stark_level_count": 3,
    "frame_update": True,
}

# Measured ancilla dephasing rate of the Ofek device (ns^-1), T_phi = 14.4 us
KAPPA_A_PHI_OFEK = 6.95e-5
# Amplitude of the 1/f offset-charge noise in units of 2e; scqubits default
CHARGE_NOISE_AMPLITUDE = 1e-4
# the incoherent-error scale divides the quality factors of the resonator and of
# the ancilla transitions, and multiplies the ancilla dephasing rate and the
# two readout assignment errors (incoherent_error_scaled_parameters)
DIVIDED_BY_INCOHERENT_ERROR_SCALE = ["Q_s", "Q_cap", "Q_ind"]
MULTIPLIED_BY_INCOHERENT_ERROR_SCALE = ["kappa_a_phi", "M_ge", "M_eg"]
# Charge states kept on each side of zero when the bare transmon is
# diagonalized for the dephasing rates
DEPHASING_NCUT = 30


def exact_cat_parameters() -> tuple[dict, dict, dict]:
    r"""Simulation, bath, and device parameters of the reference parameters.

    Returns
    -------
    simulation_parameters : dict
        Truncations of the dynamics and the switches of the tree builder
        (``FullCatTreeBuilder.SWITCH_NAMES``): all seven steps non-ideal.
    fixed_parameters : dict
        Bath, readout, pulse, and cat parameters. ``kappa_a_phi`` is the
        ancilla dephasing rate :math:`\kappa_\phi` (the solver weight of
        :math:`\mathcal D[|1\rangle\langle1|]` is :math:`2\kappa_\phi`);
        ``kappa_s_phi`` is the weight of :math:`\mathcal D[\hat a^\dagger\hat a]`.
    device_parameters : dict
        Circuit energies (GHz), resonator frequency (GHz), resonator--ancilla
        coupling (GHz), and the measurement and idle durations (ns).
    """
    # every step uses its physical (non-ideal) map; the step maps (qubit_dim)
    # and the master equation (qubit_me_dim) keep the levels of the dynamics
    simulation_parameters = {
        "res_dim": DYNAMICS_RESONATOR_DIMENSION,
        "qubit_dim": DYNAMICS_ANCILLA_DIMENSION,
        "res_me_dim": DYNAMICS_RESONATOR_DIMENSION,
        "qubit_me_dim": DYNAMICS_ANCILLA_DIMENSION,
        # charge states of the bare transmon; chencrafts enlarges this basis
        # until the kept eigenvectors have no weight at its edge
        "qubit_ncut": 20,
        "num_cpus": 1,
        "QEC_rounds": 1,
        "with_check_point": False,
        "idling_is_ideal": False,
        "gate_1_is_ideal": False,
        "parity_mapping_is_ideal": False,
        "gate_2_is_ideal": False,
        "qubit_measurement_is_ideal": False,
        "feedback_latency_is_ideal": False,
        "qubit_reset_is_ideal": False,
        "use_improved_gate": True,
        "use_improved_parity_mapping": True,
        # ancilla decoherence during the ancilla pulses (relaxation,
        # excitation, dephasing, leakage). Photon loss during the pulses is
        # not simulated: it is never the dominant error of a pulse
        "gate_qubit_decoherence": True,
        "higher_order_decay": False,
        "drive_operator": BUDGET_PARAMETERS["drive_operator"],
        # ancilla levels kept while the pulses are integrated
        "gate_simulated_qubit_dim": DYNAMICS_ANCILLA_DIMENSION,
        "use_frame_update": BUDGET_PARAMETERS["frame_update"],
    }
    # placeholder of the dynamics space; build_cat_sweep computes every
    # eigenstate of the (larger) diagonalized space
    simulation_parameters["evals_count"] = (
        DYNAMICS_ANCILLA_DIMENSION * DYNAMICS_RESONATOR_DIMENSION)

    # baths, readout, and pulse-width rule of the Ofek device
    fixed_parameters = {
        "Q_s": 1.38e7,
        "temp_a": 0.094,
        "temp_s": 0.102,
        "Q_cap": 1.48e6,
        "Q_ind": 1e20,
        "kappa_a_phi": KAPPA_A_PHI_OFEK,
        # bare resonator dephasing: zero, because the resonator dephasing that
        # Ofek et al. report is caused by ancilla excitations, which the
        # master equation already contains
        "kappa_s_phi": 0.0,
        # readout assignment errors. The measurement step already contains the
        # ancilla transitions during tau_m; this value brings the single-shot
        # measurement fidelity to the 99.3 % that Ofek et al. measure.
        # Fitted by assignment-error-fit.py (data assignment-error-fit.json,
        # key by_resonator_fock_state.0.M_fit = 5.79e-4); the value is
        # entered here by hand and is not read from the JSON
        "M_ge": 5.8e-4,
        "M_eg": 5.8e-4,
        "sigma_2_K_a": 3 * 0.297 * np.pi * 2,
        "tau_p_by_sigma": 20,
        "tau_p_eff_by_sigma": 4,
        "sigma_omega": np.pi * 2,
        # displacement of the cat states
        "disp": np.sqrt(2),
        # feedback latency: the idle between the end of the measurement pulse
        # and the conditional pulse (Ofek et al., SI Eq. (37) and Fig. 7)
        "tau_FD": 332,
        # offset charge of the transmon
        "ng": 0.3,
    }
    device_parameters = {
        "E_osc_GHz": 8.3005,
        "tau_m": 400,
        "T_W": 18000,
        "EJ_GHz": 20.0948,
        "EC_GHz": 0.2684,
        "g_GHz": 0.1276,
    }
    return simulation_parameters, fixed_parameters, device_parameters


def dynamics_dimensions(simulation_parameters: dict, resonator_dimension: int | None = None,
                        ancilla_dimension: int | None = None) -> dict:
    r"""Simulation parameters with the dimensions of the dynamics replaced.

    The step maps, the master equation, and the integration of the pulses
    keep the same number of resonator states and of ancilla levels, so each
    dimension is replaced in every key that holds it.

    Parameters
    ----------
    simulation_parameters : dict
        Simulation parameters, e.g. the first output of
        :func:`exact_cat_parameters`. They are not changed.
    resonator_dimension : int or None
        Resonator states of the dynamics; ``None`` keeps the given ones.
    ancilla_dimension : int or None
        Dressed ancilla levels of the dynamics and of the pulses; ``None``
        keeps the given ones.

    Returns
    -------
    dict
        Copy of ``simulation_parameters`` with ``res_dim`` and ``res_me_dim``,
        or ``qubit_dim``, ``qubit_me_dim``, and ``gate_simulated_qubit_dim``,
        replaced, and ``evals_count`` set to the dimension of the space of the
        dynamics.
    """
    replaced = {}
    if resonator_dimension is not None:
        replaced |= {"res_dim": resonator_dimension, "res_me_dim": resonator_dimension}
    if ancilla_dimension is not None:
        replaced |= {"qubit_dim": ancilla_dimension, "qubit_me_dim": ancilla_dimension,
                     "gate_simulated_qubit_dim": ancilla_dimension}
    out = simulation_parameters | replaced
    # placeholder of the dynamics space; build_cat_sweep computes every
    # eigenstate of the (larger) diagonalized space
    out["evals_count"] = out["qubit_dim"] * out["res_dim"]
    return out


def bare_transmon(EJ_GHz: float, EC_GHz: float, ng: float) -> scq.Transmon:
    r"""The bare transmon of given circuit energies, outside the joint space.

    Parameters
    ----------
    EJ_GHz, EC_GHz : float
        Circuit energies (GHz).
    ng : float
        Offset charge.

    Returns
    -------
    scq.Transmon
        Transmon in the charge basis of :data:`DEPHASING_NCUT` charge states
        on each side of zero; it gives the dephasing rates and the bare
        transition frequencies.
    """
    return scq.Transmon(EJ=EJ_GHz, EC=EC_GHz, ng=ng, ncut=DEPHASING_NCUT)


def transmon_1_over_f_rates(EJ_GHz: float, EC_GHz: float, ng: float) -> tuple[float, float]:
    r"""First-order 1/f dephasing rates of a transmon per unit noise amplitude.

    ``scqubits`` gives the pure-dephasing rate of the 0--1 transition as
    :math:`|\partial\omega_{\mathrm a}/\partial\lambda|\,A_\lambda
    \sqrt{2|\ln(\omega_{\mathrm{low}}t_{\mathrm{exp}})|}` for a noisy
    parameter :math:`\lambda` with 1/f amplitude :math:`A_\lambda`; the rate
    is linear in the amplitude.

    Parameters
    ----------
    EJ_GHz, EC_GHz : float
        Circuit energies (GHz).
    ng : float
        Offset charge.

    Returns
    -------
    charge_rate, critical_current_rate : float
        Rates (ns^-1) for unit amplitude of the offset-charge noise and of the
        critical-current noise.
    """
    transmon = bare_transmon(EJ_GHz, EC_GHz, ng)
    return (
        float(transmon.tphi_1_over_f_ng(A_noise=1.0, get_rate=True)),
        float(transmon.tphi_1_over_f_cc(A_noise=1.0, get_rate=True)),
    )


@lru_cache(maxsize=None)
def fit_critical_current_noise() -> float:
    r"""Critical-current noise amplitude that reproduces the measured dephasing.

    The charge noise has the amplitude :data:`CHARGE_NOISE_AMPLITUDE`; the
    critical-current amplitude is chosen so that both rates add up to
    :data:`KAPPA_A_PHI_OFEK` at the reference parameters.

    Returns
    -------
    float
        Amplitude of the 1/f critical-current noise (in units of the critical
        current).
    """
    _, fixed_parameters, device_parameters = exact_cat_parameters()
    charge_rate, critical_current_rate = transmon_1_over_f_rates(
        device_parameters["EJ_GHz"], device_parameters["EC_GHz"], fixed_parameters["ng"])
    return (KAPPA_A_PHI_OFEK - CHARGE_NOISE_AMPLITUDE * charge_rate) / critical_current_rate


def transmon_dephasing_rate(parameters: dict) -> dict:
    r"""Ancilla dephasing rate :math:`\kappa_\phi` of a transmon, by noise source.

    Parameters
    ----------
    parameters : dict
        Fixed and device parameters merged; holds ``EJ_GHz``, ``EC_GHz``, and
        ``ng``.

    Returns
    -------
    dict
        ``charge`` (1/f offset-charge noise of amplitude
        :data:`CHARGE_NOISE_AMPLITUDE`), ``critical_current`` (1/f
        critical-current noise of amplitude :func:`fit_critical_current_noise`),
        and ``total`` (their sum, the value of ``kappa_a_phi``); rates in
        ns^-1. At the reference parameters ``total`` is the measured rate.
    """
    charge_rate, critical_current_rate = transmon_1_over_f_rates(
        parameters["EJ_GHz"], parameters["EC_GHz"], parameters["ng"])
    out = dict(charge=CHARGE_NOISE_AMPLITUDE * charge_rate,
               critical_current=fit_critical_current_noise() * critical_current_rate)
    return out | dict(total=sum(out.values()))


def with_dephasing_model(parameters: dict) -> dict:
    r"""Parameters with the ancilla dephasing rate that follows the circuit energies.

    Parameters
    ----------
    parameters : dict
        Fixed and device parameters merged; holds ``EJ_GHz``, ``EC_GHz``, and
        ``ng``.

    Returns
    -------
    dict
        Copy of ``parameters`` with ``kappa_a_phi`` replaced by the sum of the
        offset-charge and critical-current dephasing rates (ns^-1,
        :func:`transmon_dephasing_rate`). At the reference parameters this is
        the measured rate.
    """
    return parameters | {"kappa_a_phi": transmon_dephasing_rate(parameters)["total"]}


def build_cat_sweep(
    simulation_parameters: dict,
    parameters: dict,
    system_class=None,
    diagonalization_ancilla_dimension: int | None = None,
    budget_parameters: dict | None = None,
    basis_size_key: str = "qubit_ncut",
    qubit_op_names: tuple = ("n_operator", "proj_11"),
) -> cqed.FlexibleSweep:
    r"""Single-point resonator--ancilla sweep with dressed operators and rates.

    Registers, in order: static dispersive quantities, the dressed bath
    operators, the Lindblad rates of the dressed jumps, the pulse durations,
    and the scalar ingredients of the one-cycle analytical formula, from which
    the tree builder reads the frame-change angles and the cycle duration
    ``t_cycle``. The joint space is diagonalized with
    ``diagonalization_ancilla_dimension`` ancilla eigenstates and
    ``res_dim`` + :data:`EXTRA_RESONATOR_STATES` resonator states, so the
    dressed quantities do not depend on the dimensions of the dynamics.

    Parameters
    ----------
    simulation_parameters : dict
        First output of :func:`exact_cat_parameters`. The size of the basis
        of the bare ancilla that ``chencrafts`` settles on is written back
        into it, so pass a copy when several parameter sets are evaluated.
    parameters : dict
        Fixed and device parameters merged, e.g. ``fixed | device``.
    system_class : type or None
        Resonator--ancilla system of ``qec_aeb``;
        ``ResonatorTransmon`` if None.
    diagonalization_ancilla_dimension : int or None
        Ancilla eigenstates of the diagonalized joint space;
        :data:`DIAGONALIZATION_ANCILLA_DIMENSION` if None.
    budget_parameters : dict or None
        Inputs of the pulses that follow the ancilla (drive operator, pulse
        scheme, levels of the Stark-shift sum, frame changes);
        :data:`BUDGET_PARAMETERS` if None.
    basis_size_key : str
        Simulation parameter that holds the size of the basis of the bare
        ancilla (``qubit_ncut``, the charge states of a transmon).
    qubit_op_names : tuple of str
        Operators of the ancilla through which it couples to its baths.

    Returns
    -------
    cqed.FlexibleSweep
        Sweep holding every key the tree builder reads.
    """
    # the transmon ancilla unless the caller names another circuit
    system_class = bsq.ResonatorTransmon if system_class is None else system_class
    if diagonalization_ancilla_dimension is None:
        diagonalization_ancilla_dimension = DIAGONALIZATION_ANCILLA_DIMENSION
    budget_parameters = BUDGET_PARAMETERS if budget_parameters is None else budget_parameters
    # the DIAGONALIZED space of resonator + ancilla, larger than that of the
    # dynamics (truncation rule in the module docstring)
    space_res_dim = simulation_parameters["res_dim"] + EXTRA_RESONATOR_STATES
    space_parameters = simulation_parameters | {
        "res_dim": space_res_dim,
        "qubit_dim": diagonalization_ancilla_dimension,
        "evals_count": diagonalization_ancilla_dimension * space_res_dim,
    }
    joint_system = system_class(sim_para=space_parameters, para=parameters)
    # the system adapts the size of the basis of the bare ancilla; report it
    # to the caller
    simulation_parameters[basis_size_key] = space_parameters[basis_size_key]
    sweep = cqed.FlexibleSweep(
        para=parameters,
        swept_para={},
        **joint_system.dict,
        **space_parameters,
        labeling_scheme="LX",
    )
    # the ancilla and the two resonator bath operators of the error model
    operator_names = dict(
        qubit_op_names=list(qubit_op_names),
        res_op_names=["a_m_adag", "adag_a"],
    )
    truncation = dict(
        res_trunc_dim=simulation_parameters["res_me_dim"],
        qubit_trunc_dim=simulation_parameters["qubit_me_dim"],
    )
    # each registration reads keys stored by the previous ones, so order matters
    bsq.batched_sweep_general(sweep.sweep)
    bsq.batched_sweep_dressed_op(sweep.sweep, **truncation, **operator_names)
    bsq.batched_sweep_jump_rates(sweep.sweep, **truncation, **operator_names)
    bsq.batched_sweep_pulse(sweep.sweep, res_mode_idx=0, qubit_mode_idx=1)
    bsq.batched_sweep_cat_budget_ingredients(sweep.sweep, budget_para=budget_parameters)
    return sweep


def configure_tree_builder(sweep: cqed.FlexibleSweep, simulation_parameters: dict):
    r"""Tree builder of a dressed sweep, with no dynamical map built.

    A script that needs only some of the dynamical maps calls the public
    builders of the returned object and reads the maps from ``step_maps``
    and the ideal operators from ``ideal_operators``.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Output of :func:`build_sweep` or :func:`build_cat_sweep`.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built; they hold the
        truncations of the dynamics and the switches of the builder.

    Returns
    -------
    bsq.QEC_graph.FullCatTreeBuilder
    """
    return bsq.QEC_graph.FullCatTreeBuilder(
        fsweep=sweep, sim_para=simulation_parameters, new_recipe=True)


def build_tree_from_sweep(
    sweep: cqed.FlexibleSweep,
    simulation_parameters: dict,
    n_cycles: int = 1,
    num_cpus: int = 1,
    prepare_builder=None,
):
    r"""Build and traverse the actual evolution tree of a given dressed sweep.

    The seven dynamical maps of one QEC cycle are built once from the sweep;
    ``generate_tree`` wires them into ``n_cycles`` QEC cycles, branching at
    every measurement, and ``traverse`` propagates the states, which gives
    every node its ideal logical states and hence the encoders that define
    its error spaces.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Output of :func:`build_cat_sweep`.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built; they hold the
        truncations of the dynamics and the switches of the builder. They
        are not changed.
    n_cycles : int
        Number of QEC cycles in the tree. The number of nodes doubles with
        every cycle.
    num_cpus : int
        Worker processes for the construction of the maps.
    prepare_builder : callable or None
        Applied to the tree builder (:func:`configure_tree_builder`) before
        any dynamical map is built, for a script that changes the model of
        the builder, e.g. its static Hamiltonian; ``None`` changes nothing.

    Returns
    -------
    EvolutionTree
        Traversed actual evolution tree.
    """
    builder = configure_tree_builder(sweep, simulation_parameters)
    if prepare_builder is not None:
        prepare_builder(builder)
    builder.build_all_processes(num_cpus=num_cpus)

    # the computational |1> populates both codewords through the linear maps
    logical_0, logical_1 = bsq.cat_ideal.cat_codewords(
        builder.res_dim, builder.qubit_dim, float(sweep["disp"]), builder.res_mode_idx)
    # generate_tree only wires the edges; traverse propagates the states
    graph = builder.generate_tree(
        init_prob_amp_01=np.array([0, 1], dtype=complex),
        logical_0=logical_0,
        logical_1=logical_1,
        QEC_rounds=n_cycles,
        with_check_point=False,
    )
    # the builder's own number of steps: prepare_builder may have merged some
    # steps into one (FullCatTreeBuilder.merge_steps)
    graph.traverse(builder.steps_per_cycle * n_cycles, evolve=True)
    return graph


def parameter_sets() -> dict:
    r"""The parameter sets of the example, as replacements of reference parameters.

    Returns
    -------
    dict
        Label (:data:`OFEK`, :data:`OPTIMIZED`) to the parameters that replace
        their reference values: none for the parameters of Ofek et al., and
        the four free parameters of the optimization for the optimized set.
    """
    optimized = json.loads(OPTIMIZED_PARAMETERS.read_text())["free_parameters"]
    return {OFEK: {}, OPTIMIZED: optimized}


def incoherent_error_scaled_parameters(scale: float) -> dict:
    r"""Parameters of Ofek et al. with every decoherence rate and both readout assignment errors multiplied by ``scale``.

    The incoherent-error scale :math:`\xi` divides the quality factors of
    :data:`DIVIDED_BY_INCOHERENT_ERROR_SCALE` (a loss rate is the mode frequency
    divided by its quality factor; the bath temperatures, and hence the ratio
    of gain to loss, are unchanged) and multiplies the ancilla dephasing rate
    and the two readout assignment errors (:data:`MULTIPLIED_BY_INCOHERENT_ERROR_SCALE`).
    The Hamiltonian, the pulses, and the durations of the steps keep their
    values.

    Parameters
    ----------
    scale : float
        Incoherent-error scale :math:`\xi`; 1 gives the parameters of Ofek et al.

    Returns
    -------
    dict
        Values that replace the reference values, e.g. in
        :func:`build_actual_tree`. The ancilla dephasing rate is among them,
        so it is the measured rate times :math:`\xi`.
    """
    _, reference, _ = exact_cat_parameters()
    scaled = {key: reference[key] / scale for key in DIVIDED_BY_INCOHERENT_ERROR_SCALE}
    scaled.update({key: reference[key] * scale for key in MULTIPLIED_BY_INCOHERENT_ERROR_SCALE})
    return scaled


def build_sweep(parameter_overrides: dict | None = None):
    r"""Dressed sweep at the reference parameters with some of them replaced.

    Two stages: the parameters, with the ancilla dephasing rate that follows
    the Josephson energy (:func:`with_dephasing_model`; at the reference
    parameters it is the measured rate), and the dressed sweep
    (:func:`build_cat_sweep`).

    Parameters
    ----------
    parameter_overrides : dict or None
        Fixed or device parameters that replace the reference values, for
        example an entry of :func:`parameter_sets`; ``None`` keeps the
        reference values. A replaced ``kappa_a_phi`` is kept as given.

    Returns
    -------
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set; ``sweep["t_cycle"]`` is the
        duration of one QEC cycle (ns).
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.
    parameters : dict
        Fixed and device parameters used, merged.
    """
    overrides = parameter_overrides or {}
    simulation_parameters, fixed_parameters, device_parameters = exact_cat_parameters()
    reference = fixed_parameters | device_parameters
    # a misspelled name would otherwise be ignored without notice
    unknown = sorted(set(overrides) - set(reference))
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    parameters = reference | overrides
    # the dephasing rate follows the circuit unless the caller sets it
    if "kappa_a_phi" not in overrides:
        parameters = with_dephasing_model(parameters)
    sweep = build_cat_sweep(simulation_parameters, parameters)
    return sweep, simulation_parameters, parameters


def build_actual_tree(
    parameter_overrides: dict | None = None,
    n_cycles: int = 1,
    num_cpus: int = 4,
):
    r"""Actual evolution tree at the reference parameters with some of them replaced.

    The dressed sweep of the parameter set (:func:`build_sweep`), then the
    tree of that sweep (:func:`build_tree_from_sweep`).

    Parameters
    ----------
    parameter_overrides : dict or None
        See :func:`build_sweep`.
    n_cycles : int
        Number of QEC cycles in the tree.
    num_cpus : int
        Worker processes for the construction of the maps.

    Returns
    -------
    graph : EvolutionTree
        Traversed actual evolution tree.
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set; ``sweep["t_cycle"]`` is the
        duration of one QEC cycle (ns).
    parameters : dict
        Fixed and device parameters used, merged.
    """
    sweep, simulation_parameters, parameters = build_sweep(parameter_overrides)
    graph = build_tree_from_sweep(sweep, simulation_parameters, n_cycles=n_cycles,
                                  num_cpus=num_cpus)
    return graph, sweep, parameters


def failure_rate_per_ms(infidelity, duration_ns):
    r"""Failure rate :math:`(1-F)/t` in 1/ms.

    Parameters
    ----------
    infidelity : float or np.ndarray
        :math:`1-F`, the infidelity accumulated over the duration.
    duration_ns : float or np.ndarray
        The duration :math:`t` (ns): one QEC cycle for the failure rate of a
        cycle, a longer time for the average over several cycles.

    Returns
    -------
    float or np.ndarray
        The rate in 1/ms.
    """
    return NS_PER_MS * infidelity / duration_ns


def formula_infidelity(ingredients: dict) -> float:
    r"""Infidelity :math:`1-F` of one QEC cycle from the analytical formula.

    The formula is scalar arithmetic on quantities of the dressed sweep
    (rates, shifts, durations); it builds no dynamical map.

    Parameters
    ----------
    ingredients : dict
        Every key of the dressed sweep of a parameter set,
        ``sweep.full_dict()`` of the first output of :func:`build_sweep`.

    Returns
    -------
    float
        The sum of the terms of ``qec_aeb.cat_code_budget`` over
        the seven steps.
    """
    return float(bsq.cat_code_budget(**ingredients))


def one_cycle_fidelity_losses(graph) -> dict:
    r"""Fidelity loss at each step of one QEC cycle from the numerical dynamical maps, two ways.

    The logical evolution tree composes the step-wise effective logical maps
    of the traversed actual evolution tree, and the failure Markov chain
    follows from it; both are built here.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle, the output of
        :func:`build_tree_from_sweep` or the first output of
        :func:`build_actual_tree` with ``n_cycles=1``.

    Returns
    -------
    dict
        :data:`ACTUAL` and :data:`CHAIN`, each to ``after_each_step`` (list of
        eight entries, before the cycle and after each step: the process
        fidelity of the actual QEC dynamics, or the total probability of the
        success states of the failure Markov chain), ``by_step`` (list of
        seven entries: the loss of the process fidelity at each step, or the
        increase of the failure probability at each step), and
        ``infidelity`` (one minus the process fidelity, or the failure
        probability, at the end of the cycle).

    Raises
    ------
    ValueError
        If the tree does not hold exactly one QEC cycle.
    """
    # entry n of the fidelities belongs to the state after n steps; the last
    # entry is the end of the cycle only if the tree holds one cycle
    F_actual = np.asarray(graph.process_fidelity_by_step(), float)
    if len(F_actual) != STEPS_PER_CYCLE + 1:
        raise ValueError(
            f"The tree has {len(F_actual) - 1} steps; one QEC cycle has {STEPS_PER_CYCLE}.")
    logical_tree = graph.logical_evolution_tree()
    F_chain = np.asarray(logical_tree.intermediate_success_prob_by_step(), float)
    return {name: dict(after_each_step=F.tolist(), by_step=(F[:-1] - F[1:]).tolist(),
                       infidelity=float(1 - F[-1]))
            for name, F in ((ACTUAL, F_actual), (CHAIN, F_chain))}


def one_cycle_infidelities(graph) -> dict:
    r"""Infidelity of one QEC cycle from the numerical dynamical maps, two ways.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree of one QEC cycle (see
        :func:`one_cycle_fidelity_losses`).

    Returns
    -------
    dict
        :data:`ACTUAL`: one minus the process fidelity of the actual QEC
        dynamics at the end of the cycle; :data:`CHAIN`: the probability that
        the failure Markov chain has failed by then.

    Raises
    ------
    ValueError
        If the tree does not hold exactly one QEC cycle.
    """
    return {name: record["infidelity"]
            for name, record in one_cycle_fidelity_losses(graph).items()}


# Environment variables that set the thread counts of the linear algebra
# libraries
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def cpu_model() -> str:
    r"""Model name of the processor that runs the script.

    Returns
    -------
    str
        The ``model name`` line of ``/proc/cpuinfo`` on Linux, otherwise the
        name that ``platform`` reports.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment_record(worker_processes: int) -> dict:
    r"""Hardware, thread settings, and package versions of a run.

    Nothing is set here: the thread counts are read from the environment and
    recorded.

    Parameters
    -------
    worker_processes : int
        Worker processes with which the script builds the dynamical maps.

    Returns
    -------
    dict
        ``cpu_model``, ``cores_available_to_process``, ``worker_processes``;
        ``thread_settings`` (the :data:`THREAD_VARIABLES` of the
        environment); ``versions`` (python, numpy, scipy, qutip, scqubits,
        and chencrafts).
    """
    # the cores that the scheduler lets this process use, where the system
    # can tell; otherwise all cores of the machine
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count()
    return dict(
        cpu_model=cpu_model(),
        cores_available_to_process=cores,
        worker_processes=worker_processes,
        thread_settings={name: os.environ.get(name) for name in THREAD_VARIABLES},
        versions=dict(
            python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, qutip=qutip.__version__,
            scqubits=scq.__version__, chencrafts=chencrafts.__version__),
    )
