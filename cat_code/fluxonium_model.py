r"""System model of the cat-code example with a fluxonium at zero flux as the ancilla.

Defines the zero-flux fluxonium circuit parameters, the :math:`1/f` flux,
critical-current, and charge dephasing model, the inductive loss, and the
dressed-sweep builder. Units: circuit energies in GHz, rates in ns^-1, times in
ns, temperatures in K.
"""

from __future__ import annotations

import json
from pathlib import Path

import scqubits as scq

from . import systems
import chencrafts.cqed as cqed

# importing device_model sets the symmetric orthonormalization of the error
# states, which must be in place before any tree is built
from . import device_model

# Stored search outputs written by scripts/4_optimization/fluxonium-optimization.py
MINIMA = device_model.DATA_DIR / "fluxonium-minima.json"
OPTIMUM = device_model.DATA_DIR / "fluxonium-optimum.json"

# Reference fluxonium at zero flux: circuit energies (EJ, EC, EL = 2.5, 1.0,
# 0.5 GHz), the resonator frequency (6 GHz), and the coupling (0.21035 GHz)
# that gives it the dispersive shift of the transmon device
REFERENCE_FLUXONIUM = {
    "E_osc_GHz": 6.0,
    "g_GHz": 0.21035,
    "EJ_GHz": 2.5,
    "EC_GHz": 1.0,
    "EL_GHz": 0.5,
    "T_W": 18000.0,
    "flux": 0.0,
}

# Search ranges of the six free parameters of the zero-flux fluxonium
FLUXONIUM_SEARCH_RANGES = {
    "E_osc_GHz": [4.0, 10.0],
    "g_GHz": [0.05, 0.2],
    "EJ_GHz": [3.0, 10.0],
    "EC_GHz": [0.5, 2.0],
    "EL_GHz": [0.15, 1.0],
    "T_W": [2000.0, 50000.0],
}
# Oscillator states of the bare fluxonium: for the dephasing rates, and the
# starting size of the basis of the joint system, which chencrafts enlarges
# until the kept eigenvectors have no weight at its edge
FLUXONIUM_CUTOFF = 100

# Quality factor of the inductor at 0.5 GHz when inductive loss is on: the
# default of ``t1_inductive`` of ``scqubits``
INDUCTIVE_QUALITY_FACTOR = 500e6

# Dressed ancilla levels that the dynamics and the pulses keep, and fluxonium
# eigenstates of the joint space that is diagonalized. Never tie the two
DYNAMICS_ANCILLA_DIMENSION = 4
DIAGONALIZATION_ANCILLA_DIMENSION = 20

# Inputs of the pulses that follow the ancilla: the fluxonium is driven through
# its charge operator by plain Gaussian pulses (its anharmonicity suppresses
# the leakage that DRAG removes for a transmon), four levels enter the
# Stark-shift sum, and the parity mapping applies the ancilla frame changes
BUDGET_PARAMETERS = {
    "drive_operator": "n_operator",
    "pulse_scheme": "plain",
    "stark_level_count": DYNAMICS_ANCILLA_DIMENSION,
    "frame_update": True,
}


def fluxonium_noise_rates(parameters: dict) -> dict:
    r"""1/f dephasing rates of the 0--1 transition of a fluxonium at unit noise amplitude.

    The first-order rates are those of ``scqubits``, linear in the noise
    amplitude; the flux rate vanishes at zero flux and at half flux. The
    second-order flux rate is ``chencrafts.cqed.tphi_1_over_f_second_order``,
    proportional to the squared amplitude. The noise cutoffs and the
    measurement time are the defaults of ``scqubits``.

    Parameters
    ----------
    parameters : dict
        Holds ``EJ_GHz``, ``EC_GHz``, ``EL_GHz``, and ``flux``.

    Returns
    -------
    dict
        ``flux_first``, ``flux_second``, and ``critical_current``: rates
        (ns^-1) for an amplitude of one flux quantum or of one critical
        current.
    """
    fluxonium = scq.Fluxonium(
        EJ=parameters["EJ_GHz"], EC=parameters["EC_GHz"], EL=parameters["EL_GHz"],
        flux=parameters["flux"], cutoff=FLUXONIUM_CUTOFF)
    return dict(
        flux_first=float(fluxonium.tphi_1_over_f_flux(A_noise=1.0, get_rate=True)),
        flux_second=float(cqed.tphi_1_over_f_second_order(
            fluxonium, A_noise=1.0, get_rate=True)),
        critical_current=float(fluxonium.tphi_1_over_f_cc(A_noise=1.0, get_rate=True)),
    )


def fluxonium_dephasing_rate(parameters: dict) -> dict:
    r"""Assumed ancilla dephasing rate :math:`\kappa_\phi` of a fluxonium.

    Parameters
    ----------
    parameters : dict
        Holds ``EJ_GHz``, ``EC_GHz``, ``EL_GHz``, ``flux``, ``A_flux``, and
        ``A_cc``.

    Returns
    -------
    dict
        ``flux_first``, ``flux_second``, ``critical_current`` (the three
        rates, ns^-1) and ``total`` (their sum, the value of
        ``kappa_a_phi``).
    """
    rates = fluxonium_noise_rates(parameters)
    flux_amplitude = parameters["A_flux"]
    out = dict(
        flux_first=flux_amplitude * rates["flux_first"],
        flux_second=flux_amplitude ** 2 * rates["flux_second"],
        critical_current=parameters["A_cc"] * rates["critical_current"],
    )
    return out | dict(total=sum(out.values()))


def fluxonium_parameters() -> tuple[dict, dict]:
    r"""Simulation parameters and parameters of the reference fluxonium.

    Returns
    -------
    simulation_parameters : dict
        Those of the transmon (``device_model.exact_cat_parameters``) with
        :data:`DIAGONALIZATION_ANCILLA_DIMENSION` fluxonium eigenstates in the
        static diagonalization, four ancilla levels in the dynamics and in the
        pulses, and the size of the oscillator basis of the bare fluxonium.
    parameters : dict
        Bath, 1/f noise amplitudes (``A_flux``, ``A_cc``), readout, pulse,
        cat, and protocol parameters, with the circuit of
        :data:`REFERENCE_FLUXONIUM` in place of the transmon.
    """
    simulation_parameters, fixed_parameters, device_parameters = (
        device_model.exact_cat_parameters())
    # the static diagonalization keeps 20 fluxonium levels; the dynamics and
    # the pulses keep four ancilla levels
    simulation_parameters = device_model.dynamics_dimensions(
        simulation_parameters, ancilla_dimension=DYNAMICS_ANCILLA_DIMENSION) | {
        "qubit_dim_for_static": DIAGONALIZATION_ANCILLA_DIMENSION,
        "qubit_cutoff": FLUXONIUM_CUTOFF,
        "drive_operator": BUDGET_PARAMETERS["drive_operator"],
        "use_frame_update": BUDGET_PARAMETERS["frame_update"],
    }
    simulation_parameters["evals_count"] = (
        simulation_parameters["qubit_dim_for_static"]
        * simulation_parameters["res_dim_for_static"]
    )
    # the circuit of the transmon leaves: its offset charge and charge noise
    fixed_parameters.pop("ng")
    fixed_parameters.pop("A_ng")
    # 1/f flux-noise amplitude (in units of Phi_0), calibrated in
    # ofek-parameter-fit.py so the half-flux reference fluxonium has T_phi = 14.4 us
    fixed_parameters["A_flux"] = 6.4873408427291855e-6
    protocol = {name: device_parameters[name] for name in ("tau_m", "T_W")}
    return simulation_parameters, fixed_parameters | protocol | REFERENCE_FLUXONIUM


def build_sweep(parameter_overrides: dict | None = None, inductive_loss: bool = True):
    r"""Dressed sweep of a fluxonium: the reference fluxonium with some parameters replaced.

    Merges the parameters of :func:`fluxonium_parameters` with
    ``parameter_overrides`` and builds the dressed sweep of
    ``device_model.build_cat_sweep`` with the fluxonium as the ancilla: all
    circuit-dependent decoherence rates (resonator loss/gain, resonator
    dephasing, ancilla dielectric/inductive loss, and ancilla 1/f dephasing
    from :func:`fluxonium_dephasing_rate`), the joint Hamiltonian diagonalized
    with :data:`DIAGONALIZATION_ANCILLA_DIMENSION` fluxonium eigenstates and
    fourteen resonator states, and the registrations of ``chencrafts``
    (static dispersive quantities, dressed bath operators, rates of the
    dressed jumps, pulse durations, ingredients of the analytical formula).

    Parameters
    ----------
    parameter_overrides : dict or None
        Parameters that replace those of the reference fluxonium, for example
        the free parameters of a minimum of :func:`minima`; ``None`` keeps
        the reference fluxonium.
    inductive_loss : bool
        Whether the ancilla also relaxes through loss in its inductor: a bath
        coupled through the phase operator, with the frequency-dependent
        quality factor of ``t1_inductive`` of ``scqubits``
        (:data:`INDUCTIVE_QUALITY_FACTOR` at 0.5 GHz) at the bath temperature
        of the ancilla. A ``Q_ind`` among the overrides takes precedence.
        ``True`` is the model of the manuscript; ``False`` is the model of the
        search of ``fluxonium-minima.json``.

    Returns
    -------
    sweep : cqed.FlexibleSweep
        Sweep holding every key that the tree builder and the analytical
        formula read; ``sweep["t_cycle"]`` is the duration of one QEC cycle
        (ns).
    simulation_parameters : dict
        Simulation parameters with which the sweep was built; the tree of
        ``device_model.build_tree_from_sweep`` takes them.
    parameters : dict
        Parameters used, with the assumed ``kappa_a_phi``.
    """
    overrides = parameter_overrides or {}
    simulation_parameters, reference = fluxonium_parameters()
    # a misspelled name would otherwise be ignored without notice
    unknown = sorted(set(overrides) - set(reference))
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    parameters = reference | overrides
    if inductive_loss and "Q_ind" not in overrides:
        parameters["Q_ind"] = INDUCTIVE_QUALITY_FACTOR

    # the joint system diagonalized as for the transmon (truncation rule in
    # the module docstring); the two ancilla bath operators of the error
    # model and, when the inductor is lossy, the phase operator as a third
    qubit_op_names = ("n_operator", "proj_11") + (("phi_operator",) if inductive_loss else ())
    sweep = device_model.build_cat_sweep(
        simulation_parameters, parameters, system_class=systems.ResonatorFluxonium,
        diagonalization_ancilla_dimension=DIAGONALIZATION_ANCILLA_DIMENSION,
        budget_parameters=BUDGET_PARAMETERS, basis_size_key="qubit_cutoff",
        qubit_op_names=qubit_op_names, dephasing_rate=fluxonium_dephasing_rate)
    return sweep, simulation_parameters, parameters


def minima() -> dict:
    r"""The minima of the search over the fluxonium parameters, as read from the input file.

    Returns
    -------
    dict
        Content of ``fluxonium-minima.json``: ``free_parameters`` (their
        names), ``search_ranges``, ``run_count``, and ``minima``, one entry
        per distinct minimum, ordered from the smallest failure rate of the
        analytical formula in the search to the largest, each with ``run``,
        ``run_count``, and ``free_parameters`` (replacements of the
        parameters of the reference fluxonium for :func:`build_sweep`).
    """
    return json.loads(MINIMA.read_text())
