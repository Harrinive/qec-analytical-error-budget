r"""Calibration and provenance of the Ofek et al. device and noise parameters.

Holds together the experimental targets reported in Ofek et al. [Nature 536, 441
(2016)] and the four calibration stages that determine the fixed and device
parameters used in ``cat_code.device_model`` and ``cat_code.fluxonium_model``:

1. **Circuit fit (Stage 1):** Bare resonator frequency ``E_osc_GHz``, coupling
   ``g_GHz``, and transmon energies ``EJ_GHz``, ``EC_GHz`` fitted to the
   measured dressed frequencies and Kerr/dispersive shifts
   (:math:`\omega_{\mathrm a}, \omega_{\mathrm s}, K_{\mathrm a}, K_{\mathrm s},
   \chi_{\mathrm{sa}}`).
2. **Bath fit (Stage 2):** Bath temperatures ``temp_a``, ``temp_s``, dielectric
   quality factors ``Q_s``, ``Q_cap``, resonator dephasing ``kappa_s_phi``, and
   reference ancilla dephasing rate :math:`\kappa_\phi` fitted to the measured
   coherence times and thermal populations (:math:`T_{1,\mathrm a},
   T_{2,\mathrm a}, T_{\phi,\mathrm a}, n_{\mathrm{th,a}}, T_{1,\mathrm s},
   T_{2,\mathrm s}, n_{\mathrm{th,s}}`) with the circuit held at Stage 1.
3. **1/f noise amplitudes (Stage 3):** Critical-current noise amplitude ``A_cc``
   (with offset-charge noise ``A_ng = 1e-4``) calibrated so the bare transmon at
   Stage 1 has :math:`\kappa_\phi = 6.95\times 10^{-5}\,\mathrm{ns}^{-1}`
   (:math:`T_\phi = 14.4\,\mu\mathrm{s}`), and flux-noise amplitude ``A_flux``
   calibrated so the half-flux reference fluxonium has the same dephasing rate.
4. **Readout assignment error (Stage 4):** Symmetric assignment error
   ``M_ge = M_eg = 5.8e-4`` calibrated so the 400 ns measurement master equation
   reproduces the reported single-shot readout fidelity :math:`F_q = 0.993`.
   Writes ``data/assignment-error-fit.json``.

Usage: ``python scripts/1_calibration/ofek-parameter-fit.py``.
"""

from __future__ import annotations

import json
import numpy as np

import chencrafts.cqed as cqed
import chencrafts.settings as cc_settings

from cat_code import device_model, fluxonium_model, ideal as cat_ideal

cc_settings.PROGRESSBAR_DISABLED = True

RESULTS_JSON = device_model.DATA_DIR / "assignment-error-fit.json"

# ---------------------------------------------------------------------------
# Experimental targets from Ofek et al., Nature 536, 441 (2016)
# ---------------------------------------------------------------------------
TWO_PI = 2 * np.pi

# Dressed frequencies (GHz) and nonlinearities / dispersive shift (rad/ns).
# Note: Ofek et al. write (K/2) a^{\dagger 2} a^2, whereas the Hamiltonian of
# Appendix F writes K a^{\dagger 2} a^2.
OFEK_SPECTROSCOPY_TARGETS = {
    "omega_a_GHz": 6.2815,
    "omega_s_GHz": 8.3056,
    "K_a_rad_ns": -297e-3 * TWO_PI,
    "K_s_rad_ns": -4.5e-6 * TWO_PI,
    "chi_sa_rad_ns": -1.97e-3 * TWO_PI,
}

# Coherence times (ns) and thermal populations
OFEK_COHERENCE_TARGETS = {
    "T1_a_ns": 35e3,
    "T2_a_ns": 12e3,
    "T_phi_a_ns": 14.5e3,
    "n_th_a": 0.04,
    "T1_s_ns": 250e3,
    "T2_s_ns": 330e3,
    "n_th_s": 0.02,
}

# Fitted ancilla dephasing rate at the Ofek operating point (ns^-1),
# corresponding to T_phi = 14.4 us from the Stage-2 bath fit
KAPPA_A_PHI_OFEK = 6.95e-5
# Default 1/f offset-charge noise amplitude of scqubits (in units of 2e)
CHARGE_NOISE_AMPLITUDE = 1e-4
# Reference fluxonium at half flux (GHz and flux quanta) used to fix A_flux
HALF_FLUX_FLUXONIUM = {
    "EJ_GHz": 4.0,
    "EC_GHz": 1.0,
    "EL_GHz": 0.5,
    "flux": 0.5,
}
# Single-shot ancilla readout fidelity (Ofek et al., SI Sec. 1.3) and its half-digit rounding
OFEK_MEASUREMENT_FIDELITY = 0.993
OFEK_ROUNDING = 0.0005
# Resonator Fock states of the prepared state: vacuum (calibration, the fit)
# and the reference photon number of the cat (for the record)
RESONATOR_FOCK_STATES = [0, 2]


# ---------------------------------------------------------------------------
# Stage 1 & Stage 2: Circuit and bath parameters of the Ofek device
# ---------------------------------------------------------------------------
FITTED_CIRCUIT_PARAMETERS = {
    "E_osc_GHz": 8.3005,
    "g_GHz": 0.1276,
    "EJ_GHz": 20.0948,
    "EC_GHz": 0.2684,
    "ng": 0.3,
}

FITTED_BATH_PARAMETERS = {
    "Q_s": 1.38e7,
    "temp_a": 0.094,
    "temp_s": 0.102,
    "Q_cap": 1.48e6,
    "Q_ind": 1e20,
    # Set to zero because the resonator dephasing reported in Ofek et al. is
    # caused by thermal ancilla excitations, which the master equation already
    # includes through ancilla excitation and the dispersive shift
    "kappa_s_phi": 0.0,
}


# ---------------------------------------------------------------------------
# Stage 3: 1/f noise amplitudes A_ng, A_cc, and A_flux
# ---------------------------------------------------------------------------
def fit_critical_current_noise(
    circuit_parameters: dict = FITTED_CIRCUIT_PARAMETERS,
    target_kappa_phi: float = KAPPA_A_PHI_OFEK,
    charge_noise_amplitude: float = CHARGE_NOISE_AMPLITUDE,
) -> float:
    r"""Critical-current 1/f noise amplitude ``A_cc`` reproducing ``target_kappa_phi``.

    At the Ofek circuit parameters, the first-order 1/f dephasing rates from
    offset-charge noise (amplitude ``charge_noise_amplitude``) and
    critical-current noise (amplitude ``A_cc``) add up to ``target_kappa_phi``.

    Parameters
    ----------
    circuit_parameters : dict
        Holds ``EJ_GHz``, ``EC_GHz``, and ``ng``.
    target_kappa_phi : float
        Target ancilla dephasing rate :math:`\kappa_\phi` (ns^-1).
    charge_noise_amplitude : float
        Amplitude ``A_ng`` of the 1/f offset-charge noise (in units of 2e).

    Returns
    -------
    float
        Amplitude ``A_cc`` of the 1/f critical-current noise (in units of the
        critical current).
    """
    charge_rate, critical_current_rate = device_model.transmon_1_over_f_rates(
        circuit_parameters["EJ_GHz"],
        circuit_parameters["EC_GHz"],
        circuit_parameters["ng"],
    )
    return float(
        (target_kappa_phi - charge_noise_amplitude * charge_rate)
        / critical_current_rate
    )


def fit_flux_noise_amplitude(
    half_flux_parameters: dict = HALF_FLUX_FLUXONIUM,
    target_kappa_phi: float = KAPPA_A_PHI_OFEK,
    critical_current_amplitude: float | None = None,
) -> float:
    r"""Flux 1/f noise amplitude ``A_flux`` at which the half-flux fluxonium has ``target_kappa_phi``.

    With the unit-amplitude flux rates :math:`r_1` (first order) and
    :math:`r_2` (second order) of ``half_flux_parameters`` and its
    critical-current rate :math:`r_{\mathrm{cc}}` at ``critical_current_amplitude``,
    ``A_flux`` solves
    :math:`r_2 A_{\mathrm{flux}}^2 + r_1 A_{\mathrm{flux}} + A_{\mathrm{cc}} r_{\mathrm{cc}} = \kappa_\phi`.

    Parameters
    ----------
    half_flux_parameters : dict
        Holds ``EJ_GHz``, ``EC_GHz``, ``EL_GHz``, and ``flux``.
    target_kappa_phi : float
        Target ancilla dephasing rate :math:`\kappa_\phi` (ns^-1).
    critical_current_amplitude : float or None
        Amplitude ``A_cc`` of the 1/f critical-current noise;
        :func:`fit_critical_current_noise` if None.

    Returns
    -------
    float
        Amplitude ``A_flux`` of the 1/f flux noise, in units of :math:`\Phi_0`.
    """
    if critical_current_amplitude is None:
        critical_current_amplitude = fit_critical_current_noise()
    rates = fluxonium_model.fluxonium_noise_rates(half_flux_parameters)
    target = target_kappa_phi - critical_current_amplitude * rates["critical_current"]
    if target <= 0:
        raise ValueError(
            "The critical-current noise alone exceeds the target dephasing rate."
        )
    first, second = rates["flux_first"], rates["flux_second"]
    return float((-first + np.sqrt(first ** 2 + 4 * second * target)) / (2 * second))


# ---------------------------------------------------------------------------
# Stage 4: Readout assignment error M_ge = M_eg
# ---------------------------------------------------------------------------
def measurement_builder(assignment_error: float | None = None):
    r"""Tree builder of the reference device, with no step map built.

    Parameters
    ----------
    assignment_error : float or None
        Value of ``M_ge = M_eg`` that replaces the reference one; ``None``
        keeps the value of ``cat_code.device_model``.

    Returns
    -------
    cat_code.tree.FullCatTreeBuilder
        Builder whose ``measurement_maps()`` returns the dynamical maps of
        the measurement step, one superoperator per reported outcome.
    """
    overrides = (
        None
        if assignment_error is None
        else {"M_ge": assignment_error, "M_eg": assignment_error}
    )
    sweep, sim_para, _ = device_model.build_sweep(overrides)
    return device_model.configure_tree_builder(sweep, sim_para)


def outcome_probabilities(builder, maps: list, photon_number: int) -> np.ndarray:
    r"""Probability of each outcome of a list of maps for each prepared ancilla level.

    Parameters
    ----------
    builder : cat_code.tree.FullCatTreeBuilder
        Supplies the dimensions of the resonator and the ancilla.
    maps : list of qutip.Qobj
        One superoperator per outcome.
    photon_number : int
        Fock state of the resonator in the prepared state.

    Returns
    -------
    np.ndarray
        Shape ``(2, len(maps))``; entry ``[i, j]`` is the probability of
        outcome j when the ancilla is prepared in level i (g or e).
    """
    probabilities = np.zeros((2, len(maps)))
    for level in range(2):
        ket = cat_ideal.res_qubit_basis(
            builder.res_dim,
            builder.qubit_dim,
            (photon_number, level),
            res_mode_idx=builder.res_mode_idx,
        )
        for outcome, map_ in enumerate(maps):
            probabilities[level, outcome] = np.real(cqed.superop_evolve(map_, ket).tr())
    return probabilities


def end_populations(builder, photon_number: int) -> np.ndarray:
    r"""Ancilla populations :math:`p_{ik}` at the end of ``tau_m``, before the assignment.

    Parameters
    ----------
    builder : cat_code.tree.FullCatTreeBuilder
        Output of :func:`measurement_builder`.
    photon_number : int
        Fock state of the resonator in the prepared state.

    Returns
    -------
    np.ndarray
        Shape ``(2, qubit_dim)``; entry ``[i, k]`` is :math:`p_{ik}`.
    """
    return outcome_probabilities(
        builder, builder.measurement_maps(confusion_matrix=None), photon_number
    )


def fit_assignment_error(populations: np.ndarray, fidelity: float) -> float:
    r"""Assignment error ``M_ge = M_eg = M`` that gives a measurement fidelity.

    :math:`F_q = [p_{gg}+p_{ee} - M\,(p_{gg}+p_{ee}-p_{ge}-p_{eg})]/2`.

    Parameters
    ----------
    populations : np.ndarray
        Output of :func:`end_populations`.
    fidelity : float
        Measurement fidelity :math:`F_q` to reproduce.

    Returns
    -------
    float
        M; negative if the ancilla transitions alone give a lower fidelity.
    """
    kept = populations[0, 0] + populations[1, 1]
    swapped = populations[0, 1] + populations[1, 0]
    return float((kept - 2 * fidelity) / (kept - swapped))


def fit_readout_assignment_error() -> dict:
    r"""Fit the symmetric readout assignment error ``M_ge = M_eg`` and build the JSON record.

    Returns
    -------
    dict
        Full calibration record written to ``data/assignment-error-fit.json``.
    """
    builder = measurement_builder()
    reference = float(builder.fsweep["M_ge"])
    record = dict(
        description=(
            "assignment error M_ge = M_eg = M for which the measurement step of the "
            "cat-code model has the ancilla measurement fidelity "
            "F_q = [P(g|g)+P(e|e)]/2 of Ofek et al."
        ),
        ofek_measurement_fidelity=OFEK_MEASUREMENT_FIDELITY,
        ofek_rounding=OFEK_ROUNDING,
        tau_m_ns=float(builder.qubit_measurement_time),
        value_in_device_model=reference,
        feedback_latency_ns=float(builder.fsweep["tau_FD"]),
        by_resonator_fock_state={},
    )
    for photon_number in RESONATOR_FOCK_STATES:
        populations = end_populations(builder, photon_number)
        record["by_resonator_fock_state"][str(photon_number)] = dict(
            end_populations=populations.tolist(),
            infidelity_without_assignment_error=float(
                1 - (populations[0, 0] + populations[1, 1]) / 2
            ),
            M_fit=fit_assignment_error(populations, OFEK_MEASUREMENT_FIDELITY),
            M_range=[
                fit_assignment_error(
                    populations, OFEK_MEASUREMENT_FIDELITY + OFEK_ROUNDING
                ),
                fit_assignment_error(
                    populations, OFEK_MEASUREMENT_FIDELITY - OFEK_ROUNDING
                ),
            ],
        )
    fit = record["by_resonator_fock_state"]["0"]
    for name, value in (
        ("at_M_fit", fit["M_fit"]),
        ("at_value_in_device_model", reference),
    ):
        checked = measurement_builder(value)
        reported = outcome_probabilities(checked, checked.measurement_maps(), 0)
        record["fidelity_of_measurement_maps_" + name] = float(
            (reported[0, 0] + reported[1, 1]) / 2
        )
    return record


def main() -> None:
    r"""Verify and print the calibrated Ofek and reference-fluxonium parameters, and write ``data/assignment-error-fit.json``."""
    _, fixed, device = device_model.exact_cat_parameters()
    _, flux_params = fluxonium_model.fluxonium_parameters()
    sweep, _, params = device_model.build_sweep()
    full = sweep.full_dict()
    jumps = np.asarray(full["jump_ij"], dtype=float)

    A_cc = fit_critical_current_noise()
    A_flux = fit_flux_noise_amplitude(critical_current_amplitude=A_cc)
    readout_record = fit_readout_assignment_error()
    RESULTS_JSON.write_text(json.dumps(readout_record, indent=1))
    fit = readout_record["by_resonator_fock_state"]["0"]

    print("=== Stage 1: Circuit parameters (GHz) ===")
    for key in ("E_osc_GHz", "g_GHz", "EJ_GHz", "EC_GHz"):
        print(f"  {key:12s} = {device[key]}")
    print(f"  dressed omega_a/2pi = {float(full['omega_a_GHz']):.4f} GHz "
          f"(Ofek: {OFEK_SPECTROSCOPY_TARGETS['omega_a_GHz']} GHz)")
    print(f"  chi_1/2pi           = {float(full['chi_1']) / TWO_PI * 1e3:.3f} MHz "
          f"(Ofek: {OFEK_SPECTROSCOPY_TARGETS['chi_sa_rad_ns'] / TWO_PI * 1e3:.3f} MHz)")
    print(f"  2*K_s/2pi           = {2 * float(full['K_s']) / TWO_PI * 1e6:.3f} kHz "
          f"(Ofek: {OFEK_SPECTROSCOPY_TARGETS['K_s_rad_ns'] / TWO_PI * 1e6:.3f} kHz)")

    print("\n=== Stage 2: Bath parameters and dressed coherence times ===")
    for key in ("Q_s", "temp_s", "Q_cap", "temp_a", "kappa_s_phi"):
        print(f"  {key:12s} = {fixed[key]}")
    print(f"  T1_a   = {1e-3 / jumps[0, 1]:.2f} us (Ofek: 35 us)")
    print(f"  T_phi  = {2e-3 / jumps[1, 1]:.2f} us (Ofek: 14.5 us)")
    print(f"  n_th_a = {jumps[1, 0] / jumps[0, 1]:.4f} (Ofek: 0.04)")
    print(f"  T1_s   = {1e-3 / float(full['jump_a']):.1f} us (Ofek: 250 us)")
    print(f"  n_th_s = {float(full['jump_adag'] / full['jump_a']):.4f} (Ofek: < 0.02)")

    print("\n=== Stage 3: 1/f noise amplitudes ===")
    print(f"  A_ng   = {CHARGE_NOISE_AMPLITUDE!r} (device_model: {fixed['A_ng']!r})")
    print(f"  A_cc   = {A_cc!r} (device_model: {fixed['A_cc']!r})")
    print(f"  A_flux = {A_flux!r} (fluxonium_model: {flux_params['A_flux']!r})")
    print(f"  kappa_a_phi at Ofek = {params['kappa_a_phi']:.6e} ns^-1 "
          f"(target: {KAPPA_A_PHI_OFEK:.6e} ns^-1)")

    print("\n=== Stage 4: Readout assignment error ===")
    print(f"  infidelity without assignment error: {fit['infidelity_without_assignment_error']:.3e}")
    print(f"  M_fit  = {fit['M_fit']:.3e} "
          f"(range {fit['M_range'][0]:.2e} to {fit['M_range'][1]:.2e}); "
          f"device_model: {readout_record['value_in_device_model']:.2e}")
    print(f"  F_q of measurement maps at fit: {readout_record['fidelity_of_measurement_maps_at_M_fit']:.6f}, "
          f"at device_model: {readout_record['fidelity_of_measurement_maps_at_value_in_device_model']:.6f}")
    print(f"  wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
