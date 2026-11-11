r"""Entries of the two :math:`\hat X_{\pm\pi/2}` pulses: coherent error, decoherence cost, formula.

Separates the failure-Markov-chain fidelity loss of the two
:math:`\hat X_{\pm\pi/2}` pulses into coherent error and ancilla decoherence
cost, and characterizes the unitary rotation error and omitted resonator jumps
during the pulses. Writes ``pulse-entries.json``.

Usage: ``python pulse-entries.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import polar

import qec_aeb as bsq
import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import gate_model
from budget_tools import ideal_idle_states, idle_error_space_probabilities

RESULTS_JSON = Path(__file__).resolve().parent / "pulse-entries.json"

# steps of the two pulses in the cycle (the idle is step 1)
PULSE_STEPS = {"X_plus_half_pi": 2, "X_minus_half_pi": 4}
# worker processes for the construction of the maps
NUM_CPUS = 1
# switch of the tree builder: ancilla relaxation, excitation, dephasing, and
# the jump to the second excited level during the three ancilla pulses
PULSE_DECOHERENCE_SWITCH = "gate_qubit_decoherence"

# the three pulses: signed rotation angle and duration in units of the pi/2 pulse
PULSES = {
    "X_plus_half_pi": (np.pi / 2, 1),
    "X_minus_half_pi": (-np.pi / 2, 1),
    "conditional_X_pi": (np.pi, 2),
}
# photon numbers m for which the unitary error of a pulse is recorded
PHOTON_NUMBERS = range(7)
# Pauli matrices of the two lowest ancilla levels
PAULI = {"x": np.array([[0, 1], [1, 0]], dtype=complex),
         "y": np.array([[0, -1j], [1j, 0]]),
         "z": np.diag([1.0, -1.0]).astype(complex)}

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def markov_chain_losses(pulse_decoherence: bool) -> tuple[np.ndarray, dict]:
    r"""Fidelity loss of the failure Markov chain at each step of one QEC cycle.

    Parameters
    ----------
    pulse_decoherence : bool
        Whether the ancilla decoheres during the three pulses. The other
        four steps keep their decoherence in either case.

    Returns
    -------
    losses : np.ndarray
        Entry :math:`n-1` is the fidelity loss at step :math:`n`, the total
        probability of the success states after :math:`n-1` steps minus that
        after :math:`n` steps.
    ingredients : dict
        Every key of the dressed sweep, the input of the analytical formula.
    """
    sweep, simulation_parameters, _ = device_model.build_sweep()
    # the dressed sweep is the same in both cases; only the maps of the
    # pulses differ
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters | {PULSE_DECOHERENCE_SWITCH: pulse_decoherence},
        num_cpus=NUM_CPUS)
    losses = device_model.one_cycle_fidelity_losses(graph)[device_model.CHAIN]["by_step"]
    return np.asarray(losses, float), sweep.full_dict()


def pulse_device():
    r"""Dressed device, closed pulses, ideal operators, and idle model of the cycle.

    Returns
    -------
    device : dict
        Output of ``gate_model.build_device``; its ``builder`` gives the
        ideal operators and ``idle`` the idle model.
    ideal : dict
        Ideal operator of each pulse of :data:`PULSES` on the
        resonator-ancilla space.
    """
    # the dressed sweep of the trees, at the parameters of Ofek et al.
    sweep, simulation_parameters, _ = device_model.build_sweep()
    dev = gate_model.build_device(sweep, simulation_parameters)
    return dev, gate_model.ideal_pulse_operators(dev["builder"], dev["t_pulse"])


def unitary_error_of_the_pulses(dev: dict, ideal: dict) -> dict:
    r"""Axis of the unitary error of each pulse in the photon-number subspaces.

    Parameters
    ----------
    dev : dict
        Output of :func:`pulse_device`.
    ideal : dict
        Ideal operators of the pulses.

    Returns
    -------
    dict
        For each pulse a list over the photon numbers m of :data:`PHOTON_NUMBERS`:
        ``amplitude`` (:math:`\sin(r/2)` of the rotation that the error
        makes within the ancilla levels 0 and 1), ``share_about_x``,
        ``share_about_y``, ``share_about_z`` (parts of its square), and
        ``population_leaving_the_two_levels`` (largest over the two initial
        levels).
    """
    anc_dim = dev["anc_dim"]
    result = {}
    for name, (theta, factor) in PULSES.items():
        closed = gate_model.framed_model_gate(dev, theta, factor * dev["t_pulse"])
        rows = {key: [] for key in (
            "amplitude", "share_about_x", "share_about_y", "share_about_z",
            "population_leaving_the_two_levels")}
        for m in PHOTON_NUMBERS:
            block = slice(m * anc_dim, (m + 1) * anc_dim)
            error = ideal[name][block, block].conj().T @ closed[block, block]
            two_levels = error[:2, :2]
            leaving = max(1.0 - np.sum(np.abs(two_levels[:, level]) ** 2) for level in range(2))
            # unitary part of the block, with the phase removed:
            # V = cos(r/2) - i (s_x x + s_y y + s_z z)
            unitary, _ = polar(two_levels)
            unitary = unitary * np.exp(-0.5j * np.angle(np.linalg.det(unitary)))
            s = {axis: float(np.real(0.5j * np.trace(matrix @ unitary)))
                 for axis, matrix in PAULI.items()}
            amplitude = float(np.sqrt(sum(v ** 2 for v in s.values())))
            rows["amplitude"].append(amplitude)
            for axis in PAULI:
                rows[f"share_about_{axis}"].append(s[axis] ** 2 / amplitude ** 2)
            rows["population_leaving_the_two_levels"].append(float(leaving))
        result[name] = rows
    return result


def omitted_photon_loss_in_pulses(dev: dict, infidelity: float) -> dict:
    r"""First-order probability of the photon loss and gain that the simulated pulses omit.

    Parameters
    ----------
    dev : dict
        Output of :func:`pulse_device`.
    infidelity : float
        Infidelity of the failure Markov chain over the cycle.

    Returns
    -------
    dict
        For each pulse and each error space the mean photon number and the
        probability of a photon loss or gain during the pulse; the
        probability of each error space; the sum over the pulses, weighted
        with these probabilities (``total``); and ``total_over_cycle_infidelity``.
    """
    model = dev["idle"]
    rates = model.rates()
    anc_dim, res_dim = model.ancilla_dimension, model.resonator_dimension
    photons = np.kron(np.diag(np.arange(res_dim)), np.eye(anc_dim))
    # ideal states of the two error spaces after the idle: the photon number
    # does not change in the pulses and the parity-mapping idling
    states = ideal_idle_states(model)
    weights = idle_error_space_probabilities(model)
    spaces = ("even_parity", "odd_parity")
    entries, total = {}, 0.0
    for name, (_, factor) in PULSES.items():
        duration = factor * dev["t_pulse"]
        entries[name] = {}
        # the conditional pulse acts only after the outcome 1 (odd parity)
        for index in ((1,) if name == "conditional_X_pi" else (0, 1)):
            v = np.asarray(states[index])
            mean_photons = float(np.real(np.trace(v.conj().T @ photons @ v)) / v.shape[1])
            probability = (rates["photon_loss"] * mean_photons
                           + rates["photon_gain"] * (mean_photons + 1.0)) * duration
            entries[name][spaces[index]] = dict(
                mean_photon_number=mean_photons, probability=float(probability))
            total += weights[index] * probability
    return dict(
        error_space_probabilities=dict(zip(spaces, (float(w) for w in weights))),
        pulses=entries, total=float(total), total_over_cycle_infidelity=float(total / infidelity))


if __name__ == "__main__":
    full, ingredients = markov_chain_losses(pulse_decoherence=True)
    no_decoherence, _ = markov_chain_losses(pulse_decoherence=False)
    # terms of the formula, keyed by the step number
    formula_by_step = bsq.cat_code_budget("step", **ingredients)
    infidelity = float(full.sum())

    pulses = {}
    for name, step in PULSE_STEPS.items():
        entry, coherent = float(full[step - 1]), float(no_decoherence[step - 1])
        formula = float(formula_by_step[step])
        pulses[name] = dict(
            step=step,
            entry=entry,
            coherent_error=coherent,
            decoherence_cost=entry - coherent,
            formula_entry=formula,
            # the formula has the decoherence cost only
            formula_over_decoherence_cost_minus_1=formula / (entry - coherent) - 1,
            formula_over_entry_minus_1=formula / entry - 1,
            entry_over_cycle_infidelity=entry / infidelity,
            coherent_error_over_cycle_infidelity=coherent / infidelity,
        )
        print(f"{name}: entry {entry:.3e}, coherent error {coherent:.3e}, formula "
              f"{formula:.3e} ({formula / (entry - coherent) - 1:+.1%} against the "
              f"decoherence cost, {formula / entry - 1:+.1%} against the entry); entry "
              f"{entry / infidelity:.2%} of the cycle infidelity")

    device, ideal = pulse_device()
    axis = unitary_error_of_the_pulses(device, ideal)
    omitted = omitted_photon_loss_in_pulses(device, infidelity)
    print(f"photon loss and gain omitted in the pulses: {omitted['total']:.2e}, "
          f"{omitted['total_over_cycle_infidelity']:.2%} of the infidelity of the cycle")

    record = dict(
        description="Coherent error, ancilla decoherence cost, unitary error axis, and omitted resonator jump probability of the ancilla pulses at the parameters of Ofek et al.",
        settings=dict(
            pulse_decoherence_switch=PULSE_DECOHERENCE_SWITCH,
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
        ),
        cycle_infidelity_failure_Markov_chain=infidelity,
        fidelity_loss_by_step=dict(
            full=full.tolist(), no_pulse_decoherence=no_decoherence.tolist()),
        pulses=pulses,
        both_pulses=dict(
            entries_over_cycle_infidelity=sum(
                p["entry_over_cycle_infidelity"] for p in pulses.values()),
            coherent_errors_over_cycle_infidelity=sum(
                p["coherent_error_over_cycle_infidelity"] for p in pulses.values()),
        ),
        unitary_error_of_the_pulses=axis,
        omitted_photon_loss_in_pulses=omitted,
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["both_pulses"], indent=1))
    print(f"wrote {RESULTS_JSON}")
