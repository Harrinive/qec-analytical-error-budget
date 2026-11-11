r"""Readout assignment error that reproduces the ancilla measurement fidelity of Ofek et al.

Integrates the measurement Lindblad master equation over the measurement
duration and solves for the symmetric readout assignment error :math:`M` that
reproduces the reported ancilla measurement fidelity :math:`F_q = 0.993`.
Writes ``assignment-error-fit.json``.

Usage: ``python assignment-error-fit.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import qec_aeb as bsq
import chencrafts.cqed as cqed
import chencrafts.settings as cc_settings

import device_model

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "assignment-error-fit.json"

# Ofek et al., Supplementary Information Sec. 1.3
OFEK_MEASUREMENT_FIDELITY = 0.993
# the fidelity is quoted to one digit after the point; half of the last digit
OFEK_ROUNDING = 0.0005
# resonator Fock states of the prepared state: vacuum (calibration, the fit)
# and the reference photon number of the cat (for the record)
RESONATOR_FOCK_STATES = [0, 2]

# progress bars would flood the log
cc_settings.PROGRESSBAR_DISABLED = True


def measurement_builder(assignment_error: float | None = None):
    r"""Tree builder of the reference device, with no step map built.

    Parameters
    ----------
    assignment_error : float or None
        Value of ``M_ge = M_eg`` that replaces the reference one; ``None``
        keeps the value of ``device_model.py``.

    Returns
    -------
    bsq.QEC_graph.FullCatTreeBuilder
        Builder whose ``measurement_maps()`` returns the dynamical maps of
        the measurement step, one superoperator per reported outcome.
    """
    # the assignment errors are fixed parameters of the device model
    overrides = (None if assignment_error is None
                 else {"M_ge": assignment_error, "M_eg": assignment_error})
    sweep, sim_para, _ = device_model.build_sweep(overrides)
    # only the measurement step is needed: its maps are made on request by
    # the builder, so no pulse is integrated and no step map is built here
    return device_model.configure_tree_builder(sweep, sim_para)


def outcome_probabilities(builder, maps: list, photon_number: int) -> np.ndarray:
    r"""Probability of each outcome of a list of maps for each prepared ancilla level.

    Parameters
    ----------
    builder : bsq.QEC_graph.FullCatTreeBuilder
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
        # product state |photon_number>|level>
        ket = bsq.cat_ideal.res_qubit_basis(
            builder.res_dim, builder.qubit_dim, (photon_number, level),
            res_mode_idx=builder.res_mode_idx,
        )
        # the probability of an outcome is the trace of its unnormalized state
        for outcome, map_ in enumerate(maps):
            probabilities[level, outcome] = np.real(cqed.superop_evolve(map_, ket).tr())
    return probabilities


def end_populations(builder, photon_number: int) -> np.ndarray:
    r"""Ancilla populations :math:`p_{ik}` at the end of ``tau_m``, before the assignment.

    Parameters
    ----------
    builder : bsq.QEC_graph.FullCatTreeBuilder
        Output of :func:`measurement_builder`.
    photon_number : int
        Fock state of the resonator in the prepared state.

    Returns
    -------
    np.ndarray
        Shape ``(2, qubit_dim)``; entry ``[i, k]`` is :math:`p_{ik}`.
    """
    # the model's own evolution during the measurement, then the projection
    # of the ancilla without a confusion matrix
    return outcome_probabilities(
        builder, builder.measurement_maps(confusion_matrix=None), photon_number)


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


def main() -> None:
    r"""Fit the assignment error, check it with the model's maps, and write the JSON.

    Returns
    -------
    None
    """
    builder = measurement_builder()
    reference = float(builder.fsweep["M_ge"])
    record = dict(
        description=("assignment error M_ge = M_eg = M for which the measurement step of the "
                     "cat-code model has the ancilla measurement fidelity "
                     "F_q = [P(g|g)+P(e|e)]/2 of Ofek et al."),
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
            # infidelity from the ancilla transitions during tau_m alone
            infidelity_without_assignment_error=float(
                1 - (populations[0, 0] + populations[1, 1]) / 2),
            M_fit=fit_assignment_error(populations, OFEK_MEASUREMENT_FIDELITY),
            # a higher measured fidelity leaves less room for assignment error
            M_range=[
                fit_assignment_error(populations, OFEK_MEASUREMENT_FIDELITY + OFEK_ROUNDING),
                fit_assignment_error(populations, OFEK_MEASUREMENT_FIDELITY - OFEK_ROUNDING),
            ],
        )
    # the fit quoted in device_model.py: resonator in vacuum
    fit = record["by_resonator_fock_state"]["0"]

    # check: the measurement maps that the tree uses, at the fitted M and at
    # the value of device_model.py
    for name, value in (("at_M_fit", fit["M_fit"]), ("at_value_in_device_model", reference)):
        checked = measurement_builder(value)
        reported = outcome_probabilities(checked, checked.measurement_maps(), 0)
        record["fidelity_of_measurement_maps_" + name] = float(
            (reported[0, 0] + reported[1, 1]) / 2)

    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"infidelity without assignment error: "
          f"{fit['infidelity_without_assignment_error']:.3e}")
    print(f"M fit: {fit['M_fit']:.3e}  (range {fit['M_range'][0]:.2e} to "
          f"{fit['M_range'][1]:.2e}); device_model.py: {reference:.2e}")
    print(f"F_q of the measurement maps at the fit: "
          f"{record['fidelity_of_measurement_maps_at_M_fit']:.6f}, at the value of "
          f"device_model.py: {record['fidelity_of_measurement_maps_at_value_in_device_model']:.6f}")
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
