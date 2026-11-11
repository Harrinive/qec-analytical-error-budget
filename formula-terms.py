r"""Terms of the analytical formula of the cat-code protocol against the numerical computation, term by term.

Evaluates each term of the analytical formula on the even- and odd-parity error
spaces alongside single-mechanism Lindblad simulations of the same step at the
parameters of Ofek et al. Writes ``formula-terms.json`` and
``figs/formula-terms.pdf``.

Usage: ``python formula-terms.py`` (or ``--plot-only``, ``--add-estimates``, ``--table PATH``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# headless backend: the script runs without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import qutip as qt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

import qec_aeb as bsq  # noqa: E402
import chencrafts.settings as cc_settings  # noqa: E402
from chencrafts.cqed.qt_helper import orthonormalize  # noqa: E402

# importing device_model sets the symmetric orthonormalization of the error
# states before any tree builder is made
import device_model  # noqa: E402
import gate_model  # noqa: E402
import plot_style  # noqa: E402
from budget_tools import (  # noqa: E402
    STEP_KEYS, collapse_operator_indices, formula_terms, ideal_idle_states,
    idle_error_space_probabilities, idle_mechanism_budget, idle_model_from_builder,
    idle_propagator, infidelity, relative_difference)

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "formula-terms.json"
# output of the figure, without extension
FIGURE = HERE / "figs" / "formula-terms"
# the two parity error spaces, in the order of the trailing axis of chencrafts
SPACES = ("even_parity", "odd_parity")
# resonator states of the dynamics in the truncation check
LARGER_RESONATOR_DIMENSION = 14
# margins of the axes box (inch): left, bottom, right, top. The left margin is
# that of the other figures; the bottom one holds the names of the steps on two
# lines and the two rows of the legend of the mechanisms
FIGURE_MARGINS = (0.50, 0.72, 0.06, 0.06)
# height of the figure (inch); its width is the text width of the manuscript,
# and the axes box has the shape ASPECT_THIN
FIGURE_HEIGHT = (FIGURE_MARGINS[1] + FIGURE_MARGINS[3] + plot_style.ASPECT_THIN
                 * (plot_style.TEXT_WIDTH - FIGURE_MARGINS[0] - FIGURE_MARGINS[2]))
# gap between the bars of two mechanisms of one step (pt); the two bars of one
# mechanism are plot_style.BAR_GAP_PT apart, and the steps share the width
# that is left, half a share at either end of the axis
MECHANISM_GAP_PT = 2.5
# the label of a mechanism (its percentages, or "omitted"), set vertically
# over the center of its bars, or over its own bar when it holds the number
# of one bar of two: its distance above the top of the highest of its bars
# and markers (pt)
LABEL_LIFT_PT = 1.5
# the legend of the mechanisms: entries in each row (the three ancilla
# mechanisms, then the two resonator mechanisms side by side with the two
# merged ones), its distance below the axes (pt), which leaves room for the
# names of the steps on two lines, and the distance of its rows (pt)
MECHANISM_LEGEND_ROWS = (3, 4)
MECHANISM_LEGEND_DROP = 30.0
MECHANISM_LEGEND_ROW = 11.0
# the three pulses: step key -> (kernel name of chencrafts, signed rotation
# angle, duration in units of the pi/2 pulse, key of the pulse in gate_model)
PULSES = {
    "X_plus_half_pi": ("gate1", np.pi / 2, 1, "plus"),
    "X_minus_half_pi": ("gate2", -np.pi / 2, 1, "minus"),
    "conditional_X_pi": ("reset", np.pi, 2, "pi"),
}
# ancilla jump of gate_model.CHANNELS -> its key in bsq.cat_pulse_decoherence
PULSE_JUMP_KEYS = {
    "ancilla_relaxation": "relaxation", "ancilla_excitation": "excitation",
    "ancilla_dephasing": "dephasing", "ancilla_jump_to_second_excited_level": "leakage"}
# Lindblad operators of the three ancilla mechanisms, by the names of
# budget_tools.JUMP_NAMES; ancilla excitation holds the two upward jumps
ANCILLA_MECHANISMS = {
    "ancilla_dephasing": ["ancilla_dephasing"],
    "ancilla_relaxation": ["ancilla_relaxation"],
    "ancilla_excitation": ["ancilla_excitation", "ancilla_jump_to_second_excited_level"],
}
# Lindblad operators of each mechanism of the three idlings after the idle
# (parity-mapping idling, measurement, feedback latency). The photon loss
# that acts only while the ancilla is excited has no term in the formula; it
# is counted with photon loss
IDLING_MECHANISMS = ANCILLA_MECHANISMS | {
    "photon_loss": ["photon_loss", "photon_loss_with_excited_ancilla"],
    "photon_gain": ["photon_gain"],
    "resonator_dephasing": ["resonator_dephasing"],
}
# terms of the figure and of the table: step key -> terms, in the fixed order
# of the mechanisms (plot_style.TERM_COLORS)
ROWS = {
    "idle": ["ancilla_excitation", "photon_loss", "photon_gain"],
    "X_plus_half_pi": ["ancilla_relaxation", "ancilla_excitation", "ancilla_dephasing",
                       "coherent_error"],
    "parity_mapping_idling": ["ancilla_relaxation", "ancilla_excitation", "ancilla_dephasing",
                              "photon_loss", "photon_gain", "chi_1_prime"],
    "X_minus_half_pi": ["ancilla_relaxation", "ancilla_excitation", "ancilla_dephasing",
                        "coherent_error"],
    "measurement": ["ancilla_relaxation", "ancilla_excitation", "photon_loss", "photon_gain",
                    "readout_assignment_error"],
    "feedback_latency": ["ancilla_relaxation", "ancilla_excitation", "photon_loss",
                         "photon_gain"],
    "conditional_X_pi": ["ancilla_relaxation", "ancilla_excitation", "ancilla_dephasing",
                         "coherent_error"],
}
# label of each term in the table (LaTeX)
TERM_LABELS = {
    "photon_loss": "photon loss",
    "photon_gain": "photon gain",
    "ancilla_relaxation": "ancilla relaxation",
    "ancilla_excitation": "ancilla excitation",
    "coherent_error": "coherent error",
    "ancilla_dephasing": "ancilla dephasing",
    "chi_1_prime": r"coherent error ($\chi_1'$)",
    "readout_assignment_error": "readout assignment error",
}
# term -> mechanism of the figure, where the two names differ: the term of
# chi_1' is the coherent error of the parity-mapping idling
TERM_MECHANISMS = {"chi_1_prime": "coherent error"}
# a relative difference is written in the figure when it is at least this large
LABEL_FLOOR = 0.01
# label of each step in the table (LaTeX); the figure takes plot_style.STEP_LABELS
STEP_TABLE_LABELS = {
    "idle": "Error-accumulation idling", "X_plus_half_pi": r"$\hat X_{\pi/2}$ pulse",
    "parity_mapping_idling": "Parity-mapping idling",
    "X_minus_half_pi": r"$\hat X_{-\pi/2}$ pulse", "measurement": "Measurement",
    "feedback_latency": "Feedback latency",
    "conditional_X_pi": r"Conditional $\hat X_{\pi}$ pulse"}


# ------------------------------------------------------------- the model ----
def build_context(resonator_dimension: int, with_pulses: bool) -> dict:
    r"""Dressed sweep, tree builder, master equation, ideal operators, and ideal states.

    Parameters
    ----------
    resonator_dimension : int
        Resonator states of the dynamics.
    with_pulses : bool
        Whether the three pulses are built (``gate_model.build_device``).

    Returns
    -------
    dict
        ``sweep``, ``builder``, ``model`` (``budget_tools.IdleModel``),
        ``index`` (positions of the Lindblad operators), ``device`` (output
        of ``gate_model.build_device`` or None), ``ideal`` (ideal operator of
        each step after the idle; a list over the outcomes for the
        measurement), ``states`` (step key to the pair of matrices whose
        columns are the two ideal states that reach the step on the
        even-parity and on the odd-parity error space), and the durations
        ``t_pulse``, ``t_parity_mapping``, ``t_measurement``,
        ``t_feedback_latency`` (ns).
    """
    # The dressed sweep of the reference parameters, as device_model.build_sweep
    # builds it, with the requested number of resonator states in the dynamics
    simulation_parameters, fixed, device = device_model.exact_cat_parameters()
    simulation_parameters = device_model.dynamics_dimensions(
        simulation_parameters, resonator_dimension=resonator_dimension)
    sweep = device_model.build_cat_sweep(
        simulation_parameters, device_model.with_dephasing_model(fixed | device))

    if with_pulses:
        gate_device = gate_model.build_device(sweep, simulation_parameters)
        builder, model = gate_device["builder"], gate_device["idle"]
    else:
        gate_device = None
        builder = device_model.configure_tree_builder(sweep, simulation_parameters)
        model = idle_model_from_builder(sweep, builder)
    # the dynamical maps and ideal operators of the measurement and of the
    # feedback latency, as they enter the actual evolution tree
    builder.build_qubit_measurement_process()
    builder.build_feedback_latency_process()

    # ideal operators of the steps after the idle
    t_pulse = float(sweep["tau_p_eff"])
    t_parity_mapping = float(builder.parity_mapping_time)
    pulses = gate_model.ideal_pulse_operators(builder, t_pulse)
    ideal = dict(
        X_plus_half_pi=pulses["X_plus_half_pi"],
        parity_mapping_idling=builder.finite_duration_parity_mapping(
            angle=float(builder.average_pm_interaction) * t_parity_mapping,
            duration=t_parity_mapping).full(),
        X_minus_half_pi=pulses["X_minus_half_pi"],
        measurement=[op.full() for op in builder.ideal_operators["qubit_measurement"]],
        feedback_latency=builder.ideal_operators["feedback_latency"][0].full(),
        conditional_X_pi=pulses["conditional_X_pi"],
    )

    # Ideal states that reach each step: the ideal operators of the earlier
    # steps applied to the ideal states after the idle. The no-jump evolution
    # of the measurement and of the feedback latency is not unitary, so its
    # final states are orthonormalized again
    states = {"X_plus_half_pi": ideal_idle_states(model)}
    states["parity_mapping_idling"] = [
        ideal["X_plus_half_pi"] @ v for v in states["X_plus_half_pi"]]
    states["X_minus_half_pi"] = [
        ideal["parity_mapping_idling"] @ v for v in states["parity_mapping_idling"]]
    states["measurement"] = [
        ideal["X_minus_half_pi"] @ v for v in states["X_minus_half_pi"]]
    # the even-parity error space reports the outcome 0, the odd-parity one 1
    states["feedback_latency"] = [
        orthonormalize(ideal["measurement"][outcome] @ v)
        for outcome, v in enumerate(states["measurement"])]
    states["conditional_X_pi"] = [
        orthonormalize(ideal["feedback_latency"] @ v) for v in states["feedback_latency"]]

    return dict(
        sweep=sweep, builder=builder, model=model,
        index=collapse_operator_indices(model), device=gate_device,
        ideal=ideal, states=states, t_pulse=t_pulse, t_parity_mapping=t_parity_mapping,
        t_measurement=float(builder.qubit_measurement_time),
        t_feedback_latency=float(sweep["tau_FD"]))


# ------------------------------------------------- terms of the formula -----
def formula_by_error_space(ingredients: dict) -> dict:
    r"""Every term of the analytical formula on each error space.

    The values are those of the kernels of ``chencrafts`` that
    ``cat_code_budget`` averages over the two error spaces, evaluated on the
    ingredients of the dressed sweep.

    Parameters
    ----------
    ingredients : dict
        ``sweep.full_dict()`` of the dressed sweep.

    Returns
    -------
    dict
        Step key -> term -> value for the idle (which starts in the logical
        subspace), and step key -> term -> ``[even, odd]`` for the other
        steps. The conditional X_pi pulse acts on the odd-parity error space
        only; its even-parity value is zero.
    """
    kw = ingredients
    budget = bsq.cat_code_budget

    def moment(key: str, space: int) -> float:
        r"""Photon-number moment of an error space after the idle.

        Parameters
        ----------
        key : str
            Ingredient of the sweep (``cat_mean_n``, ``cat_var_n``, ...).
        space : int
            0 for the even-parity error space, 1 for the odd-parity one.

        Returns
        -------
        float
        """
        return float(np.asarray(kw[key])[..., space])

    out = {}
    # step 1: the idle terms do not depend on an error space
    out["idle"] = {
        "two_or_more_photon_losses": float(budget["idle_multiple_photon_loss"](**kw)),
        "kerr_term": float(budget["idle_kerr_after_loss"](**kw)),
        "photon_gain": float(budget["idle_photon_gain"](**kw)),
        "ancilla_excitation": float(budget["idle_ancilla_excitation"](**kw)),
        "resonator_dephasing": float(budget["idle_storage_dephasing"](**kw)),
    }
    out["idle"]["photon_loss"] = (
        out["idle"]["two_or_more_photon_losses"] + out["idle"]["kerr_term"])

    # steps 2, 4, 7: coherent error and decoherence cost of the pulses
    t_pulse = float(kw["tau_p_eff"])
    for step_key, (kernel, theta, duration_factor, _) in PULSES.items():
        half = abs(theta) < np.pi
        shape = "half" if half else "pi"
        duration = duration_factor * t_pulse
        # the ideal operator of a pi/2 pulse keeps the first-order phase of
        # the dispersive shift; that of the pi pulse keeps none
        phi_s, phi_chi = bsq.cat_pulse_tilt_angles(
            abs(theta), duration, kw["chi_1"], kw["anc_stark_coeff"],
            kw[f"gate_mu_{shape}"], kw[f"gate_nu_{shape}"],
            retained_shape_nu=kw["gate_nu_half"] if half else 0.0)
        coherent = [bsq.cat_pulse_coherent_error(
            shape, phi_s, phi_chi, mean_d=moment("cat_mean_d", s),
            mean_d2=moment("cat_mean_d2", s), var_n=moment("cat_var_n", s)) for s in (0, 1)]
        # the frame changes of the protocol remove the Stark-shift term of a
        # pi/2 pulse
        stark_factor = 1 - float(kw["frame_update"]) if half else 1.0
        costs = bsq.cat_pulse_decoherence_by_space(
            kernel, kw["jump_ij"], duration,
            kw[f"gate_pop_{shape}"], kw[f"gate_pop_sq_{shape}"])
        # the conditional pulse is applied after the odd-parity syndrome only
        applied = [1.0, 1.0] if half else [0.0, 1.0]
        terms = {
            "stark_shift": [a * stark_factor * float(c["stark_shift"])
                            for a, c in zip(applied, coherent)],
            "dispersive_shift": [a * float(c["dispersive_shift"])
                                 for a, c in zip(applied, coherent)],
            "stark_shift_dispersive_shift_cross": [
                a * float(c["cross"]) for a, c in zip(applied, coherent)],
        }
        # the term of a mechanism is the sum of the costs of its jumps
        for name, operators in ANCILLA_MECHANISMS.items():
            terms[name] = [a * sum(float(costs[PULSE_JUMP_KEYS[operator]][s])
                                   for operator in operators)
                           for s, a in enumerate(applied)]
        # the coherent error is the sum of the three parts, which stay here
        # because cat_code_budget has them as separate terms
        terms["coherent_error"] = [
            terms["stark_shift"][s] + terms["dispersive_shift"][s]
            + terms["stark_shift_dispersive_shift_cross"][s] for s in (0, 1)]
        out[step_key] = terms

    # step 3: parity-mapping idling; the frame change at its end refers to
    # the even-parity error space
    reference_d2 = float(kw["frame_update"]) * moment("cat_mean_d2", 0)
    per_space = [bsq.cat_parity_mapping_idling_error(
        kw["jump_ij"], kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"],
        kw["chi_prime_sa"], kw["t_pm"], mean_n=moment("cat_mean_n", s),
        var_n=moment("cat_var_n", s), mean_d2=moment("cat_mean_d2", s),
        mean_d4=moment("cat_mean_d4", s), reference_d2=reference_d2) for s in (0, 1)]
    out["parity_mapping_idling"] = {
        ("chi_1_prime" if name == "nonlinearity" else name):
            [float(per_space[s][name]) for s in (0, 1)]
        for name in per_space[0]}

    # steps 5 and 6: the ancilla is in |0> (even parity) or |1> (odd parity),
    # and every jump out of that level is a failure
    jump_ij = np.asarray(kw["jump_ij"])
    downward_rates = [0.0, float(jump_ij[0, 1])]
    upward_rates = [float(jump_ij[1, 0]), float(jump_ij[2, 1])]
    for step_key, time_key in (("measurement", "tau_m"), ("feedback_latency", "tau_FD")):
        duration = float(kw[time_key])
        resonator = [bsq.cat_resonator_decoherence(
            kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"], duration,
            mean_n=moment("cat_mean_n", s), var_n=moment("cat_var_n", s)) for s in (0, 1)]
        terms = {name: [float(resonator[s][name]) for s in (0, 1)] for name in resonator[0]}
        terms["ancilla_relaxation"] = [rate * duration for rate in downward_rates]
        terms["ancilla_excitation"] = [rate * duration for rate in upward_rates]
        # the formula has no ancilla-dephasing term for an ancilla in a level
        terms["ancilla_dephasing"] = [0.0, 0.0]
        if step_key == "measurement":
            terms["readout_assignment_error"] = [float(kw["M_ge"]), float(kw["M_eg"])]
        out[step_key] = terms
    return out


def check_against_budget(per_space: dict, ingredients: dict, probabilities: list) -> float:
    r"""Check that the terms on the error spaces sum to the terms of ``cat_code_budget``.

    Parameters
    ----------
    per_space : dict
        Output of :func:`formula_by_error_space`.
    ingredients : dict
        ``sweep.full_dict()`` of the dressed sweep.
    probabilities : list of float
        Probabilities of the even-parity and of the odd-parity error space
        in the formula.

    Returns
    -------
    float
        Largest absolute difference between a term of ``cat_code_budget`` and
        the sum of the same term over the error spaces with their
        probabilities.

    Raises
    ------
    ValueError
        If a difference exceeds 1e-12.
    """
    worst = 0.0
    for term in formula_terms(ingredients):
        value = per_space[term["step_key"]][term["term"]]
        if term["step_key"] != "idle":
            value = sum(p * x for p, x in zip(probabilities, value))
        worst = max(worst, abs(value - term["value"]))
        if abs(value - term["value"]) > 1e-12:
            raise ValueError(
                f"{term['chencrafts_name']}: {value} on the error spaces, "
                f"{term['value']} in cat_code_budget")
    return worst


# ------------------------------------------- numerical contributions --------
def idle_contributions(context: dict) -> dict:
    r"""Numerically computed contribution of each mechanism of the idle.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.

    Returns
    -------
    dict
        ``terms`` (mechanism to contribution: photon loss alone, and the
        increase when a mechanism is added to photon loss) and
        ``all_mechanisms`` (the infidelity with every Lindblad operator).
    """
    full, contributions = idle_mechanism_budget(context["model"])
    return dict(terms={k: float(v) for k, v in contributions.items()},
                all_mechanisms=float(full))


def zero_rate(context: dict, operators: list) -> bool:
    r"""Whether every Lindblad operator of a list has zero rate.

    A mechanism with zero rate adds nothing; its contribution is set to zero
    instead of computed, because two evaluations of the same infidelity can
    differ in the last digit.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.
    operators : list of str
        Plain names of the Lindblad operators.

    Returns
    -------
    bool
    """
    model, index = context["model"], context["index"]
    return not any(np.any(model.full_collapse_operators[index[name]]) for name in operators)


def idling_contributions(context: dict, step_key: str) -> list:
    r"""Contributions of the mechanisms of an idling after the idle, on each error space.

    The step is the parity-mapping idling, the measurement, or the feedback
    latency. Each has the dynamical map of the master equation with a subset
    of the Lindblad operators, followed by the operation that closes the
    step: the frame change of the parity-mapping idling, the projection of
    the ancilla with the confusion matrix of the measurement, or nothing.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.
    step_key : str
        ``"parity_mapping_idling"``, ``"measurement"``, or
        ``"feedback_latency"``.

    Returns
    -------
    list of dict
        For the even-parity and the odd-parity error space: ``terms``
        (mechanism to contribution), ``unitary_dynamics`` (the infidelity
        with the Hamiltonian alone), and ``all_mechanisms`` (the infidelity
        with every Lindblad operator and, for the measurement, the readout
        assignment errors: the dynamical map of the actual evolution tree
        for the measurement and the feedback latency).
    """
    model, index, builder = context["model"], context["index"], context["builder"]
    res_dim, anc_dim = model.resonator_dimension, model.ancilla_dimension
    duration = context[{"parity_mapping_idling": "t_parity_mapping",
                        "measurement": "t_measurement",
                        "feedback_latency": "t_feedback_latency"}[step_key]]
    everything = list(range(len(model.full_collapse_operators)))

    def closing(space: int, assignment_errors: bool) -> np.ndarray:
        r"""Superoperator of the operation that closes the step.

        Parameters
        ----------
        space : int
            0 for the even-parity error space, 1 for the odd-parity one.
        assignment_errors : bool
            Whether the measurement has its readout assignment errors.

        Returns
        -------
        np.ndarray
            Column-stacked superoperator.
        """
        if step_key == "parity_mapping_idling":
            # frame change by -chi_1' t <d^2>_+ at the end of the idling
            z = bsq.cat_ideal.qubit_frame_change(
                res_dim, anc_dim, builder.res_mode_idx, -float(builder.pm_frame_phase)).full()
            return np.kron(z.conj(), z)
        if step_key == "measurement":
            # the map of the reported outcome: 0 for even parity, 1 for odd
            return bsq.cat_real.qubit_projectors(
                res_dim, anc_dim, builder.res_mode_idx,
                confusion_matrix=(builder.measurement_confusion_matrix
                                  if assignment_errors else None),
                ensamble_average=True)[space].full()
        return np.eye((res_dim * anc_dim) ** 2)

    def step_infidelities(collapse_indices: list, assignment_errors: bool = False) -> list:
        r"""Process infidelity of the step on the two error spaces.

        Parameters
        ----------
        collapse_indices : list of int
            Positions of the Lindblad operators that act.
        assignment_errors : bool
            Whether the measurement has its readout assignment errors.

        Returns
        -------
        list of float
            Even-parity and odd-parity error space.
        """
        propagator = idle_propagator(model, collapse_indices, duration)
        result = []
        for space, states in enumerate(context["states"][step_key]):
            ideal = context["ideal"][step_key]
            ideal = ideal[space] if step_key == "measurement" else ideal
            result.append(infidelity(
                closing(space, assignment_errors) @ propagator, ideal, states))
        return result

    unitary = step_infidelities([])
    terms = {}
    for name, operators in IDLING_MECHANISMS.items():
        if zero_rate(context, operators):
            terms[name] = [0.0, 0.0]
            continue
        with_mechanism = step_infidelities([index[operator] for operator in operators])
        terms[name] = [a - b for a, b in zip(with_mechanism, unitary)]
    if step_key == "parity_mapping_idling":
        # the unitary dynamics leave the part of the chi_1' phase that neither
        # the ideal operator nor the frame change removes
        terms["chi_1_prime"] = list(unitary)
    if step_key == "measurement":
        with_assignment = step_infidelities([], assignment_errors=True)
        terms["readout_assignment_error"] = [a - b for a, b in zip(with_assignment, unitary)]
    full = step_infidelities(everything, assignment_errors=True)

    # the measurement and the feedback latency of the tree must be the maps
    # computed here with every mechanism
    if step_key != "parity_mapping_idling":
        tree_key = {"measurement": "qubit_measurement",
                    "feedback_latency": "feedback_latency"}[step_key]
        for space, states in enumerate(context["states"][step_key]):
            tree_map, ideal = builder.step_maps[tree_key], context["ideal"][step_key]
            if step_key == "measurement":
                tree_map, ideal = tree_map[space], ideal[space]
            if abs(infidelity(tree_map.full(), ideal, states) - full[space]) > 1e-10:
                raise ValueError(f"{step_key}: the map of the tree differs from the map built here")

    return [dict(terms={name: float(values[space]) for name, values in terms.items()},
                 unitary_dynamics=float(unitary[space]), all_mechanisms=float(full[space]))
            for space in (0, 1)]


def pulse_contributions(context: dict, step_key: str) -> list:
    r"""Contributions of the mechanisms of an ancilla pulse, on each error space.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context` with the pulses.
    step_key : str
        Key of :data:`PULSES`.

    Returns
    -------
    list of dict or None
        For the even-parity and the odd-parity error space: ``terms``
        (coherent error; each ancilla mechanism of
        :data:`ANCILLA_MECHANISMS` alone),
        ``unitary_dynamics`` and ``all_mechanisms`` (the
        rotating-wave model without and with every ancilla jump), and
        ``pulse_of_the_tree`` (the same two infidelities for the pulse of
        ``chencrafts``). The conditional X_pi pulse is not applied on the
        even-parity error space; its entry there is None.
    """
    dev = context["device"]
    _, theta, duration_factor, pulse_key = PULSES[step_key]
    duration = duration_factor * context["t_pulse"]
    ideal = context["ideal"][step_key]
    result = []
    for space, states in enumerate(context["states"][step_key]):
        if step_key == "conditional_X_pi" and space == 0:
            result.append(None)
            continue
        closed = infidelity(gate_model.framed_model_gate(dev, theta, duration), ideal, states)
        # one ancilla mechanism at a time
        terms = {
            name: infidelity(
                gate_model.model_gate_map(dev, theta, duration, operators), ideal, states) - closed
            for name, operators in ANCILLA_MECHANISMS.items()}
        terms["coherent_error"] = closed
        opened = infidelity(gate_model.model_gate_map(dev, theta, duration), ideal, states)
        result.append(dict(
            terms={name: float(value) for name, value in terms.items()},
            unitary_dynamics=float(closed), all_mechanisms=float(opened),
            pulse_of_the_tree=dict(
                unitary_dynamics=infidelity(dev["pulses_closed"][pulse_key], ideal, states),
                all_mechanisms=infidelity(dev["pulses_open"][pulse_key], ideal, states))))
        print(f"{step_key}, {SPACES[space]}: done", flush=True)
    return result


def numerical_contributions(context: dict, with_pulses: bool) -> dict:
    r"""Numerically computed contributions of every step.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.
    with_pulses : bool
        Whether the three pulses are computed.

    Returns
    -------
    dict
        Step key to the output of :func:`idle_contributions` (idle), or to
        the list over the two error spaces of :func:`idling_contributions`
        and :func:`pulse_contributions`.
    """
    out = {"idle": idle_contributions(context)}
    for step_key in ("parity_mapping_idling", "measurement", "feedback_latency"):
        out[step_key] = idling_contributions(context, step_key)
        print(f"{step_key}: done", flush=True)
    if with_pulses:
        for step_key in PULSES:
            out[step_key] = pulse_contributions(context, step_key)
    return out


# ------------------------------------------------------------- the record ---
def compare(formula: float, numerical: float) -> dict:
    r"""One term of the formula next to its numerically computed contribution.

    Parameters
    ----------
    formula, numerical : float
        Term of the analytical formula and numerical contribution.

    Returns
    -------
    dict
        ``formula``, ``numerical``, ``formula_minus_numerical``, and
        ``formula_over_numerical_minus_1`` (None when the numerical
        contribution vanishes).
    """
    return dict(formula=float(formula), numerical=float(numerical),
                formula_minus_numerical=float(formula - numerical),
                formula_over_numerical_minus_1=relative_difference(formula, numerical))


def idle_excitation_excess(context: dict, measured: float) -> dict:
    r"""Estimate of the excess of the idle ancilla-excitation term over its numerical contribution.

    The ancilla is excited at a time that is uniform over the idle. If it is
    still excited at the end, the state is outside the error space and the
    excitation is a failure with probability 1. If it relaxed before the end,
    the cat state has rotated by an angle that is random, and the failure
    probability is one minus :math:`\sum_m P_{\mathrm L}(m)^2`; the formula
    counts this excitation as a failure with probability 1.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.
    measured : float
        ``formula_over_numerical_minus_1`` of the idle ancilla-excitation
        term, from the comparison.

    Returns
    -------
    dict
        ``return_fidelity`` (:math:`\sum_m P_{\mathrm L}(m)^2`),
        ``relaxation_probability_in_idle`` (:math:`x=\gamma_{\downarrow}t`),
        ``fraction_relaxed_before_the_end``, ``predicted``
        (formula over numerical, minus one), and ``measured``.
    """
    model = context["model"]
    gamma_down = model.rates()["ancilla_relaxation"]
    # ideal states of the even-parity error space at the end of the idle: the
    # two columns are the logical basis states, the ancilla is in its ground state
    states = np.asarray(ideal_idle_states(model)[0])
    anc_dim = model.ancilla_dimension
    # photon-number distribution of the maximally mixed logical state
    p_logical = (np.abs(states) ** 2).reshape(model.resonator_dimension, anc_dim, -1)
    p_logical = p_logical.sum(axis=1).mean(axis=1)
    return_fidelity = float(np.sum(p_logical ** 2))
    x = gamma_down * model.t_idle
    # an excitation at the time t_e relaxes before the end with the
    # probability 1 - exp(-gamma_down (t_idle - t_e)); average over t_e
    fraction_relaxed = float(1.0 - (1.0 - np.exp(-x)) / x)
    failure_over_formula = 1.0 - return_fidelity * fraction_relaxed
    return dict(
        return_fidelity=return_fidelity,
        relaxation_probability_in_idle=float(x),
        fraction_relaxed_before_the_end=fraction_relaxed,
        predicted=float(1.0 / failure_over_formula - 1.0),
        measured=float(measured),
    )


def parity_mapping_chi_prime_removal(context: dict) -> dict:
    r"""Infidelity of the closed parity-mapping idling with and without chi_1'.

    Parameters
    ----------
    context : dict
        Output of :func:`build_context`.

    Returns
    -------
    dict
        ``with_chi_prime`` and ``chi_prime_removed``: process infidelity of
        the step on the even-parity and the odd-parity error space (the
        first is the term of chi_1' of the comparison);
        ``chi_prime_rad_per_ns``; ``constant_phase_rad``, the phase between
        the ancilla levels that the removal leaves.
    """
    model, builder, sweep = context["model"], context["builder"], context["sweep"]
    res_dim, anc_dim = model.resonator_dimension, model.ancilla_dimension
    duration = context["t_parity_mapping"]
    hamiltonian = np.array(model.full_hamiltonian)
    chi_prime, chi_bar = float(sweep["chi_prime_sa"]), float(sweep["chi_sa"])
    # chi_1' a^dag^2 a^2 on the ancilla level 1, with a^dag^2 a^2 = n (n - 1)
    photons = np.diag(np.arange(res_dim))
    level_1 = np.diag([0.0, 1.0] + [0.0] * (anc_dim - 2))
    chi_prime_term = np.kron(photons @ photons - photons, level_1)

    def infidelities(h: np.ndarray, ideal: np.ndarray, frame_phase: float) -> list:
        r"""Process infidelity of the closed step on the two error spaces.

        Parameters
        ----------
        h : np.ndarray
            Hamiltonian of the step (rad/ns).
        ideal : np.ndarray
            Ideal operator of the step.
        frame_phase : float
            Angle of the frame change that closes the step.

        Returns
        -------
        list of float
            Even-parity and odd-parity error space.
        """
        propagator = bsq.cat_real.idling_propagator(qt.Qobj(h), [], float(duration)).full()
        z = bsq.cat_ideal.qubit_frame_change(
            res_dim, anc_dim, builder.res_mode_idx, -frame_phase).full()
        return [infidelity(np.kron(z.conj(), z) @ propagator, ideal, states)
                for states in context["states"]["parity_mapping_idling"]]

    with_chi_prime = infidelities(
        hamiltonian, context["ideal"]["parity_mapping_idling"], float(builder.pm_frame_phase))

    # without chi_1': the ideal operator has no chi_1' phase, and its
    # conditional rotation per photon is that of the changed Hamiltonian
    ideal_without = bsq.cat_ideal.finite_duration_parity_mapping(
        res_dim=res_dim, qubit_dim=anc_dim, res_mode_idx=builder.res_mode_idx,
        angle=chi_bar * duration, duration=duration, K_s=sweep["K_s"], chi_prime=0.0,
        n_ref=builder.n_bar_ref, qubit_state=1).full()
    h_without = hamiltonian - chi_prime * chi_prime_term
    # the removal leaves a constant phase between the two ancilla levels (an
    # ancilla frame change of the protocol removes it): read it at m = 0
    energies = np.real(np.diag(h_without)).reshape(res_dim, anc_dim)
    ideal_phases = np.diag(ideal_without).reshape(res_dim, anc_dim)
    residual = np.angle(np.conj(ideal_phases) * np.exp(-1j * energies * duration))
    constant_phase = float(residual[0, 1] - residual[0, 0])
    h_without = h_without + constant_phase / duration * np.kron(np.eye(res_dim), level_1)
    chi_prime_removed = infidelities(h_without, ideal_without, 0.0)
    return dict(
        with_chi_prime=[float(v) for v in with_chi_prime],
        chi_prime_removed=[float(v) for v in chi_prime_removed],
        chi_prime_rad_per_ns=chi_prime,
        constant_phase_rad=constant_phase,
    )


def idle_excitation_measured(record: dict) -> float:
    r"""Formula over numerical, minus one, of the idle ancilla-excitation term.

    Parameters
    ----------
    record : dict
        Content of ``formula-terms.json`` without the two estimates, or its
        ``cycle_contributions`` list under that key.

    Returns
    -------
    float
        The value of ``formula_over_numerical_minus_1``.
    """
    entries = record["cycle_contributions"] if isinstance(record, dict) else record
    return next(c["formula_over_numerical_minus_1"] for c in entries
                if c["step"] == "idle" and c["term"] == "ancilla_excitation")


def compute() -> dict:
    r"""Compare every term of the formula with its numerical contribution.

    Returns
    -------
    dict
        The record that is written to ``formula-terms.json``.
    """
    context = build_context(device_model.DYNAMICS_RESONATOR_DIMENSION, with_pulses=True)
    ingredients = context["sweep"].full_dict()
    formula = formula_by_error_space(ingredients)
    p_formula = [float(p) for p in bsq.cat_branch_probabilities(**ingredients)]
    budget_check = check_against_budget(formula, ingredients, p_formula)
    p_numerical = idle_error_space_probabilities(context["model"])
    numerical = numerical_contributions(context, with_pulses=True)

    # every term of every step, on each error space
    steps = {}
    for step_key in STEP_KEYS.values():
        if step_key == "idle":
            terms = {name: compare(formula["idle"][name], value)
                     for name, value in numerical["idle"]["terms"].items()}
            steps[step_key] = dict(logical_subspace=dict(
                terms=terms,
                formula_parts_of_photon_loss={
                    k: formula["idle"][k] for k in ("two_or_more_photon_losses", "kerr_term")},
                formula_sum=sum(formula["idle"][name] for name in terms),
                numerical_sum=sum(numerical["idle"]["terms"].values()),
                numerical_all_mechanisms=numerical["idle"]["all_mechanisms"]))
            continue
        steps[step_key] = {}
        for space, space_key in enumerate(SPACES):
            entry = numerical[step_key][space]
            if entry is None:
                continue
            terms = {name: compare(formula[step_key][name][space], value)
                     for name, value in entry["terms"].items()}
            # the terms whose sum is the failure probability of the step
            rows = ROWS[step_key] if step_key in PULSES else list(terms)
            steps[step_key][space_key] = dict(
                terms=terms,
                terms_that_add_up=rows,
                formula_sum=sum(terms[name]["formula"] for name in rows),
                numerical_sum=sum(terms[name]["numerical"] for name in rows),
                numerical_unitary_dynamics=entry["unitary_dynamics"],
                numerical_all_mechanisms=entry["all_mechanisms"])
            if "pulse_of_the_tree" in entry:
                steps[step_key][space_key]["pulse_of_the_tree"] = entry["pulse_of_the_tree"]

    # each row as a contribution to the infidelity of the cycle: the formula
    # with its probabilities of the error spaces, the numerical contribution
    # with the numerically computed probabilities
    cycle = []
    for step_key, row_terms in ROWS.items():
        for name in row_terms:
            if step_key == "idle":
                term = steps["idle"]["logical_subspace"]["terms"][name]
                f, n = term["formula"], term["numerical"]
            else:
                f = sum(p * steps[step_key][s]["terms"][name]["formula"]
                        for p, s in zip(p_formula, SPACES) if s in steps[step_key])
                n = sum(p * steps[step_key][s]["terms"][name]["numerical"]
                        for p, s in zip(p_numerical, SPACES) if s in steps[step_key])
            cycle.append(dict(step=step_key, term=name) | compare(f, n))
    entries = dict(
        formula_by_step={k: sum(c["formula"] for c in cycle if c["step"] == k) for k in ROWS},
        numerical_all_mechanisms_by_step={"idle": numerical["idle"]["all_mechanisms"]} | {
            k: sum(p * steps[k][s]["numerical_all_mechanisms"]
                   for p, s in zip(p_numerical, SPACES) if s in steps[k])
            for k in ROWS if k != "idle"})

    # where the formula is least accurate: per error space by the relative
    # difference, and over the cycle by the absolute difference
    relative = []
    for step_key, row_terms in ROWS.items():
        for space_key, entry in steps[step_key].items():
            for name in row_terms:
                term = entry["terms"][name]
                if term["formula"] != 0 and term["formula_over_numerical_minus_1"] is not None:
                    relative.append(dict(
                        step=step_key, error_space=space_key, term=name,
                        formula_over_numerical_minus_1=term["formula_over_numerical_minus_1"]))
    relative.sort(key=lambda r: -abs(r["formula_over_numerical_minus_1"]))
    absolute = sorted(cycle, key=lambda c: -abs(c["formula_minus_numerical"]))

    # truncation check: the idlings with more resonator states in the dynamics
    larger = build_context(LARGER_RESONATOR_DIMENSION, with_pulses=False)
    numerical_larger = numerical_contributions(larger, with_pulses=False)
    truncation = {"idle": {
        name: abs(numerical_larger["idle"]["terms"][name] - value)
        for name, value in numerical["idle"]["terms"].items()}}
    for step_key in ("parity_mapping_idling", "measurement", "feedback_latency"):
        truncation[step_key] = {
            space_key: {
                name: abs(numerical_larger[step_key][space]["terms"][name] - value)
                for name, value in numerical[step_key][space]["terms"].items()}
            for space, space_key in enumerate(SPACES)}

    model = context["model"]
    return dict(
        description="Term-by-term comparison of the analytical formula with single-mechanism numerical simulations on each parity error space at the parameters of Ofek et al.",
        settings=dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
            truncation_check_resonator_dimension=LARGER_RESONATOR_DIMENSION,
            orthonormalization=bsq.QEC_graph.StateNode.ORTHOGONALIZE_METHOD,
            pulse_model="rotating-wave model of gate_model.py, one ancilla mechanism at a time",
            mechanisms_of_the_pulses=ANCILLA_MECHANISMS,
            mechanisms_of_the_idlings=IDLING_MECHANISMS,
        ),
        environment=device_model.environment_record(1),
        durations_ns=dict(
            t_idle=model.t_idle, t_pulse=context["t_pulse"],
            t_parity_mapping=context["t_parity_mapping"],
            t_measurement=context["t_measurement"],
            t_feedback_latency=context["t_feedback_latency"],
            t_conditional_pulse=2 * context["t_pulse"]),
        rates_per_ns=model.rates(),
        readout_assignment_errors=dict(
            M_ge=float(ingredients["M_ge"]), M_eg=float(ingredients["M_eg"])),
        error_space_probabilities=dict(
            formula=dict(zip(SPACES, p_formula)), numerical=dict(zip(SPACES, p_numerical))),
        largest_difference_from_cat_code_budget=budget_check,
        steps=steps,
        cycle_contributions=cycle,
        cycle_entries=entries,
        least_accurate=dict(
            by_relative_difference_on_an_error_space=relative[:8],
            by_absolute_difference_over_the_cycle=[
                {k: c[k] for k in ("step", "term", "formula", "numerical",
                                   "formula_minus_numerical")} for c in absolute[:8]]),
        truncation_check_absolute_change=truncation,
        idle_excitation_excess_estimate=idle_excitation_excess(
            context, idle_excitation_measured(cycle)),
        parity_mapping_chi_prime_removal=parity_mapping_chi_prime_removal(context),
    )


# ------------------------------------------------------------- the figure ---
def percent(value: float | None) -> str:
    r"""Relative difference as a signed percentage: two digits, or two decimals when small.

    Parameters
    ----------
    value : float or None
        ``formula / numerical - 1``.

    Returns
    -------
    str
        For example ``"+11%"``, ``"-0.8%"``, ``"+0.02%"``.
    """
    if value is None:
        return ""
    x = 100 * value
    if abs(x) >= 9.95:
        return f"{x:+.0f}%"
    if abs(x) >= 0.095:
        return f"{x:+.1f}%"
    return f"{x:+.2f}%"


def mechanism_label(bars: list) -> tuple[str | None, list]:
    r"""The label of a mechanism of a step: its relative differences, or ``omitted``.

    Parameters
    ----------
    bars : list of dict
        The terms of the mechanism on the error spaces of the step, as in
        ``formula-terms.json`` (``formula``, ``numerical``,
        ``formula_over_numerical_minus_1``), or None for an error space on
        which the step is not applied.

    Returns
    -------
    text : str or None
        The relative differences that are at least :data:`LABEL_FLOOR`, in
        the order of the bars, as whole signed percents with one percent sign,
        for example ``$+20/+27\%$``; ``omitted`` where the formula has no term
        for a numerical contribution; None where nothing is written.
    written : list of int
        Indices in ``bars`` of the bars whose relative differences are in
        ``text``; empty for ``omitted`` and where nothing is written.
    """
    numbers, written, omitted = [], [], False
    for index, term in enumerate(bars):
        # a numerical zero of the solver is not a contribution; None is an
        # error space on which the step is not applied
        if term is None or term["numerical"] <= 1e-12:
            continue
        if term["formula"] == 0:
            omitted = True
        elif abs(term["formula_over_numerical_minus_1"]) >= LABEL_FLOOR:
            numbers.append(100 * term["formula_over_numerical_minus_1"])
            written.append(index)
    if omitted:
        return "omitted", []
    if not numbers:
        return None, []
    # each number in braces, so that its sign is the unary sign of the math font
    return plot_style.percent_text("/".join(f"{{{x:+.0f}}}" for x in numbers)), written


def plot(record: dict) -> None:
    r"""Draw the terms of the seven steps in one panel, colored by mechanism.

    The steps follow one another along the horizontal axis. A mechanism of a
    step has two neighboring bars, the numerical contribution on the
    even-parity error space and on the odd-parity one (hatched); a term of the idle,
    which starts in the logical subspace, is one bar, and so is a term of the
    conditional X_pi pulse, which is applied on the odd-parity error space
    only. Every bar has
    the width ``plot_style.BAR_WIDTH_PT``; the bars of one mechanism are
    ``plot_style.BAR_GAP_PT`` apart, the mechanisms of one step
    :data:`MECHANISM_GAP_PT`. The oblique cross on a bar is the term of the
    analytical formula for the same mechanism. The relative differences of
    the bars of a mechanism are written in one vertical label, centered over
    them and :data:`LABEL_LIFT_PT` above the highest bar or marker, where
    they are at least :data:`LABEL_FLOOR`; a label that holds the number of
    only one bar of two is centered over that bar; ``omitted`` marks a mechanism
    with a numerical contribution for which the formula has no term (a bar
    without a cross, which keeps the color of its mechanism).

    Parameters
    ----------
    record : dict
        Output of :func:`compute`, or the content of the JSON file.

    Returns
    -------
    None
    """
    plot_style.use_manuscript_style()
    colors = plot_style.TERM_COLORS
    # the short names of the steps of every figure, set horizontally
    step_labels = [label for _, label in sorted(plot_style.STEP_LABELS.items())]
    y_low, y_high = 3e-6, 0.25

    # the axes box is placed in inch, so that the widths of the bars in pt
    # can be converted to data units
    left, bottom, right, top = FIGURE_MARGINS
    fig = plt.figure(figsize=(plot_style.TEXT_WIDTH, FIGURE_HEIGHT))
    plot_style.set_canvas(fig, plot_style.TEXT_WIDTH, FIGURE_HEIGHT)
    ax = plot_style.axes_in_inches(fig, left, bottom, plot_style.TEXT_WIDTH - left - right,
                                   FIGURE_HEIGHT - bottom - top)
    ax.set_yscale("log")
    ax.set_ylim(y_low, y_high)
    # the horizontal axis is only a place: its width is one data unit
    ax.set_xlim(0.0, 1.0)

    # The mechanisms of each step, with their terms on the error spaces. A
    # mechanism has one bar per error space on which the step is applied: one
    # for the idle (the logical subspace) and for the conditional X_pi pulse
    # (the odd-parity error space only), two for every other step
    steps = []
    for step_key, names in ROWS.items():
        spaces = record["steps"][step_key]
        keys = (list(spaces) if "logical_subspace" in spaces
                else [key for key in SPACES if key in spaces])
        steps.append([dict(mechanism=TERM_MECHANISMS.get(name, TERM_LABELS[name]),
                           spaces=keys,
                           terms=[spaces[key]["terms"][name] if key in spaces else None
                                  for key in keys])
                      for name in names])
    # widths in data units: the bars of a mechanism, the mechanisms of a step,
    # and the gap between two steps, which takes what the steps leave
    bar_width = plot_style.points_to_data(ax, plot_style.BAR_WIDTH_PT)
    bar_gap = plot_style.points_to_data(ax, plot_style.BAR_GAP_PT)
    mechanism_gap = plot_style.points_to_data(ax, MECHANISM_GAP_PT)
    for groups in steps:
        for group in groups:
            count = len(group["terms"])
            group["width"] = count * bar_width + (count - 1) * bar_gap
    step_widths = [sum(g["width"] for g in groups) + (len(groups) - 1) * mechanism_gap
                   for groups in steps]
    step_gap = (1.0 - sum(step_widths)) / len(steps)
    # the steps must stand further apart than the mechanisms of one step
    if step_gap < 2 * mechanism_gap:
        raise ValueError("the bars leave too little room between the steps")

    # bars (numerical contributions), markers (terms of the formula), and
    # labels, step by step
    marker = plot_style.bar_marker_style("analytical formula")
    # the marker reaches this far above and below its value (pt)
    marker_radius = (marker["ms"] + marker["markeredgewidth"]) / 2
    centers, x = [], step_gap / 2
    for groups, width in zip(steps, step_widths):
        slot = (x - step_gap / 2, x + width + step_gap / 2)
        centers.append(x + width / 2)
        for group in groups:
            center = x + group["width"] / 2
            _, offsets = plot_style.group_offsets(ax, len(group["terms"]))
            face = colors[group["mechanism"]]
            # height and distance of the top above it (pt) of each bar and marker
            tops = []
            for space_key, term, offset in zip(group["spaces"], group["terms"], offsets):
                # a step that is not applied on an error space leaves its place empty
                if term is None:
                    continue
                # a numerical zero of the solver is not a contribution
                if term["numerical"] > 1e-12:
                    hatch = (plot_style.hatch_style(plot_style.DIAGONAL_HATCH, face)
                             if space_key == "odd_parity" else {})
                    ax.bar(center + offset, term["numerical"] - y_low, bar_width, bottom=y_low,
                           facecolor=face, linewidth=0, **hatch)
                    tops.append((term["numerical"], 0.0))
                # a mechanism without a term in the formula has no cross
                if term["formula"] > 0:
                    ax.plot([center + offset], [term["formula"]], **marker)
                    tops.append((term["formula"], marker_radius))
            text, written = mechanism_label(group["terms"])
            if text is not None:
                # a lone number of two bars stands over its own bar, so that
                # it is clear which error space it belongs to
                label_x = center + (offsets[written[0]] if len(written) == 1 else 0.0)
                # the highest top of the mechanism on the page, bar or marker,
                # so that a lone label also clears the marker of the other bar
                y, above = max(tops, key=lambda top: (
                    ax.transData.transform((0.0, top[0]))[1] * 72.0 / fig.dpi + top[1]))
                # vertical, so that it is no wider than a bar and stays over its own
                ax.annotate(text, (label_x, y), xytext=(0, above + LABEL_LIFT_PT),
                            textcoords="offset points", rotation=90, ha="center",
                            va="bottom", fontsize=plot_style.SMALL_TEXT_SIZE,
                            color=plot_style.DARK_GRAY)
            x += group["width"] + mechanism_gap
        # the next step starts half a gap after the end of this slot
        x = slot[1] + step_gap / 2
        # a line between two steps
        if slot[1] < 1.0 - step_gap / 2:
            ax.axvline(slot[1], color=plot_style.LIGHT_GRAY, lw=plot_style.THIN_WIDTH)
    ax.set_xticks(centers, step_labels)
    ax.tick_params(axis="x", length=0)
    plot_style.value_grid(ax)
    ax.set_ylabel("infidelity contribution")

    # the error spaces of the bars and the formula inside the axes, at its
    # upper right corner
    handles = [
        Patch(facecolor=plot_style.NEUTRAL, linewidth=0, label="even-parity error space"),
        Patch(facecolor=plot_style.NEUTRAL, linewidth=0, label="odd-parity error space",
              **plot_style.hatch_style(plot_style.DIAGONAL_HATCH, plot_style.NEUTRAL)),
        Line2D([], [], label="AEB by mechanism", **marker)]
    # on a white ground without an edge, so that the lines between the steps
    # and the grid stop at it
    parity_legend = ax.legend(handles=handles, ncol=3, loc="upper right",
                              **plot_style.BAR_LEGEND | dict(
                                  frameon=True, facecolor="white", edgecolor="none",
                                  framealpha=1.0, fancybox=False, borderpad=0.2))
    ax.add_artist(parity_legend)
    # the mechanisms below the names of the steps, in their order: the seven
    # entries are wider than the axes, so they are set in two rows, each a
    # legend of one row that is centered under the axes
    patches = [Patch(facecolor=color, linewidth=0, label=name) for name, color in colors.items()]
    if sum(MECHANISM_LEGEND_ROWS) != len(patches):
        raise ValueError("the rows of the legend do not hold every mechanism")
    start = 0
    for row, size in enumerate(MECHANISM_LEGEND_ROWS):
        drop = MECHANISM_LEGEND_DROP + row * MECHANISM_LEGEND_ROW
        below = ax.transAxes + ScaledTranslation(0, -drop / 72, fig.dpi_scale_trans)
        entries = patches[start:start + size]
        start += size
        legend = ax.legend(
            handles=entries, ncol=len(entries), loc="upper center", bbox_to_anchor=(0.5, 0),
            bbox_transform=below, **plot_style.BAR_LEGEND | dict(borderaxespad=0.0))
        # the axes keep only the legend made last; an earlier one is added to them
        if start < len(patches):
            ax.add_artist(legend)

    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


# -------------------------------------------------------------- the table ---
def latex_number(value: float) -> str:
    r"""A contribution in units of 10^-4, for the table.

    Parameters
    ----------
    value : float
        Contribution to the infidelity.

    Returns
    -------
    str
        Three significant digits; ``---`` for zero.
    """
    if value == 0:
        return "---"
    scaled = 1e4 * value
    return f"{scaled:.3g}" if abs(scaled) >= 0.0995 else f"{scaled:.2g}"


def latex_table(record: dict) -> str:
    r"""The content of the figure as the body of a LaTeX table.

    Parameters
    ----------
    record : dict
        Output of :func:`compute`, or the content of the JSON file.

    Returns
    -------
    str
        A ``tabular`` environment: one row per term, and for each error
        space the term of the formula, the numerical contribution (both in
        units of 10^-4), and their relative difference.
    """
    lines = [
        "% Written by formula-terms.py --table from formula-terms.json.",
        "% Units: 10^-4 (contribution to the process infidelity of the step on the error space).",
        "% The idle starts in the logical subspace; its values are in the even-parity columns.",
        r"\begin{tabular}{l rrr rrr}",
        r"\hline\hline",
        r" & \multicolumn{3}{c}{even-parity error space} & "
        r"\multicolumn{3}{c}{odd-parity error space} \\",
        r"Term & formula & numerical & diff. & formula & numerical & diff. \\",
        r"\hline",
    ]
    for step_key, row_terms in ROWS.items():
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{STEP_TABLE_LABELS[step_key]}}}}} \\")
        entry = record["steps"][step_key]
        for name in row_terms:
            cells = []
            for space_key in ("even_parity", "odd_parity"):
                key = "logical_subspace" if (
                    step_key == "idle" and space_key == "even_parity") else space_key
                if key not in entry:
                    cells += ["", "", ""]
                    continue
                term = entry[key]["terms"][name]
                difference = (percent(term["formula_over_numerical_minus_1"]).replace("%", r"\%")
                              if term["formula"] != 0 else "")
                cells += [latex_number(term["formula"]), latex_number(term["numerical"]),
                          f"${difference}$" if difference else ""]
            lines.append(rf"\quad {TERM_LABELS[name]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from the JSON file")
    parser.add_argument("--add-estimates", action="store_true",
                        help="compute only the idle-excitation estimate and the chi_1' removal "
                             "and add them to the JSON file")
    parser.add_argument("--table", type=Path, default=None,
                        help="also write the content of the figure as a LaTeX table to this file")
    arguments = parser.parse_args()
    if arguments.add_estimates:
        # the two results need the dressed sweep and the ideal operators only
        record = json.loads(RESULTS_JSON.read_text())
        context = build_context(device_model.DYNAMICS_RESONATOR_DIMENSION, with_pulses=False)
        record["idle_excitation_excess_estimate"] = idle_excitation_excess(
            context, idle_excitation_measured(record))
        record["parity_mapping_chi_prime_removal"] = parity_mapping_chi_prime_removal(context)
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps({k: record[k] for k in (
            "idle_excitation_excess_estimate", "parity_mapping_chi_prime_removal")}, indent=1))
        print(f"wrote {RESULTS_JSON}")
        raise SystemExit
    if arguments.plot_only:
        record = json.loads(RESULTS_JSON.read_text())
    else:
        record = compute()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["least_accurate"], indent=1))
        print(f"wrote {RESULTS_JSON}")
    plot(record)
    if arguments.table is not None:
        arguments.table.write_text(latex_table(record))
        print(f"wrote {arguments.table}")
