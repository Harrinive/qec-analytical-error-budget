r"""Shared tools of the scripts on the parameter optimization of the cat-code example.

Provides the three reference parameter sets, dispersive-regime checks, local
minimization helpers, derived static/decoherence quantities, and one-cycle
fidelity evaluations. Units: circuit energies and frequencies in GHz, durations
in ns, rates in 1/ns unless a key ends with ``per_ms``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

import cycle_target as ct
import device_model

# labels of the three parameter sets, as keys of the result files
OFEK = device_model.OFEK
POSITIVE_DETUNING, NEGATIVE_DETUNING = "Delta>0 minimum", "Delta<0 minimum"
# result of the search, written by optimization.py
OPTIMIZATION_JSON = Path(__file__).resolve().parent / "optimization.json"

REFERENCE = ct.REFERENCE
FREE_NAMES = ct.FREE_NAMES
NS_PER_MS = ct.NS_PER_MS
# the resonator must be detuned from the 0-1 and the 1-2 transition by at
# least this many couplings g, otherwise the parameter set is left out of a sweep
MIN_DETUNING_OVER_COUPLING = 4.0
# names of the three computations of one cycle in the result files
FORMULA, CHAIN, ACTUAL = device_model.FORMULA, device_model.CHAIN, device_model.ACTUAL


def parameter_sets() -> dict:
    r"""Free parameters of the three parameter sets of the example.

    Returns
    -------
    dict
        ``OFEK``, ``POSITIVE_DETUNING``, ``NEGATIVE_DETUNING`` to the four
        free parameters: the reference values, and the two minima of
        ``optimization.json``, each at its own idle duration.
    """
    minima = json.loads(OPTIMIZATION_JSON.read_text())["minima"]
    sets = {OFEK: {name: REFERENCE[name] for name in FREE_NAMES}}
    sets |= {label: minima[label]["free_parameters"]
             for label in (POSITIVE_DETUNING, NEGATIVE_DETUNING)}
    return {label: {name: float(values[name]) for name in FREE_NAMES}
            for label, values in sets.items()}


def on_bound(value: float, bound: float, tolerance: float) -> bool:
    r"""Whether a parameter lies on a bound of the range in which it was searched.

    Parameters
    ----------
    value : float
        Value of the parameter.
    bound : float
        Lower or upper bound of the range.
    tolerance : float
        Largest distance from the bound that counts as lying on it, in the
        unit of the parameter.

    Returns
    -------
    bool
        True if the distance is below the tolerance.
    """
    return bool(abs(value - bound) < tolerance)


def bounds_reached(free_parameters: dict, relative_tolerance: float = 1e-6) -> dict:
    r"""The bounds of the search ranges on which a parameter set lies.

    Parameters
    ----------
    free_parameters : dict
        Free parameters; those without a search range are ignored.
    relative_tolerance : float
        Tolerance of :func:`on_bound` in units of the width of the range.

    Returns
    -------
    dict
        Name of each parameter that lies on a bound to ``"lower"`` or
        ``"upper"``.
    """
    reached = {}
    for name, (lower, upper) in ct.RANGES.items():
        if name not in free_parameters:
            continue
        tolerance = relative_tolerance * (upper - lower)
        if on_bound(free_parameters[name], lower, tolerance):
            reached[name] = "lower"
        elif on_bound(free_parameters[name], upper, tolerance):
            reached[name] = "upper"
    return reached


def transmon_transitions(EJ_GHz: float) -> list[float]:
    r"""Frequencies of the 0-1 and the 1-2 transition of the bare transmon.

    Parameters
    ----------
    EJ_GHz : float
        Josephson energy; the charging energy and the offset charge are those
        of the reference parameters.

    Returns
    -------
    list of float
        The two transition frequencies (GHz), from the bare transmon of the
        dephasing model (``device_model.bare_transmon``).
    """
    transmon = device_model.bare_transmon(EJ_GHz, REFERENCE["EC_GHz"], REFERENCE["ng"])
    return [float(transition) for transition in np.diff(transmon.eigenvals(evals_count=3))]


def detuning(free_parameters: dict) -> float:
    r"""Detuning of the resonator from the ancilla transition.

    Parameters
    ----------
    free_parameters : dict
        Holds ``E_osc_GHz`` and ``EJ_GHz``.

    Returns
    -------
    float
        :math:`\Delta/2\pi=(\omega_{\mathrm r}-\omega_{\mathrm a})/2\pi` (GHz), with
        the 0-1 transition of the bare transmon.
    """
    return free_parameters["E_osc_GHz"] - transmon_transitions(free_parameters["EJ_GHz"])[0]


def detuning_over_coupling(free_parameters: dict) -> float:
    r"""Smallest detuning of the resonator from the two transitions, in couplings.

    Parameters
    ----------
    free_parameters : dict
        Holds ``E_osc_GHz``, ``g_GHz``, and ``EJ_GHz``.

    Returns
    -------
    float
        :math:`\min_{j}|\omega_{j,j+1}-\omega_{\mathrm r}|/g` over the 0-1 and
        the 1-2 transition.
    """
    detunings = [abs(transition - free_parameters["E_osc_GHz"])
                 for transition in transmon_transitions(free_parameters["EJ_GHz"])]
    return min(detunings) / free_parameters["g_GHz"]


def in_dispersive_regime(free_parameters: dict) -> bool:
    r"""Whether a parameter set is kept in a sweep.

    Parameters
    ----------
    free_parameters : dict
        Holds ``E_osc_GHz``, ``g_GHz``, and ``EJ_GHz``.

    Returns
    -------
    bool
        True if :func:`detuning_over_coupling` is at least
        ``MIN_DETUNING_OVER_COUPLING``.
    """
    return detuning_over_coupling(free_parameters) >= MIN_DETUNING_OVER_COUPLING


def formula_record(free_parameters: dict, evaluation: dict | None = None) -> dict:
    r"""Analytical formula at one parameter set, in the form that the result files store.

    Parameters
    ----------
    free_parameters : dict
        The four free parameters; they replace their reference values.
    evaluation : dict, optional
        Output of ``cycle_target.evaluate_formula`` with the breakdown at
        this parameter set, if the caller already holds it.

    Returns
    -------
    dict
        ``free_parameters``; ``omega_01_GHz`` (0-1 transition of the bare
        transmon) and ``detuning_GHz``; ``t_cycle_ns``; ``infidelity``
        (:math:`1-F` of one cycle); ``failure_rate_per_ms``;
        ``rate_by_step_per_ms`` (contribution of each of the seven steps, in
        the order of the steps) and ``rate_by_mechanism_per_ms`` (mechanism
        name to contribution); ``ingredients`` (the
        ``cycle_target.INGREDIENT_KEYS``; rad/ns, 1/ns, ns, GHz) and
        ``kappa_a_phi`` (ancilla dephasing rate, 1/ns).
    """
    if evaluation is None:
        evaluation = ct.evaluate_formula(REFERENCE | free_parameters)
    omega_01 = transmon_transitions(free_parameters["EJ_GHz"])[0]
    return dict(
        free_parameters={name: float(free_parameters[name]) for name in FREE_NAMES},
        omega_01_GHz=omega_01,
        detuning_GHz=float(free_parameters["E_osc_GHz"] - omega_01),
        t_cycle_ns=evaluation["t_cycle"],
        infidelity=evaluation["infidelity"],
        failure_rate_per_ms=evaluation["rate"] * NS_PER_MS,
        rate_by_step_per_ms=[evaluation["rate_by_step"][n] * NS_PER_MS
                             for n in sorted(evaluation["rate_by_step"])],
        rate_by_mechanism_per_ms={name: value * NS_PER_MS for name, value
                                  in evaluation["rate_by_mechanism"].items()},
        ingredients=evaluation["ingredients"],
        kappa_a_phi=evaluation["kappa_a_phi"],
    )


def derived_quantities(full_dict: dict) -> dict:
    r"""Quantities of the table that follow from the parameters of one set.

    Every rate is the coefficient :math:`\gamma_{\hat L}` of the dissipator
    :math:`\gamma_{\hat L}\mathcal D[\hat L]` of the master equation; for the
    ancilla dephasing :math:`\hat L=|1\rangle\langle1|`, so the dephasing time
    is :math:`T_\phi=2/\gamma_{|1\rangle\langle1|}`.

    Parameters
    ----------
    full_dict : dict
        Every key of the dressed sweep of the parameter set.

    Returns
    -------
    dict
        ``omega_01_GHz`` (ancilla transition frequency of the formula's
        ingredients); ``chi_1_MHz``, ``chi_1_prime_kHz``, ``K_kHz`` (each
        divided by :math:`2\pi`, with its sign); ``rates_per_ms``: photon
        loss, photon gain, ancilla relaxation (level 1 to 0), ancilla
        excitation (0 to 1), ancilla dephasing (jump operator :math:`|1\rangle\langle1|`), the
        two jumps that involve level 2 (1 to 2 and 2 to 1), and photon loss
        with the ancilla in level 1 (jump operator :math:`\hat a|1\rangle\langle1|`); the durations
        ``t_cycle_us``, ``t_idle_us``, ``t_parity_mapping_idling_ns``,
        ``t_pulse_ns`` (each of the three pulses).
    """
    # jump_ij[i, j] is the rate of the ancilla jump |i><j|
    ancilla_jumps = np.array(full_dict["jump_ij"], dtype=float)
    return dict(
        omega_01_GHz=float(full_dict["omega_a_GHz"]),
        chi_1_MHz=float(full_dict["chi_1"]) / (2 * np.pi) * 1e3,
        chi_1_prime_kHz=float(full_dict["chi_prime_sa"]) / (2 * np.pi) * 1e6,
        K_kHz=float(full_dict["K_s"]) / (2 * np.pi) * 1e6,
        rates_per_ms=dict(
            photon_loss=float(full_dict["jump_a"]) * NS_PER_MS,
            photon_gain=float(full_dict["jump_adag"]) * NS_PER_MS,
            ancilla_relaxation=float(ancilla_jumps[0, 1]) * NS_PER_MS,
            ancilla_excitation=float(ancilla_jumps[1, 0]) * NS_PER_MS,
            ancilla_dephasing=float(ancilla_jumps[1, 1]) * NS_PER_MS,
            ancilla_level_1_to_2=float(ancilla_jumps[2, 1]) * NS_PER_MS,
            ancilla_level_2_to_1=float(ancilla_jumps[1, 2]) * NS_PER_MS,
            photon_loss_with_excited_ancilla=float(
                np.array(full_dict["jump_a_ij"], dtype=float)[1, 1]) * NS_PER_MS),
        t_cycle_us=float(full_dict["t_cycle"]) / 1e3,
        t_idle_us=float(full_dict["T_W"]) / 1e3,
        t_parity_mapping_idling_ns=float(full_dict["t_pm"]),
        t_pulse_ns=float(full_dict["tau_p"]),
    )


def one_cycle_fidelities(sweep, simulation_parameters: dict, num_cpus: int) -> dict:
    r"""Fidelity loss at each step of one QEC cycle, from the two numerically computed models.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set (``device_model.build_cat_sweep``).
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.
    num_cpus : int
        Worker processes that build the dynamical maps of the seven steps.

    Returns
    -------
    dict
        The output of ``device_model.one_cycle_fidelity_losses``: ``ACTUAL``
        and ``CHAIN``, each to ``by_step`` (list of seven entries: the loss of
        the process fidelity of the actual QEC dynamics at each step, or the
        increase of the failure probability of the failure Markov chain at
        each step), ``infidelity`` (one minus the process fidelity, or the
        failure probability, at the end of the cycle), and
        ``after_each_step``.
    """
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=num_cpus)
    losses = device_model.one_cycle_fidelity_losses(graph)
    # the tree holds the maps of every step; free it before the next one
    del graph
    gc.collect()
    return losses


def three_model_record(free_parameters: dict, num_cpus: int) -> dict:
    r"""Formula, failure Markov chain, and actual QEC dynamics of one cycle at one parameter set.

    Builds the actual evolution tree of one cycle (about a minute on four
    cores).

    Parameters
    ----------
    free_parameters : dict
        The four free parameters; they replace their reference values.
    num_cpus : int
        Worker processes that build the dynamical maps.

    Returns
    -------
    dict
        ``free_parameters``; ``t_cycle_ns``; ``infidelity`` and
        ``failure_rate_per_ms``, each a dictionary with the keys ``FORMULA``,
        ``CHAIN``, ``ACTUAL``; ``infidelity_by_step`` (the same keys to the
        seven steps: terms of the formula, and fidelity losses
        of the two numerically computed models); ``ingredients`` (the
        ``cycle_target.INGREDIENT_KEYS``).
    """
    # the formula; its sweep also feeds the tree
    evaluation = ct.evaluate_formula(REFERENCE | free_parameters, keep_sweep=True)
    numerical = one_cycle_fidelities(evaluation["sweep"], evaluation["simulation_parameters"],
                                     num_cpus)
    by_step = {FORMULA: [evaluation["by_step"][n] for n in sorted(evaluation["by_step"])],
               CHAIN: numerical[CHAIN]["by_step"], ACTUAL: numerical[ACTUAL]["by_step"]}
    t_cycle = evaluation["t_cycle"]
    infidelity = {FORMULA: evaluation["infidelity"], CHAIN: numerical[CHAIN]["infidelity"],
                  ACTUAL: numerical[ACTUAL]["infidelity"]}
    return dict(
        free_parameters={name: float(free_parameters[name]) for name in FREE_NAMES},
        t_cycle_ns=t_cycle,
        infidelity=infidelity,
        failure_rate_per_ms={name: float(device_model.failure_rate_per_ms(value, t_cycle))
                             for name, value in infidelity.items()},
        infidelity_by_step=by_step,
        ingredients=evaluation["ingredients"],
    )


def minimize_formula(start: dict, searched: list[str], fixed: dict,
                     target=None, ranges: dict | None = None) -> tuple[dict, dict]:
    r"""Minimize the failure rate of the formula over some of the free parameters.

    Parameters
    ----------
    start : dict
        Starting values of the searched parameters; a value outside its range
        starts at the nearest bound.
    searched : list of str
        Names of the parameters that the minimization varies, each within its
        search range.
    fixed : dict
        Values of the free parameters that stay fixed.
    target : callable, optional
        Function of the full parameter dictionary that is minimized;
        ``cycle_target.target_func`` if None.
    ranges : dict, optional
        Searched parameter to ``[lower, upper]``, replacing its search range
        of ``search_ranges.py``; a searched parameter that is not listed keeps
        its search range.

    Returns
    -------
    free_parameters : dict
        The free parameters at the minimum (searched and fixed).
    result : dict
        ``evaluations`` (evaluations of the target by the minimizer, ``nfev``
        of scipy; the one more per iteration that records the trajectory is
        not counted), ``iterations``, ``success``, and ``message`` of scipy.
    """
    fixed = {name: float(value) for name, value in fixed.items()}
    # L-BFGS-B in the searched parameters scaled to [0, 1] over their ranges,
    # with the settings cycle_target.LOCAL_OPTIONS
    optimization = ct.build_optimization(
        free_ranges={name: (ranges or {}).get(name, ct.RANGES[name]) for name in searched},
        fixed=fixed, target=target)
    trajectory = optimization.run(init_x={name: start[name] for name in searched},
                                  clip_init=True)
    scipy_result = trajectory.opt_result
    minimum = optimization.scaled_to_physical(scipy_result.x, clip=True)
    return ({name: float(minimum[name]) for name in searched} | fixed,
            dict(evaluations=int(scipy_result.nfev), iterations=int(scipy_result.nit),
                 success=bool(scipy_result.success), message=str(scipy_result.message)))
