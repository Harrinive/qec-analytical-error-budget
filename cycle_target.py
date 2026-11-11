r"""Analytical formula of one QEC cycle and the target of the parameter optimization.

Wraps the dressed sweep and ``qec_aeb.cat_code_budget`` into a
``chencrafts.toolbox.Optimization`` problem over the four free parameters in
``search_ranges.py``. Units: circuit energies in GHz, rates in 1/ns, times in
ns.
"""

from __future__ import annotations

import qec_aeb as bsq
import chencrafts.settings as cc_settings
import chencrafts.toolbox as tb

import device_model
from search_ranges import FREE_PARAMETER_RANGES

# progress bars would print once per evaluation of the target
cc_settings.PROGRESSBAR_DISABLED = True

# simulation parameters and reference parameters (those of Ofek et al.)
SIMULATION_PARAMETERS, _FIXED, _DEVICE = device_model.exact_cat_parameters()
REFERENCE = _FIXED | _DEVICE
RANGES = FREE_PARAMETER_RANGES
FREE_NAMES = list(RANGES)

# a rate in 1/ns times this factor is the rate in 1/ms
NS_PER_MS = device_model.NS_PER_MS
# The three coherent terms that the formula has for every ancilla pulse. With
# the ancilla frame changes on (``frame_update`` of
# ``device_model.BUDGET_PARAMETERS``) these terms vanish for the two
# X(+-pi/2) pulses, so their sum over the cycle is the coherent error of the
# conditional X(pi) pulse, one contribution of the formula as the manuscript writes it
PULSE_COHERENT = ("drive Stark shift", "dispersive shift",
                  "Stark-shift/dispersive-shift cross")
PULSE_COHERENT_LABEL = "coherent error of the conditional X_pi pulse"
# mechanism names of cat_code_budget that the result files store under the
# name of the manuscript; the other names are stored as they are
MECHANISM_NAMES = {
    "storage dephasing": "resonator dephasing",
    "readout assignment": "readout assignment error",
    "qubit-state-dependent non-linearity": "chi_1'",
}
# Dressed quantities that the scripts quote: dispersive shift chi_1,
# qubit-state-dependent resonator non-linearity chi_1', resonator self-Kerr
# coefficient K (rad/ns), photon-loss and photon-gain rate (1/ns), duration of
# the parity-mapping idling (ns), and the ancilla transition frequency of the
# formula's ingredients (GHz)
INGREDIENT_KEYS = ("chi_1", "chi_prime_sa", "K_s", "jump_a", "jump_adag", "t_pm",
                   "omega_a_GHz")
# Options of L-BFGS-B, the local minimizer of every minimization of this
# folder. Each free parameter is scaled to [0, 1] over its range and the target
# is of order one. "eps" is the finite-difference step of the gradient in that
# scaled coordinate: the scipy default (1e-8) is below the numerical noise of
# the target, and the line search then fails after a few iterations. "tol" is
# the convergence threshold on the target and on its projected gradient
LOCAL_OPTIONS = {"eps": 1e-5, "tol": 1e-8, "maxiter": 100}


def mechanism_entries(full_dict: dict) -> dict:
    r"""
    Entries of the analytical formula by mechanism, under the names of the manuscript.

    Parameters
    ----------
    full_dict : dict
        Every key of the dressed sweep of a parameter set, of either ancilla.

    Returns
    -------
    dict
        Mechanism name (those of ``MECHANISM_NAMES`` where they differ from
        ``cat_code_budget``) to its contribution to :math:`1-F`; the three coherent
        terms of the pulses are one entry, ``PULSE_COHERENT_LABEL``.
    """
    by_mechanism = {PULSE_COHERENT_LABEL: 0.0}
    for name, value in bsq.cat_code_budget("mechanism", **full_dict).items():
        if name in PULSE_COHERENT:
            by_mechanism[PULSE_COHERENT_LABEL] += float(value)
        else:
            by_mechanism[MECHANISM_NAMES.get(name, name)] = float(value)
    return by_mechanism


def evaluate_formula(parameters: dict, breakdown: bool = True, keep_sweep: bool = False) -> dict:
    r"""
    Analytical formula of one QEC cycle at one set of parameters.

    Parameters
    ----------
    parameters : dict
        Fixed and device parameters merged. The ancilla dephasing rate is
        replaced by that of ``device_model.with_dephasing_model``.
    breakdown : bool
        Whether to evaluate the contributions by step and by mechanism. The
        minimizer needs only the failure rate and leaves them out.
    keep_sweep : bool
        Whether to return the dressed sweep. It holds the dressed operators,
        so a caller that collects many evaluations leaves it out.

    Returns
    -------
    dict
        ``infidelity`` (:math:`1-F` of one cycle), ``t_cycle`` (ns), ``rate``
        (failure rate, 1/ns), ``kappa_a_phi`` (ancilla dephasing rate of the
        dephasing model, 1/ns), and ``ingredients`` (the ``INGREDIENT_KEYS``).
        With ``keep_sweep``, also ``sweep``, ``full_dict`` (every key of the
        sweep), and ``simulation_parameters`` (those the sweep was built with,
        which the tree builder takes). With ``breakdown``, also the entries of
        :math:`1-F`: ``by_step`` (step number 1 to 7 to entry) and
        ``by_mechanism`` (mechanism name to entry, with the names of
        ``MECHANISM_NAMES``; the coherent terms of the pulses are one entry,
        see ``PULSE_COHERENT``); and the same entries
        divided by the cycle duration, the contributions to the failure rate
        (1/ns): ``rate_by_step`` and ``rate_by_mechanism``.
    """
    # the ancilla dephasing rate follows the Josephson energy; at the reference
    # parameters it is the measured rate
    parameters = device_model.with_dephasing_model(parameters)
    # build_cat_sweep writes the adapted size of the charge basis back into
    # the simulation parameters; a copy keeps every evaluation independent of
    # the ones before
    simulation_parameters = dict(SIMULATION_PARAMETERS)
    sweep = device_model.build_cat_sweep(simulation_parameters, parameters)
    full_dict = sweep.full_dict()

    # the formula: scalar arithmetic on the ingredients
    infidelity = device_model.formula_infidelity(full_dict)
    t_cycle = float(full_dict["t_cycle"])
    result = dict(
        infidelity=infidelity, t_cycle=t_cycle, rate=infidelity / t_cycle,
        kappa_a_phi=float(parameters["kappa_a_phi"]),
        ingredients={key: float(full_dict[key]) for key in INGREDIENT_KEYS},
    )
    if keep_sweep:
        result |= dict(sweep=sweep, full_dict=full_dict,
                       simulation_parameters=simulation_parameters)
    if not breakdown:
        return result

    # entries of 1 - F by step and by mechanism, as cat_code_budget groups them
    by_step = {int(n): float(v) for n, v in bsq.cat_code_budget("step", **full_dict).items()}
    by_mechanism = mechanism_entries(full_dict)
    # an entry divided by the cycle duration is its contribution to the
    # failure rate, so the entries of either grouping add up to the rate
    return result | dict(
        by_step=by_step, by_mechanism=by_mechanism,
        rate_by_step={n: v / t_cycle for n, v in by_step.items()},
        rate_by_mechanism={name: v / t_cycle for name, v in by_mechanism.items()},
    )


def failure_rate(parameters: dict) -> float:
    r"""
    Failure rate of the formula at one trial point of the minimizer.

    Parameters
    ----------
    parameters : dict
        Fixed parameters plus the physical values of the free parameters.

    Returns
    -------
    float
        Failure rate (1/ns); a NaN of the formula is replaced by the flat
        value of ``chencrafts.toolbox.nan_2_flat_val``, which a minimizer
        needs and no other caller does.
    """
    evaluation = evaluate_formula(parameters, breakdown=False, keep_sweep=True)
    return tb.nan_2_flat_val(evaluation["full_dict"], evaluation["rate"])


# the failure rate at the parameters of Ofek et al. is the unit of the target
RATE_REFERENCE = failure_rate(REFERENCE)


def target_func(parameters: dict) -> float:
    r"""
    Failure rate at one trial point in units of the rate at the parameters of Ofek et al.

    The minimizer needs a target of order one: the convergence thresholds of
    the scipy minimizers are absolute for a target below one, so a rate of
    order 1e-6 1/ns stops L-BFGS-B after its first step.

    Parameters
    ----------
    parameters : dict
        Fixed parameters plus the physical values of the free parameters.

    Returns
    -------
    float
        Failure rate divided by ``RATE_REFERENCE``.
    """
    return failure_rate(parameters) / RATE_REFERENCE


def build_optimization(free_ranges: dict | None = None, fixed: dict | None = None,
                       target=None, optimizer: str = "L-BFGS-B",
                       opt_options: dict | None = None) -> tb.Optimization:
    r"""
    Minimization of the target over free parameters scaled to [0, 1].

    Parameters
    ----------
    free_ranges : dict, optional
        Free parameter to ``[lower, upper]``; the four search ranges if None.
        A free parameter of ``search_ranges.py`` that is not listed is fixed.
    fixed : dict, optional
        Values of parameters that replace those of the reference parameters;
        a free parameter that is not searched needs its value here.
    target : callable, optional
        Function of the full parameter dictionary that is minimized;
        :func:`target_func` if None.
    optimizer : str
        Optimizer name of ``tb.Optimization``; ``"L-BFGS-B"`` is the local
        minimizer of the search.
    opt_options : dict, optional
        Options of the optimizer in the form of ``tb.Optimization``;
        ``LOCAL_OPTIONS`` if None.

    Returns
    -------
    tb.Optimization
        The minimization; every parameter that is not free is fixed at its
        reference value or at the value of ``fixed``.
    """
    free_ranges = RANGES if free_ranges is None else free_ranges
    # a free parameter is left out of the fixed ones: Optimization would
    # otherwise warn and delete it from the dictionary it was given
    fixed_variables = {name: value for name, value in (REFERENCE | (fixed or {})).items()
                       if name not in free_ranges}
    return tb.Optimization(
        fixed_variables=fixed_variables,
        free_variable_ranges=free_ranges,
        target_func=target_func if target is None else target,
        optimizer=optimizer,
        opt_options=LOCAL_OPTIONS if opt_options is None else opt_options,
    )


# the minimization of the search: all four free parameters
OPTIMIZATION = build_optimization()
