r"""The trade-off that sets the dispersive shift at the Delta<0 minimum.

Sweeps the coupling :math:`g` (and hence the dispersive shift :math:`\chi_1`)
through the :math:`\Delta < 0` minimum of ``optimization.json`` and evaluates
the failure-rate contributions of the EA idling, the parity-mapping idling, and
the remaining steps from the analytical formula, the failure Markov chain, and
the actual QEC dynamics. Writes ``parameter-tradeoff.json``.

Usage: ``python parameter-tradeoff.py --processes 9``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import qec_aeb as bsq

import cycle_target as ct
import device_model
import sweep_tools as st

RESULTS_JSON = Path(__file__).resolve().parent / "parameter-tradeoff.json"
NEG = st.NEGATIVE_DETUNING
# rad/ns to MHz of ordinary frequency
MHZ = 1e3 / (2 * np.pi)

# the dispersive shift of the sweep in units of its value at the minimum; the
# coupling follows as g = g_min sqrt(factor), since chi_1 grows as g^2
SHIFT_FACTORS = np.linspace(0.3, 2.4, 43)
# the same for the two numerically computed models; 1.0 is the minimum itself
TREE_SHIFT_FACTORS = np.linspace(0.4, 2.0, 9)
# steps of the two named contributions; the remaining steps are "other"
IDLE_STEP, PM_STEP = 1, 3
CONTRIBUTIONS = ("idle", "parity-mapping idling", "other five steps", "all seven steps")
# relative step of the coupling in the derivatives of the terms of the formula
DERIVATIVE_STEP = 0.01
MODELS = (st.FORMULA, st.CHAIN, st.ACTUAL)


def contributions(by_step: list, t_cycle: float) -> dict:
    r"""Contributions of the idle, the parity-mapping idling, and the other steps.

    Parameters
    ----------
    by_step : list of float
        Entry of :math:`1-F` of each step, in the order of the steps.
    t_cycle : float
        Cycle duration (ns).

    Returns
    -------
    dict
        The names of ``CONTRIBUTIONS`` to the contribution to the failure
        rate (1/ms).
    """
    by_step = np.asarray(by_step, float) * st.NS_PER_MS / t_cycle
    idle, mapping = by_step[IDLE_STEP - 1], by_step[PM_STEP - 1]
    return dict(zip(CONTRIBUTIONS, map(float, (
        idle, mapping, by_step.sum() - idle - mapping, by_step.sum()))))


def with_shift_factor(minimum: dict, factor: float) -> dict:
    r"""The minimum with its coupling changed so that the dispersive shift scales by a factor.

    Parameters
    ----------
    minimum : dict
        Free parameters of the minimum.
    factor : float
        Dispersive shift in units of its value at the minimum.

    Returns
    -------
    dict
        The free parameters with ``g_GHz`` multiplied by ``sqrt(factor)``.
    """
    return minimum | {"g_GHz": float(minimum["g_GHz"] * np.sqrt(factor))}


def formula_sweep(minimum: dict) -> dict:
    r"""The analytical formula along the coupling through the minimum.

    Parameters
    ----------
    minimum : dict
        Free parameters of the Delta<0 minimum.

    Returns
    -------
    dict
        Lists along the couplings :math:`g_{\min}\sqrt{f}`, :math:`f` in
        ``SHIFT_FACTORS``: ``shift_factor``, ``g_GHz``, ``chi_1_MHz``
        (:math:`|\chi_1|/2\pi`), ``in_search_range`` (whether the coupling
        lies in the search range of the optimization), ``K_kHz``
        (:math:`K/2\pi`), ``t_parity_mapping_idling_ns``, and the four
        ``CONTRIBUTIONS`` (1/ms). A coupling at which the resonator is closer
        than ``MIN_DETUNING_OVER_COUPLING`` couplings to a transition is left
        out.
    """
    lower, upper = ct.RANGES["g_GHz"]
    rows = []
    for factor in SHIFT_FACTORS:
        free_parameters = with_shift_factor(minimum, factor)
        # the formula assumes the dispersive regime
        if not st.in_dispersive_regime(free_parameters):
            continue
        evaluation = ct.evaluate_formula(st.REFERENCE | free_parameters)
        ingredients = evaluation["ingredients"]
        rows.append(dict(
            shift_factor=float(factor), g_GHz=free_parameters["g_GHz"],
            chi_1_MHz=abs(ingredients["chi_1"]) * MHZ,
            in_search_range=bool(lower <= free_parameters["g_GHz"] <= upper),
            K_kHz=ingredients["K_s"] * MHZ * 1e3,
            t_parity_mapping_idling_ns=ingredients["t_pm"],
            **contributions([evaluation["by_step"][n] for n in sorted(evaluation["by_step"])],
                            evaluation["t_cycle"])))
    return {key: [row[key] for row in rows] for key in rows[0]}


def tree_point(factor: float, minimum: dict) -> dict:
    r"""The three computations of one cycle at one coupling of the coarse grid.

    Parameters
    ----------
    factor : float
        Dispersive shift in units of its value at the minimum.
    minimum : dict
        Free parameters of the Delta<0 minimum.

    Returns
    -------
    dict
        ``shift_factor``, ``g_GHz``, ``chi_1_MHz``, ``failure_rate_per_ms``
        and ``infidelity`` (over the three computations), ``contributions``
        (computation to the four ``CONTRIBUTIONS``, 1/ms), and
        ``self_Kerr_angle`` (:math:`|K|` times the idle duration, rad). One
        tree of one cycle, built in this process.
    """
    free_parameters = with_shift_factor(minimum, factor)
    record = st.three_model_record(free_parameters, num_cpus=1)
    return dict(
        shift_factor=float(factor), g_GHz=free_parameters["g_GHz"],
        chi_1_MHz=abs(record["ingredients"]["chi_1"]) * MHZ,
        failure_rate_per_ms=record["failure_rate_per_ms"], infidelity=record["infidelity"],
        contributions={model: contributions(record["infidelity_by_step"][model],
                                            record["t_cycle_ns"]) for model in MODELS},
        self_Kerr_angle=abs(record["ingredients"]["K_s"]) * free_parameters["T_W"])


def stationarity(minimum: dict) -> dict:
    r"""Which terms of the formula balance at the minimum when the coupling varies.

    Parameters
    ----------
    minimum : dict
        Free parameters of the minimum.

    Returns
    -------
    dict
        Parts of :math:`dr/d\ln|\chi_1|` (1/ms): ``by_term`` (name of the
        term of ``cat_code_budget`` to its part), ``groups`` (the self-Kerr
        term of the idle, the other terms of the idle, the parity-mapping
        idling, the other five steps, the cycle duration), and ``sum`` (zero
        at a minimum, up to the differences);
        ``idle_over_parity_mapping_idling`` (the rise of all terms of the
        idle over the fall of the parity-mapping idling) and
        ``self_Kerr_term_over_idle`` (the part of that rise that is the
        self-Kerr term).
    """
    sides = []
    for factor in (1 - DERIVATIVE_STEP, 1 + DERIVATIVE_STEP):
        evaluation = ct.evaluate_formula(
            st.REFERENCE | minimum | {"g_GHz": minimum["g_GHz"] * factor}, keep_sweep=True)
        ingredients = evaluation["full_dict"]
        sides.append(dict(
            terms={name: float(value) for name, value
                   in bsq.cat_code_budget("name", **ingredients).items()},
            chi=abs(float(ingredients["chi_1"])), t_cycle=evaluation["t_cycle"],
            rate=evaluation["rate"]))
    low, high = sides
    step = np.log(high["chi"] / low["chi"])
    t_cycle, rate = (low["t_cycle"] + high["t_cycle"]) / 2, (low["rate"] + high["rate"]) / 2
    by_term = {name: (high["terms"][name] - low["terms"][name]) / step / t_cycle * st.NS_PER_MS
               for name in low["terms"]}
    duration = float(-rate * np.log(high["t_cycle"] / low["t_cycle"]) / step * st.NS_PER_MS)
    # the step of the cycle to which each term of the formula belongs
    steps = {name: term.step for name, term in bsq.cat_code_budget.error_contribs.items()}
    kerr = by_term["idle_kerr_after_loss"]
    idle = sum(v for n, v in by_term.items() if steps[n] == IDLE_STEP)
    mapping = sum(v for n, v in by_term.items() if steps[n] == PM_STEP)
    groups = {
        "self-Kerr term of the idle": kerr,
        "other terms of the idle": idle - kerr,
        "parity-mapping idling": mapping,
        "other five steps": sum(v for n, v in by_term.items()
                                if steps[n] not in (IDLE_STEP, PM_STEP)),
        "cycle duration": duration}
    return dict(by_term=by_term, groups=groups, sum=float(sum(by_term.values()) + duration),
                idle_over_parity_mapping_idling=float(-idle / mapping),
                self_Kerr_term_over_idle=float(kerr / idle))


def parabola_minimum(x, y) -> dict:
    r"""Vertex of the parabola through the lowest value of a sweep and its two neighbours.

    Parameters
    ----------
    x, y : array-like
        Values of the swept quantity and of the failure rate, at least three.

    Returns
    -------
    dict
        ``x`` and ``y`` of the vertex, and ``bracketed``: whether the lowest
        value is not an end of the sweep and the parabola opens upward.
        Without it the vertex is an extrapolation and no minimum was found.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lowest = int(np.argmin(y))
    # the lowest value and its two neighbours; at an end, the three outermost
    middle = min(max(lowest, 1), len(x) - 2)
    a, b, c = np.polyfit(x[middle - 1:middle + 2], y[middle - 1:middle + 2], 2)
    vertex = -b / (2 * a)
    return dict(x=float(vertex), y=float(np.polyval([a, b, c], vertex)),
                bracketed=bool(0 < lowest < len(x) - 1 and a > 0))


def main(processes: int) -> None:
    r"""Compute the sweep and the stationarity and write ``parameter-tradeoff.json``.

    Parameters
    ----------
    processes : int
        Worker processes of the trees.

    Returns
    -------
    None
    """
    environment = device_model.environment_record(processes)
    print(json.dumps(environment, indent=1), flush=True)
    minimum = st.parameter_sets()[NEG]

    # ---- the formula along the coupling, and the balance at the minimum ----
    formula = formula_sweep(minimum)
    balance = stationarity(minimum)
    print("d rate / d ln|chi_1| at the minimum (1/ms): " + json.dumps(balance["groups"]),
          flush=True)

    # ---- the two numerically computed models on the coarse grid ----
    factors = [float(factor) for factor in TREE_SHIFT_FACTORS
               if st.in_dispersive_regime(with_shift_factor(minimum, factor))]
    start = time.time()
    # forked workers hold this module, whose name cannot be imported
    with multiprocessing.get_context("fork").Pool(min(processes, len(factors))) as pool:
        points = pool.starmap(tree_point, [(factor, minimum) for factor in factors], chunksize=1)
    seconds = time.time() - start
    for point in points:
        print(f"|chi_1|/2pi = {point['chi_1_MHz']:.3f} MHz: {point['failure_rate_per_ms']}",
              flush=True)

    # ---- summary metrics ----
    upper = ct.RANGES["g_GHz"][1]
    at_minimum = next(point for point in points if np.isclose(point["shift_factor"], 1.0))
    chi = [point["chi_1_MHz"] for point in points]
    lowest = {model: parabola_minimum(chi, [point["failure_rate_per_ms"][model]
                                            for point in points]) for model in MODELS}
    lowest[st.FORMULA + ", fine grid"] = parabola_minimum(formula["chi_1_MHz"],
                                                          formula["all seven steps"])
    numbers = dict(
        chi_1_MHz_at_the_minimum=at_minimum["chi_1_MHz"],
        failure_rate_per_ms_at_the_minimum=at_minimum["failure_rate_per_ms"],
        # dispersive shift (MHz) and failure rate (1/ms) of the lowest value
        # along the sweep, from a parabola through three points
        lowest_along_the_sweep={model: dict(chi_1_MHz=vertex["x"], failure_rate_per_ms=vertex["y"],
                                            bracketed=vertex["bracketed"])
                                for model, vertex in lowest.items()},
        chi_1_MHz_at_the_upper_bound_of_the_coupling=float(
            np.interp(upper, formula["g_GHz"], formula["chi_1_MHz"])),
        formula_over_markov_chain_minus_1={
            name: [point["contributions"][st.FORMULA][name]
                   / point["contributions"][st.CHAIN][name] - 1 for point in points]
            for name in CONTRIBUTIONS},
        markov_chain_over_actual_minus_1=[
            point["infidelity"][st.CHAIN] / point["infidelity"][st.ACTUAL] - 1
            for point in points],
        stationarity=balance)
    record = dict(
        description="Failure-rate contributions (1/ms) against the dispersive shift |chi_1|/2pi (MHz) around the Delta<0 minimum from the analytical formula, failure Markov chain, and actual QEC dynamics.",
        environment=environment,
        settings=dict(
            minimum=minimum, search_range_of_the_coupling_GHz=ct.RANGES["g_GHz"],
            shift_factors=SHIFT_FACTORS.tolist(), tree_shift_factors=factors,
            derivative_step=DERIVATIVE_STEP,
            min_detuning_over_coupling=st.MIN_DETUNING_OVER_COUPLING,
            seconds_of_the_trees=seconds),
        formula=formula,
        numerical=points,
        numbers=numbers,
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(numbers, indent=1))
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=4, help="worker processes of the trees")
    main(parser.parse_args().processes)
