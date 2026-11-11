r"""Failure rate of the analytical formula with some parameters held at the device values.

Minimizes the one-cycle failure rate over subsets of the four free parameters
while holding the remaining parameters at the values of Ofek et al., and
evaluates the optimum error-accumulation idling duration on a grid. Writes
``idle-attribution.json``.

Usage: ``python idle-attribution.py`` (or ``--add-idle-limit``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qec_aeb.batched_sweeps_recipe import cat_kerr_rotation_infidelity

import budget_tools as bt
import cycle_target as ct
import device_model
import sweep_tools as st

RESULTS_JSON = Path(__file__).resolve().parent / "idle-attribution.json"

# seed of the random starting points of the search at the idle duration of the device
SEED = 1
# number of random starting points of that search, in addition to the two named ones
RANDOM_STARTS = 3
# the three circuit parameters; the idle duration is the fourth free parameter
CIRCUIT = ["E_osc_GHz", "g_GHz", "EJ_GHz"]
# grid of EA idling durations (us) of the key idle_limit: from below the
# optimum of the device to beyond that of the Delta<0 minimum
T_EA_GRID_US = np.arange(5.0, 120.0 + 0.125, 0.25)
# name of the sum of the entries of steps 2 to 7 next to the EA-idling terms
STEPS_2_TO_7 = "steps_2_to_7"


def rate(free_parameters: dict) -> float:
    r"""Failure rate of the analytical formula at one parameter set.

    Parameters
    ----------
    free_parameters : dict
        The four free parameters (GHz, GHz, GHz, ns).

    Returns
    -------
    float
        Failure rate of one QEC cycle (1/ms).
    """
    return float(ct.evaluate_formula(ct.REFERENCE | free_parameters, breakdown=False)["rate"]
                 * ct.NS_PER_MS)


def search(starts: list[dict], searched: list[str], fixed: dict) -> dict:
    r"""Lowest failure rate of local minimizations from several starting points.

    Parameters
    ----------
    starts : list of dict
        Starting values of the searched parameters, one dictionary per run.
    searched : list of str
        Parameters that the minimization varies.
    fixed : dict
        Values of the free parameters that stay fixed.

    Returns
    -------
    dict
        ``held_fixed`` (the fixed parameters); ``free_parameters`` and
        ``failure_rate_per_ms`` of the lowest end; ``ends``, the failure rate
        of the end of every run, in the order of ``starts``.
    """
    ends = []
    for start in starts:
        minimum, _ = st.minimize_formula(start, searched, fixed)
        ends.append((rate(minimum), minimum))
    best_rate, best = min(ends, key=lambda end: end[0])
    return dict(held_fixed=fixed, free_parameters=best, failure_rate_per_ms=best_rate,
                ends=[end[0] for end in ends])


def main() -> dict:
    r"""Run the four searches and the two reference evaluations.

    Returns
    -------
    dict
        The record that is written to ``idle-attribution.json``.
    """
    sets = st.parameter_sets()
    device = sets[st.OFEK]
    optimized = sets[st.NEGATIVE_DETUNING]
    rng = np.random.default_rng(SEED)

    # reference values: the device and the optimized parameters of optimization.json
    device_rate = rate(device)
    optimized_rate = rate(optimized)

    # idle duration alone, at the other three parameters of the device; two
    # starting durations on either side of the optimum
    idle_only = search([dict(T_W=18e3), dict(T_W=40e3)], ["T_W"],
                       {name: device[name] for name in CIRCUIT})

    # coupling and idle duration, at the resonator frequency and the Josephson
    # energy of the device
    coupling_idle = search(
        [dict(g_GHz=device["g_GHz"], T_W=18e3), dict(g_GHz=0.2, T_W=40e3)],
        ["g_GHz", "T_W"], {name: device[name] for name in ("E_osc_GHz", "EJ_GHz")})

    # the three circuit parameters at the idle duration of the device. Starts:
    # the device, the optimized circuit parameters, and random points of the
    # search ranges; the random points find the Delta>0 branch as well
    ranges = np.array([ct.RANGES[name] for name in CIRCUIT])
    starts = [{name: device[name] for name in CIRCUIT}, {name: optimized[name] for name in CIRCUIT}]
    for _ in range(RANDOM_STARTS):
        point = ranges[:, 0] + rng.uniform(0, 1, len(CIRCUIT)) * (ranges[:, 1] - ranges[:, 0])
        starts.append(dict(zip(CIRCUIT, point.tolist())))
    circuit_at_device_idle = search(starts, CIRCUIT, {"T_W": device["T_W"]})

    def reduction(failure_rate: float) -> float:
        r"""Relative reduction of the failure rate against the device.

        Parameters
        ----------
        failure_rate : float
            Failure rate (1/ms).

        Returns
        -------
        float
            One minus the failure rate over that of the device.
        """
        return 1 - failure_rate / device_rate

    reduced = dict(idle_only=idle_only, coupling_idle=coupling_idle,
                   circuit_at_device_idle=circuit_at_device_idle)
    omega = {name: st.transmon_transitions(case["free_parameters"]["EJ_GHz"])[0]
             for name, case in reduced.items() if "EJ_GHz" in case["free_parameters"]}
    return dict(
        description="Failure rate of one QEC cycle (1/ms) from the analytical formula when optimizing subsets of the four free parameters with the others held at the values of Ofek et al.",
        environment=device_model.environment_record(1),
        settings=dict(seed=SEED, random_starts=RANDOM_STARTS, search_ranges=ct.RANGES),
        device=dict(free_parameters=device, failure_rate_per_ms=device_rate),
        optimized=dict(free_parameters=optimized, failure_rate_per_ms=optimized_rate),
        searches=reduced,
        ancilla_transition_GHz=dict(
            device=st.transmon_transitions(device["EJ_GHz"])[0],
            optimized=st.transmon_transitions(optimized["EJ_GHz"])[0], **omega),
        quoted=dict(
            reduction_idle_only=reduction(idle_only["failure_rate_per_ms"]),
            reduction_coupling_and_idle=reduction(coupling_idle["failure_rate_per_ms"]),
            reduction_circuit_parameters_at_device_idle=reduction(
                circuit_at_device_idle["failure_rate_per_ms"]),
            reduction_full_optimization=reduction(optimized_rate),
            idle_duration_of_the_idle_only_optimum_us=idle_only["free_parameters"]["T_W"] / 1e3,
            idle_duration_ratio_optimized_over_device=optimized["T_W"] / device["T_W"],
        ),
        idle_limit=idle_limit(),
    )


def cycle_entries(full_dict: dict, K_s: float | None = None,
                  jump_a: float | None = None) -> dict:
    r"""Entries of one QEC cycle: the EA-idling terms by name and steps 2 to 7 together.

    Parameters
    ----------
    full_dict : dict
        Every key of the dressed sweep at one parameter set,
        ``sweep.full_dict()``.
    K_s : float, optional
        Resonator self-Kerr coefficient (rad/ns) that replaces that of the
        sweep in the Kerr term; that of the sweep if None.
    jump_a : float, optional
        Photon-loss rate (1/ns) that replaces that of the sweep in every term
        of the formula; that of the sweep if None.

    Returns
    -------
    dict
        Each term of the EA idling (``budget_tools.TERM_KEYS`` names) and
        ``STEPS_2_TO_7`` to its contribution to the infidelity of one cycle, and
        ``t_cycle``, the cycle duration (ns).
    """
    ingredients = dict(full_dict)
    if jump_a is not None:
        ingredients["jump_a"] = jump_a
    # the infidelity of the Kerr rotation is the one ingredient that the sweep
    # computes from K and the photon-loss rate; with the same arguments as
    # chencrafts' _sweep_cat_kerr_rotation_infidelity: odd error space, cat
    # displacement after the shrinkage by the no-jump evolution of the idle
    if K_s is not None or jump_a is not None:
        t_ea = float(ingredients["T_W"])
        amp_sq_after = (abs(complex(ingredients["disp"])) ** 2
                        * np.exp(-float(ingredients["jump_a"]) * t_ea))
        kerr = float(ingredients["K_s"]) if K_s is None else K_s
        ingredients["cat_kerr_rotation_infidelity"] = cat_kerr_rotation_infidelity(
            amp_sq_after, 1, kerr * t_ea)

    # the terms of the formula with the names of the manuscript; step 1 is the EA idling
    terms = bt.formula_terms(ingredients)
    entries = {term["term"]: term["value"] for term in terms if term["step"] == 1}
    entries[STEPS_2_TO_7] = sum(term["value"] for term in terms if term["step"] != 1)
    entries["t_cycle"] = float(ingredients["t_cycle"])
    return entries


def rate_of_entries(entries: dict, dropped: tuple = (), steps_2_to_7: float | None = None) -> float:
    r"""Failure rate from the entries of one cycle, with terms left out or replaced.

    Parameters
    ----------
    entries : dict
        Output of :func:`cycle_entries`.
    dropped : tuple of str
        EA-idling terms that are left out.
    steps_2_to_7 : float, optional
        Entry of steps 2 to 7 that replaces that of ``entries``.

    Returns
    -------
    float
        Failure rate of one QEC cycle (1/ms), on the cycle duration of ``entries``.
    """
    infidelity = sum(value for name, value in entries.items()
                     if name not in (STEPS_2_TO_7, "t_cycle") and name not in dropped)
    infidelity += entries[STEPS_2_TO_7] if steps_2_to_7 is None else steps_2_to_7
    return infidelity / entries["t_cycle"] * ct.NS_PER_MS


def grid_optimum(rows: list[dict], dropped: tuple = (),
                 steps_2_to_7: list[float] | None = None) -> dict:
    r"""EA idling duration of the lowest failure rate on ``T_EA_GRID_US``.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`cycle_entries` at every duration of the grid.
    dropped : tuple of str
        EA-idling terms that are left out.
    steps_2_to_7 : list of float, optional
        Entry of steps 2 to 7 at every duration of the grid that replaces that
        of ``rows``.

    Returns
    -------
    dict
        ``t_ea_us`` and ``failure_rate_per_ms`` at the optimum, and
        ``on_grid_edge``, whether the optimum is an end of the grid (then it
        is not a minimum).
    """
    rates = np.array([
        rate_of_entries(row, dropped, None if steps_2_to_7 is None else steps_2_to_7[i])
        for i, row in enumerate(rows)])
    index = int(np.argmin(rates))
    return dict(t_ea_us=float(T_EA_GRID_US[index]), failure_rate_per_ms=float(rates[index]),
                on_grid_edge=index in (0, len(rates) - 1))


def rate_derivatives(rows: list[dict], t_ea_us: float) -> dict:
    r"""Derivatives of the parts of the failure rate with respect to the EA idling duration.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`cycle_entries` at every duration of the grid.
    t_ea_us : float
        Duration (us) of the grid at which the derivatives are taken; not an
        end of the grid.

    Returns
    -------
    dict
        Each EA-idling term, ``STEPS_2_TO_7``, and ``total`` to the derivative
        of its contribution to the failure rate (entry over cycle duration),
        in 1/ms per us; central difference over the neighbouring grid points.
    """
    index = int(np.argmin(abs(T_EA_GRID_US - t_ea_us)))
    below, above = rows[index - 1], rows[index + 1]
    step_us = T_EA_GRID_US[index + 1] - T_EA_GRID_US[index - 1]
    # contribution of a part to the failure rate: its entry over the cycle
    # duration, which also grows with the EA idling duration
    names = [name for name in rows[index] if name != "t_cycle"]
    derivatives = {
        name: float((above[name] / above["t_cycle"] - below[name] / below["t_cycle"])
                    * ct.NS_PER_MS / step_us)
        for name in names}
    derivatives["total"] = float(
        (rate_of_entries(above) - rate_of_entries(below)) / step_us)
    return derivatives


def idle_limit() -> dict:
    r"""Optimum EA idling duration at two circuit sets, its slopes, and counterfactual optima.

    Returns
    -------
    dict
        The record of the key ``idle_limit`` of ``idle-attribution.json``;
        ``description`` says what each entry is.
    """
    sets = st.parameter_sets()
    device_label, minimum_label = st.OFEK, st.NEGATIVE_DETUNING
    circuits = {label: {name: sets[label][name] for name in CIRCUIT}
                for label in (device_label, minimum_label)}

    # K and the photon-loss rate of each set; both depend on the circuit only,
    # not on the EA idling duration
    dressed = {label: ct.evaluate_formula(ct.REFERENCE | sets[label],
                                          breakdown=False)["ingredients"]
               for label in circuits}
    K = {label: dressed[label]["K_s"] for label in circuits}
    loss = {label: dressed[label]["jump_a"] for label in circuits}

    # the formula variants evaluated at every duration of the grid: each set
    # with its own ingredients, and with K or the photon-loss rate of the other set
    variants = {
        device_label: dict(own={}, minimum_K=dict(K_s=K[minimum_label]),
                           minimum_K_and_loss=dict(K_s=K[minimum_label],
                                                   jump_a=loss[minimum_label])),
        minimum_label: dict(own={}, device_K=dict(K_s=K[device_label])),
    }
    rows = {label: {name: [] for name in variants[label]} for label in circuits}
    for label, circuit in circuits.items():
        for t_ea_us in T_EA_GRID_US:
            evaluation = ct.evaluate_formula(ct.REFERENCE | circuit | {"T_W": t_ea_us * 1e3},
                                             breakdown=False, keep_sweep=True)
            full_dict = evaluation["full_dict"]
            # the terms must add up to the failure rate of the formula, and the
            # recomputed Kerr rotation must reproduce that of the sweep
            own = cycle_entries(full_dict)
            if not np.isclose(rate_of_entries(own), evaluation["rate"] * ct.NS_PER_MS,
                              rtol=1e-10, atol=0):
                raise ValueError(f"terms do not add up to the rate at {label}, {t_ea_us} us")
            if not np.isclose(rate_of_entries(cycle_entries(full_dict, K_s=K[label])),
                              rate_of_entries(own), rtol=1e-10, atol=0):
                raise ValueError(f"recomputed Kerr term differs from the sweep at {label}")
            for name, replaced in variants[label].items():
                rows[label][name].append(own if not replaced
                                         else cycle_entries(full_dict, **replaced))

    device_rows, minimum_rows = rows[device_label], rows[minimum_label]
    optima = {label: grid_optimum(rows[label]["own"]) for label in circuits}
    minimum_steps_2_to_7 = [row[STEPS_2_TO_7] for row in minimum_rows["own"]]
    counterfactual = dict(
        device_without_kerr_term=grid_optimum(device_rows["own"], dropped=("kerr_term",)),
        device_without_two_or_more_photon_losses=grid_optimum(
            device_rows["own"], dropped=("two_or_more_photon_losses",)),
        device_with_minimum_K=grid_optimum(device_rows["minimum_K"]),
        device_with_minimum_K_and_photon_loss_rate=grid_optimum(device_rows["minimum_K_and_loss"]),
        device_with_minimum_K_photon_loss_rate_and_steps_2_to_7=grid_optimum(
            device_rows["minimum_K_and_loss"], steps_2_to_7=minimum_steps_2_to_7),
        minimum_with_device_K=grid_optimum(minimum_rows["device_K"]),
    )
    return dict(
        description="Optimum EA idling duration, failure-rate derivatives, and counterfactual optima on t_ea_grid_us from the analytical formula.",
        environment=device_model.environment_record(1),
        t_ea_grid_us=dict(first=float(T_EA_GRID_US[0]), last=float(T_EA_GRID_US[-1]),
                          step=float(T_EA_GRID_US[1] - T_EA_GRID_US[0])),
        circuit_parameters=circuits,
        dressed={label: dict(K_rad_per_ns=K[label],
                             K_over_2pi_kHz=abs(K[label]) / (2 * np.pi) * 1e6,
                             photon_loss_rate_per_ms=loss[label] * ct.NS_PER_MS)
                 for label in circuits},
        optimum=optima,
        derivatives_at_device_optimum=dict(
            t_ea_us=optima[device_label]["t_ea_us"],
            per_ms_per_us=rate_derivatives(device_rows["own"], optima[device_label]["t_ea_us"])),
        counterfactual_optima=counterfactual,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--add-idle-limit", action="store_true",
                        help="compute only the key idle_limit and add it to the JSON file")
    arguments = parser.parse_args()
    if arguments.add_idle_limit:
        # the searches are kept; only the key idle_limit is replaced
        record = json.loads(RESULTS_JSON.read_text())
        record["idle_limit"] = idle_limit()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps({key: value for key, value in record["idle_limit"].items()
                          if key not in ("description", "environment")}, indent=1))
        print(f"added the key idle_limit to {RESULTS_JSON}")
    else:
        record = main()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["quoted"], indent=1))
        print(f"wrote {RESULTS_JSON}")
