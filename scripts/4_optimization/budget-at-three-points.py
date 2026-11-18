r"""Failure rate by error mechanism at the parameter sets of the example, and the table of these sets.

Evaluates the failure rate by step and by error mechanism from the analytical
formula, the failure Markov chain, and the actual QEC dynamics at the
parameters of Ofek et al. and at the two minima of ``optimization.json``.
Writes ``budget-at-three-points.json``.

Usage: ``python budget-at-three-points.py --processes 4`` (or ``--update-derived``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cat_code import device_model, optimization as opt
from cat_code.optimization import FREE_PARAMETER_MEANING

RESULTS_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "budget-at-three-points.json"

# The fixed parameters that the table lists: key of the reference parameters
# to its meaning with the unit
FIXED_PARAMETER_MEANING = {
    "EC_GHz": "charging energy of the ancilla (GHz)",
    "Q_s": "quality factor of the resonator",
    "Q_cap": "capacitive quality factor of the ancilla",
    "temp_s": "temperature of the bath of the resonator (K)",
    "temp_a": "temperature of the bath of the ancilla (K)",
    "tau_m": "duration of the measurement (ns)",
    "tau_FD": "duration of the feedback latency (ns)",
    "M_ge": "readout assignment error, ground state read as excited",
    "M_eg": "readout assignment error, excited state read as ground",
    "disp": "initial cat state displacement beta",
}
# mechanisms whose change from the parameters of Ofek et al. the text quotes
QUOTED_MECHANISMS = ("ancilla relaxation", "ancilla excitation", "ancilla dephasing",
                     "photon loss", "photon gain")


def set_record(free_parameters: dict, num_cpus: int) -> dict:
    r"""Formula, numerically computed failure rates, and table quantities of one set.

    Parameters
    ----------
    free_parameters : dict
        The four free parameters of the set.
    num_cpus : int
        Worker processes that build the dynamical maps.

    Returns
    -------
    dict
        The entries of ``cat_code.optimization.formula_record`` (among them
        ``rate_by_mechanism_per_ms`` and ``rate_by_step_per_ms`` of the
        formula); ``failure_rate_per_ms`` and ``infidelity`` replaced by
        dictionaries over the three computations; ``rate_by_step_per_ms``
        likewise; ``bounds_reached``; and ``derived`` (``cat_code.optimization.derived_quantities``).
    """
    # one dressed sweep feeds the formula, the tree, and the table
    evaluation = opt.evaluate_formula(opt.REFERENCE | free_parameters, keep_sweep=True)
    record = opt.formula_record(free_parameters, evaluation)
    numerical = opt.one_cycle_fidelities(evaluation["sweep"],
                                        evaluation["simulation_parameters"], num_cpus)
    t_cycle = evaluation["t_cycle"]
    record["infidelity"] = {opt.FORMULA: record["infidelity"]} | {
        name: numerical[name]["infidelity"] for name in (opt.CHAIN, opt.ACTUAL)}
    record["failure_rate_per_ms"] = {
        name: float(device_model.failure_rate_per_ms(value, t_cycle))
        for name, value in record["infidelity"].items()}
    record["rate_by_step_per_ms"] = {opt.FORMULA: record["rate_by_step_per_ms"]} | {
        name: [float(device_model.failure_rate_per_ms(value, t_cycle))
               for value in numerical[name]["by_step"]] for name in (opt.CHAIN, opt.ACTUAL)}
    record["bounds_reached"] = opt.bounds_reached(free_parameters)
    record["derived"] = opt.derived_quantities(evaluation["full_dict"])
    return record


def comparisons(sets: dict) -> dict:
    r"""Relative changes between each minimum of ``optimization.json`` and the parameters of Ofek et al.

    Parameters
    ----------
    sets : dict
        Label of the parameter set to :func:`set_record`.

    Returns
    -------
    dict
        For each minimum: ``reduction_of_the_failure_rate`` (one minus the
        ratio of the failure rates, for each of the three computations),
        ``idle_duration_ratio``, ``cycle_duration_ratio``,
        ``mechanism_ratio`` (contribution of each mechanism of
        ``QUOTED_MECHANISMS`` at the minimum over that at the parameters of
        Ofek et al., from the formula), and ``formula_over_markov_chain_minus_1``
        at the minimum. Also the last quantity at the parameters of Ofek et
        al.
    """
    ofek = sets[opt.OFEK]

    def excess(record: dict) -> float:
        r"""Relative excess of the formula's failure rate over that of the failure Markov chain."""
        rate = record["failure_rate_per_ms"]
        return rate[opt.FORMULA] / rate[opt.CHAIN] - 1

    result = {opt.OFEK: dict(formula_over_markov_chain_minus_1=excess(ofek))}
    for label in (opt.POSITIVE_DETUNING, opt.NEGATIVE_DETUNING):
        minimum = sets[label]
        result[label] = dict(
            reduction_of_the_failure_rate={
                name: 1 - minimum["failure_rate_per_ms"][name] / ofek["failure_rate_per_ms"][name]
                for name in (opt.FORMULA, opt.CHAIN, opt.ACTUAL)},
            idle_duration_ratio=minimum["derived"]["t_idle_us"] / ofek["derived"]["t_idle_us"],
            cycle_duration_ratio=minimum["t_cycle_ns"] / ofek["t_cycle_ns"],
            mechanism_ratio={
                name: (minimum["rate_by_mechanism_per_ms"][name]
                       / ofek["rate_by_mechanism_per_ms"][name])
                for name in QUOTED_MECHANISMS},
            formula_over_markov_chain_minus_1=excess(minimum))
    return result


def main(num_cpus: int) -> None:
    r"""Evaluate the parameter sets and write ``budget-at-three-points.json``.

    Parameters
    ----------
    num_cpus : int
        Worker processes that build the dynamical maps.

    Returns
    -------
    None
    """
    environment = device_model.environment_record(num_cpus)
    print(json.dumps(environment, indent=1), flush=True)
    sets = {}
    for label, free_parameters in opt.parameter_sets().items():
        sets[label] = set_record(free_parameters, num_cpus)
        print(f"{label}: failure rate (1/ms) {sets[label]['failure_rate_per_ms']}", flush=True)
    record = dict(
        description="Failure rate of one cat-code QEC cycle by error mechanism and by step at the parameters of Ofek et al. and the two minima of optimization.json.",
        environment=environment,
        settings=dict(
            search_ranges=opt.RANGES, free_parameters=FREE_PARAMETER_MEANING,
            fixed_parameters={name: dict(value=float(opt.REFERENCE[name]), meaning=meaning)
                              for name, meaning in FIXED_PARAMETER_MEANING.items()},
            units=("free parameters as in search_ranges; chi_1/2pi in MHz, chi_1'/2pi and "
                   "K/2pi in kHz; rates in 1/ms; ingredients in rad/ns, 1/ns, ns, GHz"),
            pulse_coherent_terms=list(opt.PULSE_COHERENT),
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION),
        parameter_sets=sets,
        comparisons=comparisons(sets),
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["comparisons"], indent=1))
    print(f"wrote {RESULTS_JSON}")


def update_derived() -> None:
    r"""Recompute the table quantities of every set and merge them into the JSON.

    Only the dressed sweep and the formula are evaluated; the numerically
    computed failure rates already in ``budget-at-three-points.json`` are kept.

    Returns
    -------
    None
    """
    # the table quantities need the dressed sweep only, not the trees
    record = json.loads(RESULTS_JSON.read_text())
    for label, free_parameters in opt.parameter_sets().items():
        evaluation = opt.evaluate_formula(opt.REFERENCE | free_parameters, keep_sweep=True)
        record["parameter_sets"][label]["derived"] = opt.derived_quantities(
            evaluation["full_dict"])
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"updated the key derived of every set in {RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=4,
                        help="worker processes that build the dynamical maps")
    parser.add_argument("--update-derived", action="store_true",
                        help="recompute only the table quantities (key derived) of every set")
    arguments = parser.parse_args()
    if arguments.update_derived:
        update_derived()
    else:
        main(arguments.processes)
