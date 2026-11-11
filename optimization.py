r"""Search for the parameters with the lowest failure rate of the analytical formula.

Minimizes the one-cycle failure rate :math:`(1-F)/t_{\text{cycle}}` of the
analytical formula over the four free parameters in ``search_ranges.py`` using
L-BFGS-B from the parameters of Ofek et al. and from random starting points.
Writes ``optimization.json``.

Usage: ``python optimization.py --processes 25``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scqubits as scq

import chencrafts.toolbox as tb

import cycle_target as ct
import device_model
import sweep_tools as st
from search_ranges import FREE_PARAMETER_MEANING

RESULTS_JSON = Path(__file__).resolve().parent / "optimization.json"

# number of runs from random starting points, and the seed of those points
RUN_COUNT = 50
MULTI_START_SEED = 20261005
# a run ends at a minimum if its final failure rate exceeds the lowest one
# among the runs with the same sign of the detuning by less than this (relative)
MINIMUM_TOLERANCE = 1e-3
STOPPED_EARLY = "stopped early"


def run_record(trajectory: tb.OptTraj) -> dict:
    r"""One run of the minimizer, in the form that the result file stores.

    Parameters
    ----------
    trajectory : tb.OptTraj
        Trajectory of the run; its targets are failure rates in units of
        ``cycle_target.RATE_REFERENCE``.

    Returns
    -------
    dict
        ``start`` and ``end`` (each: ``free_parameters``,
        ``failure_rate_per_ms``, and ``formula_infidelity``, the infidelity
        :math:`1-F` of one cycle from the formula there); ``detuning_GHz`` at
        the end; ``bounds_reached`` at the end; ``iterations``,
        ``evaluations`` (of the target by the minimizer), ``success`` and
        ``message`` of scipy; and ``trajectory``: the free parameters (in the
        order of ``parameter_names``) and the failure rate (1/ms) at the
        start and after every iteration.
    """
    scipy_result = trajectory.opt_result
    rates = trajectory.target_traj * ct.RATE_REFERENCE * ct.NS_PER_MS
    ends = {}
    for key, parameters, rate in (("start", trajectory.init_para, rates[0]),
                                  ("end", trajectory.final_para, rates[-1])):
        parameters = {name: float(parameters[name]) for name in ct.FREE_NAMES}
        # one evaluation of the formula for the infidelity of the cycle
        evaluation = ct.evaluate_formula(ct.REFERENCE | parameters, breakdown=False)
        ends[key] = dict(free_parameters=parameters, failure_rate_per_ms=float(rate),
                         formula_infidelity=evaluation["infidelity"])
    final = ends["end"]["free_parameters"]
    return ends | dict(
        detuning_GHz=st.detuning(final),
        bounds_reached=st.bounds_reached(final),
        iterations=int(scipy_result.nit), evaluations=int(scipy_result.nfev),
        success=bool(scipy_result.success), message=str(scipy_result.message),
        trajectory=dict(
            parameter_names=list(trajectory.para_name),
            free_parameters=np.asarray(trajectory.para_traj, float).tolist(),
            failure_rate_per_ms=rates.tolist()),
    )


def classify(runs: list[dict]) -> dict:
    r"""Find the two minima among the ends of the runs and say where each run ends.

    Parameters
    ----------
    runs : list of dict
        Outputs of :func:`run_record`; each receives the entry ``ended_at``
        (``sweep_tools.POSITIVE_DETUNING``, ``NEGATIVE_DETUNING``, or
        ``STOPPED_EARLY``) and ``excess_over_minimum`` (its final failure
        rate over the lowest one with the same sign of the detuning, minus
        one).

    Returns
    -------
    dict
        Label of each minimum to: ``run`` (index of the run with the lowest
        final failure rate on that side), ``free_parameters``,
        ``failure_rate_per_ms``, ``formula_infidelity``, ``detuning_GHz``,
        ``omega_01_GHz``, ``bounds_reached``, ``runs`` (indices of the runs
        that end there), ``largest_excess_among_these_runs``, and the range of
        each free parameter among these runs (``spread_of_free_parameters``).
    """
    minima = {}
    for label, sign in ((st.POSITIVE_DETUNING, 1), (st.NEGATIVE_DETUNING, -1)):
        side = [index for index, run in enumerate(runs) if np.sign(run["detuning_GHz"]) == sign]
        best = min(side, key=lambda index: runs[index]["end"]["failure_rate_per_ms"])
        lowest = runs[best]["end"]["failure_rate_per_ms"]
        for index in side:
            excess = runs[index]["end"]["failure_rate_per_ms"] / lowest - 1
            runs[index]["excess_over_minimum"] = float(excess)
            runs[index]["ended_at"] = label if excess < MINIMUM_TOLERANCE else STOPPED_EARLY
        here = [index for index in side if runs[index]["ended_at"] == label]
        final = runs[best]["end"]["free_parameters"]
        minima[label] = dict(
            run=best, free_parameters=final, failure_rate_per_ms=lowest,
            formula_infidelity=runs[best]["end"]["formula_infidelity"],
            detuning_GHz=runs[best]["detuning_GHz"],
            omega_01_GHz=st.transmon_transitions(final["EJ_GHz"])[0],
            bounds_reached=runs[best]["bounds_reached"], runs=here,
            largest_excess_among_these_runs=max(runs[index]["excess_over_minimum"]
                                                for index in here),
            spread_of_free_parameters={
                name: [min(runs[index]["end"]["free_parameters"][name] for index in here),
                       max(runs[index]["end"]["free_parameters"][name] for index in here)]
                for name in ct.FREE_NAMES})
    return minima


def main(processes: int) -> None:
    r"""Run both searches and write ``optimization.json``.

    Parameters
    ----------
    processes : int
        Worker processes over which the runs from random starts are spread.

    Returns
    -------
    None
    """
    environment = device_model.environment_record(processes)
    print(json.dumps(environment, indent=1), flush=True)
    optimization = ct.OPTIMIZATION

    # ---- one run from the parameters of Ofek et al. ----
    start = time.time()
    from_ofek = run_record(optimization.run(
        init_x={name: ct.REFERENCE[name] for name in ct.FREE_NAMES}, clip_init=True))
    print(f"run from the parameters of Ofek et al.: {from_ofek['end']}, "
          f"{time.time() - start:.0f} s", flush=True)

    # ---- runs from random starting points ----
    # the workers are processes of the standard library, one BLAS thread each
    scq.settings.MULTIPROC = "multiprocessing"
    scq.settings.MULTIPROC_BLAS_THREADS = 1
    # MultiOpt draws every start here, in this process, before the runs are
    # spread, so the seed fixes the starts and, with the deterministic
    # L-BFGS-B, the whole search
    np.random.seed(MULTI_START_SEED)
    start = time.time()
    finished = tb.MultiOpt(optimization).run(RUN_COUNT, node_num=processes)
    seconds = time.time() - start
    # MultiOpt leaves out a run that raises an error; the indices of the runs
    # would then no longer be those of the starts
    if finished.length != RUN_COUNT:
        raise RuntimeError(f"{finished.length} of {RUN_COUNT} runs finished")
    runs = [run_record(trajectory) for trajectory in finished.traj_list]
    minima = classify(runs)

    # where the run from the parameters of Ofek et al. ends
    for label, minimum in minima.items():
        same_side = np.sign(from_ofek["detuning_GHz"]) == np.sign(minimum["detuning_GHz"])
        excess = from_ofek["end"]["failure_rate_per_ms"] / minimum["failure_rate_per_ms"] - 1
        if same_side:
            from_ofek["excess_over_minimum"] = float(excess)
            from_ofek["ended_at"] = label if abs(excess) < MINIMUM_TOLERANCE else STOPPED_EARLY

    ended_at = [run["ended_at"] for run in runs]
    record = dict(
        description="L-BFGS-B minimization of the one-cycle failure rate from the analytical formula over the four free parameters, starting from the parameters of Ofek et al. and from random points.",
        environment=environment,
        settings=dict(
            search_ranges=ct.RANGES, free_parameters=FREE_PARAMETER_MEANING,
            minimizer="L-BFGS-B, free parameters scaled to [0, 1] over their ranges, target "
                      "in units of the failure rate at the parameters of Ofek et al.",
            minimizer_options=ct.LOCAL_OPTIONS,
            run_count=RUN_COUNT, seed_of_the_random_starts=MULTI_START_SEED,
            relative_tolerance_of_a_run_ending_at_a_minimum=MINIMUM_TOLERANCE,
            seconds_of_the_runs_from_random_starts=seconds),
        parameters_of_Ofek_et_al=dict(
            free_parameters={name: float(ct.REFERENCE[name]) for name in ct.FREE_NAMES},
            failure_rate_per_ms=ct.RATE_REFERENCE * ct.NS_PER_MS),
        minima=minima,
        summary=dict(
            runs_ending_at={label: ended_at.count(label)
                            for label in (st.NEGATIVE_DETUNING, st.POSITIVE_DETUNING,
                                          STOPPED_EARLY)},
            runs_stopped_early=[index for index, label in enumerate(ended_at)
                                if label == STOPPED_EARLY],
            largest_formula_infidelity_at_a_start=max(
                run["start"]["formula_infidelity"] for run in runs),
            largest_formula_infidelity_at_the_end_of_a_run_that_reached_a_minimum=max(
                run["end"]["formula_infidelity"] for run in runs
                if run["ended_at"] != STOPPED_EARLY),
            failure_rate_of_Delta_positive_over_Delta_negative_minimum=(
                minima[st.POSITIVE_DETUNING]["failure_rate_per_ms"]
                / minima[st.NEGATIVE_DETUNING]["failure_rate_per_ms"]),
            run_from_the_parameters_of_Ofek_et_al_ended_at=from_ofek["ended_at"]),
        run_from_the_parameters_of_Ofek_et_al=from_ofek,
        runs=runs,
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["summary"], indent=1))
    print(json.dumps(minima, indent=1))
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processes", type=int, default=1,
                        help="worker processes of the runs from random starts")
    main(parser.parse_args().processes)
