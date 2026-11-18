r"""Multi-start parameter optimization for the zero-flux fluxonium ancilla.

Runs the two-stage L-BFGS-B search over the six free parameters of the
zero-flux fluxonium ancilla (:data:`cat_code.fluxonium_model.FLUXONIUM_SEARCH_RANGES`):

1. **Stage 1 (without inductive loss, ``--stage1``):** Multi-start L-BFGS-B
   minimization of the one-cycle failure rate from the analytical formula
   without inductive loss from random starting points, grouping final points
   within ``CLUSTER_TOLERANCE = 0.02`` in scaled coordinates into distinct
   candidate local minima. Writes ``data/fluxonium-minima.json``.
2. **Stage 2 (with inductive loss, ``--stage2``):** Multi-start L-BFGS-B
   minimization of the one-cycle failure rate with inductive loss
   (:math:`Q_{\mathrm{ind}} = 500\times 10^6` at :math:`0.5\,\mathrm{GHz}`),
   starting from the candidate minima of ``data/fluxonium-minima.json`` and
   from random starting points, and evaluating the best minima with the
   failure Markov chain and the actual QEC dynamics. Writes
   ``data/fluxonium-optimum.json``.

Units: circuit energies and frequencies in GHz, idle duration ``T_W`` in ns,
failure rates in 1/ms.

Usage:
  ``python scripts/4_optimization/fluxonium-optimization.py --stage1 --random 50``
  ``python scripts/4_optimization/fluxonium-optimization.py --stage2 --from-minima 44 --random 50 --numerical 4``
"""

from __future__ import annotations

import argparse
import functools
import json
import time

import numpy as np
import scqubits as scq
from scqubits.utils.cpu_switch import get_map_method

import chencrafts.settings as cc_settings
import chencrafts.toolbox as tb

from cat_code import device_model, fluxonium_model, optimization as opt

cc_settings.PROGRESSBAR_DISABLED = True

MINIMA_JSON = fluxonium_model.MINIMA
OPTIMUM_JSON = fluxonium_model.OPTIMUM
RANGES = fluxonium_model.FLUXONIUM_SEARCH_RANGES
FREE_NAMES = list(RANGES)

# Largest difference of one scaled parameter between two points of one minimum
CLUSTER_TOLERANCE = 0.02
# Seed of the random starting points
MULTI_START_SEED = 20261013


def evaluate(free_parameters: dict, inductive_loss: bool = True) -> dict:
    r"""Analytical formula of one QEC cycle of a zero-flux fluxonium.

    Parameters
    ----------
    free_parameters : dict
        Replacements of the parameters of :data:`fluxonium_model.REFERENCE_FLUXONIUM`.
    inductive_loss : bool
        Whether the inductor of the fluxonium is lossy.

    Returns
    -------
    dict
        ``infidelity`` (:math:`1-F` of one cycle), ``t_cycle_ns``,
        ``failure_rate_per_ms``, ``ancilla_transition_GHz``,
        ``chi_1_over_2pi_MHz``, ``kappa_a_phi_per_ns``, and ``full_dict``
        (every key of the dressed sweep).
    """
    sweep, _, parameters = fluxonium_model.build_sweep(
        free_parameters, inductive_loss=inductive_loss
    )
    full_dict = sweep.full_dict()
    infidelity = device_model.formula_infidelity(full_dict)
    t_cycle = float(full_dict["t_cycle"])
    return dict(
        infidelity=infidelity,
        t_cycle_ns=t_cycle,
        failure_rate_per_ms=float(device_model.failure_rate_per_ms(infidelity, t_cycle)),
        ancilla_transition_GHz=float(full_dict["omega_a_GHz"]),
        chi_1_over_2pi_MHz=float(full_dict["chi_1"]) / (2 * np.pi) * 1e3,
        kappa_a_phi_per_ns=float(parameters["kappa_a_phi"]),
        full_dict=full_dict,
    )


def summary(evaluation: dict) -> dict:
    r"""Scalar entries of an output of :func:`evaluate` (without ``full_dict``)."""
    return {key: value for key, value in evaluation.items() if key != "full_dict"}


@functools.lru_cache(maxsize=2)
def rate_reference(inductive_loss: bool = True) -> float:
    r"""Failure rate (1/ms) of the reference fluxonium, used as the unit of the target."""
    return evaluate({}, inductive_loss=inductive_loss)["failure_rate_per_ms"]


def make_optimization(inductive_loss: bool = True) -> tb.Optimization:
    r"""Build the L-BFGS-B optimization object over the six fluxonium parameters."""
    ref_rate = rate_reference(inductive_loss=inductive_loss)

    def target_func(parameters: dict) -> float:
        evaluation = evaluate(
            {name: parameters[name] for name in FREE_NAMES},
            inductive_loss=inductive_loss,
        )
        rate = tb.nan_2_flat_val(
            evaluation["full_dict"], evaluation["failure_rate_per_ms"]
        )
        return rate / ref_rate

    return tb.Optimization(
        fixed_variables={},
        free_variable_ranges=RANGES,
        target_func=target_func,
        optimizer="L-BFGS-B",
        opt_options=opt.LOCAL_OPTIONS,
    )


def group_minima(scaled: np.ndarray, tolerance: float = CLUSTER_TOLERANCE) -> np.ndarray:
    r"""Label sorted runs by the distinct minimum where they end.

    Two final points belong to one minimum if every free parameter, scaled to
    its search range, differs by less than ``tolerance``.

    Parameters
    ----------
    scaled : np.ndarray
        Final free parameters scaled to ``[0, 1]`` over their search ranges,
        one row per run, ordered from smallest final target to largest.
    tolerance : float
        Cluster tolerance in scaled coordinates.

    Returns
    -------
    np.ndarray
        Integer cluster label of each run; label 0 is the minimum of the best run.
    """
    labels = -np.ones(len(scaled), dtype=int)
    next_label = 0
    for i in range(len(scaled)):
        if labels[i] >= 0:
            continue
        close = np.abs(scaled - scaled[i]).max(axis=1) < tolerance
        labels[close & (labels < 0)] = next_label
        next_label += 1
    return labels


def numerical_record(free_parameters: dict, processes: int) -> dict:
    r"""Infidelity and failure rate of one QEC cycle from the tree, with inductive loss."""
    sweep, simulation_parameters, _ = fluxonium_model.build_sweep(
        free_parameters, inductive_loss=True
    )
    t_cycle = float(sweep["t_cycle"])
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=processes
    )
    infidelity = {
        name: float(value)
        for name, value in device_model.one_cycle_infidelities(graph).items()
    }
    return dict(
        infidelity=infidelity,
        failure_rate_per_ms={
            name: float(device_model.failure_rate_per_ms(value, t_cycle))
            for name, value in infidelity.items()
        },
    )


def run_stage1_without_inductive_loss(random_runs: int, processes: int) -> dict:
    r"""Stage 1: multi-start search without inductive loss, writing ``data/fluxonium-minima.json``."""
    scq.settings.MULTIPROC = "multiprocessing"
    scq.settings.MULTIPROC_BLAS_THREADS = 1
    opt = make_optimization(inductive_loss=False)

    np.random.seed(MULTI_START_SEED)
    start = time.time()
    finished = tb.MultiOpt(opt).run(random_runs, node_num=processes).sort_traj()
    runs = finished.traj_list
    print(
        f"Stage 1: {len(runs)} of {random_runs} runs finished in {time.time() - start:.0f} s",
        flush=True,
    )

    scaled = np.array([opt.physical_to_scaled(run.final_para) for run in runs])
    labels = group_minima(scaled)
    minima_list = []
    for label in range(int(labels.max()) + 1 if len(labels) else 0):
        indices = [idx for idx, lab in enumerate(labels) if lab == label]
        best_idx = indices[0]
        best = runs[best_idx]
        minima_list.append(
            dict(
                run=str(best_idx),
                run_count=len(indices),
                free_parameters={
                    name: float(best.final_para[name]) for name in FREE_NAMES
                },
            )
        )

    record = dict(
        description="Local minima of the analytical formula's one-cycle failure rate for the zero-flux fluxonium ancilla from multi-start L-BFGS-B optimization.",
        source=(
            f"Multi-start L-BFGS-B minimization of the analytical formula from "
            f"{random_runs} random starts without inductive loss "
            f"({len(runs)} valid runs grouped into {len(minima_list)} distinct minima)."
        ),
        grouping=(
            "Two final points belong to one minimum if every free parameter, scaled "
            "to its search range, differs by less than 0.02. A minimum is represented "
            "by its run with the smallest failure rate. The minima are ordered from "
            "the smallest failure rate of the analytical formula at the end of the "
            "search to the largest."
        ),
        units="circuit energies, resonator frequency, and coupling in GHz; idle duration T_W in ns",
        run_count=len(runs),
        free_parameters=FREE_NAMES,
        search_ranges=RANGES,
        reference_fluxonium=fluxonium_model.REFERENCE_FLUXONIUM,
        minima=minima_list,
    )
    MINIMA_JSON.write_text(json.dumps(record, indent=2))
    print(f"wrote {MINIMA_JSON}")
    return record


def run_stage2_with_inductive_loss(
    from_minima: int,
    random_runs: int,
    processes: int,
    numerical: int = 3,
) -> dict:
    r"""Stage 2: multi-start search with inductive loss, writing ``data/fluxonium-optimum.json``."""
    search = fluxonium_model.minima()
    opt = make_optimization(inductive_loss=True)

    scq.settings.MULTIPROC = "multiprocessing"
    scq.settings.MULTIPROC_BLAS_THREADS = 1

    def run_record(trajectory: tb.OptTraj, started_from: str) -> dict:
        end = {name: float(trajectory.final_para[name]) for name in FREE_NAMES}
        ev = evaluate(end, inductive_loss=True)
        return dict(
            started_from=started_from,
            free_parameters=end,
            failure_rate_per_ms=dict(
                analytical_formula=ev["failure_rate_per_ms"],
            ),
        )

    def _from_min(index: int) -> dict:
        minimum = search["minima"][index]
        traj = opt.run(init_x=minimum["free_parameters"], clip_init=True)
        return run_record(traj, str(minimum["run"]))

    warm = []
    n_warm = min(from_minima, len(search["minima"]))
    if n_warm > 0:
        start = time.time()
        map_method = get_map_method(processes) if processes > 1 else map
        warm = list(map_method(_from_min, range(n_warm)))
        print(f"Stage 2: {n_warm} runs from existing minima in {time.time() - start:.0f} s", flush=True)

    random_list = []
    if random_runs > 0:
        np.random.seed(MULTI_START_SEED)
        start = time.time()
        finished = tb.MultiOpt(opt).run(random_runs, node_num=processes)
        print(
            f"Stage 2: {finished.length} of {random_runs} random runs in {time.time() - start:.0f} s",
            flush=True,
        )
        random_list = [run_record(traj, "random start") for traj in finished.traj_list]

    all_runs = sorted(
        warm + random_list,
        key=lambda r: r["failure_rate_per_ms"]["analytical_formula"],
    )

    scq.settings.MULTIPROC = "pathos"
    for run in all_runs[:numerical]:
        num = numerical_record(run["free_parameters"], min(processes, 4))
        run["failure_rate_per_ms"].update(num["failure_rate_per_ms"])

    # Cluster runs in scaled coordinates to identify the kept optimum (g_GHz at upper bound)
    scaled = np.array([opt.physical_to_scaled(r["free_parameters"]) for r in all_runs])
    labels = group_minima(scaled)
    best_run = all_runs[0]
    best_label = labels[0]
    cluster_runs = [r for r, lab in zip(all_runs, labels) if lab == best_label]

    record = dict(
        description="Optimized circuit parameters and EA idling duration of the zero-flux fluxonium ancilla.",
        source=(
            f"Multi-start L-BFGS-B minimization of the analytical formula with inductive loss: "
            f"{n_warm} runs from the minima of fluxonium-minima.json and {random_runs} runs from random starts."
        ),
        units="circuit energies, resonator frequency, and coupling in GHz; idle duration T_W in ns; failure rates in 1/ms",
        search_ranges=RANGES,
        run_count=len(all_runs),
        free_parameters=best_run["free_parameters"],
        failure_rate_per_ms_in_the_search=best_run["failure_rate_per_ms"],
        runs_ending_at_this_minimum=cluster_runs,
    )
    OPTIMUM_JSON.write_text(json.dumps(record, indent=2))
    print(f"wrote {OPTIMUM_JSON}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="run Stage 1 (without inductive loss) and write data/fluxonium-minima.json",
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="run Stage 2 (with inductive loss) and write data/fluxonium-optimum.json",
    )
    parser.add_argument("--from-minima", type=int, default=44)
    parser.add_argument("--random", type=int, default=50)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--numerical", type=int, default=3)
    args = parser.parse_args()
    if not args.stage1 and not args.stage2:
        parser.error("specify --stage1 and/or --stage2")
    if args.stage1:
        run_stage1_without_inductive_loss(args.random, args.processes)
    if args.stage2:
        run_stage2_with_inductive_loss(
            args.from_minima, args.random, args.processes, args.numerical
        )
