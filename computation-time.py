r"""Time to compute the failure rate of one QEC cycle: numerically against the analytical formula.

Benchmarks the static joint diagonalization, the analytical formula, the actual
evolution tree, and the failure Markov chain over repeated single-core runs at
the reference and optimized parameter sets. Writes ``computation-time.json``.

Usage: ``python computation-time.py``.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model

RESULTS_JSON = Path(__file__).resolve().parent / "computation-time.json"

# timed repetitions at each parameter set, after one discarded warm-up
REPETITIONS = 9
# the formula takes far less than the resolution that the other parts need,
# so one sample of its time is the mean over this many consecutive calls
FORMULA_CALLS = 200
# worker processes that build the dynamical maps: one, so that the numerical
# computation and the formula use the same single core
NUM_CPUS = 1
# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def time_once(parameter_overrides: dict) -> dict:
    r"""One repetition: the three parts of the computation, each timed alone.

    Parameters
    ----------
    parameter_overrides : dict
        Parameters that replace their reference values, an entry of
        ``device_model.parameter_sets``.

    Returns
    -------
    dict
        Times in seconds: ``t_diagonalization_s``, ``t_formula_s`` (one
        evaluation), ``t_actual_tree_s``, ``t_markov_chain_s``; the cycle
        duration ``t_cycle_ns``; and ``infidelity`` of one cycle from the
        analytical formula, the failure Markov chain, and the actual QEC
        dynamics, which shows that the timed computations are the ones
        whose results the manuscript quotes.
    """
    # shared by both computations: the dressed sweep and its ingredients
    start = time.perf_counter()
    sweep, simulation_parameters, _ = device_model.build_sweep(parameter_overrides)
    ingredients = sweep.full_dict()
    t_diagonalization = time.perf_counter() - start

    # analytical formula: scalar arithmetic on the ingredients
    start = time.perf_counter()
    for _ in range(FORMULA_CALLS):
        formula = device_model.formula_infidelity(ingredients)
    t_formula = (time.perf_counter() - start) / FORMULA_CALLS

    # numerical computation, first part: the dynamical maps of the seven
    # steps from the master equation and the traversed tree of one cycle
    start = time.perf_counter()
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=NUM_CPUS)
    t_actual_tree = time.perf_counter() - start

    # second part: logical evolution tree and failure Markov chain (the
    # process fidelity of the actual QEC dynamics, read in the same call, is
    # one pass over the nodes of the tree)
    start = time.perf_counter()
    numerical = device_model.one_cycle_infidelities(graph)
    t_markov_chain = time.perf_counter() - start

    result = dict(
        t_diagonalization_s=t_diagonalization, t_formula_s=t_formula,
        t_actual_tree_s=t_actual_tree, t_markov_chain_s=t_markov_chain,
        t_cycle_ns=float(sweep["t_cycle"]),
        infidelity=dict(analytical_formula=formula, **numerical),
    )
    # the tree holds the maps of every step; free it before the next clock starts
    del graph, sweep, ingredients
    gc.collect()
    return result


def summary(samples) -> dict:
    r"""Median and spread of the samples of one timed quantity.

    Parameters
    ----------
    samples : array_like
        One value per repetition.

    Returns
    -------
    dict
        ``median``, ``min``, ``max``, the quartiles ``q25`` and ``q75``, and
        the ``samples`` themselves, in the unit of the input.
    """
    samples = np.asarray(samples, float)
    q25, median, q75 = np.percentile(samples, [25, 50, 75])
    return dict(median=float(median), min=float(samples.min()), max=float(samples.max()),
                q25=float(q25), q75=float(q75), samples=samples.tolist())


def benchmark(parameter_overrides: dict) -> dict:
    r"""Times and their ratios at one parameter set.

    Parameters
    ----------
    parameter_overrides : dict
        See :func:`time_once`.

    Returns
    -------
    dict
        ``parameter_overrides``; ``t_cycle_us``; ``failure_rate_per_ms`` of
        the three computations; the :func:`summary` of every time
        (``t_diagonalization_s``, ``t_formula_s``, ``t_actual_tree_s``,
        ``t_markov_chain_s``, and their sum ``t_numerical_s`` of the last
        two) and of the two ratios, formed within each repetition:
        ``ratio_numerical_over_formula_without_diagonalization`` and
        ``ratio_numerical_over_formula_with_diagonalization``. The ``median``
        of a ratio is the median of the per-repetition ratios, not the ratio
        of the median times.
    """
    # warm-up, discarded: it pays the deferred imports and the one-time fit
    # of the dephasing model, which no later evaluation pays again
    time_once(parameter_overrides)
    repetitions = [time_once(parameter_overrides) for _ in range(REPETITIONS)]

    times = {key: np.array([repetition[key] for repetition in repetitions])
             for key in ("t_diagonalization_s", "t_formula_s",
                         "t_actual_tree_s", "t_markov_chain_s")}
    times["t_numerical_s"] = times["t_actual_tree_s"] + times["t_markov_chain_s"]
    # ratios within a repetition, so that a slow moment of the node enters
    # numerator and denominator of the complete evaluation together
    ratios = dict(
        ratio_numerical_over_formula_without_diagonalization=(
            times["t_numerical_s"] / times["t_formula_s"]),
        ratio_numerical_over_formula_with_diagonalization=(
            (times["t_diagonalization_s"] + times["t_numerical_s"])
            / (times["t_diagonalization_s"] + times["t_formula_s"])),
    )
    # the results do not depend on the repetition; the last one is recorded
    last = repetitions[-1]
    return dict(
        parameter_overrides=parameter_overrides,
        t_cycle_us=last["t_cycle_ns"] / 1e3,
        failure_rate_per_ms={
            name: float(device_model.failure_rate_per_ms(value, last["t_cycle_ns"]))
            for name, value in last["infidelity"].items()},
        **{key: summary(values) for key, values in (times | ratios).items()},
    )


if __name__ == "__main__":
    record = dict(
        description="Single-core wall-clock times (seconds) of the static diagonalization, the analytical formula, and the numerical cycle evaluation.",
        environment=device_model.environment_record(NUM_CPUS),
        settings=dict(
            repetitions=REPETITIONS,
            warm_up_repetitions_discarded=1,
            formula_calls_per_sample=FORMULA_CALLS,
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
        ),
        parameter_sets={},
    )
    print(json.dumps(record["environment"], indent=1), flush=True)
    for label, parameter_overrides in device_model.parameter_sets().items():
        result = record["parameter_sets"][label] = benchmark(parameter_overrides)
        print(f"{label}: median times (s): diagonalization "
              f"{result['t_diagonalization_s']['median']:.3f}, formula "
              f"{result['t_formula_s']['median']:.2e}, numerical "
              f"{result['t_numerical_s']['median']:.2f}; ratio without / with the "
              f"diagonalization: "
              f"{result['ratio_numerical_over_formula_without_diagonalization']['median']:.3g}"
              f" / {result['ratio_numerical_over_formula_with_diagonalization']['median']:.3g}",
              flush=True)
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")
