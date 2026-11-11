r"""Failure rate over the two mode frequencies, minimized over the coupling and the idle duration.

Evaluates the two-dimensional landscape of the analytical formula's failure rate
over the ancilla transition frequency and resonator frequency in coarse, dense,
recheck, and merge stages. Writes ``parameter-landscape.json``.

Usage: ``python parameter-landscape.py coarse|dense|recheck|merge ...``.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import cycle_target as ct
import device_model
import sweep_tools as st

FOLDER = Path(__file__).resolve().parent
RESULTS_JSON = FOLDER / "parameter-landscape.json"
PARTS = FOLDER / "parameter-landscape-parts"
NEG, POS, OFEK = st.NEGATIVE_DETUNING, st.POSITIVE_DETUNING, st.OFEK

# ---- grids and the band ----
GRID, COARSE_GRID = 100, 10
# a pair this close to the 0-1 or the 1-2 transition is left out (GHz)
BAND_GHZ = 0.6

# ---- minimization ----
# starting points of the coarse grid and of the recheck: fractions of the
# interval of the coupling, and idle durations (ns)
START_COUPLING_FRACTIONS = (0.0, 0.5, 1.0)
START_IDLE_NS = (15e3, 45e3)
# a pair whose failure rate exceeds the median of its neighbours by more than
# this fraction is rechecked with six starts
RECHECK_EXCESS = 0.02
# the searched parameters
SEARCHED = ["g_GHz", "T_W"]


# ----------------------------------------------------------------------------
# Pairs of frequencies
# ----------------------------------------------------------------------------

def josephson_energy(omega_01_GHz: float) -> float:
    r"""Josephson energy at which the bare transmon has a given 0-1 transition.

    Parameters
    ----------
    omega_01_GHz : float
        Target transition frequency (GHz), between the transitions at the two
        bounds of the search range of the Josephson energy.

    Returns
    -------
    float
        :math:`E_J/h` (GHz), at the charging energy of the reference parameters.
    """
    lower, upper = ct.RANGES["EJ_GHz"]
    # the ends of the frequency grid are the transitions at the bounds
    # themselves, so the bracket is widened by a rounding margin
    return float(brentq(lambda EJ: st.transmon_transitions(EJ)[0] - omega_01_GHz,
                        lower - 1e-6, upper + 1e-6))


def frequency_axes(grid: int) -> dict:
    r"""Uniform grid of the ancilla transition and the resonator frequency over the search ranges.

    Parameters
    ----------
    grid : int
        Number of points along each frequency.

    Returns
    -------
    dict
        ``omega_01_GHz`` (between the 0-1 transitions of the bare transmon at
        the two bounds of the Josephson energy), ``EJ_GHz`` (the Josephson
        energy of each), ``omega_12_GHz`` (the 1-2 transition at each), and
        ``omega_r_GHz`` (over the search range of the resonator frequency),
        as lists of ``grid`` values.
    """
    omega_01 = np.linspace(*(st.transmon_transitions(bound)[0] for bound in ct.RANGES["EJ_GHz"]),
                           grid)
    energies = [josephson_energy(float(transition)) for transition in omega_01]
    return dict(
        omega_01_GHz=omega_01.tolist(), EJ_GHz=energies,
        omega_12_GHz=[float(st.transmon_transitions(energy)[1]) for energy in energies],
        omega_r_GHz=np.linspace(*ct.RANGES["E_osc_GHz"], grid).tolist())


def frequency_pairs(grid: int) -> list[dict]:
    r"""Every pair of frequencies of a grid that lies outside the band.

    Parameters
    ----------
    grid : int
        Number of points along each frequency.

    Returns
    -------
    list of dict
        For each drawn pair: ``index`` (``row * grid + column``), ``row``
        (resonator frequency), ``column`` (ancilla transition frequency),
        ``omega_r_GHz``, ``omega_01_GHz``, ``EJ_GHz``, ``side`` (+1 for
        Delta>0, -1 for Delta<0), and ``coupling_range_GHz``.
    """
    axes = frequency_axes(grid)
    g_low, g_high = ct.RANGES["g_GHz"]
    pairs = []
    for row, column in itertools.product(range(grid), range(grid)):
        omega_r, omega_01 = axes["omega_r_GHz"][row], axes["omega_01_GHz"][column]
        smallest = min(abs(omega_r - omega_01), abs(omega_r - axes["omega_12_GHz"][column]))
        if smallest < BAND_GHZ:
            continue
        # outside the band the resonator is detuned from both transitions by
        # at least BAND_GHZ, so the cap of the coupling lies above its lower bound
        upper = min(g_high, smallest / st.MIN_DETUNING_OVER_COUPLING)
        pairs.append(dict(index=row * grid + column, row=row, column=column,
                          omega_r_GHz=omega_r, omega_01_GHz=omega_01,
                          EJ_GHz=axes["EJ_GHz"][column], side=int(np.sign(omega_r - omega_01)),
                          coupling_range_GHz=[g_low, upper]))
    return pairs


# ----------------------------------------------------------------------------
# Minimization at one pair
# ----------------------------------------------------------------------------

def minimize_pair(pair: dict, starts: list[dict]) -> dict:
    r"""Minimize the failure rate of the formula over the coupling and the idle duration.

    Parameters
    ----------
    pair : dict
        One entry of :func:`frequency_pairs`.
    starts : list of dict
        ``g_GHz`` and ``T_W`` of each start; a start outside the range of
        the pair begins at the nearest bound.

    Returns
    -------
    dict
        ``index``; ``failure_rate_per_ms``, ``g_GHz``, ``t_idle_ns`` at the
        lowest end; ``starts`` (their number); ``ends_per_ms`` (the failure
        rate at every end); ``seconds``.
    """
    begin = time.time()
    frequencies = {"E_osc_GHz": pair["omega_r_GHz"], "EJ_GHz": pair["EJ_GHz"]}
    ends = []
    for start in starts:
        end, _ = st.minimize_formula(start, SEARCHED, frequencies,
                                     ranges={"g_GHz": pair["coupling_range_GHz"]})
        rate = ct.evaluate_formula(st.REFERENCE | end, breakdown=False)["rate"] * st.NS_PER_MS
        ends.append((float(rate), end))
    rate, best = min(ends, key=lambda item: item[0])
    return dict(index=pair["index"], failure_rate_per_ms=rate, g_GHz=best["g_GHz"],
                t_idle_ns=best["T_W"], starts=len(starts),
                ends_per_ms=[item[0] for item in ends], seconds=time.time() - begin)


def six_starts(pair: dict) -> list[dict]:
    r"""The six starts of the coarse grid and of the recheck.

    Parameters
    ----------
    pair : dict
        One entry of :func:`frequency_pairs`.

    Returns
    -------
    list of dict
        Three couplings across the range of the pair times two idle durations.
    """
    g_low, upper = pair["coupling_range_GHz"]
    return [{"g_GHz": g_low + fraction * (upper - g_low), "T_W": t_idle}
            for fraction, t_idle in itertools.product(START_COUPLING_FRACTIONS, START_IDLE_NS)]


def nearest_coarse_start(pair: dict, coarse: list[dict]) -> dict:
    r"""The minimum of the nearest coarse pair on the same side of the transitions.

    Parameters
    ----------
    pair : dict
        One entry of :func:`frequency_pairs` of the dense grid.
    coarse : list of dict
        Results of the coarse stage, each joined with its pair.

    Returns
    -------
    dict
        ``g_GHz`` and ``T_W`` of that minimum.
    """
    candidates = [entry for entry in coarse if entry["side"] == pair["side"]]
    # distance in the plane of the two frequencies (GHz)
    distance = [np.hypot(entry["omega_r_GHz"] - pair["omega_r_GHz"],
                         entry["omega_01_GHz"] - pair["omega_01_GHz"]) for entry in candidates]
    nearest = candidates[int(np.argmin(distance))]
    return {"g_GHz": nearest["g_GHz"], "T_W": nearest["t_idle_ns"]}


# ----------------------------------------------------------------------------
# Parts written by the tasks
# ----------------------------------------------------------------------------

def read_parts(parts: Path, pattern: str) -> list[dict]:
    r"""Results of every part file that matches a pattern.

    Parameters
    ----------
    parts : Path
        Folder of the part files.
    pattern : str
        Glob pattern of the file names.

    Returns
    -------
    list of dict
        The ``results`` of all matching files, concatenated.
    """
    return [result for path in sorted(glob.glob(str(parts / pattern)))
            for result in json.loads(Path(path).read_text())["results"]]


def best_values(dense: list[dict], rechecks: list[dict]) -> dict:
    r"""The lowest result at every pair of the dense grid.

    Parameters
    ----------
    dense, rechecks : list of dict
        Results of the dense stage and of the recheck rounds.

    Returns
    -------
    dict
        Index of the pair to its lowest result.
    """
    best = {}
    for result in dense + rechecks:
        index = result["index"]
        if index not in best or result["failure_rate_per_ms"] < best[index]["failure_rate_per_ms"]:
            best[index] = result
    return best


def flagged(best: dict) -> list[int]:
    r"""Pairs whose failure rate exceeds the median of their drawn neighbours by more than ``RECHECK_EXCESS``.

    Parameters
    ----------
    best : dict
        Output of :func:`best_values`.

    Returns
    -------
    list of int
        Indices of the flagged pairs; a pair needs at least three drawn
        neighbours among its eight to be judged.
    """
    rate = np.full((GRID, GRID), np.nan)
    for index, result in best.items():
        rate[divmod(index, GRID)] = result["failure_rate_per_ms"]
    found = []
    for row, column in zip(*np.nonzero(~np.isnan(rate))):
        block = rate[max(row - 1, 0):row + 2, max(column - 1, 0):column + 2].copy()
        # the pair itself is not its own neighbour
        block[min(row, 1), min(column, 1)] = np.nan
        neighbours = block[~np.isnan(block)]
        if len(neighbours) >= 3 and rate[row, column] > (1 + RECHECK_EXCESS) * np.median(neighbours):
            found.append(int(row * GRID + column))
    return found


def pairs_of_task(stage: str, parts: Path, round_: int) -> tuple[list[dict], callable]:
    r"""The pairs of a stage and how each is started.

    Parameters
    ----------
    stage : str
        ``coarse``, ``dense``, or ``recheck``.
    parts : Path
        Folder of the part files; the dense stage reads the coarse results
        and a recheck the results before its round.
    round_ : int
        Round of the recheck.

    Returns
    -------
    pairs : list of dict
        Entries of :func:`frequency_pairs` that the stage minimizes, in a
        fixed order.
    starts : callable
        Function of a pair that returns its starts.
    """
    if stage == "coarse":
        return frequency_pairs(COARSE_GRID), six_starts
    pairs = frequency_pairs(GRID)
    if stage == "dense":
        by_index = {pair["index"]: pair for pair in frequency_pairs(COARSE_GRID)}
        coarse = [by_index[result["index"]] | result
                  for result in read_parts(parts, "coarse-*.json")]
        return pairs, lambda pair: [nearest_coarse_start(pair, coarse)]
    # recheck: the flagged pairs that no earlier round has rechecked; the files
    # of this round and of later ones are not read, so that every task of the
    # round sees the same pairs
    rechecks = [result for earlier in range(1, round_)
                for result in read_parts(parts, f"recheck{earlier}-*.json")]
    best = best_values(read_parts(parts, "dense-*.json"), rechecks)
    done = {result["index"] for result in rechecks}
    pending = set(flagged(best)) - done
    return [pair for pair in pairs if pair["index"] in pending], six_starts


def run_task(stage: str, task: int, tasks: int, parts: Path, round_: int) -> None:
    r"""Minimize the pairs of one task of a stage and write its part file.

    Parameters
    ----------
    stage : str
        ``coarse``, ``dense``, or ``recheck``.
    task, tasks : int
        Index of the task and number of tasks of the stage; the pairs of the
        stage are split into ``tasks`` consecutive blocks.
    parts : Path
        Folder of the part files.
    round_ : int
        Round of the recheck (ignored by the other stages).

    Returns
    -------
    None
    """
    begin = time.time()
    pairs, starts = pairs_of_task(stage, parts, round_)
    block = np.array_split(np.arange(len(pairs)), tasks)[task]
    results = []
    for position in block:
        pair = pairs[position]
        results.append(minimize_pair(pair, starts(pair)))
        print(f"pair {pair['index']}: {results[-1]['failure_rate_per_ms']:.4f} /ms "
              f"after {results[-1]['seconds']:.0f} s", flush=True)
    name = f"recheck{round_}" if stage == "recheck" else stage
    parts.mkdir(exist_ok=True)
    (parts / f"{name}-{task:04d}.json").write_text(json.dumps(dict(
        stage=stage, round=round_, task=task, tasks=tasks, results=results,
        environment=device_model.environment_record(1), seconds=time.time() - begin)))
    print(f"{len(results)} pairs after {time.time() - begin:.0f} s", flush=True)


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------

def merge(parts: Path) -> None:
    r"""Collect the parts, keep the lowest value at every pair, and write ``parameter-landscape.json``.

    Parameters
    ----------
    parts : Path
        Folder of the part files.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a drawn pair of the dense grid has no result.
    """
    dense, rechecks = read_parts(parts, "dense-*.json"), read_parts(parts, "recheck*-*.json")
    best = best_values(dense, rechecks)
    pairs = frequency_pairs(GRID)
    missing = [pair["index"] for pair in pairs if pair["index"] not in best]
    if missing:
        raise RuntimeError(f"{len(missing)} pairs have no result, the first {missing[:5]}")
    pending = sorted(set(flagged(best)) - {result["index"] for result in rechecks})
    if pending:
        print(f"{len(pending)} flagged pairs are not rechecked: run the next round", flush=True)

    # grids in the index order [omega_r][omega_01], None where a pair is left out
    grids = {key: [[None] * GRID for _ in range(GRID)]
             for key in ("failure_rate_per_ms", "g_GHz", "t_idle_ns", "starts")}
    for index, result in best.items():
        row, column = divmod(index, GRID)
        for key in grids:
            grids[key][row][column] = result[key]
    # the lowest pair on each side of the transitions
    lowest = {}
    for label, side in (("Delta<0", -1), ("Delta>0", 1)):
        pair = min((p for p in pairs if p["side"] == side),
                   key=lambda p: best[p["index"]]["failure_rate_per_ms"])
        result = best[pair["index"]]
        lowest[label] = dict(omega_r_GHz=pair["omega_r_GHz"], omega_01_GHz=pair["omega_01_GHz"],
                             failure_rate_per_ms=result["failure_rate_per_ms"],
                             g_GHz=result["g_GHz"], t_idle_ns=result["t_idle_ns"])
    # pairs at which the six starts of the recheck end lower than the one start
    single = {result["index"]: result["failure_rate_per_ms"] for result in dense}
    lowered = sum(1 for result in rechecks
                  if result["failure_rate_per_ms"] < single[result["index"]] * (1 - 1e-6))
    tasks = [json.loads(Path(path).read_text()) for path in sorted(glob.glob(str(parts / "*-*.json")))]
    record = dict(
        description="Failure rate of one QEC cycle from the analytical formula (1/ms) minimized over the coupling and idle duration across the ancilla and resonator frequencies.",
        environment=device_model.environment_record(1),
        environment_of_the_tasks=tasks[0]["environment"],
        settings=dict(
            search_ranges=ct.RANGES, grid_points=GRID, coarse_grid_points=COARSE_GRID,
            band_GHz=BAND_GHZ, min_detuning_over_coupling=st.MIN_DETUNING_OVER_COUPLING,
            start_coupling_fractions=START_COUPLING_FRACTIONS, start_idle_ns=START_IDLE_NS,
            recheck_excess=RECHECK_EXCESS, local_options=ct.LOCAL_OPTIONS),
        parameter_sets={label: st.formula_record(free_parameters) | dict(
            bounds_reached=st.bounds_reached(free_parameters))
            for label, free_parameters in st.parameter_sets().items()},
        map=frequency_axes(GRID) | grids | dict(
            index_order="[omega_r][omega_01 (E_J)]",
            pairs_drawn=len(pairs),
            pairs_rechecked=len({result["index"] for result in rechecks}),
            pairs_lowered_by_the_recheck=lowered,
            flagged_and_not_rechecked=pending,
            lowest_failure_rate_per_ms=min(lowest[side]["failure_rate_per_ms"] for side in lowest),
            lowest_pair=lowest),
        core_seconds=sum(task["seconds"] for task in tasks),
    )
    RESULTS_JSON.write_text(json.dumps(record))
    print(f"wrote {RESULTS_JSON}: lowest {record['map']['lowest_failure_rate_per_ms']:.4f} /ms, "
          f"{len(pairs)} pairs, {record['core_seconds'] / 3600:.1f} core-hours", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("coarse", "dense", "recheck", "merge"))
    parser.add_argument("--task", type=int, default=0, help="index of the task of the stage")
    parser.add_argument("--tasks", type=int, default=1, help="number of tasks of the stage")
    parser.add_argument("--round", type=int, default=1, help="round of the recheck")
    parser.add_argument("--parts", type=Path, default=PARTS, help="folder of the part files")
    arguments = parser.parse_args()
    if arguments.stage == "merge":
        merge(arguments.parts)
    else:
        run_task(arguments.stage, arguments.task, arguments.tasks, arguments.parts, arguments.round)
