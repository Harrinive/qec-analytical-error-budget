r"""Agreement of the failure Markov chain with the actual QEC dynamics over every case of the example.

Collects the relative fidelity difference
:math:`(F_{\mathbb T} - F_{\mathbb M})/(1 - F_{\mathbb T})` after one QEC cycle
across the stored parameter sets, sweeps, and ancilla circuits. Writes
``range-of-validity.json``.

Usage: ``python range-of-validity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "range-of-validity.json"
# the data files sit next to this script
DATA = HERE

# incoherent-error scales at or above which a case counts as "away from the small end"
SMALLEST_SCALE_AWAY = 0.25
# label of the optimized fluxonium in ancilla-comparison.json
FLUXONIUM = "fluxonium, optimized"


def load(name: str) -> dict:
    r"""Read one data file.

    Parameters
    ----------
    name : str
        File name in the folder of the data files.

    Returns
    -------
    dict
        The parsed JSON.
    """
    return json.loads((DATA / name).read_text())


def relative_difference(infidelity_chain: float, infidelity_actual: float) -> float:
    r"""Relative fidelity difference from the two one-cycle infidelities.

    Parameters
    ----------
    infidelity_chain : float
        :math:`1-F_{\mathbb M}`.
    infidelity_actual : float
        :math:`1-F_{\mathbb T}`.

    Returns
    -------
    float
        :math:`(F_{\mathbb T}-F_{\mathbb M})/(1-F_{\mathbb T})`, which equals
        the ratio of the two infidelities minus one.
    """
    return infidelity_chain / infidelity_actual - 1


def collect() -> list[dict]:
    r"""The quantity for every case of the example.

    Returns
    -------
    list of dict
        One entry per case: ``group``, ``label``, ``value``, and, for the
        incoherent-error scale, ``scale``.
    """
    cases = []
    # three parameter sets of the transmon cycle
    for label, record in load("budget-at-three-points.json")["parameter_sets"].items():
        infidelity = record["infidelity"]
        cases.append(dict(group="parameter sets", label=label, value=relative_difference(
            infidelity["failure_Markov_chain"], infidelity["actual_QEC_dynamics"])))
    # idle duration sweep: the same cycle duration for both models, so the
    # ratio of the failure rates is the ratio of the infidelities
    for point in load("idle-duration-sweep.json")["points"]:
        rates = point["failure_rate_per_ms"]
        cases.append(dict(group="idle duration", label=f"{point['t_idle_us']:g} us",
                          value=relative_difference(rates["failure_Markov_chain"],
                                                    rates["actual_QEC_dynamics"])))
    # dispersive shift sweep
    for point in load("parameter-tradeoff.json")["numerical"]:
        infidelity = point["infidelity"]
        cases.append(dict(group="dispersive shift", label=f"{point['chi_1_MHz']:.3g} MHz",
                          value=relative_difference(infidelity["failure_Markov_chain"],
                                                    infidelity["actual_QEC_dynamics"])))
    # incoherent-error scale
    for row in load("accuracy-sweep.json")["rows"]:
        fidelity = row["fidelity"]
        cases.append(dict(group="incoherent-error scale", label=f"xi = {row['scale']:g}",
                          scale=row["scale"],
                          value=(fidelity["F_T"] - fidelity["F_M"]) / (1 - fidelity["F_T"])))
    # fluxonium ancilla: the optimized fluxonium
    infidelity = load("ancilla-comparison.json")["parameter_sets"][FLUXONIUM]["infidelity"]
    cases.append(dict(group="fluxonium", label=FLUXONIUM,
                      value=relative_difference(infidelity["failure_Markov_chain"],
                                                infidelity["actual_QEC_dynamics"])))
    return cases


def main() -> dict:
    r"""Collect all cases and compute summary statistics.

    Returns
    -------
    dict
        The record that is written to ``range-of-validity.json``.
    """
    cases = collect()
    # cases away from the small end of the incoherent-error scale; the fluxonium
    # cases are a different circuit and are reported apart
    away = [case for case in cases
            if case["group"] != "fluxonium"
            and case.get("scale", float("inf")) >= SMALLEST_SCALE_AWAY]
    largest = max(away, key=lambda case: case["value"])
    scales = {case["scale"]: case["value"] for case in cases if case["group"] == "incoherent-error scale"}
    fluxonium = next(case for case in cases if case["group"] == "fluxonium")
    return dict(
        description=(
            "(F_T - F_M)/(1 - F_T) after one QEC cycle for every case of the example: three "
            "parameter sets, twelve idle durations, nine dispersive shifts, nine incoherent-error "
            "scales, and the optimized fluxonium."),
        quantity="(F_T - F_M)/(1 - F_T); positive when the failure Markov chain overestimates the infidelity",
        cases=cases,
        quoted=dict(
            number_of_cases_away_from_the_small_end=len(away),
            largest_away_from_the_small_end=largest["value"],
            largest_away_group=largest["group"],
            largest_away_label=largest["label"],
            smallest_scale_counted_as_away=SMALLEST_SCALE_AWAY,
            at_scale_1=scales[1.0],
            at_smallest_scale=scales[min(scales)],
            smallest_scale=min(scales),
            by_scale=scales,
            fluxonium_optimized=fluxonium["value"],
            group_ranges={group: [min(c["value"] for c in cases if c["group"] == group),
                                  max(c["value"] for c in cases if c["group"] == group)]
                          for group in dict.fromkeys(c["group"] for c in cases)},
        ),
    )


if __name__ == "__main__":
    record = main()
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["quoted"], indent=1))
    print(f"wrote {RESULTS_JSON}")
