r"""Failure rate of the AEB by mechanism against the coupling, at a fixed ratio of coupling to detuning.

Scans the resonator--ancilla coupling :math:`g` while adjusting the resonator
frequency to keep :math:`g/\Delta` fixed at its value at the :math:`\Delta>0`
minimum of ``optimization.json``. Writes ``coupling-scan.json``.

Usage: ``python coupling-scan.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cycle_target as ct
import device_model
import sweep_tools as st

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "coupling-scan.json"
# the start point: the Delta>0 minimum as budget-at-three-points.json stores it
START_JSON = HERE / "budget-at-three-points.json"

# couplings of the scan (GHz); 0.05 is the lower bound of the search range
COUPLINGS_GHZ = [0.02, 0.03, 0.05, 0.08, 0.12]
# the coupling against which the relative changes are taken: the lower bound
BOUND_GHZ = ct.RANGES["g_GHz"][0]
# couplings over which the quoted largest relative change is taken (GHz)
QUOTED_WINDOW_GHZ = (0.03, 0.08)


def start_point() -> dict:
    r"""Free parameters of the Delta>0 minimum and its stored failure rate.

    Returns
    -------
    dict
        ``free_parameters`` (the four free parameters) and
        ``failure_rate_per_ms`` (that of the analytical formula stored in
        ``budget-at-three-points.json``, for the check of this evaluation).
    """
    stored = json.loads(START_JSON.read_text())["parameter_sets"][st.POSITIVE_DETUNING]
    return dict(free_parameters={name: float(value)
                                 for name, value in stored["free_parameters"].items()},
                failure_rate_per_ms=float(stored["failure_rate_per_ms"][st.FORMULA]))


def point_record(free_parameters: dict) -> dict:
    r"""Failure rate of the AEB by mechanism and the dressed quantities at one point.

    Parameters
    ----------
    free_parameters : dict
        The four free parameters; they replace their reference values.

    Returns
    -------
    dict
        ``failure_rate_per_ms`` (formula), ``omega_01_GHz`` (ancilla
        transition frequency of the formula's ingredients), ``chi_1_MHz``
        (:math:`\chi_1/2\pi` with its sign), and
        ``t_parity_mapping_idling_ns``.
    """
    # one dressed sweep gives the formula and the derived quantities
    evaluation = ct.evaluate_formula(st.REFERENCE | free_parameters, breakdown=False,
                                     keep_sweep=True)
    derived = st.derived_quantities(evaluation["full_dict"])
    return dict(
        failure_rate_per_ms=float(evaluation["rate"] * st.NS_PER_MS),
        omega_01_GHz=derived["omega_01_GHz"],
        chi_1_MHz=derived["chi_1_MHz"],
        t_parity_mapping_idling_ns=derived["t_parity_mapping_idling_ns"],
    )


def quoted_numbers(rows: list[dict]) -> dict:
    r"""Summary metrics of the coupling scan at fixed coupling-to-detuning ratio.

    Parameters
    ----------
    rows : list of dict
        One row of :func:`main` per coupling, with ``g_GHz``,
        ``relative_change_from_bound``, and ``chi_1_MHz``.

    Returns
    -------
    dict
        ``relative_change_at_<g>_MHz`` for every coupling other than the bound;
        ``largest_absolute_relative_change_30_to_80_MHz`` (over the couplings
        of ``QUOTED_WINDOW_GHZ``); ``chi_1_MHz_range`` (smallest and largest
        :math:`\chi_1/2\pi` of the scan, which the fixed ratio keeps close).
    """
    # each coupling against the bound, then the largest over the window
    quoted = {f"relative_change_at_{round(row['g_GHz'] * 1e3)}_MHz":
              row["relative_change_from_bound"]
              for row in rows if row["g_GHz"] != BOUND_GHZ}
    lower, upper = QUOTED_WINDOW_GHZ
    window = [abs(row["relative_change_from_bound"]) for row in rows
              if lower - 1e-12 <= row["g_GHz"] <= upper + 1e-12]
    quoted["largest_absolute_relative_change_30_to_80_MHz"] = max(window)
    chi = [row["chi_1_MHz"] for row in rows]
    quoted["chi_1_MHz_range"] = [min(chi), max(chi)]
    return quoted


def main() -> None:
    r"""Scan the coupling at a fixed ratio of coupling to detuning and write the JSON.

    Returns
    -------
    None
    """
    environment = device_model.environment_record(1)
    start = start_point()
    free = start["free_parameters"]

    # ancilla transition and the ratio g / Delta at the Delta>0 minimum
    at_start = point_record(free)
    omega_01 = at_start["omega_01_GHz"]
    ratio = free["g_GHz"] / (free["E_osc_GHz"] - omega_01)
    print(f"Delta>0 minimum: omega_01 = {omega_01:.4f} GHz, g/Delta = {ratio:.4f}, "
          f"rate {at_start['failure_rate_per_ms']:.4f} /ms "
          f"(stored {start['failure_rate_per_ms']:.4f} /ms)", flush=True)

    rows = []
    for coupling in COUPLINGS_GHZ:
        # resonator frequency that keeps g / Delta at its value at the minimum
        resonator = omega_01 + coupling / ratio
        point = point_record(free | {"g_GHz": coupling, "E_osc_GHz": resonator})
        rows.append(dict(g_GHz=coupling, E_osc_GHz=resonator,
                         detuning_GHz=resonator - omega_01,
                         inside_search_range=bool(coupling >= BOUND_GHZ - 1e-12)) | point)
        print(f"g = {coupling * 1e3:.0f} MHz, resonator {resonator:.4f} GHz: "
              f"chi_1/2pi = {point['chi_1_MHz']:.4f} MHz, "
              f"rate {point['failure_rate_per_ms']:.4f} /ms", flush=True)

    # every rate against the rate at the lower bound of the coupling
    at_bound = next(row for row in rows if row["g_GHz"] == BOUND_GHZ)["failure_rate_per_ms"]
    for row in rows:
        row["relative_change_from_bound"] = row["failure_rate_per_ms"] / at_bound - 1

    record = dict(
        description="Failure rate of one QEC cycle (1/ms) from the analytical formula against the coupling g at fixed g/Delta around the Delta>0 minimum.",
        environment=environment,
        settings=dict(
            start_point=st.POSITIVE_DETUNING,
            start_source="budget-at-three-points.json",
            free_parameters_at_start=free,
            stored_failure_rate_per_ms_at_start=start["failure_rate_per_ms"],
            failure_rate_per_ms_at_start=at_start["failure_rate_per_ms"],
            omega_01_held_GHz=omega_01,
            g_over_Delta=ratio,
            couplings_GHz=COUPLINGS_GHZ,
            search_range_g_GHz=ct.RANGES["g_GHz"],
            units=("frequencies in GHz, chi_1/2pi in MHz, failure rate in 1/ms, "
                   "relative change dimensionless"),
        ),
        rows=rows,
        quoted=quoted_numbers(rows),
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["quoted"], indent=1))
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
