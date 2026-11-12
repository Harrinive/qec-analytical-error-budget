r"""Failure rate of one QEC cycle against the EA idling duration, at the optimized parameters.

Sweeps the error-accumulation idling duration at the optimized circuit
parameters and evaluates the one-cycle failure rate from the analytical
formula, the failure Markov chain, and the actual QEC dynamics. Writes
``idle-duration-sweep.json`` and ``figs/idle-duration-sweep.pdf``.

Usage: ``python idle-duration-sweep.py`` (or ``--plot-only``, ``--add-formula-curve``, ``--add-loss-count``).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

import qec_aeb as bsq
import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import plot_style

RESULTS_JSON = Path(__file__).resolve().parent / "idle-duration-sweep.json"
# output figure, without extension (``plot_style.save_figure`` adds it)
FIGURE = Path(__file__).resolve().parent / "figs" / "idle-duration-sweep"
# the sweep is at the Delta<0 minimum: its color in every figure
POINT_LABEL = r"$\Delta<0$ minimum"

# EA idling durations (us): from a fifth of the bound of the optimization to three
# times the bound, denser around the bound, which is one of the points
T_IDLE_US = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150)
# EA idling durations (us) at which the analytical formula alone is evaluated, so
# that it is drawn as a smooth curve: every microsecond over the sweep
FORMULA_CURVE_T_IDLE_US = tuple(range(10, 151))
# Upper bound of the EA idling duration in the optimization that produced the
# optimized parameters (us), from search_ranges.py
T_IDLE_UPPER_BOUND_US = 50
T_IDLE_UPPER_BOUND_SOURCE = (
    "upper end of the search range of T_W in search_ranges.py; "
    "recorded here, not produced by this script")
# lower end (failure rate, 1/ms) of the name of the bound, written along its line
BOUND_NAME_PER_MS = 2.47
# distance of the name of the bound from its line (pt)
BOUND_NAME_GAP_PT = 2.0
# units of the parameters as device_model names them
PARAMETER_UNITS = "E_osc_GHz, g_GHz, EJ_GHz in GHz; T_W (idle duration) in ns"
# resonator states of the dynamics in the truncation check at the longest EA idling
RESONATOR_DIMENSION_CHECK = 14
# worker processes that build the dynamical maps
NUM_CPUS = 4

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def point_record(sweep, simulation_parameters: dict) -> dict:
    r"""Failure rates and photon-loss numbers of one cycle, from a dressed sweep.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set (``device_model.build_sweep`` or
        ``device_model.build_cat_sweep``).
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.

    Returns
    -------
    dict
        ``t_idle_us``, ``t_cycle_us``; ``failure_rate_per_ms`` from the
        ``analytical_formula``, the ``failure_Markov_chain``, and the
        ``actual_QEC_dynamics``; ``formula_over_markov_chain_minus_1`` and
        ``formula_over_actual_minus_1``, the relative excess of the formula;
        ``mean_number_of_lost_photons`` and
        ``probability_of_two_or_more_photon_losses`` during the EA idling.
    """
    ingredients = sweep.full_dict()
    t_cycle_ns = float(ingredients["t_cycle"])
    t_idle_ns = float(ingredients["T_W"])

    # the three infidelities of one cycle: the formula, then the chain and
    # the actual QEC dynamics from the same traversed tree
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=NUM_CPUS)
    infidelity = dict(
        analytical_formula=device_model.formula_infidelity(ingredients),
        **device_model.one_cycle_infidelities(graph))
    rate = {name: float(device_model.failure_rate_per_ms(value, t_cycle_ns))
            for name, value in infidelity.items()}
    # the tree holds the maps of every step; free it before the next point
    del graph
    gc.collect()

    # photon loss during the EA idling: the mean of the Poissonian number of
    # losses (rate times mean photon number times duration), and the
    # probability of two or more, as the formula evaluates it
    mean_lost = float(ingredients["jump_a"]) * abs(float(ingredients["disp"])) ** 2 * t_idle_ns
    two_or_more = float(bsq.cat_code_budget["idle_multiple_photon_loss"](**ingredients))
    return dict(
        t_idle_us=t_idle_ns / 1e3,
        t_cycle_us=t_cycle_ns / 1e3,
        failure_rate_per_ms=rate,
        formula_over_markov_chain_minus_1=rate["analytical_formula"] / rate["failure_Markov_chain"] - 1,
        formula_over_actual_minus_1=rate["analytical_formula"] / rate["actual_QEC_dynamics"] - 1,
        mean_number_of_lost_photons=mean_lost,
        probability_of_two_or_more_photon_losses=two_or_more,
    )


def truncation_check(simulation_parameters: dict, parameters: dict, reference: dict) -> dict:
    r"""One point repeated with more resonator states in the dynamics.

    Parameters
    ----------
    simulation_parameters : dict
        Simulation parameters of the point, the second output of
        ``device_model.build_sweep``. They are not changed.
    parameters : dict
        Fixed and device parameters of the point, with the dephasing rate of
        the model: the third output of ``device_model.build_sweep``.
    reference : dict
        Output of :func:`point_record` at the same point with the resonator
        dimension of ``device_model``.

    Returns
    -------
    dict
        ``t_idle_us``; the two resonator dimensions of the dynamics;
        ``failure_rate_per_ms`` at the larger one; and
        ``relative_change_of_failure_rate``, the rate at the larger
        dimension over the rate at the smaller one, minus one, for each of
        the three computations.
    """
    # only the dimension of the dynamics is replaced; the dressed sweep is
    # built again with it (the diagonalized space grows by the same number)
    simulation_parameters = device_model.dynamics_dimensions(
        simulation_parameters, resonator_dimension=RESONATOR_DIMENSION_CHECK)
    sweep = device_model.build_cat_sweep(simulation_parameters, parameters)
    larger = point_record(sweep, simulation_parameters)
    return dict(
        t_idle_us=larger["t_idle_us"],
        dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
        dynamics_resonator_dimension_check=RESONATOR_DIMENSION_CHECK,
        failure_rate_per_ms=larger["failure_rate_per_ms"],
        relative_change_of_failure_rate={
            name: larger["failure_rate_per_ms"][name] / value - 1
            for name, value in reference["failure_rate_per_ms"].items()},
    )


def photon_loss_count(points: list) -> dict:
    r"""Probability of two or more photon losses during the EA idling from binomial and Poisson counts.

    Parameters
    ----------
    points : list of dict
        Outputs of :func:`point_record` in order of increasing idle duration.

    Returns
    -------
    dict
        Binomial (decaying photon number) and Poisson (constant mean photon
        number) probabilities of two or more photon losses and their
        logarithmic slopes between neighbouring durations.
    """
    # the cat state displacement is not a free parameter of the optimization,
    # so the reference value holds at every point of the sweep
    mean_photon_number = abs(device_model.exact_cat_parameters()[1]["disp"]) ** 2
    t_idle_ns = 1e3 * np.array([point["t_idle_us"] for point in points])
    # photon-loss rate from the stored mean number of lost photons, which is
    # rate times mean photon number times duration; the same at every point
    rates = np.array([point["mean_number_of_lost_photons"] for point in points]) / (
        mean_photon_number * t_idle_ns)
    assert np.allclose(rates, rates[0], rtol=1e-9), "the photon-loss rate changes over the sweep"
    photon_loss_rate = float(rates[0])
    surviving = np.exp(-photon_loss_rate * t_idle_ns)

    # photon-number distribution of the maximally mixed logical state: the
    # even-parity error space at the start of the EA idling
    distribution = bsq.cat_error_space_distribution(mean_photon_number, parity=0)
    photon_numbers = np.arange(len(distribution))
    probability = dict(
        # each photon is lost independently: binomial count per Fock state
        binomial_decaying_photon_number=np.array([
            distribution @ binom.sf(1, photon_numbers, 1 - fraction) for fraction in surviving]),
        # the term of the formula, as stored with the sweep
        poisson_constant_photon_number=np.array([
            point["probability_of_two_or_more_photon_losses"] for point in points]),
    )
    # local exponent of each probability in the EA idling duration
    slope = {name: np.diff(np.log(values)) / np.diff(np.log(t_idle_ns))
             for name, values in probability.items()}
    return dict(
        description="Probability of two or more photon losses during the EA idling from the binomial (decaying photon number) and Poisson (constant mean photon number) counts.",
        photon_loss_rate_per_ns=photon_loss_rate,
        mean_photon_number_of_the_formula=float(mean_photon_number),
        mean_photon_number_of_the_logical_state=float(distribution @ photon_numbers),
        points=[
            dict(t_idle_us=point["t_idle_us"],
                 surviving_fraction_of_photons=float(surviving[index]),
                 probability_of_two_or_more_photon_losses={
                     name: float(values[index]) for name, values in probability.items()})
            for index, point in enumerate(points)],
        logarithmic_slope_between_neighbours=[
            dict(t_idle_us=[points[index]["t_idle_us"], points[index + 1]["t_idle_us"]],
                 **{name: float(values[index]) for name, values in slope.items()})
            for index in range(len(points) - 1)],
    )


def formula_curve() -> dict:
    r"""Failure rate of the analytical formula on the fine grid of EA idling durations.

    The formula needs the dressed sweep only, not the tree, so it is
    evaluated at ``FORMULA_CURVE_T_IDLE_US``, with every other parameter at
    its optimized value as in :func:`compute_record`.

    Returns
    -------
    dict
        ``t_idle_us`` and ``failure_rate_per_ms`` of the analytical formula,
        lists of the same length.
    """
    optimized = device_model.parameter_sets()[device_model.OPTIMIZED]
    rates = []
    for t_idle_us in FORMULA_CURVE_T_IDLE_US:
        # the EA idling duration replaces its optimized value; the builder reads it in ns
        sweep, _, _ = device_model.build_sweep(optimized | {"T_W": 1e3 * t_idle_us})
        ingredients = sweep.full_dict()
        rates.append(float(device_model.failure_rate_per_ms(
            device_model.formula_infidelity(ingredients), float(ingredients["t_cycle"]))))
    return dict(t_idle_us=[float(t) for t in FORMULA_CURVE_T_IDLE_US],
                failure_rate_per_ms=rates)


def compute_record() -> dict:
    r"""Run the sweep and the truncation check, and write the data file.

    Returns
    -------
    dict
        The content of ``idle-duration-sweep.json``: ``description``,
        ``environment``, ``settings``, ``points`` (one output of
        :func:`point_record` for each EA idling duration),
        ``truncation_check_at_longest_idle``,
        ``photon_loss_count_during_idle`` (:func:`photon_loss_count`), and
        ``formula_curve`` (:func:`formula_curve`).
    """
    environment = device_model.environment_record(NUM_CPUS)
    print(json.dumps(environment, indent=1), flush=True)
    optimized = device_model.parameter_sets()[device_model.OPTIMIZED]
    points = []
    for t_idle_us in T_IDLE_US:
        # the EA idling duration replaces its optimized value; the builder reads it in ns
        overrides = optimized | {"T_W": 1e3 * t_idle_us}
        sweep, simulation_parameters, parameters = device_model.build_sweep(overrides)
        point = point_record(sweep, simulation_parameters)
        points.append(point)
        rate = point["failure_rate_per_ms"]
        print(f"t_idle = {point['t_idle_us']:5.0f} us: failure rate (1/ms) formula "
              f"{rate['analytical_formula']:.4f}, Markov chain {rate['failure_Markov_chain']:.4f}, "
              f"actual {rate['actual_QEC_dynamics']:.4f}; P(two or more losses) = "
              f"{point['probability_of_two_or_more_photon_losses']:.4f}", flush=True)

    # the longest EA idling again, with more resonator states in the dynamics; the
    # loop ends at that point, so its parameters are the ones still held
    check = truncation_check(simulation_parameters, parameters, points[-1])
    print("truncation check:", json.dumps(check, indent=1), flush=True)

    record = dict(
        description=(
            "Failure rate of one QEC cycle (infidelity over cycle duration, 1/ms) against "
            "the idle duration, every other parameter fixed at its optimized value; "
            "computed from the analytical formula, the failure Markov chain, and the "
            "actual QEC dynamics."),
        environment=environment,
        settings=dict(
            optimized_parameters=optimized,
            optimized_parameters_units=PARAMETER_UNITS,
            t_idle_optimized_us=optimized["T_W"] / 1e3,
            t_idle_upper_bound_of_the_optimization_us=float(T_IDLE_UPPER_BOUND_US),
            t_idle_upper_bound_source=T_IDLE_UPPER_BOUND_SOURCE,
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
            diagonalization_ancilla_dimension=device_model.DIAGONALIZATION_ANCILLA_DIMENSION,
            diagonalization_resonator_dimension=(
                device_model.DYNAMICS_RESONATOR_DIMENSION + device_model.EXTRA_RESONATOR_STATES),
        ),
        points=points,
        truncation_check_at_longest_idle=check,
        photon_loss_count_during_idle=photon_loss_count(points),
        formula_curve=formula_curve(),
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")
    return record


def draw_figure(record: dict) -> None:
    r"""Failure rate of one QEC cycle against the EA idling duration, three ways.

    One panel at the width of a column, on the canvas that the figures with
    one panel share (``plot_style.single_panel``), with the flat axes box
    (``plot_style.ASPECT_FLAT``). The analytical formula is a curve
    through its values on the fine grid (``formula_curve``); the actual QEC
    dynamics and the failure Markov chain are the markers of
    ``plot_style.MODEL_MARKERS``. Everything has the color of the
    :math:`\Delta<0` minimum, at which the other three parameters are held. A
    vertical line marks the upper bound of the EA idling duration in the search,
    named beside it.

    Parameters
    ----------
    record : dict
        Content of ``idle-duration-sweep.json``.

    Returns
    -------
    None
        Writes ``FIGURE`` as PDF and PNG.
    """
    import matplotlib

    # headless backend: the script runs without a display
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    from matplotlib.transforms import ScaledTranslation

    plot_style.use_manuscript_style()
    color = plot_style.POINT_STYLES[POINT_LABEL]["color"]
    t_idle = np.array([point["t_idle_us"] for point in record["points"]])
    rate = {name: np.array([point["failure_rate_per_ms"][name] for point in record["points"]])
            for name in ("analytical_formula", "actual_QEC_dynamics", "failure_Markov_chain")}
    t_bound = record["settings"]["t_idle_upper_bound_of_the_optimization_us"]

    fig, ax = plot_style.single_panel(plot_style.ASPECT_FLAT)
    # the bound of the search first, as a thin guide, so that the data lie above it
    bound = plot_style.BOUND_LINE
    ax.axvline(t_bound, zorder=1, **bound)
    # its name along the line, on its left, in the empty space above the curve
    beside = ax.transData + ScaledTranslation(-BOUND_NAME_GAP_PT / 72, 0, fig.dpi_scale_trans)
    ax.text(t_bound, BOUND_NAME_PER_MS, "upper bound", transform=beside,
            rotation=90, ha="right", va="bottom", color=bound["color"],
            fontsize=plot_style.SMALL_TEXT_SIZE)
    curve = dict(color=color, lw=plot_style.CURVE_WIDTH)
    # the formula on its fine grid, a smooth curve that is not taken for data
    fine = record["formula_curve"]
    ax.plot(fine["t_idle_us"], fine["failure_rate_per_ms"], zorder=3, **curve)
    # the two numerically computed models, by the shape of their marker
    models = {"actual QEC dynamics": "actual_QEC_dynamics",
              "failure Markov chain": "failure_Markov_chain"}
    for label, key in models.items():
        ax.plot(t_idle, rate[key], **plot_style.model_marker_style(label, color))
    ax.set_xlabel(r"EA idling duration $t_{\mathrm{ea}}$ ($\mu$s)")
    ax.set_ylabel(r"failure rate (ms$^{-1}$)")
    ax.set_xlim(0, 1.04 * t_idle.max())
    # a tick every half failure per ms, as on the other axes of a failure rate
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    handles = [Line2D([], [], label="AEB", **curve)]
    handles += [Line2D([], [], label=label, **plot_style.model_marker_style(label, color))
                for label in models]
    ax.legend(handles=handles, loc="upper right", **plot_style.LEGEND_STYLE)
    # the canvas has the width of a column and is saved as it is
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from idle-duration-sweep.json; no computation")
    parser.add_argument("--add-formula-curve", action="store_true",
                        help="evaluate only the analytical formula on the fine grid of idle "
                             "durations and add it to idle-duration-sweep.json; no tree")
    parser.add_argument("--add-loss-count", action="store_true",
                        help="compute only the photon-loss count from the stored points and "
                             "add it to idle-duration-sweep.json; no tree, no figure")
    arguments = parser.parse_args()
    if arguments.add_loss_count:
        # arithmetic on the stored points; the figure does not show it
        record = json.loads(RESULTS_JSON.read_text())
        record["photon_loss_count_during_idle"] = photon_loss_count(record["points"])
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(json.dumps(record["photon_loss_count_during_idle"], indent=1))
        print(f"wrote {RESULTS_JSON}")
        raise SystemExit
    if arguments.add_formula_curve:
        record = json.loads(RESULTS_JSON.read_text())
        record["formula_curve"] = formula_curve()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(f"wrote {RESULTS_JSON}")
        draw_figure(record)
        raise SystemExit
    if arguments.plot_only:
        draw_figure(json.loads(RESULTS_JSON.read_text()))
    else:
        draw_figure(compute_record())
