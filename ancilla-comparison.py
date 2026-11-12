r"""Error budget of one QEC cycle with a transmon and with a fluxonium as the ancilla.

Evaluates the failure rate by step and by error mechanism, together with static
circuit quantities and ancilla decoherence rates, at three parameter sets: the
transmon of Ofek et al., the optimized transmon, and an optimized zero-flux
fluxonium. Writes ``ancilla-comparison.json`` and
``figs/ancilla-comparison.pdf``.

Usage: ``python ancilla-comparison.py`` (or ``--plot-only``, ``--add-properties``).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import qec_aeb as bsq
import chencrafts.settings as cc_settings

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import cycle_target as ct
import fluxonium_model
import model_chain
import plot_style
import sweep_tools as st

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "ancilla-comparison.json"
# free parameters of the fluxonium at the minimum of its search that is kept
FLUXONIUM_OPTIMUM = HERE / "fluxonium-optimum.json"
# output file without extension (plot_style.save_figure adds .pdf and .png)
FIGURE = HERE / "figs" / "ancilla-comparison"

# worker processes that build the dynamical maps
NUM_CPUS = 4
# the three parameter sets, in the order of the bars
OFEK, TRANSMON, FLUXONIUM = "transmon, Ofek et al.", "transmon, optimized", "fluxonium, optimized"
# hatching of the bars of each set, as the odd-parity error space of
# formula-terms.py: lines over the color of the bar, dark on a light face
HATCHES = {OFEK: None, TRANSMON: plot_style.DIAGONAL_HATCH, FLUXONIUM: plot_style.CROSSED_HATCH}
# Layout (inch) on a canvas of the text width: the two axes side by side,
# with the same upper and lower edges. The margin under them holds the names
# of the steps and mechanisms (two rows) and the legend of the hatchings;
# the margin above them holds the panel tags.
# panel -> left edge, lower edge, width, height of its axes, of the shape
# ASPECT_REGULAR; the gap between the two holds the tick labels and the axis
# label of (b)
PANEL_LEFT, PANEL_GAP, PANEL_RIGHT = 0.50, 0.55, 0.06
PANEL_BOTTOM, PANEL_TOP = 0.60, 0.22
PANEL_WIDTH = (plot_style.TEXT_WIDTH - PANEL_LEFT - PANEL_GAP - PANEL_RIGHT) / 2
PANEL_HEIGHT = plot_style.ASPECT_REGULAR * PANEL_WIDTH
FIGURE_HEIGHT = PANEL_BOTTOM + PANEL_HEIGHT + PANEL_TOP
PANELS = {"a": (PANEL_LEFT, PANEL_BOTTOM, PANEL_WIDTH, PANEL_HEIGHT),
          "b": (PANEL_LEFT + PANEL_WIDTH + PANEL_GAP, PANEL_BOTTOM, PANEL_WIDTH, PANEL_HEIGHT)}
# upper left corner of the legend of the hatchings: one row under the names
# of the steps, flush with the left edge of (a)
LEGENDS = {"sets": (PANEL_LEFT, 0.17)}
# distance of the ticks of the failure rate (1/ms), in both panels
RATE_TICK_STEP = 0.5
# free part of the vertical axis above the highest bar, in parts of its height
TOP_MARGIN = 0.08
FORMULA, CHAIN, ACTUAL = st.FORMULA, st.CHAIN, st.ACTUAL
# baths of the error model, as keys of the spectral densities that chencrafts
# stores, with what each is; a fluxonium without inductive loss and the
# transmon have no "qubit_phi_operator"
BATHS = {
    "qubit_n_operator": "capacitive (dielectric) loss of the ancilla",
    "qubit_phi_operator": "inductive loss of the ancilla",
    "qubit_proj_11": "1/f dephasing of the ancilla",
    "res_a_m_adag": "dielectric loss of the resonator, through the dressing",
    "res_adag_a": "dephasing of the resonator",
}
# ancilla operators of the baths through which the ancilla relaxes, with the
# bath key; their matrix elements explain the relaxation rates
LOSS_OPERATORS = {"qubit_n_operator": "n_operator", "qubit_phi_operator": "phi_operator"}
US_PER_NS = 1e-3

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


def set_record(sweep, simulation_parameters: dict, free_parameters: dict) -> dict:
    r"""Contributions to the failure rate of one QEC cycle at one parameter set.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set, of either ancilla.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.
    free_parameters : dict
        The parameters that tell the set from the reference set of its
        ancilla; stored, not used.

    Returns
    -------
    dict
        ``free_parameters``; ``t_cycle_ns``; ``infidelity`` and
        ``failure_rate_per_ms`` (each with the keys ``FORMULA``, ``CHAIN``,
        ``ACTUAL``); ``rate_by_step_per_ms`` (``ACTUAL``: the loss of the
        process fidelity of the actual QEC dynamics at each step; ``CHAIN``:
        the increase of the failure probability of the failure Markov chain;
        ``chain_by_manifestation``: its split into leakage and the logical
        X, Y, Z errors; all over the cycle duration);
        ``rate_by_mechanism_per_ms`` (the contributions of the formula
        over the cycle duration); ``formula_over_failure_Markov_chain_minus_1``
        and ``formula_over_actual_QEC_dynamics_minus_1``.
    """
    ingredients = sweep.full_dict()
    t_cycle = float(ingredients["t_cycle"])
    by_mechanism = ct.mechanism_entries(ingredients)

    # the three models from one traversed tree of one cycle
    graph = device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=NUM_CPUS)
    budget = model_chain.losses_by_step(graph)
    fidelity, loss = budget["fidelity"], budget["loss"]
    split = budget["chain_by_manifestation"]
    # the tree holds the maps of every step; free it before the next one
    del graph
    gc.collect()

    def rate(value):
        r"""A part of the infidelity as a contribution to the failure rate (1/ms)."""
        return device_model.failure_rate_per_ms(np.asarray(value, float), t_cycle).tolist()

    infidelity = {FORMULA: device_model.formula_infidelity(ingredients),
                  CHAIN: float(1 - fidelity["F_M"][-1]), ACTUAL: float(1 - fidelity["F_T"][-1])}
    return dict(
        free_parameters={name: float(value) for name, value in free_parameters.items()},
        t_cycle_ns=t_cycle,
        infidelity=infidelity,
        failure_rate_per_ms={name: rate(value) for name, value in infidelity.items()},
        rate_by_step_per_ms={
            ACTUAL: rate(loss["F_T"]), CHAIN: rate(loss["F_M"]),
            "chain_by_manifestation": {key: rate(values) for key, values in split.items()}},
        rate_by_mechanism_per_ms={name: rate(value) for name, value in by_mechanism.items()},
        formula_over_failure_Markov_chain_minus_1=infidelity[FORMULA] / infidelity[CHAIN] - 1,
        formula_over_actual_QEC_dynamics_minus_1=infidelity[FORMULA] / infidelity[ACTUAL] - 1,
    )


def jump_rates_by_bath(sweep, simulation_parameters: dict) -> dict:
    r"""Rates of the dressed ancilla jumps :math:`|i\rangle\langle j|`, split by bath.

    ``chencrafts`` adds the contributions of all baths into one array
    ``jump_ij``. Registering the rates again with one bath at a time gives
    each contribution with the same code; a last registration with every
    bath restores the sweep.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of one parameter set, with the rates registered.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built (truncations of
        the dressed operators).

    Returns
    -------
    dict
        Key of :data:`BATHS` to the array of rates (1/ns) of
        :math:`|i\rangle\langle j|` that this bath gives, ``[i, j]``.
    """
    # the baths of this sweep: the keys of its spectral densities
    densities = next(iter(np.asarray(sweep.sweep["spectral_density"], dtype=object).flat))
    names = list(densities)
    total = np.array(sweep.full_dict()["jump_ij"], dtype=float)
    truncation = dict(res_trunc_dim=simulation_parameters["res_me_dim"],
                      qubit_trunc_dim=simulation_parameters["qubit_me_dim"])

    def register(selected):
        r"""Register the jump rates of the baths ``selected`` only.

        Parameters
        ----------
        selected : list of str
            Keys of :data:`BATHS`.

        Returns
        -------
        np.ndarray
            ``jump_ij`` of these baths (1/ns).
        """
        bsq.batched_sweep_jump_rates(
            sweep.sweep, **truncation,
            qubit_op_names=[name[len("qubit_"):] for name in selected if name.startswith("qubit_")],
            res_op_names=[name[len("res_"):] for name in selected if name.startswith("res_")])
        return np.array(sweep.full_dict()["jump_ij"], dtype=float)

    by_bath = {name: register([name]) for name in names}
    # the sweep again holds the rates of every bath, as it was built
    restored = register(names)
    if np.abs(restored - total).max() > 0 or np.abs(sum(by_bath.values()) - total).max() > 1e-15:
        raise RuntimeError("the rates by bath do not add up to the rates of the sweep")
    return by_bath


def ancilla_properties(sweep, simulation_parameters: dict, dephasing: dict,
                       unit_dephasing: dict) -> dict:
    r"""Parameters, static quantities, and decoherence rates of the ancilla at one set.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of the parameter set, of either ancilla.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.
    dephasing : dict
        Ancilla dephasing rate :math:`\kappa_\phi` by noise source and its
        ``total`` (1/ns), from ``fluxonium_model.fluxonium_dephasing_rate``
        or ``device_model.transmon_dephasing_rate``.
    unit_dephasing : dict
        The same rates at unit noise amplitude (1/ns): the sensitivity of the
        0--1 transition to each source.

    Returns
    -------
    dict
        ``parameters`` (circuit energies, resonator frequency, coupling,
        EA idling duration); ``static`` (bare and dressed 0--1 transition
        frequency, detuning, and ``sweep_tools.derived_quantities``);
        ``jump_rates_per_ms`` (``"|i><j|"`` to the rate, every dressed ancilla
        jump of the model); ``times_us`` (``T1``, ``T_phi``, and the lifetime
        of level 1 against every jump out of it); ``relaxation_by_bath_per_ms``,
        ``excitation_by_bath_per_ms``, and ``dephasing_jump_by_bath_per_ms``
        (the rates of :math:`|0\rangle\langle1|`, :math:`|1\rangle\langle0|`,
        and :math:`|1\rangle\langle1|` by bath);
        ``dephasing_by_source_per_ms`` (:math:`\kappa_\phi` by noise source)
        and ``dephasing_sensitivity_per_ms`` (at unit amplitude);
        ``loss_matrix_elements`` (the squared 0--1 matrix element of each
        ancilla operator through which the ancilla relaxes, of the bare
        circuit, and the spectral density of its bath at the dressed 0--1
        frequency).
    """
    full_dict = sweep.full_dict()
    jumps = np.array(full_dict["jump_ij"], dtype=float)
    levels = jumps.shape[-1]
    # rates are stored in 1/ns and recorded in 1/ms
    per_ms = st.NS_PER_MS

    # the parameters that define the set; the transmon has no inductor and no flux
    names = ("EJ_GHz", "EC_GHz", "EL_GHz", "flux", "E_osc_GHz", "g_GHz")
    parameters = {name: float(full_dict[name]) for name in names if name in full_dict}
    parameters["t_ea_us"] = float(full_dict["T_W"]) * US_PER_NS

    # dressed 0-1 transition with the resonator empty (bare labels: resonator, ancilla)
    energy = sweep.sweep.energy_by_bare_index
    omega_dressed = float(np.squeeze(energy((0, 1)) - energy((0, 0))))
    static = dict(
        omega_01_bare_GHz=float(full_dict["omega_a_GHz"]),
        omega_01_dressed_GHz=omega_dressed,
        detuning_GHz=float(full_dict["E_osc_GHz"] - full_dict["omega_a_GHz"]),
        bare_ancilla_levels_GHz=[float(value) for value in np.squeeze(
            full_dict["bare_evals"][1])[:levels] - np.squeeze(full_dict["bare_evals"][1])[0]],
    ) | st.derived_quantities(full_dict)

    # times: T1 from the downward jump, T_phi from the coherence decay
    # gamma_{|1><1|}/2, and the lifetime of level 1 against every jump out of it
    out_of_level_1 = sum(jumps[i, 1] for i in range(levels) if i != 1)
    times = dict(T1=US_PER_NS / jumps[0, 1], T_phi=US_PER_NS * 2 / jumps[1, 1],
                 T_phi_from_kappa_phi=US_PER_NS / dephasing["total"],
                 level_1_lifetime=US_PER_NS / out_of_level_1)

    # the contribution of every bath to relaxation and to the dephasing jump
    by_bath = jump_rates_by_bath(sweep, simulation_parameters)

    # charge (and phase) matrix element of the bare circuit and the bath
    # spectral density at the transition: their product is the relaxation rate
    # of that bath up to the dressing
    circuit = sweep.sweep.hilbertspace.subsystem_list[1]
    densities = next(iter(np.asarray(sweep.sweep["spectral_density"], dtype=object).flat))
    loss = {}
    for bath, operator in LOSS_OPERATORS.items():
        if bath not in densities:
            continue
        element = circuit.matrixelement_table(operator, evals_count=2)[0, 1]
        density = float(densities[bath](omega_dressed))
        loss[bath] = dict(operator=operator, matrix_element_squared=float(abs(element) ** 2),
                          spectral_density_per_ms=density * per_ms,
                          product_per_ms=float(abs(element) ** 2) * density * per_ms)

    return dict(
        parameters=parameters,
        static=static,
        jump_rates_per_ms={f"|{i}><{j}|": float(jumps[i, j]) * per_ms
                           for i in range(levels) for j in range(levels)},
        times_us={name: float(value) for name, value in times.items()},
        relaxation_by_bath_per_ms={BATHS[name]: float(rates[0, 1]) * per_ms
                                   for name, rates in by_bath.items()},
        excitation_by_bath_per_ms={BATHS[name]: float(rates[1, 0]) * per_ms
                                   for name, rates in by_bath.items()},
        dephasing_jump_by_bath_per_ms={BATHS[name]: float(rates[1, 1]) * per_ms
                                       for name, rates in by_bath.items()},
        dephasing_by_source_per_ms={name: float(value) * per_ms
                                    for name, value in dephasing.items()},
        dephasing_sensitivity_per_ms={name: float(value) * per_ms
                                      for name, value in unit_dephasing.items()},
        loss_matrix_elements={BATHS[name]: value for name, value in loss.items()},
    )


def properties_record() -> dict:
    r"""The ancilla properties at the three parameter sets; no tree.

    Returns
    -------
    dict
        ``description``, ``conventions``, ``noise_amplitudes``, ``units``, and
        ``parameter_sets`` (label of a set to the output of
        :func:`ancilla_properties`).
    """
    transmon_sets = st.parameter_sets()
    optimum = json.loads(FLUXONIUM_OPTIMUM.read_text())
    sets = {}

    # the transmon: the sweep of the formula, with the dephasing model that
    # follows the Josephson energy
    for label, key in ((OFEK, st.OFEK), (TRANSMON, st.NEGATIVE_DETUNING)):
        evaluation = ct.evaluate_formula(st.REFERENCE | transmon_sets[key], breakdown=False,
                                         keep_sweep=True)
        full_dict = evaluation["full_dict"]
        charge, critical_current = device_model.transmon_1_over_f_rates(
            full_dict["EJ_GHz"], full_dict["EC_GHz"], full_dict["ng"])
        sets[label] = ancilla_properties(
            evaluation["sweep"], evaluation["simulation_parameters"],
            device_model.transmon_dephasing_rate(full_dict),
            dict(charge=charge, critical_current=critical_current))

    # the fluxonium: the sweep of the figure, with inductive loss
    sweep, simulation_parameters, parameters = fluxonium_model.build_sweep(
        optimum["free_parameters"], inductive_loss=True)
    sets[FLUXONIUM] = ancilla_properties(
        sweep, simulation_parameters, fluxonium_model.fluxonium_dephasing_rate(parameters),
        fluxonium_model.fluxonium_noise_rates(parameters))

    return dict(
        description="Ancilla circuit parameters, static quantities, and decoherence rates at the three parameter sets.",
        noise_amplitudes=dict(
            flux_flux_quanta=fluxonium_model.fit_flux_noise_amplitude(),
            critical_current=device_model.fit_critical_current_noise(),
            charge_2e=device_model.CHARGE_NOISE_AMPLITUDE),
        units=("energies and frequencies in GHz (divided by h or 2 pi), t_ea in us, times in "
               "us, rates in 1/ms; chi_1_MHz, chi_1_prime_kHz, K_kHz divided by 2 pi"),
        parameter_sets=sets,
    )


def compute() -> dict:
    r"""The record of the three parameter sets.

    Returns
    -------
    dict
        The record that is written to ``ancilla-comparison.json``:
        ``steps``, ``parameter_sets`` (label of a set to the output of
        :func:`set_record`), ``settings``, and ``environment``.
    """
    environment = device_model.environment_record(NUM_CPUS)
    transmon_sets = st.parameter_sets()
    optimum = json.loads(FLUXONIUM_OPTIMUM.read_text())

    sets = {}
    for label, key in ((OFEK, st.OFEK), (TRANSMON, st.NEGATIVE_DETUNING)):
        evaluation = ct.evaluate_formula(st.REFERENCE | transmon_sets[key], breakdown=False,
                                         keep_sweep=True)
        sets[label] = set_record(evaluation["sweep"], evaluation["simulation_parameters"],
                                 transmon_sets[key])
        print(label, sets[label]["failure_rate_per_ms"], flush=True)
    sweep, simulation_parameters, _ = fluxonium_model.build_sweep(
        optimum["free_parameters"], inductive_loss=True)
    sets[FLUXONIUM] = set_record(sweep, simulation_parameters, optimum["free_parameters"])
    print(FLUXONIUM, sets[FLUXONIUM]["failure_rate_per_ms"], flush=True)

    return dict(
        description="Failure-rate contributions by step and by error mechanism for the cat-code protocol with transmon and integer-fluxonium ancillas.",
        steps=model_chain.STEP_NAMES,
        parameter_sets=sets,
        settings=dict(
            transmon_optimized=st.NEGATIVE_DETUNING,
            fluxonium_optimum=optimum,
            fluxonium_inductive_quality_factor=fluxonium_model.INDUCTIVE_QUALITY_FACTOR,
            units="energies and frequencies in GHz, T_W (idle duration) in ns, rates in 1/ms"),
        ancilla_properties=properties_record(),
        environment=environment,
    )


def plot(record: dict) -> None:
    r"""Draw the two-panel figure from the record of ``ancilla-comparison.json``.

    Parameters
    ----------
    record : dict
        Output of :func:`compute`.

    Returns
    -------
    None
    """
    import matplotlib

    # headless backend: the script also runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    plot_style.use_manuscript_style()
    sets = record["parameter_sets"]
    # a marker on a bar near zero overlaps the axis and is drawn whole
    actual = plot_style.bar_marker_style("actual QEC dynamics") | dict(clip_on=False)
    markov = plot_style.bar_marker_style("failure Markov chain") | dict(clip_on=False)

    def bars(ax, x, heights, colors, label, width, bottom=0.0):
        r"""Bars of one parameter set, with the hatching of the set.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        x : np.ndarray
            Centers of the bars.
        heights : array-like
            Contributions to the failure rate (1/ms).
        colors : str or list of str
            Face color of the bars, one for all or one for each.
        label : str
            Key of ``HATCHES``: the parameter set.
        width : float
            Width of a bar (data units), from ``plot_style.group_offsets``.
        bottom : float or np.ndarray
            Lower ends of the bars, for a stacked bar.

        Returns
        -------
        None
        """
        x, heights = np.asarray(x, float), np.asarray(heights, float)
        bottom = np.broadcast_to(np.asarray(bottom, float), x.shape)
        if isinstance(colors, str):
            colors = [colors] * len(x)
        # one bar at a time: the color of the hatching depends on the face
        for center, height, low, color in zip(x, heights, bottom, colors):
            ax.bar(center, height, width, bottom=low, facecolor=color, linewidth=0,
                   **plot_style.hatch_style(HATCHES[label], color))

    # the legend of the hatchings, shared by both panels, on the gray of the sum
    set_handles = [
        Patch(facecolor=plot_style.NEUTRAL, linewidth=0, label=label,
              **plot_style.hatch_style(hatch, plot_style.NEUTRAL))
        for label, hatch in HATCHES.items()]

    # the axes are placed in inch on a canvas of the final size, so that a
    # length in pt (the bars) converts to data units once the limits are set
    fig = plt.figure(figsize=(plot_style.TEXT_WIDTH, FIGURE_HEIGHT))
    plot_style.set_canvas(fig, plot_style.TEXT_WIDTH, FIGURE_HEIGHT)
    ax_a, ax_b = (plot_style.axes_in_inches(fig, *PANELS[tag]) for tag in "ab")

    # (a) by step: the failure Markov chain stacked by logical manifestation,
    # and the actual QEC dynamics as a diamond on each bar
    ax = ax_a
    steps = np.arange(1, len(record["steps"]) + 1)
    plot_style.step_axis(ax)
    width, offsets = plot_style.group_offsets(ax, len(HATCHES), pitch=1.0)
    offsets = dict(zip(HATCHES, offsets))
    highest = 0.0
    for label, point in sets.items():
        by_step = point["rate_by_step_per_ms"]
        bottom = np.zeros(len(steps))
        # stacked from the last manifestation up: the bar reads from the top
        # down in the order of the legend
        for key, (_, color) in reversed(plot_style.MANIFESTATION_STYLES.items()):
            part = np.array(by_step["chain_by_manifestation"][key])
            bars(ax, steps + offsets[label], part, color, label, width, bottom)
            bottom += part
        ax.plot(steps + offsets[label], by_step[ACTUAL], **actual)
        highest = max(highest, bottom.max(), max(by_step[ACTUAL]))
    ax.set_ylim(0, (1 + TOP_MARGIN) * highest)
    # the steps after the first are low: the legend sits in the upper right
    ax.legend(
        handles=[Patch(facecolor=color, linewidth=0, label=text)
                 for text, color in plot_style.MANIFESTATION_STYLES.values()]
        + [Line2D([], [], label="actual QEC dynamics", **actual)],
        title="bars: failure Markov chain", alignment="left", loc="upper right",
        **plot_style.BAR_LEGEND)

    # (b) by mechanism: the analytical formula; the two numerically computed
    # failure rates on the group of the sum
    ax = ax_b
    names = list(plot_style.MECHANISM_GROUPS)
    groups = np.arange(len(names), dtype=float)
    ax.set_xlim(-0.6, len(names) - 0.4)
    width, offsets = plot_style.group_offsets(ax, len(HATCHES), pitch=1.0)
    offsets = dict(zip(HATCHES, offsets))
    colors = [plot_style.NEUTRAL] + [plot_style.MECHANISM_COLORS[name] for name in names[1:]]
    highest = 0.0
    for label, point in sets.items():
        values = plot_style.mechanism_group_values(point["failure_rate_per_ms"][FORMULA],
                                                   point["rate_by_mechanism_per_ms"])
        bars(ax, groups + offsets[label], values, colors, label, width)
        ax.plot(offsets[label], point["failure_rate_per_ms"][ACTUAL], **actual)
        ax.plot(offsets[label], point["failure_rate_per_ms"][CHAIN], **markov)
        highest = max(highest, *values, point["failure_rate_per_ms"][ACTUAL],
                      point["failure_rate_per_ms"][CHAIN])
    ax.set_ylim(0, (1 + TOP_MARGIN) * highest)
    ax.axvline(0.5, color=plot_style.LIGHT_GRAY, lw=plot_style.THIN_WIDTH)
    # the half width of a group, to which the line under a mode extends
    plot_style.mechanism_axis(ax, groups, names,
                              half_width=1.5 * width + plot_style.points_to_data(ax, 2.0))
    # the mechanisms are low next to their sum: the legend sits in the upper right
    ax.legend(
        handles=[Line2D([], [], label="failure Markov chain", **markov),
                 Line2D([], [], label="actual QEC dynamics", **actual)],
        title="bars: AEB", alignment="left", loc="upper right",
        **plot_style.BAR_LEGEND)

    # Both panels: the failure rate from 0 with the same ticks, one label of
    # one line each, since the two panels have their own scales; the panel
    # labels above the upper left corners, at the same height
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.yaxis.set_major_locator(MultipleLocator(RATE_TICK_STEP))
        ax.set_ylabel(r"failure rate (ms$^{-1}$)")
        plot_style.value_grid(ax)
        plot_style.panel_tag(ax, tag)
    # the hatchings in one row under the names of the steps
    plot_style.legend_in_inches(fig, set_handles, *LEGENDS["sets"], columns=len(HATCHES))

    FIGURE.parent.mkdir(exist_ok=True)
    plot_style.save_figure(fig, FIGURE, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from ancilla-comparison.json")
    parser.add_argument("--add-properties", action="store_true",
                        help="compute only the ancilla properties (static quantities and "
                             "rates) and add them to ancilla-comparison.json; no tree, no figure")
    args = parser.parse_args()
    if args.add_properties:
        record = json.loads(RESULTS_JSON.read_text())
        record["ancilla_properties"] = properties_record()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(f"wrote {RESULTS_JSON}")
        raise SystemExit
    if args.plot_only:
        record = json.loads(RESULTS_JSON.read_text())
    else:
        record = compute()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(f"wrote {RESULTS_JSON}")
    plot(record)
