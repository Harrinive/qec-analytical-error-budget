r"""Which physical mechanism produces which neglected process, step by step.

Evaluates the step-by-step magnitudes and coherence lower bounds of the
neglected processes for the cat-code QEC cycle at the parameters of Ofek et al.
with the full noise model, without decoherence, and with each decoherence
mechanism alone, plus few-level toy-step scaling checks. Writes
``neglected-process-mechanisms.json``.

Usage: ``python neglected-process-mechanisms.py`` (or ``--pairs-only``, ``--toy-checks``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed
import chencrafts.settings as cc_settings
from qec_aeb.QEC_graph import component_projector_map

# importing device_model sets the symmetric orthonormalization of the error
# states (paper convention) before any tree is built
import device_model
import model_chain
from step_magnitudes import (
    absent_transition_magnitude, leakage_operator, leakage_projector, most_seeping_state,
    restricted_spectrum, step_groups, unexpected_outcome_magnitude)

RESULTS_JSON = Path(__file__).resolve().parent / "neglected-process-mechanisms.json"

FULL, NO_DECOHERENCE = "full_noise_model", "no_decoherence"
SINGLE_MECHANISMS = [*model_chain.MECHANISMS, model_chain.ASSIGNMENT]
# computations: name -> mechanisms kept (None: the full noise model)
COMPUTATIONS = {FULL: None, NO_DECOHERENCE: []} | {name: [name] for name in SINGLE_MECHANISMS}
# pairs of mechanisms that act together in ``measurement_pairs``: photon loss
# with one ancilla jump
MECHANISM_PAIRS = [["photon_loss", "ancilla_relaxation"],
                   ["photon_loss", "ancilla_excitation"],
                   ["photon_loss", "ancilla_dephasing"]]
# the quantities of a step that are tabulated and attributed
QUANTITIES = ["leak_probability_mixed", "eps_leak", "coherence_lower_bound",
              "eps_err_traj", "eps_unexp", "eps_seep", "returned_coherence_lower_bound"]
# worker processes for the construction of the maps
NUM_CPUS = 1
# a difference between the full model and the model without decoherence
# below this value is rounding; no share is then defined
SHARE_FLOOR = 1e-12
# a state of the leakage subspace with a seepage probability below this value
# does not return; no returned coherence is then computed
SEEPAGE_FLOOR = 1e-14
# if the value without decoherence exceeds this part of the value of the full
# model, the quantity is attributed to the unitary dynamics
UNITARY_SHARE = 0.5

# states of the logical qubit for the coherence measure: the six eigenstates
# of the logical Pauli operators
CARDINAL_STATES = {
    "0": qt.basis(2, 0),
    "1": qt.basis(2, 1),
    "+": (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
    "-": (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
    "+i": (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),
    "-i": (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit(),
}

# toy checks: the grid of the jump probability and of the rotation angle
TOY_PROBABILITIES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
TOY_ANGLES = [1e-2, 3e-2, 1e-1]
TOY_MAGNITUDES = ["eps_leak", "eps_seep", "eps_leak_coh", "eps_seep_coh"]
# solver of the semidefinite program of the two coherence magnitudes
# (``chencrafts.cqed.dnorm_hp``: SCS, with the relative tolerance 1e-6)
TOY_SOLVER = "SCS"
# a toy map or magnitude below this value is zero up to rounding
TOY_ZERO = 1e-12

# progress bars would flood the log of a batch run
cc_settings.PROGRESSBAR_DISABLED = True


# ------------------------------------------------------------------ trees ----
def tree_without_self_kerr(kept: list[str]):
    r"""One-cycle tree with the resonator self-Kerr removed from the dynamics.

    The term :math:`K\hat a^{\dagger2}\hat a^2` is subtracted from the static
    Hamiltonian of the tree builder on every ancilla level, and the self-Kerr
    coefficient that the ideal one-loss operator uses for its mean rotation
    is set to zero, so the ideal error states are those of the changed
    Hamiltonian. :math:`\chi_1'` and the higher non-linearities of the
    dressed spectrum stay. Only the idle (the first step) of this tree is
    read.

    Parameters
    ----------
    kept : list of str
        Mechanisms that act (``model_chain.sweep_with_mechanisms``).

    Returns
    -------
    EvolutionTree
        Traversed actual evolution tree of one QEC cycle.
    """
    sweep, simulation_parameters = model_chain.sweep_with_mechanisms(kept)
    kerr = float(sweep["K_s"])

    def remove_self_kerr(builder) -> None:
        r"""Remove the self-Kerr term from the builder and from the ideal operators.

        Parameters
        ----------
        builder : qec_aeb.QEC_graph.FullCatTreeBuilder
            Tree builder of ``sweep``, before any dynamical map is built.

        Returns
        -------
        None
            The static Hamiltonian of ``builder`` and the self-Kerr
            coefficient stored in ``sweep`` are changed in place.
        """
        # K n(n-1) on every ancilla level, in rad/ns like the static Hamiltonian
        photons = qt.tensor(qt.num(builder.res_dim), qt.qeye(builder.qubit_dim))
        builder.static_hamiltonian = builder.static_hamiltonian - kerr * qt.Qobj(
            (photons * photons - photons).full(), dims=builder.static_hamiltonian.dims)
        # the ideal operators read the coefficient from the sweep
        sweep.sweep.store_data(K_s=0.0 * np.asarray(sweep.sweep["K_s"]))

    return device_model.build_tree_from_sweep(
        sweep, simulation_parameters, n_cycles=1, num_cpus=NUM_CPUS,
        prepare_builder=remove_self_kerr)


# ---------------------------------------------------------- small helpers ----


def trace_norm(operator: qt.Qobj) -> float:
    r"""Trace norm of an operator, the sum of its singular values.

    Parameters
    ----------
    operator : qutip.Qobj
        Any operator.

    Returns
    -------
    float
    """
    return float(np.linalg.svd(operator.full(), compute_uv=False).sum())


# ------------------------------------------------- quantities of one step ----


def returned_coherence_lower_bound(group: dict, most_seeping: qt.Qobj) -> dict:
    r"""Lower bound on the magnitude of the returned error--leakage coherence at one step.

    For an error--leakage coherence :math:`\hat X` before the step, the
    returned part is :math:`\hat P_{\mathrm{err}}\mathcal E(\hat X)\hat
    P_{\mathrm{err}}`. Its trace norm, added over the outcomes of the step
    and divided by the trace norm of :math:`\hat X`, is at most the
    magnitude of the return: the diamond norm of the projected dynamical
    maps that return the coherence, with the outcomes kept apart. The
    largest value over the tested coherences is therefore a lower bound on
    that magnitude.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.
    most_seeping : qutip.Qobj
        Ket of the leakage subspace with the largest seepage probability.

    Returns
    -------
    dict
        ``largest`` (the lower bound) and ``largest_state`` (the label of
        the coherence that gives it: ``a:0``, ``a:+``, ... for the state of
        error space ``a``, followed by the phase ``0`` or ``pi/2``).
    """
    by_coherence = {}
    for index, encoder in enumerate(group["node"].ideal_encoders()):
        for label, ket in CARDINAL_STATES.items():
            # |e><l| has the trace norm 1, so the Hermitian coherence built
            # from it is divided by 2
            outer = (encoder * ket) * most_seeping.dag()
            for phase_label, phase in (("0", 1.0), ("pi/2", 1j)):
                coherence = (phase * outer + (phase * outer).dag()) / 2
                total = 0.0
                for edge in group["edges"]:
                    after = edge.final_state.joint_error_space_projector()
                    image = cqed.superop_evolve(edge.physical_map(), coherence)
                    total += trace_norm(after * image * after)
                by_coherence[f"{index}:{label}, phase {phase_label}"] = total
    largest_state = max(by_coherence, key=by_coherence.get)
    return {"largest": by_coherence[largest_state], "largest_state": largest_state}


def coherence_lower_bound(group: dict, most_leaking: qt.Qobj) -> dict:
    r"""Lower bound on the magnitude of the created error--leakage coherence at one step.

    For a state :math:`\hat\rho` of the joint error space, the created
    coherence is :math:`\hat P_{\mathrm{leak}}\mathcal E(\hat\rho)\hat
    P_{\mathrm{err}}+\mathrm{h.c.}` Its trace norm, added over the outcomes
    of the step, is at most the magnitude of the creation: the diamond norm
    of the projected dynamical maps that create the coherence, with the
    outcomes kept apart. The largest value over the tested states is
    therefore a lower bound on that magnitude.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.
    most_leaking : qutip.Qobj
        Ket of the joint error space with the largest leakage probability.

    Returns
    -------
    dict
        ``largest`` (the lower bound) and ``largest_state`` (the label of
        the state that gives it: ``mixed joint``, ``mixed a`` and ``a:0``,
        ``a:+``, ... for error space ``a``, or ``largest leakage``).
    """
    node = group["node"]
    error = node.joint_error_space_projector()
    # the tested states of the joint error space
    states = {"mixed joint": error / error.tr(), "largest leakage": qt.ket2dm(most_leaking)}
    for index, encoder in enumerate(node.ideal_encoders()):
        states[f"mixed {index}"] = encoder * encoder.dag() / 2
        for label, ket in CARDINAL_STATES.items():
            states[f"{index}:{label}"] = qt.ket2dm(encoder * ket)
    by_state = {}
    for label, rho in states.items():
        total = 0.0
        for edge in group["edges"]:
            final = edge.final_state
            after, leak = final.joint_error_space_projector(), leakage_projector(final)
            image = cqed.superop_evolve(edge.physical_map(), rho)
            total += trace_norm(leak * image * after + after * image * leak)
        by_state[label] = total
    largest_state = max(by_state, key=by_state.get)
    return {"largest": by_state[largest_state], "largest_state": largest_state}


def step_record(group: dict) -> dict:
    r"""Every tabulated quantity of one step for one measurement record before it.

    Parameters
    ----------
    group : dict
        Entry of :func:`step_groups`.

    Returns
    -------
    dict
        ``step``, ``step_name``, ``record``, and the quantities of
        :data:`QUANTITIES` (module docstring), with
        ``coherence_lower_bound_state``, the state that gives the lower
        bound, and ``returned_coherence_lower_bound_state``, the coherence
        that gives the lower bound on the returned coherence. The two
        quantities of the return are ``None`` for the idle.
    """
    node = group["node"]
    error = node.joint_error_space_projector()
    operator = leakage_operator(group)
    values, vectors = restricted_spectrum(operator, node)
    most_leaking = qt.Qobj(vectors[:, -1], dims=[error.dims[0], [1] * len(error.dims[0])])
    coherence = coherence_lower_bound(group, most_leaking)
    # no step precedes the idle, so nothing has leaked before it
    seepage, returned = None, {"largest": None, "largest_state": None}
    if group["step"] > 1:
        seepage, most_seeping = most_seeping_state(group)
        if seepage > SEEPAGE_FLOOR:
            returned = returned_coherence_lower_bound(group, most_seeping)
        else:
            returned = {"largest": 0.0, "largest_state": None}
    return {
        "step": group["step"],
        "step_name": model_chain.STEP_NAMES[group["step"] - 1],
        "record": group["record"],
        "leak_probability_mixed": float(np.real((operator * error).tr() / error.tr())),
        # a largest eigenvalue can come out at -1e-17
        "eps_leak": float(max(values[-1], 0.0)),
        "coherence_lower_bound": coherence["largest"],
        "coherence_lower_bound_state": coherence["largest_state"],
        "eps_err_traj": absent_transition_magnitude(group),
        "eps_unexp": unexpected_outcome_magnitude(group),
        "eps_seep": seepage,
        "returned_coherence_lower_bound": returned["largest"],
        "returned_coherence_lower_bound_state": returned["largest_state"],
    }


def row_label(row: dict) -> str | None:
    r"""Label of a step and of the measurement record before it in the tables.

    Parameters
    ----------
    row : dict
        Output of :func:`step_record`.

    Returns
    -------
    str or None
        The name of the step, followed by ``, after outcome r`` at the two
        steps that follow the measurement. ``None`` for the conditional
        pulse after the outcome 0, where no pulse is applied.
    """
    if not row["record"]:
        return row["step_name"]
    if row["step"] == device_model.STEPS_PER_CYCLE and row["record"] == [0]:
        return None
    return f"{row['step_name']}, after outcome {row['record'][0]}"


# --------------------------------------------------- tables and shares ----
def quantity_tables(computations: dict) -> dict:
    r"""The quantities as tables: quantity -> step (and record) -> computation.

    Parameters
    ----------
    computations : dict
        Name of a computation to its record (``steps``: the outputs of
        :func:`step_record`).

    Returns
    -------
    dict
        For each quantity of :data:`QUANTITIES`, the rows of
        :func:`row_label` with one value for each computation.
    """
    tables = {name: {} for name in QUANTITIES}
    for computation, record in computations.items():
        for row in record["steps"]:
            label = row_label(row)
            if label is None:
                continue
            for name in QUANTITIES:
                if row[name] is not None:
                    tables[name].setdefault(label, {})[computation] = row[name]
    return tables


def attribution(tables: dict) -> dict:
    r"""Share of each mechanism in each quantity at each step.

    Parameters
    ----------
    tables : dict
        Output of :func:`quantity_tables`.

    Returns
    -------
    dict
        quantity -> step -> ``full_noise_model``, ``no_decoherence`` (the
        two values), ``no_decoherence_over_full``, ``sum_of_shares``
        (mechanisms alone, added; 1 if the mechanisms add up), ``shares``
        (:math:`(x_{\mathrm m}-x_0)/(x-x_0)` for each mechanism), and
        ``order`` (the mechanisms with a share above 1%, the largest first;
        ``["unitary dynamics"]`` where the value without decoherence is more
        than half of the value of the full model).
        Where the full model and the model without decoherence agree within
        rounding, no share is defined and ``shares`` is empty.
    """
    result = {}
    for name, table in tables.items():
        result[name] = {}
        for label, values in table.items():
            full, closed = values[FULL], values[NO_DECOHERENCE]
            excess = full - closed
            shares = ({} if abs(excess) < SHARE_FLOOR else
                      {run: (values[run] - closed) / excess for run in SINGLE_MECHANISMS})
            # where the unitary dynamics give most of the value, the shares are
            # ratios of small differences and do not order the mechanisms
            unitary = abs(full) > SHARE_FLOOR and closed / full > UNITARY_SHARE
            result[name][label] = {
                FULL: full, NO_DECOHERENCE: closed,
                "no_decoherence_over_full": closed / full if abs(full) > SHARE_FLOOR else None,
                "sum_of_shares": sum(shares.values()) if shares else None,
                "shares": shares,
                "order": ["unitary dynamics"] if unitary else [
                    run for run in sorted(shares, key=shares.get, reverse=True)
                    if shares[run] > 0.01],
            }
    return result


def idle_row(graph) -> dict:
    r"""Quantities of the idle (the first step) of a tree.

    Parameters
    ----------
    graph : EvolutionTree
        Traversed actual evolution tree.

    Returns
    -------
    dict
        Output of :func:`step_record` for the first step.
    """
    return step_record(next(group for group in step_groups(graph) if group["step"] == 1))


def measurement_pairs() -> dict:
    r"""Quantities of the measurement when two mechanisms act together.

    Returns
    -------
    dict
        For each pair of :data:`MECHANISM_PAIRS`, named ``a+b``, the
        ``eps_err_traj``, ``eps_unexp``, and ``eps_leak`` of the measurement
        (the step 5; one entry for each measurement record before it, which is
        one here), computed as in the single-mechanism trees.
    """
    result = {}
    for kept in MECHANISM_PAIRS:
        sweep, simulation_parameters = model_chain.sweep_with_mechanisms(kept)
        graph = device_model.build_tree_from_sweep(
            sweep, simulation_parameters, num_cpus=NUM_CPUS)
        rows = [step_record(group) for group in step_groups(graph)]
        result["+".join(kept)] = [
            {key: row[key] for key in ("eps_err_traj", "eps_unexp", "eps_leak")}
            for row in rows if row["step"] == 5]
        print("+".join(kept), result["+".join(kept)], flush=True)
    return result


# ------------------------------------------------------------ toy checks ----
def toy_step(jump: qt.Qobj, p: float, rotation: qt.Qobj | None = None) -> qt.Qobj:
    r"""Dynamical map of a toy step: at most one jump, and a rotation after the jump.

    The jump :math:`\hat L` (with :math:`\hat L^\dagger\hat L\le\hat 1`)
    occurs with the probability :math:`p\langle\hat L^\dagger\hat L\rangle`:
    the Kraus operators are :math:`\sqrt p\,\hat L` and
    :math:`(\hat 1-p\,\hat L^\dagger\hat L)^{1/2}`. With a rotation
    :math:`\hat R`, the first is :math:`\sqrt p\,\hat R\hat L`: the
    rotation follows the jump only, as the rotation relative to the ideal
    operation does when the ideal operation is that of the state without a
    jump.

    Parameters
    ----------
    jump : qutip.Qobj
        The jump operator :math:`\hat L`.
    p : float
        The jump probability of a state with :math:`\langle\hat
        L^\dagger\hat L\rangle=1`.
    rotation : qutip.Qobj or None
        The unitary :math:`\hat R`; ``None`` for no rotation.

    Returns
    -------
    qutip.Qobj
        The dynamical map (qutip ``super`` representation).
    """
    no_jump = (qt.qeye(jump.dims[0]) - p * jump.dag() * jump).sqrtm()
    after_jump = jump if rotation is None else rotation * jump
    return qt.kraus_to_super([no_jump, np.sqrt(p) * after_jump])


def toy_coherence_dnorm(superop: qt.Qobj) -> float:
    r"""Diamond norm of a toy projected dynamical map that is not completely positive.

    The maps that create and that return error--leakage coherence are not
    completely positive, so their diamond norm is the solution of a
    semidefinite program; they preserve Hermiticity, so the smaller program
    of ``chencrafts.cqed.dnorm_hp`` applies. The norm is
    homogeneous: the map is divided by its Frobenius norm before the
    program and the result is multiplied by it, so the accuracy of the
    solver is relative and a magnitude of :math:`10^{-6}` is resolved.

    Parameters
    ----------
    superop : qutip.Qobj
        The projected dynamical map on a space of a few levels.

    Returns
    -------
    float
        The diamond norm; exactly 0 for a map below :data:`TOY_ZERO`.
    """
    scale = float(np.linalg.norm(superop.full()))
    # a map that vanishes up to rounding has no direction to rescale
    if scale < TOY_ZERO:
        return 0.0
    return scale * cqed.dnorm_hp(superop / scale, solver=TOY_SOLVER)["value"]


def toy_magnitudes(step: qt.Qobj, error_states: list[qt.Qobj]) -> dict:
    r"""The four magnitudes of leakage, seepage, and error--leakage coherence of a toy step.

    The projected dynamical maps are :math:`\mathcal P^B\mathcal E\mathcal
    P^A` with the projector maps of ``chencrafts`` onto the component within
    the joint error space, the component within the leakage subspace, and
    the error--leakage coherence. The toy step has no measurement, so each
    magnitude is one diamond norm.

    Parameters
    ----------
    step : qutip.Qobj
        The dynamical map :math:`\mathcal E` of the step.
    error_states : list of qutip.Qobj
        Kets that span the joint error space, one for each
        (one-dimensional) error space; the same before and after the step.

    Returns
    -------
    dict
        ``eps_leak`` and ``eps_seep`` (completely positive maps: the
        largest probability), ``eps_leak_coh`` and ``eps_seep_coh`` (the
        creation and the return of the coherence: semidefinite program).
    """
    onto_error = component_projector_map(error_states, model_chain.WITHIN_ERROR)
    onto_leakage = component_projector_map(error_states, model_chain.WITHIN_LEAKAGE)
    onto_coherence = component_projector_map(error_states, model_chain.COHERENCE)
    return {
        "eps_leak": model_chain.completely_positive_dnorm(onto_leakage * step * onto_error),
        "eps_seep": model_chain.completely_positive_dnorm(onto_error * step * onto_leakage),
        "eps_leak_coh": toy_coherence_dnorm(onto_coherence * step * onto_error),
        "eps_seep_coh": toy_coherence_dnorm(onto_error * step * onto_coherence),
    }


def toy_power_law(rows: list[dict], name: str, with_theta: bool) -> dict:
    r"""Power law :math:`c\,p^a\theta^b` fitted to one magnitude over the grid.

    Parameters
    ----------
    rows : list of dict
        One entry for each point of the grid, with ``p``, ``theta`` (or
        ``None``), and the magnitudes.
    name : str
        Key of the magnitude.
    with_theta : bool
        Whether the angle is a variable of the fit.

    Returns
    -------
    dict
        ``zero`` (True if the magnitude is below :data:`TOY_ZERO` at every
        point; then only ``largest_value`` follows), ``power_p``,
        ``power_theta`` (``None`` without an angle), ``prefactor``, and
        ``largest_relative_deviation`` of the values from the fitted law.
    """
    values = np.array([row[name] for row in rows])
    if values.max() < TOY_ZERO:
        return {"zero": True, "largest_value": float(values.max())}
    # least squares for log(value) = log(c) + a log(p) + b log(theta)
    columns = [np.ones(len(rows)), np.log([row["p"] for row in rows])]
    if with_theta:
        columns.append(np.log([row["theta"] for row in rows]))
    design = np.column_stack(columns)
    solution = np.linalg.lstsq(design, np.log(values), rcond=None)[0]
    deviation = np.abs(np.exp(design @ solution) / values - 1).max()
    return {"zero": False, "power_p": float(solution[1]),
            "power_theta": float(solution[2]) if with_theta else None,
            "prefactor": float(np.exp(solution[0])),
            "largest_relative_deviation": float(deviation)}


def toy_checks() -> dict:
    r"""Toy checks of the return of error--leakage coherence and of seepage by dephasing.

    Three kinds of toy steps. (i) An ancilla qubit whose error space is
    :math:`|+\rangle` and whose leakage subspace is :math:`|-\rangle`, as
    during the parity mapping, with one jump: into the leakage subspace
    (:math:`|-\rangle\langle+|`), a relaxation (:math:`|0\rangle\langle1|`,
    which takes either state into the superposition :math:`|0\rangle`), and
    a phase flip (:math:`\hat\sigma_z`). The phase flip is repeated with the
    error space :math:`\cos(\pi/8)|0\rangle+\sin(\pi/8)|1\rangle`, the
    ancilla halfway through a :math:`\pi/2` pulse, where it takes the state
    into a superposition of the error space and the leakage subspace.
    (ii) Four levels, two error spaces :math:`|e_0\rangle`,
    :math:`|e_1\rangle` and two leaked states :math:`|l_0\rangle`,
    :math:`|l_1\rangle`: a jump from index 0 to index 1, followed by the
    rotation :math:`\exp(-\mathrm i\theta\hat\sigma/2)` with
    :math:`\hat\sigma=|e_1\rangle\langle l_1|+\mathrm{h.c.}`; the jump acts
    on the leaked state as well (:math:`|e_1\rangle\langle e_0|+|l_1\rangle
    \langle l_0|`, as a photon loss does) or on the error space only.

    Returns
    -------
    dict
        ``grid`` (the values of :math:`p` and :math:`\theta`), ``solver``,
        and ``cases``: for each case its ``statement`` (``A``: the return
        of the coherence; ``B``: seepage by dephasing), ``levels``,
        ``error_space``, ``jump_operator``, ``rotation``, the ``values``
        on the grid, and the ``fit`` of :func:`toy_power_law` for each of
        the four magnitudes.
    """
    zero, one = qt.basis(2, 0), qt.basis(2, 1)
    plus, minus = (zero + one).unit(), (zero - one).unit()
    tilted = np.cos(np.pi / 8) * zero + np.sin(np.pi / 8) * one
    e0, l0, e1, l1 = (qt.basis(4, index) for index in range(4))
    # case -> statement, levels, error space, jump operator (label, operator),
    # kets of the error spaces, whether a rotation follows the jump
    cases = {
        "jump_into_leakage": ("A", "ancilla qubit", "|+>", "|-><+|",
                              minus * plus.dag(), [plus], False),
        "jump_into_superposition_relaxation": ("A", "ancilla qubit", "|+>", "|0><1|",
                                               zero * one.dag(), [plus], False),
        "jump_into_superposition_phase_flip": (
            "A", "ancilla qubit", "cos(pi/8)|0> + sin(pi/8)|1>", "sigma_z",
            qt.sigmaz(), [tilted], False),
        "rotation_after_jump": ("A", "e0, l0, e1, l1", "e0, e1", "|e1><e0| + |l1><l0|",
                                e1 * e0.dag() + l1 * l0.dag(), [e0, e1], True),
        "rotation_after_jump_from_error_space_only": (
            "A", "e0, l0, e1, l1", "e0, e1", "|e1><e0|", e1 * e0.dag(), [e0, e1], True),
        "phase_flip": ("B", "ancilla qubit", "|+>", "sigma_z", qt.sigmaz(), [plus], False),
    }
    exchange = e1 * l1.dag() + l1 * e1.dag()
    result = {}
    for name, (statement, levels, space, label, jump, error_states, rotated) in cases.items():
        rows = []
        for p in TOY_PROBABILITIES:
            for theta in (TOY_ANGLES if rotated else [None]):
                rotation = (-0.5j * theta * exchange).expm() if rotated else None
                rows.append({"p": p, "theta": theta}
                            | toy_magnitudes(toy_step(jump, p, rotation), error_states))
        result[name] = {
            "statement": statement, "levels": levels, "error_space": space,
            "jump_operator": label,
            "rotation": "exp(-i theta (|e1><l1| + h.c.)/2), after the jump only" if rotated else None,
            "fit": {key: toy_power_law(rows, key, rotated) for key in TOY_MAGNITUDES},
            "values": rows,
        }
        print(name, {key: fit for key, fit in result[name]["fit"].items()}, flush=True)
    return {
        "description": "Diamond-norm magnitudes and power-law fits for few-level toy steps with a jump of probability p and an optional rotation by angle theta.",
        "grid": {"p": TOY_PROBABILITIES, "theta": TOY_ANGLES},
        "solver": TOY_SOLVER,
        "cases": result,
    }


if __name__ == "__main__":
    import sys
    if "--toy-checks" in sys.argv:
        # add the toy checks to the existing record; every other entry is kept
        record = json.loads(RESULTS_JSON.read_text())
        record["toy_checks"] = toy_checks()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(f"wrote {RESULTS_JSON}")
        raise SystemExit
    if "--pairs-only" in sys.argv:
        # add the pairs to the existing record; every other entry is kept
        record = json.loads(RESULTS_JSON.read_text())
        record["measurement_pairs"] = measurement_pairs()
        RESULTS_JSON.write_text(json.dumps(record, indent=1))
        print(f"wrote {RESULTS_JSON}")
        raise SystemExit
    computations = {}
    # one tree at a time; each tree is released before the next is built
    for name, kept in COMPUTATIONS.items():
        t0 = time.time()
        sweep, simulation_parameters = model_chain.sweep_with_mechanisms(kept)
        graph = device_model.build_tree_from_sweep(
            sweep, simulation_parameters, num_cpus=NUM_CPUS)
        computations[name] = {
            "mechanisms_kept": "all" if kept is None else kept,
            "steps": [step_record(group) for group in step_groups(graph)],
            "infidelity_actual_QEC_dynamics":
                device_model.one_cycle_infidelities(graph)["actual_QEC_dynamics"],
        }
        computations[name]["seconds"] = time.time() - t0
        print(f"{name}: {computations[name]['seconds']:.0f} s", flush=True)

    # photon loss alone, with and without the resonator self-Kerr: the idle
    with_kerr = next(row for row in computations["photon_loss"]["steps"] if row["step"] == 1)
    without_kerr = idle_row(tree_without_self_kerr(["photon_loss"]))
    self_kerr = {
        "description": "idle with photon loss alone, with the resonator self-Kerr and "
                       "with it removed from the Hamiltonian and from the ideal operators",
        "with_self_kerr": with_kerr,
        "without_self_kerr": without_kerr,
        "eps_leak_ratio_without_over_with": without_kerr["eps_leak"] / with_kerr["eps_leak"],
        "leak_probability_mixed_ratio_without_over_with":
            without_kerr["leak_probability_mixed"] / with_kerr["leak_probability_mixed"],
    }
    print(f"idle, photon loss alone: eps_leak {with_kerr['eps_leak']:.3e} with the "
          f"self-Kerr, {without_kerr['eps_leak']:.3e} without", flush=True)

    tables = quantity_tables(computations)
    record = {
        "description": "Step-by-step magnitudes and coherence lower bounds of the neglected processes for the cat-code cycle at the parameters of Ofek et al. under the full noise model, no decoherence, and single decoherence mechanisms.",
        "mechanisms": {name: {"sweep_key": key, "ancilla_labels": labels}
                       for name, (key, labels) in model_chain.MECHANISMS.items()}
        | {model_chain.ASSIGNMENT: {"parameters": model_chain.ASSIGNMENT_KEYS}},
        "steps": model_chain.STEP_NAMES,
        "quantities": QUANTITIES,
        "tables": tables,
        "attribution": attribution(tables),
        "idle_photon_loss_self_kerr": self_kerr,
        "measurement_pairs": measurement_pairs(),
        # the full run writes the whole record, so the toy checks are part of it
        "toy_checks": toy_checks(),
        "computations": computations,
        "settings": dict(
            dynamics_resonator_dimension=device_model.DYNAMICS_RESONATOR_DIMENSION,
            dynamics_ancilla_dimension=device_model.DYNAMICS_ANCILLA_DIMENSION,
        ),
        "environment": device_model.environment_record(NUM_CPUS),
    }
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")
