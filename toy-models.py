r"""Coherent and incoherent errors in three minimal models: exact discrepancies against their leading orders.

Evaluates the chain of models for three minimal multi-step systems (logical
qubit with leaked levels, logical qubit with a syndrome qubit, and logical
qubit alone) under coherent and incoherent errors of probability :math:`p` per
step, and compares each discrepancy with its leading-order power law in
:math:`N` and :math:`p`. Writes ``toy-models.json``.

Usage: ``python toy-models.py``.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import qutip as qt

import chencrafts.cqed as cqed

RESULTS_JSON = Path(__file__).resolve().parent / "toy-models.json"

# numbers of steps and error probabilities per step of the comparison; the
# largest N p is 0.02, so that the leading order holds everywhere
STEP_NUMBERS = (2, 3, 5, 10, 20)
ERROR_PROBABILITIES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
# an expression is confirmed if, for every N, the fitted power is within
# POWER_TOLERANCE of its power and the ratio extrapolated to p -> 0 is within
# RATIO_TOLERANCE of one; a contribution stated as zero must stay below
# ZERO_TOLERANCE (rounding of process fidelities of order one)
POWER_TOLERANCE = 0.01
RATIO_TOLERANCE = 2e-3
ZERO_TOLERANCE = 1e-13
# reductions of the chain in their order, with the pair of models each connects
REDUCTIONS = {
    "delta_unexp": ("actual", "i"),
    "delta_leak-coh": ("i", "ii"),
    "delta_leak": ("ii", "iii"),
    "delta_err-coh": ("iii", "iv"),
    "delta_err-traj": ("iv", "v"),
    "delta_logi-coh": ("v", "twirled"),
    "delta_logi-err": ("twirled", "failure_Markov_chain"),
}
PAULI = dict(I=qt.qeye(2), X=qt.sigmax(), Y=qt.sigmay(), Z=qt.sigmaz())


def transfer(dimension: int, to: int, source: int) -> qt.Qobj:
    r"""Operator :math:`|\mathrm{to}\rangle\langle\mathrm{source}|` of a single system.

    Parameters
    ----------
    dimension : int
        Dimension of the Hilbert space.
    to, source : int
        Indices of the two basis states.

    Returns
    -------
    qt.Qobj
        The operator.
    """
    return qt.basis(dimension, to) * qt.basis(dimension, source).dag()


def leaked_level_model(error: str, reading: str, p: float) -> dict:
    r"""A logical qubit with two leaked levels: the step map and the error spaces.

    The logical basis states are :math:`|0\rangle,|1\rangle`; the leaked
    levels are :math:`|2\rangle` and :math:`|3\rangle`. The error acts
    between :math:`|0\rangle` and :math:`|2\rangle` and, in the same way,
    between :math:`|1\rangle` and :math:`|3\rangle`, so that both logical
    basis states leak with the same probability.

    Parameters
    ----------
    error : str
        ``"coherent"``: the rotation :math:`\exp(-\mathrm i\varphi\hat G)`
        with :math:`\hat G=|2\rangle\langle0|+|3\rangle\langle1|+\mathrm{h.c.}`
        and :math:`\sin^2\varphi=p`. ``"incoherent"``: see ``reading``.
    reading : str
        Reading of the incoherent error (ignored for the coherent one).
        ``"jumps_in_both_directions"``: Kraus operators
        :math:`\sqrt p\,|2\rangle\langle0|`, :math:`\sqrt p\,|3\rangle\langle1|`,
        their adjoints, and :math:`\sqrt{1-p}\,\hat I`; a jump tells which
        logical basis state leaked.
        ``"joint_jumps"``: Kraus operators
        :math:`\sqrt p\,(|2\rangle\langle0|+|3\rangle\langle1|)`, its
        adjoint, and :math:`\sqrt{1-p}\,\hat I`; the leaked levels keep the
        logical information.
        ``"master_equation"``: the four jump operators of
        ``"jumps_in_both_directions"`` with equal rates, for the time after
        which the population :math:`p` of :math:`|0\rangle` has moved to
        :math:`|2\rangle`.
    p : float
        Population transferred from each logical basis state to its leaked
        level in one step.

    Returns
    -------
    dict
        ``step`` (superoperator of one step), ``error_spaces`` (one matrix for
        each error space, its error states as columns; the first is the error
        space of the ideal protocol), and ``ideal`` (indices of the error
        spaces that the ideal protocol occupies).
    """
    # leakage of each logical basis state to its own leaked level, and seepage back
    leaks = [transfer(4, 2, 0), transfer(4, 3, 1)]
    seeps = [leak.dag() for leak in leaks]
    no_jump = np.sqrt(1 - p) * qt.qeye(4)
    if error == "coherent":
        generator = sum(leaks) + sum(seeps)
        step = qt.to_super((-1j * np.arcsin(np.sqrt(p)) * generator).expm())
    elif reading == "jumps_in_both_directions":
        step = qt.kraus_to_super([np.sqrt(p) * jump for jump in leaks + seeps] + [no_jump])
    elif reading == "joint_jumps":
        step = qt.kraus_to_super([np.sqrt(p) * sum(leaks), np.sqrt(p) * sum(seeps), no_jump])
    elif reading == "master_equation":
        # each pair of levels with equal rates in both directions: the
        # transferred population after the time t is [1 - exp(-2 t)] / 2 at unit rate
        t_step = -np.log(1 - 2 * p) / 2
        step = (qt.liouvillian(0 * qt.qeye(4), leaks + seeps) * t_step).expm()
    else:
        raise ValueError(f"unknown reading {reading}")
    return dict(step=step, error_spaces=[np.eye(4)[:, :2]], ideal=[0])


def syndrome_qubit_model(error: str, reading: str, p: float) -> dict:
    r"""A logical qubit with one syndrome qubit: the step map and the error spaces.

    The Hilbert space is logical qubit :math:`\otimes` syndrome qubit. The
    error state :math:`|\ell_e\rangle=|\ell\rangle\otimes|e\rangle` belongs to
    the error space of the syndrome :math:`e\in\{0,1\}`; both error spaces
    belong to the same measurement record during the steps, and the ideal
    protocol stays in :math:`e=0`. The error operator is
    :math:`\hat Z\otimes\hat X`: it flips the syndrome qubit and applies a
    logical :math:`Z`.

    Parameters
    ----------
    error : str
        ``"coherent"``: :math:`\exp(-\mathrm i\varphi\,\hat Z\otimes\hat X)`
        with :math:`\sin^2\varphi=p`. ``"incoherent"``: the error operator
        with probability :math:`p`.
    reading : str
        Only ``"random_error_operator"`` (ignored for the coherent error).
    p : float
        Population transferred to the other error space in one step.

    Returns
    -------
    dict
        As :func:`leaked_level_model`.
    """
    error_operator = qt.tensor(PAULI["Z"], PAULI["X"])
    if error == "coherent":
        step = qt.to_super((-1j * np.arcsin(np.sqrt(p)) * error_operator).expm())
    elif reading == "random_error_operator":
        step = qt.kraus_to_super(
            [np.sqrt(1 - p) * qt.tensor(PAULI["I"], PAULI["I"]), np.sqrt(p) * error_operator])
    else:
        raise ValueError(f"unknown reading {reading}")
    # error states |l> (x) |e> as columns, for e = 0 and e = 1
    error_spaces = [np.kron(np.eye(2), np.eye(2)[:, [e]]) for e in (0, 1)]
    return dict(step=step, error_spaces=error_spaces, ideal=[0])


def logical_qubit_model(error: str, reading: str, p: float) -> dict:
    r"""A logical qubit alone: the step map and its single error space.

    Parameters
    ----------
    error : str
        ``"coherent"``: :math:`\exp(-\mathrm i\varphi\hat X/2)` with
        :math:`\sin^2(\varphi/2)=p`. ``"incoherent"``: see ``reading``.
    reading : str
        ``"three_Pauli_operators_equally"``: each of the three logical Pauli
        operators with probability :math:`p/3`. ``"one_Pauli_operator"``:
        :math:`\hat X` with probability :math:`p`.
    p : float
        Logical-error probability of one step.

    Returns
    -------
    dict
        As :func:`leaked_level_model`.
    """
    if error == "coherent":
        step = qt.to_super((-1j * np.arcsin(np.sqrt(p)) * PAULI["X"]).expm())
    elif reading == "three_Pauli_operators_equally":
        step = qt.kraus_to_super([np.sqrt(1 - p) * PAULI["I"]]
                                 + [np.sqrt(p / 3) * PAULI[name] for name in "XYZ"])
    elif reading == "one_Pauli_operator":
        step = qt.kraus_to_super([np.sqrt(1 - p) * PAULI["I"], np.sqrt(p) * PAULI["X"]])
    else:
        raise ValueError(f"unknown reading {reading}")
    return dict(step=step, error_spaces=[np.eye(2)], ideal=[0])


def chain_of_models(step: qt.Qobj, error_spaces: list, ideal: list, n_steps: int) -> dict:
    r"""Process fidelity of every model of the chain, for identical steps without a measurement.

    Parameters
    ----------
    step : qt.Qobj
        Dynamical map of one step (superoperator, column stacking).
    error_spaces : list of np.ndarray
        For each error space, the matrix whose columns are its error states
        :math:`|\ell_{\mathbf e}\rangle`. All belong to the one measurement
        record of the steps; the first is the logical subspace at the start.
    ideal : list of int
        Indices of the error spaces of the ideal evolution tree. The steps
        are no error-accumulation steps, so its only error-space transitions
        are from each of these error spaces to itself. The minimal models
        have one, the logical subspace, so the logical evolution tree is one
        sequence of identical SELMs.
    n_steps : int
        Number of steps :math:`N`.

    Returns
    -------
    dict
        Process fidelity of ``actual``, ``i``, ``ii``, ``iii``, ``iv``,
        ``v``, ``twirled``, the final success probability
        ``failure_Markov_chain``, and ``twirled_minus_failure_Markov_chain``,
        the difference of the last two evaluated without cancellation
        (module docstring).
    """
    if len(ideal) != 1:
        raise ValueError("the minimal models have one ideal error space")
    dims = step.dims[0][0]
    kets = [[qt.Qobj(space[:, [column]], dims=[dims, [1] * len(dims)])
             for column in range(space.shape[1])] for space in error_spaces]
    # projectors of the individual error spaces, of the joint error space, and
    # of the leakage subspace, as maps rho -> P rho P
    onto_space = [qt.to_super(cqed.projector_w_basis(basis)) for basis in kets]
    joint = [ket for basis in kets for ket in basis]
    onto_err = qt.to_super(cqed.projector_w_basis(joint))
    leak_projector = qt.qeye(dims) - cqed.projector_w_basis(joint)
    onto_leak = qt.to_super(leak_projector)

    # what each model keeps of the dynamical map of one step
    kept = {
        # no measurement during the steps: no unexpected measurement record
        "actual": step,
        "i": step,
        # the components err and leak, without the error--leakage coherence
        "ii": sum(onto_B * step * onto_A
                  for onto_A in (onto_err, onto_leak) for onto_B in (onto_err, onto_leak)),
        "iii": cqed.projected_superop(step, joint),
        # every error-space transition; the product over the steps is the sum
        # over the error-space histories
        "iv": sum(onto_e * step * onto_g for onto_e in onto_space for onto_g in onto_space),
        # the transitions of the ideal protocol: each ideal error space to itself
        "v": sum(cqed.projected_superop(step, kets[index]) for index in ideal),
    }
    fidelity = {}
    for name, kept_map in kept.items():
        protocol = np.linalg.matrix_power(kept_map.full(), n_steps)
        # the state starts in the logical subspace (the first error space); the
        # intermediate recovery map of each error space reads the logical
        # information after the last step
        fidelity[name] = sum(
            cqed.subspace_process_fidelity(protocol, error_spaces[0], space)
            for space in error_spaces)

    # the SELM of the ideal error space, a map on the logical qubit, and its
    # Pauli twirl; the twirled model composes the twirled SELMs
    encoder = qt.Qobj(error_spaces[ideal[0]], dims=[dims, [2]])
    twirled = cqed.Pauli_twirl(cqed.superop_between_spaces(step, encoder)).full()
    fidelity["twirled"] = float(np.real(np.trace(np.linalg.matrix_power(twirled, n_steps)))) / 4

    # the failure Markov chain starts in the success state of the logical
    # subspace, which is the first ideal error space; the SELM of each step is
    # the same, so the final success probability is a power
    g = cqed.subspace_process_fidelity(step, error_spaces[ideal[0]], error_spaces[ideal[0]])
    fidelity["failure_Markov_chain"] = g ** n_steps

    # The twirled SELM is g_tw id + D, with D the twirled logical-error part
    # (no identity entry); g_tw id commutes with D, so the binomial expansion of
    # its N-th power holds, and the term with one D has zero process fidelity.
    # The terms with two or more D are the twirled model minus the failure
    # Markov chain, without the cancellation of two numbers close to one
    g_tw = float(np.real(np.trace(twirled))) / 4
    error_part = twirled - g_tw * np.eye(4)
    fidelity["twirled_minus_failure_Markov_chain"] = sum(
        comb(n_steps, k) * g_tw ** (n_steps - k)
        * float(np.real(np.trace(np.linalg.matrix_power(error_part, k)))) / 4
        for k in range(2, n_steps + 1))
    # the direct difference agrees up to the rounding of the two fidelities
    direct = fidelity["twirled"] - fidelity["failure_Markov_chain"]
    if abs(direct - fidelity["twirled_minus_failure_Markov_chain"]) > ZERO_TOLERANCE:
        raise RuntimeError(f"twirled model minus failure Markov chain: direct {direct}, "
                           f"expanded {fidelity['twirled_minus_failure_Markov_chain']}")
    return fidelity


MODELS = {
    "leaked_level": leaked_level_model,
    "syndrome_qubit": syndrome_qubit_model,
    "logical_qubit": logical_qubit_model,
}

# Cases: (model, error, reading) -> the leading-order expressions that are
# checked. An expression is (coefficient c, power k): the value is
# c N (N-1) p^k, or c N p for the overall failure probability;
# coefficient 0 states that the quantity vanishes. Quantities without an entry
# are recorded and not judged.
CASES = {
    ("leaked_level", "coherent", ""): {
        "infidelity_M": (1, 1),
        "deviation": (-1, 1),
        # cos^2(N phi) against (1 - p)^N
        "delta_leak-coh": (-1, 1),
        # the two-step paths leak and seep: -p I on the logical qubit
        "delta_leak": (1 / 2, 2),
        # the logical part cos(phi) I of a step has no logical error
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (0, 0),
        "delta_err-coh": (0, 0),
        "delta_err-traj": (0, 0),
    },
    ("leaked_level", "incoherent", "jumps_in_both_directions"): {
        "infidelity_M": (1, 1),
        "deviation": (1 / 4, 2),
        # a jump creates no error--leakage coherence
        "delta_leak-coh": (0, 0),
        # 2 binom(N, 2) paths of a leakage and a seepage, each p |l><l|
        "delta_leak": (1 / 4, 2),
        # the no-jump operator sqrt(1-p) I has no logical error
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (0, 0),
        "delta_err-coh": (0, 0),
        "delta_err-traj": (0, 0),
    },
    ("leaked_level", "incoherent", "joint_jumps"): {
        "infidelity_M": (1, 1),
        # binom(N, 2) paths of a leakage and a seepage, each p I
        "deviation": (1 / 2, 2),
        "delta_leak-coh": (0, 0),
        "delta_leak": (1 / 2, 2),
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (0, 0),
    },
    ("leaked_level", "incoherent", "master_equation"): {
        "infidelity_M": (1, 1),
        "deviation": (1 / 4, 2),
        "delta_leak-coh": (0, 0),
        "delta_leak": (1 / 4, 2),
        "delta_logi-coh": (0, 0),
        # a leakage and a seepage within one step leave a logical Z error
        "delta_logi-err": (1 / 32, 4),
    },
    ("syndrome_qubit", "coherent", ""): {
        "infidelity_M": (1, 1),
        "deviation": (-1, 1),
        "delta_err-coh": (-1, 1),
        "delta_err-traj": (1 / 2, 2),
        "delta_leak-coh": (0, 0),
        "delta_leak": (0, 0),
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (0, 0),
    },
    ("syndrome_qubit", "incoherent", "random_error_operator"): {
        "infidelity_M": (1, 1),
        "deviation": (1 / 2, 2),
        "delta_err-traj": (1 / 2, 2),
        "delta_err-coh": (0, 0),
        "delta_leak-coh": (0, 0),
        "delta_leak": (0, 0),
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (0, 0),
    },
    ("logical_qubit", "coherent", ""): {
        "infidelity_M": (1, 1),
        "deviation": (-1, 1),
        # cos^2(N phi/2) against the twirled [1 + (1 - 2p)^N]/2 and (1 - p)^N
        "delta_logi-coh": (-1, 1),
        "delta_logi-err": (1 / 2, 2),
        "delta_leak-coh": (0, 0),
        "delta_leak": (0, 0),
        "delta_err-coh": (0, 0),
        "delta_err-traj": (0, 0),
    },
    ("logical_qubit", "incoherent", "three_Pauli_operators_equally"): {
        "infidelity_M": (1, 1),
        "deviation": (1 / 6, 2),
        # a Pauli map is its own Pauli twirl
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (1 / 6, 2),
        "delta_leak-coh": (0, 0),
        "delta_leak": (0, 0),
        "delta_err-coh": (0, 0),
        "delta_err-traj": (0, 0),
    },
    ("logical_qubit", "incoherent", "one_Pauli_operator"): {
        "infidelity_M": (1, 1),
        "deviation": (1 / 6, 2),
        "delta_logi-coh": (0, 0),
        "delta_logi-err": (1 / 6, 2),
    },
}


def exact_quantities(model: str, error: str, reading: str) -> dict:
    r"""Exact infidelities and signed contributions of one case on the grid.

    Parameters
    ----------
    model, error, reading : str
        Key of ``CASES``.

    Returns
    -------
    dict
        Quantity name to an array of shape ``(len(STEP_NUMBERS),
        len(ERROR_PROBABILITIES))``: ``infidelity_M`` (:math:`1-F_{\mathbb M}`),
        ``infidelity_actual``, ``deviation`` (:math:`F_{\mathbb T}-F_{\mathbb M}`,
        the infidelity of the failure Markov chain minus that of the actual
        QEC dynamics), one signed contribution for each reduction of
        ``REDUCTIONS`` (process fidelity before the reduction minus after it,
        so that the seven sum to ``deviation``), and ``sum_minus_deviation``
        (the sum of the seven minus ``deviation``, rounding only).
    """
    names = ["infidelity_M", "infidelity_actual", "deviation", *REDUCTIONS,
             "sum_minus_deviation"]
    values = {name: np.zeros((len(STEP_NUMBERS), len(ERROR_PROBABILITIES))) for name in names}
    grid = [(i, j, n_steps, p) for i, n_steps in enumerate(STEP_NUMBERS)
            for j, p in enumerate(ERROR_PROBABILITIES)]
    for i, j, n_steps, p in grid:
        fidelity = chain_of_models(n_steps=n_steps, **MODELS[model](error, reading, p))
        values["infidelity_M"][i, j] = 1 - fidelity["failure_Markov_chain"]
        values["infidelity_actual"][i, j] = 1 - fidelity["actual"]
        values["deviation"][i, j] = fidelity["actual"] - fidelity["failure_Markov_chain"]
        # a reduction that lowers the process fidelity means that the actual
        # infidelity falls short of that of the later models by this amount
        for name, (before, after) in REDUCTIONS.items():
            values[name][i, j] = fidelity[before] - fidelity[after]
        # the last contribution without the cancellation (chain_of_models)
        values["delta_logi-err"][i, j] = fidelity["twirled_minus_failure_Markov_chain"]
        values["sum_minus_deviation"][i, j] = (
            sum(values[name][i, j] for name in REDUCTIONS) - values["deviation"][i, j])
    return values


def judge(values: np.ndarray, coefficient: float, power: int, per_step: bool) -> dict:
    r"""Compare exact values on the grid with one leading-order expression.

    Parameters
    ----------
    values : np.ndarray
        Exact values, rows :math:`N` of ``STEP_NUMBERS`` and columns
        :math:`p` of ``ERROR_PROBABILITIES``.
    coefficient : float
        Coefficient :math:`c` of the expression; 0 states that the quantity
        vanishes.
    power : int
        Power :math:`k` of :math:`p`.
    per_step : bool
        Whether the expression is :math:`c\,N\,p^k` (the infidelity of the
        failure Markov chain) instead of :math:`c\,N(N-1)\,p^k`.

    Returns
    -------
    dict
        For a vanishing quantity: ``largest_absolute_value`` and
        ``confirmed``. Otherwise, for each :math:`N`: ``fitted_power`` (slope
        of :math:`\log|\mathrm{value}|` against :math:`\log p` over the three
        smallest :math:`p`), ``ratio_at_smallest_p`` (exact value over the
        expression), and ``ratio_extrapolated`` (intercept of a quadratic fit
        of the ratio in :math:`p`); and ``confirmed``.
    """
    if coefficient == 0:
        largest = float(np.abs(values).max())
        return dict(largest_absolute_value=largest, confirmed=bool(largest < ZERO_TOLERANCE))
    n = np.array(STEP_NUMBERS, float)[:, None]
    p = np.array(ERROR_PROBABILITIES)
    leading = coefficient * (n if per_step else n * (n - 1)) * p ** power
    ratio = values / leading
    # the power from the smallest probabilities, where the next order is weakest
    fitted_power = [float(np.polyfit(np.log(p[:3]), np.log(np.abs(row[:3])), 1)[0])
                    for row in values]
    # the ratio is 1 + O(p): its limit p -> 0 is the intercept of a polynomial fit
    extrapolated = [float(np.polyfit(p, row, 2)[-1]) for row in ratio]
    confirmed = (np.abs(np.array(fitted_power) - power).max() < POWER_TOLERANCE
                 and np.abs(np.array(extrapolated) - 1).max() < RATIO_TOLERANCE)
    return dict(fitted_power=fitted_power, ratio_at_smallest_p=ratio[:, 0].tolist(),
                ratio_extrapolated=extrapolated, confirmed=bool(confirmed))


def case_record(model: str, error: str, reading: str, expressions: dict) -> dict:
    r"""Exact values and the verdict on every expression of one case.

    Parameters
    ----------
    model, error, reading : str
        Key of ``CASES``.
    expressions : dict
        Value of ``CASES``: quantity name to (coefficient, power).

    Returns
    -------
    dict
        ``model``, ``error``, ``reading``, ``largest_sum_minus_deviation``
        (largest absolute value on the grid of the seven contributions minus
        the deviation), ``exact`` (output of :func:`exact_quantities` as
        lists), and ``expressions``: quantity name to ``expression`` (text)
        and the output of :func:`judge`.
    """
    values = exact_quantities(model, error, reading)
    judged = {}
    for name, (coefficient, power) in expressions.items():
        per_step = name == "infidelity_M"
        text = ("0" if coefficient == 0 else
                f"{coefficient:.6g} * {'N' if per_step else 'N (N - 1)'} * p^{power}")
        judged[name] = dict(expression=text,
                            **judge(values[name], coefficient, power, per_step))
    return dict(model=model, error=error, reading=reading,
                largest_sum_minus_deviation=float(np.abs(values["sum_minus_deviation"]).max()),
                exact={name: array.tolist() for name, array in values.items()},
                expressions=judged)


if __name__ == "__main__":
    cases = []
    for (model, error, reading), expressions in CASES.items():
        case = case_record(model, error, reading, expressions)
        cases.append(case)
        print(f"{model}, {error} {reading}: the seven contributions minus the deviation "
              f"{case['largest_sum_minus_deviation']:.1e} at most")
        for name, result in case["expressions"].items():
            detail = (f"largest value {result['largest_absolute_value']:.1e}"
                      if "largest_absolute_value" in result else
                      f"power {np.round(result['fitted_power'], 4).tolist()}, ratio p->0 "
                      f"{np.round(result['ratio_extrapolated'], 5).tolist()}")
            print(f"  {name:15s} {result['expression']:28s} "
                  f"{'CONFIRMED' if result['confirmed'] else 'NOT CONFIRMED'}: {detail}")
    record = dict(
        description="Exact discrepancies and leading-order power-law comparisons for three minimal N-step models under coherent and incoherent errors of probability p per step.",
        settings=dict(
            step_numbers=STEP_NUMBERS, error_probabilities=ERROR_PROBABILITIES,
            power_tolerance=POWER_TOLERANCE, ratio_tolerance=RATIO_TOLERANCE,
            zero_tolerance=ZERO_TOLERANCE,
            array_axes="rows: step_numbers; columns: error_probabilities",
        ),
        versions=dict(numpy=np.__version__, qutip=qt.__version__),
        cases=cases,
    )
    RESULTS_JSON.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS_JSON}")
