r"""Shared tools of the scripts that compare the analytical formula with the numerical computation.

Provides term-level evaluation of ``qec_aeb.cat_code_budget``, single-step
Lindblad master-equation models, and the ideal states after the
error-accumulation idling. Units: durations in ns, rates in 1/ns, angular
frequencies in rad/ns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

import qec_aeb as bsq
import chencrafts.cqed as cqed
from chencrafts.cqed.qt_helper import orthonormalize, subspace_process_fidelity

# step number of the cycle -> key of the step in the JSON files
STEP_KEYS = {
    1: "idle", 2: "X_plus_half_pi", 3: "parity_mapping_idling", 4: "X_minus_half_pi",
    5: "measurement", 6: "feedback_latency", 7: "conditional_X_pi"}
# prefix of a term name of qec_aeb.cat_code_budget -> step number
BUDGET_PREFIX_STEPS = {
    "idle": 1, "gate1": 2, "pm": 3, "gate2": 4, "meas": 5, "latency": 6, "reset": 7}
# rest of a term name of cat_code_budget -> key of the term in the JSON files
TERM_KEYS = {
    "ancilla_excitation": "ancilla_excitation",
    "storage_dephasing": "resonator_dephasing",
    "photon_gain": "photon_gain",
    "photon_loss": "photon_loss",
    "multiple_photon_loss": "two_or_more_photon_losses",
    "kerr_after_loss": "kerr_term",
    "stark": "stark_shift",
    "dispersive": "dispersive_shift",
    "cross": "stark_shift_dispersive_shift_cross",
    "ancilla_dephasing": "ancilla_dephasing",
    "ancilla_relaxation": "ancilla_relaxation",
    "nonlinearity": "chi_1_prime",
    "assignment": "readout_assignment_error",
}
# mechanism name of cat_code_budget -> name of the mechanism in the manuscript.
# Ancilla relaxation is the downward jump |0><1|; ancilla excitation holds the
# two upward jumps, |1><0| and |2><1|
MECHANISM_LABELS = {
    "ancilla relaxation": "ancilla relaxation",
    "ancilla excitation": "ancilla excitation",
    "storage dephasing": "resonator dephasing",
    "photon gain": "photon gain",
    "photon loss": "photon loss",
    "drive Stark shift": "Stark shift",
    "dispersive shift": "dispersive shift",
    "Stark-shift/dispersive-shift cross": r"Stark shift $\times$ dispersive shift",
    "ancilla dephasing": "ancilla dephasing",
    "qubit-state-dependent non-linearity": r"$\chi_1'$",
    "readout assignment": "readout assignment error",
}
# Plain name of each Lindblad operator of the master equation -> the name
# that chencrafts gives it (FullCatTreeBuilder.collapse_operators):
# ``jump_ij`` is the ancilla transition |i><j| and ``jump_a_ij`` is a|i><j|
JUMP_NAMES = {
    "photon_loss": "jump_a",
    "photon_gain": "jump_adag",
    "resonator_dephasing": "jump_adag_a",
    "ancilla_relaxation": "jump_01",
    "ancilla_excitation": "jump_10",
    "ancilla_dephasing": "jump_11",
    "ancilla_jump_to_second_excited_level": "jump_21",
    "photon_loss_with_excited_ancilla": "jump_a_11",
}


# ------------------------------------------------------------------ names ---
def formula_terms(ingredients: dict) -> list[dict]:
    r"""Every term of the analytical formula, with the names of the manuscript.

    Parameters
    ----------
    ingredients : dict
        Every key of the dressed sweep of a parameter set,
        ``sweep.full_dict()``.

    Returns
    -------
    list of dict
        One entry per term of ``qec_aeb.cat_code_budget``, in its
        order: ``step`` (1 to 7), ``step_key`` (:data:`STEP_KEYS`), ``term``
        (:data:`TERM_KEYS`), ``mechanism`` (:data:`MECHANISM_LABELS`),
        ``chencrafts_name``, and ``value``, the contribution of the term to
        the infidelity of one QEC cycle (for steps 2 to 7, summed over the
        two parity error spaces with their probabilities).
    """
    budget = bsq.cat_code_budget
    terms = []
    for name, value in budget("name", **ingredients).items():
        # a term is named <step prefix>_<term>
        prefix, rest = name.split("_", 1)
        step = BUDGET_PREFIX_STEPS[prefix]
        # the name and the step of the term must agree with chencrafts
        if budget[name].step != step:
            raise ValueError(f"term {name!r} belongs to step {budget[name].step}, not {step}")
        terms.append(dict(
            step=step, step_key=STEP_KEYS[step], term=TERM_KEYS[rest],
            mechanism=MECHANISM_LABELS[budget[name].mechanism],
            chencrafts_name=name, value=float(value)))
    return terms


def relative_difference(formula: float, numerical: float) -> float | None:
    r"""Relative difference ``formula / numerical - 1``, or None when the reference vanishes.

    A mechanism whose rate is zero at the parameters has a zero contribution
    and a zero term; their ratio is undefined.

    Parameters
    ----------
    formula, numerical : float
        Term of the analytical formula and numerically computed contribution.

    Returns
    -------
    float or None
    """
    if numerical == 0.0:
        return None
    return float(formula / numerical - 1)


# ------------------------------------------------------- model of a step ----
@dataclass
class IdleModel:
    r"""Master equation of the idling of the resonator-ancilla system.

    Attributes
    ----------
    resonator_dimension : int
        Number of resonator Fock states of the dynamics.
    ancilla_dimension : int
        Number of ancilla levels of the dynamics.
    t_idle : float
        Idle duration (ns).
    displacement : float
        Cat state displacement :math:`\beta`.
    full_hamiltonian : np.ndarray
        Dressed Hamiltonian (rad/ns) in the current frame of the builder.
    full_collapse_operators : list of np.ndarray
        :math:`\sqrt{\gamma_{\hat L}}\hat L` in the same basis.
    collapse_operator_names : list of str
        Name that ``chencrafts`` gives each entry of
        ``full_collapse_operators``, in the same order.
    resonator_hamiltonian : np.ndarray
        Block of ``full_hamiltonian`` with the ancilla in its ground state.
    logical_basis : np.ndarray
        The two logical states of the resonator, symmetrically
        orthonormalized, shape ``(resonator_dimension, 2)``.
    loss_rate, gain_rate, resonator_dephasing_rate : float
        :math:`\gamma_{\hat a}`, :math:`\gamma_{\hat a^\dagger}`,
        :math:`\gamma_{\hat a^\dagger\hat a}` (1/ns).
    jump_ij : np.ndarray
        Rates of the ancilla transitions (1/ns): entry ``[i, j]`` is
        :math:`\gamma_{|i\rangle\langle j|}`.
    jump_a_ij : np.ndarray
        Rates of the photon loss that depends on the ancilla level (1/ns):
        entry ``[i, j]`` is :math:`\gamma_{\hat a|i\rangle\langle j|}`.
    kerr : float
        Resonator self-Kerr coefficient :math:`K` (rad/ns).
    """

    resonator_dimension: int
    ancilla_dimension: int
    t_idle: float
    displacement: float
    full_hamiltonian: np.ndarray
    full_collapse_operators: list
    collapse_operator_names: list
    resonator_hamiltonian: np.ndarray
    logical_basis: np.ndarray
    loss_rate: float
    gain_rate: float
    resonator_dephasing_rate: float
    jump_ij: np.ndarray
    jump_a_ij: np.ndarray
    kerr: float

    def rates(self) -> dict:
        r"""Rate of each Lindblad operator of :data:`JUMP_NAMES`.

        Returns
        -------
        dict
            Plain name of the operator to its rate (1/ns).
        """
        return dict(
            photon_loss=self.loss_rate, photon_gain=self.gain_rate,
            resonator_dephasing=self.resonator_dephasing_rate,
            ancilla_relaxation=float(self.jump_ij[0, 1]),
            ancilla_excitation=float(self.jump_ij[1, 0]),
            ancilla_dephasing=float(self.jump_ij[1, 1]),
            ancilla_jump_to_second_excited_level=float(self.jump_ij[2, 1]),
            photon_loss_with_excited_ancilla=float(self.jump_a_ij[1, 1]))


def idle_model_from_builder(sweep: cqed.FlexibleSweep, builder) -> IdleModel:
    r"""Master equation of the idling, from a dressed sweep and its tree builder.

    The model reads from the builder only what the builder derives from the
    sweep when it is created: the static Hamiltonian and the Lindblad
    operators. Neither depends on the switches of the builder or on which
    dynamical maps have been built.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Output of ``device_model.build_sweep``.
    builder : bsq.QEC_graph.FullCatTreeBuilder
        Output of ``device_model.configure_tree_builder`` for that sweep.

    Returns
    -------
    IdleModel
    """
    resonator_dimension = builder.res_dim
    ancilla_dimension = builder.qubit_dim
    full_hamiltonian = builder.static_hamiltonian.full()
    named_collapse_operators = builder.collapse_operators

    # logical states of the resonator alone (a one-level ancilla),
    # orthonormalized symmetrically
    logical_basis = orthonormalize([
        codeword.full().ravel() for codeword in bsq.cat_ideal.cat_codewords(
            resonator_dimension, 1, float(sweep["disp"]))])

    # every ancilla_dimension-th row has the ancilla in its ground state
    ground = np.arange(0, ancilla_dimension * resonator_dimension, ancilla_dimension)
    return IdleModel(
        resonator_dimension=resonator_dimension,
        ancilla_dimension=ancilla_dimension,
        t_idle=float(sweep["T_W"]),
        displacement=float(sweep["disp"]),
        full_hamiltonian=full_hamiltonian,
        full_collapse_operators=[c.full() for c in named_collapse_operators.values()],
        collapse_operator_names=list(named_collapse_operators),
        resonator_hamiltonian=full_hamiltonian[np.ix_(ground, ground)],
        logical_basis=logical_basis,
        loss_rate=float(sweep["jump_a"]),
        gain_rate=float(sweep["jump_adag"]),
        resonator_dephasing_rate=float(sweep["jump_adag_a"]),
        jump_ij=np.asarray(sweep["jump_ij"], dtype=float),
        jump_a_ij=np.asarray(sweep["jump_a_ij"], dtype=float),
        kerr=float(sweep["K_s"]),
    )


def collapse_operator_indices(model: IdleModel) -> dict:
    r"""Positions of the Lindblad operators of the model, by their plain names.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.

    Returns
    -------
    dict
        Key of :data:`JUMP_NAMES` to the position of the operator in
        ``model.full_collapse_operators``.
    """
    names = model.collapse_operator_names
    # a mechanism whose operator is absent cannot be compared
    missing = sorted(set(JUMP_NAMES.values()) - set(names))
    if missing:
        raise ValueError(f"Lindblad operators not identified: {missing}")
    return {plain: names.index(jump) for plain, jump in JUMP_NAMES.items()}


def idle_propagator(model: IdleModel, collapse_indices: list, duration: float) -> np.ndarray:
    r"""Dynamical map of an idling with a chosen subset of the Lindblad operators.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.
    collapse_indices : list of int
        Positions in ``model.full_collapse_operators`` of the operators that
        act; the Hamiltonian always acts.
    duration : float
        Duration of the idling (ns).

    Returns
    -------
    np.ndarray
        Column-stacked superoperator of the resonator-ancilla space.
    """
    return bsq.cat_real.idling_propagator(
        qt.Qobj(model.full_hamiltonian),
        [qt.Qobj(model.full_collapse_operators[index]) for index in collapse_indices],
        float(duration),
    ).full()


def infidelity(u: np.ndarray, ideal: np.ndarray, states: np.ndarray) -> float:
    r"""Process infidelity of an operation on the error space spanned by ``states``.

    Parameters
    ----------
    u : np.ndarray
        The operation on the resonator-ancilla space: an operator, which need
        not be unitary, or a dynamical map (column-stacked superoperator).
    ideal : np.ndarray
        Ideal operator of the step.
    states : np.ndarray
        The two ideal states that reach the step (columns).

    Returns
    -------
    float
        One minus the process fidelity from ``states`` to the orthonormalized
        ``ideal @ states``.
    """
    targets = orthonormalize(ideal @ states)
    superop = u if u.shape[0] == ideal.shape[0] ** 2 else np.kron(u.conj(), u)
    return float(1 - subspace_process_fidelity(superop, states, targets))


# ------------------------------------------------------------------ idle ----
def ideal_idle_isometries(model: IdleModel, duration: float) -> tuple:
    r"""Ideal states of the two error spaces after an idle, on the resonator alone.

    The even-parity error space propagates the logical states with the
    no-jump factor :math:`e^{-\gamma_{\hat a}t\hat N/2}` and the Hamiltonian
    phase. The odd-parity error space applies :math:`\hat a` and the mean
    self-Kerr rotation :math:`e^{-iKt\hat N}` of a loss at an unknown time.
    Both are the ideal operators ``cat_ideal.idling_maps`` of the resonator
    alone.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.
    duration : float
        Idle duration (ns).

    Returns
    -------
    no_loss, one_loss : np.ndarray
        Isometries of shape ``(resonator_dimension, 2)``.
    """
    dimension = model.resonator_dimension
    # the resonator alone is the resonator with a one-level ancilla; its
    # dressed Hamiltonian is diagonal in the Fock basis
    resonator_hamiltonian = qt.Qobj(
        np.diag(np.diag(model.resonator_hamiltonian)), dims=[[dimension, 1], [dimension, 1]])
    ideal_operators = bsq.cat_ideal.idling_maps(
        dimension, 1, 0, resonator_hamiltonian, duration,
        decay_rate=model.loss_rate, self_Kerr=model.kerr)
    return tuple(
        orthonormalize([operator.full() @ model.logical_basis[:, k] for k in range(2)])
        for operator in ideal_operators)


def embed_ground_isometry(resonator_isometry: np.ndarray, ancilla_dimension: int) -> np.ndarray:
    r"""Embed states of the resonator into the full space, with the ancilla in its ground state.

    Parameters
    ----------
    resonator_isometry : np.ndarray
        Shape ``(resonator_dimension, 2)``.
    ancilla_dimension : int
        Ancilla levels of the dynamics.

    Returns
    -------
    np.ndarray
        Shape ``(ancilla_dimension * resonator_dimension, 2)``.
    """
    dimension = resonator_isometry.shape[0]
    embedded = np.zeros((ancilla_dimension * dimension, 2), dtype=complex)
    # every ancilla_dimension-th row has the ancilla in its ground state
    embedded[0::ancilla_dimension, :] = resonator_isometry
    return embedded


def ideal_idle_states(model: IdleModel) -> list:
    r"""Ideal states of the even-parity and odd-parity error spaces after the idle.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.

    Returns
    -------
    list of np.ndarray
        Two matrices of shape ``(dimension, 2)`` on the resonator-ancilla
        space, the ancilla in its ground state: even parity (no photon
        loss), odd parity (one photon loss).
    """
    return [embed_ground_isometry(isometry, model.ancilla_dimension)
            for isometry in ideal_idle_isometries(model, model.t_idle)]


def idle_infidelity(model: IdleModel, collapse_indices: list, duration: float) -> float:
    r"""Process infidelity of the idle with a chosen subset of the Lindblad operators.

    The ideal error spaces always use the actual photon-loss rate and
    Hamiltonian, so switching a mechanism off changes only the dynamics.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.
    collapse_indices : list of int
        Positions of the Lindblad operators that act.
    duration : float
        Idle duration (ns).

    Returns
    -------
    float
        One minus the process fidelity retained in the even-parity and the
        odd-parity error space.
    """
    logical = embed_ground_isometry(model.logical_basis, model.ancilla_dimension)
    propagator = idle_propagator(model, collapse_indices, duration)
    # the logical state ends in the even-parity or the odd-parity error space
    retained = sum(
        subspace_process_fidelity(
            propagator, logical, embed_ground_isometry(isometry, model.ancilla_dimension))
        for isometry in ideal_idle_isometries(model, duration))
    return float(1.0 - retained)


def idle_error_space_probabilities(model: IdleModel) -> list:
    r"""Probability that the idle ends in each ideal error space, numerically.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.

    Returns
    -------
    list of float
        Even-parity and odd-parity error space: the process fidelity of the
        idle with every Lindblad operator to the ideal states of that space.
    """
    propagator = idle_propagator(
        model, list(range(len(model.full_collapse_operators))), model.t_idle)
    logical = embed_ground_isometry(model.logical_basis, model.ancilla_dimension)
    return [float(subspace_process_fidelity(propagator, logical, states))
            for states in ideal_idle_states(model)]


def idle_mechanism_budget(model: IdleModel) -> tuple[float, dict]:
    r"""Process infidelity of the idle and the contribution of each mechanism.

    Parameters
    ----------
    model : IdleModel
        Output of :func:`idle_model_from_builder`.

    Returns
    -------
    full : float
        Idle infidelity with every Lindblad operator.
    contributions : dict
        ``photon_loss``: the infidelity with photon loss alone;
        ``resonator_dephasing``, ``photon_gain``, ``ancilla_excitation``:
        the increase when the resonator dephasing, the photon gain, or every
        Lindblad operator that acts on the ancilla is added to photon loss.
    """
    index = collapse_operator_indices(model)
    everything = list(range(len(model.full_collapse_operators)))
    loss = index["photon_loss"]
    # every operator that is not one of the three resonator operators acts
    # on the ancilla; from the ground state only the excitation acts first
    resonator = {loss, index["photon_gain"], index["resonator_dephasing"]}
    ancilla = [position for position in everything if position not in resonator]
    duration = model.t_idle
    loss_only = idle_infidelity(model, [loss], duration)
    added = {
        "resonator_dephasing": [loss, index["resonator_dephasing"]],
        "photon_gain": [loss, index["photon_gain"]],
        "ancilla_excitation": [loss] + ancilla,
    }
    contributions = {"photon_loss": loss_only}
    for name, indices in added.items():
        # A mechanism with zero rate has zero operators and adds nothing. Its
        # contribution is set to zero instead of computed, because two
        # evaluations of the same infidelity can differ in the last digit
        if not any(np.any(model.full_collapse_operators[i]) for i in indices if i != loss):
            contributions[name] = 0.0
            continue
        contributions[name] = idle_infidelity(model, indices, duration) - loss_only
    return idle_infidelity(model, everything, duration), contributions
