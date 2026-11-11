r"""
Dynamical maps of the real operations of the cat-code QEC cycle on a
resonator coupled to an ancilla: ingredients of the master equation, the
idle, the ancilla pulse, and the ancilla measurement with assignment
errors.

In the identifiers, ``qubit`` is the ancilla, ``res`` is the resonator, and
a propagator of an open system is its dynamical map, a superoperator in
the qutip ``super`` representation (column stacking).
"""

import numpy as np
import qutip as qt
import copy

import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.utils.cpu_switch import get_map_method
try:
    from qutip.solver.integrator.integrator import IntegratorException
except ImportError:
    # it seems that the IntegratorException is not available in qutip<5
    IntegratorException = Exception
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.cqed.qt_helper import (
    oprt_in_basis, direct_sum, component_liouvillian, superop_from_component_maps,
)
from chencrafts.cqed.pulses import (
    GeneralPulse,
    DRAGGaussian, Gaussian
)
from chencrafts.settings import QUTIP_VERSION

from typing import Dict, List, Tuple, Callable, Any, Literal
import warnings

Esys = Tuple[np.ndarray, np.ndarray]

# ##############################################################################
def _collapse_operators_by_rate(
    hilbertspace: HilbertSpace,
    mode_idx: int, 
    collapse_parameters: Dict[str, Any] = {},
    basis: List[qt.Qobj] | np.ndarray | None = None,
) -> List[qt.Qobj]:
    r"""
    Collapse operators :math:`\sqrt{\gamma}\hat L` of one mode of
    ``hilbertspace`` for the rates in ``collapse_parameters`` (keys as in
    ``cavity_ancilla_me_ingredients``), transformed to ``basis`` if one is
    given.
    """

    hilbertspace = copy.deepcopy(hilbertspace)
    mode = hilbertspace.subsystem_list[mode_idx]
    dim = hilbertspace.subsystem_dims[mode_idx]

    a_oprt = hilbertspace.annihilate(mode)
    if basis is not None:
        a_oprt = oprt_in_basis(a_oprt, basis)

    collapse_ops = []
    if "res_decay" in collapse_parameters.keys():
        collapse_ops.append(np.sqrt(collapse_parameters["res_decay"]) * a_oprt)

    if "res_excite" in collapse_parameters.keys():
        collapse_ops.append(np.sqrt(collapse_parameters["res_excite"]) * a_oprt.dag())

    if "res_dephase" in collapse_parameters.keys():
        collapse_ops.append(
            np.sqrt(collapse_parameters["res_dephase"]) * a_oprt.dag() * a_oprt)

    if "qubit_decay" in collapse_parameters.keys():
        rate_arr = np.array(collapse_parameters["qubit_decay"])
        if len(rate_arr.shape) != 2:
            raise ValueError("The qubit decay rate should be a 2D array.")
        for idx, value in np.ndenumerate(rate_arr):
            # no self transition, neglect small rates
            if idx[0] == idx[1] or value < 1e-14:
                continue

            oprt_ij = hilbertspace.hubbard_operator(idx[1], idx[0], mode)
            if basis is not None:
                oprt_ij = oprt_in_basis(oprt_ij, basis)

            collapse_ops.append(np.sqrt(value) * oprt_ij)

    if "qubit_dephase" in collapse_parameters.keys():
        rate_arr = np.array(collapse_parameters["qubit_dephase"])
        if len(rate_arr.shape) != 1:
            raise ValueError("The qubit pure dephasing rate should be a 1D array.")
        
        diag_elem = np.zeros(dim)
        len_rate = len(rate_arr)
        if dim > len_rate:
            diag_elem[:len_rate] = np.sqrt(rate_arr)
        else:
            diag_elem = np.sqrt(rate_arr[:dim])

        oprt_ii = scq.identity_wrap(
            np.diag(diag_elem), mode, hilbertspace.subsystem_list, 
            op_in_eigenbasis=True,
        )
        if basis is not None:
            oprt_ii = oprt_in_basis(oprt_ii, basis)

        collapse_ops.append(oprt_ii)

    return collapse_ops
    
def cavity_ancilla_me_ingredients(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int, 
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2, 
    dressed_indices: np.ndarray | None = None, eigensys = None,
    collapse_parameters: Dict[str, Any] = {},
    res_n_bar: int | None = None,
    in_rot_frame: bool = True,
) -> Tuple[qt.Qobj, List[qt.Qobj], Esys, qt.Qobj]:
    r"""
    Hamiltonian and collapse operators of the master equation of a
    resonator-ancilla system, in the basis of its dressed states truncated to
    the two modes.

    The Hamiltonian is diagonal with the dressed energies, counted from the
    ground state. The collapse operators are the operators of the bare modes
    written in the dressed basis, with the rates of ``collapse_parameters``.

    Parameters
    ----------
    hilbertspace: HilbertSpace
        ``scq.HilbertSpace`` that contains a resonator and an ancilla.
    res_mode_idx, qubit_mode_idx: int
        Subsystem indices of the resonator and of the ancilla.
    res_truncated_dim: int | None
        Number of resonator levels kept. If None, the resonator is not
        truncated unless a nan eigenvalue is found.
    qubit_truncated_dim: int
        Number of ancilla levels kept.
    dressed_indices: np.ndarray | None
        Array that maps the flattened bare state label to the eigenstate
        label, e.g. ``ParameterSweep["dressed_indices"]``.
    eigensys:
        Eigenenergies and eigenstates of the Hamiltonian of ``hilbertspace``,
        e.g. ``ParameterSweep["evals"]`` and ``ParameterSweep["evecs"]``.
    collapse_parameters: Dict[str, Any]
        Rates of the collapse operators; a channel is added if its key
        exists.

        - ``"res_decay"``: rate of the jump :math:`\hat a`.
        - ``"res_excite"``: rate of the jump :math:`\hat a^\dagger`.
        - ``"res_dephase"``: rate of the jump :math:`\hat a^\dagger\hat a`.
        - ``"qubit_decay"``: two-dimensional array whose element ``[i, j]``
          is the rate of the ancilla transition from level i to level j,
          with the jump :math:`|j\rangle\langle i|`. Diagonal elements and
          rates below 1e-14 are skipped.
        - ``"qubit_dephase"``: one-dimensional array of the pure-dephasing
          rates :math:`\gamma_i` of the ancilla levels; they enter through
          the single collapse operator
          :math:`\sum_i\sqrt{\gamma_i}\,|i\rangle\langle i|`, under which the
          coherence between the levels i and j decays at the rate
          :math:`(\sqrt{\gamma_i}-\sqrt{\gamma_j})^2/2`.
    res_n_bar: int | None
        Has no effect; a value other than None emits a ``UserWarning``.
    in_rot_frame: bool
        If True, the Hamiltonian is transformed into the frame that rotates
        at the resonator frequency and at the 0-1 frequency of the ancilla,
        both taken at the ground state of the other mode. The collapse
        operators would pick up only a phase factor, which drops out of the
        master equation, so they are not transformed.

    Returns
    -------
    hamiltonian: qt.Qobj
        Hamiltonian in the truncated dressed basis, in the unit of the
        eigenenergies (GHz for a scqubits ``HilbertSpace``), with
        ``dims=[[res_dim, qubit_dim], [res_dim, qubit_dim]]``: the resonator
        is first in the tensor product, whatever ``res_mode_idx`` is.
    c_ops: List[qt.Qobj]
        Collapse operators :math:`\sqrt{\gamma}\hat L` in the same basis.
    eigensys: Tuple[np.ndarray, np.ndarray]
        Flattened eigenenergies and eigenstates of the truncated basis.
    frame_hamiltonian: qt.Qobj
        Hamiltonian of the frame transformation:
        :math:`\hat H_{\mathrm{new}}=\hat H_{\mathrm{old}}-\hat H_{\mathrm{frame}}`.
        Zero if ``in_rot_frame`` is False.
    """
    if res_n_bar is not None:
        warnings.warn(
            "The argument res_n_bar has no effect: the rotating frame is "
            "taken at the resonator frequency and the ancilla 0-1 frequency "
            "of the ground state of the other mode.",
            UserWarning, stacklevel=2,
        )

    # prepare
    hilbertspace = copy.deepcopy(hilbertspace)
    dims = hilbertspace.subsystem_dims
    if len(dims) > 2:
        warnings.warn("More than 2 subsystems detected. The 'smart truncation' is not "
                      "smart for more than 2 subsystems. It can't determine when to "
                      "truncate for other subsystems and keep the ground state for the mode "
                      "only. It's also not tested."
                      "Please specify the truncation when initialize the HilbertSpace obj.\n")

    # truncate the basis
    # 1. for qubit mode, keep up to the next excited state of the qubit initial state
    # 2. for res mode, keep all levels unless the bare label are not found (eval=nan)
    # 3. for other modes, keep only ground states
    truncated_evals, truncated_evecs = two_mode_dressed_esys(
        hilbertspace, res_mode_idx, qubit_mode_idx,
        state_label=list(np.zeros_like(dims).astype(int)),
        res_truncated_dim=res_truncated_dim, qubit_truncated_dim=qubit_truncated_dim,
        dressed_indices=dressed_indices, eigensys=eigensys,
        adjust_phase=True,
    )
    truncated_dims = list(truncated_evals.shape)

    # hamiltonian in this basis
    flattened_evals = truncated_evals.ravel() - truncated_evals.ravel()[0]
    hamiltonian = qt.Qobj(np.diag(flattened_evals), dims=[truncated_dims, truncated_dims])

    if in_rot_frame:
        if res_n_bar is None:
            res_n_bar = 0

        # in the dispersive regime, the Hamiltonian of the frame transformation is
        # freq_res * a^dag a * identity_qubit + identity_res * freq_qubit * qubit^dag qubit
        res_freq = truncated_evals[1, 0] - truncated_evals[0, 0]
        qubit_freq = truncated_evals[0, 1] - truncated_evals[0, 0]

        rot_hamiltonian = (
            qt.tensor(res_freq * qt.num(truncated_dims[0]), qt.qeye(qubit_truncated_dim))
            + qt.tensor(qt.qeye(truncated_dims[0]), qubit_freq * qt.num(qubit_truncated_dim))
        )
    else:
        rot_hamiltonian = qt.Qobj(np.zeros_like(hamiltonian.data), dims=hamiltonian.dims)

    hamiltonian -= rot_hamiltonian

    # collapse operators in this basis
    res_collapse_parameters = {
        key: collapse_parameters[key] for key in collapse_parameters.keys()
        if key.startswith("res")
    }
    res_collapse_operators = _collapse_operators_by_rate(
        hilbertspace, res_mode_idx, res_collapse_parameters, basis=truncated_evecs.ravel()
    )
    qubit_collapse_parameters = {
        key: collapse_parameters[key] for key in collapse_parameters.keys()
        if key.startswith("qubit")
    }
    qubit_collapse_operators = _collapse_operators_by_rate(
        hilbertspace, qubit_mode_idx, qubit_collapse_parameters, basis=truncated_evecs.ravel()
    )
    c_ops = [
        qt.Qobj(op, dims=[truncated_dims, truncated_dims])
        for op in res_collapse_operators + qubit_collapse_operators
    ]       # give the collapse operators the dims of the truncated basis
    
    return (
        hamiltonian, 
        c_ops, 
        (truncated_evals.ravel(), truncated_evecs.ravel()),
        rot_hamiltonian,
    )

def idling_propagator(
    hamiltonian: qt.Qobj, 
    c_ops: List[qt.Qobj],
    time: float,
) -> qt.Qobj:
    r"""
    Dynamical map :math:`e^{\mathcal L t}` of an idle of duration :math:`t`,
    with the Liouvillian :math:`\mathcal L` of the Hamiltonian and of the
    collapse operators.

    Parameters
    ----------
    hamiltonian: qt.Qobj
        Hamiltonian of the system (rad/ns for a time in ns).
    c_ops: List[qt.Qobj]
        Collapse operators :math:`\sqrt{\gamma}\hat L` of the system.
    time: float
        Idle duration :math:`t`.

    Returns
    -------
    qt.Qobj
        The dynamical map, a superoperator in the qutip ``super``
        representation (column stacking).
    """
    liouv = qt.liouvillian(hamiltonian, c_ops)

    return (liouv * time).expm()

def assignment_confusion_matrix(
    dim: int,
    M_ge: float,
    M_eg: float,
) -> np.ndarray:
    r"""
    Confusion matrix of a measurement with assignment errors between the
    outcomes 0 and 1.

    The element :math:`C_{ij}` is the probability of reporting the outcome j when
    the measured state is i. The state 0 is reported as 1 with the probability
    ``M_ge``, and the state 1 is reported as 0 with the probability ``M_eg``;
    every other state is reported correctly.

    Parameters
    ----------
    dim : int
        Number of measured states (ancilla levels, or the two parities), at
        least 2.
    M_ge : float
        Probability of reporting 1 when the state is 0.
    M_eg : float
        Probability of reporting 0 when the state is 1.

    Returns
    -------
    np.ndarray
        The ``dim`` x ``dim`` matrix; each row sums to 1.
    """
    confusion_matrix = np.eye(dim)
    confusion_matrix[0, 0] = 1 - M_ge
    confusion_matrix[0, 1] = M_ge
    confusion_matrix[1, 0] = M_eg
    confusion_matrix[1, 1] = 1 - M_eg

    return confusion_matrix

from qec_aeb.cat_ideal import qubit_projectors as ideal_qubit_projectors
def qubit_projectors(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    confusion_matrix: np.ndarray | None = None,
    ensamble_average: bool = False,
) -> List[qt.Qobj]:
    r"""
    Dynamical maps of an ancilla measurement with assignment errors.

    The ancilla is measured projectively in the basis of its levels, and the
    reported outcome differs from the measured level with some probability.
    The element :math:`C_{ij}` of the confusion matrix is the probability of
    reporting the outcome j when the ancilla is found in level i. The map of
    the reported outcome j is
    :math:`\hat\rho\mapsto\sum_iC_{ij}\hat P_i\hat\rho\hat P_i`, where
    :math:`\hat P_i` projects the ancilla on level i.

    Parameters
    ----------
    res_dim, qubit_dim: int
        Dimensions of the resonator and the ancilla.
    res_mode_idx: Literal[0, 1]
        Position of the resonator in the tensor product.
    confusion_matrix: np.ndarray | None
        The confusion matrix, a square matrix of dimension ``qubit_dim``
        whose rows sum to 1. If None, the measurement is ideal.
    ensamble_average: bool
        If True, return one superoperator per reported outcome j, summed
        over the measured levels i. If False, return the superoperator
        :math:`\hat\rho\mapsto C_{ij}\hat P_i\hat\rho\hat P_i` of every pair
        (i, j), ordered by i first.

    Returns
    -------
    List[qt.Qobj]
        The superoperators, in the qutip ``super`` representation.
    """
    projs = ideal_qubit_projectors(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        superop=False,
    )

    if confusion_matrix is None:
        confusion_matrix = np.eye(qubit_dim)
    else:
        # check the normalization of the confusion matrix
        confusion_matrix = np.array(confusion_matrix)
        if not np.allclose(np.sum(confusion_matrix, axis=1), 1):
            raise ValueError("Each row of the confusion matrix should be "
                             "normalized to 1.")
        
        # check the dimension of the confusion matrix
        if confusion_matrix.shape != (qubit_dim, qubit_dim):
            raise ValueError("The confusion matrix should be a square matrix "
                             "with dimension qubit_dim.")
        
    # construct the measurement operators
    measurement_ops = np.empty_like(confusion_matrix, dtype=object)
    for idx, prob in np.ndenumerate(confusion_matrix):
        measurement_ops[idx] = qt.to_super(projs[idx[0]]) * prob

    if ensamble_average:
        measurement_ops = np.sum(measurement_ops, axis=0)
    else:
        measurement_ops = measurement_ops.ravel()

    return measurement_ops.tolist()
    
def dressed_subspace_operators(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int,
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2,
    dressed_indices: np.ndarray | None = None, eigensys = None,
    drive_operator: str | None = None,
    drive_matrix_elements: bool = True,
) -> Dict[str, Any]:
    r"""
    Static Hamiltonian and drive operator of an ancilla pulse, on the full
    Hilbert space and on the dressed states of each photon-number subspace.

    The static Hamiltonian conserves the dressed photon number, and a drive on
    the ancilla nearly does. A pulse is therefore described subspace by
    subspace: in the subspace of m photons, spanned by the dressed states
    :math:`|\overline{m, q}\rangle` of the ancilla levels q, the static
    Hamiltonian is diagonal with the dressed energies :math:`E_{m,q}`, and the
    drive operator :math:`\hat D` has the matrix elements
    :math:`\langle\overline{m, q}|\hat D|\overline{m, q'}\rangle`.
    :func:`qubit_gate` integrates the pulse with these ingredients.

    With the static Hamiltonian :math:`\hat H` of ``hilbertspace`` in GHz, the
    returned :math:`2\pi\hat H` and the returned energies are in rad/ns. The
    drive operator is multiplied by the same factor :math:`2\pi`, as in
    :func:`qubit_gate`, where the pulse envelope multiplies the drive
    operator.

    Parameters
    ----------
    hilbertspace: HilbertSpace
        scq.HilbertSpace object that contains an ancilla and a resonator.
    res_mode_idx, qubit_mode_idx: int
        The indices of the resonator and of the ancilla in the HilbertSpace.
    res_truncated_dim: int | None
        Number of photon-number subspaces. If None, the resonator is not
        truncated unless a nan eigenvalue is found.
    qubit_truncated_dim: int
        Number of ancilla levels kept in each subspace.
    dressed_indices: np.ndarray | None
        An array mapping the flattened bare state label to the eigenstate
        label. Usually given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the Hamiltonian of
        ``hilbertspace``. Usually given by `ParameterSweep["evals"]` and
        `ParameterSweep["evecs"]`.
    drive_operator: str | None
        Name of the ancilla operator that the drive couples to, e.g.
        "n_operator". If None, use the default of the ancilla circuit: the
        charge operator for a transmon, the phase operator for a fluxonium.
    drive_matrix_elements: bool
        If False, the matrix elements of the drive operator in the subspaces
        are not computed (:func:`qubit_gate` projects the operator itself,
        after fixing its phase).

    Returns
    -------
    Dict[str, Any]
        - ``"static_hamiltonian"``: qt.Qobj, :math:`2\pi\hat H` on the full
          Hilbert space (rad/ns); the ground-state energy is not subtracted.
        - ``"drive_hamiltonian"``: qt.Qobj, :math:`2\pi\hat D` on the full Hilbert
          space.
        - ``"drive_operator"``: str, the name of the ancilla operator.
        - ``"energies"``: np.ndarray of shape (photon numbers, ancilla
          levels), the dressed energies :math:`E_{m,q}` (rad/ns); the
          ground-state energy is not subtracted.
        - ``"dressed_states"``: np.ndarray of kets with the same shape, the
          dressed states :math:`|\overline{m, q}\rangle` on the full space.
        - ``"drive_matrix_elements"``: np.ndarray of shape (photon numbers, ancilla
          levels, ancilla levels), the matrix elements of :math:`2\pi\hat D` in
          each subspace. Absent if ``drive_matrix_elements`` is False.
    """
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    qubit_type = qubit.__class__.__name__

    # static Hamiltonian, from GHz to rad/ns
    H0 = hilbertspace.hamiltonian() * np.pi * 2
    default_drive_operator = {
        "Transmon": "n_operator", "Fluxonium": "phi_operator",
    }
    if qubit_type not in default_drive_operator:
        raise ValueError(f"Unknown qubit type {qubit_type}")
    if drive_operator is None:
        drive_operator = default_drive_operator[qubit_type]
    H1 = scq.identity_wrap(
        getattr(qubit, drive_operator)(), 
        qubit, 
        hilbertspace.subsystem_list
    ) * np.pi * 2

    # dressed states of every photon-number subspace
    evals, evec_arr = two_mode_dressed_esys(
        hilbertspace,
        res_mode_idx, qubit_mode_idx,
        state_label=(0, 0),
        res_truncated_dim=res_truncated_dim,
        qubit_truncated_dim=qubit_truncated_dim,
        dressed_indices=dressed_indices, 
        eigensys=eigensys,
    )
    evals = evals * np.pi * 2

    operators = {
        "static_hamiltonian": H0,
        "drive_hamiltonian": H1,
        "drive_operator": drive_operator,
        "energies": evals,
        "dressed_states": evec_arr,
    }
    if drive_matrix_elements:
        # projection of the drive operator on the dressed states of each subspace
        operators["drive_matrix_elements"] = np.array([
            qt.Qobj(oprt_in_basis(H1, basis)).full() for basis in evec_arr
        ])

    return operators

def qubit_gate(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int,
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2,
    dressed_indices: np.ndarray | None = None, eigensys = None,
    rotation_angle: float = np.pi / 2,
    gate_params: Dict[str, Any] = {},
    num_cpus: int = 8,
    t_steps: int | None = None,
    apply_direct_sum: bool = True,
    drive_operator: str | None = None,
    simulated_qubit_dim: int | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    collapse_operators: List[qt.Qobj] | None = None,
) -> qt.Qobj | List[qt.Qobj] | np.ndarray:
    r"""
    Unitary or dynamical map of an ancilla pulse in the lab frame.

    The pulse is resonant with the 0-1 transition of the ancilla in the
    subspace of ``round(disp**2)`` photons. It has a Gaussian envelope, with a
    DRAG quadrature at a constant carrier frequency for a transmon and
    without one for a fluxonium. Without collapse operators, the pulse is a
    closed-system unitary: the direct sum of the propagators of the
    photon-number subspaces. With collapse operators, it is a dynamical map.
    The density matrix then splits into components between two photon-number
    subspaces, the master equation of each component is integrated
    separately (``component_liouvillian``), and the maps of the components
    are composed into the map of the whole system
    (``superop_from_component_maps``).

    Parameters
    ----------
    hilbertspace: HilbertSpace
        ``scq.HilbertSpace`` that contains a resonator and an ancilla (a
        ``Transmon`` or a ``Fluxonium``).
    res_mode_idx, qubit_mode_idx: int
        Subsystem indices of the resonator and of the ancilla.
    res_truncated_dim: int | None
        Number of photon-number subspaces. If None, the resonator is not
        truncated unless a nan eigenvalue is found.
    qubit_truncated_dim: int
        Number of ancilla levels of the returned operator, at least 3; the
        default 2 raises ``ValueError``.
    dressed_indices: np.ndarray | None
        Array that maps the flattened bare state label to the eigenstate
        label, e.g. ``ParameterSweep["dressed_indices"]``.
    eigensys:
        Eigenenergies and eigenstates of the Hamiltonian of ``hilbertspace``,
        e.g. ``ParameterSweep["evals"]`` and ``ParameterSweep["evecs"]``.
    rotation_angle: float
        Rotation angle of the pulse (rad), signed.
    gate_params: Dict[str, Any]
        Pulse parameters: ``"disp"``, the cat amplitude; ``"sigma"`` and
        ``"tau_p_eff"``, the Gaussian width and the duration of a
        :math:`\pi/2` pulse (ns). Width and duration are multiplied by
        ``abs(rotation_angle) / (pi / 2)``.
    num_cpus: int
        Number of processes over which the subspaces, or the components, are
        distributed.
    t_steps: int | None
        Number of times at which the propagator is recorded. If None, only
        the final propagator is recorded. For testing only.
    apply_direct_sum: bool
        If True, return the direct sum of the subspace propagators as one
        operator. If False, return the array of subspace propagators. For
        testing only.
    drive_operator: str | None
        Name of the ancilla operator that the drive couples to, e.g.
        ``"n_operator"``. If None, the charge operator for a transmon and the
        phase operator for a fluxonium.
    simulated_qubit_dim: int | None
        Number of ancilla levels kept while the pulse is integrated in each
        photon-number subspace. The returned operator keeps only the first
        ``qubit_truncated_dim`` of them, so population driven into the extra
        levels appears as a loss of norm (leakage). If None, equal to
        ``qubit_truncated_dim``.
    rtol, atol: float
        Relative and absolute tolerances of the ODE integrator. The lab-frame
        integration spans many carrier periods; qutip's default relative
        tolerance (1e-6) loses about 1e-4 of norm per pulse; 1e-12 is
        converged (1e-13 changes the pulse infidelity by less than 1e-4 of
        itself).
    collapse_operators: List[qt.Qobj] | None
        Collapse operators :math:`\sqrt{\gamma}\hat L` (ns^-1/2) on the full
        Hilbert space of ``hilbertspace``. None of them may change the
        photon number of the resonator; an example is the ``jump_ij`` family
        of ``cat_recipe.dressed_jump_ops`` written on the full space with
        ``trans_by_kets``. Like the Hamiltonian and the drive operator, each
        is projected on the dressed subspace of every photon number. If None
        or empty, the pulse is the closed-system unitary. Collapse operators
        need ``apply_direct_sum=True``, ``t_steps=None``, and qutip 5 or
        later.

    Returns
    -------
    qt.Qobj
        The lab-frame unitary (closed system) or dynamical map (with collapse
        operators, qutip ``super`` representation) on the
        ``res_truncated_dim * qubit_truncated_dim`` space, with the resonator
        first in the tensor product. Only the first ``qubit_truncated_dim`` of
        the simulated ancilla levels are kept, so the dynamical map is
        trace-decreasing if population reaches the other levels. With
        ``t_steps``, a list of such operators; with
        ``apply_direct_sum=False``, the array of subspace propagators.
    """

    # the pulse is built from the 0-2 matrix element and the 1-2 detuning of
    # the ancilla, which two levels do not have
    if qubit_truncated_dim < 3:
        raise ValueError(
            "qubit_truncated_dim must be at least 3, got "
            f"{qubit_truncated_dim}."
        )

    # for testing, the propagator can be recorded at several times
    if t_steps is not None:
        warnings.warn(
            "Recording the propagator at multiple time steps is only supported "
            "for testing purposes."
        )
    if not apply_direct_sum:
        warnings.warn(
            "Not applying direct sum to the propagator is only supported "
            "for testing purposes."
        )
    
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    qubit_type = qubit.__class__.__name__

    # gate params
    if simulated_qubit_dim is None:
        simulated_qubit_dim = qubit_truncated_dim
    if simulated_qubit_dim < qubit_truncated_dim:
        raise ValueError(
            "simulated_qubit_dim must not be smaller than qubit_truncated_dim."
        )
    res_n_bar = np.round(gate_params["disp"]**2).astype(int)

    # static Hamiltonian, drive operator, and the dressed states of every
    # photon-number subspace with the simulated ancilla levels. The drive
    # operator is projected on the subspaces below, after its phase is fixed.
    subspace_ingredients = dressed_subspace_operators(
        hilbertspace,
        res_mode_idx, qubit_mode_idx,
        res_truncated_dim=res_truncated_dim,
        qubit_truncated_dim=simulated_qubit_dim,
        dressed_indices=dressed_indices, 
        eigensys=eigensys,
        drive_operator=drive_operator,
        drive_matrix_elements=False,
    )
    H0 = subspace_ingredients["static_hamiltonian"]
    H1 = subspace_ingredients["drive_hamiltonian"]
    evals = subspace_ingredients["energies"]
    evec_arr = subspace_ingredients["dressed_states"]

    bare_angular_freq = (evals[res_n_bar, 1] - evals[res_n_bar, 0])
    non_lin = (evals[res_n_bar, 2] - 2*evals[res_n_bar, 1] + evals[res_n_bar, 0])

    gate_in_subspace = oprt_in_basis(
        H1, evec_arr[res_n_bar, 0:qubit_truncated_dim]
    )
    tgt_mat_elem = gate_in_subspace[0, 1]
    leaking_mat_elem = gate_in_subspace[0, 2]
    
    # shift the energies so that the ground state has zero energy, then fix the
    # phase of the drive matrix element (its sign for a real element)
    H0 = H0 - evals[0, 0]
    tgt_phase = tgt_mat_elem / np.abs(tgt_mat_elem)
    if np.abs(tgt_phase.imag) < 1e-10:
        # real element: remove its sign from the operator
        H1 = H1 / tgt_phase
        carrier_phase = 0.0
    elif np.abs(tgt_phase.real) < 1e-10:
        # purely imaginary element, e.g. the charge operator of a fluxonium in
        # a real eigenbasis: a quarter-period shift of the carrier makes the
        # rotation axis x (dividing a Hermitian operator by i would break its
        # hermiticity)
        carrier_phase = -np.angle(tgt_phase)
    else:
        raise ValueError(
            "Target matrix element is neither real nor purely imaginary. "
            "Check the drive operator."
        )

    # scale the pulse duration for different rotation angle
    sigma = gate_params["sigma"] * np.abs(rotation_angle) / (np.pi / 2)
    duration = gate_params["tau_p_eff"] * np.abs(rotation_angle) / (np.pi / 2)

    if qubit_type == "Transmon":
        pulse = DRAGGaussian(
            base_angular_freq = bare_angular_freq,
            duration = duration,
            sigma = sigma, 
            non_lin = non_lin,
            order = 2, 
            rotation_angle = rotation_angle, 
            tgt_mat_elem = tgt_mat_elem,
            leaking_mat_elem = leaking_mat_elem,
            init_phase = carrier_phase,
            # constant carrier frequency: the DRAG quadrature acts without a
            # frequency chirp
            dynamic_drive_freq = False,
        )
        
        # # square pulse
        # pulse = GeneralPulse(
        #     base_angular_freq = bare_angular_freq,
        #     duration = duration,
        #     rotation_angle = rotation_angle, 
        #     tgt_mat_elem = tgt_mat_elem,
        #     with_freq_shift = False,
        # )
    elif qubit_type == "Fluxonium":
        pulse = Gaussian(
            base_angular_freq = bare_angular_freq,
            duration = duration,
            sigma = sigma, 
            rotation_angle = rotation_angle, 
            tgt_mat_elem = tgt_mat_elem,
            init_phase = carrier_phase,
        )
    
    # for testing, the propagator can be recorded at several times
    sim_time = pulse.duration if t_steps is None else np.linspace(0, pulse.duration, t_steps)

    def shifted_subspace_hamiltonian(basis):
        r"""
        Static Hamiltonian projected on the dressed states ``basis`` of one
        subspace, shifted by a multiple of 2 pi / duration to keep the integrated
        phase small; the shift multiplies the final propagator by a phase of 2 pi
        times an integer, which does not change it.
        """
        H0_ss = oprt_in_basis(H0, basis)
        phase_quantum = np.pi * 2 / pulse.duration
        cycle_num = np.round(H0_ss[1, 1] / phase_quantum)
        return H0_ss - phase_quantum * cycle_num

    # closed system: integrate subspace by subspace
    def calculate_prop(basis):
        H0_ss = shifted_subspace_hamiltonian(basis)
        H1_ss = oprt_in_basis(H1, basis)

        try:
            pulse.reset()
        except AttributeError:
            pass

        options = dict(
            nsteps=10000000,
            atol=atol,
            rtol=rtol,
        )
        if QUTIP_VERSION[0] < 5:
            options = qt.Options(**options)

        try:
            prop = qt.propagator(
                H = lambda t, args: H0_ss + H1_ss * pulse(t),
                t = sim_time,
                options = options,
            )
        except IntegratorException as e:
            raise IntegratorException(
                f"The qutip solver fails: {e}. "
                f"Current gate time is {pulse.duration}. "
                f"Try to increase the number of time steps."
            )

        # keep the retained qubit levels; population in the extra simulated
        # levels is lost from the returned propagator
        def truncate(p):
            return qt.Qobj(p.full()[:qubit_truncated_dim, :qubit_truncated_dim])
        if isinstance(prop, list):
            return [truncate(p) for p in prop]
        return truncate(prop)

    if collapse_operators:
        # open system: the Hamiltonian and the collapse operators conserve the
        # photon number, so the density matrix splits into components between
        # two photon-number subspaces, and each evolves under its own master
        # equation
        if t_steps is not None or not apply_direct_sum:
            raise ValueError(
                "Collapse operators need apply_direct_sum=True and t_steps=None."
            )
        if QUTIP_VERSION[0] < 5:
            raise ValueError("The open-system gate needs qutip >= 5.")
        num_photon_numbers = len(evec_arr)

        # Hamiltonian, drive operator, and collapse operators of each subspace:
        # projections of the operators of the full system on its dressed states
        subspace_operators = []
        for basis in evec_arr:
            subspace_operators.append((
                shifted_subspace_hamiltonian(basis),
                oprt_in_basis(H1, basis),
                [oprt_in_basis(op, basis) for op in collapse_operators],
            ))

        component_options = dict(nsteps=10000000, atol=atol, rtol=rtol)

        def calculate_component_map(subspaces):
            r"""
            Dynamical map of the component of the density matrix between the
            photon-number subspaces ``subspaces`` = (m, m'), on the retained ancilla
            levels: a matrix of dimension ``qubit_truncated_dim**2`` in the
            column-stacking convention.
            """
            m, m_prime = subspaces
            static_part, drive_part = component_liouvillian(
                *subspace_operators[m], *subspace_operators[m_prime],
            )
            try:
                pulse.reset()
            except AttributeError:
                pass
            prop = qt.propagator(
                H = [static_part, [drive_part, lambda t, args: pulse(t)]],
                t = sim_time,
                options = component_options,
            )
            # keep the retained qubit levels of the initial and the final matrix;
            # population in the extra simulated levels is lost from the trace
            # of the map
            full_map = prop.full().reshape((simulated_qubit_dim,) * 4)
            kept = full_map[(slice(0, qubit_truncated_dim),) * 4]
            return kept.reshape(qubit_truncated_dim ** 2, qubit_truncated_dim ** 2)

        # the component (m', m) is the Hermitian conjugate of (m, m'), so only m <= m'
        # is integrated
        components = [(m, m_prime) for m in range(num_photon_numbers) for m_prime in range(m, num_photon_numbers)]
        if num_cpus > 1:
            map_method = get_map_method(num_cpus, cpu_per_node=1)
            component_maps = list(map_method(calculate_component_map, components))
        else:
            component_maps = list(map(calculate_component_map, components))

        return superop_from_component_maps(dict(zip(components, component_maps)))

    if num_cpus > 1:
        # try:
        #     from multiprocess import Pool
        # except ImportError:
        #     raise ImportError(
        #         "multiprocess is a optional dependency for qec_aeb."
        #         "Please install it via 'pip install multiprocess' or 'conda install multiprocess'."
        #     )
        # with Pool(num_cpus) as pool:
        #     prop_list = list(pool.map(
        #         calculate_prop, evec_arr
        #     ))
        
        # use the scqubits cpu_switch
        map_method = get_map_method(num_cpus, cpu_per_node=1)
        prop_list = list(map_method(calculate_prop, evec_arr))
        
    else:
        prop_list = list(map(
            calculate_prop, evec_arr
        ))
        
    prop_list = np.array(prop_list, dtype=object)
    def apply_ds_and_set_dims(prop_list_1d):
        prop_ds = direct_sum(*prop_list_1d)
        # these dims may be wrong if two_mode_dressed_esys returns a partial
        # basis
        prop_ds.dims = [[res_truncated_dim, qubit_truncated_dim], [res_truncated_dim, qubit_truncated_dim]]
        return prop_ds
    
    if apply_direct_sum:
        if t_steps is None:
            return apply_ds_and_set_dims(prop_list)
        else:
            return [apply_ds_and_set_dims(prop) for prop in prop_list.T]
    else:
        return prop_list
