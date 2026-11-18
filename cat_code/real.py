r"""Dressed jump operators and dynamical maps of the real operations of the cat-code QEC cycle.

Combines the dressed-jump master-equation recipe (``get_jump_ops``,
``dressed_jump_ops``, ``cavity_ancilla_me_ingredients``) and the dynamical maps
of the idle, the ancilla pulses, and the ancilla measurement with assignment
errors (``idling_propagator``, ``assignment_confusion_matrix``,
``qubit_projectors``, ``dressed_subspace_operators``, ``qubit_gate``).

In the identifiers, ``qubit`` is the ancilla, ``res`` is the resonator, and
a propagator of an open system is its dynamical map, a superoperator in
the qutip ``super`` representation (column stacking).
"""

__all__ = [
    "get_jump_ops",
    "dressed_jump_ops",
    "cavity_ancilla_me_ingredients",
    "idling_propagator",
    "assignment_confusion_matrix",
    "qubit_projectors",
    "dressed_subspace_operators",
    "qubit_gate",
]

import copy
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import qutip as qt
import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.param_sweep import ParameterSweep
from scqubits.utils.cpu_switch import get_map_method

try:
    from qutip.solver.integrator.integrator import IntegratorException
except ImportError:
    IntegratorException = Exception

from chencrafts.cqed.flexible_sweep import FlexibleSweep
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.cqed.pulses import DRAGGaussian, Gaussian, GeneralPulse
from chencrafts.cqed.qt_helper import (
    component_liouvillian,
    direct_sum,
    oprt_in_basis,
    superop_from_component_maps,
)
from chencrafts.settings import QUTIP_VERSION

from . import ideal as cat_ideal

Esys = Tuple[np.ndarray, np.ndarray]


def keep_second_order_jump(
    op_family: str,
    i: int | None = None,
    j: int | None = None,
) -> bool:
    r"""
    Whether a dressed jump operator is kept at second order in the
    coupling-to-detuning ratio (``higher_order_decay = False``).

    At second order, the kept operators are:

    - ``jump_a``, ``jump_adag``, ``jump_adag_a``: the bare resonator jumps;
    - ``jump_ij`` with ``(i, j)`` in ``{(0, 1), (1, 0), (1, 1), (2, 1)}``:
      relaxation, excitation, dephasing in the 0-1 subspace, and leakage from
      level 1 to level 2;
    - ``jump_a_ij`` with ``(i, j) = (1, 1)``: photon loss conditioned on the
      ancilla being in level 1.
    """
    if op_family in ("jump_a", "jump_adag", "jump_adag_a"):
        return True
    if op_family == "jump_ij":
        return (i, j) in ((0, 1), (1, 0), (1, 1), (2, 1))
    if op_family == "jump_a_ij":
        return (i, j) == (1, 1)
    return False


def get_jump_ops(
    res_mode_idx: int = 0, 
    res_trunc_dim: int = 5,
    qubit_trunc_dim: int = 2,
) -> Dict[str, qt.Qobj | np.ndarray]:
    r"""
    Jump operators of the resonator-ancilla system, before dressing and
    without rates.

    Parameters
    ----------
    res_mode_idx : int
        Index of the resonator in the tensor product.
    res_trunc_dim, qubit_trunc_dim : int
        Dimensions of the resonator and the ancilla.

    Returns
    -------
    Dict[str, qt.Qobj | np.ndarray]
        ``jump_a``, ``jump_adag``, and ``jump_adag_a`` are the operators
        :math:`a`, :math:`a^\dagger`, and :math:`a^\dagger a`. ``jump_ij``,
        ``jump_a_ij``, and ``jump_adag_ij`` are arrays whose element ``[i, j]``
        is :math:`|i\rangle\langle j|`, :math:`a|i\rangle\langle j|`, and
        :math:`a^\dagger|i\rangle\langle j|` of the ancilla. Each operator
        acts as the identity on the other mode.
    """
    a_op = cat_ideal.res_destroy(
        res_trunc_dim, qubit_trunc_dim, res_mode_idx,
    )
    ij_ops = np.ndarray(
        (qubit_trunc_dim, qubit_trunc_dim), 
        dtype=object
    )
    for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
        ij_ops[i, j] = cat_ideal.qubit_proj(
            res_trunc_dim, qubit_trunc_dim, res_mode_idx,
            qubit_state = i, qubit_state_2 = j,
        )
    vectorized_multiply = np.vectorize(lambda x, y: x * y, otypes=[object])
    a_ij_ops = vectorized_multiply(a_op, ij_ops)
    adag_ij_ops = vectorized_multiply(a_op.dag(), ij_ops)
    
    jump_ops = {
        "jump_a": a_op,
        "jump_adag": a_op.dag(),
        "jump_adag_a": a_op.dag() * a_op,
        
        "jump_ij": ij_ops,
        
        "jump_a_ij": a_ij_ops,
        "jump_adag_ij": adag_ij_ops,
    }
     
    return jump_ops


def dressed_jump_ops(
    fsweep: FlexibleSweep,
    res_mode_idx: int,
    res_dim: int,
    qubit_dim: int,
    qubit_me_dim: int,
    higher_order_decay: bool = True,
    op_families: Optional[List[str]] = None,
) -> Dict[str, qt.Qobj]:
    r"""
    Dressed jump operators :math:`\sqrt{\gamma}\hat L` of the master equation.

    Each operator of :func:`get_jump_ops` is multiplied by the square root of
    its rate ``fsweep["jump_*"]``. If ``higher_order_decay`` is False, the jumps
    of higher than second order are dropped (:func:`keep_second_order_jump`).

    Parameters
    ----------
    fsweep : FlexibleSweep
        Sweep that holds the ``jump_*`` rates.
    res_mode_idx : int
        Subsystem index of the resonator.
    res_dim, qubit_dim : int
        Dimensions of the operators.
    qubit_me_dim : int
        Number of ancilla levels i and j for which the jumps
        :math:`|i\rangle\langle j|` and their products with :math:`a` and
        :math:`a^\dagger` are built.
    higher_order_decay : bool
        If True, keep every computed jump. If False, keep only the second-order
        jumps.
    op_families : list of str, optional
        Keys of :func:`get_jump_ops` to keep. For example, ``["jump_ij"]``
        keeps the jumps that act on the ancilla alone and do not change the
        resonator photon number. Default: every family.

    Returns
    -------
    Dict[str, qt.Qobj]
        The operators, labeled by their family and ancilla levels
        (``jump_01``, ``jump_a_11``, ...), in the order of the families.
    """
    jump_ops = get_jump_ops(res_mode_idx, res_dim, qubit_dim)
    jump_ops_w_rate = {}
    for op_family, op_val in jump_ops.items():
        if op_families is not None and op_family not in op_families:
            continue
        rate = fsweep[op_family]

        if op_family.endswith("ij"):
            for i, j in np.ndindex(qubit_me_dim, qubit_me_dim):
                if (not higher_order_decay) and (
                    not keep_second_order_jump(op_family, i, j)
                ):
                    continue
                labeled_key = op_family[:-2] + f"{i}{j}"
                jump_ops_w_rate[labeled_key] = op_val[i, j] * np.sqrt(rate[i, j])
        else:
            if (not higher_order_decay) and (
                not keep_second_order_jump(op_family)
            ):
                continue
            jump_ops_w_rate[op_family] = op_val * np.sqrt(rate)
    return jump_ops_w_rate


def cavity_ancilla_me_ingredients(
    fsweep: FlexibleSweep,
    res_mode_idx: int, qubit_mode_idx: int, 
    res_dim: int = 5, qubit_dim: int = 2, 
    res_me_dim: int = 5, qubit_me_dim: int = 2, 
    in_rot_frame: bool = True,
    res_n_ref: int = 0,
    higher_order_decay: bool = True,
    return_jump_names: bool = False,
) -> (
    Tuple[qt.Qobj, List[qt.Qobj], Esys, qt.Qobj]
    | Tuple[qt.Qobj, List[qt.Qobj], Esys, qt.Qobj, List[str]]
):
    r"""
    Hamiltonian and collapse operators of the master equation of a
    resonator-ancilla system, in the dressed basis and, if ``in_rot_frame`` is
    True, in the rotating frame.

    Each jump operator acts on the dressed states as an operator of the bare
    modes acts on the bare states with the same labels: :math:`a`,
    :math:`a^\dagger`, :math:`a^\dagger a`, :math:`|i\rangle\langle j|`, and the
    products :math:`a|i\rangle\langle j|` and :math:`a^\dagger|i\rangle\langle j|`.
    Its rate is taken from ``fsweep["jump_*"]``. If ``higher_order_decay`` is
    False, only the second-order subset documented in
    :func:`keep_second_order_jump` is kept.

    Parameters
    ----------
    fsweep : FlexibleSweep
        Sweep holding dressed energies and the ``jump_*`` rates.
    res_mode_idx, qubit_mode_idx : int
        Subsystem indices of the resonator and the ancilla.
    res_dim, qubit_dim : int
        Hilbert-space dimensions used to build the Hamiltonian.
    res_me_dim, qubit_me_dim : int
        Dimensions of the jump operators in the master equation. They must not
        exceed ``res_dim`` and ``qubit_dim``.
    in_rot_frame : bool
        If True, subtract the free rotation of the resonator and of the 0-1
        transition of the ancilla.
    res_n_ref : int
        Photon number at which the ancilla frequency of the rotating frame is
        evaluated. Only used when ``in_rot_frame`` is True.
    higher_order_decay : bool
        If True (default), keep every computed jump. If False, drop
        jumps past second order in
        :math:`\{(g/\Delta)^2, n_{\mathrm{th}}, \gamma_{\hat a},
        \gamma_\downarrow, \gamma_\phi, \gamma_{\hat a^\dagger\hat a}\}`.
    return_jump_names : bool
        If True, also return the names of the collapse operators as a fifth
        value. If False (default), return the first four values only.

    Returns
    -------
    hamiltonian : qutip.Qobj
        Truncated Hamiltonian in the unit of ``fsweep["evals"]``. The two
        modes are in the order of the subsystems of the ``HilbertSpace``:
        dims ``[[res_dim, qubit_dim], ...]`` if ``res_mode_idx`` is 0 and
        ``[[qubit_dim, res_dim], ...]`` if it is 1.
    c_ops : list of qutip.Qobj
        Collapse operators :math:`\sqrt{\gamma}\,L` in the same basis.
    eigensys : tuple
        Flattened truncated eigenvalues and eigenvectors.
    frame_hamiltonian : qutip.Qobj
        Generator of the rotating-frame map:
        :math:`H_{\mathrm{new}}=H_{\mathrm{old}}-H_{\mathrm{frame}}`.
    jump_names : list of str
        Only if ``return_jump_names`` is True. The name of each element of
        ``c_ops``, in the same order: the keys of :func:`dressed_jump_ops`
        (``jump_a``, ``jump_adag``, ``jump_adag_a``, ``jump_01``,
        ``jump_a_11``, ...).
    """
    if not fsweep.sweep.all_params_fixed(fsweep.sweep._current_param_indices):
        raise ValueError("It's a multi-parameter sweep and the slice is not "
                         "specified. Please try a single-parameter sweep or "
                         "specify the slice.")
        
    if res_n_ref != 0 and not in_rot_frame:
        raise ValueError("res_n_ref is only used when in_rot_frame is True.")
    if not isinstance(res_n_ref, int):
        raise ValueError("res_n_ref must be an integer.")
    
    hilbertspace = fsweep.hilbertspace
    dims = hilbertspace.subsystem_dims
    if len(dims) > 2:
        raise ValueError("More than 2 subsystems detected. The 'smart truncation' is not "
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
        res_truncated_dim=res_dim, qubit_truncated_dim=qubit_dim,
        dressed_indices=fsweep["dressed_indices"], 
        eigensys=(fsweep["evals"], fsweep["evecs"]),
        adjust_phase=True,
        # the axes follow the order of the tensor product, which the
        # frequencies below and the jump operators assume
        keep_resonator_first_mode=False,
    )
    truncated_dims = list(truncated_evals.shape)
    
    # hamiltonian in this basis
    flattened_evals = truncated_evals.ravel() - truncated_evals.ravel()[0]
    hamiltonian = qt.Qobj(np.diag(flattened_evals), dims=[truncated_dims, truncated_dims])

    if in_rot_frame:
        # in the dispersive regime, the Hamiltonian of the frame transformation is
        # freq_res * a^dag a * identity_qubit + identity_res * freq_qubit * qubit^dag qubit
        if res_mode_idx == 0:
            mode0_freq = truncated_evals[1, 0] - truncated_evals[0, 0]
            mode1_freq = truncated_evals[res_n_ref, 1] - truncated_evals[res_n_ref, 0]
        else:
            mode0_freq = truncated_evals[1, res_n_ref] - truncated_evals[0, res_n_ref]
            mode1_freq = truncated_evals[0, 1] - truncated_evals[0, 0]

        rot_hamiltonian = (
            qt.tensor(mode0_freq * qt.num(truncated_dims[0]), qt.qeye(truncated_dims[1]))
            + qt.tensor(qt.qeye(truncated_dims[0]), mode1_freq * qt.num(truncated_dims[1]))
        )
    else:
        rot_hamiltonian = qt.Qobj(np.zeros_like(hamiltonian.data), dims=hamiltonian.dims)

    hamiltonian -= rot_hamiltonian

    jump_ops_w_rate = dressed_jump_ops(
        fsweep, res_mode_idx, res_dim, qubit_dim, qubit_me_dim,
        higher_order_decay,
    )

    ingredients = (
        hamiltonian, 
        list(jump_ops_w_rate.values()), 
        (truncated_evals.ravel(), truncated_evecs.ravel()),
        rot_hamiltonian,
    )
    # the list of collapse operators does not carry their names; the names
    # are returned in the same order
    if return_jump_names:
        return ingredients + (list(jump_ops_w_rate.keys()),)
    return ingredients

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

from .ideal import qubit_projectors as ideal_qubit_projectors
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

    if not collapse_operators:
        # closed system: integrate the unitary evolution subspace by subspace
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

        if num_cpus > 1:
            map_method = get_map_method(num_cpus, cpu_per_node=1)
            prop_list = list(map_method(calculate_prop, evec_arr))
        else:
            prop_list = list(map(calculate_prop, evec_arr))

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
    else:
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
