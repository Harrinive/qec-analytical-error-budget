r"""
Master equation of the resonator-ancilla system with dressed jump
operators.

The jump operators act on the dressed states as the operators of the bare
modes act on the bare states with the same labels; their rates are the
``jump_*`` keys stored by ``batched_sweep_jump_rates``. In the
identifiers, ``qubit`` is the ancilla and ``res`` is the resonator.
"""

import numpy as np
import qutip as qt
import scqubits as scq
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.hilbert_space import HilbertSpace

from qec_aeb import cat_ideal as cat_ideal
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.cqed.flexible_sweep import FlexibleSweep

from typing import List, Tuple, Any, Dict, Callable, Optional
Esys = Tuple[np.ndarray, np.ndarray]


def keep_second_order_jump(
    op_family: str,
    i: Optional[int] = None,
    j: Optional[int] = None,
) -> bool:
    r"""
    Whether a dressed jump belongs to the second-order list of jumps of the
    Lindblad master equation.

    The undressed baths are the ohmic loss of the resonator, the dephasing of
    the resonator, the capacitive loss of the ancilla, and the dephasing of the
    ancilla. After dispersive dressing, the rate of a second-order jump is a
    leading bath rate times at most one factor of :math:`(g/\Delta)^2` or
    :math:`n_{\mathrm{th}}`. The kept jumps are

    - :math:`a`, :math:`a^\dagger`, :math:`a^\dagger a`;
    - :math:`|0\rangle\langle 1|`, :math:`|1\rangle\langle 0|`,
      :math:`|1\rangle\langle 1|`;
    - :math:`a|1\rangle\langle 1|`;
    - :math:`|n\rangle\langle 1|` with :math:`n>1` (thermal leakage from
      :math:`|1\rangle`).

    Examples of jumps of higher order are :math:`a^\dagger|1\rangle\langle 1|`
    (rate :math:`\sim n_{\mathrm{th}}(g/\Delta)^2\gamma_\downarrow`) and the
    correlated jumps :math:`a|i\rangle\langle j|` other than
    :math:`a|1\rangle\langle 1|`, such as :math:`a|0\rangle\langle 1|`, which
    removes a photon and an ancilla excitation together.

    Parameters
    ----------
    op_family : str
        One of ``jump_a``, ``jump_adag``, ``jump_adag_a``,
        ``jump_ij``, ``jump_a_ij``, ``jump_adag_ij``.
    i, j : int or None
        Ancilla labels of the factor :math:`|i\rangle\langle j|`. Ignored for
        the jumps of the resonator alone.

    Returns
    -------
    bool
        True if the jump is kept at second order.
    """
    if op_family in ("jump_a", "jump_adag", "jump_adag_a"):
        return True
    if op_family == "jump_ij":
        # |0><1|, |1><0|, |1><1|, and |n>1><1|
        return (j == 1) or (i, j) == (1, 0)
    if op_family == "jump_a_ij":
        # photon loss only when the ancilla is excited
        return (i, j) == (1, 1)
    if op_family == "jump_adag_ij":
        # a†|i><j| is at least n_th (g/Δ)^2, i.e. third order
        return False
    raise ValueError(f"Unknown jump family: {op_family}")


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
