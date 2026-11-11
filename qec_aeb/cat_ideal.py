r"""
Elementary operators and ideal operators of the cat-code QEC cycle on the
space of a resonator and an ancilla.

Every operator acts on the tensor product of the resonator (dimension
``res_dim``) and the ancilla (dimension ``qubit_dim``; ``qubit`` in the
identifiers is the ancilla). ``res_mode_idx`` is the position of the
resonator in the tensor product: 0 puts the resonator first, 1 the
ancilla first. A superoperator is in the qutip ``super`` representation
(column stacking).
"""

import numpy as np
import math
import qutip as qt
from typing import Literal, Callable, List, Tuple, overload, TYPE_CHECKING

from chencrafts.cqed import oprt_in_basis, superop_evolve, cat
if TYPE_CHECKING:
    from qec_aeb.QEC_graph.node import MeasurementRecord

# ##############################################################################
def _res_qubit_tensor(
    res_op: qt.Qobj, qubit_op: qt.Qobj, 
    res_mode_idx: Literal[0, 1],
) -> qt.Qobj:
    r"""
    Tensor product of a resonator operator and an ancilla operator, with the
    resonator first if ``res_mode_idx`` is 0 and the ancilla first if it is 1.
    """
    if res_mode_idx == 0:
        return qt.tensor(res_op, qubit_op)
    elif res_mode_idx == 1:
        return qt.tensor(qubit_op, res_op)

def eye(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
) -> qt.Qobj:
    r"""
    Identity operator on the resonator-ancilla space.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.

    Returns
    -------
    qt.Qobj
        The identity.
    """
    return _res_qubit_tensor(qt.qeye(res_dim), qt.qeye(qubit_dim), res_mode_idx)

def res_destroy(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0, 
) -> qt.Qobj:
    r"""
    Annihilation operator :math:`\hat a` of the resonator, times the identity
    on the ancilla.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.

    Returns
    -------
    qt.Qobj
        Operator on the resonator-ancilla space.
    """
    return _res_qubit_tensor(qt.destroy(res_dim), qt.qeye(qubit_dim), res_mode_idx)
    
def qubit_pauli(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    axis: Literal['x', 'y', 'z'] = 'x',
) -> qt.Qobj:
    r"""
    Pauli operator of the two lowest ancilla levels, times the identity on the
    resonator.

    The operator acts as the identity on the ancilla levels above 1.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    axis : Literal['x', 'y', 'z']
        Axis of the Pauli operator.

    Returns
    -------
    qt.Qobj
        Operator on the resonator-ancilla space.
    """
    qubit_oprt = np.eye(qubit_dim, dtype=complex)
    if axis == 'x':
        qubit_oprt[:2, :2] = qt.sigmax().full()
    elif axis == 'y':
        qubit_oprt[:2, :2] = qt.sigmay().full()
    elif axis == 'z':
        qubit_oprt[:2, :2] = qt.sigmaz().full()
    else:
        raise ValueError(f'Invalid axis {axis}')
    qubit_oprt = qt.Qobj(qubit_oprt)

    return _res_qubit_tensor(qt.qeye(res_dim), qubit_oprt, res_mode_idx)

def res_number(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    qubit_state: int | None = None,
) -> qt.Qobj:
    r"""
    Photon-number operator :math:`\hat a^\dagger\hat a` of the resonator, times
    the identity on the ancilla or times the projector on one ancilla level.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    qubit_state : int | None
        If None, the ancilla factor is the identity. Otherwise it is the
        projector :math:`|q\rangle\langle q|` on the ancilla level
        :math:`q` = ``qubit_state``.

    Returns
    -------
    qt.Qobj
        Operator on the resonator-ancilla space.
    """
    res_oprt = qt.num(res_dim)
    if qubit_state is None:
        qubit_oprt = qt.qeye(qubit_dim)
    else:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)
    
def qubit_proj(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    qubit_state: int = 0,
    qubit_state_2: int | None = None,
) -> qt.Qobj:
    r"""
    Ancilla operator :math:`|q\rangle\langle q'|`, times the identity on the
    resonator.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    qubit_state : int
        Ancilla level :math:`q`.
    qubit_state_2 : int | None
        Ancilla level :math:`q'`. If None, :math:`q'=q` and the operator is
        the projector on level :math:`q`, the operator of an ancilla
        measurement with that outcome.

    Returns
    -------
    qt.Qobj
        Operator on the resonator-ancilla space.
    """
    res_oprt = qt.qeye(res_dim)
    
    if qubit_state_2 is not None:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state_2)
    else:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)

def res_rotation(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = np.pi / 2,
) -> qt.Qobj:
    r"""
    Rotation :math:`e^{-i\varphi\,\hat a^\dagger\hat a}` of the resonator by
    the angle :math:`\varphi` = ``angle`` per photon, times the identity on the
    ancilla.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Rotation angle :math:`\varphi` (rad).

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    res_oprt = (-1j * angle * qt.num(res_dim)).expm()
    qubit_oprt = qt.qeye(qubit_dim)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)

def res_Kerr_rotation(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    Kerr: float = 0.0,
    time: float = 0.0,
) -> qt.Qobj:
    r"""
    Phase :math:`e^{-iKt\,\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)}` that
    the self-Kerr :math:`K` of the resonator produces in the time :math:`t`,
    times the identity on the ancilla.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    Kerr : float
        Self-Kerr :math:`K`, the coefficient of
        :math:`\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)` in the
        Hamiltonian (rad/ns for a time in ns).
    time : float
        Duration :math:`t`.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    num_op = qt.num(res_dim)
    kerr_op = num_op * (num_op - 1)
    res_oprt = (-1j * Kerr * time * kerr_op).expm()
    qubit_oprt = qt.qeye(qubit_dim)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)

def res_qubit_basis(
    res_dim: int, qubit_dim: int,
    bare_label: Tuple[int, int] = (0, 0),
    res_mode_idx: Literal[0, 1] = 0,
) -> qt.Qobj:
    r"""
    Product state of a Fock state of the resonator and a level of the ancilla.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    bare_label : Tuple[int, int]
        Labels of the two modes in the order of the tensor product:
        ``bare_label[res_mode_idx]`` is the photon number and the other entry
        is the ancilla level.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.

    Returns
    -------
    qt.Qobj
        Ket of the resonator-ancilla space.
    """
    res_basis = qt.basis(res_dim, bare_label[res_mode_idx])
    qubit_basis = qt.basis(qubit_dim, bare_label[1 - res_mode_idx])

    return _res_qubit_tensor(res_basis, qubit_basis, res_mode_idx)

# #############################################################################
def idling_maps(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1], 
    static_hamiltonian: qt.Qobj,
    time: float,
    decay_rate: float = 0.0,
    self_Kerr: float = 0.0,
) -> List[qt.Qobj]:
    r"""
    Ideal operators of the idle for no photon loss and for one photon loss.

    With the Hamiltonian :math:`\hat H`, the photon-loss rate :math:`\kappa`
    and the idle duration :math:`t`, the evolution without a loss is
    :math:`\hat E_0=e^{-\kappa t\,\hat a^\dagger\hat a/2}e^{-i\hat Ht}`. The
    operator for one loss is
    :math:`\hat E_0\,e^{-iKt\,\hat a^\dagger\hat a}\,\hat a`: the time of the
    loss is not recorded, and the rotation is the average over that time of
    the rotation that the self-Kerr :math:`K` produces after a loss. The
    operators are not normalized to the probabilities of the two outcomes.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    static_hamiltonian : qt.Qobj
        Hamiltonian :math:`\hat H` on the resonator-ancilla space (rad/ns for
        a time in ns).
    time : float
        Idle duration :math:`t`.
    decay_rate : float
        Photon-loss rate :math:`\kappa` (1/ns for a time in ns).
    self_Kerr : float
        Self-Kerr :math:`K` of the resonator (rad/ns for a time in ns).

    Returns
    -------
    List[qt.Qobj]
        The operators for no loss and for one loss.
    """
    # free evolution
    shrinkage_oprt = (
        -decay_rate * time / 2 * res_number(res_dim, qubit_dim, res_mode_idx)
    ).expm()
    free_evolution_oprt = shrinkage_oprt * (-1j * static_hamiltonian * time).expm()

    # single-photon loss related operators
    spl_rotation_oprt = res_rotation(
        res_dim, qubit_dim, res_mode_idx, angle = self_Kerr * time
    )   # average rotation due to self-Kerr
    a_oprt = res_destroy(res_dim, qubit_dim, res_mode_idx)

    return [
        free_evolution_oprt,    # zero-photon loss
        free_evolution_oprt * spl_rotation_oprt * a_oprt    # single-photon loss
    ]

@overload
def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: Literal[False] = False,
) -> List[qt.Qobj]:
    ...

@overload
def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: Literal[True] = True,
) -> qt.Qobj:
    ...

def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: bool = False,
) -> qt.Qobj | List[qt.Qobj]:
    r"""
    Kraus operators, or superoperator, of the photon loss of the resonator
    during an idle, truncated at a maximum number of lost photons.

    With the decay probability :math:`p=\kappa t` (loss rate times duration),
    the operator for :math:`k` lost photons is

    .. math::
        \hat E_k = e^{-p\,\hat a^\dagger\hat a/2}\,
        \frac{(1-e^{-p})^{k/2}}{\sqrt{k!}}\,\hat a^k .

    The sum over :math:`k\le` ``max_photon_loss`` does not preserve the trace
    when states with more photons than ``max_photon_loss`` are occupied.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    decay_prob : float
        Decay probability :math:`p`.
    max_photon_loss : int
        Largest number :math:`k` of lost photons that is kept.
    superop : bool
        If False, return the list of the operators :math:`\hat E_k`. If True,
        return the superoperator
        :math:`\hat\rho\mapsto\sum_k\hat E_k\hat\rho\hat E_k^\dagger`.

    Returns
    -------
    qt.Qobj | List[qt.Qobj]
        The superoperator, or the ``max_photon_loss + 1`` Kraus operators.
    """
    shrinkage_oprt = (-decay_prob / 2 * res_number(res_dim, qubit_dim, res_mode_idx)).expm()
    a_oprt = res_destroy(res_dim, qubit_dim, res_mode_idx)

    # Kraus representation of the decay channel
    kraus_op = lambda k: (
        1 / np.sqrt(math.factorial(k))
        * (1 - np.exp(-decay_prob)) ** (k / 2)
        * a_oprt ** k
    )    
    kraus_op_list = [kraus_op(k) for k in range(max_photon_loss + 1)]

    if superop:
        super_kraus = [qt.to_super(kraus) for kraus in kraus_op_list]
        return qt.to_super(shrinkage_oprt) * sum(super_kraus)
    else:
        shrink_kraus = [shrinkage_oprt * kraus for kraus in kraus_op_list]
        return shrink_kraus

def qubit_rot_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = np.pi / 2,
    axis: Literal['x', 'y', 'z', '-x', '-y', '-z'] = 'x',
    superop: bool = False,
) -> qt.Qobj:
    r"""
    Ideal rotation :math:`e^{-i\theta\hat\sigma/2}` of the ancilla in the
    rotating frame, times the identity on the resonator.

    :math:`\hat\sigma` is the Pauli operator of the two lowest ancilla levels
    for the given axis (``qubit_pauli``); the ancilla levels above 1 acquire
    the phase :math:`e^{-i\theta/2}`.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Rotation angle :math:`\theta` (rad).
    axis : Literal['x', 'y', 'z', '-x', '-y', '-z']
        Rotation axis. A minus sign reverses the sign of the angle.
    superop : bool
        If True, return the superoperator of the unitary.

    Returns
    -------
    qt.Qobj
        Unitary, or its superoperator, on the resonator-ancilla space.
    """
    if axis.startswith('-'):
        angle = -angle
        axis = axis[1:]

    generator = qubit_pauli(res_dim, qubit_dim, res_mode_idx, axis) / 2
    unitary = (-1j * angle * generator).expm()

    if superop:
        return qt.to_super(unitary)
    else:
        return unitary
    
def parity_mapping_propagator(
    res_dim: int, qubit_dim: int,
    angle: float = np.pi,
    res_mode_idx: Literal[0, 1] = 0,
    qubit_state: int = 1,
    superop: bool = False,
) -> qt.Qobj:
    r"""
    Ideal operator of the parity mapping,
    :math:`\exp\left[-i\varphi\,\hat a^\dagger\hat a\,|q\rangle\langle q|\right]`:
    a rotation of the resonator by :math:`\varphi` per photon if the ancilla
    is in the level :math:`q`.

    For :math:`\varphi=\pi`, the level :math:`q` acquires the sign of the
    photon-number parity.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    angle : float
        Phase :math:`\varphi` per photon (rad).
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    qubit_state : int
        Ancilla level :math:`q` that acquires the phase.
    superop : bool
        If True, return the superoperator of the unitary.

    Returns
    -------
    qt.Qobj
        Unitary, or its superoperator, on the resonator-ancilla space.
    """
    generator = res_number(res_dim, qubit_dim, res_mode_idx, qubit_state)
    unitary = (-1j * angle * generator).expm()

    if superop:
        return qt.to_super(unitary)
    else:
        return unitary
    
def parity_mapping_n_ref_frame(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = np.pi,
    n_ref: int = 0,
    qubit_state: int = 1,
) -> qt.Qobj:
    r"""
    Unitary of the parity mapping for an arbitrary angle, in the current
    frame: a phase on one ancilla level that is proportional to the photon
    number counted from ``n_ref``,

    .. math::
        \exp\left[-i\,\mathrm{angle}\,(\hat a^\dagger\hat a - n_{\mathrm{ref}})
        \,|q\rangle\langle q|\right],

    with :math:`q` = ``qubit_state``.

    The current frame rotates at the frequency that the ancilla has when the
    resonator holds ``n_ref`` photons, so the ancilla level ``qubit_state``
    acquires no phase in the subspace of ``n_ref`` photons, whereas
    ``parity_mapping_propagator`` gives no phase at zero photons. With
    ``qubit_state`` = 1, every other ancilla level is left unchanged. With
    ``qubit_state`` = 0, level 1 is left unchanged, and the ancilla levels
    above 1 acquire the phase :math:`e^{i\,\mathrm{angle}\,n_{\mathrm{ref}}}`,
    which does not depend on the photon number.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla (``qubit_dim`` >= 2).
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Phase per photon (rad).
    n_ref : int
        Photon number at which the ancilla frequency of the current frame is
        evaluated.
    qubit_state : int
        Ancilla level that acquires the phase, 0 or 1.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    assert qubit_state in [0, 1]

    parity_mapping_ideal = parity_mapping_propagator(
        res_dim=res_dim, qubit_dim=qubit_dim,
        angle=angle,
        res_mode_idx=res_mode_idx, superop=False,
        qubit_state=qubit_state,
    )

    # phase gate of the ancilla that moves the zero of the phase from zero
    # photons to n_ref photons: the product is the identity in the subspace of
    # n_ref photons
    extra_phase = - (angle * n_ref) % (2 * np.pi)
    qubit_rot_ideal = qubit_rot_propagator(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        axis="z" if qubit_state == 0 else "-z",
        # the axis gives the level 0 or 1 that is not qubit_state the lower
        # eigenvalue (-1/2) of the rotation generator
        angle= extra_phase,
        superop=False,
    ) * np.exp(
        -1 *        # cancel the phase of the level that is not qubit_state
        -1/2 *      # eigenvalue of that level
        -1j *       # sign convention exp(-1j * phase) of the rotation
        extra_phase
    )

    return parity_mapping_ideal * qubit_rot_ideal

def res_Kerr_rotation_n_ref_frame(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = 0.0,
    n_ref: int = 0,
) -> qt.Qobj:
    r"""
    Phase of the resonator that is quadratic in the photon number counted from
    ``n_ref``,

    .. math::
        \exp\left[-i\,\mathrm{angle}\,
        (\hat a^\dagger\hat a - n_{\mathrm{ref}})^2\right].

    With ``angle = chi_prime / 2 * time``, the unitary is the phase that the
    qubit-state-dependent resonator non-linearity :math:`\chi_1'`
    (``chi_prime``) produces in the duration ``time``, averaged over the two
    ancilla levels. The unitary is written in the current frame, which
    rotates at the ancilla frequency for ``n_ref`` photons.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Phase of the states with ``n_ref`` + 1 or ``n_ref`` - 1 photons (rad).
    n_ref : int
        Photon number at which the ancilla frequency of the current frame is
        evaluated.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space, the identity on the ancilla.
    """
    res_num_op = res_number(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
    ) - n_ref
    return (
        -1j * angle
        * (res_num_op)**2
    ).expm()

def finite_duration_qubit_rotation(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1],
    angle: float,
    duration: float,
    chi_1: float,
    K_s: float,
    chi_prime: float,
    n_ref: int,
    nu: float,
    axis: Literal['x', 'y', 'z', '-x', '-y', '-z'] = 'x',
) -> qt.Qobj:
    r"""
    Ideal operator of a :math:`\pm\pi/2` ancilla pulse of finite duration, in the
    current frame.

    During a pulse of duration :math:`t`, the dispersive shift :math:`\chi_1`, the
    self-Kerr :math:`K_s` of the resonator, and the qubit-state-dependent
    resonator non-linearity :math:`\chi_1'` keep acting. To first order in
    :math:`\chi_1`, the pulse is the ancilla rotation
    :math:`\hat R_{\mathrm{axis}}(\mathrm{angle})` between two controlled
    rotations :math:`\hat C(\varphi)` of the resonator, which rotate the
    resonator by :math:`\varphi` per photon if the ancilla is in :math:`|1\rangle`,
    times phases that depend on the photon number only. With
    :math:`\hat d=\hat a^\dagger\hat a-n_{\mathrm{ref}}` the photon number counted
    from ``n_ref``,

    .. math::
        \hat U = e^{-iK_st\,\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)}\,
        e^{-i\chi_1't\,\hat d^2/2}\,
        e^{-i\chi_1t(1/2-\nu)\hat d}\,
        \hat C(\varphi)\,\hat R_{\mathrm{axis}}(\mathrm{angle})\,\hat C(\varphi),
        \qquad
        \hat C(\varphi)=e^{-i\varphi\,\hat d\,|1\rangle\langle 1|},
        \quad\varphi=\chi_1t\,\nu .

    The envelope shape number :math:`\nu` (the second value of
    ``batched_sweeps_recipe.pulse_shape_numbers``) is the average of
    :math:`\sin\theta(t')` over the pulse, where :math:`\theta(t')` is the rotation
    angle completed at time :math:`t'`.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla (``qubit_dim`` >= 2). The
        rotation acts on the two lowest ancilla levels.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Rotation angle (rad), signed: :math:`+\pi/2` or :math:`-\pi/2`.
    duration : float
        Pulse duration :math:`t` (ns).
    chi_1 : float
        Dispersive shift :math:`\chi_1` per photon, evaluated at the mean photon
        number (rad/ns).
    K_s : float
        Self-Kerr :math:`K_s` of the resonator (rad/ns), the coefficient of
        :math:`\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)`.
    chi_prime : float
        :math:`\chi_1'` (rad/ns).
    n_ref : int
        Photon number at which the ancilla frequency of the current frame is
        evaluated.
    nu : float
        Envelope shape number :math:`\nu` of the pulse; :math:`2/\pi` for a square
        :math:`\pi/2` pulse.
    axis : Literal['x', 'y', 'z', '-x', '-y', '-z']
        Rotation axis of the ancilla.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    # rotation of the ancilla alone
    gate_ideal = qubit_rot_propagator(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=angle, axis=axis, superop=False,
    )

    # Fictitious Rabi frequency such that chi_1 / (2 * rabi_freq) equals the
    # controlled-rotation angle chi_1 * duration * nu. It uses the envelope
    # shape number of the pulse in place of the square-pulse value 2 / pi.
    rabi_freq = 1 / (2 * duration * nu)

    # rotation of the resonator: in each photon-number subspace, the unitary
    # of the pulse has a phase that depends on the photon number
    res_angle = chi_1 * (
        duration - 1 / rabi_freq
    ) / 2
    gate_res_rot_op = res_rotation(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=res_angle,
    ) * np.exp(1j * res_angle * n_ref)

    # controlled rotation of the resonator by chi_1 * duration * nu per
    # photon, the part of the parity mapping that one side of the pulse
    # contributes; its phase vanishes at n_ref photons, as the current frame
    # requires
    gate_pm_op = parity_mapping_n_ref_frame(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=chi_1 / 2 / rabi_freq,
        n_ref=n_ref,
        qubit_state=1,
    )

    # phases of the self-Kerr and of chi_prime during the pulse
    gate_res_Kerr_op_1 = res_Kerr_rotation(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        Kerr=K_s, time=duration,
    )
    gate_res_Kerr_op_2 = res_Kerr_rotation_n_ref_frame(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=chi_prime/2 * duration,
        n_ref=n_ref,
    )

    return (
        gate_res_Kerr_op_1 * gate_res_Kerr_op_2
        * gate_res_rot_op * gate_pm_op * gate_ideal * gate_pm_op
    )

def finite_duration_qubit_reset(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1],
    duration: float,
    chi_1: float,
    K_s: float,
    chi_prime: float,
    n_ref: int,
    axis: Literal['x', 'y', 'z', '-x', '-y', '-z'] = 'x',
) -> qt.Qobj:
    r"""
    Ideal operator of the reset :math:`\pi` pulse of finite duration, in the
    current frame.

    A :math:`\pi` pulse of duration :math:`t` exchanges the two ancilla levels, so
    the ancilla spends half of the duration in :math:`|1\rangle` whichever level
    it starts in. The dispersive shift :math:`\chi_1` therefore rotates the
    resonator by :math:`\chi_1t/2` per photon for both initial levels. The
    self-Kerr :math:`K_s` of the resonator and the qubit-state-dependent
    resonator non-linearity :math:`\chi_1'` act as in
    :func:`finite_duration_qubit_rotation`. With
    :math:`\hat d=\hat a^\dagger\hat a-n_{\mathrm{ref}}` the photon number counted
    from ``n_ref`` and :math:`\hat R_{\mathrm{axis}}(\pi)` the ancilla rotation,

    .. math::
        \hat U = e^{-iK_st\,\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)}\,
        e^{-i\chi_1't\,\hat d^2/2}\,
        e^{-i\chi_1t\,\hat a^\dagger\hat a/2}\,
        \hat R_{\mathrm{axis}}(\pi) .

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla (``qubit_dim`` >= 2). The
        rotation acts on the two lowest ancilla levels.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    duration : float
        Pulse duration :math:`t` (ns).
    chi_1 : float
        Dispersive shift :math:`\chi_1` per photon, evaluated at the mean photon
        number (rad/ns).
    K_s : float
        Self-Kerr :math:`K_s` of the resonator (rad/ns), the coefficient of
        :math:`\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)`.
    chi_prime : float
        :math:`\chi_1'` (rad/ns).
    n_ref : int
        Photon number at which the ancilla frequency of the current frame is
        evaluated.
    axis : Literal['x', 'y', 'z', '-x', '-y', '-z']
        Rotation axis of the ancilla.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    reset_ideal = qubit_rot_propagator(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=np.pi, axis=axis, superop=False,
    )
    # rotation of the resonator: in each photon-number subspace, the unitary
    # of the pulse has a phase that depends on the photon number
    res_rot_op = res_rotation(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=chi_1 * duration / 2,
    )
    # phases of the self-Kerr and of chi_prime during the pulse
    gate_res_Kerr_op_1 = res_Kerr_rotation(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        Kerr=K_s, time=duration,
    )
    gate_res_Kerr_op_2 = res_Kerr_rotation_n_ref_frame(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=chi_prime/2 * duration,
        n_ref=n_ref,
    )

    # the rotation of the resonator counts the photons from zero, not from
    # n_ref; the two choices differ by a global phase
    return (
        gate_res_Kerr_op_1 * gate_res_Kerr_op_2 * res_rot_op
        * reset_ideal
    )

def finite_duration_parity_mapping(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1],
    angle: float,
    duration: float,
    K_s: float,
    chi_prime: float,
    n_ref: int,
    qubit_state: int = 1,
) -> qt.Qobj:
    r"""
    Ideal operator of the parity-mapping idling of finite duration, in the
    current frame.

    While the ancilla idles for the duration :math:`t`, the dispersive shift
    rotates the resonator by ``angle`` per photon if the ancilla is in the
    level ``qubit_state``, and the self-Kerr :math:`K_s` of the resonator and the
    qubit-state-dependent resonator non-linearity :math:`\chi_1'` act on both
    ancilla levels. With :math:`\hat d=\hat a^\dagger\hat a-n_{\mathrm{ref}}` the
    photon number counted from ``n_ref``, and for ``qubit_state`` = 1,

    .. math::
        \hat U = e^{-iK_st\,\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)}\,
        e^{-i\chi_1't\,\hat d^2/2}\,
        e^{-i\,\mathrm{angle}\,\hat d\,|1\rangle\langle 1|} .

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla (``qubit_dim`` >= 2).
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        Angle of the controlled rotation per photon (rad). An idling of
        duration :math:`t` gives :math:`\chi_1t`, with :math:`\chi_1` the dispersive
        shift per photon. In the parity mapping of the protocol, this angle
        is :math:`\pm\pi` minus the angles that the two :math:`\pi/2` pulses
        contribute.
    duration : float
        Idling duration :math:`t` (ns).
    K_s : float
        Self-Kerr :math:`K_s` of the resonator (rad/ns), the coefficient of
        :math:`\hat a^\dagger\hat a(\hat a^\dagger\hat a-1)`.
    chi_prime : float
        :math:`\chi_1'` (rad/ns).
    n_ref : int
        Photon number at which the ancilla frequency of the current frame is
        evaluated.
    qubit_state : int
        Ancilla level that controls the rotation, 0 or 1.

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    # controlled rotation of the resonator; its phase vanishes at n_ref
    # photons, as the current frame requires
    parity_mapping_ideal = parity_mapping_n_ref_frame(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=angle,
        n_ref=n_ref,
        qubit_state=qubit_state,
    )

    # phases of the self-Kerr and of chi_prime during the idling
    pm_res_Kerr_op_1 = res_Kerr_rotation(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        Kerr=K_s, time=duration,
    )
    pm_res_Kerr_op_2 = res_Kerr_rotation_n_ref_frame(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        angle=chi_prime/2 * duration,
        n_ref=n_ref,
    )

    return pm_res_Kerr_op_1 * pm_res_Kerr_op_2 * parity_mapping_ideal

def cat_codewords(
    res_dim: int, qubit_dim: int,
    alpha: complex,
    res_mode_idx: Literal[0, 1] = 0,
) -> Tuple[qt.Qobj, qt.Qobj]:
    r"""
    The two logical codewords of the four-legged cat code, with the ancilla in
    its ground state:

    .. math::
        |0_L\rangle\propto(|\alpha\rangle+|-\alpha\rangle)|0\rangle,
        \qquad
        |1_L\rangle\propto(|i\alpha\rangle+|-i\alpha\rangle)|0\rangle .

    Each codeword is a superposition of two coherent states with even
    photon-number parity, normalized in the truncated space. The two
    codewords are not orthogonal to each other at a finite :math:`\alpha`.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    alpha : complex
        Amplitude :math:`\alpha` of the coherent states.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.

    Returns
    -------
    logical_0, logical_1 : qt.Qobj
        Kets of the resonator-ancilla space.
    """
    # product states |n>|0> of the photon number n and the ancilla ground
    # state, in which cat() expands the coherent states
    basis = [
        res_qubit_basis(
            res_dim, qubit_dim,
            bare_label=(n, 0) if res_mode_idx == 0 else (0, n),
            res_mode_idx=res_mode_idx,
        )
        for n in range(res_dim)
    ]
    logical_0 = cat([(1, alpha), (1, -alpha)], basis=basis)
    logical_1 = cat([(1, 1j * alpha), (1, -1j * alpha)], basis=basis)
    return logical_0, logical_1

def qubit_projectors(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
) -> List[qt.Qobj]:
    r"""
    Projectors :math:`|q\rangle\langle q|` of an ideal ancilla measurement,
    for the ancilla levels :math:`q=0,\dots,` ``qubit_dim`` - 1, times the
    identity on the resonator.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    superop : bool
        If True, return the superoperators
        :math:`\hat\rho\mapsto\hat P_q\hat\rho\hat P_q` of the projectors
        :math:`\hat P_q`.

    Returns
    -------
    List[qt.Qobj]
        One projector, or superoperator, per ancilla level.
    """
    ops = [
        qubit_proj(res_dim, qubit_dim, res_mode_idx, qubit_state=i) 
        for i in range(qubit_dim)
    ]
    if superop:
        return [qt.to_super(op) for op in ops]
    else:
        return ops
    
def qubit_measurement_func(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    order_by_prob: bool = True,
) -> Callable: 
    r"""
    Ideal projective measurement of the ancilla, as a function that maps a
    state to its outcomes.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    order_by_prob : bool
        If True, the outcomes are ordered by decreasing probability;
        otherwise by ancilla level.

    Returns
    -------
    Callable
        Function of a ket or a density matrix that returns a list of pairs
        (probability, normalized state after the measurement), one per
        ancilla level.
    """
    proj_list = qubit_projectors(
        res_dim, qubit_dim, res_mode_idx, superop=False
    )

    def measurement(state: qt.Qobj):
        prob_list = np.array([qt.expect(proj, state) for proj in proj_list], dtype=float)

        if order_by_prob:
            sorted_idx = np.argsort(prob_list)[::-1]
            sorted_prob_list = prob_list[sorted_idx]
            sorted_proj_list = [proj_list[i] for i in sorted_idx]
        else:
            sorted_prob_list = prob_list
            sorted_proj_list = proj_list

        if qt.isket(state):
            post_state_list = [
                proj * state / np.sqrt(prob) for proj, prob in zip(sorted_proj_list, sorted_prob_list)
            ]
        else:
            post_state_list = [
                proj * state * proj / prob for proj, prob in zip(sorted_proj_list, sorted_prob_list)
            ]
        
        return list(zip(sorted_prob_list, post_state_list))

    return measurement

def qubit_reset_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
    axis: Literal['x', 'y', 'z'] = 'x',
) -> qt.Qobj:
    r"""
    Ideal reset of the ancilla: the rotation by :math:`\pi` about ``axis``
    (``qubit_rot_propagator``), times the identity on the resonator.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    superop : bool
        If True, return the superoperator of the unitary.
    axis : Literal['x', 'y', 'z']
        Rotation axis.

    Returns
    -------
    qt.Qobj
        Unitary, or its superoperator, on the resonator-ancilla space.
    """
    return qubit_rot_propagator(
        res_dim, qubit_dim, res_mode_idx, angle=np.pi, axis=axis, superop=superop
    )

def qubit_frame_change(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = 0.0,
) -> qt.Qobj:
    r"""
    Ancilla frame change (virtual Z gate) :math:`e^{-i\Phi \hat n_\mathrm{q}}`.

    Advancing the frame of the ancilla by :math:`\Phi` shifts the phase of every 
    later drive pulse. In the current frame, it acts as the phase gate 
    :math:`e^{-i\Phi \hat n_\mathrm{q}}` with :math:`\hat n_\mathrm{q}` the 
    ancilla level index, on every photon-number subspace.

    Parameters
    ----------
    res_dim : int
        Number of resonator levels.
    qubit_dim : int
        Number of ancilla levels.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    angle : float
        :math:`\Phi` (rad).

    Returns
    -------
    qt.Qobj
        Unitary on the resonator-ancilla space.
    """
    level_phase = qt.Qobj(np.diag(np.exp(-1j * angle * np.arange(qubit_dim))))
    return _res_qubit_tensor(qt.qeye(res_dim), level_phase, res_mode_idx)

def identity(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
) -> qt.Qobj:
    r"""
    Identity operator, or identity superoperator, on the resonator-ancilla
    space.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    res_mode_idx : Literal[0, 1]
        Position of the resonator in the tensor product.
    superop : bool
        If True, return the superoperator.

    Returns
    -------
    qt.Qobj
        The identity.
    """
    if superop:
        return qt.to_super(eye(res_dim, qubit_dim, res_mode_idx))
    else:
        return eye(res_dim, qubit_dim, res_mode_idx)