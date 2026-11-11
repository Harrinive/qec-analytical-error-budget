r"""Rotating-wave model of the three ancilla pulses of the cat-code protocol.

Integrates the closed and open rotating-wave master equations of the DRAG
Gaussian ancilla pulses (:math:`\hat X_{\pi/2}`, :math:`\hat X_{-\pi/2}`,
conditional :math:`\hat X_\pi`) across photon-number subspaces with selectable
ancilla jump channels. Units: durations in ns, angular frequencies in rad/ns.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

import qec_aeb as bsq
from chencrafts.cqed.pulses import DRAGGaussian
from chencrafts.cqed.qt_helper import component_liouvillian, superop_from_component_maps

# importing device_model sets the symmetric orthonormalization of the error
# states before any tree builder is made
import device_model
from budget_tools import JUMP_NAMES, ideal_idle_states, idle_model_from_builder

# ancilla jumps that act during a pulse, by the names of budget_tools.JUMP_NAMES
CHANNELS = ("ancilla_relaxation", "ancilla_excitation", "ancilla_dephasing",
            "ancilla_jump_to_second_excited_level")


# ---------------------------------------------------------------- device ----
def build_device(sweep, simulation_parameters: dict) -> dict:
    r"""Dressed subspace Hamiltonians, the pulses of ``chencrafts``, and the ideal states.

    Parameters
    ----------
    sweep : cqed.FlexibleSweep
        Dressed sweep of ``device_model.py``.
    simulation_parameters : dict
        Simulation parameters with which the sweep was built.

    Returns
    -------
    dict
        The tree builder (``builder``, which gives the ideal operators);
        dressed energies and drive matrix elements of every subspace m;
        pulse and device constants; the ancilla jumps of one subspace
        (``jumps``); the master equation of the idling (``idle``) and the
        ideal states of the two error spaces after the idle
        (``idle_states``); ``pulses_closed``, the operators of the three
        pulses of ``chencrafts`` without ancilla decoherence, in the current
        frame (``plus`` and ``minus`` with the frame changes of the protocol,
        ``pi``); and ``pulses_open``, the dynamical maps of the three pulses
        with the ancilla jumps, as they enter the actual evolution tree.
    """
    sim = simulation_parameters
    builder = device_model.configure_tree_builder(sweep, sim)
    # the pulses of chencrafts without ancilla decoherence: unitaries
    builder.gate_qubit_decoherence = False
    builder.build_qubit_gate_process(num_cpus=1)
    builder.build_qubit_reset_process(num_cpus=1)

    # Dressed energies and drive matrix elements in each subspace m. The
    # pulses are integrated with gate_levels levels; the dynamical maps keep
    # the ancilla levels of the dynamics, anc_dim
    res_dim, anc_dim = builder.res_dim, builder.qubit_dim
    gate_levels = sim["gate_simulated_qubit_dim"]
    subspaces = bsq.cat_real.dressed_subspace_operators(
        sweep.hilbertspace, builder.res_mode_idx, builder.qubit_mode_idx,
        res_truncated_dim=res_dim, qubit_truncated_dim=gate_levels,
        dressed_indices=sweep["dressed_indices"],
        eigensys=(sweep["evals"], sweep["evecs"]),
        drive_operator=device_model.BUDGET_PARAMETERS["drive_operator"])
    energies = subspaces["energies"] - subspaces["energies"][0, 0]

    # The pulses as operators in the current frame; the pi/2 pulses have the
    # frame changes of the protocol on both sides, the pi pulse has none and
    # lasts twice as long
    t_pulse = float(sweep["tau_p_eff"])
    stark_frame = builder.ancilla_frame_change(-builder.gate_frame_phase)
    lab_pulses = builder.lab_frame_pulses
    pulses_closed = dict(
        plus=builder.gate_map_in_current_frame(
            lab_pulses["qubit_gate_p"], t_pulse, stark_frame, as_operator=True).full(),
        minus=builder.gate_map_in_current_frame(
            lab_pulses["qubit_gate_m"], t_pulse, stark_frame, as_operator=True).full(),
        pi=builder.gate_map_in_current_frame(
            lab_pulses["qubit_reset"], 2 * t_pulse, as_operator=True).full())

    # The ancilla jumps sqrt(gamma)|i><j| of the master equation. They are
    # the same in every photon-number subspace, so the block of the vacuum is
    # the operator of the ancilla alone
    collapse_operators = builder.collapse_operators
    jumps = {
        channel: collapse_operators[JUMP_NAMES[channel]].full()
        .reshape(res_dim, anc_dim, res_dim, anc_dim)[0, :, 0, :]
        for channel in CHANNELS}

    idle = idle_model_from_builder(sweep, builder)
    n_ref = builder.n_bar_ref
    device = dict(
        builder=builder,
        res_dim=res_dim,
        anc_dim=anc_dim,
        drag=device_model.BUDGET_PARAMETERS["pulse_scheme"] == "drag",
        gate_levels=gate_levels,
        energies=energies, drive=subspaces["drive_matrix_elements"],
        frame=np.real(np.diag(builder.frame_hamiltonian.full())),
        t_pulse=t_pulse, n_ref=n_ref,
        # the carrier is resonant with the ancilla in the subspace n_ref
        omega_d=energies[n_ref, 1] - energies[n_ref, 0],
        # ancilla constants of the subspace n_ref, as the pulses and the
        # analytical formula use them
        eta=float(sweep["anc_eta"]), stark_coeff=float(sweep["anc_stark_coeff"]),
        chi1=builder.average_pm_interaction,
        frame_update=bool(builder.use_frame_update),
        pulses_closed=pulses_closed,
        jumps=jumps,
        # ideal operator of the parity-mapping idling of the protocol
        parity_mapping=builder.finite_duration_parity_mapping().full(),
        idle=idle,
        idle_states=ideal_idle_states(idle),
    )
    # the pulses of the actual evolution tree: the same builder with the
    # ancilla jumps
    builder.gate_qubit_decoherence = True
    builder.build_qubit_gate_process(num_cpus=1)
    builder.build_qubit_reset_process(num_cpus=1)
    step_maps = builder.step_maps
    device["pulses_open"] = dict(plus=step_maps["qubit_gate_p"].full(),
                                 minus=step_maps["qubit_gate_m"].full(),
                                 pi=step_maps["qubit_reset"].full())
    return device


def ideal_pulse_operators(builder, t_pulse: float) -> dict:
    r"""Ideal operators of the three ancilla pulses of the cycle, from the tree builder.

    Parameters
    ----------
    builder : bsq.QEC_graph.FullCatTreeBuilder
        Tree builder of the dressed sweep (``builder`` of :func:`build_device`).
    t_pulse : float
        Duration of a :math:`\pi/2` pulse (ns), ``sweep["tau_p_eff"]``; the
        conditional :math:`\hat X_\pi` pulse lasts twice as long.

    Returns
    -------
    dict
        ``X_plus_half_pi``, ``X_minus_half_pi``, ``conditional_X_pi`` to the
        ideal operator of the pulse on the resonator-ancilla space, a matrix.
    """
    return dict(
        X_plus_half_pi=builder.finite_duration_qubit_rotation(np.pi / 2, t_pulse).full(),
        X_minus_half_pi=builder.finite_duration_qubit_rotation(-np.pi / 2, t_pulse).full(),
        conditional_X_pi=builder.finite_duration_qubit_reset(2 * t_pulse).full(),
    )


# ----------------------------------------------------------------- pulses ---
def pulse_envelopes(theta: float, duration: float, eta: float, drag: bool = True):
    r"""Rabi rate and DRAG quadrature of the pulse of ``cat_real.qubit_gate``.

    The Rabi rate is the Gaussian of width ``duration / 4`` shifted to vanish
    at the ends, scaled to the rotation angle ``theta``; the quadrature is
    ``-Omega'(t) / eta`` (first-order DRAG).

    Parameters
    ----------
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).
    eta : float
        Anharmonicity of the ancilla (rad/ns).
    drag : bool
        If False the pulse is a plain Gaussian: the quadrature is zero.

    Returns
    -------
    rabi, quadrature : callable
        Functions of the time ``t`` (ns), for scalar ``t``.
    """
    pulse = DRAGGaussian(
        base_angular_freq=1.0, duration=duration, sigma=duration / 4,
        non_lin=eta, order=2, rotation_angle=theta,
    )
    if drag:
        return pulse.envelope_I, pulse.envelope_Q

    def no_quadrature(t):
        r"""Quadrature of a plain Gaussian pulse.

        Parameters
        ----------
        t : float
            Time (ns).

        Returns
        -------
        float
            Zero.
        """
        return 0.0

    return pulse.envelope_I, no_quadrature


def pulse_has_frame_changes(dev: dict, theta: float) -> bool:
    r"""Whether the protocol applies ancilla frame changes around a pulse.

    The frame changes by minus the Stark-shift phase belong to the pi/2
    pulses; the conditional pi pulse has none.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.

    Returns
    -------
    bool
    """
    return bool(abs(theta) < np.pi and dev["frame_update"])


def tilt_angles(dev: dict, theta: float, duration: float) -> tuple:
    r"""Tilt angles of a pulse from the Stark shift and from the dispersive shift.

    The angles are those of ``qec_aeb.cat_pulse_tilt_angles``,
    the kernel of the analytical formula. The ideal operator of a +-pi/2
    pulse keeps the shape number of the pulse itself (``GATE_NU_HALF``, with
    the sign of the angle), so the dispersive-shift angle vanishes for it;
    the ideal operator of the pi pulse keeps none.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed; both angles are odd in it.
    duration : float
        Pulse duration (ns).

    Returns
    -------
    phi_s, phi_chi : float
        Stark-shift angle (rad) and dispersive-shift angle (rad per photon).
    """
    # the shape numbers carry the sign of the rotation angle
    mu, nu = bsq.pulse_shape_numbers(theta)
    ideal_nu = np.sign(theta) * bsq.QEC_graph.GATE_NU_HALF if abs(theta) < np.pi else 0.0
    return bsq.cat_pulse_tilt_angles(
        theta, duration, dev["chi1"], dev["stark_coeff"], mu, nu, ideal_nu)


def frame_change_angle(dev: dict, theta: float, duration: float) -> float:
    r"""Angle of the ancilla frame change before and after a pulse.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).

    Returns
    -------
    float
        Minus the Stark-shift phase of the pulse of angle ``abs(theta)`` (the
        two pi/2 pulses have the same frame change); zero for a pulse without
        frame changes (:func:`pulse_has_frame_changes`).
    """
    if not pulse_has_frame_changes(dev, theta):
        return 0.0
    return -tilt_angles(dev, abs(theta), duration)[0]


def current_frame_phases(dev: dict, m: int, duration: float, levels: int | None = None):
    r"""Phase factors that take a propagator of subspace m to the current frame of the builder.

    The model integrates in the frame that rotates level q of subspace m at
    ``E[m, 0] + q omega_d``; the lab-frame propagator is that propagator times
    ``exp(-i (E[m, 0] + q omega_d) t)`` on its final level q, and the current
    frame of ``chencrafts`` multiplies the lab-frame propagator by
    ``exp(i frame t)``.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    m : int
        Photon number of the subspace.
    duration : float
        Pulse duration (ns).
    levels : int or None
        Number of ancilla levels; the levels of the dynamics by default.

    Returns
    -------
    np.ndarray
        One factor per ancilla level, for the final index of a propagator.
    """
    anc_dim = dev["anc_dim"]
    levels = anc_dim if levels is None else levels
    frame = dev["frame"][anc_dim * m:anc_dim * m + levels]
    return np.exp(1j * frame * duration) * np.exp(
        -1j * (np.arange(levels) * dev["omega_d"] + dev["energies"][m, 0]) * duration)


def subspace_operators(dev: dict, m: int, levels: int | None = None):
    r"""Detuning and raising operator of subspace m in the frame rotating at the drive.

    The frame rotates level q at ``E[m, 0] + q omega_d``; the Hamiltonian is
    ``detuning + c raising + c* raising^dag`` with ``c = (Omega + i Q) / 2``.

    Rotating-wave approximation: in this frame the drive matrix element
    between the levels q and q' oscillates at ``(|q - q'| - 1) omega_d`` and
    ``(|q - q'| + 1) omega_d``. The raising operator holds the elements with
    q - q' = 1, the only ones with a static part; the elements with
    q - q' >= 2, the counter-rotating terms, and the diagonal elements of the
    drive operator are dropped.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    m : int
        Photon number of the subspace.
    levels : int or None
        Ancilla levels; ``dev["gate_levels"]`` by default.

    Returns
    -------
    detuning, raising : np.ndarray
        ``levels`` x ``levels`` matrices.
    """
    levels = dev["gate_levels"] if levels is None else levels
    e, h = dev["energies"], dev["drive"]
    n_ref = dev["n_ref"]
    detuning = np.diag(e[m, :levels] - e[m, 0] - np.arange(levels) * dev["omega_d"])
    # The pulse of chencrafts removes the sign of the 0-1 element from the
    # drive operator, which acts on the raising part as the unit-modulus
    # phase of that element
    element_01 = h[n_ref][0, 1]
    coupling = h[m][:levels, :levels] / abs(element_01)
    # only the elements between neighbouring levels are static in the frame
    lower = np.diag(np.diag(coupling, -1), -1)
    raising = lower * (element_01 / abs(element_01))
    return detuning.astype(complex), raising


def subspace_unitary(dev: dict, m: int, theta: float, duration: float,
                     levels: int | None = None, drag: bool | None = None) -> np.ndarray:
    r"""Rotating-wave propagator of subspace m, in the current frame of the builder.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    m : int
        Photon number of the subspace.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).
    levels : int or None
        Ancilla levels that are integrated; ``dev["gate_levels"]`` by
        default. ``levels=2`` drops the second excited level.
    drag : bool or None
        Whether the pulse has the DRAG quadrature; that of the protocol by
        default.

    Returns
    -------
    np.ndarray
        ``anc_dim`` x ``anc_dim`` matrix: the integrated levels, and the
        identity on a level that is not integrated.
    """
    levels = dev["gate_levels"] if levels is None else levels
    drag = dev["drag"] if drag is None else drag
    rabi, quadrature = pulse_envelopes(theta, duration, dev["eta"], drag)
    detuning, raising = subspace_operators(dev, m, levels)

    def rhs(t, y):
        r"""Schroedinger equation of the propagator in the drive frame.

        Parameters
        ----------
        t : float
            Time (ns).
        y : np.ndarray
            Flattened propagator.

        Returns
        -------
        np.ndarray
            Its time derivative, flattened.
        """
        c = (rabi(t) + 1j * quadrature(t)) / 2
        ham = detuning + c * raising + np.conj(c) * raising.conj().T
        return (-1j * ham @ y.reshape(levels, levels)).ravel()

    sol = solve_ivp(rhs, [0, duration], np.eye(levels, dtype=complex).ravel(),
                    method="DOP853", rtol=1e-10, atol=1e-12)
    # keep the levels of the dynamics and go to the current frame
    kept = min(levels, dev["anc_dim"])
    u_rot = sol.y[:, -1].reshape(levels, levels)[:kept, :kept]
    u = np.eye(dev["anc_dim"], dtype=complex)
    u[:kept, :kept] = np.diag(current_frame_phases(dev, m, duration, kept)) @ u_rot
    return u


def model_gate(dev: dict, theta: float, duration: float, **switches) -> np.ndarray:
    r"""Closed model of the pulse on the whole resonator-ancilla space, without frame changes.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).
    **switches
        ``levels`` and ``drag`` of :func:`subspace_unitary`.

    Returns
    -------
    np.ndarray
        The operator that acts as the propagator of subspace m in subspace m.
    """
    # the pulse does not change the photon number: one block per subspace
    return block_diag(*[subspace_unitary(dev, m, theta, duration, **switches)
                        for m in range(dev["res_dim"])])


def frame_change_operator(dev: dict, theta: float, duration: float) -> np.ndarray:
    r"""Ancilla frame change (virtual Z gate) of the protocol before and after a pulse.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).

    Returns
    -------
    np.ndarray
        The frame change by the angle of :func:`frame_change_angle`, on the
        resonator-ancilla space.
    """
    return bsq.cat_ideal.qubit_frame_change(
        dev["res_dim"], dev["anc_dim"], dev["builder"].res_mode_idx,
        frame_change_angle(dev, theta, duration)).full()


def framed_model_gate(dev: dict, theta: float, duration: float) -> np.ndarray:
    r"""Closed model of a pulse with the frame changes of the protocol.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).

    Returns
    -------
    np.ndarray
        :func:`model_gate` between the frame changes by minus the Stark-shift
        phase for a +-pi/2 pulse; :func:`model_gate` itself for the pi pulse.
    """
    u = model_gate(dev, theta, duration)
    if not pulse_has_frame_changes(dev, theta):
        return u
    z = frame_change_operator(dev, theta, duration)
    return z @ u @ z


# ------------------------------------------------------------ open system ---
def component_map(dev: dict, m: int, m_prime: int, theta: float, duration: float,
                  channels: list) -> np.ndarray:
    r"""Dynamical map of the component rho_{mm'} of the density matrix over one pulse.

    The component between the photon-number subspaces m and m' obeys a master
    equation in which the operators to its left are those of subspace m and
    the operators to its right those of subspace m'. The map is integrated in
    the frame that rotates at the drive frequency and returned in the current
    frame of the builder.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    m, m_prime : int
        Photon numbers of the two subspaces.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).
    channels : list of str
        Names of the ancilla jumps (entries of :data:`CHANNELS`) that act.

    Returns
    -------
    np.ndarray
        ``anc_dim**2`` x ``anc_dim**2`` matrix in the column-stacking convention.
    """
    levels = dev["gate_levels"]
    rabi, quadrature = pulse_envelopes(theta, duration, dev["eta"], dev["drag"])
    # the jumps are the same on both sides of the component
    jump_operators = [qt.Qobj(dev["jumps"][name]) for name in channels]

    # Hamiltonian detuning + Omega X + Q Y of each subspace
    parts = []
    for photon_number in (m, m_prime):
        detuning, raising = subspace_operators(dev, photon_number, levels)
        parts.append((qt.Qobj(detuning),
                      qt.Qobj((raising + raising.conj().T) / 2),
                      qt.Qobj(1j * (raising - raising.conj().T) / 2)))
    (det_a, x_a, y_a), (det_b, x_b, y_b) = parts
    static, in_phase = component_liouvillian(
        det_a, x_a, jump_operators, det_b, x_b, jump_operators)
    # the quadrature drive: the same construction without detuning and jumps
    _, in_quadrature = component_liouvillian(0 * det_a, y_a, [], 0 * det_b, y_b, [])
    static, in_phase, in_quadrature = static.full(), in_phase.full(), in_quadrature.full()

    def rhs(t, y):
        r"""Master equation of the dynamical map of the component in the drive frame.

        Parameters
        ----------
        t : float
            Time (ns).
        y : np.ndarray
            Flattened superoperator.

        Returns
        -------
        np.ndarray
            Its time derivative, flattened.
        """
        generator = static + rabi(t) * in_phase + quadrature(t) * in_quadrature
        return (generator @ y.reshape(levels ** 2, levels ** 2)).ravel()

    sol = solve_ivp(rhs, [0, duration], np.eye(levels ** 2, dtype=complex).ravel(),
                    method="DOP853", rtol=1e-10, atol=1e-12)
    # axes (final column, final row, initial column, initial row)
    tensor = sol.y[:, -1].reshape((levels,) * 4)[(slice(0, dev["anc_dim"]),) * 4]

    # to the current frame, as in subspace_unitary: rho_mm' -> P_m rho P_m'^dag
    phases = [current_frame_phases(dev, photon_number, duration)
              for photon_number in (m, m_prime)]
    tensor = tensor * (phases[1].conj()[:, None, None, None] * phases[0][None, :, None, None])
    return tensor.reshape(dev["anc_dim"] ** 2, dev["anc_dim"] ** 2)


def model_gate_map(dev: dict, theta: float, duration: float,
                   channels: list | None = None) -> np.ndarray:
    r"""Open model: dynamical map of the pulse on the whole resonator-ancilla space.

    For a +-pi/2 pulse the frame changes of the protocol, by minus the
    Stark-shift phase before and after the pulse, are included; the pi pulse
    has none.

    Parameters
    ----------
    dev : dict
        Output of :func:`build_device`.
    theta : float
        Rotation angle (rad), signed.
    duration : float
        Pulse duration (ns).
    channels : list of str or None
        Names of the ancilla jumps that act; all of :data:`CHANNELS` by
        default.

    Returns
    -------
    np.ndarray
        Superoperator in the column-stacking convention.
    """
    res_dim = dev["res_dim"]
    channels = list(CHANNELS) if channels is None else channels
    # the component (m', m) is the Hermitian conjugate of (m, m')
    maps = {(m, m_prime): component_map(dev, m, m_prime, theta, duration, channels)
            for m in range(res_dim) for m_prime in range(m, res_dim)}
    superop = superop_from_component_maps(maps).full()
    if pulse_has_frame_changes(dev, theta):
        # the frame change Z rho Z^dag on both sides of the pulse
        z = frame_change_operator(dev, theta, duration)
        frame_change = np.kron(z.conj(), z)
        superop = frame_change @ superop @ frame_change
    return superop
