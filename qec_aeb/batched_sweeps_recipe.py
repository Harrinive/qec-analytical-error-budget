r"""
Batched sweeps for the cat-code QEC cycle of a storage resonator coupled to
an ancilla.

The sweeps are registered on a ``ParameterSweep`` (``fs.sweep`` of a
``FlexibleSweep``) in this order:

0. ``batched_sweep_general``: dispersive shift, self-Kerr, and the detuning
   ``non_lin`` of the nearest leakage transition.
1. ``batched_sweep_dressed_op``: system-bath coupling operators in the
   dressed basis.
2. ``batched_sweep_jump_rates``: rates of the dressed jump operators of the
   Lindblad master equation.
3. ``batched_sweep_pulse``: width and duration of the ancilla pulses.
4. ``batched_sweep_cat_budget_ingredients``: scalar ingredients of the
   analytical one-cycle error budget ``error_budgeting.cat_code_budget``.

In the identifiers, ``qubit`` is the ancilla and ``res`` is the storage
resonator.
"""

__all__ = [
    'batched_sweep_general',
    'batched_sweep_pulse',
    'batched_sweep_dressed_op',
    'batched_sweep_jump_rates',
    'batched_sweep_cat_budget_ingredients',
    'cat_error_space_distribution',
    'cat_error_space_moments',
    'cat_kerr_rotation_infidelity',
    'photon_number_moments',
    'cat_stark_shift_coefficient',
    'pulse_shape_numbers',
    'pulse_population_averages',
    'DEFAULT_CAT_BUDGET_PARA',
]

import warnings

import numpy as np
import qutip as qt
import scipy as sp
import scqubits as scq
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import gammaln
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

from chencrafts.cqed.custom_sweeps import sweep_convergence
from chencrafts.cqed.clustered_me import (
    ohmic_spectral_density, cap_spectral_density, ind_spectral_density,
    delta_spectral_density, q_cap_fun, q_ind_fun,
)
from qec_aeb import cat_ideal as cat_ideal
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from qec_aeb.cat_recipe import get_jump_ops
from qec_aeb.error_budgeting import cat_pulse_tilt_angles

from typing import List, Tuple, Any, Dict, Callable

# When considering dephasing, we only consider the zero-frequency transition
# because the noise spectral density is 1/f.
TPHI_HI_FREQ_CUTOFF = 1e-8

PI2 = np.pi * 2

# Step0: Static properties and pulse parameters ########################
# ######################################################################

def batched_sweep_general(
    sweep: ParameterSweep, res_mode_idx: int = 0, qubit_mode_idx: int = 1,
    **kwargs,
) -> None:
    r"""
    Store the static properties of the resonator--ancilla system.

    The sweep needs the scqubits sweep data ``bare_evals``, ``chi``,
    ``chi_prime`` and ``kerr``.

    Stored keys (angular frequencies in rad/ns unless stated):

    - ``omega_a_GHz``: bare ancilla 0-1 transition frequency (GHz).
    - ``omega_s_GHz``: bare resonator frequency (GHz).
    - ``chi_sa``: dispersive shift :math:`2\pi\chi_{01}` of the resonator
      frequency when the ancilla is in :math:`|1\rangle`.
    - ``K_s``: resonator self-Kerr :math:`2\pi K`, the coefficient of
      :math:`a^{\dagger 2}a^2` in the Hamiltonian.
    - ``chi_prime_sa``: qubit-state-dependent resonator non-linearity
      :math:`2\pi\chi'`, the coefficient of :math:`a^{\dagger 2}a^2\otimes|1\rangle\langle 1|`; NaN if
      the resonator has only two levels.
    - ``non_lin``: smallest detuning between the ancilla 0-1 transition and
      any transition from level 0 or 1 into a higher level; NaN if the
      ancilla has only two levels.
    - ``convergence``: truncation check of the ancilla
      (``cqed.custom_sweeps.sweep_convergence``).

    Parameters
    ----------
    sweep:
        The parameter sweep.
    res_mode_idx, qubit_mode_idx:
        Subsystem indices of the resonator and the ancilla.
    **kwargs:
        Ignored; accepted so that a shared keyword dictionary can be passed.
    """
    # Bare transition frequencies and dispersive coefficients
    qubit_evals = sweep["bare_evals"][qubit_mode_idx]
    sweep.store_data(
        omega_a_GHz = (qubit_evals[..., 1]
            - qubit_evals[..., 0]),
        omega_s_GHz = (sweep["bare_evals"][res_mode_idx][..., 1]
            - sweep["bare_evals"][res_mode_idx][..., 0]),
        chi_sa = sweep["chi"][qubit_mode_idx, res_mode_idx][..., 1] * PI2,
        K_s = sweep["kerr"][res_mode_idx, res_mode_idx] * PI2,
    )

    # in case the "resonator" has only 2 levels
    try:
        sweep.store_data(
            chi_prime_sa = sweep["chi_prime"][qubit_mode_idx, res_mode_idx][..., 1] * PI2,
        )
    except IndexError:
        sweep.store_data(
            chi_prime_sa = np.zeros_like(sweep["chi_sa"]) * np.nan,
        )

    # non-linearities; in case the "qubit" has only 2 levels
    try:
        non_lin = np.min(np.abs([
            qubit_evals[..., 2:] - qubit_evals[..., 0:1] - sweep["omega_a_GHz"][..., None],
            qubit_evals[..., 2:] - qubit_evals[..., 1:2] - sweep["omega_a_GHz"][..., None],
        ]), axis=(0, -1)) * PI2
        sweep.store_data(non_lin = NamedSlotsNdarray(non_lin, sweep.param_info))
    except IndexError:
        sweep.store_data(
            non_lin = np.ones_like(sweep["chi_sa"]) * np.nan,
        )

    # convergence
    sweep.add_sweep(
        sweep_convergence, "convergence", mode_idx = qubit_mode_idx
    )


def batched_sweep_pulse(
    sweep: ParameterSweep, res_mode_idx: int = 0, qubit_mode_idx: int = 1,
    sigma_exists: bool = False,
    bound_by_nonlin: bool = True, bound_by_freq: bool = True,
    min_sigma: float = 2.0, max_sigma: float = 50.0,
    **kwargs
) -> None:
    r"""
    Store the Gaussian width and the duration of the ancilla pulses.

    The sweep needs the keys of ``batched_sweep_general``. Unless
    ``sigma_exists``, the width is

    .. math::
        \sigma = \min\Bigl[\sigma_{\max},\ \max\bigl(\sigma_{\min},
        \ |\texttt{sigma\_2\_K\_a}/\texttt{non\_lin}|,
        \ |\texttt{sigma\_omega}/(2\pi\,\omega_{a})|\bigr)\Bigr].

    The second bound keeps the pulse spectrally narrow compared with the
    detuning ``non_lin`` of the nearest leakage transition, which limits
    leakage to the second excited level. The third bound keeps the pulse at
    least about one carrier period long (with ``sigma_omega`` =
    :math:`2\pi`), which keeps the rotating-wave approximation valid.

    Stored keys: ``sigma`` (ns), ``tau_p = sigma * tau_p_by_sigma`` (ns), and
    ``tau_p_eff = sigma * tau_p_eff_by_sigma`` (ns). ``tau_p_eff`` is the
    duration of a :math:`\pi/2` pulse, a Gaussian of width ``tau_p_eff / 4``
    truncated at :math:`\pm 2\sigma`.

    Parameters
    ----------
    sweep:
        The parameter sweep. Must contain ``omega_a_GHz`` (the ancilla
        frequency :math:`\omega_a`, GHz) and ``non_lin`` (rad/ns).
    res_mode_idx, qubit_mode_idx:
        Subsystem indices of the resonator and the ancilla; not used.
    sigma_exists:
        If True, read ``sigma`` from the parameters instead of choosing it.
    bound_by_nonlin, bound_by_freq:
        Enable the lower bounds set by ``non_lin`` and by the carrier
        frequency.
    min_sigma, max_sigma:
        Absolute lower and upper bounds :math:`\sigma_{\min}` and
        :math:`\sigma_{\max}` on :math:`\sigma` (ns).
    **kwargs:
        Parameters that are not swept: ``sigma_2_K_a``, ``sigma_omega``,
        ``tau_p_by_sigma``, ``tau_p_eff_by_sigma`` (and ``sigma`` if
        ``sigma_exists``). A keyword given here overrides a swept parameter
        of the same name.
    """
    # Merge the swept-parameter grids with any fallback kwargs before
    # choosing sigma (a kwarg here overrides a same-named swept parameter)
    params = sweep.parameters.meshgrids_by_paramname() | kwargs

    if not sigma_exists:
        # find sigma based on:
        # 1. should be larger than (1 / non_lin) to reduce leakage
        # 2. should be larger than (1 / freq) to reduce non-RWA error
        # 3. should be larger than a minimum value (2) empirically
        # 4. should be smaller than a maximum value (50) empirically
        min_bounds = [np.ones_like(sweep["omega_a_GHz"]) * min_sigma]
        if bound_by_nonlin:
            min_sigma_by_nonlin = np.abs(params["sigma_2_K_a"] / sweep["non_lin"])
            min_bounds.append(min_sigma_by_nonlin)
        if bound_by_freq:
            min_sigma_by_freq = np.abs(params["sigma_omega"] / sweep["omega_a_GHz"] / np.pi / 2)
            min_bounds.append(min_sigma_by_freq)
        sigma = np.max(min_bounds, axis=0)

        max_bounds = [
            sigma, np.ones_like(sweep["omega_a_GHz"]) * max_sigma
        ]
        sigma = np.min(max_bounds, axis=0)
        sweep.store_data(
            sigma = sigma,
        )
    else:
        sweep.store_data(
            sigma = params["sigma"],
        )

    sweep.store_data(
        tau_p = sweep["sigma"] * np.abs(params["tau_p_by_sigma"]),
        tau_p_eff = sweep["sigma"] * np.abs(params["tau_p_eff_by_sigma"])
    )

# Step1: Transform the system-bath coupling operators ##################
# ######################################################################
# Requirements: 
# 1. Two modes: resonator and qubit. 
# 2. Dimension of the resonator: smaller than the original sweep by at least 1,
#    Dimension of the qubit: 2, 

def _sweep_organized_evecs(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
):
    r"""
    Dressed states of the resonator-ancilla system at one grid point, ordered
    by their bare labels and truncated to ``res_trunc_dim`` resonator levels
    and ``qubit_trunc_dim`` ancilla levels, as a flat array.
    """
    hilbertspace = sweep.hilbertspace

    evals, evecs = two_mode_dressed_esys(
        hilbertspace=hilbertspace,
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        state_label=(-1, -1),
        res_truncated_dim=res_trunc_dim, 
        qubit_truncated_dim=qubit_trunc_dim,
        dressed_indices=sweep["dressed_indices"][idx],
        eigensys=(sweep["evals"][idx], sweep["evecs"][idx]),
        adjust_phase=True,
        keep_resonator_first_mode=False,
    )
    
    if np.prod(evecs.shape) < res_trunc_dim * qubit_trunc_dim:
        print(evecs.shape)
        raise ValueError(
            "The truncated dimension is too large, please make it smaller."
        )
    
    return evecs.ravel()

def _sweep_dressed_qubit_op(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0,
    qubit_mode_idx = 1, 
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    operator_name = "n_operator"
):
    r"""
    Ancilla operator in the dressed basis ``org_evecs``, as a ``Qobj`` with the
    truncated dimensions.

    ``operator_name`` is an operator attribute of the ancilla circuit (e.g.
    ``"n_operator"``) or ``"proj_ij"``, the operator
    :math:`|i\rangle\langle j|` of the bare ancilla levels :math:`i` and
    :math:`j`.
    """
    hilbertspace = sweep.hilbertspace
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    
    # obtain and transform the operator
    if operator_name.startswith("proj_"):
        op = hilbertspace.hubbard_operator(
            j = int(operator_name[5]),
            k = int(operator_name[6]),
            subsystem = qubit,
        )
    else:
        op = scq.identity_wrap(
            operator = operator_name,
        subsystem = qubit,
        subsys_list = hilbertspace.subsystem_list,
    )
    op_dressed = oprt_in_basis(op, sweep["org_evecs"][idx])
    
    # change it to the desired dimension
    dim = hilbertspace.subsystem_dims
    dim[qubit_mode_idx] = qubit_trunc_dim
    dim[res_mode_idx] = res_trunc_dim

    return qt.Qobj(op_dressed, dims=[dim, dim])

def _sweep_dressed_res_op(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, 
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    operator_name = "a_m_adag",
):
    r"""
    Resonator operator in the dressed basis ``org_evecs``, as a ``Qobj`` with
    the truncated dimensions: ``"a_m_adag"`` is
    :math:`i(\hat a-\hat a^\dagger)` and ``"adag_a"`` is
    :math:`\hat a^\dagger\hat a`.
    """
    hilbertspace = sweep.hilbertspace
    res = hilbertspace.subsystem_list[res_mode_idx]
    
    # obtain and transform the operator
    a_op = hilbertspace.annihilate(res)
    
    if operator_name == "a_m_adag":
        op = 1j * a_op - 1j * a_op.dag()
    elif operator_name == "adag_a":
        op = a_op.dag() * a_op
    else:
        raise ValueError(f"Invalid resonator operator name: {operator_name}")

    op_dressed = oprt_in_basis(op, sweep["org_evecs"][idx])

    # change it to the desired dimension
    dim = hilbertspace.subsystem_dims
    dim[res_mode_idx] = res_trunc_dim
    dim[qubit_mode_idx] = qubit_trunc_dim

    return qt.Qobj(
        op_dressed, 
        dims=[dim, dim]
    )

def batched_sweep_dressed_op(
    sweep: ParameterSweep, 
    res_mode_idx: int = 0, qubit_mode_idx: int = 1,
    res_trunc_dim: int = 5,
    qubit_trunc_dim: int = 2,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    **kwargs
) -> None:
    r"""
    Store the system-bath coupling operators of the resonator and of the
    ancilla in the dressed basis.

    Stored keys:

    - ``org_evecs``: dressed states ordered by their bare labels, truncated
      to ``res_trunc_dim`` resonator levels and ``qubit_trunc_dim`` ancilla
      levels.
    - ``qubit_<name>`` for each name of ``qubit_op_names``: the ancilla
      operator in the basis ``org_evecs``. A name is an operator attribute of
      the ancilla circuit (``"n_operator"``, ``"phi_operator"``) or
      ``"proj_ij"``, the operator :math:`|i\rangle\langle j|` of the bare
      ancilla levels :math:`i` and :math:`j`.
    - ``res_<name>`` for each name of ``res_op_names``: ``"a_m_adag"``, the
      operator :math:`i(\hat a-\hat a^\dagger)`, or ``"adag_a"``, the operator
      :math:`\hat a^\dagger\hat a`, in the same basis.

    Parameters
    ----------
    sweep:
        The parameter sweep, with ``evals``, ``evecs`` and
        ``dressed_indices``.
    res_mode_idx, qubit_mode_idx:
        Subsystem indices of the resonator and the ancilla.
    res_trunc_dim, qubit_trunc_dim:
        Numbers of resonator and ancilla levels kept. ``res_trunc_dim`` must
        be smaller than the resonator dimension of the sweep.
    qubit_op_names, res_op_names:
        Names of the ancilla and resonator operators to transform.
    **kwargs:
        Ignored.
    """    
    
    sweep.add_sweep(
        _sweep_organized_evecs, 
        sweep_name="org_evecs",
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        res_trunc_dim=res_trunc_dim,
        qubit_trunc_dim=qubit_trunc_dim,
    )
    for op_name in qubit_op_names:
        sweep.add_sweep(
            _sweep_dressed_qubit_op, 
            sweep_name=f"qubit_{op_name}",
            qubit_mode_idx=qubit_mode_idx,
            res_mode_idx=res_mode_idx,
            res_trunc_dim=res_trunc_dim,
            qubit_trunc_dim=qubit_trunc_dim,
            operator_name=op_name,
        )
    for op_name in res_op_names:
        sweep.add_sweep(
            _sweep_dressed_res_op, 
            sweep_name=f"res_{op_name}",
            res_mode_idx=res_mode_idx,
            qubit_mode_idx=qubit_mode_idx,
            res_trunc_dim=res_trunc_dim,
            qubit_trunc_dim=qubit_trunc_dim,
            operator_name=op_name,
        )

# Step2: Extract the jump operators from the system-bath coupling operators
# and compute the associated rates
# #########################################################################
def _sweep_spectral_density(
    sweep: ParameterSweep,
    idx,
    qubit_mode_idx = 1,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    **kwargs
):
    r"""
    Spectral-density functions of the baths at one grid point, as a dictionary
    from ``"qubit_<name>"`` or ``"res_<name>"`` to a function of frequency.

    The ancilla baths read ``temp_a`` (K), ``Q_cap``, ``Q_ind`` and
    ``kappa_a_phi`` (1/ns), and the resonator baths read ``temp_s``, ``Q_s``
    and ``kappa_s_phi`` (1/ns), from the sweep parameters or from ``kwargs``.
    """
    params = {
        key: val[idx] for key, val in sweep.parameters.meshgrids_by_paramname().items()
    } | kwargs
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    
    spec_dens = {}
    for op in qubit_op_names:
        if op == "n_operator":
            EC = qubit.EC
            func = lambda freq: cap_spectral_density(
                freq, T = params["temp_a"], EC = EC, Q_cap = params["Q_cap"]
            )
        elif op == "phi_operator":
            EL = qubit.EL
            func = lambda freq: ind_spectral_density(
                freq, T = params["temp_a"], EL = EL, Q_ind = params["Q_ind"]
            )
        elif op == "proj_11":
            func = lambda freq: delta_spectral_density(
                freq, peak_value = 2 * params["kappa_a_phi"], peak_loc = 0, peak_width = TPHI_HI_FREQ_CUTOFF
            )
        else:
            raise ValueError(f"Invalid qubit operator name: {op}")
        spec_dens[f"qubit_{op}"] = func
    
    for op in res_op_names:
        if op == "a_m_adag":
            func = lambda freq: ohmic_spectral_density(
                freq, T = params["temp_s"], Q = params["Q_s"]
            )
        elif op == "adag_a":
            func = lambda freq: delta_spectral_density(
                freq, peak_value = params["kappa_s_phi"], peak_loc = 0, peak_width = TPHI_HI_FREQ_CUTOFF
            )
        else:
            raise ValueError(f"Invalid resonator operator name: {op}")
        spec_dens[f"res_{op}"] = func
        
    return spec_dens


# Just a note here, for a qubit, coefficient_of_proj_11 = 2 --> dephasing rate = 1
# to see this, run the following code:

# import qutip as qt
# import chencrafts.cqed as cqed
# import chencrafts.toolbox as tb
# liouv = qt.liouvillian(
#     c_ops = [qt.projection(2, 1, 1) * np.sqrt(1)]
# )
# prop = lambda t: (liouv * t).expm()
# init_state = qt.ket2dm((qt.basis(2, 1) + qt.basis(2, 0)).unit())
# t_array = np.linspace(0, 10, 100)
# pop1 = []
# for t in t_array:
#     pop1.append(qt.expect(init_state.dag() * init_state, cqed.superop_evolve(prop(t), init_state)))
# print(tb.decay_rate(t_array, pop1, extract_envelope=False))
    

def _ravel_res_qubit_index(
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5, qubit_trunc_dim = 2,
    res_state = 0, qubit_state = 0,
) -> int:
    r"""
    Index of the bare state (``res_state``, ``qubit_state``) in the flattened
    two-mode basis, for either order of the two modes.
    """
    if res_mode_idx == 0 and qubit_mode_idx == 1:
        return np.ravel_multi_index(
            (res_state, qubit_state), 
            (res_trunc_dim, qubit_trunc_dim)
        )
    elif res_mode_idx == 1 and qubit_mode_idx == 0:
        return np.ravel_multi_index(
            (qubit_state, res_state), 
            (qubit_trunc_dim, res_trunc_dim)
        )
    raise ValueError(f"Invalid mode indices: {res_mode_idx}, {qubit_mode_idx}")

def _calc_mat_elem(
    op: qt.Qobj,
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5, qubit_trunc_dim = 2,
    res_state_1 = 0, qubit_state_1 = 0,
    res_state_2 = 1, qubit_state_2 = 0,
    **kwargs
) -> float:
    r"""
    Matrix element :math:`\langle 1|\hat O|2\rangle` of ``op`` between the
    states with the bare labels (``res_state_1``, ``qubit_state_1``) and
    (``res_state_2``, ``qubit_state_2``); it belongs to the jump operator
    :math:`|1\rangle\langle 2|`.
    """
    idx1 = _ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, res_trunc_dim, qubit_trunc_dim, 
        res_state_1, qubit_state_1
    )
    idx2 = _ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, res_trunc_dim, qubit_trunc_dim, 
        res_state_2, qubit_state_2
    )
    
    # matrix element
    mat_elem = op[idx1, idx2]
    
    return mat_elem
    
def _calc_freq_diff(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_state_1 = 0, qubit_state_1 = 0,
    res_state_2 = 1, qubit_state_2 = 0,
    **kwargs
) -> float:
    r"""
    Dressed energy of state 2 minus that of state 1 (GHz), the two states
    being given by their bare labels.
    """
    res = sweep.hilbertspace.subsystem_list[res_mode_idx]
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    
    # note that we don't use "res_trunc_dim" and "qubit_trunc_dim" here
    # because evals are still organized before organization & truncation
    res_dim = res.truncated_dim
    qubit_dim = qubit.truncated_dim
    
    idx1 = _ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, 
        res_dim, qubit_dim, 
        res_state_1, qubit_state_1
    )
    idx2 = _ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, 
        res_dim, qubit_dim, 
        res_state_2, qubit_state_2
    )
    
    # frequency difference
    drs_idx1 = sweep["dressed_indices"][idx][idx1]
    drs_idx2 = sweep["dressed_indices"][idx][idx2]
    omega_1 = sweep["evals"][idx][drs_idx1]
    omega_2 = sweep["evals"][idx][drs_idx2]
    omega_diff = omega_2 - omega_1
    
    return omega_diff

def _compute_rate_by_a_m_adag(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    res_ref_state = 0,
    **kwargs
):
    r"""
    Add the rates that the bath coupled through
    :math:`i(\hat a-\hat a^\dagger)` produces to ``jump_a``, ``jump_adag`` and
    ``jump_ij``, at every grid point.

    A rate is the spectral density of the bath at the transition frequency
    times the squared matrix element of the dressed coupling operator. The
    elements of :math:`\hat a` and :math:`\hat a^\dagger` are read between the
    resonator levels ``res_ref_state`` and ``res_ref_state + 1``, and those of
    :math:`|i\rangle\langle j|` at ``res_ref_state``.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density["res_a_m_adag"]
        except KeyError:
            return
        sys_bath_inter_op = sweep["res_a_m_adag"][idx]
        
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )

        # a 
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = res_ref_state, qubit_state_1 = 0,
            res_state_2 = res_ref_state + 1, qubit_state_2 = 0,
        )
        a_mat_elem = _calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (a op has this factor)
        a_freq_diff = _calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_a = spec_dens(a_freq_diff) * np.abs(a_mat_elem)**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + jump_a

        # adag
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = res_ref_state + 1, qubit_state_1 = 0,
            res_state_2 = res_ref_state, qubit_state_2 = 0,
        )
        adag_mat_elem = _calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (adag op has this factor)
        adag_freq_diff = _calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_adag = spec_dens(adag_freq_diff) * np.abs(adag_mat_elem)**2
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # qubit |i><j|
        jump_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state, qubit_state_1 = i,
                res_state_2 = res_ref_state, qubit_state_2 = j,
            )
            q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **state_kwargs
            )   # don't need to normalize, |i><j| op is supposed to be the same across resonator levels
            q_freq_diff = _calc_freq_diff(
                sweep, idx, **state_kwargs
            )
            jump_q_matelems[i, j] = spec_dens(q_freq_diff) * np.abs(q_mat_elem)**2
            
        # <00|op|00> should already be 0, so the following is not needed
        # q_matelems = q_matelems - np.eye(qubit_trunc_dim) * q_matelems[0, 0]
        sweep[f"jump_ij"][idx] = sweep[f"jump_ij"][idx] + np.real(jump_q_matelems)
    
def _compute_rate_by_adag_a(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    res_ref_state = 0,
    **kwargs
):
    r"""
    Add the rates that the bath coupled through :math:`\hat a^\dagger\hat a`
    produces to ``jump_adag_a``, ``jump_a``, ``jump_adag``, ``jump_a_ij`` and
    ``jump_adag_ij``, at every grid point.

    A rate is the spectral density of the bath at the transition frequency
    times the squared matrix element of the dressed coupling operator. The
    element of :math:`\hat a^\dagger\hat a` is read at the resonator level
    ``res_ref_state + 1``, and those of :math:`\hat a|i\rangle\langle j|` and
    :math:`\hat a^\dagger|i\rangle\langle j|` between ``res_ref_state`` and
    ``res_ref_state + 1``.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density["res_adag_a"]
        except KeyError:
            return
        
        sys_bath_inter_op = sweep["res_adag_a"][idx].copy()
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )
        
        # adag_a 
        # Since we don't have adag_a * |i><j| and <00|op|00> is already 0, 
        # we just need to calculate one matrix element.
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = res_ref_state + 1, qubit_state_1 = 0,
            res_state_2 = res_ref_state + 1, qubit_state_2 = 0,
        )
        adag_a_mat_elem = _calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        ) / (res_ref_state + 1)  # extracted at high level, normalize (adag_a op has this factor)
        adag_a_freq_diff = _calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_adag_a = spec_dens(adag_a_freq_diff) * np.abs(adag_a_mat_elem)**2
        sweep[f"jump_adag_a"][idx] = sweep[f"jump_adag_a"][idx] + np.real(jump_adag_a)

        # a/adag * |i><j|
        a_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        a_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            a_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state, qubit_state_1 = i,
                res_state_2 = res_ref_state + 1, qubit_state_2 = j,
            )
            a_q_freq_diff = _calc_freq_diff(
                sweep, idx, **a_q_state_kwargs
            )
            a_q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **a_q_state_kwargs
            ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (a op has this factor)
            a_q_freq_diffs[i, j] = a_q_freq_diff
            a_q_matelems[i, j] = a_q_mat_elem
            
            adag_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state + 1, qubit_state_1 = i,
                res_state_2 = res_ref_state, qubit_state_2 = j,
            )
            adag_q_freq_diff = _calc_freq_diff(
                sweep, idx, **adag_q_state_kwargs
            )
            adag_q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **adag_q_state_kwargs
            ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (adag op has this factor)
            adag_q_freq_diffs[i, j] = adag_q_freq_diff
            adag_q_matelems[i, j] = adag_q_mat_elem
        
        # identity component: a / adag (add to jump_a and jump_adag)
        jump_a = spec_dens(a_q_freq_diffs[0, 0]) * np.abs(a_q_matelems[0, 0])**2
        jump_adag = spec_dens(adag_q_freq_diffs[0, 0]) * np.abs(adag_q_matelems[0, 0])**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + np.real(jump_a)
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # the rest: correlated jumps
        a_q_matelems = (
            a_q_matelems 
            - np.eye(qubit_trunc_dim) * a_q_matelems[0, 0]
        )
        adag_q_matelems = (
            adag_q_matelems 
            - np.eye(qubit_trunc_dim) * adag_q_matelems[0, 0]
        )
        sweep[f"jump_a_ij"][idx] = np.real(
            sweep[f"jump_a_ij"][idx] 
            + spec_dens(a_q_freq_diffs) * np.abs(a_q_matelems)**2
        )
        sweep[f"jump_adag_ij"][idx] = np.real(
            sweep[f"jump_adag_ij"][idx] 
            + spec_dens(adag_q_freq_diffs) * np.abs(adag_q_matelems)**2
        )
        
def _compute_rate_by_qubit_ops(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    op_name: str = "n_operator",
    res_ref_state = 0,
    **kwargs
):
    r"""
    Add the rates that the bath coupled through the ancilla operator
    ``op_name`` produces to ``jump_ij``, ``jump_a``, ``jump_adag``,
    ``jump_a_ij`` and ``jump_adag_ij``, at every grid point.

    A rate is the spectral density of the bath at the transition frequency
    times the squared matrix element of the dressed coupling operator. The
    elements of :math:`|i\rangle\langle j|` are read at the resonator level
    ``res_ref_state``, and those of :math:`\hat a|i\rangle\langle j|` and
    :math:`\hat a^\dagger|i\rangle\langle j|` between ``res_ref_state`` and
    ``res_ref_state + 1``.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density[f"qubit_{op_name}"]
        except KeyError:
            return
    
        sys_bath_inter_op = sweep[f"qubit_{op_name}"][idx].copy()
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )
        
        # qubit |i><j|
        jump_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state, qubit_state_1 = i,
                res_state_2 = res_ref_state, qubit_state_2 = j,
            )
            q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **state_kwargs
            )   # don't need to normalize, |i><j| op is supposed to be the same across resonator levels
            q_freq_diff = _calc_freq_diff(
                sweep, idx, **state_kwargs
            )
            jump_q_matelems[i, j] = spec_dens(q_freq_diff) * np.abs(q_mat_elem)**2
            
        # <00|op|00> should already be 0, so the following is not needed
        # q_matelems = q_matelems - np.eye(qubit_trunc_dim) * q_matelems[0, 0]
        sweep[f"jump_ij"][idx] = sweep[f"jump_ij"][idx] + np.real(jump_q_matelems)
        
        # a/adag * |i><j|
        a_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        a_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            a_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state, qubit_state_1 = i,
                res_state_2 = res_ref_state + 1, qubit_state_2 = j,
            )
            a_q_freq_diff = _calc_freq_diff(
                sweep, idx, **a_q_state_kwargs
            )
            a_q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **a_q_state_kwargs
            ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (a op has this factor)
            a_q_freq_diffs[i, j] = a_q_freq_diff
            a_q_matelems[i, j] = a_q_mat_elem
            
            adag_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = res_ref_state + 1, qubit_state_1 = i,
                res_state_2 = res_ref_state, qubit_state_2 = j,
            )
            adag_q_freq_diff = _calc_freq_diff(
                sweep, idx, **adag_q_state_kwargs
            )
            adag_q_mat_elem = _calc_mat_elem(
                sys_bath_inter_op, **adag_q_state_kwargs
            ) / np.sqrt(res_ref_state + 1)  # extracted at high level, normalize (adag op has this factor)
            adag_q_freq_diffs[i, j] = adag_q_freq_diff
            adag_q_matelems[i, j] = adag_q_mat_elem
        
        # identity component: a / adag (add to jump_a and jump_adag)
        jump_a = spec_dens(a_q_freq_diffs[0, 0]) * np.abs(a_q_matelems[0, 0])**2
        jump_adag = spec_dens(adag_q_freq_diffs[0, 0]) * np.abs(adag_q_matelems[0, 0])**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + np.real(jump_a)
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # the rest: correlated jumps (add to jump_a_ij and jump_adag_ij)
        a_q_matelems = (
            a_q_matelems 
            - np.eye(qubit_trunc_dim) * a_q_matelems[0, 0]
        )
        adag_q_matelems = (
            adag_q_matelems 
            - np.eye(qubit_trunc_dim) * adag_q_matelems[0, 0]
        )
        sweep[f"jump_a_ij"][idx] = np.real(
            sweep[f"jump_a_ij"][idx] 
            + spec_dens(a_q_freq_diffs) * np.abs(a_q_matelems)**2
        )
        sweep[f"jump_adag_ij"][idx] = np.real(
            sweep[f"jump_adag_ij"][idx] 
            + spec_dens(adag_q_freq_diffs) * np.abs(adag_q_matelems)**2
        )
        
def batched_sweep_jump_rates(
    sweep: ParameterSweep,
    res_mode_idx: int = 0, qubit_mode_idx: int = 1,
    res_trunc_dim: int = 5,
    qubit_trunc_dim: int = 2,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    res_ref_state: int = 0,
    **kwargs
) -> None:
    r"""
    Store the rates of the dressed jump operators of the Lindblad master
    equation.

    Each bath couples to the system through one operator
    (``batched_sweep_dressed_op``). In the dressed basis, that operator has
    matrix elements between states that differ in the photon number, in the
    ancilla level, or in both. Each such transition is a jump whose rate is
    the spectral density of the bath at the transition frequency times the
    squared matrix element.

    Stored keys (rates in 1/ns):

    - ``spectral_density``: dictionary of the spectral-density functions of
      the baths, one per coupling operator.
    - ``jump_a``, ``jump_adag``, ``jump_adag_a``: rates of the jumps
      :math:`\hat a`, :math:`\hat a^\dagger` and
      :math:`\hat a^\dagger\hat a`.
    - ``jump_ij``: rates of the ancilla jumps :math:`|i\rangle\langle j|`, in
      an array with the trailing shape ``(qubit_trunc_dim, qubit_trunc_dim)``.
    - ``jump_a_ij``, ``jump_adag_ij``: rates of the correlated jumps
      :math:`\hat a|i\rangle\langle j|` and
      :math:`\hat a^\dagger|i\rangle\langle j|`, with the same shape.
    - ``kappa_s``: the photon-loss rate, equal to ``jump_a``.
    - ``K_s``: self-Kerr of the resonator (rad/ns).

    Parameters
    ----------
    sweep:
        The parameter sweep, with the keys of ``batched_sweep_dressed_op``.
        The bath parameters ``temp_a`` (K), ``Q_cap``, ``Q_ind``,
        ``kappa_a_phi`` (1/ns), ``temp_s``, ``Q_s`` and ``kappa_s_phi``
        (1/ns) are sweep parameters.
    res_mode_idx, qubit_mode_idx:
        Subsystem indices of the resonator and the ancilla.
    res_trunc_dim, qubit_trunc_dim:
        Numbers of resonator and ancilla levels of the dressed operators.
    qubit_op_names, res_op_names:
        Names of the coupling operators, as in ``batched_sweep_dressed_op``.
        The ancilla baths are ``"n_operator"`` (capacitive loss),
        ``"phi_operator"`` (inductive loss) and ``"proj_11"`` (dephasing); the
        resonator baths are ``"a_m_adag"`` (ohmic loss) and ``"adag_a"``
        (dephasing).
    res_ref_state:
        Resonator level at which the matrix elements are read: those of
        :math:`\hat a` and :math:`\hat a^\dagger` between ``res_ref_state``
        and ``res_ref_state + 1``, those of :math:`|i\rangle\langle j|` at
        ``res_ref_state``.
    **kwargs:
        Ignored.
    """
    zeros = np.zeros(
        sweep.parameters.counts
    )
    zeros_qij = np.zeros(
        sweep.parameters.counts + (qubit_trunc_dim, qubit_trunc_dim),
    )
    sweep.store_data(
        # resonator jumps
        jump_a = zeros.copy(),
        jump_adag = zeros.copy(),
        jump_adag_a = zeros.copy(),
        
        # qubit jumps
        jump_ij = zeros_qij.copy(), # depolarization + dephasing (11)
        
        # resonator-qubit jumps
        jump_a_ij = zeros_qij.copy(),
        jump_adag_ij = zeros_qij.copy(),
    )
    
    sweep.add_sweep(
        _sweep_spectral_density,
        sweep_name = "spectral_density",
        qubit_mode_idx = qubit_mode_idx,
        qubit_op_names = qubit_op_names,
        res_op_names = res_op_names,
    )
    
    _compute_rate_by_a_m_adag(
        sweep,
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
        res_ref_state = res_ref_state,
    )
    _compute_rate_by_adag_a(
        sweep,
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
        res_ref_state = res_ref_state,
    )
    for op_name in qubit_op_names:
        _compute_rate_by_qubit_ops(
            sweep,
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            op_name = op_name,
            res_ref_state = res_ref_state,
        )
        
    # for fitting into `cat_tree.py`, we need to store some dict values 
    # with a different name
    sweep.store_data(
        kappa_s = sweep["jump_a"],
        K_s = sweep["kerr"][res_mode_idx, res_mode_idx] * np.pi * 2,
    )

# We can stop here and get the master equation and simulate it numerically

# Step 3: Compute the grouped jump probability (unfinished)
# #########################################################################
def _sweep_jump_ops(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    r"""
    Jump operators of ``cat_recipe.get_jump_ops`` at one grid point (part of
    an unfinished sweep of the jump probabilities).
    """
    ops = get_jump_ops(
        res_mode_idx = res_mode_idx,
        res_trunc_dim = res_trunc_dim,
        qubit_trunc_dim = qubit_trunc_dim,
    )
    
    return ops


def _sweep_1jump_prob(
    sweep: ParameterSweep, idx,
    **kwargs
):
    r"""
    Unfinished: the probability of a single jump is not computed, and the
    function returns nothing.
    """
    jump_ops = sweep["jump_ops"][idx]
    
    
    pass
    
    

def _batched_sweep_jump_prob(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    r"""
    Unfinished sweep of the jump probabilities; stores ``jump_ops`` and an
    empty ``1jump_prob``.
    """
    sweep.add_sweep(
        _sweep_jump_ops,
        sweep_name = "jump_ops",
        res_mode_idx = res_mode_idx,
        res_trunc_dim = res_trunc_dim,
        qubit_trunc_dim = qubit_trunc_dim,
    )
    
    sweep.add_sweep(
        _sweep_1jump_prob,
        sweep_name = "1jump_prob",
    )

# Step 4: Ingredients of the analytical one-cycle error budget
# #########################################################################
# The analytical error budget ``error_budgeting.cat_code_budget`` is a
# closed-form expression in a few scalars per grid point: Lindblad rates (from
# Step 2), durations, dispersive coefficients, two ancilla constants, the
# photon-number moments of the error spaces, and four pulse-shape numbers.
# The functions below compute the scalars that are not already stored.

def photon_number_moments(
    prob: np.ndarray,
    n_ref: int,
) -> Dict[str, float]:
    r"""
    Moments of the photon number and of its offset from the reference photon
    number, for one photon-number distribution.

    The ancilla drive is resonant in the photon-number subspace
    :math:`n_{\mathrm{ref}}`, the reference photon number, so the dispersive
    shift detunes the ancilla transition in subspace :math:`m` in proportion
    to the offset :math:`d = m - n_{\mathrm{ref}}`. For a distribution
    :math:`P(m)`, the moments are the mean photon number
    :math:`\langle N\rangle`, the photon-number variance :math:`\Delta N^2`
    and the means of three powers of the offset:

    .. math::
        \langle N\rangle = \sum_m P(m)\,m,\quad
        \Delta N^2 = \sum_m P(m)\,m^2 - \langle N\rangle^2,\quad
        \langle d^k\rangle = \sum_m P(m)\,(m-n_{\mathrm{ref}})^k,
        \quad k = 1, 2, 4 .

    Parameters
    ----------
    prob:
        One-dimensional array; ``prob[m]`` is the probability :math:`P(m)` of
        photon number :math:`m`, e.g. in the maximally mixed state of an error
        space. The function does not normalize it.
    n_ref:
        Reference photon number :math:`n_{\mathrm{ref}}`.

    Returns
    -------
    dict
        Keys ``mean_n``, ``var_n``, ``mean_d``, ``mean_d2``, ``mean_d4``:
        :math:`\langle N\rangle`, :math:`\Delta N^2`, :math:`\langle d\rangle`,
        :math:`\langle d^2\rangle` and :math:`\langle d^4\rangle`.
    """
    # Photon numbers of the entries of prob, and their offsets d = m - n_ref
    prob = np.asarray(prob)
    m = np.arange(len(prob))
    d = m - n_ref

    # Moments of N and of the offset d
    mean_n = float(prob @ m)
    return dict(
        mean_n=mean_n,
        var_n=float(prob @ m**2 - mean_n**2),
        mean_d=float(prob @ d),
        mean_d2=float(prob @ d**2),
        mean_d4=float(prob @ d**4),
    )


def cat_error_space_distribution(
    amp_sq: float,
    parity: int,
    n_max: int | None = None,
) -> np.ndarray:
    r"""
    Photon-number distribution of the maximally mixed state of a cat error
    space.

    An error space of the four-legged cat code is spanned by the cats
    :math:`|C^{\pm}_{\beta}\rangle` and :math:`|C^{\pm}_{i\beta}\rangle` of one
    photon-number parity. Expanding them in Fock states shows that the same
    two-dimensional space is spanned by the two mod-4 components

    .. math::
        |u_j\rangle \propto \sum_{m\equiv j\ (\mathrm{mod}\ 4)}
        \frac{\beta^m}{\sqrt{m!}}\,|m\rangle,
        \qquad j\in\{0,2\}\ (\text{even}),\ \{1,3\}\ (\text{odd}),

    which are orthogonal. The maximally mixed state of the error space is
    :math:`\hat P/2 = (|u_j\rangle\langle u_j| + |u_{j+2}\rangle\langle u_{j+2}|)/2`
    in any orthonormal basis of the same span, so it does not depend on how
    the codewords are orthonormalized (symmetric or Gram--Schmidt). Its
    photon-number distribution is

    .. math::
        P(m) = \frac{1}{2}\,\frac{|\beta|^{2m}/m!}{S_j},
        \qquad S_j = \sum_{k\equiv j\ (\mathrm{mod}\ 4)} \frac{|\beta|^{2k}}{k!},

    for :math:`m\equiv j` (mod 4).

    Parameters
    ----------
    amp_sq:
        :math:`|\beta|^2`, squared amplitude of the cats that span the space.
        After an idle of duration :math:`t` with photon-loss rate
        :math:`\gamma`, :math:`|\beta|^2 = |\alpha|^2 e^{-\gamma t}`.
    parity:
        0 for the even-parity error space, 1 for the odd-parity one.
    n_max:
        Largest Fock number kept in the sums. Default: a cutoff far in the
        Poisson tail, :math:`|\beta|^2 + 12\sqrt{|\beta|^2+1} + 20`.

    Returns
    -------
    np.ndarray
        One-dimensional array of length ``n_max + 1``; entry ``m`` is
        :math:`P(m)`.
    """
    # Fock cutoff: the Poisson weights beyond it are negligible
    if n_max is None:
        n_max = int(amp_sq + 12 * np.sqrt(amp_sq + 1) + 20)
    m = np.arange(n_max + 1)

    # Poisson weights |beta|^{2m}/m! in log form, so that large m does not
    # overflow; the common factor e^{-|beta|^2} cancels in the normalization
    log_w = m * np.log(amp_sq) - gammaln(m + 1) if amp_sq > 0 else np.where(
        m == 0, 0.0, -np.inf
    )
    w = np.exp(log_w - np.max(log_w))

    # Each of the two mod-4 components carries probability 1/2
    residues = (0, 2) if parity == 0 else (1, 3)
    prob = np.zeros_like(w)
    for j in residues:
        in_component = (m % 4 == j)
        prob[in_component] = 0.5 * w[in_component] / np.sum(w[in_component])
    return prob


def cat_error_space_moments(
    amp_sq: float,
    parity: int,
    n_ref: int,
    n_max: int | None = None,
) -> Dict[str, float]:
    r"""
    Photon-number moments of the maximally mixed state of a cat error space.

    The moments are those of ``photon_number_moments`` for the photon-number
    distribution :math:`P(m)` of ``cat_error_space_distribution``.

    Parameters
    ----------
    amp_sq:
        :math:`|\beta|^2`, squared amplitude of the cats that span the space.
        After an idle of duration :math:`t` with photon-loss rate
        :math:`\gamma`, :math:`|\beta|^2 = |\alpha|^2 e^{-\gamma t}`.
    parity:
        0 for the even-parity error space, 1 for the odd-parity one.
    n_ref:
        Reference photon number :math:`n_{\mathrm{ref}}`, from which the
        offset :math:`d = m - n_{\mathrm{ref}}` is counted.
    n_max:
        Largest Fock number kept in the sums. Default: a cutoff far in the
        Poisson tail, :math:`|\beta|^2 + 12\sqrt{|\beta|^2+1} + 20`.

    Returns
    -------
    dict
        Keys ``mean_n``, ``var_n``, ``mean_d``, ``mean_d2``, ``mean_d4``.
    """
    # Moments of N and of the offset d = N - n_ref
    return photon_number_moments(
        cat_error_space_distribution(amp_sq, parity, n_max), n_ref)


def cat_kerr_rotation_infidelity(
    amp_sq: float,
    parity: int,
    kerr_angle: float,
    n_max: int | None = None,
) -> float:
    r"""
    Infidelity of the Kerr rotation that a photon loss at an unrecorded time
    leaves on a cat error space.

    With the self-Kerr Hamiltonian :math:`K\hat a^{\dagger2}\hat a^2`, a
    photon lost at time :math:`\tau` of an idle of duration :math:`t` leaves
    the rotation :math:`e^{-i\vartheta\hat N}`, with
    :math:`\vartheta=2K(\tau-t/2)`, about the mean rotation
    :math:`e^{-iKt\hat N}`. Its process fidelity with the identity on the
    error space that the loss leads to is
    :math:`|\sum_mP(m)e^{-i\vartheta m}|^2`, where :math:`P(m)` is the
    photon-number distribution of the maximally mixed state of that space
    (``cat_error_space_distribution``). For :math:`\tau` uniform on
    :math:`[0,t]`, the angle is uniform on :math:`[-Kt,Kt]` and the average
    infidelity is

    .. math::
        1-\sum_{m,m'}P(m)P(m')\,\frac{\sin[(m-m')Kt]}{(m-m')Kt}
        \simeq \tfrac13(Kt)^2\,\Delta N^2 ,

    where the term :math:`m=m'` is 1 and :math:`\Delta N^2` is the
    photon-number variance of :math:`P`. The second form holds for
    :math:`|K|t\ll1`; for a large cat, :math:`\Delta N^2\to|\beta|^2`. The
    function returns the first form.

    Parameters
    ----------
    amp_sq:
        :math:`|\beta|^2`, squared amplitude of the cats that span the error
        space after the loss.
    parity:
        Photon-number parity of that error space: 0 even, 1 odd.
    kerr_angle:
        :math:`Kt` (rad): self-Kerr coefficient times idle duration.
    n_max:
        Largest Fock number kept in the sums; default of
        ``cat_error_space_distribution``.

    Returns
    -------
    float
        The infidelity averaged over the loss time, between 0 and 1.
    """
    prob = cat_error_space_distribution(amp_sq, parity, n_max)
    m = np.arange(len(prob))
    # np.sinc(x) is sin(pi x)/(pi x), hence the division by pi
    fidelity = prob @ np.sinc(np.subtract.outer(m, m) * kerr_angle / np.pi) @ prob
    return float(1 - fidelity)


def _pulse_rotation_profile(
    theta: float,
    envelope: str = "gaussian",
    n_grid: int = 20001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Time grid on :math:`[0,1]`, Rabi rate :math:`\Omega_{\mathrm R}(t)` and
    completed rotation angle
    :math:`\theta(t)=\int_0^t\Omega_{\mathrm R}\,dt'` of an ancilla pulse of
    rotation angle ``theta`` and unit duration.

    The ``"gaussian"`` envelope is a Gaussian of width
    :math:`\sigma=t_\theta/4`, truncated at :math:`\pm2\sigma` and shifted
    down so that it vanishes at the two ends; ``"square"`` is constant.
    """
    # The shape numbers are independent of the duration, so use t_theta = 1
    t = np.linspace(0.0, 1.0, n_grid)

    # Envelope shape f(t); its scale is fixed below by the rotation angle
    if envelope == "gaussian":
        sigma = 0.25
        shape = np.exp(-(t - 0.5) ** 2 / (2 * sigma**2))
        shape = shape - shape[0]    # the truncated pulse starts and ends at 0
    elif envelope == "square":
        shape = np.ones_like(t)
    else:
        raise ValueError(f"Unknown envelope {envelope!r}.")

    # Scale the envelope so that the pulse area equals theta
    rabi = theta * shape / trapezoid(shape, t)
    angle = cumulative_trapezoid(rabi, t, initial=0.0)
    return t, rabi, angle


def pulse_shape_numbers(
    theta: float,
    envelope: str = "gaussian",
    n_grid: int = 20001,
) -> Tuple[float, float]:
    r"""
    Envelope shape numbers :math:`\mu_\theta` and :math:`\nu_\theta` of an
    ancilla pulse (dimensionless).

    Let :math:`\Omega_{\mathrm R}(t)` be the Rabi rate of a pulse of rotation
    angle :math:`\theta` and duration :math:`t_\theta`, and
    :math:`\theta(t)=\int_0^t\Omega_{\mathrm R}\,dt'` the rotation completed at
    time :math:`t`. A small detuning :math:`\delta(t)` tilts the rotation by
    :math:`\varphi=\int_0^{t_\theta}\delta(t)\sin\theta(t)\,dt`. For the Stark
    shift that the drive induces, :math:`\delta\propto\Omega_{\mathrm R}^2`,
    and for a constant detuning by the dispersive shift, computing the tilt
    angles requires

    .. math::
        \mu_\theta = \frac{t_\theta}{\theta^2}\int_0^{t_\theta}
            \Omega_{\mathrm R}^2\sin\theta(t)\,dt,
        \qquad
        \nu_\theta = \frac{1}{t_\theta}\int_0^{t_\theta}\sin\theta(t)\,dt .

    Both depend only on the envelope shape and :math:`\theta`, not on
    :math:`t_\theta`. For a square pulse,
    :math:`\mu_\theta=\nu_\theta=(1-\cos\theta)/\theta`. The ``"gaussian"``
    envelope is a Gaussian of width :math:`\sigma=t_\theta/4`, truncated at
    :math:`\pm2\sigma` and shifted down so that it vanishes at the two ends,
    the envelope of the pulses of ``cat_real.qubit_gate``.

    Parameters
    ----------
    theta:
        Rotation angle (rad), positive. The numbers of a negative angle
        follow from :math:`\mu_{-\theta}=-\mu_\theta`,
        :math:`\nu_{-\theta}=-\nu_\theta`.
    envelope:
        ``"gaussian"`` or ``"square"``.
    n_grid:
        Number of time points of the trapezoidal quadrature.

    Returns
    -------
    (mu, nu): tuple of float
        :math:`\mu_\theta` and :math:`\nu_\theta`.
    """
    t, rabi, angle = _pulse_rotation_profile(theta, envelope, n_grid)

    # Shape integrals with t_theta = 1
    mu = trapezoid(rabi**2 * np.sin(angle), t) / theta**2
    nu = trapezoid(np.sin(angle), t)
    return float(mu), float(nu)


def pulse_population_averages(
    theta: float,
    envelope: str = "gaussian",
    n_grid: int = 20001,
) -> Tuple[float, float]:
    r"""
    Time averages of the excited-state population and of its square during an
    ancilla pulse.

    An ancilla that starts in :math:`|0\rangle` and is rotated by the resonant
    pulse has the excited-state population
    :math:`P_1(t)=\sin^2[\theta(t)/2]`, with :math:`\theta(t)` the rotation
    completed at time :math:`t`. The averages over the pulse duration
    :math:`t_\theta` are

    .. math::
        \overline{P_1} = \frac{1}{t_\theta}\int_0^{t_\theta}P_1(t)\,dt,
        \qquad
        \overline{P_1^2} = \frac{1}{t_\theta}\int_0^{t_\theta}P_1(t)^2\,dt .

    Both depend only on the envelope shape and the rotation angle
    :math:`\theta`. For a square pulse,
    :math:`\overline{P_1}=\tfrac12(1-\sin\theta/\theta)` and
    :math:`\overline{P_1^2}=\tfrac38-\sin\theta/(2\theta)+\sin2\theta/(16\theta)`.
    For a pulse symmetric in time, :math:`\overline{P_1}=(1-\nu_{\pi/2})/2` and
    :math:`\overline{P_1^2}=(3-4\nu_{\pi/2})/8` at :math:`\theta=\pi/2`, and
    :math:`\overline{P_1}=\tfrac12` at :math:`\theta=\pi`, with the envelope
    shape number :math:`\nu_\theta` of ``pulse_shape_numbers``.

    An ancilla that starts in :math:`|1\rangle`, or that the pulse rotates
    from the equator to :math:`|1\rangle`, has the populations of the two
    levels exchanged: :math:`\overline{P_1}\to1-\overline{P_1}` and
    :math:`\overline{P_1^2}\to1-2\overline{P_1}+\overline{P_1^2}`.

    Parameters
    ----------
    theta:
        Rotation angle (rad), positive.
    envelope:
        ``"gaussian"`` or ``"square"``, as in ``pulse_shape_numbers``.
    n_grid:
        Number of time points of the trapezoidal quadrature.

    Returns
    -------
    (mean_population, mean_population_squared): tuple of float
        :math:`\overline{P_1}` and :math:`\overline{P_1^2}`.
    """
    t, _, angle = _pulse_rotation_profile(theta, envelope, n_grid)

    # Excited-state population of the resonant rotation from |0>
    population = np.sin(angle / 2) ** 2
    return float(trapezoid(population, t)), float(trapezoid(population**2, t))


def _param_at(
    sweep: ParameterSweep,
    idx: Tuple[int, ...],
    name: str,
    kwargs: Dict[str, Any],
) -> Any:
    r"""
    Value of the parameter ``name`` at the grid point ``idx``: the sweep
    parameter of this name if there is one (a ``FlexibleSweep`` registers its
    fixed parameters as sweep axes of length 1), otherwise the entry of
    ``kwargs``.
    """
    # Swept (or FlexibleSweep-fixed) parameters take precedence
    meshgrids = sweep.parameters.meshgrids_by_paramname()
    if name in meshgrids:
        return meshgrids[name][idx]
    # Otherwise fall back to a value supplied by the caller
    try:
        return kwargs[name]
    except KeyError:
        raise KeyError(
            f"Parameter {name!r} is neither a sweep parameter nor a keyword "
            "argument of batched_sweep_cat_budget_ingredients."
        )


def _sweep_cat_error_space_moments(
    sweep: ParameterSweep,
    idx: Tuple[int, ...],
    **kwargs,
) -> np.ndarray:
    r"""
    Photon-number moments (``cat_error_space_moments``) of three maximally
    mixed states at one grid point, as an array of shape ``(3, 5)``.

    The rows are the logical subspace at encoding (cats of amplitude
    :math:`\alpha`, ``disp``), the even error space after the idle and the
    odd error space after the idle; the columns are ``mean_n``, ``var_n``,
    ``mean_d``, ``mean_d2``, ``mean_d4``. After an idle of duration
    :math:`t_{\mathrm w}` (``T_W``) the cats have the amplitude
    :math:`\beta=\alpha e^{-\gamma_{\hat a}t_{\mathrm w}/2}`, and the reference
    photon number is :math:`n_{\mathrm{ref}}=\mathrm{round}(|\alpha|^2)`.
    """
    # Cat size and the rotating-frame reference photon number
    amp_sq = np.abs(_param_at(sweep, idx, "disp", kwargs)) ** 2
    n_ref = int(np.round(amp_sq))

    # Amplitude after the idle: shrinkage by the no-jump evolution
    idle_time = _param_at(sweep, idx, "T_W", kwargs)
    amp_sq_after = amp_sq * np.exp(-sweep["jump_a"][idx] * idle_time)

    # One row per space, in the documented order
    keys = ["mean_n", "var_n", "mean_d", "mean_d2", "mean_d4"]
    rows = [
        cat_error_space_moments(amp_sq, 0, n_ref),
        cat_error_space_moments(amp_sq_after, 0, n_ref),
        cat_error_space_moments(amp_sq_after, 1, n_ref),
    ]
    return np.array([[row[k] for k in keys] for row in rows])


def _sweep_cat_kerr_rotation_infidelity(
    sweep: ParameterSweep,
    idx: Tuple[int, ...],
    **kwargs,
) -> float:
    r"""
    Infidelity of the Kerr rotation after a photon loss at an unrecorded time
    of the idle (``cat_kerr_rotation_infidelity``) at one grid point, on the
    odd error space after the idle, with the angle
    :math:`Kt_{\mathrm w}` (``K_s`` times ``T_W``).
    """
    # Amplitude after the idle: shrinkage by the no-jump evolution
    amp_sq = np.abs(_param_at(sweep, idx, "disp", kwargs)) ** 2
    idle_time = _param_at(sweep, idx, "T_W", kwargs)
    amp_sq_after = amp_sq * np.exp(-sweep["jump_a"][idx] * idle_time)

    return cat_kerr_rotation_infidelity(
        amp_sq_after, 1, sweep["K_s"][idx] * idle_time)


# Ancilla-dependent inputs of the analytical budget. The default describes a
# DRAG-driven transmon; passing another dict to
# ``batched_sweep_cat_budget_ingredients`` changes the physical system.
DEFAULT_CAT_BUDGET_PARA = dict(
    # ancilla operator through which the drive acts (attribute of the qubit)
    drive_operator="n_operator",
    # "plain": Gaussian pulse; "drag": Gaussian pulse with a DRAG quadrature
    # and no chirp
    pulse_scheme="drag",
    # number of lowest ancilla eigenstates in the Stark-shift sum, at least 3
    stark_level_count=3,
    # True: the parity mapping applies ancilla frame changes (virtual Z
    # gates) that remove the Stark-shift phase of each pi/2 pulse and the
    # cat-averaged chi_1' phase of the parity-mapping idling; must match
    # FullCatTreeBuilder.use_frame_update
    frame_update=False,
)
_PULSE_SCHEMES = ("plain", "drag")


def _check_cat_budget_para(budget_para: Dict[str, Any] | None) -> Dict[str, Any]:
    r"""
    ``budget_para`` merged over ``DEFAULT_CAT_BUDGET_PARA``; an unknown key or
    an invalid value raises.
    """
    # A misspelled key would silently keep the default, so reject it here
    unknown = set(budget_para or {}) - set(DEFAULT_CAT_BUDGET_PARA)
    if unknown:
        raise KeyError(
            f"Unknown budget parameters {sorted(unknown)}; "
            f"the accepted keys are {sorted(DEFAULT_CAT_BUDGET_PARA)}."
        )
    para = DEFAULT_CAT_BUDGET_PARA | (budget_para or {})
    if para["pulse_scheme"] not in _PULSE_SCHEMES:
        raise ValueError(
            f"pulse_scheme must be one of {_PULSE_SCHEMES}, "
            f"got {para['pulse_scheme']!r}."
        )
    if para["stark_level_count"] < 3:
        raise ValueError("stark_level_count must be at least 3.")
    return para


def cat_stark_shift_coefficient(
    energies: np.ndarray,
    drive_elements: np.ndarray,
    pulse_scheme: str,
    anharmonicity: float | None = None,
) -> float:
    r"""
    Stark-shift coefficient :math:`s_{\mathrm S}` of the ancilla transition
    under its own drive (ns).

    A drive of Rabi rate :math:`\Omega_{\mathrm R}` on the 0-1 transition,
    applied through the operator :math:`\hat O`, shifts that transition by
    :math:`\delta_{\mathrm S}=s_{\mathrm S}\Omega_{\mathrm R}^2/4`.
    Second-order perturbation theory in the drive, keeping the co-rotating
    term of every off-resonant transition, gives for a plain pulse

    .. math::
        s_{\mathrm S} = -\sum_{k\ge2}\frac{1}{|O_{01}|^2}\left[
        \frac{|O_{1k}|^2}{\omega_{1k}-\omega_{01}}
        - \frac{|O_{0k}|^2}{\omega_{0k}-\omega_{01}}\right],

    with :math:`O_{jk}=\langle j|\hat O|k\rangle` and
    :math:`\omega_{jk}=E_k-E_j`. For a transmon, whose drive operator couples
    only neighboring levels, only the :math:`k=2` term of level 1 survives and
    :math:`s_{\mathrm S}=-\lambda^2/\eta`, with :math:`\lambda=|O_{12}/O_{01}|`
    and the anharmonicity :math:`\eta=\omega_{12}-\omega_{01}`. A DRAG
    quadrature without chirp adds :math:`4/\eta` to first order in
    :math:`\Omega_{\mathrm R}/\eta`. Counter-rotating terms are neglected,
    which is justified when the pulse of duration :math:`t_\theta` and
    rotation angle :math:`\theta` lasts many carrier periods,
    :math:`\omega_{01}t_\theta\gg\theta^2`.

    Parameters
    ----------
    energies:
        Ascending ancilla eigenenergies :math:`E_k` (rad/ns), at least three.
    drive_elements:
        Moduli :math:`|O_{jk}|` of the drive-operator matrix elements, a
        square array of the size of ``energies``.
    pulse_scheme:
        ``"plain"`` or ``"drag"``.
    anharmonicity:
        :math:`\eta` (rad/ns) used in the 1-2 term and in the DRAG term, e.g.
        the dressed value. None uses the value :math:`E_2-2E_1+E_0` of
        ``energies``.

    Returns
    -------
    float
        :math:`s_{\mathrm S}` in ns.
    """
    # Detunings of the 1->k and 0->k transitions from the drive frequency
    upper_levels = np.arange(2, len(energies))
    drive_freq = energies[1] - energies[0]
    if anharmonicity is None:
        anharmonicity = energies[2] - 2 * energies[1] + energies[0]
    detuning_from_1 = energies[upper_levels] - energies[1] - drive_freq
    detuning_from_1[0] = anharmonicity
    detuning_from_0 = energies[upper_levels] - energies[0] - drive_freq

    # Shift of level 1 minus shift of level 0, in units of Omega_R^2 / 4
    coefficient = -np.sum(
        drive_elements[1, upper_levels] ** 2 / detuning_from_1
        - drive_elements[0, upper_levels] ** 2 / detuning_from_0
    ) / drive_elements[0, 1] ** 2
    if pulse_scheme == "drag":
        coefficient += 4 / anharmonicity
    return coefficient


def _sweep_anc_drive_constants(
    sweep: ParameterSweep,
    idx: Tuple[int, ...],
    res_mode_idx: int = 0,
    qubit_mode_idx: int = 1,
    budget_para: Dict[str, Any] | None = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Ancilla anharmonicity :math:`\eta` (rad/ns), drive-matrix-element ratio
    :math:`\lambda=|O_{12}/O_{01}|` and Stark-shift coefficient
    :math:`s_{\mathrm S}` (ns, ``cat_stark_shift_coefficient``) at one grid
    point, as the array ``[eta, lam, stark_coeff]``.

    :math:`\eta=\omega_{12}-\omega_{01}` and :math:`\lambda` are taken from
    the dressed states of the photon-number subspace
    :math:`n_{\mathrm{ref}}=\mathrm{round}(|\alpha|^2)`, where the pulses are
    resonant (:math:`\alpha` is the cat amplitude ``disp``). All three are NaN
    if the ancilla keeps fewer than three levels.
    """
    # A two-level ancilla has no |2> level to define eta or lambda
    budget_para = _check_cat_budget_para(budget_para)
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    if qubit.truncated_dim < 3:
        return np.full(3, np.nan)
    n_ref = int(np.round(np.abs(_param_at(sweep, idx, "disp", kwargs)) ** 2))

    # Dressed 0-1 and 1-2 transition frequencies in subspace n_ref (GHz)
    state_kwargs = dict(res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx)
    omega_01 = _calc_freq_diff(
        sweep, idx, res_state_1=n_ref, qubit_state_1=0,
        res_state_2=n_ref, qubit_state_2=1, **state_kwargs,
    )
    omega_12 = _calc_freq_diff(
        sweep, idx, res_state_1=n_ref, qubit_state_1=1,
        res_state_2=n_ref, qubit_state_2=2, **state_kwargs,
    )
    eta = (omega_12 - omega_01) * PI2

    # Dressed energies and drive-operator matrix elements of the lowest
    # ``level_count`` ancilla levels in subspace n_ref, the spectrum that the
    # pulses drive. If the Hilbert space keeps fewer ancilla levels than the
    # Stark-shift sum needs, the bare ancilla spectrum replaces the dressed one.
    level_count = budget_para["stark_level_count"]
    if qubit.truncated_dim >= level_count:
        hilbertspace = sweep.hilbertspace
        evals, evecs = two_mode_dressed_esys(
            hilbertspace=hilbertspace,
            res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
            state_label=(0, 0),
            res_truncated_dim=n_ref + 1, qubit_truncated_dim=level_count,
            dressed_indices=sweep["dressed_indices"][idx],
            eigensys=(sweep["evals"][idx], sweep["evecs"][idx]),
        )
        drive_op = scq.identity_wrap(
            getattr(qubit, budget_para["drive_operator"])(), qubit,
            hilbertspace.subsystem_list,
        )
        energies = np.real(evals[n_ref, :level_count]) * PI2
        elements = np.abs(qt.Qobj(
            oprt_in_basis(drive_op, evecs[n_ref, :level_count])
        ).full())
    else:
        energies = qubit.eigenvals(level_count) * PI2
        elements = np.abs(qubit.matrixelement_table(
            budget_para["drive_operator"], evals_count=level_count,
        ))
    lam = elements[1, 2] / elements[0, 1]
    stark_coeff = cat_stark_shift_coefficient(
        energies, elements, budget_para["pulse_scheme"], anharmonicity=eta,
    )
    return np.array([eta, lam, stark_coeff])


def batched_sweep_cat_budget_ingredients(
    sweep: ParameterSweep,
    res_mode_idx: int = 0,
    qubit_mode_idx: int = 1,
    budget_para: Dict[str, Any] | None = None,
    **kwargs,
) -> None:
    r"""
    Store the scalar ingredients of the analytical one-cycle error budget.

    The sweep needs the keys of ``batched_sweep_general``,
    ``batched_sweep_jump_rates`` and ``batched_sweep_pulse``. The budget
    ``error_budgeting.cat_code_budget`` reads the stored keys together with
    the rates ``jump_a``, ``jump_adag``, ``jump_adag_a``, ``jump_ij`` and
    the protocol parameters ``disp``, ``T_W``, ``tau_m``, ``tau_FD``,
    ``tau_p_eff``, ``M_ge``, ``M_eg``, ``K_s`` and ``chi_prime_sa``.

    Stored keys (rad/ns and ns unless stated):

    - ``n_ref``: reference photon number :math:`\mathrm{round}(|\alpha|^2)`.
    - ``chi_1``: dispersive shift per photon at the mean photon number
      :math:`\bar n=|\alpha|^2` of the cat of amplitude :math:`\alpha`,
      :math:`\chi_1=\chi_{sa}+(2\bar n-1)\chi'_{sa}`. The parity mapping is
      designed for this linearized shift.
    - ``t_pm``: compensated duration of the parity-mapping idling,
      :math:`t_{\mathrm{pm}}=(\pi-2|\varphi_g|)/|\chi_1|`. Here
      :math:`\varphi_g=\chi_1t_g\nu_g` is the controlled phase accumulated
      during one :math:`\pi/2` pulse of duration :math:`t_g` (``tau_p_eff``),
      with the envelope shape number :math:`\nu_g` of the pulse
      (``gate_nu_half``).
    - ``t_cycle``: duration of one cycle,
      :math:`t_{\mathrm w}+4t_g+t_{\mathrm{pm}}+\tau_m+\tau_{\mathrm{FD}}`
      (two :math:`\pi/2` pulses plus the :math:`2t_g` reset slot; the feedback
      latency ``tau_FD`` is zero if it is neither a sweep parameter nor a
      keyword). The infidelity of the cycle divided by ``t_cycle`` is the
      failure rate per unit time.
    - ``anc_eta``, ``anc_lambda``, ``anc_stark_coeff``: ancilla
      anharmonicity :math:`\eta=\omega_{12}-\omega_{01}` (rad/ns) and ratio
      :math:`\lambda=|O_{12}/O_{01}|` of the drive matrix elements, both in
      the photon-number subspace ``n_ref``, and Stark-shift coefficient
      :math:`s_{\mathrm S}` (ns, ``cat_stark_shift_coefficient``). All three
      are NaN if the ancilla keeps fewer than three levels. The pulse terms
      of the budget read only ``anc_stark_coeff``.
    - ``cat_logical_mean_n``, ``cat_logical_var_n``: :math:`\langle N\rangle`
      and :math:`\Delta N^2` of the logical subspace at encoding.
    - ``cat_mean_n``, ``cat_var_n``, ``cat_mean_d``, ``cat_mean_d2``,
      ``cat_mean_d4``: moments of the even (index 0) and odd (index 1)
      error spaces after the idle; trailing axis of length 2.
    - ``cat_kerr_rotation_infidelity``: infidelity of the Kerr rotation that
      one photon loss at an unrecorded time of the idle leaves on the odd
      error space after the idle (``cat_kerr_rotation_infidelity`` at the
      angle :math:`Kt_{\mathrm w}`, ``K_s`` times ``T_W``).
    - ``gate_mu_half``, ``gate_nu_half``, ``gate_mu_pi``, ``gate_nu_pi``:
      envelope shape numbers of the :math:`\pi/2` pulses and of the
      :math:`\pi` pulse of the reset
      (``pulse_shape_numbers``, Gaussian envelope).
    - ``gate_pop_half``, ``gate_pop_sq_half``, ``gate_pop_pi``,
      ``gate_pop_sq_pi``: time averages of the excited-state population and
      of its square during the :math:`\pi/2` pulses and the :math:`\pi` pulse
      of the reset, for an ancilla that starts in :math:`|0\rangle`
      (``pulse_population_averages``, Gaussian envelope).
    - ``gate_frame_phase``: Stark-shift angle of a :math:`\pi/2` pulse,
      :math:`\varphi_{\mathrm S}=s_{\mathrm S}\mu_{\pi/2}(\pi/2)^2/(4t_g)`. In the
      two-level ancilla, the pulse leaves the ancilla phase gate
      :math:`e^{-i\varphi_{\mathrm S}|1\rangle\langle1|}` on each side of the
      rotation; frame changes by :math:`-\varphi_{\mathrm S}` remove it.
    - ``pm_frame_phase``: ancilla phase of the qubit-state-dependent
      resonator non-linearity :math:`\chi_1'` (``chi_prime_sa``) during the
      parity-mapping idling, averaged over the even error space after the
      idle, :math:`\chi_1't_{\mathrm{pm}}\langle d^2\rangle_+`, where
      :math:`d` is the offset of the photon number from ``n_ref``;
      a frame change by its negative removes it.
    - ``frame_update``: 1 if the protocol applies these frame changes
      (``budget_para["frame_update"]``), else 0.

    Parameters
    ----------
    sweep:
        The parameter sweep.
    res_mode_idx, qubit_mode_idx:
        Subsystem indices of the resonator and the ancilla.
    budget_para:
        Ancilla-dependent inputs, merged over ``DEFAULT_CAT_BUDGET_PARA``
        (a DRAG-driven transmon): ``drive_operator``, the ancilla operator
        through which the drive acts; ``pulse_scheme``, ``"plain"`` or
        ``"drag"``; ``stark_level_count``, the number of ancilla levels
        in the Stark-shift sum; and ``frame_update``, whether the parity
        mapping applies the ancilla frame changes. The drive operator and
        the pulse scheme must match the simulated pulses.
    **kwargs:
        Fallback values of ``disp``, ``T_W``, ``tau_m`` and ``tau_FD`` if
        they are not sweep parameters.
    """
    # Validate the ancilla dict before any sweep runs
    budget_para = _check_cat_budget_para(budget_para)

    # Parameter meshgrids of the full grid (fixed FlexibleSweep parameters
    # are length-1 axes); kwargs fill in names that are not parameters
    params = kwargs | sweep.parameters.meshgrids_by_paramname()
    grid_ones = np.ones(sweep.parameters.counts)
    wrap = lambda arr: NamedSlotsNdarray(np.asarray(arr), sweep.param_info)

    # Pulse-shape integrals do not depend on the grid point; broadcast them.
    # Computed here (ahead of phi_g below) because phi_g needs gate_nu_half.
    mu_half, nu_half = pulse_shape_numbers(np.pi / 2)
    mu_pi, nu_pi = pulse_shape_numbers(np.pi)
    pop_half, pop_sq_half = pulse_population_averages(np.pi / 2)
    pop_pi, pop_sq_pi = pulse_population_averages(np.pi)
    sweep.store_data(
        gate_mu_half=wrap(mu_half * grid_ones),
        gate_nu_half=wrap(nu_half * grid_ones),
        gate_mu_pi=wrap(mu_pi * grid_ones),
        gate_nu_pi=wrap(nu_pi * grid_ones),
        gate_pop_half=wrap(pop_half * grid_ones),
        gate_pop_sq_half=wrap(pop_sq_half * grid_ones),
        gate_pop_pi=wrap(pop_pi * grid_ones),
        gate_pop_sq_pi=wrap(pop_sq_pi * grid_ones),
    )

    # Linearized dispersive slope and the compensated parity-mapping time,
    # with the same definitions as FullCatTreeBuilder in QEC_graph. phi_g
    # uses the pi/2 pulse's own shape number nu_half in place of the
    # square-pulse value 2/pi, so the ideal pi/2-pulse gate retains the
    # pulse's own first-order dispersive-shift tilt and phase.
    n_bar = np.abs(params["disp"]) ** 2 * grid_ones
    chi_1 = sweep["chi_sa"] + (2 * n_bar - 1) * sweep["chi_prime_sa"]
    phi_g = chi_1 * sweep["tau_p_eff"] * nu_half
    t_pm = (np.pi - 2 * np.abs(phi_g)) / np.abs(chi_1)
    sweep.store_data(
        n_ref=wrap(np.round(n_bar)),
        chi_1=wrap(chi_1),
        t_pm=wrap(t_pm),
    )

    # Duration of one cycle: idle, two pi/2 pulses, parity mapping,
    # measurement, feedback latency (absent from parameter sets that
    # predate it, hence the default 0), and the reset slot (2 t_g),
    # reserved even when the outcome is even. cat_code_budget / t_cycle
    # gives a failure rate per unit time.
    if "tau_FD" not in params:
        # a forgotten key silently shortens the cycle, so the default is announced
        warnings.warn(
            "Neither the sweep parameters nor the keywords hold 'tau_FD': "
            "t_cycle is computed without feedback latency. Add tau_FD (ns), "
            "or 0 explicitly for a cycle without latency.",
            UserWarning, stacklevel=2,
        )
    sweep.store_data(t_cycle=wrap(
        params["T_W"] * grid_ones + 4 * sweep["tau_p_eff"] + t_pm
        + params["tau_m"] * grid_ones
        + params.get("tau_FD", 0.0) * grid_ones
    ))

    # Ancilla constants: one kernel returns all three, then unpack and delete
    sweep.add_sweep(
        _sweep_anc_drive_constants, "anc_drive_constants_raw",
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        budget_para=budget_para, **kwargs,
    )
    raw = np.asarray(sweep["anc_drive_constants_raw"], dtype=float)
    sweep.store_data(
        anc_eta=wrap(raw[..., 0]), anc_lambda=wrap(raw[..., 1]),
        anc_stark_coeff=wrap(raw[..., 2]),
    )
    del sweep._data["anc_drive_constants_raw"]

    # Photon-number moments: rows = (logical, even, odd), columns = moments
    sweep.add_sweep(_sweep_cat_error_space_moments, "cat_moments_raw", **kwargs)
    raw = np.asarray(sweep["cat_moments_raw"], dtype=float)
    sweep.store_data(
        cat_logical_mean_n=wrap(raw[..., 0, 0]),
        cat_logical_var_n=wrap(raw[..., 0, 1]),
        cat_mean_n=wrap(raw[..., 1:, 0]),
        cat_var_n=wrap(raw[..., 1:, 1]),
        cat_mean_d=wrap(raw[..., 1:, 2]),
        cat_mean_d2=wrap(raw[..., 1:, 3]),
        cat_mean_d4=wrap(raw[..., 1:, 4]),
    )
    del sweep._data["cat_moments_raw"]

    # Kerr rotation after one photon loss of the idle, averaged over the
    # loss time: the idle term of the budget multiplies it by the
    # probability of exactly one loss
    sweep.add_sweep(
        _sweep_cat_kerr_rotation_infidelity, "cat_kerr_rotation_infidelity",
        **kwargs)

    # Frame changes of the parity mapping remove two ancilla phases: the
    # phase that the Stark shift of one pi/2 pulse leaves, and the
    # even-parity average of the chi_1' phase of the parity-mapping idling.
    # The first equals the Stark-shift angle phi_S of the budget, so the
    # function of the budget computes it. The residual dispersive-shift
    # angle that the function also returns is not an ancilla phase and is
    # discarded.
    stark_shift_phase, _ = cat_pulse_tilt_angles(
        np.pi / 2, sweep["tau_p_eff"], chi_1, sweep["anc_stark_coeff"],
        mu_half, nu_half, retained_shape_nu=nu_half,
    )
    sweep.store_data(
        gate_frame_phase=wrap(stark_shift_phase),
        pm_frame_phase=wrap(
            sweep["chi_prime_sa"] * t_pm * np.asarray(sweep["cat_mean_d2"])[..., 0]
        ),
        frame_update=wrap(float(budget_para["frame_update"]) * grid_ones),
    )
