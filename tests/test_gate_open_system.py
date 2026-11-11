r"""Open-system ancilla gate of ``qec_aeb.cat_real.qubit_gate``.

If the Hamiltonian and the collapse operators of a gate do not change the
photon number, the master equation of every component of the density matrix
between two photon-number subspaces is integrated separately, and the maps of
the components are composed into the map of the whole system. These tests
check the selection of the ancilla-only jump operators, the composition
against the Liouvillian of the whole system, and the gate on a small
resonator-transmon device.
"""

import numpy as np
import pytest
import qutip as qt
from scipy.linalg import expm

import qec_aeb as bsq
from tests.sweep_view_helpers import requires_store_data
from qec_aeb import cat_real, cat_recipe
from chencrafts.cqed.qt_helper import (
    component_liouvillian, qobj_submatrix, superop_evolve,
    superop_from_component_maps, trans_by_kets,
)


def _random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    r"""A random Hermitian matrix.

    Parameters
    ----------
    dim : int
        Dimension.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
    """
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


def _block_diagonal(blocks, sim_dim):
    r"""Operator of the whole system that acts as ``blocks[m]`` in subspace m.

    Parameters
    ----------
    blocks : list of np.ndarray
        One ``sim_dim`` x ``sim_dim`` matrix per subspace.
    sim_dim : int
        Ancilla levels per subspace.

    Returns
    -------
    qt.Qobj
    """
    count = len(blocks)
    return qt.Qobj(np.block([
        [blocks[m] if m == n else np.zeros((sim_dim, sim_dim)) for n in range(count)]
        for m in range(count)
    ]))


def test_dressed_jump_ops_family_selection():
    """A: the ``jump_ij`` family is the subset of the dressed jumps that act on the ancilla alone."""
    res_dim, qubit_dim = 4, 3
    rates = {
        "jump_a": 0.1, "jump_adag": 0.02, "jump_adag_a": 0.0,
        "jump_ij": np.array([[0, 0.3, 0], [0.01, 0.4, 0], [0, 0.05, 0]]),
        "jump_a_ij": np.full((qubit_dim, qubit_dim), 0.001),
        "jump_adag_ij": np.full((qubit_dim, qubit_dim), 0.001),
    }
    everything = cat_recipe.dressed_jump_ops(
        rates, 0, res_dim, qubit_dim, qubit_dim, higher_order_decay=False)
    ancilla = cat_recipe.dressed_jump_ops(
        rates, 0, res_dim, qubit_dim, qubit_dim, higher_order_decay=False,
        op_families=["jump_ij"])
    # at second order, the ancilla-only jumps are |0><1|, |1><0|, |1><1|, and |2><1|
    assert list(ancilla) == ["jump_01", "jump_10", "jump_11", "jump_21"]
    for key, op in ancilla.items():
        assert np.allclose(op.full(), everything[key].full())
        # the identity on the resonator: no matrix element between photon numbers
        blocks = op.full().reshape(res_dim, qubit_dim, res_dim, qubit_dim)
        assert all(np.allclose(blocks[n, :, n, :], blocks[0, :, 0, :]) for n in range(res_dim))
        assert np.allclose(blocks[0, :, 1, :], 0)
    # the full list also contains the jumps of the resonator
    assert {"jump_a", "jump_adag", "jump_a_11"} <= set(everything)


class _FixedPointSweep:
    r"""Sweep with a single parameter point, with the entries that ``cat_recipe`` reads.

    Parameters
    ----------
    qubit_first : bool
        Whether the ancilla precedes the resonator in the ``HilbertSpace``.
    """

    class _Sweep:
        _current_param_indices = (0,)

        def all_params_fixed(self, idx):
            return True

    def __init__(self, qubit_first: bool):
        import scqubits as scq

        res = scq.Oscillator(E_osc=8.3, truncated_dim=6, id_str="res", l_osc=1)
        qubit = scq.Transmon(EJ=20.0, EC=0.27, ng=0.3, ncut=15, truncated_dim=3, id_str="qubit")
        self.hilbertspace = scq.HilbertSpace([qubit, res] if qubit_first else [res, qubit])
        self.hilbertspace.add_interaction(
            g=0.12, op1=res.n_operator, op2=qubit.n_operator, add_hc=False)
        self.hilbertspace.generate_lookup()
        self.sweep = self._Sweep()
        self._data = {
            key: self.hilbertspace[key][0] for key in ("dressed_indices", "evals", "evecs")
        } | {
            "jump_a": 0.1, "jump_adag": 0.02, "jump_adag_a": 0.03,
            "jump_ij": np.array([[0, 0.3, 0], [0.01, 0.4, 0], [0, 0.05, 0]]),
            "jump_a_ij": np.full((3, 3), 0.001), "jump_adag_ij": np.full((3, 3), 0.002),
        }

    def __getitem__(self, key):
        return self._data[key]


def test_master_equation_ingredients_with_the_ancilla_first():
    """A, C: with the ancilla first in the tensor product, the rotating frame makes the resonator at zero ancilla excitation and the ancilla at n_ref photons static, and the operators equal those of the resonator-first system with the tensor factors exchanged."""
    res_dim, qubit_dim, n_ref = 4, 3, 2
    arguments = dict(res_dim=res_dim, qubit_dim=qubit_dim, res_me_dim=res_dim,
                     qubit_me_dim=qubit_dim, in_rot_frame=True, res_n_ref=n_ref)
    ham_qr, jumps_qr, (evals_qr, _), frame_qr = cat_recipe.cavity_ancilla_me_ingredients(
        _FixedPointSweep(qubit_first=True), res_mode_idx=1, qubit_mode_idx=0, **arguments)
    assert ham_qr.dims == [[qubit_dim, res_dim]] * 2

    # energies in the rotating frame, indexed by (ancilla level, photon number)
    energy = np.real(np.diag(ham_qr.full())).reshape(qubit_dim, res_dim)
    lab = evals_qr.reshape(qubit_dim, res_dim)
    assert energy[0, 1] == pytest.approx(energy[0, 0], abs=1e-12)
    assert energy[1, n_ref] == pytest.approx(energy[0, n_ref], abs=1e-12)
    # the frame removes the resonator frequency at ancilla level 0 per photon and
    # the ancilla frequency at n_ref photons per ancilla level
    res_freq = lab[0, 1] - lab[0, 0]
    qubit_freq = lab[1, n_ref] - lab[0, n_ref]
    q, n = np.meshgrid(np.arange(qubit_dim), np.arange(res_dim), indexing="ij")
    assert np.allclose(energy, lab - lab[0, 0] - res_freq * n - qubit_freq * q, atol=1e-10)

    # the same system with the resonator first
    ham_rq, jumps_rq, _, frame_rq = cat_recipe.cavity_ancilla_me_ingredients(
        _FixedPointSweep(qubit_first=False), res_mode_idx=0, qubit_mode_idx=1, **arguments)

    def exchanged(op):
        # matrix on (ancilla x resonator) of an operator given on (resonator x ancilla)
        tensor = op.full().reshape(res_dim, qubit_dim, res_dim, qubit_dim)
        return tensor.transpose(1, 0, 3, 2).reshape((res_dim * qubit_dim,) * 2)

    assert np.allclose(ham_qr.full(), exchanged(ham_rq), atol=1e-10)
    assert np.allclose(frame_qr.full(), exchanged(frame_rq), atol=1e-10)
    assert len(jumps_qr) == len(jumps_rq)
    for jump_qr, jump_rq in zip(jumps_qr, jumps_rq):
        assert np.allclose(jump_qr.full(), exchanged(jump_rq), atol=1e-12)


def test_qubit_gate_rejects_fewer_than_three_ancilla_levels():
    """D: the pulse needs the third ancilla level; two levels raise a ValueError that says so."""
    with pytest.raises(ValueError, match="qubit_truncated_dim must be at least 3"):
        cat_real.qubit_gate(None, 0, 1, res_truncated_dim=4, gate_params={"disp": 1.0})
    with pytest.raises(ValueError, match="qubit_truncated_dim must be at least 3"):
        cat_real.qubit_gate(None, 0, 1, qubit_truncated_dim=2, simulated_qubit_dim=4)


@pytest.mark.parametrize("sim_dim", [3, 4])
def test_composed_component_maps_equal_full_liouvillian(sim_dim):
    """A: the maps of the components compose into the map of the whole system, coherences included.

    The Hamiltonian differs between subspaces, and the collapse operators are
    block diagonal. With four simulated levels and three kept, the composition
    of the truncated component maps equals the map of the whole system
    restricted to three levels.
    """
    rng = np.random.default_rng(7)
    res_dim, keep_dim, time, amplitude = 4, 3, 1.3, 0.6
    hams = [_random_hermitian(sim_dim, rng) for _ in range(res_dim)]
    # one drive operator per subspace, applied with a constant amplitude
    drives = [_random_hermitian(sim_dim, rng) for _ in range(res_dim)]
    base = (rng.normal(size=(sim_dim, sim_dim)) + 1j * rng.normal(size=(sim_dim, sim_dim))) * 0.2
    jump_ops = [base, 0.1 * _random_hermitian(sim_dim, rng)]

    component_maps = {}
    for m in range(res_dim):
        for mp in range(m, res_dim):
            static_part, drive_part = component_liouvillian(
                qt.Qobj(hams[m]), qt.Qobj(drives[m]), [qt.Qobj(j) for j in jump_ops],
                qt.Qobj(hams[mp]), qt.Qobj(drives[mp]), [qt.Qobj(j) for j in jump_ops],
            )
            full = ((static_part + amplitude * drive_part) * time).expm().full()
            # keep the first levels of the initial and the final matrix
            kept = full.reshape((sim_dim,) * 4)[(slice(0, keep_dim),) * 4]
            component_maps[(m, mp)] = kept.reshape(keep_dim ** 2, keep_dim ** 2)
    composed = superop_from_component_maps(component_maps)
    assert composed.type == "super"
    assert composed.dims == [[[res_dim, keep_dim]] * 2] * 2

    # the map of the whole system, from its block-diagonal Hamiltonian and jumps
    full_ham = _block_diagonal([h + amplitude * d for h, d in zip(hams, drives)], sim_dim)
    full_jumps = [_block_diagonal([op] * res_dim, sim_dim) for op in jump_ops]
    full_map = cat_real.idling_propagator(full_ham, full_jumps, time)

    # compare the action on random operators of the kept levels
    kept_levels = [sim_dim * m + a for m in range(res_dim) for a in range(keep_dim)]
    for _ in range(3):
        kept_rho = qt.Qobj(rng.normal(size=(res_dim * keep_dim,) * 2)
                           + 1j * rng.normal(size=(res_dim * keep_dim,) * 2),
                           dims=[[res_dim, keep_dim]] * 2)
        # the same operator on the whole system, zero in the other levels
        embedded = np.zeros((res_dim * sim_dim,) * 2, dtype=complex)
        embedded[np.ix_(kept_levels, kept_levels)] = kept_rho.full()
        expected = qobj_submatrix(superop_evolve(full_map, qt.Qobj(embedded)), kept_levels)
        result = superop_evolve(composed, kept_rho)
        assert np.allclose(result.full(), expected.full(), atol=1e-10)


def test_component_maps_of_unitaries_compose_to_the_unitary_map():
    """A: without collapse operators, the composition is the superoperator of the direct-sum unitary."""
    rng = np.random.default_rng(3)
    res_dim, sub_dim = 3, 3
    unitaries = [expm(-1j * _random_hermitian(sub_dim, rng)) for _ in range(res_dim)]
    component_maps = {
        (m, mp): np.kron(unitaries[mp].conj(), unitaries[m])
        for m in range(res_dim) for mp in range(m, res_dim)
    }
    composed = superop_from_component_maps(component_maps)
    direct = qt.Qobj(np.block([
        [unitaries[m] if m == n else np.zeros((sub_dim, sub_dim)) for n in range(res_dim)]
        for m in range(res_dim)
    ]), dims=[[res_dim, sub_dim], [res_dim, sub_dim]])
    assert np.allclose(composed.full(), qt.to_super(direct).full(), atol=1e-12)


def test_superop_from_component_maps_rejects_missing_components():
    """D: every component with m <= m' must be given."""
    identity = np.eye(4)
    with pytest.raises(ValueError, match="missing"):
        superop_from_component_maps({(0, 0): identity, (1, 1): identity})


@pytest.mark.integration
@requires_store_data
def test_qubit_gate_with_collapse_operators_on_small_device():
    """H: a vanishing rate gives the closed-system map; real jumps keep the trace and lower the purity."""
    import chencrafts.cqed as cqed
    from chencrafts.cqed.mode_assignment import two_mode_dressed_esys

    res_dim, qubit_dim = 4, 3
    sim_para = dict(res_dim=res_dim + 2, qubit_dim=qubit_dim, qubit_ncut=12,
                    num_cpus=1, evals_count=3 * (res_dim + 2))
    para = dict(
        E_osc_GHz=8.3005, g_GHz=0.1276, EJ_GHz=20.0948, EC_GHz=0.2684,
        ng=0.3, disp=np.sqrt(2), tau_m=400, M_ge=0.01, M_eg=0.01, T_W=18000,
        Q_s=1.38e7, temp_a=0.094, temp_s=0.102, Q_cap=1.48e6, Q_ind=1e20,
        kappa_a_phi=6.95e-5, kappa_s_phi=0.0,
        sigma_2_K_a=3 * 0.297 * 2 * np.pi, sigma_omega=2 * np.pi,
        tau_p_by_sigma=20, tau_p_eff_by_sigma=4,
    )
    system = bsq.ResonatorTransmon(sim_para=sim_para, para=para)
    fs = cqed.FlexibleSweep(para=para, swept_para={}, **system.dict, **sim_para,
                            lookup_scheme="LX")
    bsq.batched_sweep_general(fs.sweep)
    bsq.batched_sweep_pulse(fs.sweep)

    kwargs = dict(
        hilbertspace=fs.hilbertspace, res_mode_idx=0, qubit_mode_idx=1,
        res_truncated_dim=res_dim, qubit_truncated_dim=qubit_dim,
        dressed_indices=fs["dressed_indices"], eigensys=(fs["evals"], fs["evecs"]),
        rotation_angle=np.pi / 2, gate_params=fs, num_cpus=1,
        rtol=1e-12, atol=1e-14,
    )
    closed = cat_real.qubit_gate(**kwargs)

    # jumps of the ancilla, written on the full space with the dressed states
    _, dressed = two_mode_dressed_esys(
        fs.hilbertspace, 0, 1, state_label=(0, 0), res_truncated_dim=res_dim,
        qubit_truncated_dim=qubit_dim, dressed_indices=fs["dressed_indices"],
        eigensys=(fs["evals"], fs["evecs"]),
    )

    isometry = trans_by_kets(dressed.ravel())

    def jump(i, j, rate):
        # |i><j| of the dressed ancilla levels, in every photon-number subspace
        truncated = np.sqrt(rate) * qt.tensor(
            qt.qeye(res_dim), qt.basis(qubit_dim, i) * qt.basis(qubit_dim, j).dag())
        return isometry * qt.Qobj(truncated.full()) * isometry.dag()

    # a vanishing rate gives the unitary map
    no_decay = cat_real.qubit_gate(**kwargs, collapse_operators=[jump(0, 1, 0.0)])
    assert np.allclose(no_decay.full(), qt.to_super(closed).full(), atol=1e-8)

    rate = 1e-3     # per ns; large enough for a clear effect during a pulse of about 12 ns
    open_map = cat_real.qubit_gate(
        **kwargs, collapse_operators=[jump(0, 1, rate), jump(1, 1, rate)])
    total = res_dim * qubit_dim
    trace = np.eye(total).reshape(-1, order="F")
    assert np.allclose(trace @ open_map.full(), trace, atol=1e-8)

    # relaxation and dephasing lower the purity of the ancilla state in every subspace
    for m in range(res_dim):
        start = qt.tensor(qt.basis(res_dim, m), qt.basis(qubit_dim, 0))
        rho = start * start.dag()
        out_closed = qt.vector_to_operator(qt.to_super(closed) * qt.operator_to_vector(rho))
        out_open = qt.vector_to_operator(open_map * qt.operator_to_vector(rho))
        pure = (out_closed.full() @ out_closed.full()).trace().real
        mixed = (out_open.full() @ out_open.full()).trace().real
        assert pure == pytest.approx(1.0, abs=1e-6)
        assert mixed < pure - 1e-3


def test_qubit_frame_change_phases_every_ancilla_level_in_every_subspace():
    """A: exp(-i angle n_q) on the ancilla; the resonator position sets the tensor order."""
    from qec_aeb import cat_ideal

    angle = 0.37
    for res_mode_idx in (0, 1):
        z = cat_ideal.qubit_frame_change(4, 3, res_mode_idx, angle)
        phases = np.exp(-1j * angle * np.arange(3))
        expected = (np.kron(np.eye(4), np.diag(phases)) if res_mode_idx == 0
                    else np.kron(np.diag(phases), np.eye(4)))
        assert np.allclose(z.full(), expected)
