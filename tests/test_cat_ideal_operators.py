r"""Ideal operators of finite duration and codewords of ``qec_aeb.cat_ideal``.

The ideal operators of the ancilla pulses and of the parity-mapping idling
keep the phases that the dispersive shift :math:`\chi_1`, the self-Kerr :math:`K_s`
of the resonator, and the qubit-state-dependent resonator non-linearity
:math:`\chi_1'` produce during a step of duration :math:`t`. In the subspace of m
photons, with d = m - n_ref the photon number counted from the reference
photon number n_ref of the frame, the operators are

- :math:`\pm\pi/2` pulse: the phase
  :math:`e^{-i[K_stm(m-1)+\chi_1'td^2/2+\chi_1t(1/2-\nu)d]}` times
  :math:`\hat Z_d\hat R_x(\theta)\hat Z_d`, with
  :math:`\hat Z_d=e^{-i\chi_1t\nu d|1\rangle\langle1|}`, :math:`\hat R_x(\theta)`
  the ancilla rotation, and :math:`\nu` the envelope shape number of the pulse;
- reset :math:`\pi` pulse: the phase
  :math:`e^{-i[K_stm(m-1)+\chi_1'td^2/2+\chi_1tm/2]}` times :math:`\hat R_x(\pi)`;
- parity-mapping idling: the phase
  :math:`e^{-i[K_stm(m-1)+\chi_1'td^2/2]}` on every ancilla level and
  :math:`e^{-i\,\mathrm{angle}\,d}` on level 1.

The tests compare the public functions with these formulas evaluated in each
photon-number subspace, and with each other for the two orders of the tensor
product.
"""

import numpy as np
import pytest
import qutip as qt

from qec_aeb import cat_ideal

# parameters of the order of magnitude of a cat-code device (duration in ns,
# coefficients in rad/ns); K_s and chi_prime are enlarged so that every factor
# of the operators changes the result
PARAMETERS = dict(
    duration=12.0, chi_1=-1.2e-2, K_s=-3.0e-4, chi_prime=5.0e-4, n_ref=2,
)
NU = 0.59
RES_DIM = 7


def _assemble(blocks) -> np.ndarray:
    r"""Operator that acts as ``blocks[m]`` on the ancilla in the subspace of m photons.

    Parameters
    ----------
    blocks : list of np.ndarray
        One ancilla matrix per photon number.

    Returns
    -------
    np.ndarray
        Matrix of the operator on the resonator-ancilla space, with the
        resonator first in the tensor product.
    """
    res_dim, qubit_dim = len(blocks), len(blocks[0])
    full = np.zeros((res_dim * qubit_dim,) * 2, complex)
    for m, block in enumerate(blocks):
        full[qubit_dim * m:qubit_dim * (m + 1), qubit_dim * m:qubit_dim * (m + 1)] = block
    return full


def _x_rotation(qubit_dim: int, theta: float) -> np.ndarray:
    r"""Rotation :math:`e^{-i\theta\hat\sigma_x/2}` of the two lowest ancilla levels.

    The levels above 1 are multiplied by :math:`e^{-i\theta/2}`, because the
    operator of ``cat_ideal.qubit_pauli`` acts as the identity on them.

    Parameters
    ----------
    qubit_dim : int
        Number of ancilla levels.
    theta : float
        Rotation angle (rad).

    Returns
    -------
    np.ndarray
        ``qubit_dim`` x ``qubit_dim`` matrix.
    """
    rotation = np.eye(qubit_dim, dtype=complex) * np.exp(-1j * theta / 2)
    rotation[:2, :2] = [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                        [-1j * np.sin(theta / 2), np.cos(theta / 2)]]
    return rotation


def _formula_half_pi(res_dim, qubit_dim, theta, duration, chi_1, K_s, chi_prime, n_ref, nu):
    r"""Ideal operator of the :math:`\pm\pi/2` pulse, from the formula of the module docstring evaluated in each photon-number subspace.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    theta : float
        Rotation angle (rad).
    duration, chi_1, K_s, chi_prime, n_ref, nu
        As in ``cat_ideal.finite_duration_qubit_rotation``.

    Returns
    -------
    np.ndarray
        Matrix of the operator, with the resonator first in the tensor
        product.
    """
    phi = chi_1 * duration * nu
    rotation = chi_1 * duration * (0.5 - nu)
    blocks = []
    for m in range(res_dim):
        d = m - n_ref
        phase = np.exp(-1j * (K_s * duration * m * (m - 1)
                              + chi_prime / 2 * duration * d ** 2 + rotation * d))
        z = np.eye(qubit_dim, dtype=complex)
        z[1, 1] = np.exp(-1j * phi * d)
        blocks.append(phase * z @ _x_rotation(qubit_dim, theta) @ z)
    return _assemble(blocks)


def _formula_reset(res_dim, qubit_dim, duration, chi_1, K_s, chi_prime, n_ref):
    r"""Ideal operator of the reset :math:`\pi` pulse, from the formula of the module docstring evaluated in each photon-number subspace.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    duration, chi_1, K_s, chi_prime, n_ref
        As in ``cat_ideal.finite_duration_qubit_reset``.

    Returns
    -------
    np.ndarray
        Matrix of the operator, with the resonator first in the tensor
        product.
    """
    return _assemble([
        np.exp(-1j * (K_s * duration * m * (m - 1)
                      + chi_prime / 2 * duration * (m - n_ref) ** 2
                      + chi_1 * duration / 2 * m)) * _x_rotation(qubit_dim, np.pi)
        for m in range(res_dim)])


def _formula_parity_mapping(res_dim, qubit_dim, angle, duration, K_s, chi_prime, n_ref):
    r"""Ideal operator of the parity-mapping idling, the diagonal matrix of the phases of the module docstring.

    Parameters
    ----------
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.
    angle, duration, K_s, chi_prime, n_ref
        As in ``cat_ideal.finite_duration_parity_mapping``.

    Returns
    -------
    np.ndarray
        Diagonal matrix of the operator, with the resonator first in the
        tensor product.
    """
    m = np.arange(res_dim)
    d = m - n_ref
    phase = np.zeros(qubit_dim * res_dim)
    for level in range(qubit_dim):
        phase[level::qubit_dim] = (K_s * duration * m * (m - 1)
                                   + chi_prime / 2 * duration * d ** 2
                                   + (level == 1) * angle * d)
    return np.diag(np.exp(-1j * phase))


def _swap_tensor_order(operator: np.ndarray, res_dim: int, qubit_dim: int) -> np.ndarray:
    r"""Matrix of an operator with the ancilla first in the tensor product, from its matrix with the resonator first.

    Parameters
    ----------
    operator : np.ndarray
        Matrix with the resonator first.
    res_dim, qubit_dim : int
        Dimensions of the resonator and the ancilla.

    Returns
    -------
    np.ndarray
        The same operator with the ancilla first.
    """
    tensor = operator.reshape(res_dim, qubit_dim, res_dim, qubit_dim)
    return tensor.transpose(1, 0, 3, 2).reshape(res_dim * qubit_dim, res_dim * qubit_dim)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [2, 3, 4])
@pytest.mark.parametrize("theta", [np.pi / 2, -np.pi / 2])
def test_finite_duration_qubit_rotation_matches_block_formula(qubit_dim, theta):
    """The ideal operator of the pi/2 pulse equals the formula of the module docstring, for two to four ancilla levels."""
    operator = cat_ideal.finite_duration_qubit_rotation(
        RES_DIM, qubit_dim, 0, angle=theta, nu=NU, **PARAMETERS)
    reference = _formula_half_pi(RES_DIM, qubit_dim, theta, nu=NU, **PARAMETERS)
    assert operator.dims == [[RES_DIM, qubit_dim]] * 2
    assert np.allclose(operator.full(), reference, atol=1e-12)


@pytest.mark.unit
def test_finite_duration_qubit_rotation_matches_script_formula_for_three_levels():
    """For three ancilla levels, the ideal operator of the pi/2 pulse equals the formula written with z = diag(1, e^{-i phi d}, 1) and the rotation of qubit_rot_propagator."""
    qubit_dim, theta = 3, np.pi / 2
    duration, chi_1, n_ref = PARAMETERS["duration"], PARAMETERS["chi_1"], PARAMETERS["n_ref"]
    phi = chi_1 * duration * NU
    rotation = chi_1 * duration * (0.5 - NU)
    x_rotation = cat_ideal.qubit_rot_propagator(1, qubit_dim, 0, angle=theta).full()
    blocks = []
    for m in range(RES_DIM):
        d = m - n_ref
        phase = np.exp(-1j * (PARAMETERS["K_s"] * duration * m * (m - 1)
                              + PARAMETERS["chi_prime"] / 2 * duration * d ** 2 + rotation * d))
        z = np.diag([1, np.exp(-1j * phi * d), 1])
        blocks.append(phase * z @ x_rotation @ z)
    operator = cat_ideal.finite_duration_qubit_rotation(
        RES_DIM, qubit_dim, 0, angle=theta, nu=NU, **PARAMETERS)
    assert np.allclose(operator.full(), _assemble(blocks), atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [2, 3, 4])
def test_finite_duration_qubit_reset_matches_block_formula(qubit_dim):
    """The ideal operator of the reset pulse equals the formula of the module docstring, with the rotation of qubit_rot_propagator."""
    operator = cat_ideal.finite_duration_qubit_reset(RES_DIM, qubit_dim, 0, **PARAMETERS)
    reference = _formula_reset(RES_DIM, qubit_dim, **PARAMETERS)
    assert np.allclose(operator.full(), reference, atol=1e-12)
    # the rotation of the formula equals qubit_rot_propagator, the rotation
    # that finite_duration_qubit_reset uses
    x_rotation = cat_ideal.qubit_rot_propagator(1, qubit_dim, 0, angle=np.pi).full()
    assert np.allclose(x_rotation, _x_rotation(qubit_dim, np.pi), atol=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [2, 3, 4])
@pytest.mark.parametrize("angle", [np.pi - 0.3, -np.pi + 0.3])
def test_finite_duration_parity_mapping_matches_phase_formula(qubit_dim, angle):
    """The ideal operator of the parity-mapping idling is the diagonal matrix of the three phases of the module docstring."""
    kwargs = {key: PARAMETERS[key] for key in ("duration", "K_s", "chi_prime", "n_ref")}
    operator = cat_ideal.finite_duration_parity_mapping(RES_DIM, qubit_dim, 0, angle=angle, **kwargs)
    reference = _formula_parity_mapping(RES_DIM, qubit_dim, angle, **kwargs)
    assert np.allclose(operator.full(), reference, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [3, 4])
def test_finite_duration_operators_in_the_other_tensor_order(qubit_dim):
    """res_mode_idx = 1 gives the same operators with the ancilla first."""
    pm_kwargs = {key: PARAMETERS[key] for key in ("duration", "K_s", "chi_prime", "n_ref")}
    pairs = [
        (lambda idx: cat_ideal.finite_duration_qubit_rotation(
            RES_DIM, qubit_dim, idx, angle=-np.pi / 2, nu=NU, **PARAMETERS)),
        (lambda idx: cat_ideal.finite_duration_qubit_reset(RES_DIM, qubit_dim, idx, **PARAMETERS)),
        (lambda idx: cat_ideal.finite_duration_parity_mapping(
            RES_DIM, qubit_dim, idx, angle=2.5, **pm_kwargs)),
    ]
    for build in pairs:
        ancilla_first = build(1)
        assert ancilla_first.dims == [[qubit_dim, RES_DIM]] * 2
        assert np.allclose(
            ancilla_first.full(),
            _swap_tensor_order(build(0).full(), RES_DIM, qubit_dim), atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("qubit_dim", [2, 3, 4])
def test_cat_codewords_match_explicit_cats(qubit_dim):
    """The codewords are the even-parity superpositions of the coherent states of amplitudes +-alpha and +-i alpha, with the ancilla in |0>."""
    res_dim, alpha = 12, np.sqrt(2)
    logical_0, logical_1 = cat_ideal.cat_codewords(res_dim, qubit_dim, alpha)

    # the states from the Fock amplitudes of a coherent state: only the
    # even photon numbers remain, and the amplitudes of the second codeword
    # change sign at the photon numbers 2 mod 4
    n = np.arange(res_dim)
    log_factorial = np.cumsum(np.log(np.maximum(n, 1)))
    amplitudes = alpha ** n / np.exp(log_factorial / 2)
    for codeword, leg in ((logical_0, 1.0), (logical_1, 1j)):
        resonator = (leg ** n * amplitudes) * (1 + (-1.0) ** n)
        resonator = resonator / np.linalg.norm(resonator)
        expected = np.kron(resonator, np.eye(qubit_dim)[0])
        assert codeword.dims[0] == [res_dim, qubit_dim]
        assert np.allclose(codeword.full().ravel(), expected, atol=1e-12)


@pytest.mark.unit
def test_cat_codewords_in_the_other_tensor_order():
    """res_mode_idx = 1 puts the ancilla first."""
    res_dim, qubit_dim, alpha = 10, 3, 1.3
    resonator_first = cat_ideal.cat_codewords(res_dim, qubit_dim, alpha, res_mode_idx=0)
    ancilla_first = cat_ideal.cat_codewords(res_dim, qubit_dim, alpha, res_mode_idx=1)
    for a, b in zip(resonator_first, ancilla_first):
        assert b.dims[0] == [qubit_dim, res_dim]
        swapped = a.full().reshape(res_dim, qubit_dim).T.ravel()
        assert np.allclose(b.full().ravel(), swapped, atol=1e-14)
