r"""Tests of the analytical cat-code error budget and its sweep ingredients.

Covers ``qec_aeb/batched_sweeps_recipe.py`` (ingredients) and
``qec_aeb/error_budgeting.py`` (``ErrorBudget``, ``cat_code_budget``).
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from scipy.integrate import trapezoid

import qec_aeb as bsq
from qec_aeb.batched_sweeps_recipe import (
    DEFAULT_CAT_BUDGET_PARA,
    _check_cat_budget_para,
    cat_error_space_distribution,
    cat_error_space_moments,
    cat_kerr_rotation_infidelity,
    cat_stark_shift_coefficient,
    photon_number_moments,
    pulse_population_averages,
    pulse_shape_numbers,
)
from qec_aeb.error_budgeting import (
    ErrorBudget,
    _rate,
    cat_branch_probabilities,
    cat_code_budget,
    cat_parity_mapping_idling_error,
    cat_pulse_coherent_error,
    cat_pulse_decoherence,
    cat_pulse_decoherence_by_space,
    cat_pulse_tilt_angles,
    cat_resonator_decoherence,
)
from tests.sweep_view_helpers import requires_idx_add_sweep


# ----------------------------------------------------------------------------
# Shared synthetic ingredients
# ----------------------------------------------------------------------------

def _ofek_like_ingredients(shape=()) -> dict:
    r"""
    Budget ingredients close to the Ofek operating point, broadcast to a grid.

    The numbers are the ones printed by ``batched_sweep_cat_budget_ingredients``
    for the Ofek transmon; the tests only need realistic magnitudes and the
    right trailing shapes, not exact device values. ``jump_adag_a`` is kept
    at its pre-upgrade nonzero value (the current operating point has
    ``kappa_s_phi=0``, hence ``jump_adag_a=0``) on purpose, so the
    storage-dephasing terms are actually exercised by the tests that use
    this fixture.

    Parameters
    ----------
    shape:
        Shape of the parameter grid; scalars are broadcast to it.

    Returns
    -------
    dict
        Keyword ingredients accepted by ``cat_code_budget``.
    """
    # Scalar ingredients on the grid (rates 1/ns, frequencies rad/ns, times ns)
    ones = np.ones(shape)
    scalars = dict(
        disp=np.sqrt(2), T_W=18000.0, tau_m=400.0, tau_FD=332.0, tau_p_eff=11.918,
        t_pm=234.8, M_ge=0.01, M_eg=0.01,
        jump_a=3.919e-6, jump_adag=7.829e-8, jump_adag_a=1.024e-6,
        K_s=-1.160e-5, chi_prime_sa=-1.099e-4, chi_1=-1.2624e-2,
        anc_eta=-1.968, anc_lambda=1.380,
        anc_stark_coeff=(4 - 1.380**2) / -1.968,
        gate_mu_half=0.9130, gate_nu_half=0.5912,
        gate_mu_pi=1.0264, gate_nu_pi=0.4197,
        gate_pop_half=0.2044, gate_pop_sq_half=0.0794,
        gate_pop_pi=0.5, gate_pop_sq_pi=0.4245,
        cat_logical_mean_n=1.898,
        # odd error space after the idle, |beta|^2 = 2 exp(-jump_a T_W),
        # at the angle |K_s| T_W = 0.2088
        cat_kerr_rotation_infidelity=2.133e-2,
    )
    ingredients = {key: value * ones for key, value in scalars.items()}

    # Ancilla jump rates with a third level, so the leakage rate is present
    jump_ij = np.zeros(shape + (3, 3))
    jump_ij[..., 0, 1] = 2.868e-5
    jump_ij[..., 1, 0] = 1.161e-6
    jump_ij[..., 1, 1] = 1.381e-4
    jump_ij[..., 2, 1] = 2.502e-6
    ingredients["jump_ij"] = jump_ij

    # Error-space moments: trailing axis (even, odd)
    per_space = dict(
        cat_mean_n=[1.742, 2.213], cat_var_n=[2.234, 1.500],
        cat_mean_d=[-0.258, 0.213], cat_mean_d2=[2.300, 1.545],
        cat_mean_d4=[13.86, 9.58],
    )
    for key, value in per_space.items():
        ingredients[key] = np.broadcast_to(np.array(value), shape + (2,)).copy()
    return ingredients


# ----------------------------------------------------------------------------
# Photon-number moments of the error spaces
# ----------------------------------------------------------------------------

def _brute_force_moments(amp_sq: float, parity: int, n_ref: int, method: str):
    r"""
    Moments of the maximally mixed error-space state from explicit cat kets.

    Builds the two cats of the requested parity with amplitudes :math:`\beta` and
    :math:`i\beta` in a truncated Fock space, orthonormalizes them (symmetric or
    Gram--Schmidt), and averages the photon-number moments over the two
    orthonormal states.

    Parameters
    ----------
    amp_sq:
        :math:`|\beta|^2`.
    parity:
        0 even, 1 odd.
    n_ref:
        Reference photon number for :math:`d = N - n_{\mathrm{ref}}`.
    method:
        ``"symm"`` or ``"gs"``.

    Returns
    -------
    dict
        Same keys as ``cat_error_space_moments``.
    """
    # Cats |beta> +- |-beta> and |i beta> +- |-i beta>
    dim, beta = 60, np.sqrt(amp_sq)
    sign = 1 if parity == 0 else -1
    kets = [
        (qt.coherent(dim, b) + sign * qt.coherent(dim, -b)).unit()
        for b in (beta, 1j * beta)
    ]
    cols = np.column_stack([k.full().ravel() for k in kets])

    # Orthonormalize the two columns with the requested convention
    if method == "symm":
        gram = cols.conj().T @ cols
        vals, vecs = np.linalg.eigh(gram)
        cols = cols @ vecs @ np.diag(vals ** -0.5) @ vecs.conj().T
    else:
        cols, _ = np.linalg.qr(cols)

    # Photon-number distribution of the maximally mixed state
    prob = np.sum(np.abs(cols) ** 2, axis=1) / 2
    m = np.arange(dim)
    d = m - n_ref
    mean_n = prob @ m
    return dict(mean_n=mean_n, var_n=prob @ m**2 - mean_n**2,
                mean_d=prob @ d, mean_d2=prob @ d**2, mean_d4=prob @ d**4)


@pytest.mark.unit
@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("method", ["symm", "gs"])
def test_moments_match_explicit_cats(parity, method):
    """E, C: the mod-4 formula equals explicit cats, for either orthonormalization."""
    amp_sq, n_ref = 1.864, 2
    expected = _brute_force_moments(amp_sq, parity, n_ref, method)
    got = cat_error_space_moments(amp_sq, parity, n_ref)
    for key, value in expected.items():
        assert np.isclose(got[key], value, rtol=1e-7, atol=1e-9), key


@pytest.mark.unit
def test_moments_large_cat_limit():
    """D: for a large cat, <N> -> |beta|^2 and Delta N^2 -> |beta|^2 (coherent state)."""
    amp_sq = 30.0
    for parity in (0, 1):
        got = cat_error_space_moments(amp_sq, parity, n_ref=30)
        assert np.isclose(got["mean_n"], amp_sq, rtol=1e-6)
        assert np.isclose(got["var_n"], amp_sq, rtol=1e-6)


# ----------------------------------------------------------------------------
# Pulse-shape integrals
# ----------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("theta", [np.pi / 2, np.pi])
def test_square_pulse_shape_numbers(theta):
    """A: for a square pulse mu = nu = (1 - cos theta) / theta."""
    mu, nu = pulse_shape_numbers(theta, envelope="square")
    expected = (1 - np.cos(theta)) / theta
    assert np.isclose(mu, expected, rtol=1e-6)
    assert np.isclose(nu, expected, rtol=1e-6)


@pytest.mark.unit
def test_gaussian_pulse_shape_numbers_reference():
    """A: truncated-Gaussian values quoted in cat-code-example.tex."""
    mu_half, nu_half = pulse_shape_numbers(np.pi / 2)
    mu_pi, nu_pi = pulse_shape_numbers(np.pi)
    assert np.allclose([mu_half, nu_half, mu_pi, nu_pi],
                       [0.913, 0.591, 1.026, 0.420], atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("theta", [np.pi / 2, np.pi])
def test_square_pulse_population_averages(theta):
    """A: square-pulse closed forms of the time averages of P_1 and P_1^2."""
    mean, mean_sq = pulse_population_averages(theta, envelope="square")
    assert np.isclose(mean, (1 - np.sin(theta) / theta) / 2, rtol=1e-6)
    assert np.isclose(
        mean_sq,
        3 / 8 - np.sin(theta) / (2 * theta) + np.sin(2 * theta) / (16 * theta),
        rtol=1e-6)


@pytest.mark.unit
def test_gaussian_pulse_population_averages_identities():
    """C, A: symmetric-pulse identities with nu_{pi/2}; reference value of the pi pulse."""
    nu_half = pulse_shape_numbers(np.pi / 2)[1]
    mean, mean_sq = pulse_population_averages(np.pi / 2)
    assert np.isclose(mean, (1 - nu_half) / 2, rtol=1e-6)
    assert np.isclose(mean_sq, (3 - 4 * nu_half) / 8, rtol=1e-6)
    mean, mean_sq = pulse_population_averages(np.pi)
    assert np.isclose(mean, 0.5, rtol=1e-6)
    assert np.isclose(mean_sq, 0.4245, atol=1e-4)


@pytest.mark.unit
def test_pulse_shape_rejects_unknown_envelope():
    """D: an unknown envelope name raises."""
    with pytest.raises(ValueError, match="envelope"):
        pulse_shape_numbers(np.pi, envelope="cosine")


# ----------------------------------------------------------------------------
# ErrorBudget container
# ----------------------------------------------------------------------------

@pytest.mark.unit
def test_budget_groupings_sum_to_total():
    """C: sums by name, step and mechanism all equal the total, on a grid."""
    ingredients = _ofek_like_ingredients(shape=(3,))
    total = cat_code_budget(**ingredients)
    assert total.shape == (3,)
    for organize_by in ("name", "step", "mechanism"):
        grouped = cat_code_budget(organize_by, **ingredients)
        assert np.allclose(sum(grouped.values()), total)


@pytest.mark.unit
def test_budget_steps_cover_the_cycle():
    """D: the budget has entries for all seven steps, in order."""
    steps = cat_code_budget("step", **_ofek_like_ingredients())
    assert list(steps.keys()) == [1, 2, 3, 4, 5, 6, 7]
    assert all(np.all(value > 0) for value in steps.values())


@pytest.mark.unit
def test_budget_disable_and_subset():
    """C, D: disabling a term removes it; indexing by a list gives a sub-budget."""
    ingredients = _ofek_like_ingredients()
    total = cat_code_budget(**ingredients)
    assignment = cat_code_budget["meas_assignment"](**ingredients)

    # Disabling a term lowers the total by exactly that term
    cat_code_budget.disable_contrib("meas_assignment")
    try:
        assert np.isclose(cat_code_budget(**ingredients), total - assignment)
    finally:
        cat_code_budget.enable_contrib("meas_assignment")

    # A sub-budget of the idle terms reproduces the step-1 entry
    idle_names = [n for n in cat_code_budget.error_names if n.startswith("idle")]
    sub = cat_code_budget[idle_names]
    assert isinstance(sub, ErrorBudget)
    assert np.isclose(sub(**ingredients),
                      cat_code_budget("step", **ingredients)[1])


@pytest.mark.unit
def test_budget_rejects_unknown_grouping():
    """D: an unknown organize_by raises."""
    with pytest.raises(ValueError, match="organize_by"):
        cat_code_budget("color", **_ofek_like_ingredients())


# ----------------------------------------------------------------------------
# Cat-code formulas
# ----------------------------------------------------------------------------

@pytest.mark.unit
def test_idle_terms_reference():
    """A: idle terms recomputed from their closed forms with a Poissonian number of photon losses."""
    kw = _ofek_like_ingredients()
    n_bar, t = 2.0, kw["T_W"]
    got = cat_code_budget("name", **kw)
    assert np.isclose(got["idle_ancilla_excitation"], 1.161e-6 * t)
    assert np.isclose(got["idle_storage_dephasing"], 1.024e-6 * n_bar * t)
    # Mean number of photon losses in the idle
    mean_loss_number = 3.919e-6 * n_bar * t
    # Probability of exactly one or two losses, which compensate a gain
    compensated = np.exp(-mean_loss_number) * (
        mean_loss_number + mean_loss_number**2 / 2)
    assert np.isclose(got["idle_photon_gain"], 7.829e-8 * (n_bar + 1) * t * (1 - compensated))
    # Probability of two or more losses
    assert np.isclose(got["idle_multiple_photon_loss"],
                      1 - np.exp(-mean_loss_number) * (1 + mean_loss_number))
    # Probability of exactly one loss times the infidelity of the Kerr rotation
    assert np.isclose(got["idle_kerr_after_loss"],
                      mean_loss_number * np.exp(-mean_loss_number) * 2.133e-2)


@pytest.mark.unit
@pytest.mark.parametrize("kerr_angle", [0.05, 0.6, 2.5])
def test_kerr_rotation_infidelity_matches_explicit_cats(kerr_angle):
    """E: rotation infidelity from explicit odd cats, averaged over the loss time by quadrature."""
    amp_sq, dim = 1.86, 60
    # Orthonormal basis of the odd error space from the cats themselves
    beta = np.sqrt(amp_sq)
    kets = [(qt.coherent(dim, b) - qt.coherent(dim, -b)).unit() for b in (beta, 1j * beta)]
    basis, _ = np.linalg.qr(np.column_stack([k.full().ravel() for k in kets]))
    photon_numbers = np.arange(dim)

    # Process fidelity |tr(P e^{-i theta N})/2|^2 on a grid of the angle,
    # which is uniform on [-Kt, Kt] for a uniform loss time
    angles = np.linspace(-kerr_angle, kerr_angle, 4001)
    rotated = np.exp(-1j * np.outer(angles, photon_numbers))
    traces = np.einsum("mi,am,mi->a", basis.conj(), rotated, basis)
    reference = 1 - trapezoid(np.abs(traces / 2) ** 2, angles) / (2 * kerr_angle)

    assert np.isclose(cat_kerr_rotation_infidelity(amp_sq, 1, kerr_angle), reference, rtol=1e-6)
    # The distribution that the function averages over is that of the cats
    assert np.allclose(cat_error_space_distribution(amp_sq, 1, dim - 1),
                       np.sum(np.abs(basis) ** 2, axis=1) / 2, atol=1e-12)


@pytest.mark.unit
def test_kerr_rotation_infidelity_small_angle_and_large_cat():
    """C: (Kt)^2 var(N)/3 at a small angle, and var(N) -> |beta|^2 for a large cat."""
    kerr_angle = 1e-3
    for amp_sq in (2.0, 12.0):
        var_n = cat_error_space_moments(amp_sq, 1, 0)["var_n"]
        assert np.isclose(cat_kerr_rotation_infidelity(amp_sq, 1, kerr_angle),
                          kerr_angle**2 * var_n / 3, rtol=1e-5)
    # Large cat: the idle term approaches gamma nbar^2 K^2 t^3 / 3 when the
    # mean number of losses is small
    n_bar, jump_a, kerr, t = 12.0, 1e-9, 1e-7, 1e4
    term = cat_code_budget["idle_kerr_after_loss"](
        jump_a=jump_a, disp=np.sqrt(n_bar), T_W=t,
        cat_kerr_rotation_infidelity=cat_kerr_rotation_infidelity(
            n_bar * np.exp(-jump_a * t), 1, kerr * t))
    assert np.isclose(term, jump_a * n_bar**2 * kerr**2 * t**3 / 3, rtol=1e-3)


@pytest.mark.unit
def test_gate_stark_angle_reference():
    """A: Stark angle of the X_{pi/2} pulse, -0.0507 as quoted in cat-code-example.tex."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    stark = cat_code_budget["gate1_stark"](**kw)
    phi_s = -np.sqrt(4 * stark / (p_plus + p_minus))
    assert np.isclose(phi_s, -0.0507, atol=5e-4)


@pytest.mark.unit
def test_pulse_decoherence_matches_master_equation():
    """E: first-order jump costs equal the infidelity of a resonant two-level rotation with jumps."""
    # square pi/2 rotation of a three-level ancilla from |0>, weak jumps
    theta, duration = np.pi / 2, 10.0
    rates = {(0, 1): 3e-5, (1, 0): 2e-6, (1, 1): 1e-4, (2, 1): 4e-6}
    jump_ij = np.zeros((3, 3))
    for (i, j), rate in rates.items():
        jump_ij[i, j] = rate
    drive = theta / duration / 2 * (qt.basis(3, 0) * qt.basis(3, 1).dag()
                                    + qt.basis(3, 1) * qt.basis(3, 0).dag())
    target = (-1j * drive * duration).expm() * qt.basis(3, 0)
    expected = cat_pulse_decoherence(
        jump_ij, duration, *pulse_population_averages(theta, envelope="square"))
    names = {(0, 1): "relaxation", (1, 0): "excitation",
             (1, 1): "dephasing", (2, 1): "leakage"}
    for (i, j), rate in rates.items():
        jump = np.sqrt(rate) * qt.basis(3, i) * qt.basis(3, j).dag()
        final = qt.mesolve(drive, qt.basis(3, 0), [0, duration], [jump],
                           options=dict(atol=1e-12, rtol=1e-10)).states[-1]
        infidelity = 1 - qt.expect(final, target)
        assert np.isclose(infidelity, expected[names[(i, j)]], rtol=2e-3), names[(i, j)]


@pytest.mark.unit
def test_pulse_decoherence_terms_reference():
    """A, C: decoherence terms of the pulses; the odd-space second pulse and the reset exchange the levels."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    t_g = kw["tau_p_eff"]
    # dephasing: mean of P_0 P_1 is 1/8 for both trajectories of a pi/2 pulse
    dephasing = 1.381e-4 * (0.2044 - 0.0794) * t_g
    assert np.isclose(got["gate1_ancilla_dephasing"], (p_plus + p_minus) * dephasing)
    assert np.isclose(got["gate2_ancilla_dephasing"], (p_plus + p_minus) * dephasing)
    # jumps from |0> to the equator, and from the equator to |1>; relaxation
    # is the downward jump, excitation the two upward jumps
    ground_sq = 1 - 2 * 0.2044 + 0.0794
    down_to_equator = 2.868e-5 * 0.0794 * t_g
    up_to_equator = (1.161e-6 * ground_sq + 2.502e-6 * 0.2044) * t_g
    down_to_excited = 2.868e-5 * ground_sq * t_g
    up_to_excited = (1.161e-6 * 0.0794 + 2.502e-6 * (1 - 0.2044)) * t_g
    assert np.isclose(got["gate1_ancilla_relaxation"], (p_plus + p_minus) * down_to_equator)
    assert np.isclose(got["gate1_ancilla_excitation"], (p_plus + p_minus) * up_to_equator)
    assert np.isclose(got["gate2_ancilla_relaxation"],
                      p_plus * down_to_equator + p_minus * down_to_excited)
    assert np.isclose(got["gate2_ancilla_excitation"],
                      p_plus * up_to_equator + p_minus * up_to_excited)
    # reset: odd error space only, pi pulse of duration 2 t_g
    assert np.isclose(got["reset_ancilla_dephasing"],
                      p_minus * 1.381e-4 * (0.5 - 0.4245) * 2 * t_g)
    assert np.isclose(got["reset_ancilla_relaxation"], p_minus * 2.868e-5 * 0.4245 * 2 * t_g)
    assert np.isclose(got["reset_ancilla_excitation"],
                      p_minus * (1.161e-6 * 0.4245 + 2.502e-6 * 0.5) * 2 * t_g)


@pytest.mark.unit
@pytest.mark.parametrize("scheme, drag_term", [("plain", 0.0), ("drag", 4.0)])
def test_stark_coefficient_ladder_limit(scheme, drag_term):
    """A: with only neighboring-level elements, s_S = (drag_term - lambda^2) / alpha."""
    energies = np.array([0.0, 6.0, 11.7, 17.0])
    lam = 1.4
    elements = np.zeros((4, 4))
    elements[0, 1] = elements[1, 0] = 1.0
    elements[1, 2] = elements[2, 1] = lam
    alpha = 11.7 - 2 * 6.0
    got = cat_stark_shift_coefficient(energies, elements, scheme)
    assert np.isclose(got, (drag_term - lam**2) / alpha)


@pytest.mark.unit
def test_stark_coefficient_extra_levels_and_dressed_alpha():
    """A: a 0-3 element adds +|O_03/O_01|^2/(E_3-E_1) to s_S; anharmonicity overrides the 1-2 term."""
    energies = np.array([0.0, 6.0, 11.7, 17.0])
    elements = np.zeros((4, 4))
    elements[0, 1] = elements[1, 0] = 1.0
    elements[1, 2] = elements[2, 1] = 1.4
    base = cat_stark_shift_coefficient(energies, elements, "plain")
    elements[0, 3] = elements[3, 0] = 0.3
    extra = cat_stark_shift_coefficient(energies, elements, "plain")
    assert np.isclose(extra - base, 0.3**2 / (17.0 - 6.0))
    # the dressed value replaces the bare alpha in the 1-2 term only
    dressed = cat_stark_shift_coefficient(energies, elements, "plain", anharmonicity=-0.5)
    assert np.isclose(dressed - extra, -1.4**2 / -0.5 + 1.4**2 / (11.7 - 12.0))


@pytest.mark.unit
def test_budget_para_defaults_and_validation():
    """D: defaults are kept, overrides merge, and bad entries are rejected."""
    assert _check_cat_budget_para(None) == DEFAULT_CAT_BUDGET_PARA
    merged = _check_cat_budget_para({"pulse_scheme": "plain"})
    assert merged["pulse_scheme"] == "plain"
    assert merged["drive_operator"] == DEFAULT_CAT_BUDGET_PARA["drive_operator"]
    with pytest.raises(KeyError):
        _check_cat_budget_para({"pulse_shceme": "plain"})
    with pytest.raises(ValueError):
        _check_cat_budget_para({"pulse_scheme": "square"})
    with pytest.raises(ValueError):
        _check_cat_budget_para({"stark_level_count": 2})


@pytest.mark.unit
def test_branch_probabilities_identity():
    """C: p_+ + p_- = 1 - (idle infidelity) and p_- / p_+ = gamma <N>_L t_w."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    idle = cat_code_budget("step", **kw)[1]
    assert np.isclose(p_plus + p_minus, 1 - idle)
    assert np.isclose(p_minus / p_plus, kw["jump_a"] * kw["cat_logical_mean_n"] * kw["T_W"])


@pytest.mark.unit
def test_two_level_ancilla_has_no_leakage_rate():
    """D: _rate returns zeros for a level beyond the truncation."""
    jump_ij = np.ones((4, 2, 2))
    assert np.array_equal(_rate(jump_ij, 2, 1), np.zeros(4))
    assert np.array_equal(_rate(jump_ij, 0, 1), np.ones(4))


@pytest.mark.unit
def test_measurement_terms_reference():
    """A: assignment and ancilla-jump terms of the measurement."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    assert np.isclose(got["meas_assignment"], 0.01 * (p_plus + p_minus))
    assert np.isclose(got["meas_ancilla_relaxation"], p_minus * 2.868e-5 * 400.0)
    assert np.isclose(got["meas_ancilla_excitation"],
                      (p_plus * 1.161e-6 + p_minus * 2.502e-6) * 400.0)


@pytest.mark.unit
def test_feedback_latency_terms_reference():
    """A, B: the latency terms are the measurement-wait terms at tau_FD and vanish without it."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    assert np.isclose(got["latency_ancilla_relaxation"], p_minus * 2.868e-5 * 332.0)
    assert np.isclose(got["latency_ancilla_excitation"],
                      (p_plus * 1.161e-6 + p_minus * 2.502e-6) * 332.0)
    # The storage terms of the two waits differ only by the duration
    for term in ("photon_loss", "photon_gain", "storage_dephasing"):
        assert np.isclose(got[f"latency_{term}"], got[f"meas_{term}"] * 332.0 / 400.0)

    # Ingredients without tau_FD describe a cycle without latency: step 6
    # vanishes and every other step is unchanged
    without = {name: value for name, value in kw.items() if name != "tau_FD"}
    steps = cat_code_budget("step", **kw)
    steps_without = cat_code_budget("step", **without)
    assert np.all(steps_without[6] == 0)
    for step in (1, 2, 3, 4, 5, 7):
        assert np.allclose(steps_without[step], steps[step])


# ----------------------------------------------------------------------------
# Terms of the analytical formula for one error space
# ----------------------------------------------------------------------------

def _average(p_plus, p_minus, even, odd):
    r"""
    Average :math:`p_+x_++p_-x_-` of one term of the analytical formula over the two error spaces.

    Parameters
    ----------
    p_plus, p_minus:
        Probabilities :math:`p_\pm` that the idle leaves the logical
        information in the even and in the odd error space.
    even, odd:
        Values :math:`x_\pm` of the term in the even and in the odd error
        space.

    Returns
    -------
    float or np.ndarray
    """
    return p_plus * even + p_minus * odd


def _space_moments(kw: dict, space: int) -> dict:
    r"""
    Photon-number moments of one error space, from the budget ingredients.

    Parameters
    ----------
    kw:
        Budget ingredients with the keys ``cat_<moment>``, trailing axis
        (even, odd).
    space:
        0 for the even error space, 1 for the odd one.

    Returns
    -------
    dict
        Keys ``mean_n``, ``var_n``, ``mean_d``, ``mean_d2``, ``mean_d4``, as
        returned by ``photon_number_moments``.
    """
    return {key: kw[f"cat_{key}"][..., space]
            for key in ("mean_n", "var_n", "mean_d", "mean_d2", "mean_d4")}


@pytest.mark.unit
def test_photon_number_moments_known_distributions():
    """A, C: moments of a two-point and of a Poissonian distribution; the cat moments use the same function."""
    # half of the probability on m = 0 and half on m = 2, reference 1: d = -1, +1
    got = photon_number_moments([0.5, 0.0, 0.5], n_ref=1)
    assert got == dict(mean_n=1.0, var_n=1.0, mean_d=0.0, mean_d2=1.0, mean_d4=1.0)

    # Poissonian distribution of mean lam: variance lam and central moments
    # mu_2 = lam, mu_4 = lam + 3 lam^2; the moments of d follow from the
    # shift lam - n_ref of the mean from the reference photon number
    lam, n_ref = 2.3, 2
    m = np.arange(80)
    prob = np.exp(m * np.log(lam) - lam - np.cumsum(np.log(np.maximum(m, 1))))
    got = photon_number_moments(prob, n_ref)
    shift = lam - n_ref
    assert np.isclose(got["mean_n"], lam)
    assert np.isclose(got["var_n"], lam)
    assert np.isclose(got["mean_d"], shift)
    assert np.isclose(got["mean_d2"], lam + shift**2)
    # <d^4> = mu_4 + 4 mu_3 shift + 6 mu_2 shift^2 + shift^4, with mu_3 = lam
    assert np.isclose(
        got["mean_d4"], lam + 3 * lam**2 + 4 * lam * shift + 6 * lam * shift**2 + shift**4)

    # the moments of a cat error space are those of its photon-number
    # distribution; in the even error space the photon numbers 0 and 2
    # modulo 4 each carry the probability 1/2
    amp_sq = 1.864
    weights = np.exp(m * np.log(amp_sq) - np.cumsum(np.log(np.maximum(m, 1))))
    prob = np.zeros_like(weights)
    for residue in (0, 2):
        component = m % 4 == residue
        prob[component] = 0.5 * weights[component] / weights[component].sum()
    expected = cat_error_space_moments(amp_sq, 0, n_ref)
    got = photon_number_moments(prob, n_ref)
    for key, value in expected.items():
        assert np.isclose(got[key], value, rtol=1e-12), key


@pytest.mark.unit
def test_pulse_tilt_angles():
    """A, C: Stark-shift angle and residual dispersive-shift angle of the pulses of the cycle; scaling with the duration; sign for a negative rotation angle."""
    kw = _ofek_like_ingredients()
    t_g, chi_1, s_s = kw["tau_p_eff"], kw["chi_1"], kw["anc_stark_coeff"]
    mu, nu = kw["gate_mu_half"], kw["gate_nu_half"]

    # pi/2 pulse: reference value -0.0507 rad of the Stark-shift angle for
    # the synthetic ingredients (tolerance 5e-4 rad); the ideal operator
    # retains the pulse's own envelope shape number, so phi_chi = 0
    phi_s, phi_chi = cat_pulse_tilt_angles(np.pi / 2, t_g, chi_1, s_s, mu, nu, nu)
    assert np.isclose(phi_s, -0.0507, atol=5e-4)
    assert phi_chi == 0

    # reset: pi pulse of duration 2 t_g whose ideal operator contains none
    # of the tilt
    phi_s_pi, phi_chi_pi = cat_pulse_tilt_angles(
        np.pi, 2 * t_g, chi_1, s_s, kw["gate_mu_pi"], kw["gate_nu_pi"], 0.0)
    assert np.isclose(phi_s_pi, s_s * kw["gate_mu_pi"] * np.pi**2 / (8 * t_g))
    assert np.isclose(phi_chi_pi, chi_1 * 2 * t_g * kw["gate_nu_pi"])

    # any duration, also on a grid of durations: phi_S is proportional to
    # 1 / t and phi_chi to t
    durations = t_g * np.array([0.5, 1.0, 3.0])
    on_grid = cat_pulse_tilt_angles(np.pi / 2, durations, chi_1, s_s, mu, nu, 0.0)
    assert np.allclose(on_grid[0] * durations, on_grid[0][1] * t_g)
    assert np.allclose(on_grid[1] / durations, chi_1 * nu)

    # negative rotation angle: the negatives of the envelope shape numbers
    # reverse the sign of both angles
    forward = cat_pulse_tilt_angles(np.pi / 2, t_g, chi_1, s_s, mu, nu, 0.2)
    backward = cat_pulse_tilt_angles(-np.pi / 2, t_g, chi_1, s_s, -mu, -nu, -0.2)
    assert np.allclose(backward, -np.array(forward))


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["half", "pi"])
def test_pulse_coherent_error_matches_distribution(kind):
    """E: the three terms add up to the population (phi_S + phi_chi d)^2 / 4 that the tilt leaves outside the intended ancilla state, averaged over the photon numbers."""
    prob = np.array([0.1, 0.0, 0.45, 0.05, 0.3, 0.1])
    n_ref, phi_s, phi_chi = 2, -0.05, 0.013
    moments = photon_number_moments(prob, n_ref)
    d = np.arange(len(prob)) - n_ref
    expected = prob @ (phi_s + phi_chi * d) ** 2 / 4
    if kind == "half":
        # a pi/2 pulse adds the cost of the rotation of the cat by the
        # phase phi_chi / 2 per photon
        expected += phi_chi**2 * moments["var_n"] / 4
    got = cat_pulse_coherent_error(
        kind, phi_s, phi_chi, moments["mean_d"], moments["mean_d2"], moments["var_n"])
    assert set(got) == {"stark_shift", "dispersive_shift", "cross"}
    assert np.isclose(sum(got.values()), expected, rtol=1e-12)
    assert np.isclose(got["stark_shift"], phi_s**2 / 4)
    assert np.isclose(got["cross"], phi_s * phi_chi * moments["mean_d"] / 2)


@pytest.mark.unit
def test_pulse_coherent_error_rejects_bad_input():
    """D: a pi/2 pulse needs the photon-number variance; an unknown pulse kind raises ValueError."""
    with pytest.raises(ValueError, match="var_n"):
        cat_pulse_coherent_error("half", 0.1, 0.1, 0.0, 1.0)
    with pytest.raises(ValueError, match="kind"):
        cat_pulse_coherent_error("quarter", 0.1, 0.1, 0.0, 1.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("frame_update", [0.0, 1.0])
def test_pulse_coherent_error_by_error_space_reproduces_terms(frame_update):
    """C: the coherent error for one error space, averaged over the two error spaces, reproduces the coherent terms of each pulse."""
    kw = _ofek_like_ingredients() | {"frame_update": frame_update}
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    even, odd = _space_moments(kw, 0), _space_moments(kw, 1)

    # pi/2 pulses: both error spaces; the frame changes remove the Stark-shift term.
    # The pulse has the duration t_g, and its ideal operator retains the
    # pulse's own envelope shape number
    t_g = kw["tau_p_eff"]
    phi_s, phi_chi = cat_pulse_tilt_angles(
        np.pi / 2, t_g, kw["chi_1"], kw["anc_stark_coeff"],
        kw["gate_mu_half"], kw["gate_nu_half"], kw["gate_nu_half"])
    costs = [cat_pulse_coherent_error(
        "half", phi_s, phi_chi, space["mean_d"], space["mean_d2"], space["var_n"])
        for space in (even, odd)]
    for prefix in ("gate1", "gate2"):
        assert np.isclose(
            got[f"{prefix}_stark"],
            (1 - frame_update) * _average(
                p_plus, p_minus, costs[0]["stark_shift"], costs[1]["stark_shift"]),
            rtol=1e-12, atol=0)
        assert got[f"{prefix}_dispersive"] == 0 == costs[0]["dispersive_shift"]
        assert got[f"{prefix}_cross"] == 0 == costs[1]["cross"]
    assert (got["gate1_stark"] > 0) == (frame_update == 0)

    # reset: odd error space only, no frame change; a pi pulse of duration
    # 2 t_g whose ideal operator contains none of the tilt
    phi_s, phi_chi = cat_pulse_tilt_angles(
        np.pi, 2 * t_g, kw["chi_1"], kw["anc_stark_coeff"],
        kw["gate_mu_pi"], kw["gate_nu_pi"], 0.0)
    cost = cat_pulse_coherent_error("pi", phi_s, phi_chi, odd["mean_d"], odd["mean_d2"])
    for term, key in [("stark", "stark_shift"), ("dispersive", "dispersive_shift"),
                      ("cross", "cross")]:
        assert got[f"reset_{term}"] != 0
        assert np.isclose(got[f"reset_{term}"], p_minus * cost[key], rtol=1e-12, atol=0)


@pytest.mark.unit
def test_pulse_decoherence_by_space_reproduces_terms():
    """C, D: the decoherence costs in the two error spaces, averaged over them, reproduce the decoherence terms; which pulse exchanges the populations in which error space."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    t_g = kw["tau_p_eff"]
    half = (kw["gate_pop_half"], kw["gate_pop_sq_half"])
    full = (kw["gate_pop_pi"], kw["gate_pop_sq_pi"])
    cases = dict(gate1=(t_g, half, (p_plus, p_minus)), gate2=(t_g, half, (p_plus, p_minus)),
                 reset=(2 * t_g, full, (0.0, p_minus)))
    for pulse, (duration, averages, weights) in cases.items():
        costs = cat_pulse_decoherence_by_space(pulse, kw["jump_ij"], duration, *averages)
        assert set(costs) == {"relaxation", "excitation", "dephasing", "leakage"}
        assert all(value.shape == (2,) for value in costs.values())
        upward = costs["excitation"] + costs["leakage"]
        assert np.isclose(got[f"{pulse}_ancilla_dephasing"],
                          _average(*weights, *costs["dephasing"]), rtol=1e-12, atol=0)
        assert np.isclose(got[f"{pulse}_ancilla_relaxation"],
                          _average(*weights, *costs["relaxation"]), rtol=1e-12, atol=0)
        assert np.isclose(got[f"{pulse}_ancilla_excitation"],
                          _average(*weights, *upward), rtol=1e-12, atol=0)

    # which time averages each pulse takes in each error space: those of an
    # ancilla that starts in |0> (direct), or those with the populations of
    # |0> and |1> exchanged
    mean, mean_sq = half
    direct = cat_pulse_decoherence(kw["jump_ij"], t_g, mean, mean_sq)
    exchanged = cat_pulse_decoherence(kw["jump_ij"], t_g, 1 - mean, 1 - 2 * mean + mean_sq)
    for pulse, (even, odd) in dict(gate1=(direct, direct), gate2=(direct, exchanged),
                                   reset=(exchanged, exchanged)).items():
        costs = cat_pulse_decoherence_by_space(pulse, kw["jump_ij"], t_g, mean, mean_sq)
        for name in direct:
            assert costs[name][0] == even[name] and costs[name][1] == odd[name]
    assert not np.isclose(direct["leakage"], exchanged["leakage"])


@pytest.mark.unit
@pytest.mark.parametrize("frame_update", [0.0, 1.0])
def test_parity_mapping_idling_error_by_error_space_reproduces_terms(frame_update):
    """C: the terms of the parity-mapping idling for one error space, averaged over the two error spaces, reproduce the budget terms of step 3."""
    kw = _ofek_like_ingredients() | {"frame_update": frame_update}
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    # reference of the frame change: <d^2> of the even error space, or zero
    reference = frame_update * kw["cat_mean_d2"][..., 0]
    costs = []
    for space in (0, 1):
        moments = _space_moments(kw, space)
        costs.append(cat_parity_mapping_idling_error(
            kw["jump_ij"], kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"],
            kw["chi_prime_sa"], kw["t_pm"],
            mean_n=moments["mean_n"], var_n=moments["var_n"],
            mean_d2=moments["mean_d2"], mean_d4=moments["mean_d4"],
            reference_d2=reference))
    term_of_key = dict(
        ancilla_dephasing="pm_ancilla_dephasing",
        ancilla_relaxation="pm_ancilla_relaxation",
        ancilla_excitation="pm_ancilla_excitation",
        photon_loss="pm_photon_loss", photon_gain="pm_photon_gain",
        resonator_dephasing="pm_storage_dephasing", nonlinearity="pm_nonlinearity")
    assert set(costs[0]) == set(term_of_key)
    for key, term in term_of_key.items():
        assert got[term] > 0
        assert np.isclose(got[term], _average(p_plus, p_minus, costs[0][key], costs[1][key]),
                          rtol=1e-12, atol=0), key
    # the seven terms are the whole of step 3
    assert np.isclose(
        cat_code_budget("step", **kw)[3],
        sum(_average(p_plus, p_minus, costs[0][key], costs[1][key]) for key in term_of_key),
        rtol=1e-12)


@pytest.mark.unit
def test_parity_mapping_idling_error_by_error_space_reference_and_durations():
    """A, E: closed forms of the terms for one error space on a grid of durations; non-linearity term from an explicit photon-number distribution."""
    prob = np.array([0.1, 0.0, 0.45, 0.05, 0.3, 0.1])
    n_ref, chi_prime, reference = 2, -1.1e-4, 1.7
    moments = photon_number_moments(prob, n_ref)
    jump_ij = np.zeros((3, 3))
    jump_ij[0, 1], jump_ij[1, 0], jump_ij[1, 1], jump_ij[2, 1] = 3e-5, 1e-6, 1.4e-4, 2.5e-6
    times = np.array([0.0, 100.0, 235.0])
    got = cat_parity_mapping_idling_error(
        jump_ij, 4e-6, 8e-8, 1e-6, chi_prime, times,
        mean_n=moments["mean_n"], var_n=moments["var_n"],
        mean_d2=moments["mean_d2"], mean_d4=moments["mean_d4"], reference_d2=reference)
    assert all(value.shape == times.shape for value in got.values())
    assert np.allclose(got["ancilla_dephasing"], 1.4e-4 * times / 4)
    assert np.allclose(got["ancilla_relaxation"], 3e-5 * times / 2)
    assert np.allclose(got["ancilla_excitation"], (1e-6 + 2.5e-6) * times / 2)
    assert np.allclose(got["photon_loss"], 4e-6 * moments["mean_n"] * times)
    assert np.allclose(got["photon_gain"], 8e-8 * (moments["mean_n"] + 1) * times)
    assert np.allclose(got["resonator_dephasing"], 1e-6 * moments["var_n"] * times)
    # mean square, over the photon numbers, of the ancilla phase
    # chi_1' t (d^2 - r) that remains after the frame change, divided by 4
    d = np.arange(len(prob)) - n_ref
    assert np.allclose(got["nonlinearity"],
                       (chi_prime * times) ** 2 / 4 * (prob @ (d**2 - reference) ** 2))
    # without the frame change the reference is zero: (chi_1' t)^2 <d^4> / 4
    unframed = cat_parity_mapping_idling_error(
        jump_ij, 4e-6, 8e-8, 1e-6, chi_prime, times,
        mean_n=moments["mean_n"], var_n=moments["var_n"],
        mean_d2=moments["mean_d2"], mean_d4=moments["mean_d4"])
    assert np.allclose(unframed["nonlinearity"],
                       (chi_prime * times) ** 2 / 4 * moments["mean_d4"])
    # a two-level ancilla has no leakage jump
    two_level = cat_parity_mapping_idling_error(
        jump_ij[:2, :2], 4e-6, 8e-8, 1e-6, chi_prime, times,
        mean_n=moments["mean_n"], var_n=moments["var_n"],
        mean_d2=moments["mean_d2"], mean_d4=moments["mean_d4"])
    assert np.allclose(two_level["ancilla_relaxation"], 3e-5 * times / 2)
    assert np.allclose(two_level["ancilla_excitation"], 1e-6 * times / 2)


@pytest.mark.unit
@pytest.mark.parametrize("prefix, time_key", [("pm", "t_pm"), ("meas", "tau_m"),
                                              ("latency", "tau_FD")])
def test_resonator_decoherence_by_error_space_reproduces_terms(prefix, time_key):
    """C: the photon-loss, photon-gain and resonator-dephasing terms for one error space, averaged over the two error spaces, reproduce the terms of steps 3, 5 and 6."""
    kw = _ofek_like_ingredients()
    p_plus, p_minus = cat_branch_probabilities(**kw)
    got = cat_code_budget("name", **kw)
    costs = [cat_resonator_decoherence(
        kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"], kw[time_key],
        mean_n=kw["cat_mean_n"][..., space], var_n=kw["cat_var_n"][..., space])
        for space in (0, 1)]
    for key, term in dict(photon_loss="photon_loss", photon_gain="photon_gain",
                          resonator_dephasing="storage_dephasing").items():
        assert got[f"{prefix}_{term}"] > 0
        assert np.isclose(got[f"{prefix}_{term}"],
                          _average(p_plus, p_minus, costs[0][key], costs[1][key]),
                          rtol=1e-12, atol=0), key


# ----------------------------------------------------------------------------
# Ingredients on a real FlexibleSweep (scqubits fixture)
# ----------------------------------------------------------------------------

@pytest.mark.integration
@requires_idx_add_sweep
def test_ingredients_on_two_point_grid():
    """H: ingredient keys have the grid shape on a 2-point T_W sweep.

    Uses the Ofek transmon with a small truncation; checks shapes, that the
    moments do not depend on anything but T_W along the swept axis, and that
    the budget evaluates to one number per grid point.
    """
    import chencrafts.cqed as cqed

    # Small truncation: the test checks plumbing and shapes, not convergence
    sim_para = dict(res_dim=8, qubit_dim=3, qubit_ncut=15, num_cpus=1,
                    evals_count=24)
    para = dict(
        E_osc_GHz=8.3005, g_GHz=0.1276, EJ_GHz=20.0948, EC_GHz=0.2684,
        ng=0.3, disp=np.sqrt(2), tau_m=400, M_ge=0.01, M_eg=0.01,
        Q_s=1.38e7, temp_a=0.094, temp_s=0.102, Q_cap=1.48e6, Q_ind=1e20,
        kappa_a_phi=6.95e-5, kappa_s_phi=1.03e-6,
        sigma_2_K_a=3 * 0.297 * 2 * np.pi, sigma_omega=2 * np.pi,
        tau_p_by_sigma=20, tau_p_eff_by_sigma=4,
    )
    swept = dict(T_W=np.array([9000.0, 18000.0]))
    system = bsq.ResonatorTransmon(sim_para=sim_para, para=para | {"T_W": 0})
    fs = cqed.FlexibleSweep(para=para, swept_para=swept, **system.dict,
                            **sim_para, lookup_scheme="LX")

    # Registration pipeline, in the documented order
    names = dict(res_trunc_dim=6, qubit_trunc_dim=3,
                 qubit_op_names=["n_operator", "proj_11"],
                 res_op_names=["a_m_adag", "adag_a"])
    bsq.batched_sweep_general(fs.sweep)
    bsq.batched_sweep_dressed_op(fs.sweep, **names)
    bsq.batched_sweep_jump_rates(fs.sweep, **names)
    bsq.batched_sweep_pulse(fs.sweep)
    bsq.batched_sweep_cat_budget_ingredients(fs.sweep)

    # Grid-shaped scalars and (grid, 2)-shaped moments
    assert np.shape(fs["t_pm"]) == (2,)
    # The cycle duration grows with the idle time by exactly the idle increment
    assert np.isclose(fs["t_cycle"][1] - fs["t_cycle"][0], 9000.0)
    assert np.shape(fs["cat_mean_n"]) == (2, 2)
    # A longer idle time shrinks the cat state, lowering <N> of the even error space
    assert fs["cat_mean_n"][1, 0] < fs["cat_mean_n"][0, 0]
    # Temporary kernel outputs are deleted after unpacking
    assert "cat_moments_raw" not in fs.sweep.keys()
    # A longer idle widens the range of the Kerr rotation after a photon loss
    assert np.shape(fs["cat_kerr_rotation_infidelity"]) == (2,)
    assert 0 < fs["cat_kerr_rotation_infidelity"][0] < fs["cat_kerr_rotation_infidelity"][1]

    # The default dict is the DRAG transmon: s_S = (4 - lambda^2) / alpha, up
    # to the small charge-offset element O_02 that the ladder-like
    # formula ignores
    assert np.allclose(
        fs["anc_stark_coeff"], (4 - fs["anc_lambda"] ** 2) / fs["anc_eta"],
        rtol=1e-2,
    )

    # One budget value per grid point, larger for the longer idle time
    total = np.asarray(cat_code_budget(**fs.full_dict()))
    assert total.shape == (2,)
    assert total[1] > total[0]


# ----------------------------------------------------------------------------
# I. Master-equation ingredients: argument without effect
# ----------------------------------------------------------------------------

@pytest.mark.unit
def test_me_ingredients_res_n_bar_warns_and_has_no_effect():
    """I: ``res_n_bar`` changes nothing that is returned, and passing it warns."""
    import warnings

    import scqubits as scq

    from qec_aeb import cat_real

    resonator = scq.Oscillator(E_osc=8.3, truncated_dim=6)
    transmon = scq.Transmon(EJ=20.0, EC=0.27, ng=0.3, ncut=12, truncated_dim=3)
    space = scq.HilbertSpace([resonator, transmon])
    space.add_interaction(
        g=0.1, op1=resonator.creation_operator, op2=transmon.n_operator,
        add_hc=True)
    space.generate_lookup()
    kwargs = dict(
        hilbertspace=space, res_mode_idx=0, qubit_mode_idx=1,
        res_truncated_dim=4, qubit_truncated_dim=2,
        dressed_indices=space["dressed_indices"][0],
        eigensys=(space["evals"][0], space["evecs"][0]),
        collapse_parameters={"res_decay": 1e-6},
    )

    # The default call is silent
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ham, c_ops, (evals, _), frame = cat_real.cavity_ancilla_me_ingredients(**kwargs)

    with pytest.warns(UserWarning, match="res_n_bar has no effect"):
        ham_n, c_ops_n, (evals_n, _), frame_n = (
            cat_real.cavity_ancilla_me_ingredients(**kwargs, res_n_bar=3))

    assert (ham - ham_n).norm() == 0
    assert (frame - frame_n).norm() == 0
    assert (c_ops[0] - c_ops_n[0]).norm() == 0
    assert np.array_equal(evals, evals_n)
