r"""Analytical error budget of one cat-code QEC cycle.

Defines ``ErrorContrib``, ``ErrorBudget``, and the 31 closed-form terms of
``cat_code_budget`` across the seven steps and physical error mechanisms of the
cat-code QEC cycle. Units: rates in 1/ns, angular frequencies in rad/ns,
durations in ns.
"""

__all__ = [
    "ErrorContrib",
    "ErrorBudget",
    "cat_code_budget",
    "cat_branch_probabilities",
    "cat_pulse_tilt_angles",
    "cat_pulse_coherent_error",
    "cat_pulse_decoherence",
    "cat_pulse_decoherence_by_space",
    "cat_resonator_decoherence",
    "cat_parity_mapping_idling_error",
]

import warnings

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List, Dict, Literal, Tuple


class ErrorContrib:
    r"""
    One term of an error budget: an infidelity contribution as a function of
    the budget ingredients, labeled by its step and its physical mechanism.

    Calling the object evaluates ``expression`` on the given ingredients.

    Parameters
    ----------
    name:
        Unique name of the term.
    expression:
        Function of keyword arguments that returns the infidelity
        contribution; it must accept (and ignore) unrelated keywords.
    step:
        Index of the QEC-cycle step that produces the contribution.
    mechanism:
        Physical mechanism responsible for the contribution, used to group
        terms across steps.
    type:
        Optional logical manifestation (``"X"``, ``"Y"``, ``"Z"``, ``"L"``).
    """

    def __init__(
        self,
        name: str,
        expression: Callable,
        step: int,
        mechanism: str | None = None,
        type: Literal["X", "Y", "Z", "L"] | None = None,
    ):
        r"""
        Store the term and its labels; see the class docstring.
        """
        self.name = name
        self.expression = expression
        self.step = step
        self.mechanism = mechanism
        self.type = type

    def __call__(self, *args, **kwargs) -> np.ndarray:
        r"""
        Infidelity contribution of the term for the given ingredients.

        Parameters
        ----------
        *args, **kwargs:
            Budget ingredients, passed to ``expression``.

        Returns
        -------
        float or np.ndarray
            Infidelity contribution; an array if the ingredients are arrays.
        """
        return self.expression(*args, **kwargs)


# ##############################################################################
class ErrorBudget:
    r"""
    A collection of named ``ErrorContrib`` terms that can be switched on and off.

    Calling the budget evaluates every enabled term on the same keyword
    ingredients. Each term is a NumPy expression, so arrays of ingredients
    (for example ``full_dict()`` of a swept ``FlexibleSweep``) evaluate the
    whole parameter grid at once. A new budget is empty; terms are added with
    ``add_contrib``.
    """

    def __init__(self):
        r"""
        Create an empty budget.
        """
        self.error_contribs: Dict[str, ErrorContrib] = {}
        self.contrib_enable_info: Dict[str, bool] = {}

    def __call__(
        self,
        organize_by: Literal["None", "name", "step", "mechanism", "type"] = "None",
        **kwargs,
    ) -> np.ndarray | Dict:
        r"""
        Total infidelity of the enabled terms, or its decomposition.

        Parameters
        ----------
        organize_by:
            ``"None"`` returns the total. ``"name"`` returns one entry per
            term; ``"step"``, ``"mechanism"`` and ``"type"`` return the sums
            over terms that share that label, in order of first appearance
            (``"step"`` is sorted by step index).
        **kwargs:
            Ingredients passed to every term.

        Returns
        -------
        float, np.ndarray or dict
            The total infidelity, or a dictionary of partial sums.
        """
        values = {
            name: contrib(**kwargs)
            for name, contrib in self.error_contribs.items()
            if self.contrib_enable_info[name]
        }

        if organize_by == "None":
            return sum(values.values())
        if organize_by == "name":
            return values
        if organize_by not in ("step", "mechanism", "type"):
            raise ValueError(f"Unknown organize_by {organize_by!r}.")

        # Group terms by the requested label and add them up
        grouped: Dict = {}
        for name, value in values.items():
            label = getattr(self.error_contribs[name], organize_by)
            grouped[label] = grouped.get(label, 0) + value
        if organize_by == "step":
            grouped = dict(sorted(grouped.items()))
        return grouped

    def __getitem__(
        self,
        contrib_name: str | List[str],
    ) -> "ErrorContrib | ErrorBudget":
        r"""
        One term, or a budget made of several terms.

        Parameters
        ----------
        contrib_name:
            A term name, or a list / tuple of names.

        Returns
        -------
        ErrorContrib or ErrorBudget
            The term for one name; for several names, a new budget that shares
            the term objects.
        """
        if isinstance(contrib_name, str):
            return self.error_contribs[contrib_name]
        new_budget = ErrorBudget()
        for name in contrib_name:
            new_budget.add_existed_contrib(self[name])
        return new_budget

    @property
    def num(self) -> int:
        r"""
        Number of terms.
        """
        return len(self.error_contribs)

    @property
    def enable_num(self) -> int:
        r"""
        Number of enabled terms.
        """
        return int(np.sum(list(self.contrib_enable_info.values())))

    @property
    def error_names(self) -> List[str]:
        r"""
        Names of all terms, in insertion order.
        """
        return list(self.error_contribs.keys())

    def add_contrib(
        self,
        name: str,
        expression: Callable,
        step: int,
        mechanism: str | None = None,
        type: Literal["X", "Y", "Z", "L"] | None = None,
    ) -> None:
        r"""
        Create a term and add it to the budget, enabled.

        Parameters
        ----------
        name, expression, step, mechanism, type:
            See ``ErrorContrib``.
        """
        self.error_contribs[name] = ErrorContrib(
            name, expression, step, mechanism, type,
        )
        self.contrib_enable_info[name] = True

    def add_existed_contrib(
        self,
        contrib: ErrorContrib,
    ) -> None:
        r"""
        Add an existing term to the budget, enabled.

        Parameters
        ----------
        contrib:
            The term.
        """
        self.error_contribs[contrib.name] = contrib
        self.contrib_enable_info[contrib.name] = True

    def remove_contrib(
        self,
        name: str,
    ) -> None:
        r"""
        Remove a term from the budget.

        Parameters
        ----------
        name:
            Name of the term.

        Raises
        ------
        KeyError
            If no term has this name.
        """
        del self.error_contribs[name]
        del self.contrib_enable_info[name]

    def disable_contrib(
        self,
        name: str,
    ) -> None:
        r"""
        Exclude a term from the evaluation without removing it.

        Parameters
        ----------
        name:
            Name of the term.
        """
        self.contrib_enable_info[name] = False

    def enable_contrib(
        self,
        name: str,
    ) -> None:
        r"""
        Include a disabled term in the evaluation again.

        Parameters
        ----------
        name:
            Name of the term.
        """
        self.contrib_enable_info[name] = True

    def bar_plot(
        self,
        ax: plt.Axes | None = None,
        organize_by: Literal["name", "step", "mechanism"] = "mechanism",
        **kwargs,
    ) -> plt.Axes:
        r"""
        Bar chart of the infidelity contributions at a single parameter point.

        Parameters
        ----------
        ax:
            Matplotlib axes; a new figure is created if None.
        organize_by:
            Grouping of the bars, as in ``__call__``.
        **kwargs:
            Ingredients; every contribution must reduce to a scalar.

        Returns
        -------
        matplotlib.axes.Axes
            The axes of the chart.
        """
        # One bar per group; float() fails loudly on a swept (array) input
        values = self(organize_by=organize_by, **kwargs)
        labels = [str(key) for key in values.keys()]
        heights = [float(np.squeeze(value)) for value in values.values()]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels, heights)
        ax.set_ylabel(r"Infidelity contribution")
        ax.tick_params(axis="x", labelrotation=45)
        return ax


# ##############################################################################
# Cat-code budget: helpers
# ##############################################################################

def _rate(jump_ij: np.ndarray, i: int, j: int) -> np.ndarray:
    r"""
    Rate ``jump_ij[..., i, j]`` of the ancilla jump :math:`|i\rangle\langle j|`
    (1/ns); zero if a level is beyond the ancilla truncation. The element
    ``[1, 1]`` is twice the pure-dephasing rate.
    """
    # A master equation with d ancilla levels has no jump into level d
    jump_ij = np.asarray(jump_ij)
    if max(i, j) >= jump_ij.shape[-1]:
        return np.zeros(jump_ij.shape[:-2])
    return jump_ij[..., i, j]


def _average_over_spaces(
    p_plus: np.ndarray,
    p_minus: np.ndarray,
    per_space: np.ndarray,
) -> np.ndarray:
    r"""
    Average :math:`p_+x_+ + p_-x_-` of a quantity over the two error spaces;
    the trailing axis of ``per_space`` holds :math:`x_+` (even, index 0) and
    :math:`x_-` (odd, index 1).
    """
    per_space = np.asarray(per_space)
    return p_plus * per_space[..., 0] + p_minus * per_space[..., 1]


def _stack_spaces(
    even: Dict[str, np.ndarray],
    odd: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    r"""
    Stack the terms of the even (index 0) and of the odd (index 1) error space
    along a new trailing axis, term by term.
    """
    # index 0 is the even error space and index 1 the odd one, the order of
    # the budget ingredients cat_mean_n, cat_var_n, ...
    return {name: np.stack([even[name], odd[name]], axis=-1) for name in even}


# ------------------------------------------------------------------------------
# Step 1: idle of duration t_w (error accumulation)
# Terms of Eq. (idle-analytical); nbar = |alpha|^2 is the cat size.

def _idle_ancilla_excitation(jump_ij, T_W, **kwargs):
    r"""
    Idle term of ancilla excitation,
    :math:`\gamma_{|1\rangle\langle 0|}t_{\mathrm w}`, with the idle duration
    :math:`t_{\mathrm w}` (``T_W``, ns); every excitation counts as a failure.
    """
    return _rate(jump_ij, 1, 0) * T_W


def _idle_storage_dephasing(jump_adag_a, disp, T_W, **kwargs):
    r"""
    Idle term of resonator dephasing,
    :math:`\gamma_{\hat a^\dagger\hat a}\bar n t_{\mathrm w}`, with
    :math:`\bar n=|\alpha|^2` and the cat amplitude :math:`\alpha` (``disp``);
    the photon-number variance is taken at its large-cat limit :math:`\bar n`.
    """
    return jump_adag_a * np.abs(disp) ** 2 * T_W


def _idle_photon_gain(jump_adag, jump_a, disp, T_W, **kwargs):
    r"""
    Idle term of photon gain,
    :math:`\gamma_{\hat a^\dagger}(\bar n+1)t_{\mathrm w}\,[1-e^{-\lambda}(\lambda+\lambda^2/2)]`
    with :math:`\lambda=\gamma_{\hat a}\bar n t_{\mathrm w}`: a gain is a
    failure unless exactly one or two photon losses of the same idle
    compensate it.
    """
    # Mean number of photon losses in the idle
    mean_loss_number = jump_a * np.abs(disp) ** 2 * T_W
    # Probability of exactly one or two losses, which compensate the gain
    compensated = np.exp(-mean_loss_number) * (
        mean_loss_number + mean_loss_number**2 / 2)
    return jump_adag * (np.abs(disp) ** 2 + 1) * T_W * (1 - compensated)


def _idle_multiple_photon_loss(jump_a, disp, T_W, **kwargs):
    r"""
    Idle term of two or more photon losses, :math:`1-e^{-\lambda}(1+\lambda)`
    with :math:`\lambda=\gamma_{\hat a}\bar n t_{\mathrm w}`, the mean of the
    Poissonian number of losses.
    """
    # Mean number of photon losses in the idle
    mean_loss_number = jump_a * np.abs(disp) ** 2 * T_W
    # Probability of two or more losses
    return 1 - np.exp(-mean_loss_number) * (1 + mean_loss_number)


def _idle_kerr_after_loss(
    jump_a, disp, T_W, cat_kerr_rotation_infidelity, **kwargs,
):
    r"""
    Idle term of the Kerr rotation after one photon loss at an unrecorded
    time, :math:`\lambda e^{-\lambda}\,\epsilon_K` with
    :math:`\lambda=\gamma_{\hat a}\bar n t_{\mathrm w}`: the probability of
    exactly one loss times the infidelity :math:`\epsilon_K` of the rotation
    on the odd error space, averaged over a loss time uniform over the idle
    (ingredient ``cat_kerr_rotation_infidelity``). For
    :math:`\lambda\ll1`, :math:`|K|t_{\mathrm w}\ll1` and a large cat it
    reduces to :math:`\tfrac13\gamma_{\hat a}\bar n^2K^2t_{\mathrm w}^3`, with
    the self-Kerr :math:`K` (``K_s``, rad/ns).
    """
    # Mean number of photon losses in the idle
    mean_loss_number = jump_a * np.abs(disp) ** 2 * T_W
    # Two or more losses are a failure already (_idle_multiple_photon_loss),
    # so the rotation costs only when exactly one photon is lost
    return (mean_loss_number * np.exp(-mean_loss_number)
            * cat_kerr_rotation_infidelity)


_IDLE_TERMS = (
    _idle_ancilla_excitation,
    _idle_storage_dephasing,
    _idle_photon_gain,
    _idle_multiple_photon_loss,
    _idle_kerr_after_loss,
)


def cat_branch_probabilities(
    jump_a: np.ndarray, T_W: np.ndarray, cat_logical_mean_n: np.ndarray, **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Probabilities :math:`p_\pm` that the idle leaves the logical information
    intact in the even (+) or in the odd (-) error space.

    To leading order, zero or one photon is lost during the idle, with
    probabilities in the ratio :math:`1:\lambda_{\mathrm{loss}}`, where
    :math:`\lambda_{\mathrm{loss}}=\gamma_{\hat a}\langle N\rangle_{\mathrm L}t_{\mathrm w}`
    and :math:`\langle N\rangle_{\mathrm L}` is the mean photon number of the
    logical subspace. The remaining idle mechanisms are taken to remove the
    same fraction :math:`1-F_{\mathrm{idle}}` from both outcomes:

    .. math::
        p_+ = \frac{F_{\mathrm{idle}}}{1+\lambda_{\mathrm{loss}}},
        \qquad
        p_- = \frac{\lambda_{\mathrm{loss}}\,F_{\mathrm{idle}}}{1+\lambda_{\mathrm{loss}}}.

    The idle fidelity :math:`F_{\mathrm{idle}}` is one minus the sum of the
    idle terms of the budget, bounded below by zero.

    Parameters
    ----------
    jump_a:
        Rate :math:`\gamma_{\hat a}` of the photon-loss jump :math:`\hat a`
        (1/ns).
    T_W:
        Idle duration :math:`t_{\mathrm w}` (ns).
    cat_logical_mean_n:
        :math:`\langle N\rangle_{\mathrm L}`.
    **kwargs:
        Ingredients of the idle terms (``jump_ij``, ``jump_adag``,
        ``jump_adag_a``, ``disp``, ``cat_kerr_rotation_infidelity``).

    Returns
    -------
    (p_plus, p_minus): tuple of np.ndarray
        :math:`p_+` and :math:`p_-`.
    """
    # Idle fidelity F_idle, from the analytical formula of the idle
    ingredients = dict(jump_a=jump_a, T_W=T_W) | kwargs
    # The idle terms estimate a small error probability, so they can sum to
    # more than one for a long idle, outside the range of the formula; a
    # probability is not negative, and without the bound every later step
    # would lower the total budget there
    survival = np.clip(1 - sum(term(**ingredients) for term in _IDLE_TERMS), 0, None)

    # Split F_idle between no photon loss and one photon loss
    loss_mean = jump_a * cat_logical_mean_n * T_W
    return survival / (1 + loss_mean), survival * loss_mean / (1 + loss_mean)


# ------------------------------------------------------------------------------
# Steps 2, 4, 7: ancilla pulses
# Coherent terms of Eq. (gate-analytical) and decoherence terms of
# Eq. (gate-decoherence).

def cat_pulse_tilt_angles(
    theta: float,
    duration: np.ndarray,
    chi_1: np.ndarray,
    stark_coeff: np.ndarray,
    shape_mu: np.ndarray,
    shape_nu: np.ndarray,
    retained_shape_nu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Stark-shift angle :math:`\varphi_{\mathrm S}` and residual dispersive-shift
    angle per photon :math:`\varphi_\chi` of an ancilla pulse.

    The drive is resonant with the ancilla transition in the photon-number
    subspace :math:`n_{\mathrm{ref}}`. To first order in the detunings, the
    tilt of the rotation in the photon-number subspace :math:`m` that the
    ideal operator of the pulse does not contain has the angle
    :math:`\varphi_{\mathrm S}+\varphi_\chi d_m`, where
    :math:`d_m=m-n_{\mathrm{ref}}` and

    .. math::
        \varphi_{\mathrm S} = \frac{s_{\mathrm S}\,\mu_\theta\theta^2}{4\,t_\theta},
        \qquad
        \varphi_\chi = \chi_1 t_\theta(\nu_\theta-\nu^{\mathrm{ret}}_\theta)

    (Eqs. gate-stark-shift-angle and gate-dispersive-shift-angle of
    ``cat-code-example.tex``); the symbols are those of the parameters. For a
    transmon with a DRAG quadrature and no chirp,
    :math:`s_{\mathrm S}=(4-\lambda^2)/\eta`, with the anharmonicity
    :math:`\eta` and the ratio :math:`\lambda` of the drive matrix elements of
    the 1-2 and 0-1 transitions.

    The envelope shape numbers are odd functions of :math:`\theta`: for a
    pulse of negative rotation angle, pass the negatives of the envelope
    shape numbers of :math:`|\theta|`; both angles then change sign.

    Parameters
    ----------
    theta:
        Rotation angle :math:`\theta` of the pulse (rad).
    duration:
        Pulse duration :math:`t_\theta` (ns).
    chi_1:
        Dispersive shift :math:`\chi_1` per photon: the local slope of the
        ancilla transition frequency with respect to the photon number
        (rad/ns).
    stark_coeff:
        Stark-shift coefficient :math:`s_{\mathrm S}` (ns): the Stark shift
        :math:`\delta_{\mathrm S}` of the ancilla transition divided by
        :math:`\Omega_{\mathrm R}^2/4`, where :math:`\Omega_{\mathrm R}` is the
        Rabi rate; see ``batched_sweeps_recipe.cat_stark_shift_coefficient``.
    shape_mu, shape_nu:
        Envelope shape numbers :math:`\mu_\theta` and :math:`\nu_\theta` of the
        pulse (dimensionless), from
        ``batched_sweeps_recipe.pulse_shape_numbers``.
    retained_shape_nu:
        Value :math:`\nu^{\mathrm{ret}}_\theta` of :math:`\nu_\theta` that the
        ideal operator of the pulse retains (dimensionless). For the
        :math:`\hat X_{\pm\pi/2}` pulses of the parity mapping it is the
        pulse's own :math:`\nu_\theta`, which gives :math:`\varphi_\chi=0`; for
        the reset it is 0.

    Returns
    -------
    (phi_s, phi_chi): tuple of np.ndarray
        :math:`\varphi_{\mathrm S}` (rad), the same in every photon-number
        subspace, and :math:`\varphi_\chi` (rad per photon).
    """
    # Stark-shift angle: the Stark shift s_S Omega_R^2 / 4 multiplied by
    # sin(theta(t)) and integrated over the pulse
    phi_s = stark_coeff * shape_mu * theta**2 / (4 * duration)
    # Residual dispersive-shift angle per photon: the part of the
    # first-order tilt that the ideal operator does not contain
    phi_chi = chi_1 * duration * (shape_nu - retained_shape_nu)
    return phi_s, phi_chi


def _gate_angles(
    kind: Literal["half", "pi"],
    chi_1, tau_p_eff, anc_stark_coeff,
    gate_mu_half, gate_nu_half, gate_mu_pi, gate_nu_pi,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Stark-shift angle :math:`\varphi_{\mathrm S}` and residual dispersive-shift
    angle per photon :math:`\varphi_\chi` of an :math:`\hat X_{\pm\pi/2}`
    pulse (``"half"``, duration :math:`t_g`, :math:`\varphi_\chi=0`) or of the
    reset (``"pi"``, duration :math:`2t_g`,
    :math:`\varphi_\chi=2\chi_1t_g\nu_\pi`), from the budget ingredients.
    """
    # The requested pulse of the cycle fixes the rotation angle, the
    # duration, the envelope shape numbers and the value of nu that its
    # ideal operator retains
    if kind == "half":
        return cat_pulse_tilt_angles(
            np.pi / 2, tau_p_eff, chi_1, anc_stark_coeff,
            gate_mu_half, gate_nu_half, retained_shape_nu=gate_nu_half)
    return cat_pulse_tilt_angles(
        np.pi, 2 * tau_p_eff, chi_1, anc_stark_coeff,
        gate_mu_pi, gate_nu_pi, retained_shape_nu=0.0)


def cat_pulse_coherent_error(
    kind: Literal["half", "pi"],
    phi_s: np.ndarray,
    phi_chi: np.ndarray,
    mean_d: np.ndarray,
    mean_d2: np.ndarray,
    var_n: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    r"""
    Coherent error of one ancilla pulse without frame changes, for one error
    space.

    The coherent error is the infidelity of the pulse when the ancilla does
    not decay. In the photon-number subspace :math:`m`, the tilt of angle
    :math:`\varphi_{\mathrm S}+\varphi_\chi d_m`, with the offset
    :math:`d_m=m-n_{\mathrm{ref}}` from the reference photon number, leaves
    the population :math:`(\varphi_{\mathrm S}+\varphi_\chi d_m)^2/4` outside
    the intended ancilla state. Averaged over the photon-number distribution
    of error space :math:`s`, and with the cost of the rotation of the cat
    added, the infidelity of the pulse is

    .. math::
        1-F_{\mathrm{gate},s} \simeq \tfrac14\varphi_{\mathrm S}^2
        + \tfrac14\varphi_\chi^2(\langle d^2\rangle+c_\theta\Delta N^2)
        + \tfrac12\varphi_{\mathrm S}\varphi_\chi\langle d\rangle ,

    (Eq. gate-analytical of ``cat-code-example.tex``), where
    :math:`F_{\mathrm{gate},s}` is the process fidelity of the pulse,
    :math:`\langle d\rangle` and :math:`\langle d^2\rangle` are the mean and
    the mean square of :math:`d_m`, and :math:`\Delta N^2` is the
    photon-number variance. The part :math:`c_\theta\Delta N^2` is the cost
    of the rotation of the cat by the phase that the ideal operator does not
    contain: :math:`c_{\pi/2}=1` for an :math:`\hat X_{\pm\pi/2}` pulse, and
    :math:`c_\pi=0` for the :math:`\pi` pulse of the reset with an envelope
    symmetric in time. Reversing the sign of both angles
    (:math:`\hat X_{-\pi/2}`) leaves every term unchanged. The terms do not
    include the frame changes of the parity mapping, which remove the
    Stark-shift term of a :math:`\pi/2` pulse.

    Parameters
    ----------
    kind:
        ``"half"`` for an :math:`\hat X_{\pm\pi/2}` pulse
        (:math:`c_\theta=1`), ``"pi"`` for the :math:`\pi` pulse of the reset
        (:math:`c_\theta=0`).
    phi_s, phi_chi:
        Stark-shift angle :math:`\varphi_{\mathrm S}` (rad) and residual
        dispersive-shift angle per photon :math:`\varphi_\chi` (rad per
        photon), as returned by ``cat_pulse_tilt_angles``.
    mean_d, mean_d2:
        Mean :math:`\langle d\rangle` and mean square
        :math:`\langle d^2\rangle` of the offset :math:`d=N-n_{\mathrm{ref}}`
        of the photon number :math:`N` from the reference photon number, in
        the maximally mixed state of the error space.
    var_n:
        Photon-number variance :math:`\Delta N^2` in the maximally mixed state
        of the error space. Required for ``kind="half"``; not used for
        ``kind="pi"``.

    Returns
    -------
    dict
        Infidelity of each term, under the keys ``"stark_shift"``,
        ``"dispersive_shift"`` and ``"cross"``.
    """
    # Moment that multiplies phi_chi^2 / 4: <d^2> from the tilt, plus the
    # photon-number variance for a pi/2 pulse, whose residual phase also
    # rotates the cat
    if kind == "half":
        if var_n is None:
            raise ValueError("A pi/2 pulse needs the photon-number variance var_n.")
        spread = np.asarray(mean_d2) + np.asarray(var_n)
    elif kind == "pi":
        spread = np.asarray(mean_d2)
    else:
        raise ValueError(f"Unknown pulse kind {kind!r}; use 'half' or 'pi'.")

    # (phi_S + phi_chi d)^2 / 4 averaged over the photon numbers, expanded
    # into its three terms
    return dict(
        stark_shift=phi_s**2 / 4,
        dispersive_shift=phi_chi**2 / 4 * spread,
        cross=phi_s * phi_chi / 2 * np.asarray(mean_d),
    )


def _half_pulse_coherent_error_by_space(**kw) -> Dict[str, np.ndarray]:
    r"""
    Coherent-error terms of an :math:`\hat X_{\pm\pi/2}` pulse without frame
    changes (``cat_pulse_coherent_error``), from the budget ingredients; the
    trailing axis holds the even (index 0) and the odd (index 1) error space.
    """
    # The angles are the same in both error spaces; only the moments differ
    phi_s, phi_chi = _gate_angles("half", **kw)
    costs = [
        cat_pulse_coherent_error(
            "half", phi_s, phi_chi,
            mean_d=np.asarray(kw["cat_mean_d"])[..., space],
            mean_d2=np.asarray(kw["cat_mean_d2"])[..., space],
            var_n=np.asarray(kw["cat_var_n"])[..., space],
        )
        for space in (0, 1)
    ]
    return _stack_spaces(*costs)


def _make_gate_terms(step: int) -> Dict[str, Callable]:
    r"""
    Build the three coherent-error terms of one :math:`\hat X_{\pm\pi/2}`
    pulse (Eq. gate-analytical of ``cat-code-example.tex``), averaged over
    the even (:math:`s=+`) and the odd (:math:`s=-`) error space with the
    probabilities :math:`p_\pm`:

    .. math::
        \Delta F(n) \simeq \sum_s p_s\bigl(1-F_{\mathrm{gate},s}\bigr)
        \simeq \sum_s p_s\Bigl[\tfrac14\varphi_{\mathrm S}^2
        + \tfrac14\varphi_\chi^2(\langle d^2\rangle_s+\Delta N^2_s)
        + \tfrac12\varphi_{\mathrm S}\varphi_\chi\langle d\rangle_s\Bigr].

    Here :math:`\Delta F(n)=F(n-1)-F(n)` is the fidelity loss at step
    :math:`n`, the step of this pulse, and :math:`1-F_{\mathrm{gate},s}` is
    the coherent error of the pulse in error space :math:`s`. Since
    :math:`\varphi_\chi=0` for these pulses, the dispersive-shift and cross
    terms are zero; the Stark-shift term is zero with the frame changes of
    the parity mapping (``frame_update`` = 1). ``step`` (2 or 4) is not used.
    The keys are ``"stark"``, ``"dispersive"`` and ``"cross"``.
    """

    # The three terms share the average over the two error spaces
    def averaged(name, **kw):
        r"""
        Term ``name`` of the coherent error of a :math:`\pi/2` pulse averaged over
        the two error spaces, :math:`\sum_sp_sx_s`.
        """
        p_plus, p_minus = cat_branch_probabilities(**kw)
        return _average_over_spaces(
            p_plus, p_minus, _half_pulse_coherent_error_by_space(**kw)[name])

    def stark(**kw):
        r"""
        Stark-shift term :math:`\sum_s p_s\,\varphi_{\mathrm S}^2/4` of a
        :math:`\pi/2` pulse; zero when the protocol applies the frame changes of
        the parity mapping (``frame_update`` = 1).
        """
        # the frame changes of the parity mapping remove this term
        uncompensated = 1 - np.asarray(kw.get("frame_update", 0.0))
        return uncompensated * averaged("stark_shift", **kw)

    def dispersive(**kw):
        r"""
        Dispersive-shift term
        :math:`\sum_s p_s\,\varphi_\chi^2(\langle d^2\rangle_s+\Delta N_s^2)/4` of
        a :math:`\pi/2` pulse.
        """
        return averaged("dispersive_shift", **kw)

    def cross(**kw):
        r"""
        Cross term
        :math:`\sum_s p_s\,\varphi_{\mathrm S}\varphi_\chi\langle d\rangle_s/2` of
        a :math:`\pi/2` pulse.
        """
        return averaged("cross", **kw)

    return dict(stark=stark, dispersive=dispersive, cross=cross)


def cat_pulse_decoherence(
    jump_ij: np.ndarray,
    duration: np.ndarray,
    mean_population: np.ndarray,
    mean_population_sq: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""
    First-order infidelity that each ancilla jump produces during one pulse.

    To zeroth order in the detunings, a jump :math:`\hat L` of rate
    :math:`\gamma` on the ancilla lowers the process fidelity at the rate
    :math:`\gamma(\langle\hat L^\dagger\hat L\rangle-|\langle\hat L\rangle|^2)`.
    With :math:`P_1(t)` the excited-state population of the resonant rotation
    and :math:`P_0=1-P_1`, the four ancilla jumps (relaxation, excitation,
    dephasing, leakage to level 2) cost

    .. math::
        \gamma_{|0\rangle\langle1|}\overline{P_1^2}\,t_\theta,\quad
        \gamma_{|1\rangle\langle0|}\overline{P_0^2}\,t_\theta,\quad
        \gamma_{|1\rangle\langle1|}\overline{P_0P_1}\,t_\theta,\quad
        \gamma_{|2\rangle\langle1|}\overline{P_1}\,t_\theta,

    where the overline is the time average over the pulse (Eq.
    gate-decoherence of ``cat-code-example.tex``).

    Parameters
    ----------
    jump_ij:
        Ancilla jump rates (1/ns) with trailing shape ``(d, d)`` for
        :math:`d` ancilla levels: ``jump_ij[..., i, j]`` is the rate of the
        jump :math:`|i\rangle\langle j|`, and ``jump_ij[..., 1, 1]`` is the
        coefficient of :math:`\mathcal D[|1\rangle\langle 1|]`, twice the
        pure-dephasing rate. With fewer than three levels the leakage cost is
        zero.
    duration:
        Pulse duration :math:`t_\theta` (ns).
    mean_population, mean_population_sq:
        :math:`\overline{P_1}` and :math:`\overline{P_1^2}` of the pulse
        (``batched_sweeps_recipe.pulse_population_averages``, with the two
        levels exchanged for an ancilla that starts in or ends in
        :math:`|1\rangle`).

    Returns
    -------
    dict
        Infidelity of each jump, under the keys ``"relaxation"``,
        ``"excitation"``, ``"dephasing"`` and ``"leakage"``.
    """
    # time averages of P_0^2 and P_0 P_1 from those of P_1 and P_1^2
    mean_ground_sq = 1 - 2 * mean_population + mean_population_sq
    mean_product = mean_population - mean_population_sq
    return dict(
        relaxation=_rate(jump_ij, 0, 1) * mean_population_sq * duration,
        excitation=_rate(jump_ij, 1, 0) * mean_ground_sq * duration,
        dephasing=_rate(jump_ij, 1, 1) * mean_product * duration,
        leakage=_rate(jump_ij, 2, 1) * mean_population * duration,
    )


def cat_pulse_decoherence_by_space(
    pulse: Literal["gate1", "gate2", "reset"],
    jump_ij: np.ndarray,
    duration: np.ndarray,
    mean_population: np.ndarray,
    mean_population_sq: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""
    Decoherence cost of one pulse of the QEC cycle in the even and in the odd
    error space.

    The cost is the first-order infidelity of the ancilla jumps
    (``cat_pulse_decoherence``) at the time averages :math:`\overline{P_1}`
    and :math:`\overline{P_1^2}` of the excited-state population along the
    path of the ancilla. The averages of a pulse that starts in
    :math:`|0\rangle` apply to the :math:`\hat X_{\pi/2}` pulse in both error
    spaces and to the :math:`\hat X_{-\pi/2}` pulse in the even error space.
    The :math:`\hat X_{-\pi/2}` pulse in the odd error space, which takes the
    ancilla on to :math:`|1\rangle`, and the reset, which starts in
    :math:`|1\rangle`, have the populations of the two levels exchanged:

    .. math::
        \overline{P_1}\to1-\overline{P_1},
        \qquad
        \overline{P_1^2}\to1-2\overline{P_1}+\overline{P_1^2}.

    The reset acts in the odd error space only; the value returned for the
    even error space is the same.

    Parameters
    ----------
    pulse:
        ``"gate1"`` (:math:`\hat X_{\pi/2}`), ``"gate2"``
        (:math:`\hat X_{-\pi/2}`) or ``"reset"`` (:math:`\hat X_\pi`).
    jump_ij:
        Ancilla jump rates (1/ns): ``jump_ij[..., i, j]`` is the rate of the
        jump :math:`|i\rangle\langle j|`, and the trailing shape is ``(d, d)``
        for :math:`d` ancilla levels. With fewer than three levels the
        leakage cost is zero.
    duration:
        Duration :math:`t_\theta` of this pulse (ns).
    mean_population, mean_population_sq:
        Time averages :math:`\overline{P_1}` and :math:`\overline{P_1^2}` over
        the pulse for an ancilla that starts in :math:`|0\rangle`
        (dimensionless), from
        ``batched_sweeps_recipe.pulse_population_averages`` at the rotation
        angle of the pulse, :math:`\pi/2` or :math:`\pi`.

    Returns
    -------
    dict
        Name of the ancilla jump (``"relaxation"``, ``"excitation"``,
        ``"dephasing"``, ``"leakage"``) to an array of infidelities whose
        trailing axis holds the even (index 0) and the odd (index 1) error
        space.
    """
    mean, mean_sq = np.asarray(mean_population), np.asarray(mean_population_sq)

    # the same averages with the populations of |0> and |1> exchanged
    exchanged = (1 - mean, 1 - 2 * mean + mean_sq)
    direct = (mean, mean_sq)
    # which of the two sets of averages applies in the (even, odd) error
    # space, for each pulse
    averages = dict(
        gate1=(direct, direct), gate2=(direct, exchanged), reset=(exchanged, exchanged),
    )[pulse]
    costs = [cat_pulse_decoherence(jump_ij, duration, *average) for average in averages]
    return _stack_spaces(*costs)


def _pulse_decoherence_by_space(
    pulse: Literal["gate1", "gate2", "reset"], **kw,
) -> Dict[str, np.ndarray]:
    r"""
    Decoherence cost of one pulse of the QEC cycle in the even and in the odd
    error space, from the budget ingredients; the duration is :math:`t_g`
    (``tau_p_eff``) for a :math:`\pi/2` pulse and :math:`2t_g` for the reset.
    """
    # The reset is a pi pulse of duration 2 t_g; the other two pulses are
    # pi/2 pulses of duration t_g
    if pulse == "reset":
        duration = 2 * np.asarray(kw["tau_p_eff"])
        mean, mean_sq = kw["gate_pop_pi"], kw["gate_pop_sq_pi"]
    else:
        duration = np.asarray(kw["tau_p_eff"])
        mean, mean_sq = kw["gate_pop_half"], kw["gate_pop_sq_half"]
    return cat_pulse_decoherence_by_space(pulse, kw["jump_ij"], duration, mean, mean_sq)


def _make_pulse_decoherence_terms(
    pulse: Literal["gate1", "gate2", "reset"],
) -> Dict[str, Callable]:
    r"""
    Build the ancilla-dephasing, the ancilla-relaxation and the
    ancilla-excitation term of one pulse, averaged over the error spaces with
    the probabilities :math:`p_\pm`; the reset has the odd error space only.
    Ancilla relaxation is the downward jump :math:`|0\rangle\langle1|`;
    ancilla excitation holds the two upward jumps,
    :math:`|1\rangle\langle0|` and the leakage :math:`|2\rangle\langle1|`.
    """

    def weighted(per_space, **kw):
        r"""
        Average of a per-error-space cost with the probabilities :math:`p_\pm`;
        for the reset only the odd error space enters.
        """
        p_plus, p_minus = cat_branch_probabilities(**kw)
        if pulse == "reset":
            # the reset acts only after the odd-parity syndrome
            p_plus = 0 * p_plus
        return _average_over_spaces(p_plus, p_minus, per_space)

    def ancilla_dephasing(**kw):
        r"""
        Ancilla dephasing during the pulse,
        :math:`\sum_sp_s\gamma_{|1\rangle\langle1|}\overline{P_0P_1}t_\theta`.
        """
        return weighted(_pulse_decoherence_by_space(pulse, **kw)["dephasing"], **kw)

    def ancilla_relaxation(**kw):
        r"""
        Ancilla relaxation during the pulse.
        """
        return weighted(_pulse_decoherence_by_space(pulse, **kw)["relaxation"], **kw)

    def ancilla_excitation(**kw):
        r"""
        Ancilla excitation and leakage to level 2 during the pulse.
        """
        costs = _pulse_decoherence_by_space(pulse, **kw)
        return weighted(costs["excitation"] + costs["leakage"], **kw)

    return dict(ancilla_dephasing=ancilla_dephasing, ancilla_relaxation=ancilla_relaxation,
                ancilla_excitation=ancilla_excitation)


def _reset_coherent_term(name: str, **kw) -> np.ndarray:
    r"""
    Term ``name`` of the coherent error of the conditional :math:`\hat X_\pi`
    reset, :math:`p_-x_-`: its value :math:`x_-` in the odd error space times
    the probability :math:`p_-` of that error space.
    """
    # Angles of the pi pulse, and moments of the odd error space (index 1),
    # the only error space in which the reset acts
    _, p_minus = cat_branch_probabilities(**kw)
    phi_s, phi_chi = _gate_angles("pi", **kw)
    costs = cat_pulse_coherent_error(
        "pi", phi_s, phi_chi,
        mean_d=np.asarray(kw["cat_mean_d"])[..., 1],
        mean_d2=np.asarray(kw["cat_mean_d2"])[..., 1],
    )
    return p_minus * costs[name]


def _reset_stark(**kw):
    r"""
    Stark-shift term of the conditional :math:`\hat X_\pi` reset,
    :math:`p_-\varphi_{\mathrm S,\pi}^2/4`; no frame change removes it.
    """
    return _reset_coherent_term("stark_shift", **kw)


def _reset_dispersive(**kw):
    r"""
    Dispersive-shift term of the reset,
    :math:`p_-\varphi_{\chi,\pi}^2\langle d^2\rangle_-/4`, for an envelope
    symmetric in time.
    """
    return _reset_coherent_term("dispersive_shift", **kw)


def _reset_cross(**kw):
    r"""
    Cross term of the reset,
    :math:`p_-\varphi_{\mathrm S,\pi}\varphi_{\chi,\pi}\langle d\rangle_-/2`.
    """
    return _reset_coherent_term("cross", **kw)


# ------------------------------------------------------------------------------
# Step 3: parity-mapping idling of duration t_pm
# Terms of Eq. (wait-analytical).

def cat_resonator_decoherence(
    jump_a: np.ndarray,
    jump_adag: np.ndarray,
    jump_adag_a: np.ndarray,
    duration: np.ndarray,
    mean_n: np.ndarray,
    var_n: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""
    Infidelity that photon loss, photon gain and resonator dephasing produce
    during one step after the idle, for one error space.

    The steps are the parity-mapping idling, the measurement and the feedback
    latency, in which every photon lost or gained counts as a failure. To
    first order in the rates, the three terms of a step of duration :math:`t`
    are

    .. math::
        \gamma_{\hat a}\langle N\rangle t,\qquad
        \gamma_{\hat a^\dagger}(\langle N\rangle+1)t,\qquad
        \gamma_{\hat a^\dagger\hat a}\Delta N^2t .

    The symbols are those of the parameters.

    Parameters
    ----------
    jump_a, jump_adag, jump_adag_a:
        Rates of the photon-loss jump :math:`\hat a`, of the photon-gain jump
        :math:`\hat a^\dagger` and of the resonator-dephasing jump
        :math:`\hat a^\dagger\hat a` (1/ns).
    duration:
        Duration :math:`t` of the step (ns); a scalar or an array.
    mean_n, var_n:
        Mean photon number :math:`\langle N\rangle` and photon-number variance
        :math:`\Delta N^2` in the maximally mixed state of the error space.

    Returns
    -------
    dict
        Infidelity of each term, under the keys ``"photon_loss"``,
        ``"photon_gain"`` and ``"resonator_dephasing"``.
    """
    # First order in the rates: the rate times <L^dag L> for loss and gain,
    # because every such jump is a failure, and the rate times the variance
    # of L = N for dephasing
    return dict(
        photon_loss=jump_a * mean_n * duration,
        photon_gain=jump_adag * (np.asarray(mean_n) + 1) * duration,
        resonator_dephasing=jump_adag_a * var_n * duration,
    )


def cat_parity_mapping_idling_error(
    jump_ij: np.ndarray,
    jump_a: np.ndarray,
    jump_adag: np.ndarray,
    jump_adag_a: np.ndarray,
    chi_prime: np.ndarray,
    duration: np.ndarray,
    mean_n: np.ndarray,
    var_n: np.ndarray,
    mean_d2: np.ndarray,
    mean_d4: np.ndarray,
    reference_d2: np.ndarray = 0.0,
) -> Dict[str, np.ndarray]:
    r"""
    Terms of the analytical formula for the infidelity of the parity-mapping
    idling, for one error space.

    During the idling of duration :math:`t` the ancilla is in an equal
    superposition of :math:`|0\rangle` and :math:`|1\rangle`. Each of the seven
    terms is the infidelity, relative to the ideal operator of the idling,
    that one group of physical error mechanisms produces (Eq. wait-analytical
    of ``cat-code-example.tex``).

    - Ancilla dephasing, the jump :math:`|1\rangle\langle1|`, flips the phase
      of the superposition with probability
      :math:`\tfrac12(1-e^{-\gamma_{|1\rangle\langle1|}t/2})\simeq
      \gamma_{|1\rangle\langle1|}t/4`, which is the term.
    - Ancilla relaxation (:math:`|0\rangle\langle1|`), excitation
      (:math:`|1\rangle\langle0|`) and leakage to level 2
      (:math:`|2\rangle\langle1|`) each start from a level that holds half of
      the population, and every such jump counts as a failure. The downward
      jump is the term of ancilla relaxation,
      :math:`\tfrac12\gamma_{|0\rangle\langle1|}t`, and the two upward jumps
      are the term of ancilla excitation,
      :math:`\tfrac12(\gamma_{|1\rangle\langle0|}+\gamma_{|2\rangle\langle1|})t`.
    - Photon loss, photon gain and resonator dephasing cost
      :math:`\gamma_{\hat a}\langle N\rangle t`,
      :math:`\gamma_{\hat a^\dagger}(\langle N\rangle+1)t` and
      :math:`\gamma_{\hat a^\dagger\hat a}\Delta N^2t`.
    - The qubit-state-dependent resonator non-linearity :math:`\chi_1'` adds
      :math:`\chi_1'd_m^2` to the dispersive shift :math:`\chi_1d_m` of the
      ancilla transition in the photon-number subspace :math:`m`, where
      :math:`d_m=m-n_{\mathrm{ref}}` is the offset from the reference photon
      number :math:`n_{\mathrm{ref}}`. The ideal operator retains the average
      of :math:`\chi_1'd_m^2` over the two ancilla levels, so the residual is
      the ancilla phase :math:`\chi_1'td_m^2` between :math:`|1\rangle` and
      :math:`|0\rangle`, of which a frame change at the end of the idling
      removes :math:`\chi_1't\,r` (``reference_d2``). The remaining phase
      :math:`\chi_1't(d_m^2-r)` costs

      .. math::
          \tfrac14(\chi_1't)^2\bigl\langle(d^2-r)^2\bigr\rangle
          = \tfrac14(\chi_1't)^2\bigl(\langle d^4\rangle
          -2r\langle d^2\rangle+r^2\bigr),

      where the averages are taken over the photon-number distribution of
      the error space. The expression holds when :math:`n_{\mathrm{ref}}`
      equals the mean photon number :math:`\bar n` at which :math:`\chi_1` is
      evaluated; otherwise a residual that is linear in :math:`d_m` remains,
      which the formula does not include.

    Parameters
    ----------
    jump_ij:
        Ancilla jump rates (1/ns): ``jump_ij[..., i, j]`` is the rate of the
        jump :math:`|i\rangle\langle j|`, and the trailing shape is ``(d, d)``
        for :math:`d` ancilla levels. With fewer than three levels the
        leakage rate is zero.
    jump_a, jump_adag, jump_adag_a:
        Rates of the photon-loss jump :math:`\hat a`, of the photon-gain jump
        :math:`\hat a^\dagger` and of the resonator-dephasing jump
        :math:`\hat a^\dagger\hat a` (1/ns).
    chi_prime:
        Qubit-state-dependent resonator non-linearity :math:`\chi_1'`
        (rad/ns), the ingredient ``chi_prime_sa``.
    duration:
        Duration :math:`t` of the idling (ns); a scalar or an array, e.g. a
        grid of durations.
    mean_n, var_n:
        Mean photon number :math:`\langle N\rangle` and photon-number variance
        :math:`\Delta N^2` in the maximally mixed state of the error space.
    mean_d2, mean_d4:
        Means :math:`\langle d^2\rangle` and :math:`\langle d^4\rangle` of the
        second and fourth powers of the offset :math:`d=N-n_{\mathrm{ref}}` of
        the photon number :math:`N` from the reference photon number, in the
        maximally mixed state of the error space.
    reference_d2:
        Reference :math:`r` of the frame change:
        :math:`\langle d^2\rangle_+` of the even error space when the protocol
        applies the frame change, 0 (default) when it does not.

    Returns
    -------
    dict
        Infidelity of each term, under the keys ``"ancilla_dephasing"``,
        ``"ancilla_relaxation"``, ``"ancilla_excitation"``, ``"photon_loss"``,
        ``"photon_gain"``, ``"resonator_dephasing"`` and ``"nonlinearity"``.
    """
    # Ancilla jumps out of the two levels of the superposition: the upward
    # jumps are the excitation and the leakage to level 2
    upward_jump_rate = _rate(jump_ij, 1, 0) + _rate(jump_ij, 2, 1)
    # <(d^2 - r)^2> expanded in the moments: mean square of the chi_1'
    # ancilla phase that remains after the frame change, in units of chi_1' t
    spread = mean_d4 - 2 * reference_d2 * mean_d2 + reference_d2 ** 2
    return dict(
        ancilla_dephasing=_rate(jump_ij, 1, 1) * duration / 4,
        ancilla_relaxation=_rate(jump_ij, 0, 1) * duration / 2,
        ancilla_excitation=upward_jump_rate * duration / 2,
        # photon loss, photon gain and resonator dephasing have the same
        # expressions in every step after the idle
        **cat_resonator_decoherence(
            jump_a, jump_adag, jump_adag_a, duration, mean_n, var_n),
        nonlinearity=(chi_prime * duration) ** 2 / 4 * spread,
    )


def _pm_error_by_space(**kw) -> Dict[str, np.ndarray]:
    r"""
    Terms of the parity-mapping idling of duration :math:`t_{\mathrm{pm}}`
    from the budget ingredients, the even (index 0) and the odd (index 1)
    error space on the trailing axis. The reference of the frame change is
    :math:`\langle d^2\rangle_+` for both error spaces (``frame_update`` = 1)
    or zero.
    """
    # reference of the frame change: <d^2> of the even error space, or zero
    d2 = np.asarray(kw["cat_mean_d2"])
    reference = np.asarray(kw.get("frame_update", 0.0)) * d2[..., 0]
    # the rates, the duration and the reference are the same in both error
    # spaces; only the moments differ
    costs = [
        cat_parity_mapping_idling_error(
            kw["jump_ij"], kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"],
            kw["chi_prime_sa"], kw["t_pm"],
            mean_n=np.asarray(kw["cat_mean_n"])[..., space],
            var_n=np.asarray(kw["cat_var_n"])[..., space],
            mean_d2=d2[..., space],
            mean_d4=np.asarray(kw["cat_mean_d4"])[..., space],
            reference_d2=reference,
        )
        for space in (0, 1)
    ]
    return _stack_spaces(*costs)


def _pm_term(name: str, **kw) -> np.ndarray:
    r"""
    Term ``name`` of the parity-mapping idling averaged over the two error
    spaces, :math:`\sum_sp_sx_s`, with the probability :math:`p_s` that the
    idle leaves the logical information in error space :math:`s`.
    """
    # the idling follows the idle in both error spaces, so both
    # probabilities enter
    p_plus, p_minus = cat_branch_probabilities(**kw)
    return _average_over_spaces(p_plus, p_minus, _pm_error_by_space(**kw)[name])


def _pm_ancilla_dephasing(**kw):
    r"""
    Ancilla dephasing during the parity-mapping idling,
    :math:`\sum_sp_s\,\gamma_{|1\rangle\langle1|}t_{\mathrm{pm}}/4`.
    """
    return _pm_term("ancilla_dephasing", **kw)


def _pm_ancilla_relaxation(**kw):
    r"""
    Ancilla relaxation during the parity-mapping idling,
    :math:`\sum_sp_s\,\tfrac12\gamma_{|0\rangle\langle1|}t_{\mathrm{pm}}`.
    """
    return _pm_term("ancilla_relaxation", **kw)


def _pm_ancilla_excitation(**kw):
    r"""
    Ancilla excitation and leakage to level 2 during the parity-mapping
    idling,
    :math:`\sum_sp_s\,\tfrac12(\gamma_{|1\rangle\langle0|}+\gamma_{|2\rangle\langle1|})t_{\mathrm{pm}}`.
    """
    return _pm_term("ancilla_excitation", **kw)


def _make_parity_change_terms(time_key: str) -> Dict[str, Callable]:
    r"""
    Build the photon-loss, photon-gain and resonator-dephasing terms of one
    step after the idle, averaged over the two error spaces with the
    probabilities :math:`p_\pm`; ``time_key`` names the ingredient that holds
    the duration of the step (ns).
    """

    # The three terms share the duration of the step, read from `time_key`,
    # and the average over the two error spaces
    def averaged(name, **kw):
        r"""
        Term ``name`` of ``cat_resonator_decoherence`` averaged over the two error
        spaces, :math:`\sum_sp_sx_s`.
        """
        p_plus, p_minus = cat_branch_probabilities(**kw)
        # the rates and the duration are the same in both error spaces;
        # only the moments differ
        costs = [
            cat_resonator_decoherence(
                kw["jump_a"], kw["jump_adag"], kw["jump_adag_a"], kw[time_key],
                mean_n=np.asarray(kw["cat_mean_n"])[..., space],
                var_n=np.asarray(kw["cat_var_n"])[..., space],
            )
            for space in (0, 1)
        ]
        return _average_over_spaces(p_plus, p_minus, _stack_spaces(*costs)[name])

    def photon_loss(**kw):
        r"""
        Photon loss during the step,
        :math:`\sum_sp_s\gamma_{\hat a}\langle N\rangle_st`.
        """
        return averaged("photon_loss", **kw)

    def photon_gain(**kw):
        r"""
        Photon gain during the step,
        :math:`\sum_sp_s\gamma_{\hat a^\dagger}(\langle N\rangle_s+1)t`.
        """
        return averaged("photon_gain", **kw)

    def resonator_dephasing(**kw):
        r"""
        Resonator dephasing during the step,
        :math:`\sum_sp_s\gamma_{\hat a^\dagger\hat a}\Delta N_s^2t`.
        """
        return averaged("resonator_dephasing", **kw)

    return dict(photon_loss=photon_loss, photon_gain=photon_gain,
                resonator_dephasing=resonator_dephasing)


def _pm_nonlinearity(**kw):
    r"""
    Term of the qubit-state-dependent resonator non-linearity :math:`\chi_1'`
    during the parity-mapping idling,
    :math:`\sum_sp_s\tfrac14(\chi_1't_{\mathrm{pm}})^2\bigl\langle(d^2-r)^2\bigr\rangle_s`,
    with :math:`r=\langle d^2\rangle_+` (``frame_update`` = 1) or :math:`r=0`.
    """
    return _pm_term("nonlinearity", **kw)


# ------------------------------------------------------------------------------
# Step 5: ancilla measurement = wait of duration tau_m + projective readout

def _meas_assignment(**kw):
    r"""
    Readout assignment error, :math:`p_+M_{ge}+p_-M_{eg}`, with the
    probability :math:`M_{ge}` of reading out :math:`|1\rangle` given the
    state :math:`|0\rangle`, and :math:`M_{eg}` the reverse.
    """
    p_plus, p_minus = cat_branch_probabilities(**kw)
    return p_plus * kw["M_ge"] + p_minus * kw["M_eg"]


def _make_ancilla_level_jump_terms(time_key: str) -> Dict[str, Callable]:
    r"""
    Build the ancilla-relaxation and the ancilla-excitation term of a wait
    during which the ancilla is in :math:`|0\rangle` (even error space) or
    :math:`|1\rangle` (odd). Every jump out of the level counts as a failure:
    the downward jump is the term of ancilla relaxation,
    :math:`p_-\gamma_{|0\rangle\langle1|}t`, and the upward jumps are the term
    of ancilla excitation,
    :math:`(p_+\gamma_{|1\rangle\langle0|}+p_-\gamma_{|2\rangle\langle1|})t`;
    ``time_key`` names the ingredient that holds the wait duration :math:`t`.
    """

    # The terms read the wait duration from `time_key`
    def ancilla_relaxation(**kw):
        r"""
        Ancilla relaxation out of :math:`|1\rangle`, the level of the odd
        error space, during the wait.
        """
        _, p_minus = cat_branch_probabilities(**kw)
        return p_minus * _rate(kw["jump_ij"], 0, 1) * kw[time_key]

    def ancilla_excitation(**kw):
        r"""
        Ancilla excitation out of the level of the error space during the
        wait: from :math:`|0\rangle` (even) and from :math:`|1\rangle` to
        level 2 (odd).
        """
        p_plus, p_minus = cat_branch_probabilities(**kw)
        jump_ij = kw["jump_ij"]
        rate = p_plus * _rate(jump_ij, 1, 0) + p_minus * _rate(jump_ij, 2, 1)
        return rate * kw[time_key]

    return dict(ancilla_relaxation=ancilla_relaxation, ancilla_excitation=ancilla_excitation)


# Ancilla jumps during the measurement wait of duration tau_m
_meas_ancilla_jumps = _make_ancilla_level_jump_terms("tau_m")


# ------------------------------------------------------------------------------
# Step 6: feedback latency = wait of duration tau_FD between the measurement
# and the reset

def _with_latency_default(term: Callable) -> Callable:
    r"""
    Wrap a feedback-latency term so that ingredients without ``tau_FD`` are
    evaluated at ``tau_FD = 0``, the cycle without feedback latency, with a
    warning.
    """

    # A missing duration means no latency
    def term_with_default(**kw):
        r"""
        The wrapped term, with ``tau_FD`` defaulting to zero.
        """
        # a forgotten key silently removes step 6, so the default is announced
        if "tau_FD" not in kw:
            warnings.warn(
                "The budget ingredients have no 'tau_FD': the feedback-latency "
                "terms are evaluated at zero duration. Pass tau_FD (ns), or 0 "
                "explicitly for a cycle without latency.",
                UserWarning, stacklevel=2,
            )
        return term(**({"tau_FD": 0.0} | kw))

    return term_with_default


# ##############################################################################
# Assemble the cat-code budget
# ##############################################################################

cat_code_budget = ErrorBudget()

# Step 1: idle
cat_code_budget.add_contrib(
    "idle_ancilla_excitation", _idle_ancilla_excitation, 1, "ancilla excitation")
cat_code_budget.add_contrib(
    "idle_storage_dephasing", _idle_storage_dephasing, 1, "storage dephasing")
cat_code_budget.add_contrib(
    "idle_photon_gain", _idle_photon_gain, 1, "photon gain")
cat_code_budget.add_contrib(
    "idle_multiple_photon_loss", _idle_multiple_photon_loss, 1, "photon loss")
cat_code_budget.add_contrib(
    "idle_kerr_after_loss", _idle_kerr_after_loss, 1, "photon loss")

# Steps 2 and 4: X_{+pi/2} and X_{-pi/2} pulses (same formula)
for _step, _prefix in [(2, "gate1"), (4, "gate2")]:
    _terms = _make_gate_terms(_step)
    cat_code_budget.add_contrib(
        f"{_prefix}_stark", _terms["stark"], _step, "drive Stark shift")
    cat_code_budget.add_contrib(
        f"{_prefix}_dispersive", _terms["dispersive"], _step, "dispersive shift")
    cat_code_budget.add_contrib(
        f"{_prefix}_cross", _terms["cross"], _step,
        "Stark-shift/dispersive-shift cross")
    _decoherence = _make_pulse_decoherence_terms(_prefix)
    cat_code_budget.add_contrib(
        f"{_prefix}_ancilla_dephasing", _decoherence["ancilla_dephasing"], _step,
        "ancilla dephasing")
    cat_code_budget.add_contrib(
        f"{_prefix}_ancilla_relaxation", _decoherence["ancilla_relaxation"], _step,
        "ancilla relaxation")
    cat_code_budget.add_contrib(
        f"{_prefix}_ancilla_excitation", _decoherence["ancilla_excitation"], _step,
        "ancilla excitation")

# Step 3: parity-mapping idling
_pm_waits = _make_parity_change_terms("t_pm")
cat_code_budget.add_contrib(
    "pm_ancilla_dephasing", _pm_ancilla_dephasing, 3, "ancilla dephasing")
cat_code_budget.add_contrib(
    "pm_ancilla_relaxation", _pm_ancilla_relaxation, 3, "ancilla relaxation")
cat_code_budget.add_contrib(
    "pm_ancilla_excitation", _pm_ancilla_excitation, 3, "ancilla excitation")
cat_code_budget.add_contrib(
    "pm_photon_loss", _pm_waits["photon_loss"], 3, "photon loss")
cat_code_budget.add_contrib(
    "pm_photon_gain", _pm_waits["photon_gain"], 3, "photon gain")
cat_code_budget.add_contrib(
    "pm_storage_dephasing", _pm_waits["resonator_dephasing"], 3,
    "storage dephasing")
cat_code_budget.add_contrib(
    "pm_nonlinearity", _pm_nonlinearity, 3, "qubit-state-dependent non-linearity")

# Step 5: measurement
_meas_waits = _make_parity_change_terms("tau_m")
cat_code_budget.add_contrib(
    "meas_assignment", _meas_assignment, 5, "readout assignment")
cat_code_budget.add_contrib(
    "meas_ancilla_relaxation", _meas_ancilla_jumps["ancilla_relaxation"], 5,
    "ancilla relaxation")
cat_code_budget.add_contrib(
    "meas_ancilla_excitation", _meas_ancilla_jumps["ancilla_excitation"], 5,
    "ancilla excitation")
cat_code_budget.add_contrib(
    "meas_photon_loss", _meas_waits["photon_loss"], 5, "photon loss")
cat_code_budget.add_contrib(
    "meas_photon_gain", _meas_waits["photon_gain"], 5, "photon gain")
cat_code_budget.add_contrib(
    "meas_storage_dephasing", _meas_waits["resonator_dephasing"], 5,
    "storage dephasing")

# Step 6: feedback latency (no ancilla-dephasing term: the ancilla is in
# an energy level, not in a superposition)
_latency_waits = _make_parity_change_terms("tau_FD")
_latency_ancilla_jumps = _make_ancilla_level_jump_terms("tau_FD")
cat_code_budget.add_contrib(
    "latency_ancilla_relaxation",
    _with_latency_default(_latency_ancilla_jumps["ancilla_relaxation"]), 6,
    "ancilla relaxation")
cat_code_budget.add_contrib(
    "latency_ancilla_excitation",
    _with_latency_default(_latency_ancilla_jumps["ancilla_excitation"]), 6,
    "ancilla excitation")
cat_code_budget.add_contrib(
    "latency_photon_loss", _with_latency_default(_latency_waits["photon_loss"]),
    6, "photon loss")
cat_code_budget.add_contrib(
    "latency_photon_gain", _with_latency_default(_latency_waits["photon_gain"]),
    6, "photon gain")
cat_code_budget.add_contrib(
    "latency_storage_dephasing",
    _with_latency_default(_latency_waits["resonator_dephasing"]), 6,
    "storage dephasing")

# Step 7: conditional reset (odd error space only)
cat_code_budget.add_contrib(
    "reset_stark", _reset_stark, 7, "drive Stark shift")
cat_code_budget.add_contrib(
    "reset_dispersive", _reset_dispersive, 7, "dispersive shift")
cat_code_budget.add_contrib(
    "reset_cross", _reset_cross, 7, "Stark-shift/dispersive-shift cross")
_reset_decoherence = _make_pulse_decoherence_terms("reset")
cat_code_budget.add_contrib(
    "reset_ancilla_dephasing", _reset_decoherence["ancilla_dephasing"], 7,
    "ancilla dephasing")
cat_code_budget.add_contrib(
    "reset_ancilla_relaxation", _reset_decoherence["ancilla_relaxation"], 7,
    "ancilla relaxation")
cat_code_budget.add_contrib(
    "reset_ancilla_excitation", _reset_decoherence["ancilla_excitation"], 7,
    "ancilla excitation")
