r"""General analytical error budget container (``ErrorContrib``, ``ErrorBudget``)."""

from __future__ import annotations

__all__ = [
    "ErrorContrib",
    "ErrorBudget",
]

from typing import Callable, Dict, List, Literal
import numpy as np


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
