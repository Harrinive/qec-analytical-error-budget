r"""
Error budgeting of a bosonic qubit: a cat-code qubit in a storage resonator
that is dispersively coupled to an ancilla (a transmon or a fluxonium).

The analytical formula for the infidelity of one QEC cycle is in
``error_budgeting``, the sweeps that compute its ingredients in
``batched_sweeps_recipe``, the ideal operators in ``cat_ideal``, the
dynamical maps of the real operations in ``cat_real``, the master equation
in ``cat_recipe``, and the tree of QEC trajectories in ``QEC_graph``.
"""

from .error_budgeting import *
from .systems import *
from .batched_sweeps_recipe import *

import qec_aeb.cat_real as cat_real
import qec_aeb.cat_recipe as cat_recipe
import qec_aeb.cat_ideal as cat_ideal
import qec_aeb.QEC_graph as QEC_graph

__all__ = [
    'cat_real',
    'cat_recipe',
    'cat_ideal',
    'QEC_graph',
]
