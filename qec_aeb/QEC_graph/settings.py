__all__ = [
    'IDEAL_STATE_THRESHOLD_0',
    'IDEAL_STATE_THRESHOLD_1',
    'IDEAL_STATE_OVERLAP_THRESHOLD',
    'ISSUE_WARNING',
    'CVX_SOLVER',
]

from typing import Literal
# An ideal operator applied to a pair of ideal logical states gives a pair
# that is dropped if the squared norm of either state is below a threshold:
# silently below the first, and with a warning between the first and the
# second, where the component is small but not negligible.
IDEAL_STATE_THRESHOLD_0 = 1e-10
IDEAL_STATE_THRESHOLD_1 = 2e-2
# A pair of ideal codewords with 1 - |<0_L|1_L>| below this value is one
# state, not a qubit (e.g. after more photon losses than the truncated
# resonator holds); such a pair is dropped from the ideal logical states.
IDEAL_STATE_OVERLAP_THRESHOLD = 1e-8
# whether the nodes, edges, and builders issue their warnings
ISSUE_WARNING = True
# cvxpy solver of the semidefinite program of utils.bound_fidelity
CVX_SOLVER: Literal["MOSEK", "SCS"] = "MOSEK"
