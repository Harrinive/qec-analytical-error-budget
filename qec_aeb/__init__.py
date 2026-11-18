r"""General QEC analytical error budgeting (AEB) and evolution-tree framework.

Provides the code-agnostic QEC evolution trees (actual evolution tree, logical
evolution tree, failure Markov chain), the eight-model reduction chain, the
step-by-step diamond-norm magnitudes and discrepancy bounds of Appendix B, and
the general error-budget container.
"""

from .budget import *
from .edge import *
from .graph import *
from .magnitudes import *
from .node import *
from .reduction_chain import *
from . import settings
from .settings import *
from . import utils
from .utils import (
    abstract_encoders,
    component_projector_map,
    effective_logical_process,
    joint_error_space_projector,
    process_block,
    process_block_dnorm,
    stacked_encoder,
    target_process_for_dnorm,
)
