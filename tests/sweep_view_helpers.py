"""Pytest skip markers for ParameterSweep features."""

from __future__ import annotations

import pytest
from scqubits.core.param_sweep import ParameterSweep

HAS_STORE_DATA = hasattr(ParameterSweep, "store_data")
requires_store_data = pytest.mark.skipif(
    not HAS_STORE_DATA,
    reason="ParameterSweep.store_data is required",
)
requires_idx_add_sweep = pytest.mark.skipif(
    not HAS_STORE_DATA,
    reason="idx-style ParameterSweep.add_sweep is required",
)
