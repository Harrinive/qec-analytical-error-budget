import numpy as np
import pytest
import qutip as qt


@pytest.fixture
def sample_ec_el():
    """Typical fluxonium-like (EC, EL) in GHz."""
    return 1.0, 0.5


@pytest.fixture
def sample_thermal_args():
    """(freq GHz, temp K) for decoherence identity checks."""
    return 5.0, 0.01


@pytest.fixture
def qubit_pauli_ops():
    """Single-qubit Pauli operators for proc_repr tests."""
    return {
        "I": qt.qeye(2),
        "X": qt.sigmax(),
        "Y": qt.sigmay(),
        "Z": qt.sigmaz(),
    }
