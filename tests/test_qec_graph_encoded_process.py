r"""Encoded-process identities of ``qec_aeb.QEC_graph.utils``.

A node of the actual evolution tree stores its process restricted to the
encoded input, :math:`\mathcal P\circ\mathcal S` with
:math:`\mathcal S(\cdot)=S\cdot S^\dagger` and :math:`S=\sum_a E_a\Pi_a^\dagger`.
These tests check that the blocks computed from it with the inclusion maps
:math:`\Pi_a` equal the blocks computed from the full process with the
physical encoders :math:`E_a`.
"""

import numpy as np
import pytest
import qutip as qt

from qec_aeb.QEC_graph.utils import (
    abstract_encoders,
    effective_logical_process,
    leakage_process,
    stacked_encoder,
)


def _random_encoders(dim: int, num_subspaces: int, seed: int) -> np.ndarray:
    r"""Isometric encoders of ``num_subspaces`` orthogonal two-dimensional codes.

    Parameters
    ----------
    dim : int
        Physical dimension; must be at least ``2 * num_subspaces``.
    num_subspaces : int
        Number of error spaces.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Object array of Qobj encoders of dims ``[[dim], [2]]``.
    """
    # orthonormal columns from a QR decomposition, split into pairs
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(mat)
    encs = np.ndarray(num_subspaces, dtype=qt.Qobj)
    for a in range(num_subspaces):
        encs[a] = qt.Qobj(q[:, 2 * a: 2 * a + 2], dims=[[dim], [2]])
    return encs


def _random_process(dim: int, seed: int) -> qt.Qobj:
    r"""A random CPTP map on a ``dim``-dimensional space.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    seed : int
        Random seed.

    Returns
    -------
    qt.Qobj
        Superoperator in the ``super`` representation.
    """
    return qt.to_super(qt.rand_super_bcsz(dim, seed=seed))


@pytest.mark.unit
def test_stacked_encoder_blocks_are_the_encoders():
    encs = _random_encoders(8, 2, seed=1)
    S = stacked_encoder(encs)
    # S Pi_a = E_a: column block a of S is the encoder of error space a
    for a, pi in enumerate(abstract_encoders(2)):
        assert np.allclose((S * pi).full(), encs[a].full())


@pytest.mark.unit
@pytest.mark.parametrize("num_init", [1, 2])
def test_effective_logical_process_from_encoded_process(num_init):
    dim = 6
    init_encs = _random_encoders(dim, num_init, seed=2)
    final_encs = _random_encoders(dim, 2, seed=3)
    process = _random_process(dim, seed=4)
    S = stacked_encoder(init_encs)
    encoded = process * qt.sprepost(S, S.dag())

    # the full process with physical encoders is the reference
    ref = effective_logical_process(process, init_encs, final_encs)
    new = effective_logical_process(encoded, abstract_encoders(num_init), final_encs)
    assert ref.shape == new.shape
    for r, n in zip(ref.ravel(), new.ravel()):
        assert np.allclose(r.full(), n.full(), atol=1e-12)


@pytest.mark.unit
def test_truncated_leakage_process_from_encoded_process():
    dim = 6
    init_encs = _random_encoders(dim, 1, seed=5)
    final_encs = _random_encoders(dim, 1, seed=6)
    process = _random_process(dim, seed=7)
    S = stacked_encoder(init_encs)
    encoded = process * qt.sprepost(S, S.dag())

    ref = leakage_process(process, init_encs, final_encs, truncate_init_space=True)
    new = leakage_process(
        encoded, abstract_encoders(1), final_encs, truncate_init_space=True
    )
    for r, n in zip(ref.ravel(), new.ravel()):
        assert np.allclose(r.full(), n.full(), atol=1e-12)
