"""Tests for the quantum-state distance metrics."""

from __future__ import annotations

import numpy as np

from bnn_qst.quantum.metrics import (fidelity_single, frob_batch,
                                     trace_distance_single)


def test_fidelity_identity():
    """F(rho, rho) = 1 for any state."""
    rho = np.eye(8, dtype=complex) / 8
    assert abs(fidelity_single(rho, rho) - 1.0) < 1e-6


def test_fidelity_symmetry_and_range():
    """F is symmetric and in [0, 1]."""
    rng = np.random.default_rng(11)
    g1 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    g2 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    rho = g1 @ g1.conj().T; rho /= np.trace(rho)
    sigma = g2 @ g2.conj().T; sigma /= np.trace(sigma)
    f = fidelity_single(rho, sigma)
    assert 0.0 <= f <= 1.0
    assert abs(f - fidelity_single(sigma, rho)) < 1e-6


def test_trace_distance_identity():
    """D(rho, rho) = 0."""
    rho = np.eye(8, dtype=complex) / 8
    assert trace_distance_single(rho, rho) < 1e-6


def test_fuchs_van_de_graaf():
    """Fuchs-van de Graaf: 1 - sqrt(F) <= D <= sqrt(1 - F)."""
    rng = np.random.default_rng(23)
    g1 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    g2 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    rho = g1 @ g1.conj().T; rho /= np.trace(rho)
    sigma = g2 @ g2.conj().T; sigma /= np.trace(sigma)
    f = fidelity_single(rho, sigma)
    d = trace_distance_single(rho, sigma)
    assert 1 - np.sqrt(f) - 1e-6 <= d <= np.sqrt(max(1 - f, 0.0)) + 1e-6


def test_frobenius_batch_shape():
    """frob_batch returns one value per state."""
    rho = np.eye(8, dtype=complex)[None] / 8
    sigma = np.eye(8, dtype=complex)[None] / 8
    out = frob_batch(rho, sigma)
    assert out.shape == (1,)
    assert out[0] < 1e-6
