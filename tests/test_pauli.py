"""Tests for the Pauli tensor basis and r <-> rho conversions."""

from __future__ import annotations

import numpy as np

from bnn_qst.quantum.pauli import build_pauli_basis, check_orthogonality, rho_from_r, r_from_rho


def test_pauli_basis_shape():
    """The 3-qubit basis has 63 matrices of shape (8, 8)."""
    sigmas = build_pauli_basis(3)
    assert sigmas.shape == (63, 8, 8)


def test_orthogonality():
    """Tr(sigma_i sigma_j) = d * delta_ij holds for the 3-qubit basis."""
    sigmas = build_pauli_basis(3)
    check_orthogonality(sigmas, d=8, n_pairs=300)


def test_round_trip():
    """rho -> r -> rho is the identity (within 1e-8)."""
    sigmas = build_pauli_basis(3)
    d = 8
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = g @ g.conj().T
        rho /= np.trace(rho)
        rt = rho_from_r(r_from_rho(rho, sigmas), sigmas)
        assert np.linalg.norm(rho - rt, "fro") < 1e-8
