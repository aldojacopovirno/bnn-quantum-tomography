"""Pauli tensor basis and density-matrix <-> Pauli-parameter conversions.

Implements the Pauli decomposition used throughout the thesis (Chapter 2,
``sec:pauli-decomposizione``)::

    rho = (1/d) * (I + sum_i r_i sigma_i),     r_i = Tr(rho sigma_i)

for an n-qubit system with ``d = 2**n`` and ``4**n - 1`` non-trivial Pauli
matrices. The basis is orthonormal under the Hilbert-Schmidt inner product,
``Tr(sigma_i sigma_j) = d * delta_ij``.
"""

from __future__ import annotations

from itertools import product as iproduct

import numpy as np

# Single-qubit Pauli operators: index 0 = I, 1 = X, 2 = Y, 3 = Z.
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SINGLE_QUBIT_PAULI = [I2, SX, SY, SZ]


def build_pauli_basis(n: int) -> np.ndarray:
    """Build the non-trivial tensor Pauli basis for n qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(4**n - 1, 2**n, 2**n)`` of complex Pauli matrices,
        excluding the identity, in lexicographic order over ``{0,1,2,3}**n``.
    """
    out = []
    for combo in iproduct(range(4), repeat=n):
        if all(c == 0 for c in combo):
            continue
        m = _SINGLE_QUBIT_PAULI[combo[0]]
        for c in combo[1:]:
            m = np.kron(m, _SINGLE_QUBIT_PAULI[c])
        out.append(m)
    return np.array(out)


def check_orthogonality(sigmas: np.ndarray, d: int, n_pairs: int = 300,
                        rng_seed: int = 0) -> None:
    """Assert ``Tr(sigma_i sigma_j) = d * delta_ij``.

    Parameters
    ----------
    sigmas : numpy.ndarray
        Pauli basis of shape ``(K, d, d)``.
    d : int
        Hilbert-space dimension.
    n_pairs : int, optional
        Number of random off-diagonal pairs to test.
    rng_seed : int, optional
        Seed for the pair sampler.

    Raises
    ------
    AssertionError
        If the orthogonality relation is violated beyond ``1e-9``.
    """
    rng = np.random.default_rng(rng_seed)
    k = sigmas.shape[0]
    pairs = rng.integers(0, k, (min(n_pairs, k * (k + 1) // 2), 2))
    for i, j in pairs:
        val = np.trace(sigmas[i] @ sigmas[j]).real
        exp = d if i == j else 0
        assert abs(val - exp) < 1e-9, f"Tr(sigma_{i} sigma_{j})={val:.4f}!={exp}"
    for i in range(k):
        assert abs(np.trace(sigmas[i] @ sigmas[i]).real - d) < 1e-9


def r_from_rho(rho: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Extract Pauli parameters ``r_i = Re(Tr(rho sigma_i))``.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix of shape ``(..., d, d)``.
    sigmas : numpy.ndarray
        Pauli basis of shape ``(K, d, d)``.

    Returns
    -------
    numpy.ndarray
        Pauli parameters of shape ``(..., K)``.
    """
    if rho.ndim == 2:
        return np.einsum("ab,iab->i", rho, sigmas.conj()).real
    return np.einsum("nab,iab->ni", rho, sigmas.conj()).real


def rho_from_r(r: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Reconstruct the density matrix from Pauli parameters.

    Parameters
    ----------
    r : numpy.ndarray
        Pauli parameters of shape ``(..., K)``.
    sigmas : numpy.ndarray
        Pauli basis of shape ``(K, d, d)``.

    Returns
    -------
    numpy.ndarray
        Density matrix of shape ``(..., d, d)``.
    """
    d = sigmas.shape[-1]
    if r.ndim == 1:
        return (np.eye(d, dtype=complex) + np.einsum("i,iab->ab", r, sigmas)) / d
    return (np.eye(d, dtype=complex) + np.einsum("ni,iab->nab", r, sigmas)) / d
