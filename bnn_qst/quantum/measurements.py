"""Pauli-tensorial measurement projectors and frequency simulation.

Measurements are performed in the ``3**n`` tensor Pauli bases
``sigma_{a_1} x ... x sigma_{a_n}`` with ``a_k in {X, Y, Z}`` (Chapter 5,
``subsec:operatori-misura``). Each basis produces ``2**n`` joint eigenstate
projectors, and the Born rule gives the outcome probabilities
``P(k | B_j) = Tr(rho Pi_j^{(k)})`` (``eq:born-rule-basi``).

Empirical frequencies are obtained by multinomial sampling with ``N_shots``
shots per basis (``eq:multinomiale``).
"""

from __future__ import annotations

from itertools import product as iproduct

import numpy as np

# Eigenvectors of the single-qubit Pauli operators:
# axis a in {1=X, 2=Y, 3=Z}, outcome k in {0, 1}.
_EIG = {
    1: [np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2)],
    2: [np.array([1, 1j], dtype=complex) / np.sqrt(2),
        np.array([1, -1j], dtype=complex) / np.sqrt(2)],
    3: [np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex)],
}


def build_projectors(n: int):
    """Build the tensor Pauli measurement projectors.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    proj : numpy.ndarray
        Projectors of shape ``(3**n, 2**n, 2**n, 2**n)`` indexed as
        ``proj[j, k] = Pi_j^{(k)}``.
    bases : list of tuple
        List of ``3**n`` bases, each a tuple of axes in ``{1, 2, 3}``.
    """
    d = 2 ** n
    bases = list(iproduct([1, 2, 3], repeat=n))
    proj = np.zeros((len(bases), d, d, d), dtype=complex)
    for j, basis in enumerate(bases):
        for k in range(d):
            # k in binary MSB first: bit i = (k >> (n-1-i)) & 1
            k_bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
            ket = _EIG[basis[0]][k_bits[0]]
            for i in range(1, n):
                ket = np.kron(ket, _EIG[basis[i]][k_bits[i]])
            proj[j, k] = np.outer(ket, ket.conj())
    return proj, bases


def simulate_frequencies(rho_batch: np.ndarray, proj: np.ndarray, n_shots: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Simulate empirical measurement frequencies via multinomial sampling.

    Parameters
    ----------
    rho_batch : numpy.ndarray
        Density matrices of shape ``(N, d, d)``.
    proj : numpy.ndarray
        Projectors of shape ``(B, K, d, d)``.
    n_shots : int
        Number of shots per basis.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Frequencies of shape ``(N, B*K)`` (float32), each block of ``K``
        summing to 1.
    """
    n_bases, n_out = proj.shape[:2]
    n = rho_batch.shape[0]
    # probs[n, j, k] = Tr(rho_n @ proj[j, k])
    probs = np.einsum("nab,jkba->njk", rho_batch, proj).real.clip(0)
    probs /= probs.sum(axis=2, keepdims=True).clip(1e-15)
    freq = np.zeros((n, n_bases, n_out), dtype=np.float32)
    for i in range(n):
        for j in range(n_bases):
            counts = rng.multinomial(n_shots, probs[i, j])
            freq[i, j] = counts / n_shots
    return freq.reshape(n, -1)
