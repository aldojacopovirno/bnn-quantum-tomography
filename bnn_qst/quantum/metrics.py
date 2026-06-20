"""Quantum-state distance metrics.

Implements the standard quantum-information metrics used for evaluation
(Chapter 4, ``sec:metriche-accuratezza``; EXPERIMENT.md Section 9.1):

* **Fidelity** ``F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2``;
* **Trace distance** ``D(rho, sigma) = 0.5 Tr|rho - sigma|``;
* **Frobenius norm** ``||rho - sigma||_F`` (equivalent to
  ``||r - s|| / sqrt(d)`` by ``eq:frobenius-pauli``);
* **Physical-violation rate** (fraction of states with a negative eigenvalue).
"""

from __future__ import annotations

import numpy as np


def fidelity_single(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the fidelity ``F(rho, sigma)`` between two states.

    Uses the spectral decomposition of ``sqrt(rho) sigma sqrt(rho)``.

    Parameters
    ----------
    rho, sigma : numpy.ndarray
        Density matrices of shape ``(d, d)``.

    Returns
    -------
    float
        Fidelity in ``[0, 1]`` (``1`` for identical states).
    """
    d, u = np.linalg.eigh(rho)
    d = d.clip(0)
    sqrt_rho = u @ np.diag(np.sqrt(d)) @ u.conj().T
    inner = sqrt_rho @ sigma @ sqrt_rho
    eig_inner = np.linalg.eigvalsh(inner).clip(0)
    return float(np.clip(np.sum(np.sqrt(eig_inner)) ** 2, 0, 1))


def trace_distance_single(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the trace distance ``D(rho, sigma) = 0.5 Tr|rho - sigma|``.

    Parameters
    ----------
    rho, sigma : numpy.ndarray
        Density matrices of shape ``(d, d)``.

    Returns
    -------
    float
        Trace distance in ``[0, 1]``.
    """
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))


def fidelity_batch(rho_pred: np.ndarray, rho_true: np.ndarray) -> np.ndarray:
    """Vectorized (loop) fidelity over a batch.

    Parameters
    ----------
    rho_pred, rho_true : numpy.ndarray
        Density matrices of shape ``(N, d, d)``.

    Returns
    -------
    numpy.ndarray
        Fidelities of shape ``(N,)``.
    """
    return np.array([fidelity_single(rho_pred[i], rho_true[i])
                     for i in range(len(rho_pred))])


def td_batch(rho_pred: np.ndarray, rho_true: np.ndarray) -> np.ndarray:
    """Vectorized (loop) trace distance over a batch.

    Parameters
    ----------
    rho_pred, rho_true : numpy.ndarray
        Density matrices of shape ``(N, d, d)``.

    Returns
    -------
    numpy.ndarray
        Trace distances of shape ``(N,)``.
    """
    return np.array([trace_distance_single(rho_pred[i], rho_true[i])
                     for i in range(len(rho_pred))])


def frob_batch(rho_pred: np.ndarray, rho_true: np.ndarray) -> np.ndarray:
    """Compute the Frobenius norm ``||rho - sigma||_F`` over a batch.

    Parameters
    ----------
    rho_pred, rho_true : numpy.ndarray
        Density matrices of shape ``(N, d, d)``.

    Returns
    -------
    numpy.ndarray
        Frobenius norms of shape ``(N,)``.
    """
    diff = rho_pred - rho_true
    return np.sqrt(np.einsum("nab,nab->n", diff, diff.conj()).real)


def phys_violations(rho_batch: np.ndarray, tol: float = -1e-4) -> float:
    """Fraction of states with at least one eigenvalue below ``tol``.

    Parameters
    ----------
    rho_batch : numpy.ndarray
        Density matrices of shape ``(N, d, d)``.
    tol : float, optional
        Tolerance below which an eigenvalue counts as a violation.

    Returns
    -------
    float
        Fraction of physically invalid states in ``[0, 1]``.
    """
    eigs = np.linalg.eigvalsh(rho_batch)
    return float((eigs.min(axis=1) < tol).mean())
