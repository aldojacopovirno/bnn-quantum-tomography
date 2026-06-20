"""Synthetic quantum-state generators for the 3-qubit dataset.

States are sampled from two standard distributions (Chapter 5,
``subsec:dati-sintetici``):

* **Haar-random pure states** (``rho = |psi><psi|``), generated from normalized
  complex Gaussian vectors;
* **Hilbert-Schmidt mixed states** (``rho = G G^dagger / Tr(G G^dagger)``),
  generated from Ginibre matrices.

Notable states (GHZ, W, ``|000>``, ``I/8``) and Werner states
``rho_W(p) = p |GHZ><GHZ| + (1-p) I/8`` are also provided for qualitative
analysis.
"""

from __future__ import annotations

import numpy as np

from .pauli import r_from_rho


def haar_states(n_samples: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Haar-random pure states ``rho = |psi><psi|``.

    Parameters
    ----------
    n_samples : int
        Number of states to generate.
    d : int
        Hilbert-space dimension.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Density matrices of shape ``(n_samples, d, d)``.
    """
    z = rng.standard_normal((n_samples, d)) + 1j * rng.standard_normal((n_samples, d))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return np.einsum("ni,nj->nij", z, z.conj())


def hs_states(n_samples: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Hilbert-Schmidt mixed states ``rho = G G^dagger / Tr(G G^dagger)``.

    Parameters
    ----------
    n_samples : int
        Number of states to generate.
    d : int
        Hilbert-space dimension.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Density matrices of shape ``(n_samples, d, d)``.
    """
    g = rng.standard_normal((n_samples, d, d)) + 1j * rng.standard_normal((n_samples, d, d))
    gg = g @ g.conj().transpose(0, 2, 1)
    tr = np.trace(gg, axis1=1, axis2=2).real[:, None, None]
    return gg / (tr + 1e-30)


def generate_dataset(n_qubits: int, n_samples: int, pure_frac: float,
                     rng: np.random.Generator, sigmas: np.ndarray):
    """Generate a ``(rho, r)`` dataset mixing pure and mixed states.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_samples : int
        Total number of states.
    pure_frac : float
        Fraction of Haar-random pure states (the rest are HS mixed states).
    rng : numpy.random.Generator
        Random generator.
    sigmas : numpy.ndarray
        Pauli basis of shape ``(K, d, d)``.

    Returns
    -------
    rho_all : numpy.ndarray
        Density matrices of shape ``(n_samples, d, d)``.
    r_all : numpy.ndarray
        Pauli parameters of shape ``(n_samples, K)``.
    """
    d = 2 ** n_qubits
    n_pure = int(n_samples * pure_frac)
    n_mixed = n_samples - n_pure
    rho_p = haar_states(n_pure, d, rng)
    rho_m = hs_states(n_mixed, d, rng)
    rho_all = np.concatenate([rho_p, rho_m], axis=0)
    r_all = r_from_rho(rho_all, sigmas)
    idx = rng.permutation(n_samples)
    return rho_all[idx], r_all[idx]


def make_special_states() -> dict:
    """Build the notable 3-qubit states for qualitative analysis.

    Returns
    -------
    dict
        Mapping ``name -> rho`` (each ``rho`` is an ``(8, 8)`` complex array)
        for ``GHZ``, ``W``, ``|000>`` and ``I/8``.
    """
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    w = np.zeros(8, dtype=complex)
    w[1] = w[2] = w[4] = 1 / np.sqrt(3)
    z000 = np.zeros(8, dtype=complex)
    z000[0] = 1.0
    return {
        "GHZ": np.outer(ghz, ghz.conj()),
        "W": np.outer(w, w.conj()),
        r"$|000\rangle$": np.outer(z000, z000.conj()),
        r"$I/8$": np.eye(8, dtype=complex) / 8,
    }


def werner_state(rho_ghz: np.ndarray, p: float) -> np.ndarray:
    """Build a Werner state ``rho_W(p) = p |GHZ><GHZ| + (1-p) I/8``.

    Parameters
    ----------
    rho_ghz : numpy.ndarray
        GHZ density matrix of shape ``(8, 8)``.
    p : float
        Mixing parameter in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Werner density matrix of shape ``(8, 8)``.
    """
    return p * rho_ghz + (1 - p) * np.eye(8, dtype=complex) / 8
