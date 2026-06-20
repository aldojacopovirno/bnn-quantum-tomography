"""Maximum Likelihood Estimation via the iterative R-rho-R algorithm of Hradil.

The R-rho-R iteration (Chapter 3, ``def:hradil-operatore``,
``eq:hradil-iterazione``) operates on ``d x d`` density matrices::

    R(rho) = sum_{j,k} (f_j^{(k)} / Tr(rho Pi_j^{(k)})) Pi_j^{(k)}
    rho_{t+1} = R(rho_t) rho_t R(rho_t) / Tr[R(rho_t) rho_t R(rho_t)]

MLE guarantees physical validity (``rho >= 0``, ``Tr = 1``) by construction and
is asymptotically efficient, at the cost of an iterative optimization whose
per-state complexity grows with the number of bases and outcomes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm.auto import tqdm


def mle_single(freq_jk: np.ndarray, proj: np.ndarray,
               max_iter: int = 2000, tol: float = 1e-8,
               eps: float = 1e-15) -> np.ndarray:
    """Run the R-rho-R iteration for a single state.

    Parameters
    ----------
    freq_jk : numpy.ndarray
        Frequencies for one state, shape ``(B, K)``.
    proj : numpy.ndarray
        Measurement projectors of shape ``(B, K, d, d)``.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance on ``||rho_new - rho||``.
    eps : float, optional
        Floor for the denominator to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Estimated density matrix of shape ``(d, d)``.
    """
    d = proj.shape[2]
    rho = np.eye(d, dtype=complex) / d
    for _ in range(max_iter):
        denom = np.einsum("ab,jkba->jk", rho, proj).real.clip(eps)  # (B, K)
        r_op = np.einsum("jk,jkab->ab", freq_jk / denom, proj)  # (d, d)
        rho_new = r_op @ rho @ r_op
        rho_new /= rho_new.trace().real
        if np.linalg.norm(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new
    return rho


def mle_batch(freq_batch: np.ndarray, proj: np.ndarray,
              max_iter: int = 2000, tol: float = 1e-8,
              subset: Optional[int] = None) -> np.ndarray:
    """Run MLE over a batch of states (optionally on a subset).

    Parameters
    ----------
    freq_batch : numpy.ndarray
        Frequencies of shape ``(N, B*K)`` (raw, not centered).
    proj : numpy.ndarray
        Measurement projectors of shape ``(B, K, d, d)``.
    max_iter : int, optional
        Maximum number of iterations per state.
    tol : float, optional
        Convergence tolerance.
    subset : int, optional
        If given, evaluate only the first ``subset`` states.

    Returns
    -------
    numpy.ndarray
        Estimated density matrices of shape ``(N_sub, d, d)``.
    """
    n = freq_batch.shape[0]
    n_bases, n_out = proj.shape[:2]
    freq_3d = freq_batch.reshape(n, n_bases, n_out)
    if subset is not None:
        n = min(n, subset)
    rho_out = []
    for i in tqdm(range(n), desc="MLE", leave=False):
        rho_out.append(mle_single(freq_3d[i], proj, max_iter, tol))
    return np.stack(rho_out)
