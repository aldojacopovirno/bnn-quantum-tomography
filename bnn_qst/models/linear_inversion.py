"""Linear Inversion baseline estimator.

Linear Inversion (Chapter 3, ``def:li-stimatore``) exploits the linear relation
between the observed frequencies and the Pauli parameters. For the Pauli
measurement bases the system ``A r = b`` is solved in least-squares sense via
the pseudo-inverse, with ``b = d * f - 1``.

LI is unbiased but does **not** guarantee ``rho >= 0``: on the 3-qubit test set
the violation rate exceeds 77% at ``N_shots = 5000``.
"""

from __future__ import annotations

import numpy as np

from ..quantum.pauli import rho_from_r


def linear_inversion(freq_batch: np.ndarray, proj: np.ndarray, sigmas: np.ndarray):
    """Estimate Pauli parameters and density matrices by Linear Inversion.

    Parameters
    ----------
    freq_batch : numpy.ndarray
        Observed frequencies of shape ``(N, B*K)`` (raw, not centered).
    proj : numpy.ndarray
        Measurement projectors of shape ``(B, K, d, d)``.
    sigmas : numpy.ndarray
        Pauli basis of shape ``(K_sig, d, d)``.

    Returns
    -------
    r_hat : numpy.ndarray
        Estimated Pauli parameters of shape ``(N, K_sig)``.
    rho_hat : numpy.ndarray
        Estimated density matrices of shape ``(N, d, d)``.
    """
    n_bases, n_out, d, _ = proj.shape
    k = sigmas.shape[0]
    a = np.einsum("iab,jkba->jki", sigmas, proj).real  # (B, K, K_sig)
    a = a.reshape(n_bases * n_out, k)
    b = d * freq_batch - 1.0  # (N, B*K)
    a_pinv = np.linalg.pinv(a)
    r_hat = (a_pinv @ b.T).T  # (N, K)
    rho_hat = np.array([rho_from_r(r, sigmas) for r in r_hat])
    return r_hat, rho_hat
