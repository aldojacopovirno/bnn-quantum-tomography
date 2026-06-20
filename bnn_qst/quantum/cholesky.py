"""Cholesky parametrization pipeline (TensorFlow).

Implements the map ``Phi_Ch`` of Definition ``def:cholesky-map`` (Chapter 5)::

    rho(c) = L(c) L(c)^dagger / Tr(L(c) L(c)^dagger)

where ``L(c)`` is a lower-triangular matrix with positive diagonal
``L_ii = exp(c_ii)`` and complex strictly sub-diagonal entries. This guarantees,
*by construction*, the three physical constraints of a density matrix:

* Hermiticity: ``(L L^dagger)^dagger = L L^dagger``;
* Positive-definiteness (stronger than ``rho >= 0``): positive diagonal makes
  ``L`` invertible, hence ``L L^dagger > 0``;
* Unit trace: explicit normalization.

As a consequence the loss terms ``L_trace`` and ``L_pos`` are identically zero
(``eq:loss-totale``) and no eigenvalue computation is needed in the training
loop.

Parameter layout (``d**2`` real parameters)
-------------------------------------------
* ``[0:d]``           -> diagonal of ``L`` (passed through softplus + eps);
* ``[d:d+2*off]``     -> complex sub-diagonal entries, interleaved
  ``[re_0, im_0, re_1, im_1, ...]``, with ``off = d(d-1)/2``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from .pauli import r_from_rho


def make_cholesky_fns(
    d: int, sigmas_np: np.ndarray
) -> Tuple[Callable[[tf.Tensor], tf.Tensor], Callable[[tf.Tensor], tf.Tensor]]:
    """Build traced TensorFlow functions for the Cholesky pipeline.

    Parameters
    ----------
    d : int
        Hilbert-space dimension.
    sigmas_np : numpy.ndarray
        Pauli basis of shape ``(K, d, d)`` (complex).

    Returns
    -------
    tuple of callables
        ``(params_to_rho, params_to_r)`` where ``params_to_rho`` maps
        ``(B, d**2)`` float tensors to ``(B, d, d)`` complex density matrices,
        and ``params_to_r`` maps the same to ``(B, K)`` Pauli parameters.
    """
    off = d * (d - 1) // 2
    assert d * d == d + 2 * off

    # Indicator tensor: ind_tf[k, r, c] = 1 places the k-th sub-diagonal param.
    rows, cols = [], []
    for row in range(1, d):
        for col in range(row):
            rows.append(row)
            cols.append(col)
    ind = np.zeros((off, d, d), dtype=np.float32)
    for k, (r, c) in enumerate(zip(rows, cols)):
        ind[k, r, c] = 1.0
    ind_tf = tf.constant(ind)
    sig_tf = tf.constant(sigmas_np, dtype=tf.complex64)

    @tf.function
    def params_to_rho(params: tf.Tensor) -> tf.Tensor:
        """Map Cholesky parameters to normalized density matrices.

        Parameters
        ----------
        params : tensorflow.Tensor
            Float tensor of shape ``(B, d**2)``.

        Returns
        -------
        tensorflow.Tensor
            Complex tensor of shape ``(B, d, d)``, Hermitian, trace-one, PSD.
        """
        diag = tf.nn.softplus(params[..., :d]) + 1e-6  # positive diagonal
        off_p = params[..., d:]  # (B, 2*off)
        re = off_p[..., 0::2]  # (B, off)
        im = off_p[..., 1::2]  # (B, off)
        lr = tf.linalg.diag(diag) + tf.einsum("...k,krc->...rc", re, ind_tf)
        li = tf.einsum("...k,krc->...rc", im, ind_tf)
        l = tf.cast(lr, tf.complex64) + tf.cast(li, tf.complex64) * 1j
        ll = l @ tf.linalg.adjoint(l)
        tr = tf.math.real(tf.linalg.trace(ll))[..., None, None]
        return ll / tf.cast(tr + 1e-10, ll.dtype)

    @tf.function
    def params_to_r(params: tf.Tensor) -> tf.Tensor:
        """Map Cholesky parameters to Pauli parameters ``r_i = Tr(rho sigma_i)``.

        Parameters
        ----------
        params : tensorflow.Tensor
            Float tensor of shape ``(B, d**2)``.

        Returns
        -------
        tensorflow.Tensor
            Float tensor of shape ``(B, K)``.
        """
        rho = params_to_rho(params)
        return tf.math.real(tf.einsum("nab,iba->ni", rho, sig_tf))

    return params_to_rho, params_to_r


def check_physicality(params_to_rho: Callable[[tf.Tensor], tf.Tensor],
                      d: int, n: int = 50, rng_seed: int = 2) -> None:
    """Assert that random Cholesky parameters yield physical density matrices.

    Parameters
    ----------
    params_to_rho : callable
        Traced function produced by :func:`make_cholesky_fns`.
    d : int
        Hilbert-space dimension.
    n : int, optional
        Number of random parameter vectors to test.
    rng_seed : int, optional
        Seed for the parameter sampler.

    Raises
    ------
    AssertionError
        If hermiticity, unit trace or positive semi-definiteness fail.
    """
    rng = np.random.default_rng(rng_seed)
    p = rng.standard_normal((n, d * d)).astype(np.float32)
    rho = params_to_rho(tf.constant(p)).numpy()
    assert np.allclose(rho, rho.conj().transpose(0, 2, 1), atol=1e-5), "Not Hermitian"
    assert np.allclose(np.trace(rho, axis1=1, axis2=2).real, 1.0, atol=1e-5), "Tr != 1"
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() >= -1e-5, f"Negative eigenvalue: {eigs.min():.2e}"


def r_from_rho_np(rho: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """NumPy convenience re-export of :func:`pauli.r_from_rho`."""
    return r_from_rho(rho, sigmas)
