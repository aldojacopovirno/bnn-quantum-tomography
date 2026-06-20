"""ELBO loss components for the BNN-Last variant.

The total loss optimized during training (Chapter 5, ``eq:loss-totale``) is::

    L(phi; B, t) = MSE(r_hat, r_true) + (beta(t) / N_train) * KL[q_phi || p]

where the MSE on the Pauli parameters is equivalent (up to a ``d`` factor) to
the Frobenius norm on the density matrix (``eq:frobenius-pauli``), and the KL
is collected automatically by TFP from the variational layers. The trace and
positivity penalty terms are identically zero thanks to the Cholesky
parametrization.
"""

from __future__ import annotations

import tensorflow as tf


def mse_on_pauli(r_pred: tf.Tensor, r_true: tf.Tensor) -> tf.Tensor:
    """Mean squared error on Pauli parameters, summed over components.

    Equivalent to ``d * ||rho_pred - rho_true||_F^2`` (``eq:frobenius-pauli``).

    Parameters
    ----------
    r_pred : tensorflow.Tensor
        Predicted Pauli parameters of shape ``(B, K)``.
    r_true : tensorflow.Tensor
        True Pauli parameters of shape ``(B, K)``.

    Returns
    -------
    tensorflow.Tensor
        Scalar MSE (sum over K, mean over the batch).
    """
    return tf.reduce_mean(tf.reduce_sum(tf.square(r_pred - r_true), axis=-1))


def kl_term(model) -> tf.Tensor:
    """Aggregate the KL divergence collected by the TFP variational layers.

    Parameters
    ----------
    model : Net
        Network exposing a ``losses`` property.

    Returns
    -------
    tensorflow.Tensor
        Scalar KL (unscaled; the ``beta / N_train`` weight is applied upstream).
    """
    losses = model.losses
    return tf.add_n(losses) if losses else tf.constant(0.0)


def kl_annealing_factor(epoch: int, warmup: int) -> float:
    """KL annealing schedule ``beta(t)`` (``eq:kl-annealing``).

    Piecewise-linear schedule over ``E_warm`` epochs:

    * ``beta = 0``           for ``t < E_warm / 2``;
    * linear ramp ``0 -> 1`` for ``E_warm / 2 <= t <= E_warm``;
    * ``beta = 1``           for ``t > E_warm``.

    Parameters
    ----------
    epoch : int
        Current epoch index ``t``.
    warmup : int
        Warmup length ``E_warm``.

    Returns
    -------
    float
        Annealing factor ``beta`` in ``[0, 1]``.
    """
    n_flat = warmup // 2
    n_ramp = warmup - n_flat
    if epoch < n_flat:
        return 0.0
    if epoch < warmup:
        return (epoch - n_flat) / max(n_ramp, 1)
    return 1.0
