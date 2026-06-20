"""Neural network builders: deterministic NN and three BNN variants.

All networks share the architecture of Chapter 5 (``sec:architettura-rete``):
three hidden layers ``[256, 256, 128]`` with ``tanh`` activation (``rmk:scelta-tanh``)
and a linear output layer of dimension ``d**2 = 64`` (Cholesky parameters).

Three Bayesian variants are provided (ablation ``tab:ablation-varianti``):

* **A — DenseReparameterization** on every layer (fully Bayesian);
* **B — DenseFlipout** on every layer (fully Bayesian);
* **C — Last-layer Bayesian** (deterministic body + single DenseFlipout head),
  the *main* variant of the thesis (``def:bnn-last``).

The mean-field posterior is initialized near-deterministically via
``sigma_init = -4.6`` (``softplus(-4.6) ~= 0.01``), and the prior is
``N(0, prior_std^2 I)`` with ``prior_std = 1.0``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfpl = tfp.layers

# Near-deterministic posterior start: softplus(-4.6) ~= 0.01 (def:bnn-last).
SIGMA_INIT = -4.6


def _prior_fn(prior_std: float):
    """Return a TFP ``kernel_prior_fn`` for ``N(0, prior_std^2 I)``.

    Parameters
    ----------
    prior_std : float
        Prior standard deviation.

    Returns
    -------
    callable
        Prior factory compatible with TFP variational layers.
    """

    def prior(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn
        dist = tfp.distributions.Normal(
            loc=tf.zeros(shape, dtype=dtype),
            scale=dtype.as_numpy_dtype(prior_std),
        )
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior


def _post_fn(sigma_init: float = SIGMA_INIT):
    """Return a TFP ``kernel_posterior_fn`` (mean-field normal).

    Parameters
    ----------
    sigma_init : float, optional
        Constant initializer for the untransformed posterior scale.

    Returns
    -------
    callable
        Posterior factory compatible with TFP variational layers.
    """
    return tfpl.default_mean_field_normal_fn(
        untransformed_scale_initializer=tf.initializers.constant(sigma_init)
    )


class Net(tf.Module):
    """Sequence of (Keras + TFP) layers bypassing Keras-3 ``isinstance`` checks.

    TFP variational layers are not recognized by Keras-3 ``Sequential``; this
    thin wrapper tracks the layer list as a ``tf.Module`` and exposes the
    interface required by the training loops.

    Parameters
    ----------
    layers_list : list
        Ordered list of Keras/TFP layers.
    name : str, optional
        Module name.
    """

    def __init__(self, layers_list: List, name: str = None):
        super().__init__(name=name)
        self._lyrs = layers_list

    def __call__(self, x, training: bool = False):
        for lyr in self._lyrs:
            x = lyr(x, training=training)
        return x

    @property
    def losses(self):
        # KL terms added via layer.add_loss() in TFP layers; reset per __call__.
        return [l for lyr in self._lyrs if hasattr(lyr, "losses") for l in lyr.losses]

    @property
    def trainable_variables(self):
        return [v for lyr in self._lyrs for v in lyr.trainable_variables]

    def get_weights(self):
        return [v.numpy() for v in self.trainable_variables]

    def set_weights(self, weights):
        for var, w in zip(self.trainable_variables, weights):
            var.assign(w)

    def count_params(self) -> int:
        return int(sum(np.prod(v.shape) for v in self.trainable_variables))


def build_det_nn(d_in: int, n_chol: int, hidden, activation: str = "tanh") -> Net:
    """Build the deterministic NN: ``d_in -> hidden -> n_chol`` (linear head).

    Parameters
    ----------
    d_in : int
        Input dimension (number of centered frequencies).
    n_chol : int
        Output dimension (number of Cholesky parameters, ``d**2``).
    hidden : sequence of int
        Hidden layer sizes.
    activation : str, optional
        Hidden activation (``tanh`` per the thesis).

    Returns
    -------
    Net
        Deterministic network.
    """
    lyrs = [tf.keras.layers.Dense(h, activation=activation) for h in hidden]
    lyrs.append(tf.keras.layers.Dense(n_chol))
    return Net(lyrs, name="det_nn")


def build_bnn_reparam(d_in: int, n_chol: int, hidden,
                      sigma_init: float = SIGMA_INIT, prior_std: float = 1.0,
                      activation: str = "tanh") -> Net:
    """Variant A: fully Bayesian DenseReparameterization on every layer.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_chol : int
        Output dimension (``d**2``).
    hidden : sequence of int
        Hidden layer sizes.
    sigma_init : float, optional
        Posterior scale initializer.
    prior_std : float, optional
        Prior standard deviation.
    activation : str, optional
        Hidden activation.

    Returns
    -------
    Net
        BNN-Reparam network.
    """
    pf, pr = _post_fn(sigma_init), _prior_fn(prior_std)
    lyrs = [tfpl.DenseReparameterization(
        h, activation=activation, kernel_posterior_fn=pf, kernel_prior_fn=pr,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=True))
        for h in hidden]
    lyrs.append(tfpl.DenseReparameterization(
        n_chol, activation=None, kernel_posterior_fn=pf, kernel_prior_fn=pr,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=True)))
    return Net(lyrs, name="bnn_reparam")


def build_bnn_flipout(d_in: int, n_chol: int, hidden,
                      sigma_init: float = SIGMA_INIT, prior_std: float = 1.0,
                      activation: str = "tanh") -> Net:
    """Variant B: fully Bayesian DenseFlipout on every layer.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_chol : int
        Output dimension (``d**2``).
    hidden : sequence of int
        Hidden layer sizes.
    sigma_init : float, optional
        Posterior scale initializer.
    prior_std : float, optional
        Prior standard deviation.
    activation : str, optional
        Hidden activation.

    Returns
    -------
    Net
        BNN-Flipout network.
    """
    pf, pr = _post_fn(sigma_init), _prior_fn(prior_std)
    lyrs = [tfpl.DenseFlipout(
        h, activation=activation, kernel_posterior_fn=pf, kernel_prior_fn=pr,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=True))
        for h in hidden]
    lyrs.append(tfpl.DenseFlipout(
        n_chol, activation=None, kernel_posterior_fn=pf, kernel_prior_fn=pr,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=True)))
    return Net(lyrs, name="bnn_flipout")


def build_bnn_last(d_in: int, n_chol: int, hidden,
                   sigma_init: float = SIGMA_INIT, prior_std: float = 1.0,
                   activation: str = "tanh") -> Net:
    """Variant C (main): last-layer Bayesian (deterministic body + Flipout head).

    This is the variant selected by the thesis (``def:bnn-last``): only the
    output layer is variational, the hidden layers are deterministic.

    Parameters
    ----------
    d_in : int
        Input dimension.
    n_chol : int
        Output dimension (``d**2``).
    hidden : sequence of int
        Hidden layer sizes.
    sigma_init : float, optional
        Posterior scale initializer.
    prior_std : float, optional
        Prior standard deviation.
    activation : str, optional
        Hidden activation.

    Returns
    -------
    Net
        BNN-Last network.
    """
    pf, pr = _post_fn(sigma_init), _prior_fn(prior_std)
    lyrs = [tf.keras.layers.Dense(h, activation=activation) for h in hidden]
    lyrs.append(tfpl.DenseFlipout(
        n_chol, activation=None, kernel_posterior_fn=pf, kernel_prior_fn=pr,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=True)))
    return Net(lyrs, name="bnn_last")


def bnn_mean_forward(model: Net, x) -> tf.Tensor:
    """Deterministic MAP-style forward using posterior means.

    For each TFP layer the posterior ``loc`` is used instead of sampling; for
    plain ``Dense`` layers (the deterministic body of BNN-Last) the layer is
    called normally. Produces the point estimate used by ``eval_bnn``.

    Parameters
    ----------
    model : Net
        Network built by one of the ``build_bnn_*`` functions.
    x : tensorflow.Tensor
        Input batch.

    Returns
    -------
    tensorflow.Tensor
        Deterministic output (Cholesky parameters).
    """
    h = tf.cast(x, tf.float32)
    for lyr in model._lyrs:
        if hasattr(lyr, "kernel_posterior") and lyr.kernel_posterior is not None:
            w = lyr.kernel_posterior.distribution.loc
            if getattr(lyr, "bias_posterior", None) is not None:
                b = lyr.bias_posterior.distribution.loc
            else:
                b = tf.zeros(tf.shape(w)[-1:], dtype=w.dtype)
            h = tf.matmul(h, w) + b
            if lyr.activation is not None:
                h = lyr.activation(h)
        else:
            h = lyr(h, training=False)
    return h
