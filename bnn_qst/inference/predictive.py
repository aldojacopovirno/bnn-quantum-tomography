"""Monte Carlo predictive distribution.

For a new centered frequency vector, the predictive distribution
(``eq:distribuzione-predittiva-mc``) is approximated by ``M`` stochastic
forward passes through the BNN, each sampling a weight configuration from the
variational posterior. The collection ``{rho^(m)}`` yields:

* a point estimate ``hat_rho = mean_m rho^(m)`` (``eq:stima-puntuale``), valid
  because the state space is convex;
* 95% credible intervals per Pauli component via the 2.5/97.5 percentiles
  (``eq:ci-95``);
* the epistemic uncertainty ``sum_k Var[r_k]`` (``eq:incertezza-epistemica-mc``).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def mc_inference(model, f_test, p2rho_fn, p2r_fn, m: int = 150, bs: int = 512):
    """Run ``M`` stochastic forward passes to build the predictive distribution.

    Parameters
    ----------
    model : Net
        Trained BNN.
    f_test : numpy.ndarray
        Centered test frequencies of shape ``(N, d_in)``.
    p2rho_fn : callable
        Traced ``params_to_rho`` Cholesky function.
    p2r_fn : callable
        Traced ``params_to_r`` Cholesky function.
    m : int, optional
        Number of Monte Carlo samples.
    bs : int, optional
        Batch size for the forward passes.

    Returns
    -------
    rho_samples : numpy.ndarray
        Density-matrix samples of shape ``(N, M, d, d)``.
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    """
    n = f_test.shape[0]
    f_tf = tf.constant(f_test, tf.float32)
    rho_all, r_all = [], []
    for _ in tqdm(range(m), desc="MC", leave=False):
        rho_m, r_m = [], []
        for i in range(0, n, bs):
            out = model(f_tf[i:i + bs], training=False)  # stochastic for TFP
            rho_m.append(p2rho_fn(out).numpy())
            r_m.append(p2r_fn(out).numpy())
        rho_all.append(np.concatenate(rho_m, 0))
        r_all.append(np.concatenate(r_m, 0))
    return np.stack(rho_all, 1), np.stack(r_all, 1)


def point_estimate(rho_samples: np.ndarray) -> np.ndarray:
    """MC point estimate ``hat_rho = mean_m rho^(m)`` (``eq:stima-puntuale``).

    Parameters
    ----------
    rho_samples : numpy.ndarray
        Density-matrix samples of shape ``(N, M, d, d)``.

    Returns
    -------
    numpy.ndarray
        Point estimate of shape ``(N, d, d)``.
    """
    return rho_samples.mean(axis=1)


def credible_intervals(r_samples: np.ndarray, level: float = 0.95):
    """Per-component credible intervals (``eq:ci-95``).

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    level : float, optional
        Credible level (``0.95`` -> 2.5/97.5 percentiles).

    Returns
    -------
    lo, hi : numpy.ndarray
        Lower/upper bounds of shape ``(N, K)``.
    """
    a = (1 - level) / 2
    lo = np.percentile(r_samples, 100 * a, axis=1)
    hi = np.percentile(r_samples, 100 * (1 - a), axis=1)
    return lo, hi


def epistemic_variance(r_samples: np.ndarray) -> float:
    """Total epistemic uncertainty ``sum_k Var[r_k]`` (``eq:incertezza-epistemica-mc``).

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.

    Returns
    -------
    float
        Mean over the test set of the summed per-component variance.
    """
    return float(r_samples.var(axis=1).mean())
