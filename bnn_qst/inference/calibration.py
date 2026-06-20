"""Uncertainty-quantification metrics and post-hoc temperature scaling.

Implements the UQ metrics of Chapter 4 (``sec:metriche-uq``): coverage
probability (``def:coverage``), Expected Calibration Error (``def:ece``) and
sharpness (``def:sharpness``); plus the post-hoc temperature scaling of
Chapter 5 (``eq:temperature-scaling``, ``eq:t-ottimale``).

Temperature scaling corrects the systematic variance underestimation of
mean-field variational inference: scaling the epistemic standard deviation by
``T`` (equivalently the credible intervals by ``T`` around their mean) is
mathematically equivalent to replacing the posterior ``sigma_i`` with
``softplus(sigma_i) * T`` inside the variational layer (``alg:forward-pass``
step 4). The optimal ``T*`` is found by bisection so that the empirical
coverage on the validation set matches the nominal level.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..quantum.metrics import (fidelity_batch, frob_batch, phys_violations,
                               td_batch)


def coverage(r_samples: np.ndarray, r_true: np.ndarray, level: float = 0.95) -> float:
    """Per-component coverage probability, averaged over N and K (``def:coverage``).

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    r_true : numpy.ndarray
        True Pauli parameters of shape ``(N, K)``.
    level : float, optional
        Nominal credible level.

    Returns
    -------
    float
        Empirical coverage in ``[0, 1]``.
    """
    a = (1 - level) / 2
    lo = np.percentile(r_samples, 100 * a, axis=1)
    hi = np.percentile(r_samples, 100 * (1 - a), axis=1)
    return float(((r_true >= lo) & (r_true <= hi)).mean())


def ece(r_samples: np.ndarray, r_true: np.ndarray,
        levels: Optional[np.ndarray] = None) -> float:
    """Expected Calibration Error (``def:ece``).

    Mean absolute deviation of the empirical coverage from the nominal level,
    averaged over a grid of nominal levels.

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    r_true : numpy.ndarray
        True Pauli parameters of shape ``(N, K)``.
    levels : numpy.ndarray, optional
        Grid of nominal levels (default ``arange(0.05, 1.0, 0.05)``).

    Returns
    -------
    float
        ECE (``0`` for perfect calibration).
    """
    if levels is None:
        levels = np.arange(0.05, 1.0, 0.05)
    return float(np.mean([abs(coverage(r_samples, r_true, lv) - lv) for lv in levels]))


def sharpness(r_samples: np.ndarray, level: float = 0.95) -> float:
    """Mean credible-interval width (``def:sharpness``).

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    level : float, optional
        Credible level.

    Returns
    -------
    float
        Mean interval width, averaged over N and K.
    """
    a = (1 - level) / 2
    lo = np.percentile(r_samples, 100 * a, axis=1)
    hi = np.percentile(r_samples, 100 * (1 - a), axis=1)
    return float((hi - lo).mean())


def scale_samples(r_samples: np.ndarray, temp: float) -> np.ndarray:
    """Scale the credible intervals by ``temp`` around their mean.

    Equivalent to multiplying the epistemic standard deviation by ``temp``
    (``eq:temperature-scaling``).

    Parameters
    ----------
    r_samples : numpy.ndarray
        Pauli-parameter samples of shape ``(N, M, K)``.
    temp : float
        Temperature factor.

    Returns
    -------
    numpy.ndarray
        Scaled samples, same shape.
    """
    if temp == 1.0:
        return r_samples
    r_mean = r_samples.mean(axis=1, keepdims=True)
    return r_mean + temp * (r_samples - r_mean)


def calibrate_temperature(r_samples_val: np.ndarray, r_true_val: np.ndarray,
                          target_ci: float = 0.95, lo: float = 0.05,
                          hi: float = 30.0, iters: int = 50) -> float:
    """Find ``T*`` so that the scaled coverage matches ``target_ci`` (``eq:t-ottimale``).

    Post-hoc temperature scaling (Guo et al. 2017, extended to credible
    intervals). If the unscaled samples are already over-calibrated, ``1.0`` is
    returned.

    Parameters
    ----------
    r_samples_val : numpy.ndarray
        Validation Pauli-parameter samples of shape ``(N, M, K)``.
    r_true_val : numpy.ndarray
        Validation true Pauli parameters of shape ``(N, K)``.
    target_ci : float, optional
        Target coverage.
    lo, hi : float, optional
        Bisection bracket.
    iters : int, optional
        Number of bisection iterations.

    Returns
    -------
    float
        Optimal temperature ``T*``.
    """

    def _cov(lam):
        return coverage(scale_samples(r_samples_val, lam), r_true_val, target_ci)

    if _cov(1.0) >= target_ci:
        return 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _cov(mid) < target_ci:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def eval_method(rho_pred: np.ndarray, rho_true: np.ndarray,
                label: str = "") -> dict:
    """Compute accuracy metrics for a point-estimate method.

    Parameters
    ----------
    rho_pred : numpy.ndarray
        Predicted density matrices of shape ``(N, d, d)``.
    rho_true : numpy.ndarray
        True density matrices of shape ``(N, d, d)``.
    label : str, optional
        Method name.

    Returns
    -------
    dict
        ``fid_mean``, ``fid_std``, ``td_mean``, ``td_std``, ``frob_mean``,
        ``viol_pct`` and ``method``.
    """
    fids = fidelity_batch(rho_pred, rho_true)
    tds = td_batch(rho_pred, rho_true)
    frob = frob_batch(rho_pred, rho_true)
    viol = phys_violations(rho_pred)
    return {
        "method": label,
        "fid_mean": float(fids.mean()), "fid_std": float(fids.std()),
        "td_mean": float(tds.mean()), "td_std": float(tds.std()),
        "frob_mean": float(frob.mean()), "viol_pct": 100 * viol,
    }


def eval_bnn(rho_point, rho_samples, r_samples, rho_true, r_true,
             label: str = "BNN", ci: float = 0.95, temp: float = 1.0) -> dict:
    """Evaluate a BNN: point-estimate accuracy + UQ metrics on MC samples.

    The point estimate is ``rho_point`` (MAP mean-weight forward); the UQ
    metrics are computed on ``r_samples`` after temperature scaling by ``temp``.

    Parameters
    ----------
    rho_point : numpy.ndarray
        Point-estimate density matrices of shape ``(N, d, d)``.
    rho_samples : numpy.ndarray
        MC density-matrix samples of shape ``(N, M, d, d)``.
    r_samples : numpy.ndarray
        MC Pauli-parameter samples of shape ``(N, M, K)``.
    rho_true : numpy.ndarray
        True density matrices of shape ``(N, d, d)``.
    r_true : numpy.ndarray
        True Pauli parameters of shape ``(N, K)``.
    label : str, optional
        Method name.
    ci : float, optional
        Credible level.
    temp : float, optional
        Temperature factor applied to the MC samples.

    Returns
    -------
    dict
        ``eval_method`` fields plus ``coverage``, ``ece``, ``sharp``,
        ``epi_var`` and ``temp``.
    """
    base = eval_method(rho_point, rho_true, label)
    r_calib = scale_samples(r_samples, temp)
    base["coverage"] = coverage(r_calib, r_true, ci)
    base["ece"] = ece(r_calib, r_true)
    base["sharp"] = sharpness(r_calib, ci)
    base["epi_var"] = float(r_samples.var(axis=1).mean())
    base["temp"] = float(temp)
    return base
