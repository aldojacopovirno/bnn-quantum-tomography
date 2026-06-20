"""Tests for the Cholesky parametrization pipeline."""

from __future__ import annotations

import numpy as np

from bnn_qst.quantum.cholesky import check_physicality, make_cholesky_fns
from bnn_qst.quantum.pauli import build_pauli_basis


def test_cholesky_physicality():
    """Random Cholesky parameters yield Hermitian, trace-one, PSD matrices."""
    sigmas = build_pauli_basis(3)
    params_to_rho, _ = make_cholesky_fns(8, sigmas)
    check_physicality(params_to_rho, d=8, n=50)


def test_cholesky_trace_unit():
    """Every generated density matrix has unit trace."""
    sigmas = build_pauli_basis(3)
    params_to_rho, _ = make_cholesky_fns(8, sigmas)
    rng = np.random.default_rng(7)
    p = rng.standard_normal((20, 64)).astype(np.float32)
    import tensorflow as tf
    rho = params_to_rho(tf.constant(p)).numpy()
    tr = np.trace(rho, axis1=1, axis2=2).real
    assert np.allclose(tr, 1.0, atol=1e-5)
