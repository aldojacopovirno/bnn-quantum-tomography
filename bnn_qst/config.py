"""Central experiment configuration for 3-qubit Bayesian Quantum State Tomography.

All scientific hyperparameters are aligned with the formal specification of
Chapter 5 ("Metodo ed Esperimenti") of the thesis and with ``EXPERIMENT.md``
Section 13.1 (definitive configuration). The defaults encoded here are the
*canonical* source of truth for the repository.

Notes
-----
The monolithic exploration notebook used a different (GPU-default) configuration
(ReLU activations, ``prior_std=3.0``, ``sigma_init=-2.0``, 100k training
samples, 1000 epochs, ``M=200``). This module deliberately enshrines the
thesis-aligned values instead. As a consequence, reproduced numbers need not
match the notebook's reported ``F=0.9704`` exactly; they reproduce the
*canonical* experiment described in the publication.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class Config:
    """Canonical configuration for the 3-qubit BNN-QST experiment.

    The dataclass is frozen so that a single validated instance is shared
    across every stage of the pipeline, guaranteeing reproducibility.
    """

    # ── Quantum system ───────────────────────────────────────────────────────
    n_qubits: int = 3

    # ── Dataset ──────────────────────────────────────────────────────────────
    n_train: int = 60_000
    n_val: int = 10_000
    n_test: int = 2_000
    n_shots: int = 5_000
    shots_sweep: Tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 50000)
    pure_fraction: float = 0.5

    # ── Architecture (Chapter 5, sec:architettura-rete) ──────────────────────
    hidden: Tuple[int, ...] = (256, 256, 128)
    activation: str = "tanh"  # rmk:scelta-tanh (NOT ReLU as in the notebook)

    # ── Deterministic NN training ────────────────────────────────────────────
    det_epochs: int = 500
    det_bs: int = 256
    det_lr: float = 1e-3

    # ── BNN training (alg:training-loop) ─────────────────────────────────────
    bnn_epochs: int = 700
    warmup: int = 200  # E_warm: beta=0 first half, linear ramp second half
    bnn_bs: int = 256
    bnn_lr: float = 1e-3
    bnn_min_lr: float = 1e-5
    clip: float = 1.0  # gradient norm clipping (alg:training-loop)
    patience: int = 40  # early stopping, active only for t >= warmup

    # ── Variational inference (def:bnn-last) ─────────────────────────────────
    prior_std: float = 1.0  # p(theta) = N(0, I)  (NOT 3.0 as in the notebook)
    sigma_init: float = -4.6  # softplus(-4.6) ~= 0.01, near-deterministic start

    # ── Monte Carlo inference (sec:inferenza-mc) ─────────────────────────────
    mc_samples: int = 150  # M  (NOT 200 as in the notebook)
    ci: float = 0.95

    # ── Baselines ────────────────────────────────────────────────────────────
    mle_max_iter: int = 2000
    mle_subset: int = 500  # MLE is the slowest baseline; evaluated on a subset

    # ── Success gates / research hypotheses ──────────────────────────────────
    gate_fid: float = 0.90
    gate_fid_gap: float = 0.03
    gate_cov_lo: float = 0.90
    gate_cov_hi: float = 0.99

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Derived quantities ───────────────────────────────────────────────────
    @property
    def d(self) -> int:
        """Hilbert-space dimension d = 2**n."""
        return 2 ** self.n_qubits

    @property
    def n_pauli(self) -> int:
        """Number of non-trivial Pauli parameters 4**n - 1."""
        return 4 ** self.n_qubits - 1

    @property
    def n_freq(self) -> int:
        """Input dimension: 3**n bases x 2**n outcomes."""
        return (3 ** self.n_qubits) * (2 ** self.n_qubits)

    @property
    def chol_params(self) -> int:
        """Number of Cholesky parameters d**2."""
        return self.d ** 2

    @property
    def centering(self) -> float:
        """Frequency centering offset 1/d (eq:centering)."""
        return 1.0 / self.d

    def with_overrides(self, **kwargs) -> "Config":
        """Return a new Config with the given fields overridden.

        Parameters
        ----------
        **kwargs
            Field names mapped to new values.

        Returns
        -------
        Config
            A new frozen Config instance.
        """
        return dataclasses.replace(self, **kwargs)
