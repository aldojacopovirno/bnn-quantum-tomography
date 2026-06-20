# Bayesian Neural Networks for Quantum State Tomography

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Modular, reproducible implementation of the 3-qubit experiment accompanying the
thesis *"Sulle Reti Neurali Bayesiane per la Tomografia degli Stati Quantistici"*.

The repository reconstructs the density matrix of a 3-qubit quantum system
from noisy measurement frequencies using a **Bayesian Neural Network (BNN)**,
and quantifies the predictive uncertainty with calibrated credible intervals.
It is benchmarked against three classical estimators: Linear Inversion (LI),
Maximum Likelihood Estimation (MLE), and a deterministic neural network.

---

## Background

**Problem.** Quantum State Tomography (QST) reconstructs the density matrix
`rho ∈ C^{d×d}` (here `d = 2^3 = 8`) of an n-qubit system from observed
measurement frequencies. A valid density matrix must satisfy three physical
constraints: Hermiticity (`rho = rho†`), positive semi-definiteness
(`rho ≥ 0`), and unit trace (`Tr(rho) = 1`). The system is measured in the `3^3 = 27` tensor Pauli bases, each producing
`2^3 = 8` outcomes, for a total of `216` observed frequencies per state. The
Pauli decomposition

```
rho = (1/d) (I + sum_i r_i sigma_i),     r_i = Tr(rho sigma_i)
```

maps `rho` to `4^3 - 1 = 63` real Pauli parameters `r`.

**Method.** The network learns the mapping from centered frequencies
`f̃ = f - 1/d ∈ R^{216}` to `d^2 = 64` Cholesky parameters `c`. The density
matrix is then built as

```
rho(c) = L(c) L(c)† / Tr(L(c) L(c)†)
```

where `L(c)` is lower-triangular with positive diagonal. This **guarantees all
three physical constraints by construction**, so no trace or positivity penalty
is needed in the loss (the corresponding terms are identically zero,
`eq:loss-totale` of the thesis).

The Bayesian variant **BNN-Last** treats only the output layer as variational
(`def:bnn-last`): hidden weights are deterministic, the last layer follows a
mean-field Gaussian posterior `q_phi(theta) = prod_i N(mu_i, sigma_i^2)` with
`sigma_i = softplus(...)` and a near-deterministic initialization
`softplus(-4.6) ≈ 0.01`. The prior is `N(0, I)`. Training maximizes the ELBO

```
L(phi) = MSE(r_hat, r_true) + (beta(t) / N_train) * KL[q_phi || p]
```

with KL annealing (`eq:kl-annealing`): `beta` is `0` for the first half of
warmup, ramps linearly to `1` over the second half, then stays `1`. An
early-stopping fix activates patience only after warmup (`alg:training-loop`).

Predictive uncertainty is obtained by Monte Carlo inference: `M = 150`
stochastic forward passes yield a point estimate (mean of the sampled
`rho^(m)`, valid by convexity of the state space), 95% credible intervals
(2.5/97.5 percentiles per Pauli component), and the epistemic variance
`sum_k Var[r_k]`. A post-hoc temperature scaling `T*` (calibrated on the
validation set per shot regime) corrects the variance underestimation typical
of mean-field variational inference.

---

## Repository structure

```
repo/
├── main.py                      # CLI orchestrator (stages: data|train|eval|sweep|plot|all)
├── bnn_qst/
│   ├── config.py                # Canonical Config dataclass (thesis-aligned)
│   ├── quantum/                 # Pauli basis, Cholesky pipeline, states, measurements, metrics
│   ├── models/                  # Deterministic NN, 3 BNN variants, Linear Inversion, MLE
│   ├── training/                # ELBO loss, KL annealing, training loops
│   ├── inference/               # MC predictive distribution, calibration/UQ metrics
│   ├── evaluation/              # Method comparison, shot sweep, gates, H1-H5, CSV export
│   ├── visualization/           # Plot style + 12 figure generators
│   └── pipeline/                # Stage orchestration + artifact I/O
├── tests/                       # pytest: Pauli orthogonality, Cholesky physicality, metric identities
├── requirements.txt
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

Generated outputs (datasets, weights, figures, CSVs) are written under
`results/` and are gitignored, the code regenerates them at runtime.

---

## Installation

```bash
git clone <repository-url>
cd repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

A GPU (e.g. Colab T4/V100) is strongly recommended for the full run; the
deterministic and BNN training of the canonical configuration takes ~2 hours on
GPU. Python 3.10+ is required.

---

## Usage

Run the full canonical experiment end-to-end:

```bash
python main.py all --results-dir results
```

Run individual stages (each persists to `results/artifacts/` and can be
resumed):

```bash
python main.py data   --results-dir results   # generate the synthetic dataset
python main.py train  --results-dir results   # train det NN + 3 BNN variants
python main.py eval   --results-dir results   # MC inference, calibration, tables
python main.py sweep  --results-dir results   # N_shots sweep
python main.py plot   --results-dir results   # regenerate the 12 figures
```

CLI options:

| Option           | Default | Description                              |
|------------------|---------|------------------------------------------|
| `--results-dir`  | `results` | Root for artifacts, tables, figures    |
| `--seed`         | `42`    | Global random seed                       |
| `--n-shots`      | `5000`  | Default shots per basis                  |

### Tests

```bash
pytest -q
```

---

## Canonical configuration

The defaults in `bnn_qst/config.py` are aligned with Chapter 5 of the thesis
and `EXPERIMENT.md` §13.1 (definitive configuration):

| Parameter | Value |
|---|---|
| System | 3 qubits, `d = 8`, 63 Pauli params, 216 frequencies |
| Architecture | `[256, 256, 128]`, `tanh` activation |
| Dataset | 60 000 train / 10 000 val / 2 000 test, `pure_fraction = 0.5` |
| Det NN | 500 epochs, batch 256, lr 1e-3, patience 40 |
| BNN | 700 epochs, warmup 200, batch 256, lr 1e-3, clip 1.0, patience 40 |
| Variational | `prior_std = 1.0`, `sigma_init = -4.6` (≈0.01) |
| MC inference | `M = 150`, 95% credible intervals |
| Baselines | MLE max_iter 2000 (subset 500), LI |

> **Note on parameter alignment.** The original exploration notebook used a
> different GPU-default configuration (ReLU activations, `prior_std = 3.0`,
> `sigma_init = -2.0`, 100k training samples, 1000 epochs, `M = 200`), which
> produced the fidelity `F = 0.9704` reported in the thesis tables. This
> repository enshrines the **thesis-aligned** configuration as canonical, so
> reproduced numbers need not match that exact value; they reproduce the
> *canonical* experiment described in the publication. The notebook remains in
> the thesis repository as historical provenance.

---

## Outputs

After `python main.py all`:

* `results/tables/results_3q.csv` — 6-method comparison at `N_shots = 5000`;
* `results/tables/results_sweep.csv` — `N_shots` sweep
  `{500, 1000, 2000, 5000, 10000, 50000}`;
* `results/plots/*.pdf` and `*.png` — 12 thesis figures (learning curves,
  reliability diagram, fidelity/UQ vs shots, predictive distributions on
  notable states, Werner sweep, violin/min-eigenvalue distributions,
  fidelity/uncertainty vs purity, temperature-scaling effect, Pauli bias/std);
* `results/artifacts/` — dataset, model weights, histories, evaluation
  artifact (for resumption).

---

## License

MIT — see [LICENSE](LICENSE).
