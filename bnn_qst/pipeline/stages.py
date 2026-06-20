"""Pipeline stages: data, train, eval, sweep, plot.

Each stage is independently runnable and persists its outputs under
``<results_dir>/artifacts/`` so the pipeline can be resumed from any stage.
The canonical entry point is :func:`run_all`, which executes the stages in
order.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from ..config import Config
from ..evaluation.comparison import (research_hypotheses, run_shot_sweep,
                                     success_gate)
from ..evaluation.tables import format_results, write_csv
from ..inference.calibration import (calibrate_temperature, coverage,
                                     eval_bnn, eval_method)
from ..inference.predictive import mc_inference
from ..models.linear_inversion import linear_inversion
from ..models.mle import mle_batch
from ..models.network import (build_bnn_flipout, build_bnn_last,
                              build_bnn_reparam, build_det_nn, bnn_mean_forward)
from ..quantum.cholesky import make_cholesky_fns
from ..quantum.measurements import build_projectors, simulate_frequencies
from ..quantum.pauli import build_pauli_basis, r_from_rho
from ..quantum.states import make_special_states, werner_state
from ..training.trainer import train_bnn, train_det
from ..visualization.plots import plot_all
from ..visualization.style import apply_style
from . import io


def setup_environment(cfg: Config) -> None:
    """Seed NumPy/TensorFlow and silence TF logging.

    Parameters
    ----------
    cfg : Config
        Configuration (uses ``seed``).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)


def artifacts_dir(results_dir) -> Path:
    """Return the artifacts directory, creating it if needed.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Artifacts directory path.
    """
    d = Path(results_dir) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_quantum_primitives(cfg: Config):
    """Build the Pauli basis, projectors and Cholesky TF functions.

    Parameters
    ----------
    cfg : Config
        Configuration (uses ``n_qubits``).

    Returns
    -------
    sigmas : numpy.ndarray
        Pauli basis of shape ``(n_pauli, d, d)``.
    proj : numpy.ndarray
        Measurement projectors.
    params_to_rho, params_to_r : callable
        Traced Cholesky functions.
    """
    sigmas = build_pauli_basis(cfg.n_qubits)
    proj, _ = build_projectors(cfg.n_qubits)
    params_to_rho, params_to_r = make_cholesky_fns(cfg.d, sigmas)
    return sigmas, proj, params_to_rho, params_to_r


def build_all_models(cfg: Config, d_in: int):
    """Build (but not train) the deterministic NN and the three BNN variants.

    Parameters
    ----------
    cfg : Config
        Configuration.
    d_in : int
        Input dimension (number of centered frequencies).

    Returns
    -------
    tuple
        ``(det, bnn_a, bnn_b, bnn_c)`` networks with variables initialized.
    """
    det = build_det_nn(d_in, cfg.chol_params, cfg.hidden, cfg.activation)
    bnn_a = build_bnn_reparam(d_in, cfg.chol_params, cfg.hidden,
                              cfg.sigma_init, cfg.prior_std, cfg.activation)
    bnn_b = build_bnn_flipout(d_in, cfg.chol_params, cfg.hidden,
                              cfg.sigma_init, cfg.prior_std, cfg.activation)
    bnn_c = build_bnn_last(d_in, cfg.chol_params, cfg.hidden,
                           cfg.sigma_init, cfg.prior_std, cfg.activation)
    # Create variables with a forward call on a dummy input.
    x0 = tf.zeros((1, d_in), tf.float32)
    det(x0)
    bnn_a(x0)
    bnn_b(x0)
    bnn_c(x0)
    return det, bnn_a, bnn_b, bnn_c


def run_data(cfg: Config, results_dir) -> Path:
    """Stage 1: generate the synthetic 3-qubit dataset.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Path to the saved ``dataset.npz``.
    """
    setup_environment(cfg)
    sigmas, proj, _, _ = build_quantum_primitives(cfg)

    rng = np.random.default_rng(cfg.seed + 1)
    n_total = cfg.n_train + cfg.n_val + cfg.n_test
    from ..quantum.states import generate_dataset
    rho_all, r_all = generate_dataset(cfg.n_qubits, n_total, cfg.pure_fraction, rng, sigmas)

    rho_tr = rho_all[:cfg.n_train]
    r_tr = r_all[:cfg.n_train]
    rho_va = rho_all[cfg.n_train:cfg.n_train + cfg.n_val]
    r_va = r_all[cfg.n_train:cfg.n_train + cfg.n_val]
    rho_te = rho_all[cfg.n_train + cfg.n_val:]
    r_te = r_all[cfg.n_train + cfg.n_val:]

    # Raw frequencies (LI/MLE) — centered derivatives are computed at load time.
    f_tr_raw = simulate_frequencies(rho_tr, proj, cfg.n_shots, rng).astype(np.float32)
    f_va_raw = simulate_frequencies(rho_va, proj, cfg.n_shots, rng).astype(np.float32)
    f_te_raw = simulate_frequencies(rho_te, proj, cfg.n_shots, rng).astype(np.float32)

    path = artifacts_dir(results_dir) / "dataset.npz"
    io.save_npz(path, rho_tr=rho_tr, r_tr=r_tr.astype(np.float32),
                rho_va=rho_va, r_va=r_va.astype(np.float32),
                rho_te=rho_te, r_te=r_te.astype(np.float32),
                f_tr_raw=f_tr_raw, f_va_raw=f_va_raw, f_te_raw=f_te_raw)
    print(f"[data] Dataset saved to {path} "
          f"(train={cfg.n_train}, val={cfg.n_val}, test={cfg.n_test})")
    return path


def _load_dataset(cfg: Config, results_dir):
    """Load the dataset and return centered frequencies for the NN inputs."""
    data = io.load_npz(artifacts_dir(results_dir) / "dataset.npz")
    f_tr = (data["f_tr_raw"] - cfg.centering).astype(np.float32)
    f_va = (data["f_va_raw"] - cfg.centering).astype(np.float32)
    f_te = (data["f_te_raw"] - cfg.centering).astype(np.float32)
    return data, f_tr, f_va, f_te


def run_train(cfg: Config, results_dir) -> Path:
    """Stage 2: train the deterministic NN and the three BNN variants.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Path to the saved training histories.
    """
    setup_environment(cfg)
    data, f_tr, f_va, f_te = _load_dataset(cfg, results_dir)
    r_tr, r_va = data["r_tr"], data["r_va"]
    sigmas, proj, _, params_to_r = build_quantum_primitives(cfg)
    d_in = f_tr.shape[1]

    det, bnn_a, bnn_b, bnn_c = build_all_models(cfg, d_in)
    print(f"[train] Det NN params: {det.count_params():,}")
    hist_det = train_det(det, f_tr, r_tr, f_va, r_va, params_to_r, cfg)

    print(f"[train] BNN-A (Reparam) params: {bnn_a.count_params()}")
    hist_a = train_bnn(bnn_a, f_tr, r_tr, f_va, r_va, params_to_r, cfg)
    print(f"[train] BNN-B (Flipout) params: {bnn_b.count_params()}")
    hist_b = train_bnn(bnn_b, f_tr, r_tr, f_va, r_va, params_to_r, cfg)
    print(f"[train] BNN-C (Last) params: {bnn_c.count_params()}")
    hist_c = train_bnn(bnn_c, f_tr, r_tr, f_va, r_va, params_to_r, cfg)

    adir = artifacts_dir(results_dir)
    io.save_model_weights(adir / "det_weights", det)
    io.save_model_weights(adir / "bnn_a_weights", bnn_a)
    io.save_model_weights(adir / "bnn_b_weights", bnn_b)
    io.save_model_weights(adir / "bnn_c_weights", bnn_c)
    hist_path = io.save_pickle(adir / "histories",
                               {"det": hist_det, "A": hist_a, "B": hist_b, "C": hist_c})
    print(f"[train] Weights and histories saved to {adir}")
    return hist_path


def _restore_models(cfg: Config, results_dir, d_in):
    """Build and restore all model weights from disk."""
    det, bnn_a, bnn_b, bnn_c = build_all_models(cfg, d_in)
    adir = artifacts_dir(results_dir)
    io.load_model_weights(adir / "det_weights.npz", det)
    io.load_model_weights(adir / "bnn_a_weights.npz", bnn_a)
    io.load_model_weights(adir / "bnn_b_weights.npz", bnn_b)
    io.load_model_weights(adir / "bnn_c_weights.npz", bnn_c)
    return det, bnn_a, bnn_b, bnn_c


def run_eval(cfg: Config, results_dir) -> Path:
    """Stage 3: MC inference, calibration, evaluation and artifact dump.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Path to the saved evaluation artifact (``eval.pkl``).
    """
    setup_environment(cfg)
    data, f_tr, f_va, f_te = _load_dataset(cfg, results_dir)
    r_tr, r_va, r_te = data["r_tr"], data["r_va"], data["r_te"]
    rho_te = data["rho_te"]
    f_te_raw = data["f_te_raw"]
    sigmas, proj, params_to_rho, params_to_r = build_quantum_primitives(cfg)
    d_in = f_te.shape[1]
    histories = io.load_pickle(artifacts_dir(results_dir) / "histories.pkl")
    det, bnn_a, bnn_b, bnn_c = _restore_models(cfg, results_dir, d_in)

    f_te_tf = tf.constant(f_te, tf.float32)
    m = cfg.mc_samples

    # ── Baselines ────────────────────────────────────────────────────────────
    _, rho_li = linear_inversion(f_te_raw, proj, sigmas)
    res_li = eval_method(rho_li, rho_te, "LI")

    rho_mle = mle_batch(f_te_raw, proj, cfg.mle_max_iter, subset=cfg.mle_subset)
    rho_te_sub = rho_te[:cfg.mle_subset]
    res_mle = eval_method(rho_mle, rho_te_sub, "MLE")

    rho_det_pred = params_to_rho(det(f_te_tf, training=False)).numpy()
    res_det = eval_method(rho_det_pred, rho_te, "NN-det")

    # ── BNN variants: MC inference + temperature scaling ─────────────────────
    bnn_models = {"A": bnn_a, "B": bnn_b, "C": bnn_c}
    names = {"A": "BNN-Reparam", "B": "BNN-Flipout", "C": "BNN-Last"}
    r_mc, rho_mc, rho_map, temps, cov_pre, results = {}, {}, {}, {}, {}, {}
    for key, model in bnn_models.items():
        rho_mc[key], r_mc[key] = mc_inference(model, f_te, params_to_rho, params_to_r, m)
        rho_map[key] = params_to_rho(bnn_mean_forward(model, f_te_tf)).numpy()
        cov_pre[key] = coverage(r_mc[key], r_te, cfg.ci)
        _, r_mcv = mc_inference(model, f_va, params_to_rho, params_to_r, m)
        temps[key] = calibrate_temperature(r_mcv, r_va, cfg.ci)
        results[key] = eval_bnn(rho_map[key], rho_mc[key], r_mc[key], rho_te, r_te,
                                names[key], cfg.ci, temps[key])
    print(f"[eval] Temperatures: A={temps['A']:.3f} B={temps['B']:.3f} C={temps['C']:.3f}")

    # ── Summary table ────────────────────────────────────────────────────────
    import pandas as pd
    df_3q = pd.DataFrame([res_li, res_mle, res_det,
                          results["A"], results["B"], results["C"]]).set_index("method")
    print("\n[eval] 3-qubit results (N_shots=5000)")
    print(format_results(df_3q).to_string())
    write_csv(df_3q, Path(results_dir) / "tables" / "results_3q.csv")

    # ── Best BNN selection ───────────────────────────────────────────────────
    bnn_results = [results["A"], results["B"], results["C"]]
    best_key = max(["A", "B", "C"], key=lambda k: results[k]["fid_mean"])
    best_name = names[best_key]
    best_model = bnn_models[best_key]
    t_best = temps[best_key]
    print(f"[eval] Best BNN: {best_name} (T={t_best:.3f})")

    # ── Success gate + research hypotheses ───────────────────────────────────
    gate = success_gate(bnn_results, res_mle["fid_mean"], cfg)
    hyp = research_hypotheses(results[best_key], res_mle["fid_mean"], cfg)
    print(f"[eval] Gate: fid={gate['gate_fid']} gap={gate['gate_gap']} "
          f"cov={gate['gate_cov']} -> {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"[eval] Hypotheses: {hyp}")

    # ── Auxiliary arrays for the plotting stage ──────────────────────────────
    art = {
        "hist_A": histories["A"], "hist_B": histories["B"], "hist_C": histories["C"],
        "hist_det": histories["det"],
        "r_mc_A": r_mc["A"], "r_mc_B": r_mc["B"], "r_mc_C": r_mc["C"],
        "rho_mc_A": rho_mc["A"], "rho_mc_B": rho_mc["B"], "rho_mc_C": rho_mc["C"],
        "rho_map_A": rho_map["A"], "rho_map_B": rho_map["B"], "rho_map_C": rho_map["C"],
        "tA": temps["A"], "tB": temps["B"], "tC": temps["C"],
        "cov_pre_A": cov_pre["A"], "cov_pre_B": cov_pre["B"], "cov_pre_C": cov_pre["C"],
        "rho_li": rho_li, "rho_mle": rho_mle, "rho_det_pred": rho_det_pred,
        "rho_te": rho_te, "r_te": r_te, "rho_te_sub": rho_te_sub,
        "res_A": results["A"], "res_B": results["B"], "res_C": results["C"],
        "res_li": res_li, "res_mle": res_mle, "res_det": res_det,
        "best_bnn_name": best_name, "best_key": best_key, "t_best": t_best,
        "df_3q": df_3q, "gate": gate, "hypotheses": hyp,
        "mle_subset": cfg.mle_subset,
    }

    # Notable states predictive distributions.
    special = make_special_states()
    rng_spec = np.random.default_rng(cfg.seed + 200)
    comps = [0, 1, 2, 62]
    art["special_comps"] = comps
    art["special"] = {}
    art["special_true"] = {}
    for sname, rho_s in special.items():
        f_s = (simulate_frequencies(rho_s[None], proj, cfg.n_shots, rng_spec)
               - cfg.centering).astype(np.float32)
        _, r_s_mc = mc_inference(best_model, f_s, params_to_rho, params_to_r, m=m)
        art["special"][sname] = r_s_mc
        art["special_true"][sname] = r_from_rho(rho_s, sigmas)

    # Werner sweep.
    rho_ghz = special["GHZ"]
    p_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    rng_wern = np.random.default_rng(cfg.seed + 300)
    werner_records = []
    for p in p_values:
        rho_w = werner_state(rho_ghz, p)
        f_raw = simulate_frequencies(rho_w[None], proj, cfg.n_shots, rng_wern).astype(np.float32)
        f_norm = (f_raw - cfg.centering).astype(np.float32)
        _, rho_li_w = linear_inversion(f_raw, proj, sigmas)
        rec = eval_method(rho_li_w, rho_w[None], "LI"); rec["p"] = p; werner_records.append(rec)
        rho_mle_w = mle_batch(f_raw, proj, cfg.mle_max_iter, subset=1)
        rec = eval_method(rho_mle_w, rho_w[None], "MLE"); rec["p"] = p; werner_records.append(rec)
        rho_det_w = params_to_rho(det(tf.constant(f_norm), training=False)).numpy()
        rec = eval_method(rho_det_w, rho_w[None], "NN-det"); rec["p"] = p; werner_records.append(rec)
        rho_map_w = params_to_rho(bnn_mean_forward(best_model, tf.constant(f_norm, tf.float32))).numpy()
        _, r_mc_w = mc_inference(best_model, f_norm, params_to_rho, params_to_r, m=m)
        rec = eval_method(rho_map_w, rho_w[None], best_name)
        rec["epi_var"] = float(r_mc_w[0].var(axis=0).sum()); rec["p"] = p
        werner_records.append(rec)
    art["df_werner"] = pd.DataFrame(werner_records)

    # Purity / epistemic-variance arrays.
    art["purity_te"] = np.trace(rho_te @ rho_te, axis1=1, axis2=2).real
    art["epi_per_state"] = r_mc[best_key].var(axis=1).sum(axis=-1)
    art["r_mc_best"] = r_mc[best_key]

    # df_sweep placeholder (filled by the sweep stage).
    art["df_sweep"] = None

    path = io.save_pickle(artifacts_dir(results_dir) / "eval", art)
    print(f"[eval] Artifact saved to {path}")
    return path


def run_sweep(cfg: Config, results_dir) -> Path:
    """Stage 4: shot-budget sweep over the four methods.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Path to the saved ``results_sweep.csv``.
    """
    setup_environment(cfg)
    data, f_tr, f_va, f_te = _load_dataset(cfg, results_dir)
    rho_te, r_te = data["rho_te"], data["r_te"]
    sigmas, proj, params_to_rho, params_to_r = build_quantum_primitives(cfg)
    d_in = f_te.shape[1]
    det, bnn_a, bnn_b, bnn_c = _restore_models(cfg, results_dir, d_in)

    art = io.load_pickle(artifacts_dir(results_dir) / "eval.pkl")
    best_name = art["best_bnn_name"]
    best_key = art["best_key"]
    t_best = art["t_best"]
    best_bnn = {"A": bnn_a, "B": bnn_b, "C": bnn_c}[best_key]

    rng_sweep = np.random.default_rng(cfg.seed + 100)
    records = run_shot_sweep(cfg, rho_te, r_te, det, best_bnn, best_name, t_best,
                             proj, sigmas, params_to_rho, params_to_r, rng_sweep)

    import pandas as pd
    df_sweep = pd.DataFrame(records)
    sweep_path = write_csv(df_sweep, Path(results_dir) / "tables" / "results_sweep.csv")
    pivot = df_sweep.groupby(["method", "n_shots"])["fid_mean"].mean().unstack().round(4)
    print("\n[sweep] Mean fidelity per N_shots")
    print(pivot.to_string())

    # Update the eval artifact so the plotting stage can access df_sweep.
    art["df_sweep"] = df_sweep
    io.save_pickle(artifacts_dir(results_dir) / "eval", art)
    print(f"[sweep] Sweep table saved to {sweep_path}")
    return sweep_path


def run_plot(cfg: Config, results_dir) -> Path:
    """Stage 5: regenerate the 12 thesis figures.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.

    Returns
    -------
    pathlib.Path
        Plots output directory.
    """
    apply_style()
    art = io.load_pickle(artifacts_dir(results_dir) / "eval.pkl")
    plots_dir = Path(results_dir) / "plots"
    plot_all(art, plots_dir, cfg)
    print(f"[plot] Figures saved to {plots_dir}")
    return plots_dir


def run_all(cfg: Config, results_dir) -> None:
    """Run every stage in order: data, train, eval, sweep, plot.

    Parameters
    ----------
    cfg : Config
        Configuration.
    results_dir : str or pathlib.Path
        Root results directory.
    """
    run_data(cfg, results_dir)
    run_train(cfg, results_dir)
    run_eval(cfg, results_dir)
    run_sweep(cfg, results_dir)
    run_plot(cfg, results_dir)
