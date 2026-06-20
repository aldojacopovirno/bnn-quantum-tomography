"""Method comparison, shot-budget sweep, success gates and research hypotheses.

Aggregates the four estimators (LI, MLE, deterministic NN, best BNN) into
summary tables, runs the ``N_shots`` sweep that probes the low-measurement
regime, and checks the three success gates plus the research hypotheses H1-H5
(Chapter 6, ``sec:validazione-sperimentale``).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ..config import Config
from ..inference.calibration import eval_bnn, eval_method
from ..inference.predictive import mc_inference
from ..models.linear_inversion import linear_inversion
from ..models.mle import mle_batch
from ..models.network import bnn_mean_forward
from ..quantum.measurements import simulate_frequencies


def run_shot_sweep(cfg: Config, rho_te, r_te, det_model, best_bnn, best_bnn_name,
                   t_best, proj, sigmas, params_to_rho, params_to_r,
                   rng_sweep):
    """Run the ``N_shots`` sweep over all four methods.

    For each shot budget the test frequencies are regenerated (same states,
    new noise) and every method is evaluated. The BNN uses the pre-calibrated
    temperature ``t_best``.

    Parameters
    ----------
    cfg : Config
        Configuration (uses ``shots_sweep``, ``ci``, ``mle_subset``).
    rho_te : numpy.ndarray
        Test density matrices.
    r_te : numpy.ndarray
        Test Pauli parameters.
    det_model : Net
        Trained deterministic NN.
    best_bnn : Net
        Best trained BNN.
    best_bnn_name : str
        Name of the best BNN variant.
    t_best : float
        Calibrated temperature of the best BNN.
    proj : numpy.ndarray
        Measurement projectors.
    sigmas : numpy.ndarray
        Pauli basis.
    params_to_rho, params_to_r : callable
        Traced Cholesky functions.
    rng_sweep : numpy.random.Generator
        Random generator for frequency resampling.

    Returns
    -------
    list of dict
        One record per (method, n_shots).
    """
    records = []
    for n_shots in tqdm(cfg.shots_sweep, desc="Shot sweep"):
        f_sw_raw = simulate_frequencies(rho_te, proj, n_shots, rng_sweep).astype(np.float32)
        f_sw = (f_sw_raw - cfg.centering).astype(np.float32)

        _, rho_li_sw = linear_inversion(f_sw_raw, proj, sigmas)
        rec = eval_method(rho_li_sw, rho_te, "LI")
        rec["n_shots"] = n_shots
        records.append(rec)

        rho_mle_sw = mle_batch(f_sw_raw, proj, 300, subset=cfg.mle_subset)
        rec = eval_method(rho_mle_sw, rho_te[:cfg.mle_subset], "MLE")
        rec["n_shots"] = n_shots
        records.append(rec)

        rho_det_sw = params_to_rho(det_model(tf.constant(f_sw), training=False)).numpy()
        rec = eval_method(rho_det_sw, rho_te, "NN-det")
        rec["n_shots"] = n_shots
        records.append(rec)

        rho_mc_sw, r_mc_sw = mc_inference(best_bnn, f_sw, params_to_rho, params_to_r, m=50)
        rho_map_sw = params_to_rho(
            bnn_mean_forward(best_bnn, tf.constant(f_sw, tf.float32))).numpy()
        rec = eval_bnn(rho_map_sw, rho_mc_sw, r_mc_sw, rho_te, r_te,
                       best_bnn_name, cfg.ci, t_best)
        rec["n_shots"] = n_shots
        records.append(rec)
    return records


def success_gate(bnn_results, mle_fid, cfg: Config) -> dict:
    """Check the three success gates on the 3-qubit BNN results.

    Parameters
    ----------
    bnn_results : list of dict
        Evaluation dicts for the three BNN variants.
    mle_fid : float
        MLE mean fidelity (reference for the gap gate).
    cfg : Config
        Configuration (uses ``gate_*`` fields).

    Returns
    -------
    dict
        ``gate_fid``, ``gate_gap``, ``gate_cov`` booleans and the global
        ``passed`` flag.
    """
    gate_fid = any(r["fid_mean"] >= cfg.gate_fid for r in bnn_results)
    gate_gap = any(r["fid_mean"] >= mle_fid - cfg.gate_fid_gap for r in bnn_results)
    gate_cov = any(cfg.gate_cov_lo <= r.get("coverage", 0) <= cfg.gate_cov_hi
                   for r in bnn_results)
    return {"gate_fid": gate_fid, "gate_gap": gate_gap, "gate_cov": gate_cov,
            "passed": bool(gate_fid and gate_gap and gate_cov)}


def research_hypotheses(best_res, mle_fid, cfg: Config) -> dict:
    """Check the research hypotheses H1-H5 (Chapter 6).

    Parameters
    ----------
    best_res : dict
        Evaluation dict of the best BNN.
    mle_fid : float
        MLE mean fidelity.
    cfg : Config
        Configuration.

    Returns
    -------
    dict
        Boolean flags ``H1`` ... ``H5`` and the global ``confirmed`` flag
        (``H1 and H2 and H3``).
    """
    h1 = best_res["fid_mean"] >= cfg.gate_fid
    h2 = best_res["fid_mean"] >= mle_fid - cfg.gate_fid_gap
    h3 = cfg.gate_cov_lo <= best_res.get("coverage", 0) <= cfg.gate_cov_hi
    h4 = "epi_var" in best_res
    h5 = best_res["viol_pct"] < 1.0
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5,
            "confirmed": bool(h1 and h2 and h3)}
