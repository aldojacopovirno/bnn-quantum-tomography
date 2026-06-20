"""Training loops for the deterministic NN and the BNN variants.

Both loops use Adam with cosine learning-rate decay. The BNN loop maximizes
the ELBO (``eq:loss-totale``) with KL annealing (``eq:kl-annealing``) and the
*early-stopping fix* of ``alg:training-loop``: the patience counter is active
only for ``t >= E_warm`` and the internal best/wait state is reset exactly at
``t == E_warm``. Without this, the validation MSE naturally worsens during
annealing and triggers premature stopping before ``beta = 1``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from ..config import Config
from .losses import kl_annealing_factor, kl_term, mse_on_pauli


def train_det(model, f_tr, r_tr, f_val, r_val, p2r_fn, cfg: Config) -> dict:
    """Train the deterministic NN by MSE on Pauli parameters.

    Parameters
    ----------
    model : Net
        Deterministic network (already built).
    f_tr, r_tr : numpy.ndarray
        Training inputs (centered frequencies) and targets.
    f_val, r_val : numpy.ndarray
        Validation inputs and targets.
    p2r_fn : callable
        Traced ``params_to_r`` Cholesky function.
    cfg : Config
        Configuration (uses ``det_*`` fields and ``seed``).

    Returns
    -------
    dict
        Training history with keys ``tr_mse`` and ``val_mse``.
    """
    lr_sch = tf.keras.optimizers.schedules.CosineDecay(
        cfg.det_lr, cfg.det_epochs * (len(f_tr) // cfg.det_bs),
        alpha=1e-5 / cfg.det_lr)
    opt = tf.keras.optimizers.Adam(lr_sch)

    f_tr_tf = tf.constant(f_tr, tf.float32)
    r_tr_tf = tf.constant(r_tr, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((f_tr_tf, r_tr_tf))
    ds = ds.shuffle(len(f_tr), seed=cfg.seed).batch(cfg.det_bs).prefetch(2)

    @tf.function
    def step(fb, rb):
        with tf.GradientTape() as tape:
            rp = p2r_fn(model(fb, training=True))
            loss = mse_on_pauli(rp, rb)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    hist = {"tr_mse": [], "val_mse": []}
    best_val, best_w, patience = np.inf, None, 0
    for _ in tqdm(range(cfg.det_epochs), desc="Det NN", leave=False):
        tr_l = [step(fb, rb).numpy() for fb, rb in ds]
        vp = p2r_fn(model(tf.constant(f_val, tf.float32), training=False)).numpy()
        vl = float(np.mean(np.sum((vp - r_val) ** 2, axis=-1)))
        hist["tr_mse"].append(float(np.mean(tr_l)))
        hist["val_mse"].append(vl)
        if vl < best_val - 1e-6:
            best_val, best_w, patience = vl, [w.copy() for w in model.get_weights()], 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
    if best_w:
        model.set_weights(best_w)
    return hist


def train_bnn(model, f_tr, r_tr, f_val, r_val, p2r_fn, cfg: Config) -> dict:
    """Train a BNN variant by ELBO maximization with KL annealing.

    Parameters
    ----------
    model : Net
        Bayesian network (already built).
    f_tr, r_tr : numpy.ndarray
        Training inputs (centered frequencies) and targets.
    f_val, r_val : numpy.ndarray
        Validation inputs and targets.
    p2r_fn : callable
        Traced ``params_to_r`` Cholesky function.
    cfg : Config
        Configuration (uses ``bnn_*``, ``warmup``, ``clip``, ``patience``,
        ``seed``).

    Returns
    -------
    dict
        Training history with keys ``tr_mse``, ``val_mse``, ``kl``, ``beta``.
    """
    n_tr = len(f_tr)
    lr_sch = tf.keras.optimizers.schedules.CosineDecay(
        cfg.bnn_lr, cfg.bnn_epochs * (n_tr // cfg.bnn_bs),
        alpha=cfg.bnn_min_lr / cfg.bnn_lr)
    opt = tf.keras.optimizers.Adam(lr_sch, clipnorm=cfg.clip)

    f_tr_tf = tf.constant(f_tr, tf.float32)
    r_tr_tf = tf.constant(r_tr, tf.float32)
    f_val_tf = tf.constant(f_val, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((f_tr_tf, r_tr_tf))
    ds = ds.shuffle(n_tr, seed=cfg.seed).batch(cfg.bnn_bs).prefetch(2)

    n_tr_f = tf.constant(float(n_tr), tf.float32)
    beta_v = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    @tf.function
    def step(fb, rb):
        with tf.GradientTape() as tape:
            rp = p2r_fn(model(fb, training=True))
            mse = mse_on_pauli(rp, rb)
            kl = kl_term(model)
            loss = mse + beta_v * kl / n_tr_f  # eq:loss-totale
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return mse, kl

    hist = {"tr_mse": [], "val_mse": [], "kl": [], "beta": []}
    best_val, best_w, patience = np.inf, None, 0
    pbar = tqdm(range(cfg.bnn_epochs), desc="BNN", leave=False)
    for ep in pbar:
        beta = kl_annealing_factor(ep, cfg.warmup)
        beta_v.assign(beta)

        mse_l, kl_l = zip(*[step(fb, rb) for fb, rb in ds])
        mse_l = [m.numpy() for m in mse_l]
        kl_l = [k.numpy() for k in kl_l]
        vp = p2r_fn(model(f_val_tf, training=False)).numpy()
        vl = float(np.mean(np.sum((vp - r_val) ** 2, axis=-1)))

        hist["tr_mse"].append(float(np.mean(mse_l)))
        hist["val_mse"].append(vl)
        hist["kl"].append(float(np.mean(kl_l)))
        hist["beta"].append(beta)

        # Early-stopping fix (alg:training-loop): reset at t == E_warm.
        if ep == cfg.warmup:
            best_val = np.inf
            patience = 0
        if vl < best_val - 1e-6:
            best_val, best_w, patience = vl, [w.copy() for w in model.get_weights()], 0
        elif ep >= cfg.warmup:
            patience += 1
        if ep % 10 == 0:
            pbar.set_postfix({"val_mse": f"{vl:.4f}", "beta": f"{beta:.2f}",
                              "kl": f"{np.mean(kl_l):.1f}"})
        if patience >= cfg.patience and ep >= cfg.warmup:
            print(f"\n  Early stop @ ep {ep + 1}")
            break

    if best_w:
        model.set_weights(best_w)
    return hist
