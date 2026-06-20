"""Figure generators reproducing the 12 thesis plots (3-qubit).

Each generator consumes a single ``art`` artifact dictionary produced by the
``eval`` pipeline stage, so the plotting stage is model-free and purely
array-driven. Figures are saved as PDF + PNG via :func:`style.save_fig`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
import seaborn as sns

from ..config import Config
from .style import COLORS, LABELS_BNN, LEVELS, MARKERS, METH_COLORS, save_fig


def plot_learning_curves(art: dict, plots_dir, cfg: Config) -> None:
    """Learning curves (MSE, KL, beta, generalisation gap) -> ``lc_3q``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with histories ``hist_A``/``hist_B``/``hist_C``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused, kept for API uniformity).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for k, hist in [("A", art["hist_A"]), ("B", art["hist_B"]), ("C", art["hist_C"])]:
        ep = np.arange(len(hist["val_mse"]))
        axes[0].semilogy(ep, hist["tr_mse"], color=COLORS[k], lw=0.9, ls="--", alpha=0.7)
        axes[0].semilogy(ep, hist["val_mse"], color=COLORS[k], lw=1.2, label=LABELS_BNN[k])
        axes[1].semilogy(ep, hist["kl"], color=COLORS[k], lw=1.2, label=LABELS_BNN[k])
        axes[2].plot(ep, hist["beta"], color=COLORS[k], lw=1.2, label=LABELS_BNN[k])
        gap = np.array(hist["val_mse"]) - np.array(hist["tr_mse"])
        axes[3].plot(ep, gap, color=COLORS[k], lw=1.2, label=LABELS_BNN[k])
    axes[0].set(xlabel="Epoch", ylabel="MSE", title=r"MSE (train ---, val $\mathbf{---}$)")
    axes[1].set(xlabel="Epoch", ylabel=r"$D_{\mathrm{KL}}[q||p]$", title="KL divergence")
    axes[2].set(xlabel="Epoch", ylabel=r"$\beta$", title=r"$\beta$ (KL annealing)")
    axes[3].axhline(0, color="k", lw=0.4, ls="--")
    axes[3].set(xlabel="Epoch", ylabel=r"$\Delta$MSE", title="Generalisation gap")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves - BNN 3-qubit", fontsize=11, y=1.01)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("lc_3q", plots_dir, fig)
    plt.close(fig)


def plot_reliability(art: dict, plots_dir, cfg: Config) -> None:
    """Reliability diagram (3 variants, post-temperature-scaling) -> ``rd_3q``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``r_mc_*``, ``t*``, ``r_te``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    from ..inference.calibration import coverage, scale_samples

    r_te = art["r_te"]
    mc_map = {"BNN-Reparam": (art["r_mc_A"], art["tA"]),
              "BNN-Flipout": (art["r_mc_B"], art["tB"]),
              "BNN-Last": (art["r_mc_C"], art["tC"])}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    x_band = np.array([0.0, 1.0])
    ax.fill_between(x_band, x_band - 0.05, x_band + 0.05, alpha=0.12,
                    color="#546E7A", zorder=0, label=r"$\pm5\%$")
    ax.plot([0, 1], [0, 1], color="#546E7A", lw=0.8, ls="--", zorder=1)
    for k, rname in [("A", "BNN-Reparam"), ("B", "BNN-Flipout"), ("C", "BNN-Last")]:
        rs, temp = mc_map[rname]
        rc = scale_samples(rs, temp)
        emp = [coverage(rc, r_te, lv) for lv in LEVELS]
        ece_v = float(np.mean(np.abs(np.array(emp) - LEVELS)))
        ax.plot(LEVELS, emp, "o-", color=COLORS[k], ms=4, lw=1.0,
                label=f"{LABELS_BNN[k]}  ECE={ece_v:.3f}")
    ax.set(xlabel="Nominal level", ylabel="Empirical coverage",
           title="Reliability diagram - 3-qubit", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("rd_3q", plots_dir, fig)
    plt.close(fig)


def plot_fid_vs_shots(art: dict, plots_dir, cfg: Config) -> None:
    """Mean fidelity vs ``N_shots`` (log-x, error bars) -> ``fid_vs_shots``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``df_sweep`` and ``best_bnn_name``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (uses ``shots_sweep``, ``gate_fid``).
    """
    df_sweep = art["df_sweep"]
    best = art["best_bnn_name"]
    mc_clr = {**METH_COLORS, best: COLORS["A"]}
    mc_mk = {"LI": "s", "MLE": "^", "NN-det": "D", best: "o"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for meth in ["LI", "MLE", "NN-det", best]:
        sub = df_sweep[df_sweep["method"] == meth]
        clr = mc_clr.get(meth, "#2196F3")
        mk = mc_mk.get(meth, "o")
        yerr = sub["fid_std"].values if "fid_std" in sub else np.zeros(len(sub))
        ax.errorbar(sub["n_shots"], sub["fid_mean"], yerr=yerr, fmt=f"{mk}-",
                    color=clr, lw=1.1, ms=5, capsize=3, elinewidth=0.7, label=meth)
    ax.axhline(cfg.gate_fid, color="#B71C1C", lw=0.8, ls="--", alpha=0.7,
               label=f'Target $\\bar{{F}}={cfg.gate_fid}$')
    ax.set_xscale("log")
    ax.set_xticks(cfg.shots_sweep)
    ax.set_xticklabels([str(n) for n in cfg.shots_sweep], fontsize=8)
    ax.set(xlabel=r"$N_{\mathrm{shots}}$  [log]",
           ylabel=r"$\bar{F}(\hat{\rho}, \rho_{\mathrm{true}})$",
           title="Mean fidelity vs measurement budget - 3-qubit")
    ax.legend(loc="lower right")
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("fid_vs_shots", plots_dir, fig)
    plt.close(fig)


def plot_unc_vs_shots(art: dict, plots_dir, cfg: Config) -> None:
    """UQ metrics vs ``N_shots`` (4 panels) -> ``unc_vs_shots``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``df_sweep`` and ``best_bnn_name``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    best = art["best_bnn_name"]
    bnn_sw = art["df_sweep"][art["df_sweep"]["method"] == best].copy()
    shots = bnn_sw["n_shots"].values

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    specs = [
        ("sharp", r"Sharpness $\langle w_{\mathrm{CI95}}\rangle$", "#1565C0", None),
        ("epi_var", r"Epistemic var $\sum_i \mathrm{Var}[r_i]$", "#C62828", None),
        ("ece", "ECE", "#EF6C00", 0.0),
        ("coverage", r"Coverage 95% $\hat{c}$", "#2E7D32", 0.95),
    ]
    for ax, (col, ylabel, clr, hline) in zip(axes, specs):
        if col not in bnn_sw.columns:
            ax.set_visible(False)
            continue
        ax.plot(shots, bnn_sw[col], "o-", color=clr, ms=4, lw=1.2)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks(shots)
        ax.set_xticklabels([str(n) for n in shots], fontsize=7, rotation=30)
        ax.set(xlabel=r"$N_{\mathrm{shots}}$", ylabel=ylabel)
    fig.suptitle(f"Epistemic coherence - {best} (3-qubit)", fontsize=11)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("unc_vs_shots", plots_dir, fig)
    plt.close(fig)


def plot_pred_special(art: dict, plots_dir, cfg: Config) -> None:
    """Predictive distribution on notable states (4x4 grid) -> ``pred_special``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``special`` (name -> ``r_s_mc`` (1,M,K)),
        ``special_true`` (name -> r_true) and ``special_comps``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    comps = art["special_comps"]
    clbl = [f"$r_{{{c + 1}}}$" for c in comps]
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    for col, sname in enumerate(art["special"].keys()):
        samp = art["special"][sname][0]  # (M, K)
        r_true = art["special_true"][sname]
        for row, ci in enumerate(comps):
            ax = axes[row, col]
            data = samp[:, ci]
            ax.hist(data, bins=25, color=COLORS["A"], alpha=0.45, density=True, linewidth=0)
            kde_x = np.linspace(data.min() - 0.1 * max(data.std(), 1e-6),
                                data.max() + 0.1 * max(data.std(), 1e-6), 200)
            kde_y = sp_stats.gaussian_kde(data)(kde_x)
            ax.plot(kde_x, kde_y, color=COLORS["A"], lw=1.2)
            lo, hi = np.percentile(data, [2.5, 97.5])
            ax.axvspan(lo, hi, alpha=0.18, color="#546E7A")
            ax.axvline(r_true[ci], color="#B71C1C", lw=1.4,
                       label="True" if (row == 0 and col == 0) else "")
            ax.axvline(data.mean(), color=COLORS["A"], lw=0.8, ls="--")
            ax.set(yticks=[], xlabel=clbl[row] if row == 3 else "")
            if col == 0:
                ax.set_ylabel(clbl[row], fontsize=9)
            if row == 0:
                ax.set_title(sname, fontsize=10, pad=4)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("MC predictive distribution on notable states - best BNN (3-qubit)",
                 fontsize=11, y=1.01)
    sns.despine(fig=fig, left=True)
    plt.tight_layout()
    save_fig("pred_special", plots_dir, fig)
    plt.close(fig)


def plot_unc_vs_purity(art: dict, plots_dir, cfg: Config) -> None:
    """Epistemic variance vs purity scatter + regression -> ``unc_vs_purity``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``purity_te`` and ``epi_per_state``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    purity = art["purity_te"]
    epi = art["epi_per_state"]
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(purity, epi)
    r_sq = r_val ** 2

    fig, ax = plt.subplots(figsize=(6, 4.8))
    is_pure = purity > 0.999
    ax.scatter(purity[~is_pure], epi[~is_pure], s=6, alpha=0.25, color="#546E7A",
               label="Mixed", rasterized=True)
    ax.scatter(purity[is_pure], epi[is_pure], s=8, alpha=0.5, color="#1565C0",
               label="Pure", rasterized=True)
    x_reg = np.linspace(purity.min(), purity.max(), 200)
    ax.plot(x_reg, slope * x_reg + intercept, color="#B71C1C", lw=1.0, ls="--",
            label=f"Regression ($r^2={r_sq:.3f}$, $p={p_val:.2e}$)")
    ax.set(xlabel=r"Purity  $\mathrm{Tr}(\rho^2)$",
           ylabel=r"Epistemic var  $\sum_i \mathrm{Var}[r_i]$",
           title=f"Uncertainty vs purity - {art['best_bnn_name']}")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("unc_vs_purity", plots_dir, fig)
    plt.close(fig)


def plot_werner_sweep(art: dict, plots_dir, cfg: Config) -> None:
    """Werner-state sweep (fidelity + epistemic variance) -> ``werner_sweep``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``df_werner`` and ``best_bnn_name``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    df = art["df_werner"]
    best = art["best_bnn_name"]
    mc_clr = {**METH_COLORS, best: COLORS["A"]}
    mc_mk = {"LI": "s", "MLE": "^", "NN-det": "D", best: "o"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for meth in ["LI", "MLE", "NN-det", best]:
        sub = df[df["method"] == meth]
        axes[0].plot(sub["p"], sub["fid_mean"], f"{mc_mk.get(meth, 'o')}-",
                     color=mc_clr.get(meth, "#9E9E9E"), ms=5, lw=1.2, label=meth)
        if meth == best and "epi_var" in sub:
            axes[1].plot(sub["p"], sub["epi_var"], f"{mc_mk.get(meth, 'o')}-",
                         color=mc_clr.get(meth, "#9E9E9E"), ms=5, lw=1.2, label=meth)
    axes[0].set(xlabel=r"$p$", ylabel=r"Fidelity $F$", title="Fidelity - Werner sweep")
    axes[0].legend()
    axes[1].set(xlabel=r"$p$", ylabel=r"Epistemic var $\sum_i \mathrm{Var}[r_i]$",
                title="BNN uncertainty - Werner sweep")
    axes[1].legend()
    fig.suptitle(r"Werner states  $\rho_W(p)=p|\mathrm{GHZ}\rangle\langle\mathrm{GHZ}|+(1-p)I/8$",
                 fontsize=10, y=1.01)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("werner_sweep", plots_dir, fig)
    plt.close(fig)


def plot_fid_violin(art: dict, plots_dir, cfg: Config) -> None:
    """Per-state fidelity distribution (violin + jitter) -> ``fid_violin``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with per-method predicted/true density matrices.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (uses ``gate_fid``).
    """
    from ..quantum.metrics import fidelity_batch

    best = art["best_bnn_name"]
    rho_map_best = {"BNN-Reparam": art["rho_map_A"],
                    "BNN-Flipout": art["rho_map_B"],
                    "BNN-Last": art["rho_map_C"]}[best]
    methods_vio = {"LI": (art["rho_li"], art["rho_te"]),
                   "MLE": (art["rho_mle"], art["rho_te_sub"]),
                   "NN-det": (art["rho_det_pred"], art["rho_te"]),
                   best: (rho_map_best, art["rho_te"])}
    fid_data, names = [], []
    for meth, (rp, rt) in methods_vio.items():
        fid_data.append(fidelity_batch(rp, rt))
        names.append(meth)

    clrs = [METH_COLORS.get(n, COLORS["A"]) for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(fid_data, positions=range(len(names)),
                          showmedians=True, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(clrs[i])
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("#212121")
    parts["cmedians"].set_linewidth(1.2)
    rng42 = np.random.default_rng(42)
    for i, fids in enumerate(fid_data):
        xj = rng42.uniform(-0.12, 0.12, len(fids))
        ax.scatter(i + xj, fids, s=2, alpha=0.15, color=clrs[i], rasterized=True)
    ax.axhline(cfg.gate_fid, color="#B71C1C", lw=0.8, ls="--",
               label=f"Target F={cfg.gate_fid}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set(ylabel=r"$F(\hat{\rho}, \rho_{\mathrm{true}})$",
           title=r"Fidelity distribution per method ($N_{\mathrm{shots}}=5000$, 3-qubit)")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("fid_violin", plots_dir, fig)
    plt.close(fig)


def plot_viol_minlambda(art: dict, plots_dir, cfg: Config) -> None:
    """Minimum-eigenvalue distribution (KDE) per method -> ``viol_minlambda``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with per-method predicted density matrices.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    best = art["best_bnn_name"]
    rho_map_best = {"BNN-Reparam": art["rho_map_A"],
                    "BNN-Flipout": art["rho_map_B"],
                    "BNN-Last": art["rho_map_C"]}[best]
    methods_eig = {"LI": art["rho_li"], "MLE": art["rho_mle"],
                   "NN-det": art["rho_det_pred"], best: rho_map_best}
    clr_eig = {"LI": METH_COLORS["LI"], "MLE": METH_COLORS["MLE"],
               "NN-det": METH_COLORS["NN-det"], best: COLORS["A"]}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for meth, rp in methods_eig.items():
        min_eig = np.linalg.eigvalsh(rp)[:, 0]
        clr = clr_eig.get(meth, "#9E9E9E")
        kde_x = np.linspace(min_eig.min() - 0.005, max(min_eig.max(), 0.05), 300)
        kde_y = sp_stats.gaussian_kde(min_eig)(kde_x)
        ax.plot(kde_x, kde_y, lw=1.4, color=clr,
                label=f"{meth}  ({100 * (min_eig < 0).mean():.1f}% neg.)")
        ax.fill_between(kde_x[kde_x < 0], kde_y[kde_x < 0], alpha=0.15, color=clr)
    ax.axvline(0, color="k", lw=0.7, ls="--", label=r"$\lambda=0$")
    ax.set(xlabel=r"$\lambda_{\min}(\hat{\rho})$", ylabel="KDE density",
           title=r"$\lambda_{\min}(\hat{\rho})$ distribution - 3-qubit")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("viol_minlambda", plots_dir, fig)
    plt.close(fig)


def plot_fid_vs_purity(art: dict, plots_dir, cfg: Config) -> None:
    """Fidelity vs purity (smoothed) per method -> ``fid_vs_purity``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with per-method predicted/true density matrices
        and ``purity_te``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    from ..quantum.metrics import fidelity_batch

    best = art["best_bnn_name"]
    rho_map_best = {"BNN-Reparam": art["rho_map_A"],
                    "BNN-Flipout": art["rho_map_B"],
                    "BNN-Last": art["rho_map_C"]}[best]
    purity = art["purity_te"]
    methods_fp = {"LI": (art["rho_li"], art["rho_te"]),
                  "NN-det": (art["rho_det_pred"], art["rho_te"]),
                  best: (rho_map_best, art["rho_te"])}
    sort_idx = np.argsort(purity)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for meth, (rp, rt) in methods_fp.items():
        fids = fidelity_batch(rp, rt)
        fs = fids[sort_idx]
        ps = purity[sort_idx]
        sm = uniform_filter1d(fs, size=max(1, len(fs) // 50))
        clr = METH_COLORS.get(meth, COLORS["A"])
        ax.plot(ps, sm, color=clr, lw=1.3, label=meth)
        ax.scatter(ps[::20], fs[::20], s=3, alpha=0.2, color=clr)
    ax.set(xlabel=r"Purity  $\mathrm{Tr}(\rho^2)$",
           ylabel=r"Fidelity  $F(\hat{\rho},\rho)$",
           title="Fidelity vs purity per method - 3-qubit")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("fid_vs_purity", plots_dir, fig)
    plt.close(fig)


def plot_cal_temp_effect(art: dict, plots_dir, cfg: Config) -> None:
    """Pre/post temperature-scaling coverage (paired bars) -> ``cal_temp_effect``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``cov_pre_*``, ``r_mc_*``, ``t*``, ``r_te``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (uses ``ci``).
    """
    from ..inference.calibration import coverage, scale_samples

    variants = ["BNN-Reparam", "BNN-Flipout", "BNN-Last"]
    mc_te = {"BNN-Reparam": art["r_mc_A"], "BNN-Flipout": art["r_mc_B"],
             "BNN-Last": art["r_mc_C"]}
    t_map = {"BNN-Reparam": art["tA"], "BNN-Flipout": art["tB"], "BNN-Last": art["tC"]}
    c_pre = {"BNN-Reparam": art["cov_pre_A"], "BNN-Flipout": art["cov_pre_B"],
             "BNN-Last": art["cov_pre_C"]}
    r_te = art["r_te"]

    x = np.arange(len(variants))
    width = 0.35
    pre = [c_pre[v] for v in variants]
    post = [coverage(scale_samples(mc_te[v], t_map[v]), r_te, cfg.ci) for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, pre, width, color="#B0BEC5", label="Pre-calibration")
    ax.bar(x + width / 2, post, width, color=list(COLORS.values())[:3], alpha=0.85,
           label="Post-calibration")
    ax.axhline(cfg.ci, color="#B71C1C", lw=0.9, ls="--", label=f"Target={cfg.ci:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9)
    ax.set(ylabel="Empirical 95% coverage",
           title="Temperature scaling effect - 3-qubit", ylim=(0.5, 1.05))
    for bars, vals in [([ax.patches[i] for i in range(3)], pre),
                       ([ax.patches[i] for i in range(3, 6)], post)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    plt.tight_layout()
    save_fig("cal_temp_effect", plots_dir, fig)
    plt.close(fig)


def plot_pauli_bias_std(art: dict, plots_dir, cfg: Config) -> None:
    """Bias and std of the 63 Pauli components (heatmaps) -> ``pauli_bias_std``.

    Parameters
    ----------
    art : dict
        Evaluation artifact with ``r_mc_best`` and ``r_te``.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration (unused).
    """
    r_pred_mean = art["r_mc_best"].mean(axis=1)  # (N_test, 63)
    bias_63 = (r_pred_mean - art["r_te"]).mean(axis=0)
    std_63 = r_pred_mean.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, grid, title, cmap in [
        (axes[0], bias_63.reshape(7, 9), r"Bias $\mathbb{E}[\hat{r}_i - r_i]$", "coolwarm"),
        (axes[1], std_63.reshape(7, 9), r"Std $\mathrm{Std}[\hat{r}_i]$", "viridis"),
    ]:
        vmax = np.abs(grid).max()
        kw = dict(vmin=-vmax, vmax=vmax) if cmap == "coolwarm" else dict(vmin=0)
        im = ax.imshow(grid, cmap=cmap, aspect="auto", **kw)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.set(xlabel="Column (0-8)", ylabel="Row (0-6)", title=title)
        ax.set_xticks(range(9))
        ax.set_yticks(range(7))
    fig.suptitle(r"Per-component error $r_i$ ($i=1\ldots63$) - best BNN 3-qubit",
                 fontsize=11)
    sns.despine(fig=fig, left=True, bottom=True)
    plt.tight_layout()
    save_fig("pauli_bias_std", plots_dir, fig)
    plt.close(fig)


def plot_all(art: dict, plots_dir, cfg: Config) -> None:
    """Generate all 12 thesis figures.

    Parameters
    ----------
    art : dict
        Evaluation artifact.
    plots_dir : str or pathlib.Path
        Output directory.
    cfg : Config
        Configuration.
    """
    plot_learning_curves(art, plots_dir, cfg)
    plot_reliability(art, plots_dir, cfg)
    # Sweep-dependent figures require the shot-sweep stage to have run.
    if art.get("df_sweep") is not None:
        plot_fid_vs_shots(art, plots_dir, cfg)
        plot_unc_vs_shots(art, plots_dir, cfg)
    else:
        print("[plot] Skipping fid_vs_shots / unc_vs_shots (run the 'sweep' stage first).")
    plot_pred_special(art, plots_dir, cfg)
    plot_unc_vs_purity(art, plots_dir, cfg)
    plot_werner_sweep(art, plots_dir, cfg)
    plot_fid_violin(art, plots_dir, cfg)
    plot_viol_minlambda(art, plots_dir, cfg)
    plot_fid_vs_purity(art, plots_dir, cfg)
    plot_cal_temp_effect(art, plots_dir, cfg)
    plot_pauli_bias_std(art, plots_dir, cfg)
