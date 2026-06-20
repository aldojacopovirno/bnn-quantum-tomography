"""Plot style, color palettes and figure saving.

Configures matplotlib to a serif, minimal-spine style (STIXGeneral math font)
consistent with the thesis figures, and exposes the global color palettes used
by the figure generators.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# BNN-variant colors and labels.
COLORS = {"A": "#1565C0", "B": "#C62828", "C": "#2E7D32"}
LABELS_BNN = {"A": "BNN-Reparam", "B": "BNN-Flipout", "C": "BNN-Last"}

# Baseline method colors and markers.
METH_COLORS = {"LI": "#757575", "MLE": "#EF6C00", "NN-det": "#2E7D32"}
MARKERS = {"LI": "s", "MLE": "^", "NN-det": "D", "BNN": "o"}

# Nominal coverage levels for reliability diagrams / ECE.
LEVELS = np.arange(0.05, 1.0, 0.05)


def apply_style() -> None:
    """Apply the thesis matplotlib/seaborn style globally."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "figure.constrained_layout.use": False,
    })
    sns.set_style("ticks")


def save_fig(name: str, plots_dir, fig=None) -> None:
    """Save a figure as both PDF (vector) and PNG (300 DPI).

    Parameters
    ----------
    name : str
        Figure base name (without extension).
    plots_dir : str or pathlib.Path
        Output directory.
    fig : matplotlib.figure.Figure, optional
        Figure to save (defaults to the current figure).
    """
    if fig is None:
        fig = plt.gcf()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    base = plots_dir / name
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
