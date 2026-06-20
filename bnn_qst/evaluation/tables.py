"""Result-table formatting and CSV export.

Provides a pretty-printer for the per-method summary table and writes the
canonical CSV deliverables (``results_3q.csv``, ``results_sweep.csv``) consumed
by the thesis (Appendix B, ``tab:bnn-varianti-estesa``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DISPLAY_COLS = {
    "fid_mean": "F(med)", "fid_std": "F(std)", "td_mean": "TD(med)",
    "viol_pct": "Viol.(%)", "coverage": "Cov.95%", "ece": "ECE",
    "sharp": "Sharp.",
}


def format_results(df: pd.DataFrame) -> pd.DataFrame:
    """Format a results DataFrame for console display.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw results (one row per method, indexed by ``method``).

    Returns
    -------
    pandas.DataFrame
        Formatted copy with renamed columns and string-formatted values.
    """
    d = df[[c for c in _DISPLAY_COLS if c in df.columns]].rename(
        columns=_DISPLAY_COLS).copy()
    for c in ["F(med)", "F(std)", "TD(med)", "Cov.95%", "ECE", "Sharp."]:
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    if "Viol.(%)" in d:
        d["Viol.(%)"] = d["Viol.(%)"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    return d


def write_csv(df: pd.DataFrame, path) -> Path:
    """Write a DataFrame to CSV, creating parent directories.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    path : str or pathlib.Path
        Destination path.

    Returns
    -------
    pathlib.Path
        Resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path
