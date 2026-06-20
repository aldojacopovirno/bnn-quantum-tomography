"""CLI orchestrator for the BNN-QST experiment.

Single entry point that runs the pipeline stages (``data``, ``train``,
``eval``, ``sweep``, ``plot``) either individually or all together. Stages
persist their outputs under ``<results_dir>/artifacts/`` and can be resumed.

Examples
--------
Run the full canonical experiment::

    python main.py all --results-dir results

Run only the plotting stage from a previous evaluation::

    python main.py plot --results-dir results

Override the default shot budget::

    python main.py all --n-shots 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bnn_qst.config import Config
from bnn_qst.pipeline.stages import (run_all, run_data, run_eval, run_plot,
                                     run_sweep, run_train)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list, optional
        Argument list (defaults to ``sys.argv``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Neural Networks for Quantum State Tomography (3-qubit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stage", choices=["data", "train", "eval", "sweep", "plot", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Root directory for artifacts, tables and figures.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--n-shots", type=int, default=5000,
                        help="Default number of measurement shots per basis.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Build the canonical Config from CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    Config
        Frozen configuration instance.
    """
    return Config().with_overrides(seed=args.seed, n_shots=args.n_shots)


def main(argv=None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list, optional
        Argument list.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = parse_args(argv)
    cfg = build_config(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "data": run_data, "train": run_train, "eval": run_eval,
        "sweep": run_sweep, "plot": run_plot, "all": run_all,
    }
    if args.stage == "all":
        dispatch["all"](cfg, results_dir)
    else:
        dispatch[args.stage](cfg, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
