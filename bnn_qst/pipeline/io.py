"""Artifact persistence for stage-based resumption.

Each pipeline stage writes its outputs to a ``results/artifacts`` directory and
later stages load what they need. Arrays use ``.npz``, structured objects use
pickle, and model weights are stored as ``.npz`` of flattened variables.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path, **arrays: np.ndarray) -> Path:
    """Save named arrays to a compressed ``.npz`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path (``.npz`` extension added if missing).
    **arrays : numpy.ndarray
        Named arrays to store.

    Returns
    -------
    pathlib.Path
        Resolved destination path.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_npz(path) -> dict:
    """Load a ``.npz`` file into a dictionary of arrays.

    Parameters
    ----------
    path : str or pathlib.Path
        Source path.

    Returns
    -------
    dict
        Mapping ``name -> numpy.ndarray``.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_pickle(path, obj: Any) -> Path:
    """Pickle an arbitrary Python object.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path.
    obj : object
        Object to serialize.

    Returns
    -------
    pathlib.Path
        Resolved destination path.
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path) -> Any:
    """Unpickle an object.

    Parameters
    ----------
    path : str or pathlib.Path
        Source path.

    Returns
    -------
    object
        Deserialized object.
    """
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_model_weights(path, model) -> Path:
    """Save a model's trainable variables to a ``.npz`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path (without extension).
    model : Net
        Network with ``get_weights``.

    Returns
    -------
    pathlib.Path
        Resolved destination path.
    """
    weights = model.get_weights()
    return save_npz(path, **{f"w{i}": np.asarray(w) for i, w in enumerate(weights)})


def load_model_weights(path, model) -> None:
    """Load trainable variables into a (already built) model.

    Parameters
    ----------
    path : str or pathlib.Path
        Source ``.npz`` path.
    model : Net
        Network whose variables have already been created by a forward call.
    """
    data = load_npz(path)
    weights = [data[f"w{i}"] for i in range(len(data))]
    model.set_weights(weights)
