"""BNN-QST: Bayesian Neural Networks for Quantum State Tomography.

Modular implementation of the 3-qubit experiment of the thesis *Quantificazione
dell'Incertezza nelle Reti Neurali Bayesiane per la Tomografia degli Stati
Quantistici*. The canonical configuration and scientific parameters live in
:mod:`bnn_qst.config`; the pipeline is orchestrated by
:mod:`bnn_qst.pipeline.stages` and ``main.py``.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "1.0.0"
