"""Helpers for loading the PatchClassifier checkpoint onto a chosen device."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from train import PatchClassifier

log = logging.getLogger(__name__)


def resolve_device(gpu: int | None) -> torch.device:
    """Pick a torch device from a CLI-style gpu argument.

    Args:
        gpu: Index of the GPU to use (e.g. 0, 1). ``None`` for auto-detect
            (CUDA if available, else CPU). ``-1`` forces CPU.

    Returns:
        Resolved ``torch.device``.
    """
    if gpu is not None and gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    if gpu == -1:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(checkpoint_path: Path, device: torch.device) -> PatchClassifier:
    """Load a ``PatchClassifier`` checkpoint in eval mode on ``device``.

    Args:
        checkpoint_path: Path to a ``.ckpt`` written by ``train.py``.
        device: Target device the model should live on.

    Returns:
        The loaded ``PatchClassifier`` in ``eval()`` mode and pinned to ``device``.
    """
    log.info("Loading classifier from %s on %s", checkpoint_path, device)
    model = PatchClassifier.load_from_checkpoint(
        str(checkpoint_path), weights_only=False, map_location=device
    )
    model.eval()
    model.to(device)
    return model
