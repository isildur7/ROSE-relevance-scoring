"""GPU inference loop shared by both scoring entry points."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch

log = logging.getLogger(__name__)


def score_tile_batches(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    use_amp: bool | None = None,
) -> np.ndarray:
    """Run inference over a stream of pre-normalized tile batches.

    The model's ``forward`` already applies a sigmoid, so the returned values
    are scores in ``[0, 1]``.

    Args:
        model: A ``PatchClassifier`` (or compatible) in ``eval()`` mode.
        batches: Iterable of ``(B, 3, 224, 224)`` float32 tensors on any device;
            each batch is moved to ``device`` inside the loop.
        device: Target device for inference.
        use_amp: If ``None``, AMP is enabled when ``device.type == "cuda"``.

    Returns:
        1-D float32 ``np.ndarray`` of length ``sum(B_i)``, with one score per
        input tile in the same order they were yielded.
    """
    if use_amp is None:
        use_amp = device.type == "cuda"

    all_scores: list[np.ndarray] = []
    with torch.inference_mode(), torch.amp.autocast(device.type, enabled=use_amp):
        for batch in batches:
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch)
            scores = outputs.squeeze(1).float().cpu().numpy()
            all_scores.append(scores)

    if not all_scores:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(all_scores).astype(np.float32)
