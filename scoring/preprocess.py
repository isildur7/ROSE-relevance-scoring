"""Convert raw uint8 tiles into the normalized batch the classifier expects.

Two entry points:

- :func:`preprocess_tiles_gpu` — for tiles already on GPU (Mode 1's hot path);
  does resize + normalize entirely on device.
- :func:`to_model_batch` — for tiles that live on the CPU (Mode 2, h5 bags);
  stacks a numpy ``(N, H, W, 3)`` uint8 batch, ships it to the device, then
  runs the same GPU preprocessing.

Both produce ``(N, 3, 224, 224)`` float tensors normalized with the dataset
statistics from ``dataset.py``, byte-for-byte matching what training used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dataset import DATASET_MEAN, DATASET_STD

_NORM_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _norm_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the normalization mean/std tensors per device."""
    cached = _NORM_CACHE.get(device)
    if cached is None:
        mean = torch.tensor(DATASET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(DATASET_STD, device=device).view(1, 3, 1, 1)
        _NORM_CACHE[device] = (mean, std)
        return mean, std
    return cached


def preprocess_tiles_gpu(tiles_hwc_u8: torch.Tensor) -> torch.Tensor:
    """Resize + normalize a batch of uint8 GPU tiles.

    Args:
        tiles_hwc_u8: ``(N, H, W, 3)`` uint8 tensor on the target device.

    Returns:
        ``(N, 3, 224, 224)`` float32 tensor on the same device. The resize uses
        bicubic + antialias, matching ``torchvision.transforms.v2.Resize``
        behavior at inference. Normalization uses ``DATASET_MEAN/STD``.
    """
    if tiles_hwc_u8.ndim != 4 or tiles_hwc_u8.shape[-1] != 3:
        raise ValueError(
            f"preprocess_tiles_gpu expects (N,H,W,3) uint8, got shape "
            f"{tuple(tiles_hwc_u8.shape)} dtype {tiles_hwc_u8.dtype}"
        )
    x = tiles_hwc_u8.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(
        x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
    )
    mean, std = _norm_tensors(x.device)
    return (x - mean) / std


def to_model_batch(images_uint8_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """Stack a CPU numpy batch onto ``device`` and apply the GPU preprocessing.

    Args:
        images_uint8_hwc: ``(N, H, W, 3)`` uint8 numpy array (e.g. from h5).
        device: Target device for inference.

    Returns:
        ``(N, 3, 224, 224)`` float32 tensor on ``device``.
    """
    if images_uint8_hwc.ndim != 4 or images_uint8_hwc.shape[-1] != 3:
        raise ValueError(
            f"to_model_batch expects (N,H,W,3) uint8, got shape "
            f"{images_uint8_hwc.shape} dtype {images_uint8_hwc.dtype}"
        )
    t = torch.from_numpy(images_uint8_hwc)
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return preprocess_tiles_gpu(t)
