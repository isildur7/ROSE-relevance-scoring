"""CPU loader for per-FOV pen-mark annotation masks.

Ported from ROSE-processing-v3's ``data_tiler/annotation.py``: each mask file
under ``<mask_folder>`` is named ``<slide>_<row>_<col>.{tif,npy}`` and is
normalized to a ``(H, W)`` uint8 array where nonzero means annotated. Missing
files become all-zero masks. The batch loader uses a small thread pool because
the work is I/O- and decode-bound.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from skimage import io


def load_annotation(
    mask_folder: Path,
    slide_name: str,
    row: int,
    col: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Load one FOV's annotation mask, padded/cropped to ``shape_hw``.

    Args:
        mask_folder: Directory containing ``<slide>_<row>_<col>.{tif,npy}``.
        slide_name: Slide identifier (slide directory name).
        row: FOV row index.
        col: FOV column index.
        shape_hw: Expected ``(H, W)``.

    Returns:
        ``(H, W)`` uint8 array. Nonzero pixels mark annotated regions.
    """
    tif_path = mask_folder / f"{slide_name}_{row}_{col}.tif"
    npy_path = mask_folder / f"{slide_name}_{row}_{col}.npy"
    h, w = shape_hw

    if npy_path.exists():
        ann = np.load(npy_path)
    elif tif_path.exists():
        ann = io.imread(tif_path)
    else:
        return np.zeros((h, w), dtype=np.uint8)

    if ann.ndim == 3:
        ann = ann[..., 0]
    ann = np.asarray(ann)

    if ann.shape != (h, w):
        padded = np.zeros((h, w), dtype=np.uint8)
        h2 = min(h, ann.shape[0])
        w2 = min(w, ann.shape[1])
        padded[:h2, :w2] = ann[:h2, :w2]
        ann = padded

    if ann.dtype != np.uint8:
        ann = (ann > 0).astype(np.uint8) * 255
    return ann


def load_annotation_batch(
    mask_folder: Path,
    slide_name: str,
    fovs: Iterable[tuple[int, int]],
    shape_hw: tuple[int, int],
    n_workers: int = 4,
) -> np.ndarray:
    """Load a batch of annotation masks in parallel into one ``(B, H, W)`` array.

    Args:
        mask_folder: Directory containing the mask files.
        slide_name: Slide identifier.
        fovs: Iterable of ``(row, col)`` FOV indices in the desired stack order.
        shape_hw: Per-FOV ``(H, W)``.
        n_workers: Thread-pool size for parallel decode.

    Returns:
        ``(B, H, W)`` uint8 array stacking the masks in the input order.
    """
    fovs = list(fovs)
    h, w = shape_hw
    out = np.zeros((len(fovs), h, w), dtype=np.uint8)
    if not fovs:
        return out

    def _one(item: tuple[int, tuple[int, int]]) -> None:
        i, (row, col) = item
        out[i] = load_annotation(mask_folder, slide_name, row, col, shape_hw)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_one, enumerate(fovs)))
    return out
