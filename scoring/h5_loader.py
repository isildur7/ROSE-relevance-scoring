"""Read pre-tiled tile bags from an ROSE-processing-v3 output directory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np

log = logging.getLogger(__name__)


class BagData(TypedDict):
    """In-memory view of one or more concatenated bag files."""

    images: np.ndarray  # (N, 256, 256, 3) uint8
    cams: np.ndarray  # (N, 2) int32 — (row, col) of the source FOV
    coords: np.ndarray  # (N, 2) int32 — (pixel_row, pixel_col) of tile within FOV


def load_bag_dir(directory: Path, pattern: str = "bag_*.h5") -> BagData:
    """Concatenate every matching bag file under ``directory`` into one dict.

    Args:
        directory: Directory containing ``bag_*.h5`` files produced by
            ROSE-processing-v3's ``Tiler``.
        pattern: Glob pattern matched directly under ``directory``.

    Returns:
        Dict with keys ``images``, ``cams``, ``coords`` concatenated along
        axis 0. Tile order follows the sorted glob order of bag files.

    Raises:
        FileNotFoundError: If no files match ``pattern`` under ``directory``.
        ValueError: If a bag file is missing one of the required datasets or
            has an unexpected per-sample shape.
    """
    bag_paths = sorted(directory.glob(pattern))
    if not bag_paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    log.info("Loading %d bag file(s) from %s", len(bag_paths), directory)

    images_chunks: list[np.ndarray] = []
    cams_chunks: list[np.ndarray] = []
    coords_chunks: list[np.ndarray] = []

    for path in bag_paths:
        with h5py.File(path, "r") as f:
            for key in ("images", "cams", "coords"):
                if key not in f:
                    raise ValueError(f"{path}: missing dataset {key!r}")
            images = f["images"][:]
            cams = f["cams"][:]
            coords = f["coords"][:]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"{path}: expected images shape (N,H,W,3), got {images.shape}"
            )
        if cams.shape[1] != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"{path}: cams/coords must be (N,2), got {cams.shape}/{coords.shape}"
            )
        images_chunks.append(images)
        cams_chunks.append(cams.astype(np.int32, copy=False))
        coords_chunks.append(coords.astype(np.int32, copy=False))
        log.info("  %s -> %d tiles", path.name, images.shape[0])

    return BagData(
        images=np.concatenate(images_chunks, axis=0),
        cams=np.concatenate(cams_chunks, axis=0),
        coords=np.concatenate(coords_chunks, axis=0),
    )
