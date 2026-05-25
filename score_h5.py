"""Mode 2: score pre-tiled tile bags from an ROSE-processing-v3 ``.h5`` directory.

Reads every matching bag file from ``--bag-dir``, scores every tile with the
trained ``PatchClassifier``, then renders the subset of visualizations that
still make sense without full slide coverage:

* top-tiles grid (+ poster variant) and top-6 row,
* full-slide thumbnail-backed score heatmap (missing tiles render as 0),
* histogram of tile scores normalized to the total number of scored tiles.

The single-FOV heatmap is intentionally omitted — sliding-window dense scoring
inside one FOV is not possible from bag tiles alone.

Example:
    python score_h5.py \\
        --bag-dir /media/data1/kanghyun/ROSE_MIL/DUMC/CF15-000507A-6 \\
        --slide-path /media/Wednesday/Temporary/amey/DUMC_second/CF15-000507A-6 \\
        --checkpoint results/.../last.ckpt \\
        --output-dir ./claude/score_h5
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from jsonargparse import CLI

from scoring.h5_loader import load_bag_dir
from scoring.model_io import load_classifier, resolve_device
from scoring.preprocess import to_model_batch
from scoring.scorer import score_tile_batches
from scoring.visualizations import (
    build_score_lookup,
    build_slide_arrays,
    overview_with_heatmap,
    score_histogram,
    top6_row,
    top_tiles_grid,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(
    bag_dir: Path,
    slide_path: Path,
    checkpoint: Path,
    output_dir: Path,
    bag_pattern: str = "bag_*.h5",
    gpu: int | None = None,
    batch_size: int = 1024,
    tile_size: int = 256,
    pixel_border: int = 0,
    fov_px: int = 160,
    export_individual: bool = False,
) -> None:
    """Score every tile in a bag directory and write Mode-2 visualizations.

    Args:
        bag_dir: Directory containing ``bag_*.h5`` files from
            ROSE-processing-v3's ``Tiler``.
        slide_path: Slide directory containing ``fullslidescan_AIF_WB.nc``
            (used only to build the thumbnail behind the score heatmap).
        checkpoint: Trained ``PatchClassifier`` checkpoint (``.ckpt``).
        output_dir: Root output directory; results land in
            ``output_dir / slide_path.name / *``.
        bag_pattern: Glob pattern matched under ``bag_dir``.
        gpu: GPU index. ``None`` auto-picks CUDA; ``-1`` forces CPU.
        batch_size: Tiles per inference batch.
        tile_size: Tile edge length (must match what the bags were tiled with).
        pixel_border: Pixel border used when the bags were tiled (for mosaic
            grid alignment).
        fov_px: Per-FOV side length in the thumbnail mosaic.
        export_individual: Also save unannotated panels alongside the combined
            overview figure.
    """
    device = resolve_device(gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    slide_name = slide_path.name

    out_dir = output_dir / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_classifier(checkpoint, device)
    data = load_bag_dir(bag_dir, pattern=bag_pattern)
    images_np = data["images"]
    cams_np = data["cams"]
    coords_np = data["coords"]
    log.info("Total tiles: %d", images_np.shape[0])

    mini_batches: list[torch.Tensor] = []
    for start in range(0, images_np.shape[0], batch_size):
        mini_batches.append(
            to_model_batch(images_np[start : start + batch_size], device)
        )
    scores_np = score_tile_batches(model, mini_batches, device)
    log.info(
        "Scored %d tiles  min=%.3f  max=%.3f  mean=%.3f",
        len(scores_np),
        float(scores_np.min()),
        float(scores_np.max()),
        float(scores_np.mean()),
    )

    score_lookup = build_score_lookup(scores_np, cams_np, coords_np, tile_size)
    score_array, image_array = build_slide_arrays(
        score_lookup,
        slide_path,
        tile_size=tile_size,
        pixel_border=pixel_border,
        fov_px=fov_px,
        device=device,
    )

    overview_with_heatmap(
        image_array,
        score_array,
        scores_np,
        slide_name,
        out_dir / f"{slide_name}_overview_heatmap.png",
        fov_px=fov_px,
        export_individual=export_individual,
    )

    top_tiles_grid(
        images_np,
        cams_np,
        coords_np,
        scores_np,
        slide_name,
        out_dir / f"{slide_name}_top_tiles.png",
        poster_grid=True,
    )
    top6_row(images_np, scores_np, out_dir / f"{slide_name}_top6_row.png")

    score_histogram(
        scores_np,
        out_dir / f"{slide_name}_score_histogram.png",
        total=int(scores_np.shape[0]),
        title=(
            f"Tile score histogram (normalized) - {slide_name} "
            f"(n={int(scores_np.shape[0])})"
        ),
    )


if __name__ == "__main__":
    CLI(main)
