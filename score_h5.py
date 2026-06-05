"""Mode 2: score pre-tiled tile bags from an ROSE-processing-v3 ``.h5`` directory.

Reads every matching bag file from ``--bag-root / <slide>``, scores every tile
with the trained ``PatchClassifier``, then renders the subset of
visualizations that still make sense without full slide coverage:

* top-tiles grid (+ poster variant) and top-6 row,
* full-slide thumbnail-backed score heatmap (missing tiles render as 0),
* histogram of tile scores normalized to the total number of scored tiles.

The single-FOV heatmap is intentionally omitted — sliding-window dense scoring
inside one FOV is not possible from bag tiles alone.

Example:
    python score_h5.py \\
        --bag-root /media/data1/kanghyun/ROSE_MIL/DUMC \\
        --slide-root /media/Wednesday/Temporary/amey/DUMC_second \\
        --slide-list '[CF15-000507A-6]' \\
        --checkpoint results/.../last.ckpt \\
        --output-dir ./claude/score_h5
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from jsonargparse import CLI
from tqdm import tqdm

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


def _write_scores_csv(
    csv_path: Path,
    cams_np: np.ndarray,
    coords_np: np.ndarray,
    scores_np: np.ndarray,
) -> None:
    """Write per-tile ``FOV_y,FOV_x,pixel_y,pixel_x,score`` rows to ``csv_path``.

    Args:
        csv_path: Output CSV path.
        cams_np: ``(N, 2)`` int array of ``(fov_y, fov_x)`` per tile.
        coords_np: ``(N, 2)`` int array of ``(pixel_y, pixel_x)`` per tile.
        scores_np: ``(N,)`` float array of tile scores.
    """
    table = np.column_stack(
        [
            cams_np[:, 0].astype(np.int64),
            cams_np[:, 1].astype(np.int64),
            coords_np[:, 0].astype(np.int64),
            coords_np[:, 1].astype(np.int64),
            scores_np.astype(np.float64),
        ]
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="FOV_y,FOV_x,pixel_y,pixel_x,score",
        comments="",
        fmt=("%d", "%d", "%d", "%d", "%.6f"),
    )


def _score_one_slide(
    bag_dir: Path,
    slide_path: Path,
    output_dir: Path,
    device: torch.device,
    model: torch.nn.Module,
    bag_pattern: str,
    batch_size: int,
    tile_size: int,
    pixel_border: int,
    fov_px: int,
    export_individual: bool,
    write_csv: bool,
) -> None:
    """Score every tile in one bag directory and write its visualizations.

    Args:
        bag_dir: Directory containing ``bag_*.h5`` files for this slide.
        slide_path: Slide directory containing ``fullslidescan_AIF_WB.nc``
            (used only to build the thumbnail behind the score heatmap).
        output_dir: Root output directory; results land in
            ``output_dir / slide_path.name / *``.
        device: Resolved inference device.
        model: Loaded ``PatchClassifier`` in eval mode.
        bag_pattern: Glob pattern matched under ``bag_dir``.
        batch_size: Tiles per inference batch.
        tile_size: Tile edge length (must match what the bags were tiled with).
        pixel_border: Pixel border used when the bags were tiled.
        fov_px: Per-FOV side length in the thumbnail mosaic.
        export_individual: Also save unannotated panels alongside combined figures.
        write_csv: When ``True``, dump per-tile scores as a CSV.
    """
    slide_name = slide_path.name

    out_dir = output_dir / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_bag_dir(bag_dir, pattern=bag_pattern)
    images_np = data["images"]
    cams_np = data["cams"]
    coords_np = data["coords"]
    log.info("Total tiles: %d", images_np.shape[0])

    # Stream batches lazily so only one batch is on the GPU at a time; building
    # the full list of preprocessed GPU tensors OOMs on large multi-bag slides.
    batches = (
        to_model_batch(images_np[start : start + batch_size], device)
        for start in range(0, images_np.shape[0], batch_size)
    )
    scores_np = score_tile_batches(model, batches, device)
    log.info(
        "Scored %d tiles  min=%.3f  max=%.3f  mean=%.3f",
        len(scores_np),
        float(scores_np.min()),
        float(scores_np.max()),
        float(scores_np.mean()),
    )

    if write_csv:
        csv_path = out_dir / f"{slide_name}_scores.csv"
        _write_scores_csv(csv_path, cams_np, coords_np, scores_np)
        log.info("Wrote per-tile CSV: %s", csv_path)

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


def main(
    bag_root: Path,
    slide_root: Path,
    slide_list: list[str],
    checkpoint: Path,
    output_dir: Path,
    bag_pattern: str = "bag_*.h5",
    gpu: int | None = None,
    batch_size: int = 1024,
    tile_size: int = 256,
    pixel_border: int = 0,
    fov_px: int = 160,
    export_individual: bool = False,
    verbose: bool = False,
    write_csv: bool = False,
) -> None:
    """Score one or more bag directories and write Mode-2 visualizations.

    Args:
        bag_root: Parent directory containing per-slide bag subdirectories.
        slide_root: Parent directory containing slide ``.nc`` subdirectories.
        slide_list: Names of subdirectories present under both ``bag_root``
            and ``slide_root``. Each slide is scored and written to
            ``output_dir / <name> / *``.
        checkpoint: Trained ``PatchClassifier`` checkpoint (``.ckpt``).
        output_dir: Root output directory.
        bag_pattern: Glob pattern matched under each bag directory.
        gpu: GPU index. ``None`` auto-picks CUDA; ``-1`` forces CPU.
        batch_size: Tiles per inference batch.
        tile_size: Tile edge length (must match what the bags were tiled with).
        pixel_border: Pixel border used when the bags were tiled (for mosaic
            grid alignment).
        fov_px: Per-FOV side length in the thumbnail mosaic.
        export_individual: Also save unannotated panels alongside the combined
            overview figure.
        verbose: When ``True``, emit detailed per-slide tile-count and score
            stats. Default is to log only warnings/errors plus the per-slide
            progress bar.
        write_csv: When ``True``, also dump per-tile
            ``FOV_y,FOV_x,pixel_y,pixel_x,score`` CSV per slide.
    """
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

    device = resolve_device(gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_classifier(checkpoint, device)
    log.info("Loaded model on %s", device)

    for slide_name in tqdm(slide_list, desc="Slides"):
        bag_dir = bag_root / slide_name
        slide_path = slide_root / slide_name
        try:
            _score_one_slide(
                bag_dir=bag_dir,
                slide_path=slide_path,
                output_dir=output_dir,
                device=device,
                model=model,
                bag_pattern=bag_pattern,
                batch_size=batch_size,
                tile_size=tile_size,
                pixel_border=pixel_border,
                fov_px=fov_px,
                export_individual=export_individual,
                write_csv=write_csv,
            )
        except Exception:
            log.exception("Failed to score slide %s; continuing", slide_name)


if __name__ == "__main__":
    CLI(main)
