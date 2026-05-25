"""Mode 1: score every pen-mark-filtered tile in a full ``.nc`` slide.

Streams tiles via :func:`scoring.gpu_tiler.iter_tile_batches`, scores them with
the trained ``PatchClassifier``, then renders the suite of visualizations
(overview + score heatmap, top-tiles grid, top-6 row, single-FOV heatmap,
score histogram).

Example:
    python score_nc.py \\
        --slide-path /media/Wednesday/Temporary/amey/DUMC_second/CF15-000507A-6 \\
        --checkpoint results/.../last.ckpt \\
        --mask-folder /media/Wednesday/Temporary/kanghyun/new_pen_marks/mask \\
        --output-dir ./claude/score_nc
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from jsonargparse import CLI

from scoring._annotation import load_annotation
from scoring.gpu_tiler import iter_tile_batches, load_slide_images
from scoring.model_io import load_classifier, resolve_device
from scoring.preprocess import preprocess_tiles_gpu, to_model_batch
from scoring.scorer import score_tile_batches
from scoring.visualizations import (
    build_score_lookup,
    build_slide_arrays,
    fov_heatmap,
    overview_with_heatmap,
    score_histogram,
    top6_row,
    top_tiles_grid,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _score_fov_sliding_window(
    model: torch.nn.Module,
    device: torch.device,
    fov_image: np.ndarray,
    annotation: np.ndarray,
    tile_size: int,
    stride: int,
    pixel_border: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-score one FOV with overlapping tiles to build a per-pixel score map.

    Tiles overlapping any pen-marked pixel are skipped. Overlapping regions in
    the output ``score_map`` are averaged across the contributing tiles.

    Args:
        model: Loaded classifier in eval mode on ``device``.
        device: Inference device.
        fov_image: ``(H, W, 3)`` uint8 FOV.
        annotation: ``(H, W)`` uint8 pen-mark mask.
        tile_size: Tile edge length.
        stride: Sliding-window stride (``< tile_size`` for overlap).
        pixel_border: Pixel border dropped on each side of the FOV.
        batch_size: Tiles per forward pass.

    Returns:
        ``(scores, score_map)`` where ``scores`` is ``(N,)`` per-tile scores
        and ``score_map`` is ``(H, W)`` per-pixel average score.
    """
    h, w = fov_image.shape[:2]
    tiles: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for i in range(pixel_border, h - pixel_border - tile_size + 1, stride):
        for j in range(pixel_border, w - pixel_border - tile_size + 1, stride):
            window_ann = annotation[i : i + tile_size, j : j + tile_size]
            if window_ann.sum() != 0:
                continue
            tiles.append(fov_image[i : i + tile_size, j : j + tile_size])
            positions.append((i, j))

    if not tiles:
        return np.zeros((0,), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    def _batches() -> "list[torch.Tensor]":  # noqa: F821 — quoted for type-check
        out = []
        for start in range(0, len(tiles), batch_size):
            chunk = np.stack(tiles[start : start + batch_size], axis=0)
            out.append(to_model_batch(chunk, device))
        return out

    scores = score_tile_batches(model, _batches(), device)

    score_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for s, (i, j) in zip(scores, positions, strict=True):
        score_sum[i : i + tile_size, j : j + tile_size] += s
        count[i : i + tile_size, j : j + tile_size] += 1.0
    mask = count > 0
    score_sum[mask] /= count[mask]
    return scores, score_sum


def _fov_rgb_from_array(images_np: np.ndarray, fov_y: int, fov_x: int) -> np.ndarray:
    """Debayer one FOV from a pre-eager-loaded ``(Y, X, H, W)`` Bayer array."""
    raw = images_np[fov_y, fov_x]
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    return cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)


def _pick_best_fov(
    score_lookup: dict[tuple[int, int], dict[tuple[int, int], float]],
) -> tuple[int, int]:
    """Return the ``(fov_y, fov_x)`` whose tiles have the highest mean score."""
    best_key: tuple[int, int] | None = None
    best_mean = -1.0
    for key, tile_scores in score_lookup.items():
        if not tile_scores:
            continue
        m = float(np.mean(list(tile_scores.values())))
        if m > best_mean:
            best_mean = m
            best_key = key
    if best_key is None:
        raise RuntimeError("No FOV with scored tiles to visualize.")
    return best_key


def main(
    slide_path: Path,
    checkpoint: Path,
    output_dir: Path,
    mask_folder: Path | None = None,
    gpu: int | None = None,
    batch_size: int = 1024,
    tile_size: int = 256,
    pixel_border: int = 0,
    x_border: int = 2,
    y_border: int = 2,
    chunk: int = 27,
    fov_px: int = 160,
    fov_y: int | None = None,
    fov_x: int | None = None,
    fov_stride: int = 16,
    export_individual: bool = False,
) -> None:
    """Score a full ``.nc`` slide and write all visualizations.

    Args:
        slide_path: Slide directory containing ``fullslidescan_AIF_WB.nc``.
        checkpoint: Trained ``PatchClassifier`` checkpoint (``.ckpt``).
        output_dir: Root output directory. Results are written under
            ``output_dir / slide_path.name / *``.
        mask_folder: Directory containing per-FOV pen-mark masks named
            ``<slide>_<row>_<col>.{tif,npy}``. Omit to score every tile.
        gpu: GPU index. ``None`` auto-picks CUDA; ``-1`` forces CPU
            (CPU is rejected by the GPU tiler and will error out).
        batch_size: Tiles per inference batch.
        tile_size: Tile edge length in pixels.
        pixel_border: Pixel border dropped inside each FOV before tiling.
        x_border: Number of border FOVs skipped on each side along x.
        y_border: Number of border FOVs skipped on each side along y.
        chunk: FOVs loaded onto GPU per streaming step.
        fov_px: Per-FOV side length in the full-slide mosaic figures.
        fov_y: Row of the FOV used for the per-FOV heatmap; auto-picked
            (highest mean score) when ``None``.
        fov_x: Column of the FOV used for the per-FOV heatmap; auto-picked
            (highest mean score) when ``None``.
        fov_stride: Sliding-window stride for the per-FOV heatmap.
        export_individual: Also save unannotated panels alongside combined figures.
    """
    device = resolve_device(gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    slide_name = slide_path.name

    out_dir = output_dir / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_model = time.time()
    model = load_classifier(checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_model = time.time() - t_model

    t_load = time.time()
    slide_images = load_slide_images(slide_path)
    t_load = time.time() - t_load
    log.info("Slide array: shape=%s dtype=%s", slide_images.shape, slide_images.dtype)

    all_images: list[np.ndarray] = []
    all_cams: list[np.ndarray] = []
    all_coords: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    log.info("Streaming + scoring slide %s on %s", slide_name, device)
    is_cuda = device.type == "cuda"

    def _sync() -> None:
        if is_cuda:
            torch.cuda.synchronize(device)

    t_score = time.time()
    t_iter_total = 0.0
    t_pre_total = 0.0
    t_score_total = 0.0
    t_collect_total = 0.0
    n_tiles_seen = 0

    gen = iter_tile_batches(
        slide_path,
        tile_size=tile_size,
        x_border=x_border,
        y_border=y_border,
        pixel_border=pixel_border,
        mask_folder=mask_folder,
        chunk=chunk,
        device=device,
        images_np=slide_images,
    )

    chunk_idx = 0
    while True:
        t0 = time.time()
        try:
            batch = next(gen)
        except StopIteration:
            break
        _sync()
        t_iter_total += time.time() - t0

        tiles_gpu = batch["tiles_gpu"]
        n_tiles_seen += tiles_gpu.shape[0]
        log.info(
            "  chunk %d: %d kept tiles (running total %d)",
            chunk_idx,
            tiles_gpu.shape[0],
            n_tiles_seen,
        )

        t0 = time.time()
        mini_batches: list[torch.Tensor] = []
        for start in range(0, tiles_gpu.shape[0], batch_size):
            mini_batches.append(
                preprocess_tiles_gpu(tiles_gpu[start : start + batch_size])
            )
        _sync()
        t_pre_total += time.time() - t0

        t0 = time.time()
        scores = score_tile_batches(model, mini_batches, device)
        _sync()
        t_score_total += time.time() - t0

        t0 = time.time()
        all_images.append(batch["images"])
        all_cams.append(batch["cams"])
        all_coords.append(batch["coords"])
        all_scores.append(scores)
        t_collect_total += time.time() - t0
        chunk_idx += 1

    _sync()
    t_score = time.time() - t_score

    if not all_scores:
        log.error("No tiles scored — nothing to visualize.")
        return

    tiles_np = np.concatenate(all_images, axis=0)
    cams_np = np.concatenate(all_cams, axis=0)
    coords_np = np.concatenate(all_coords, axis=0)
    scores_np = np.concatenate(all_scores, axis=0)
    log.info(
        "Scored %d tiles  min=%.3f  max=%.3f  mean=%.3f  (%.1f tiles/s)",
        len(scores_np),
        float(scores_np.min()),
        float(scores_np.max()),
        float(scores_np.mean()),
        len(scores_np) / max(t_score, 1e-6),
    )
    log.info(
        "Phase timings: model_load=%.2fs eager_load=%.2fs score_loop=%.2fs",
        t_model,
        t_load,
        t_score,
    )
    log.info(
        "  score_loop breakdown: iter=%.2fs preprocess=%.2fs inference=%.2fs collect=%.2fs",
        t_iter_total,
        t_pre_total,
        t_score_total,
        t_collect_total,
    )

    score_lookup = build_score_lookup(scores_np, cams_np, coords_np, tile_size)
    score_array, image_array = build_slide_arrays(
        score_lookup,
        slide_path,
        tile_size=tile_size,
        pixel_border=pixel_border,
        fov_px=fov_px,
        images_np=slide_images,
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
        tiles_np,
        cams_np,
        coords_np,
        scores_np,
        slide_name,
        out_dir / f"{slide_name}_top_tiles.png",
        poster_grid=True,
    )
    top6_row(tiles_np, scores_np, out_dir / f"{slide_name}_top6_row.png")

    score_histogram(
        scores_np,
        out_dir / f"{slide_name}_score_histogram.png",
        total=None,
        title=f"Tile score histogram - {slide_name} (n={len(scores_np)})",
    )

    if fov_y is None or fov_x is None:
        fov_y, fov_x = _pick_best_fov(score_lookup)
    log.info("Rendering per-FOV heatmap for FOV (%d, %d)", fov_y, fov_x)
    fov_image = _fov_rgb_from_array(slide_images, fov_y, fov_x)
    if mask_folder is not None:
        annotation = load_annotation(
            mask_folder, slide_name, fov_y, fov_x, fov_image.shape[:2]
        )
    else:
        annotation = np.zeros(fov_image.shape[:2], dtype=np.uint8)

    _, score_map = _score_fov_sliding_window(
        model,
        device,
        fov_image,
        annotation,
        tile_size=tile_size,
        stride=fov_stride,
        pixel_border=pixel_border,
        batch_size=batch_size,
    )
    fov_heatmap(
        fov_image,
        score_map,
        annotation,
        out_dir / f"{slide_name}_fov_{fov_y}_{fov_x}_heatmap.png",
        tile_size=tile_size,
        stride=fov_stride,
        export_individual=export_individual,
    )


if __name__ == "__main__":
    CLI(main)
