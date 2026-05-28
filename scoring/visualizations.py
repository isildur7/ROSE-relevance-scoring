"""Source-agnostic plotting + per-FOV mosaic helpers shared by both entry scripts.

These functions take plain arrays / lookup dicts, never the original scorer
object, so they can be driven by either Mode 1 (full-slide from ``.nc``) or
Mode 2 (collated tile bags from ``.h5``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from owl.mcam_data import load as owl_load

log = logging.getLogger(__name__)

_SLIDE_FILENAME = "fullslidescan_AIF_WB.nc"


def _gpu_mosaic_chunks(
    images_np: np.ndarray,
    fov_px: int,
    device: torch.device,
    chunk_size: int = 27,
) -> np.ndarray:
    """Debayer + downsample every FOV on GPU; return ``(Y, X, fov_px, fov_px, 3)`` uint8."""
    import kornia.color as kc  # local import keeps module importable without CUDA

    # Release cached-but-unused allocations from scoring, then size each chunk
    # so it fits in available memory.  kornia raw_to_rgb peaks at ~13× a single
    # float32 channel per FOV (input + r/g/b intermediates + cat output).
    torch.cuda.empty_cache()
    fov_h, fov_w = images_np.shape[2], images_np.shape[3]
    bytes_per_fov = 13 * fov_h * fov_w * 4  # empirical kornia peak per 2048×2048 FOV
    free_bytes, _ = torch.cuda.mem_get_info(device)
    safe_chunk = max(1, int(free_bytes * 0.7 // bytes_per_fov))
    chunk_size = min(chunk_size, safe_chunk)
    log.info("mosaic chunk_size=%d (%.0f MiB free)", chunk_size, free_bytes / 2**20)

    y_max, x_max = images_np.shape[:2]
    out = np.zeros((y_max, x_max, fov_px, fov_px, 3), dtype=np.uint8)
    fovs: list[tuple[int, int]] = [(y, x) for y in range(y_max) for x in range(x_max)]

    cursor = 0
    while cursor < len(fovs):
        chunk_fovs = fovs[cursor : cursor + chunk_size]
        rows = np.fromiter((y for y, _ in chunk_fovs), dtype=np.int64)
        cols = np.fromiter((x for _, x in chunk_fovs), dtype=np.int64)
        bayer_chunk = np.ascontiguousarray(images_np[rows, cols])
        if bayer_chunk.ndim == 4 and bayer_chunk.shape[-1] == 1:
            bayer_chunk = bayer_chunk[..., 0]
        try:
            bayer_t = (
                torch.from_numpy(bayer_chunk).pin_memory().to(device, non_blocking=True)
            )
            x = bayer_t.unsqueeze(1).float() / 255.0  # (B, 1, H, W)
            del bayer_t
            rgb = kc.raw_to_rgb(x, kc.CFA.GR).clamp(0.0, 1.0)  # (B, 3, H, W)
            del x
            small = F.interpolate(rgb, size=(fov_px, fov_px), mode="area")
            del rgb
            small_u8 = (
                (small * 255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            del small
            for i, (y_idx, x_idx) in enumerate(chunk_fovs):
                out[y_idx, x_idx] = small_u8[i]
            cursor += len(chunk_fovs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
            log.warning("OOM in mosaic chunk; retrying with chunk_size=%d", chunk_size)
    return out


def _cpu_mosaic(images_np: np.ndarray, fov_px: int) -> np.ndarray:
    """CV2 debayer + resize fallback when CUDA isn't available."""
    y_max, x_max = images_np.shape[:2]
    out = np.zeros((y_max, x_max, fov_px, fov_px, 3), dtype=np.uint8)
    for y in range(y_max):
        for x in range(x_max):
            raw = images_np[y, x]
            if raw.ndim == 3 and raw.shape[-1] == 1:
                raw = raw[..., 0]
            rgb = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)
            out[y, x] = cv2.resize(rgb, (fov_px, fov_px), interpolation=cv2.INTER_AREA)
    return out


ScoreLookup = dict[tuple[int, int], dict[tuple[int, int], float]]


def build_score_lookup(
    scores: np.ndarray,
    cams: np.ndarray,
    coords: np.ndarray,
    tile_size: int,
) -> ScoreLookup:
    """Group per-tile scores by FOV into a nested ``{cam: {tile: score}}`` dict.

    Args:
        scores: ``(N,)`` float array of tile scores.
        cams: ``(N, 2)`` int array of ``(fov_row, fov_col)`` per tile.
        coords: ``(N, 2)`` int array of ``(pixel_row, pixel_col)`` per tile.
        tile_size: Tile edge length in pixels (used to convert pixel coords
            into tile-grid indices).

    Returns:
        Nested dict: ``lookup[(cam_y, cam_x)][(tile_y, tile_x)] = score``.
    """
    lookup: ScoreLookup = {}
    for score, (cam_y, cam_x), (pix_y, pix_x) in zip(scores, cams, coords, strict=True):
        tile_y = int(pix_y) // tile_size
        tile_x = int(pix_x) // tile_size
        lookup.setdefault((int(cam_y), int(cam_x)), {})[(tile_y, tile_x)] = float(score)
    return lookup


def build_slide_arrays(
    score_lookup: ScoreLookup,
    slide_nc_dir: Path,
    *,
    tile_size: int = 256,
    pixel_border: int = 0,
    fov_px: int = 160,
    images_np: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-FOV downsampled mosaic and matching tile-score grid.

    Every FOV in the slide is debayered, downsampled to ``fov_px``, and tiled
    in score space at the tile granularity used during scoring. Missing tiles
    (e.g. pen-marked in Mode 1, or simply absent from the bag files in Mode 2)
    are assigned score 0.

    Args:
        score_lookup: Result of :func:`build_score_lookup`.
        slide_nc_dir: Slide directory containing ``fullslidescan_AIF_WB.nc``.
            Only consulted when ``images_np`` is ``None``.
        tile_size: Tile edge length the scores were computed on.
        pixel_border: Pixel border that was dropped inside each FOV at scoring
            time (so tile-grid dimensions match the scorer's view).
        fov_px: Per-FOV side length of the output mosaic; should be divisible
            by the per-FOV tile count for a clean upsample (``160`` matches the
            8x8 grid used in the project).
        images_np: Pre-eager-loaded ``(Y, X, H, W)`` uint8 Bayer array. When
            provided the slide file is not re-opened. Pass the same array used
            for tiling to keep memory flat.
        device: CUDA device for the GPU debayer+downsample path. When ``None``
            or a CPU device, falls back to a per-FOV ``cv2`` loop.

    Returns:
        ``(score_array, image_array)``:
            * ``score_array`` — ``(y_max, x_max, fov_px, fov_px)`` float32.
            * ``image_array`` — ``(y_max, x_max, fov_px, fov_px, 3)`` uint8.
    """
    if images_np is None:
        nc_path = slide_nc_dir / _SLIDE_FILENAME
        log.info("Eager-loading %s for mosaic", nc_path)
        dataset = owl_load(nc_path)
        images_np = np.asarray(dataset["images"].values)

    y_max, x_max = images_np.shape[:2]
    fov_h, fov_w = images_np.shape[2:4]
    log.info(
        "build_slide_arrays: shape=%s fov_h=%d fov_w=%d tile_size=%d pixel_border=%d fov_px=%d",
        images_np.shape,
        fov_h,
        fov_w,
        tile_size,
        pixel_border,
        fov_px,
    )

    tiles_per_row = (fov_w - 2 * pixel_border) // tile_size
    tiles_per_col = (fov_h - 2 * pixel_border) // tile_size
    if tiles_per_row == 0 or tiles_per_col == 0:
        raise ValueError(
            f"FOV {fov_h}x{fov_w} with pixel_border={pixel_border} "
            f"yields no full tiles of size {tile_size}"
        )
    scale = fov_px // tiles_per_row

    log.info(
        "Building mosaic: %dx%d FOVs, %dx%d tiles/FOV, scale=%dx, fov_px=%d (%s)",
        y_max,
        x_max,
        tiles_per_col,
        tiles_per_row,
        scale,
        fov_px,
        "GPU" if (device is not None and device.type == "cuda") else "CPU",
    )

    if device is not None and device.type == "cuda":
        image_array = _gpu_mosaic_chunks(images_np, fov_px, device)
    else:
        image_array = _cpu_mosaic(images_np, fov_px)

    score_array = np.zeros((y_max, x_max, fov_px, fov_px), dtype=np.float32)
    for (cam_y, cam_x), fov_scores in score_lookup.items():
        grid = np.zeros((tiles_per_col, tiles_per_row), dtype=np.float32)
        for (ty, tx), s in fov_scores.items():
            if 0 <= ty < tiles_per_col and 0 <= tx < tiles_per_row:
                grid[ty, tx] = s
        score_array[cam_y, cam_x] = np.repeat(
            np.repeat(grid, scale, axis=0), scale, axis=1
        )

    return score_array, image_array


def overview_with_heatmap(
    image_array: np.ndarray,
    score_array: np.ndarray,
    scores: np.ndarray,
    slide_name: str,
    save_path: Path,
    *,
    fov_px: int = 160,
    export_individual: bool = False,
    individual_dpi: int = 600,
) -> None:
    """Render the stacked full-slide-image / score-heatmap figure.

    Args:
        image_array: ``(y_max, x_max, fov_px, fov_px, 3)`` uint8 mosaic from
            :func:`build_slide_arrays`.
        score_array: ``(y_max, x_max, fov_px, fov_px)`` float32 score map.
        scores: 1-D array of all per-tile scores (used for the figure caption).
        slide_name: Title-bar slide identifier.
        save_path: Output path for the combined figure.
        fov_px: Per-FOV mosaic side length (must match ``image_array``).
        export_individual: Also save unannotated image, heatmap, and
            heatmap-with-colorbar panels alongside ``save_path``.
        individual_dpi: DPI for the individual panel exports.
    """
    y_max, x_max = image_array.shape[:2]
    slide_image = image_array.transpose(0, 2, 1, 3, 4).reshape(
        y_max * fov_px, x_max * fov_px, 3
    )
    slide_scores = score_array.transpose(0, 2, 1, 3).reshape(
        y_max * fov_px, x_max * fov_px
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
    ax1.imshow(slide_image)
    ax1.set_title(
        f"Full Slide Overview - {slide_name}\n(FOV size: {fov_px}x{fov_px})",
        fontsize=14,
    )
    ax1.axis("off")

    im = ax2.imshow(slide_scores, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    score_summary = (
        f"Total tiles: {len(scores)}, "
        f"Score range: {scores.min():.3f}-{scores.max():.3f}"
        if len(scores)
        else "No tiles scored"
    )
    ax2.set_title(f"Tile-Level Score Heatmap\n{score_summary}", fontsize=14)
    ax2.axis("off")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="3%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Tile Score", labelpad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved overview+heatmap -> %s", save_path)

    if export_individual:
        stem = save_path.with_suffix("")
        plt.imsave(f"{stem}_slide_image.png", slide_image)
        plt.imsave(f"{stem}_heatmap.png", slide_scores, cmap="hot", vmin=0, vmax=1)

        h_px, w_px = slide_scores.shape
        fig_w = 10
        fig_h = fig_w * h_px / w_px + 0.6
        fig2, ax = plt.subplots(figsize=(fig_w, fig_h))
        im2 = ax.imshow(
            slide_scores, cmap="hot", interpolation="nearest", vmin=0, vmax=1
        )
        ax.axis("off")
        cbar2 = fig2.colorbar(
            im2, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02
        )
        cbar2.set_label("Tile Score")
        plt.savefig(
            f"{stem}_heatmap_colorbar.png",
            dpi=individual_dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig2)


def top_tiles_grid(
    images: np.ndarray,
    cams: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    slide_name: str,
    save_path: Path,
    *,
    top_k: int = 20,
    poster_grid: bool = False,
    poster_dpi: int = 600,
) -> None:
    """Render a 4x5 grid of the ``top_k`` highest-scoring tiles.

    Args:
        images: ``(N, ts, ts, 3)`` uint8 array of tiles, aligned with ``scores``.
        cams: ``(N, 2)`` int array — ``(fov_row, fov_col)`` per tile.
        coords: ``(N, 2)`` int array — ``(pixel_row, pixel_col)`` per tile.
        scores: ``(N,)`` float array of tile scores.
        slide_name: Title-bar slide identifier.
        save_path: Output path for the combined figure.
        top_k: Number of top tiles to display.
        poster_grid: Also write a 2x4 transparent-background variant for posters.
        poster_dpi: DPI for the poster export.
    """
    if len(scores) == 0:
        log.warning("top_tiles_grid: no scores to plot")
        return

    top_indices = np.argsort(scores)[-top_k:][::-1]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f"Top {top_k} Scoring Tiles - {slide_name}", fontsize=16)

    for idx, ax_idx in enumerate(top_indices):
        row, col = idx // 5, idx % 5
        axes[row, col].imshow(images[ax_idx])
        axes[row, col].set_title(
            f"Score: {scores[ax_idx]:.3f}\n"
            f"FOV: ({int(cams[ax_idx, 0])}, {int(cams[ax_idx, 1])})\n"
            f"Coord: ({int(coords[ax_idx, 0])}, {int(coords[ax_idx, 1])})",
            fontsize=8,
        )
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved top-%d tiles -> %s", top_k, save_path)

    if poster_grid:
        n_cols, n_rows = 4, 2
        poster_indices = top_indices[: n_cols * n_rows]
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(9, 5), facecolor="none")
        fig2.patch.set_alpha(0.0)
        fig2.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
        for flat_idx, ax_idx in enumerate(poster_indices):
            r, c = flat_idx // n_cols, flat_idx % n_cols
            axes2[r, c].imshow(images[ax_idx])
            axes2[r, c].set_axis_off()
            axes2[r, c].set_aspect("equal")
        poster_path = save_path.with_name(save_path.stem + "_poster_grid.png")
        fig2.savefig(
            poster_path,
            dpi=poster_dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig2)
        log.info("Saved poster grid -> %s", poster_path)


def top6_row(
    images: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    *,
    dpi: int = 300,
) -> None:
    """Render the 6 highest-scoring tiles as a single transparent row.

    Args:
        images: ``(N, ts, ts, 3)`` uint8 array of tiles.
        scores: ``(N,)`` float array of tile scores aligned with ``images``.
        save_path: Output path for the saved PNG.
        dpi: Resolution of the saved figure.
    """
    if len(scores) < 6:
        log.warning("top6_row: need at least 6 tiles, got %d", len(scores))
        return

    top_indices = np.argsort(scores)[-6:][::-1]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3), facecolor="none")
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0, wspace=0.02)
    for ax, idx in zip(axes, top_indices, strict=True):
        ax.imshow(images[idx])
        ax.set_title(f"Score: {scores[idx]:.2f}", fontsize=15, color="white")
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    log.info("Saved top-6 row -> %s", save_path)


def fov_heatmap(
    fov_image: np.ndarray,
    score_map: np.ndarray,
    annotation: np.ndarray,
    save_path: Path,
    *,
    tile_size: int,
    stride: int,
    export_individual: bool = False,
    individual_dpi: int = 600,
) -> None:
    """Render a single-FOV 3-panel figure (raw / score / overlay).

    Args:
        fov_image: ``(H, W, 3)`` uint8 FOV image.
        score_map: ``(H, W)`` float array — per-pixel averaged tile score.
        annotation: ``(H, W)`` uint8 pen-mark mask (zero if not provided).
        save_path: Output path for the combined figure.
        tile_size: Tile edge length used during scoring (for the caption).
        stride: Sliding-window stride used during scoring (for the caption).
        export_individual: Also save the three panels as standalone PNGs.
        individual_dpi: DPI for the individual panel exports.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(fov_image)
    axes[0].set_title("Original FOV")
    axes[0].axis("off")

    im1 = axes[1].imshow(score_map, cmap="hot", interpolation="bilinear")
    axes[1].set_title(f"Score Heatmap\n(stride={stride}, tile_size={tile_size})")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    axes[2].imshow(fov_image)
    overlay = axes[2].imshow(score_map, cmap="hot", alpha=0.5, interpolation="bilinear")
    axes[2].set_title("FOV with Score Overlay")
    axes[2].axis("off")
    plt.colorbar(overlay, ax=axes[2], shrink=0.8)

    pen_mask = annotation > 0 if annotation.sum() > 0 else None
    if pen_mask is not None:
        for ax in axes:
            ax.contour(pen_mask, colors="cyan", linewidths=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved FOV heatmap -> %s", save_path)

    if export_individual:
        stem = save_path.with_suffix("")
        plt.imsave(f"{stem}_fov_image.png", fov_image)
        plt.imsave(f"{stem}_score_heatmap.png", score_map, cmap="hot")

        cmap = plt.get_cmap("hot")
        denom = score_map.max() - score_map.min() + 1e-8
        score_norm = (score_map - score_map.min()) / denom
        heat_rgba = (np.asarray(cmap(score_norm)) * 255).astype(np.uint8)
        fov_f = fov_image.astype(np.float32)
        heat_f = heat_rgba[..., :3].astype(np.float32)
        composite = np.clip(0.5 * fov_f + 0.5 * heat_f, 0, 255).astype(np.uint8)
        if pen_mask is not None:
            contour_img = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            contours, _ = cv2.findContours(
                pen_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)
            composite = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
        plt.imsave(f"{stem}_overlay.png", composite)
        # individual_dpi is consumed when saving via plt.savefig; imsave writes
        # raw pixel grids so the parameter only governs the figure-based export
        # above (none here). Kept in the signature for API parity with the
        # full-slide overview helper.
        _ = individual_dpi


def score_histogram(
    scores: np.ndarray,
    save_path: Path,
    *,
    total: int | None = None,
    title: str,
    bins: int = 50,
) -> None:
    """Render a histogram of tile scores in ``[0, 1]``.

    Args:
        scores: 1-D array of tile scores.
        save_path: Output path for the saved PNG.
        total: If provided, y-axis shows ``count / total`` instead of raw counts.
            Mode 2 passes ``len(scores)`` so the histogram normalizes across
            slides with different tile counts; Mode 1 passes ``None``.
        title: Figure title.
        bins: Number of histogram bins between 0 and 1.
    """
    if len(scores) == 0:
        log.warning("score_histogram: no scores to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    weights = None
    ylabel = "Tile count"
    if total is not None and total > 0:
        weights = np.full_like(scores, 1.0 / float(total), dtype=np.float64)
        ylabel = "Fraction of total tiles"

    ax.hist(scores, bins=bins, range=(0.0, 1.0), weights=weights, color="steelblue")
    ax.set_xlabel("Tile score")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved score histogram -> %s", save_path)
