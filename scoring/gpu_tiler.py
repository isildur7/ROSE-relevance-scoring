"""GPU-streamed tile producer for Mode 1 (full-slide scoring from a ``.nc`` file).

Ported subset of ROSE-processing-v3's ``data_tiler`` package:

- ``debayer``      — kornia ``raw_to_rgb`` on a batched Bayer FOV chunk.
- ``to_tiles``     — view-reshape ``(B, H, W, 3)`` into ``(B, ny, nx, ts, ts, 3)``.
- ``ann_block_ok`` — block-sum a per-FOV mask and return True where no pen pixel
  overlaps the tile.

The ``iter_tile_batches`` generator drives these kernels chunk-by-chunk and
yields kept tiles plus their FOV/coord metadata. Unlike v3 it does not
HSV-filter (Mode 1 wants every non-pen tile scored), does not write HDF5, and
does not produce the FOV-mask overview PNG.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from owl.mcam_data import load as owl_load

from scoring._annotation import load_annotation_batch

log = logging.getLogger(__name__)

_SLIDE_FILENAME = "fullslidescan_AIF_WB.nc"

try:
    import kornia.color as _kc
except ImportError as e:  # pragma: no cover — surfaced at import time
    raise ImportError(
        "kornia is required for scoring.gpu_tiler; install with `pip install kornia`."
    ) from e


class TileBatch(TypedDict):
    """One streamed batch of pen-mark-filtered tiles.

    ``tiles_gpu`` is the hot-path payload: a uint8 ``(B, ts, ts, 3)`` GPU
    tensor that the score loop preprocesses + classifies without ever copying
    back to CPU. ``images`` is a CPU mirror of the same tiles, used only by
    visualization (top-tiles grid). Drop the reference to ``tiles_gpu`` once
    you've scored the batch so its GPU memory is released for the next chunk.
    """

    tiles_gpu: torch.Tensor  # (B, ts, ts, 3) uint8 on the streaming device
    images: np.ndarray  # (B, ts, ts, 3) uint8 CPU mirror for viz
    cams: np.ndarray  # (B, 2) int32 — (fov_row, fov_col)
    coords: np.ndarray  # (B, 2) int32 — (pixel_row, pixel_col) within FOV


def _largest_divisible_crop(
    h: int, w: int, tile: int, border: int
) -> tuple[slice, slice]:
    """Crop ``border`` pixels off each side then trim to multiples of ``tile``.

    Args:
        h: FOV height in pixels.
        w: FOV width in pixels.
        tile: Tile edge length in pixels.
        border: Pixel border to drop on every side before tiling.

    Returns:
        ``(slice_y, slice_x)`` selecting the largest tile-aligned region of the
        FOV. The slices may be empty if ``border`` is too large.
    """
    y0, x0 = border, border
    y1, x1 = h - border, w - border
    if y1 <= y0 or x1 <= x0:
        return slice(0, 0), slice(0, 0)
    hh = ((y1 - y0) // tile) * tile
    ww = ((x1 - x0) // tile) * tile
    return slice(y0, y0 + hh), slice(x0, x0 + ww)


def _debayer(bayer: torch.Tensor) -> torch.Tensor:
    """Demosaic a batch of GR-pattern Bayer FOVs to uint8 RGB.

    Args:
        bayer: ``(B, H, W)`` uint8 tensor on the target device.

    Returns:
        ``(B, H, W, 3)`` uint8 tensor on the same device.
    """
    x = bayer.unsqueeze(1).float() / 255.0
    rgb = _kc.raw_to_rgb(x, _kc.CFA.GR)
    rgb = rgb.clamp_(0.0, 1.0).mul_(255.0)
    return rgb.to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def _to_tiles(rgb: torch.Tensor, tile_size: int) -> torch.Tensor:
    """View-reshape a batched RGB FOV into a tile grid.

    Args:
        rgb: ``(B, H, W, 3)`` tensor with H, W divisible by ``tile_size``.
        tile_size: Tile edge length in pixels.

    Returns:
        ``(B, ny, nx, tile_size, tile_size, 3)`` contiguous tensor.
    """
    b, h, w, c = rgb.shape
    ny, nx = h // tile_size, w // tile_size
    return (
        rgb.view(b, ny, tile_size, nx, tile_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
    )


def _ann_block_ok(ann: torch.Tensor, tile_size: int) -> torch.Tensor:
    """True where a tile-sized block of the annotation mask has zero pixels set.

    Args:
        ann: ``(B, H, W)`` uint8 tensor; nonzero marks pen-annotated pixels.
        tile_size: Tile edge length in pixels.

    Returns:
        ``(B, ny, nx)`` bool tensor; True means the tile can be kept.
    """
    b, h, w = ann.shape
    ny, nx = h // tile_size, w // tile_size
    blocks = ann.view(b, ny, tile_size, nx, tile_size).permute(0, 1, 3, 2, 4)
    return blocks.to(torch.int32).sum(dim=(3, 4)) == 0


def _interior_fovs(
    y_max: int, x_max: int, y_border: int, x_border: int
) -> list[tuple[int, int]]:
    """Build the FOV iteration order in (row, col), skipping the border ring."""
    return [
        (y, x)
        for y in range(y_border, y_max - y_border)
        for x in range(x_border, x_max - x_border)
    ]


def load_slide_images(slide_nc_dir: Path) -> np.ndarray:
    """Eager-load a slide's ``images`` xarray into a ``(Y, X, H, W)`` uint8 array.

    Args:
        slide_nc_dir: Slide directory containing ``fullslidescan_AIF_WB.nc``.

    Returns:
        Numpy view of the full slide. ~6 GB for a 27x54 grid of 3000x4000 FOVs.
    """
    nc_path = Path(slide_nc_dir) / _SLIDE_FILENAME
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing slide file: {nc_path}")
    log.info("Eager-loading %s into RAM", nc_path)
    dataset = owl_load(nc_path)
    return np.asarray(dataset["images"].values)


def iter_tile_batches(
    slide_nc_dir: Path,
    *,
    tile_size: int = 256,
    x_border: int = 2,
    y_border: int = 2,
    pixel_border: int = 0,
    mask_folder: Path | None = None,
    chunk: int = 27,
    device: torch.device | None = None,
    eager_load: bool = True,
    images_np: np.ndarray | None = None,
) -> Iterator[TileBatch]:
    """Stream pen-mark-filtered tiles from a ``.nc`` slide.

    For each chunk of FOVs the generator loads Bayer (plus annotations if
    provided), debayers on GPU, builds the tile grid, applies the pen-mark
    block-sum filter, and yields the kept tiles as numpy arrays so the
    downstream scorer can pipeline GPU inference against the next chunk.

    Args:
        slide_nc_dir: Slide directory containing ``fullslidescan_AIF_WB.nc``.
        tile_size: Tile edge length in pixels.
        x_border: Number of border FOVs to skip on each side along the x axis.
        y_border: Number of border FOVs to skip on each side along the y axis.
        pixel_border: Pixel border dropped inside each FOV before tiling.
        mask_folder: Directory containing per-FOV pen-mark masks named
            ``<slide>_<row>_<col>.{tif,npy}``. ``None`` disables filtering.
        chunk: Number of FOVs loaded into GPU memory per step.
        device: Target CUDA device. ``None`` auto-picks CUDA (required — no
            CPU fallback for the GPU kernels used here).
        eager_load: If True, load the full slide images array into host RAM
            once and fancy-index it per chunk (fast). If False, ``isel`` each
            FOV from the underlying xarray (slower, lower peak RAM). Ignored
            when ``images_np`` is supplied.
        images_np: Optional pre-eager-loaded ``(Y, X, H, W)`` uint8 array.
            Skips re-opening the slide; pass this when the same array will be
            reused downstream (e.g. for building the visualization mosaic).

    Yields:
        :class:`TileBatch` dicts with ``images`` ``(B, tile_size, tile_size, 3)``
        uint8, ``cams`` ``(B, 2)`` int32 ``(row, col)``, and ``coords``
        ``(B, 2)`` int32 ``(pixel_row, pixel_col)`` within the source FOV.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("scoring.gpu_tiler requires CUDA; no GPU detected.")
        device = torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError(f"scoring.gpu_tiler requires a CUDA device, got {device}.")

    slide_nc_dir = Path(slide_nc_dir)
    slide_name = slide_nc_dir.name
    nc_path = slide_nc_dir / _SLIDE_FILENAME
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing slide file: {nc_path}")

    if images_np is not None:
        log.info("Reusing supplied slide array (shape=%s)", images_np.shape)
        images_xr = None
        y_max, x_max = images_np.shape[:2]
    else:
        log.info("Opening %s", nc_path)
        dataset = owl_load(nc_path)
        images_xr = dataset["images"]
        y_max, x_max = images_xr.shape[:2]
        if eager_load:
            log.info("Eager-loading full slide images array into RAM")
            images_np = np.asarray(images_xr.values)

    # Upload the entire bayer array to the GPU once. The slide is ~6 GB uint8;
    # this fits easily on a 24 GB+ GPU and eliminates the per-chunk
    # pin_memory + host->device transfer that dominated the previous run
    # (~4 s per chunk × 43 chunks = ~170 s of CPU stalls).
    bayer_full_gpu: torch.Tensor | None = None
    if images_np is not None:
        log.info(
            "Uploading slide bayer (%.2f GB) to %s",
            images_np.nbytes / 2**30,
            device,
        )
        bayer_full_gpu = torch.from_numpy(images_np).to(device, non_blocking=False)
        if bayer_full_gpu.ndim == 5 and bayer_full_gpu.shape[-1] == 1:
            bayer_full_gpu = bayer_full_gpu[..., 0]

    fovs = _interior_fovs(y_max, x_max, y_border, x_border)
    log.info(
        "Streaming %d interior FOVs (grid=%dx%d, border=(y=%d,x=%d)) in chunks of %d",
        len(fovs),
        y_max,
        x_max,
        y_border,
        x_border,
        chunk,
    )

    # Preload every interior pen-mark mask once. The masks live on slow NFS
    # (~100 ms per file) so doing it per chunk was costing tens of seconds.
    # Total mask data for the full slide is small (~25 MB uint8).
    full_ann_gpu: torch.Tensor | None = None
    fov_to_ann_idx: dict[tuple[int, int], int] = {}
    if mask_folder is not None:
        # Need the FOV pixel shape to allocate the mask buffer.
        if bayer_full_gpu is not None:
            h0_global, w0_global = (
                int(bayer_full_gpu.shape[2]),
                int(bayer_full_gpu.shape[3]),
            )
        else:
            assert images_xr is not None
            sample = np.asarray(images_xr.isel(image_y=0, image_x=0).values)
            if sample.ndim == 3 and sample.shape[-1] == 1:
                sample = sample[..., 0]
            h0_global, w0_global = sample.shape
        log.info(
            "Preloading %d pen-mark masks (%dx%d) from %s",
            len(fovs),
            h0_global,
            w0_global,
            mask_folder,
        )
        ann_np = load_annotation_batch(
            mask_folder, slide_name, fovs, (h0_global, w0_global), n_workers=16
        )
        for idx, fov in enumerate(fovs):
            fov_to_ann_idx[fov] = idx
        full_ann_gpu = torch.from_numpy(ann_np).to(device, non_blocking=False)

    rows_idx_tensor: torch.Tensor | None = None
    cols_idx_tensor: torch.Tensor | None = None
    ann_idx_tensor: torch.Tensor | None = None
    if bayer_full_gpu is not None:
        rows_idx_tensor = torch.empty(chunk, dtype=torch.long, device=device)
        cols_idx_tensor = torch.empty(chunk, dtype=torch.long, device=device)
    if full_ann_gpu is not None:
        ann_idx_tensor = torch.empty(chunk, dtype=torch.long, device=device)

    for start in range(0, len(fovs), chunk):
        chunk_fovs = fovs[start : start + chunk]
        b_chunk = len(chunk_fovs)

        if bayer_full_gpu is not None:
            rows_np = np.fromiter((r for r, _ in chunk_fovs), dtype=np.int64)
            cols_np = np.fromiter((c for _, c in chunk_fovs), dtype=np.int64)
            assert rows_idx_tensor is not None and cols_idx_tensor is not None
            rows_idx_tensor[:b_chunk].copy_(
                torch.from_numpy(rows_np), non_blocking=True
            )
            cols_idx_tensor[:b_chunk].copy_(
                torch.from_numpy(cols_np), non_blocking=True
            )
            bayer_t = bayer_full_gpu[
                rows_idx_tensor[:b_chunk], cols_idx_tensor[:b_chunk]
            ]
            h0, w0 = int(bayer_t.shape[1]), int(bayer_t.shape[2])
        else:
            bayer_np_chunk = _load_chunk_bayer(chunk_fovs, images_np, images_xr)
            h0, w0 = bayer_np_chunk.shape[1], bayer_np_chunk.shape[2]
            bayer_t = (
                torch.from_numpy(bayer_np_chunk)
                .pin_memory()
                .to(device, non_blocking=True)
            )

        ys, xs = _largest_divisible_crop(h0, w0, tile_size, pixel_border)
        if ys.stop - ys.start <= 0 or xs.stop - xs.start <= 0:
            continue
        bayer_t = bayer_t[:, ys, xs]

        if full_ann_gpu is not None:
            assert ann_idx_tensor is not None
            ann_idx_np = np.fromiter(
                (fov_to_ann_idx[fov] for fov in chunk_fovs), dtype=np.int64
            )
            ann_idx_tensor[:b_chunk].copy_(
                torch.from_numpy(ann_idx_np), non_blocking=True
            )
            ann_t = full_ann_gpu[ann_idx_tensor[:b_chunk]][:, ys, xs]
            ok = _ann_block_ok(ann_t, tile_size)
        else:
            b, h, w = bayer_t.shape
            ny, nx = h // tile_size, w // tile_size
            ok = torch.ones((b, ny, nx), dtype=torch.bool, device=device)

        if not bool(ok.any()):
            continue

        rgb = _debayer(bayer_t)
        tiles = _to_tiles(rgb, tile_size)

        kept_idx = ok.nonzero(as_tuple=False)  # (K, 3) — (b, iy, ix)
        kept_tiles = tiles[
            kept_idx[:, 0], kept_idx[:, 1], kept_idx[:, 2]
        ]  # (K, ts, ts, 3)

        b_idx = kept_idx[:, 0].cpu().numpy()
        iy_idx = kept_idx[:, 1].cpu().numpy().astype(np.int32)
        ix_idx = kept_idx[:, 2].cpu().numpy().astype(np.int32)

        cams_np = np.array(
            [(chunk_fovs[b][0], chunk_fovs[b][1]) for b in b_idx], dtype=np.int32
        )
        coords_np = np.stack(
            [ys.start + iy_idx * tile_size, xs.start + ix_idx * tile_size],
            axis=1,
        ).astype(np.int32)
        # CPU mirror for visualizations; the score loop consumes ``kept_tiles``
        # directly on-device.
        images_out = kept_tiles.cpu().numpy()

        yield TileBatch(
            tiles_gpu=kept_tiles,
            images=images_out,
            cams=cams_np,
            coords=coords_np,
        )


def _load_chunk_bayer(
    chunk_fovs: list[tuple[int, int]],
    images_np: np.ndarray | None,
    images_xr,  # noqa: ANN001 — xarray.DataArray, kept untyped to avoid hard dep
) -> np.ndarray:
    """Pull a chunk of Bayer FOVs into a contiguous host array."""
    if images_np is not None:
        rows = np.fromiter((r for r, _ in chunk_fovs), dtype=np.int32)
        cols = np.fromiter((c for _, c in chunk_fovs), dtype=np.int32)
        bayer = images_np[rows, cols]
    else:
        arrs = []
        for r, c in chunk_fovs:
            a = np.asarray(images_xr.isel(image_y=r, image_x=c).values)
            arrs.append(a)
        bayer = np.stack(arrs, axis=0)

    if bayer.ndim == 4 and bayer.shape[-1] == 1:
        bayer = bayer[..., 0]
    return np.ascontiguousarray(bayer)
