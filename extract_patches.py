"""Extract annotated 256×256 patches from MCAM .nc slide scans and save as RGB JPEGs.

Reads the annotation parquet, filters to labels 0 and 1, and for each annotated
patch extracts the correct region from the raw Bayer FOV, debayers it, undoes the
90° CW rotation applied during JPEG tile generation, and saves as a JPEG.

Also writes a ``patches_annotations.parquet`` with columns ``filepath`` (relative
to ``output_dir``) and ``label``.

Slides are processed in parallel via ``ProcessPoolExecutor`` (one worker per slide).
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
from jsonargparse import CLI
from owl import mcam_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PATCH_SIZE: int = 256
BAYER_FLAG: int = cv2.COLOR_BayerGR2RGB


def find_nc(slide: str, slide_dirs: list[Path]) -> Path:
    """Locate the fullslidescan_AIF_WB.nc file for a given slide name.

    Args:
        slide: Slide directory name (e.g. ``'CF14-003066A-1'``).
        slide_dirs: Ordered list of base directories to search.

    Returns:
        Path to the .nc file.

    Raises:
        FileNotFoundError: If no .nc file is found in any of the provided directories.
    """
    for base in slide_dirs:
        p = base / slide / "fullslidescan_AIF_WB.nc"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No fullslidescan_AIF_WB.nc found for slide {slide!r} in {slide_dirs}"
    )


def process_slide(
    slide: str,
    slide_df: pd.DataFrame,
    slide_dirs: list[Path],
    output_dir: Path,
    overwrite: bool,
) -> list[dict]:
    """Extract all annotated patches for a single slide.

    FOV tiles were pre-generated with PIL.rotate(270) = 90° CW applied to raw data.
    Annotation pixel coords reference the rotated JPEG. To recover the correct region
    from the raw array: ``raw[top_left_px_x : +256, top_left_px_y : +256]``, then
    ``np.rot90`` (90° CCW) to restore the annotator's view orientation.

    Unless ``overwrite`` is set, patches whose JPEG already exists in ``output_dir``
    are skipped: a FOV is only loaded and debayered if it has at least one missing
    patch, and a slide whose patches all exist never opens its .nc file. Skipped
    patches are still recorded so the returned table stays complete.

    Args:
        slide: Slide name used for filenames and locating the .nc file.
        slide_df: Annotation rows for this slide.
        slide_dirs: Ordered list of base directories to search for .nc files.
        output_dir: Directory where JPEG patches will be written.
        overwrite: If True, (re)write every patch even if its JPEG already exists.
            If False, skip patches already present on disk.

    Returns:
        List of ``{"filepath": str, "label": int}`` dicts for every annotated patch
        (both newly written and pre-existing).
    """
    nc_path = find_nc(slide, slide_dirs)
    images = None  # loaded lazily; a fully-extracted slide never opens its .nc

    records: list[dict] = []
    for (fov_r, fov_c), fov_df in slide_df.groupby(["fov_r", "fov_c"]):
        targets = [
            (
                row,
                f"{slide}__fovr{fov_r}-fovc{fov_c}"
                f"__pr{int(row['patch_r'])}-pc{int(row['patch_c'])}.jpg",
            )
            for _, row in fov_df.iterrows()
        ]
        # Record every annotated patch so the output table stays complete.
        records.extend(
            {"filepath": fname, "label": int(row["label"])} for row, fname in targets
        )

        to_write = (
            targets
            if overwrite
            else [(r, f) for r, f in targets if not (output_dir / f).exists()]
        )
        if not to_write:
            log.info(
                "  [%s] FOV (%s, %s): all %d patches present, skipping.",
                slide,
                fov_r,
                fov_c,
                len(targets),
            )
            continue

        if images is None:
            log.info("Loading %s for slide %s", nc_path.name, slide)
            dataset = mcam_data.load(nc_path, delayed=True)
            images = dataset["images"]  # (image_y, image_x, y, x)

        bayer = images.isel(image_y=fov_r, image_x=fov_c).values
        rgb = cv2.cvtColor(bayer, BAYER_FLAG)

        for row, fname in to_write:
            raw_row = int(row["top_left_px_x"])
            raw_col = int(row["top_left_px_y"])
            patch = rgb[raw_row : raw_row + PATCH_SIZE, raw_col : raw_col + PATCH_SIZE]
            patch = np.rot90(patch)
            iio.imwrite(output_dir / fname, patch)

        log.info(
            "  [%s] FOV (%s, %s): %d patches written, %d skipped.",
            slide,
            fov_r,
            fov_c,
            len(to_write),
            len(targets) - len(to_write),
        )

    return records


def extract_patches(
    df: pd.DataFrame,
    slide_dirs: list[Path],
    output_dir: Path,
    num_workers: int,
    overwrite: bool,
) -> pd.DataFrame:
    """Extract patches for all slides, processing slides in parallel.

    Args:
        df: Annotation rows with columns ``slide``, ``fov_r``, ``fov_c``,
            ``patch_r``, ``patch_c``, ``top_left_px_x``, ``top_left_px_y``, ``label``.
        slide_dirs: Ordered list of base directories to search for .nc files.
        output_dir: Directory where JPEG patches will be written.
        num_workers: Number of parallel worker processes.
        overwrite: If True, (re)write every patch even if its JPEG already exists.

    Returns:
        DataFrame with columns ``filepath`` (relative to ``output_dir``) and ``label``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slides = list(df.groupby("slide"))

    all_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_slide, str(slide), slide_df, slide_dirs, output_dir, overwrite
            ): slide
            for slide, slide_df in slides
        }
        for future in as_completed(futures):
            slide = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                log.info("Slide %s complete: %d patches.", slide, len(records))
            except Exception:
                log.exception("Slide %s failed.", slide)

    return pd.DataFrame(all_records)


def main(
    annotations: Path,
    output_dir: Path,
    slide_dirs: list[Path],
    num_workers: int = 8,
    test: bool = False,
    overwrite: bool = False,
) -> None:
    """Run patch extraction.

    Args:
        annotations: Path to the annotation parquet file.
        output_dir: Directory to write JPEG patches and the output annotation parquet.
        slide_dirs: Ordered list of base directories to search for slide .nc files.
        num_workers: Number of parallel worker processes. 0 = auto (min of slide count
            and CPU count). Defaults to 8.
        test: If True, process only one slide/FOV (CF14-003066A-1, FOV 12/8) for
            visual verification.
        overwrite: If True, (re)write every patch even if its JPEG already exists.
            Defaults to False, which skips patches already present in ``output_dir``
            and only extracts newly annotated ones.
    """
    df = pd.read_parquet(annotations)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    if test:
        df = df[
            (df["slide"] == "CF14-003066A-1") & (df["fov_r"] == 12) & (df["fov_c"] == 8)
        ]
        log.info(
            "TEST MODE: %d patches from slide CF14-003066A-1, FOV (12, 8)", len(df)
        )

    n_slides = df["slide"].nunique()
    resolved_workers = (
        num_workers if num_workers > 0 else min(n_slides, os.cpu_count() or 8)
    )
    log.info("Processing %d slides with %d workers.", n_slides, resolved_workers)

    result_df = extract_patches(df, slide_dirs, output_dir, resolved_workers, overwrite)

    ann_path = output_dir / "patches_annotations.parquet"
    result_df.to_parquet(ann_path, index=False)
    log.info("Done. %d patches saved. Annotation table: %s", len(result_df), ann_path)


if __name__ == "__main__":
    CLI(main)
