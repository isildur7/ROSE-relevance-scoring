"""Join score_nc blurred scores into existing bag/feature h5 files.

``score_nc.py`` can compute a spatially-blurred per-tile relevance score
(``smoothed_score``) because it lays every tile out on the full-slide grid.
The h5 bag pipeline cannot, since each slide is split across multiple bags.
This script bridges the gap without re-running anything spatial: it reads the
``score_nc`` CSV for each slide and writes a ``blurred_rs`` dataset into the
targeted bag h5 files and their matching ``features_<stem>*.h5`` files, joined
by the coordinate key ``(FOV_y, FOV_x, pixel_y, pixel_x)``.

No existing dataset is overwritten; unmatched rows are filled with ``0.0`` and
warned about. This is an experiment scaffold — a production blurred-score
pipeline is built only if MIL training improves with these scores.

Example:
    python join_blurred_scores.py \\
        --bag_root /media/data1/kanghyun/ROSE_MIL/DUMC \\
        --csv_root claude/score_nc_blurred \\
        --bag_pattern 'gammasat3bags_*.h5' \\
        --slide_info_json /home/amey/ROSE-processing-v2/data_jsons/test.json
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import h5py
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

from process_h5_scores import _patient_ids_from_jsonl, _replace_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_blurred_array(
    coords: np.ndarray,
    cams: np.ndarray,
    lut: dict[tuple[int, int, int, int], float],
) -> tuple[np.ndarray, int]:
    """Build a per-row blurred-score array by coordinate lookup.

    Args:
        coords: ``(N, 2)`` int array of ``(pixel_y, pixel_x)`` per row.
        cams: ``(N, 2)`` int array of ``(fov_y, fov_x)`` per row (the bag's
            ``cams`` dataset or a feature file's ``cam_yx`` dataset).
        lut: Map ``(fov_y, fov_x, pixel_y, pixel_x) -> score`` built from the
            slide's CSV.

    Returns:
        ``(blurred, n_unmatched)`` where ``blurred`` is an ``(N,)`` float32
        array (``0.0`` for rows with no lut entry) and ``n_unmatched`` is the
        number of such rows.
    """
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.float32)
    unmatched = 0
    for i in range(n):
        key = (
            int(cams[i, 0]),
            int(cams[i, 1]),
            int(coords[i, 0]),
            int(coords[i, 1]),
        )
        score = lut.get(key)
        if score is None:
            unmatched += 1
        else:
            out[i] = score
    return out, unmatched


def _read_csv_lut(
    csv_path: Path,
    score_column: str,
) -> dict[tuple[int, int, int, int], float]:
    """Read a ``score_nc`` CSV into a coordinate-keyed score lookup.

    Args:
        csv_path: Path to ``<slide>_scores.csv`` with header
            ``FOV_y,FOV_x,pixel_y,pixel_x,score[,smoothed_score]``.
        score_column: Column whose value becomes the lookup value
            (e.g. ``"smoothed_score"`` or ``"score"``).

    Returns:
        Map ``(FOV_y, FOV_x, pixel_y, pixel_x) -> float`` for every row.

    Raises:
        ValueError: If ``score_column`` is not present in the CSV header.
    """
    lut: dict[tuple[int, int, int, int], float] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or score_column not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: column {score_column!r} not in header "
                f"{reader.fieldnames}"
            )
        for row in reader:
            key = (
                int(row["FOV_y"]),
                int(row["FOV_x"]),
                int(row["pixel_y"]),
                int(row["pixel_x"]),
            )
            lut[key] = float(row[score_column])
    return lut


def _write_blurred_into_h5(
    path: Path,
    lut: dict[tuple[int, int, int, int], float],
    dataset_name: str,
    cam_key: str,
) -> None:
    """Write ``dataset_name`` into an h5 file by coordinate-joining ``lut``.

    Opens ``path`` in ``r+`` and adds (or replaces) ``dataset_name``. If the
    file lacks ``coords`` or ``cam_key`` it is left untouched and a warning is
    logged. Unmatched rows are filled with ``0.0`` and warned about.

    Args:
        path: Bag or feature HDF5 path to update in place.
        lut: Coordinate-keyed score lookup from :func:`_read_csv_lut`.
        dataset_name: Name of the dataset to create (e.g. ``"blurred_rs"``).
        cam_key: Dataset holding the FOV index — ``"cams"`` for bag files,
            ``"cam_yx"`` for feature files.
    """
    with h5py.File(path, "r+") as f:
        if "coords" not in f or cam_key not in f:
            log.warning(
                "%s: missing coords/%s; skipping", path, cam_key
            )
            return
        coords = f["coords"][:]
        cams = f[cam_key][:]
        blurred, n_unmatched = build_blurred_array(coords, cams, lut)
        _replace_dataset(f, dataset_name, blurred)

    if n_unmatched:
        log.warning(
            "%s: %d/%d rows had no matching CSV entry (filled with 0.0)",
            path,
            n_unmatched,
            coords.shape[0],
        )


def _feature_paths_for_bag(bag_path: Path) -> list[Path]:
    """Find feature h5 files for ``bag_path`` (underscores kept).

    For this bag family feature files are named ``features_<bag.stem>*.h5``
    with the bag stem's underscores preserved. Unlike
    ``process_h5_scores``, the underscore-stripped form is intentionally not
    matched (it resolves to nothing for these slides).

    Args:
        bag_path: Path to the bag h5 whose feature files are sought.

    Returns:
        Sorted list of matching feature-file paths in the bag's directory.
    """
    return sorted(bag_path.parent.glob(f"features_{bag_path.stem}*.h5"))


def _process_one_slide(
    slide_name: str,
    bag_root: Path,
    csv_root: Path,
    bag_pattern: str,
    score_column: str,
    dataset_name: str,
) -> None:
    """Join one slide's CSV scores into its bag and feature h5 files.

    Args:
        slide_name: Slide subdirectory name (under ``bag_root`` and
            ``csv_root``).
        bag_root: Parent dir with per-slide subdirectories of bag h5 files.
        csv_root: Root of ``score_nc`` output; CSV expected at
            ``csv_root/<slide>/<slide>_scores.csv``.
        bag_pattern: Glob for the targeted bags under the slide dir.
        score_column: CSV column to write into the h5 dataset.
        dataset_name: Name of the h5 dataset to create.
    """
    csv_path = csv_root / slide_name / f"{slide_name}_scores.csv"
    if not csv_path.is_file():
        log.warning("CSV not found: %s; skipping slide %s", csv_path, slide_name)
        return

    lut = _read_csv_lut(csv_path, score_column)
    log.info("%s: loaded %d CSV rows", slide_name, len(lut))

    slide_dir = bag_root / slide_name
    bag_paths = sorted(slide_dir.glob(bag_pattern))
    if not bag_paths:
        log.warning("No bags matching %r under %s; skipping", bag_pattern, slide_dir)
        return

    for bag_path in bag_paths:
        try:
            _write_blurred_into_h5(bag_path, lut, dataset_name, cam_key="cams")
            log.info("Wrote %s into %s", dataset_name, bag_path.name)
            for feat_path in _feature_paths_for_bag(bag_path):
                _write_blurred_into_h5(feat_path, lut, dataset_name, cam_key="cam_yx")
                log.info("Wrote %s into %s", dataset_name, feat_path.name)
        except Exception:
            log.exception("Failed to process bag %s; continuing", bag_path)


def main(
    bag_root: Path,
    csv_root: Path,
    bag_pattern: str,
    slide_list: list[str] | None = None,
    slide_info_json: Path | None = None,
    score_column: str = "smoothed_score",
    dataset_name: str = "blurred_rs",
    verbose: bool = False,
) -> None:
    """Join score_nc CSV scores into bag/feature h5 files as ``blurred_rs``.

    Provide exactly one of ``slide_list`` or ``slide_info_json``.

    Args:
        bag_root: Parent dir with per-slide subdirectories of bag h5 files.
        csv_root: Root of ``score_nc`` output; CSV at
            ``csv_root/<slide>/<slide>_scores.csv``.
        bag_pattern: Glob for the targeted bags under each slide dir
            (e.g. ``'gammasat3bags_*.h5'``).
        slide_list: Slide subdir names. Mutually exclusive with
            ``slide_info_json``.
        slide_info_json: JSON-Lines file with a ``patient_id`` per record.
            Mutually exclusive with ``slide_list``.
        score_column: CSV column whose value is written into the h5 dataset.
        dataset_name: Name of the h5 dataset to create.
        verbose: When ``True``, emit per-slide INFO logging; otherwise show
            only the progress bar plus warnings/errors.
    """
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

    if (slide_list is None) == (slide_info_json is None):
        raise ValueError("Provide exactly one of --slide_list or --slide_info_json")
    if slide_info_json is not None:
        slide_list = _patient_ids_from_jsonl(slide_info_json)
        log.info("Loaded %d slide IDs from %s", len(slide_list), slide_info_json)
    assert slide_list is not None  # type narrowing for pyrefly

    for slide_name in tqdm(slide_list, desc="Slides"):
        _process_one_slide(
            slide_name,
            bag_root=bag_root,
            csv_root=csv_root,
            bag_pattern=bag_pattern,
            score_column=score_column,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    CLI(main, as_positional=False)
