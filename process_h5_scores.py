"""Data-pipeline step: score bag tiles and write results back into the h5 files.

Mirrors the side effects of the legacy
``ROSE-processing-v2/compute_relevance_scores.py`` against the new
``PatchClassifier``:

1. For each ``<bag_pattern>`` h5 file under ``<bag_root>/<slide_name>/``, score
   every tile and write a ``relevance_scores`` dataset into the bag file in
   place (replacing any existing dataset of that name).
2. Save a sibling ``<bag_stem>_relevance.npz`` with ``relevance_scores``,
   ``coords``, ``cams``.
3. Map the bag's scores onto every matching
   ``features_<bag_stem-no-underscore>*.h5`` in the same directory by
   ``(coords, cam_xy)`` and write a ``relevance_scores`` dataset there too.
   Unmatched feature rows default to ``0.0`` and trigger a warning.

This is a pure processing tool — no visualizations, no CSV. Run ``score_h5.py``
for the visualization suite.

Slide selection accepts either an inline ``--slide_list`` or a JSON-Lines
file (``--slide_info_json``) with one record per slide and a ``patient_id``
field (mirrors ``ROSE-processing-v2/data_jsons/*.json``).

Example:
    python process_h5_scores.py \\
        --bag_root /media/data1/kanghyun/ROSE_MIL/DUMC \\
        --slide_info_json /home/amey/ROSE-processing-v2/data_jsons/test.json \\
        --checkpoint results/.../last.ckpt \\
        --bag_pattern 'Bag*.h5'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from jsonargparse import CLI
from tqdm import tqdm

from scoring.model_io import load_classifier, resolve_device
from scoring.preprocess import to_model_batch
from scoring.scorer import score_tile_batches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _patient_ids_from_jsonl(path: Path) -> list[str]:
    """Read a JSON-Lines file and return its ``patient_id`` column in order.

    Args:
        path: Path to a ``.json`` / ``.jsonl`` file with one JSON object per
            line. Each line must contain a ``patient_id`` field, which is
            used as the slide subdirectory name under ``bag_root``.

    Returns:
        List of ``patient_id`` strings in file order. Blank lines are skipped.
    """
    ids: list[str] = []
    with path.open("r") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if "patient_id" not in record:
                raise ValueError(f"{path}:{line_num}: missing 'patient_id' field")
            ids.append(str(record["patient_id"]))
    if not ids:
        raise ValueError(f"{path}: no records found")
    return ids


def _score_images(
    images_np: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score every tile in ``images_np`` with ``model``.

    Args:
        images_np: ``(N, H, W, 3)`` uint8 tile array.
        model: Loaded ``PatchClassifier`` in eval mode.
        device: Inference device.
        batch_size: Tiles per inference batch.

    Returns:
        ``(N,)`` float array of per-tile scores in ``[0, 1]``.
    """
    mini_batches: list[torch.Tensor] = []
    for start in range(0, images_np.shape[0], batch_size):
        mini_batches.append(
            to_model_batch(images_np[start : start + batch_size], device)
        )
    return score_tile_batches(model, mini_batches, device)


def _replace_dataset(f: h5py.File, name: str, data: np.ndarray) -> None:
    """Delete ``name`` if present, then create it with ``data``.

    Args:
        f: Open HDF5 file in ``r+`` mode.
        name: Dataset name to replace.
        data: Array to write.
    """
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)


def _map_scores_to_feature_file(
    feature_path: Path,
    bag_lookup: dict[tuple[int, int, int, int], float],
) -> None:
    """Write ``relevance_scores`` into a feature file by ``(coords, cam_xy)`` match.

    Args:
        feature_path: Feature HDF5 path to update in place.
        bag_lookup: Map ``(coord_y, coord_x, cam_y, cam_x) -> score`` built
            from the parent bag.
    """
    with h5py.File(feature_path, "r+") as f:
        if "coords" not in f or "cam_xy" not in f:
            log.warning(
                "%s: missing coords/cam_xy; skipping feature-file mapping",
                feature_path,
            )
            return
        coords = f["coords"][:]
        cam_xy = f["cam_xy"][:]

        n = coords.shape[0]
        mapped = np.zeros(n, dtype=np.float32)
        unmatched = 0
        for i in range(n):
            key = (
                int(coords[i, 0]),
                int(coords[i, 1]),
                int(cam_xy[i, 0]),
                int(cam_xy[i, 1]),
            )
            score = bag_lookup.get(key)
            if score is None:
                unmatched += 1
            else:
                mapped[i] = score

        _replace_dataset(f, "relevance_scores", mapped)

    if unmatched:
        log.warning(
            "%s: %d/%d feature rows had no matching bag entry (filled with 0.0)",
            feature_path,
            unmatched,
            n,
        )


def _process_one_bag(
    bag_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> None:
    """Score one bag h5, write back in place, save NPZ, map onto feature files.

    Args:
        bag_path: Path to the bag HDF5 file.
        model: Loaded ``PatchClassifier`` in eval mode.
        device: Inference device.
        batch_size: Tiles per inference batch.
    """
    with h5py.File(bag_path, "r+") as f:
        images = f["images"][:]
        coords = f["coords"][:]
        cams = f["cams"][:]

        scores = _score_images(images, model, device, batch_size)
        log.info(
            "%s: scored %d tiles  min=%.3f  max=%.3f  mean=%.3f",
            bag_path.name,
            len(scores),
            float(scores.min()) if len(scores) else float("nan"),
            float(scores.max()) if len(scores) else float("nan"),
            float(scores.mean()) if len(scores) else float("nan"),
        )

        _replace_dataset(f, "relevance_scores", scores)

    npz_path = bag_path.with_name(f"{bag_path.stem}_relevance.npz")
    np.savez(
        npz_path,
        relevance_scores=scores,
        coords=coords,
        cams=cams,
    )
    log.info("Wrote %s", npz_path)

    bag_lookup: dict[tuple[int, int, int, int], float] = {}
    for i in range(scores.shape[0]):
        key = (
            int(coords[i, 0]),
            int(coords[i, 1]),
            int(cams[i, 0]),
            int(cams[i, 1]),
        )
        bag_lookup[key] = float(scores[i])

    feature_glob = f"features_{bag_path.stem.replace('_', '')}*.h5"
    feature_paths = sorted(bag_path.parent.glob(feature_glob))
    for feature_path in feature_paths:
        _map_scores_to_feature_file(feature_path, bag_lookup)
        log.info("Mapped scores into %s", feature_path.name)


def main(
    bag_root: Path,
    checkpoint: Path,
    bag_pattern: str,
    slide_list: list[str] | None = None,
    slide_info_json: Path | None = None,
    gpu: int | None = None,
    batch_size: int = 1024,
    verbose: bool = False,
) -> None:
    """Score bag h5 files in place for every slide selected.

    Provide exactly one of ``slide_list`` or ``slide_info_json``.

    Args:
        bag_root: Parent directory containing per-slide subdirectories with
            ``bag_pattern`` files (and optionally matching ``features_*.h5``
            files).
        checkpoint: Trained ``PatchClassifier`` checkpoint (``.ckpt``).
        bag_pattern: Glob pattern matched directly under each slide
            subdirectory (e.g. ``"Bag*.h5"`` or ``"bag_*.h5"``).
        slide_list: Subdirectory names under ``bag_root`` to process.
            Mutually exclusive with ``slide_info_json``.
        slide_info_json: Path to a JSON-Lines file (one JSON object per
            line) with a ``patient_id`` field per record. ``patient_id``
            is used as the slide subdirectory name under ``bag_root``.
            Mutually exclusive with ``slide_list``.
        gpu: GPU index. ``None`` auto-picks CUDA; ``-1`` forces CPU.
        batch_size: Tiles per inference batch.
        verbose: When ``True``, emit per-bag tile-count and score stats at
            INFO level. Default shows only the tqdm progress bar plus
            warnings/errors.
    """
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

    if (slide_list is None) == (slide_info_json is None):
        raise ValueError("Provide exactly one of --slide_list or --slide_info_json")
    if slide_info_json is not None:
        slide_list = _patient_ids_from_jsonl(slide_info_json)
        log.info("Loaded %d slide IDs from %s", len(slide_list), slide_info_json)
    assert slide_list is not None  # type narrowing for pyrefly

    device = resolve_device(gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_classifier(checkpoint, device)
    log.info("Loaded model on %s", device)

    for slide_name in tqdm(slide_list, desc="Slides"):
        slide_dir = bag_root / slide_name
        if not slide_dir.is_dir():
            log.warning("Slide directory not found: %s; skipping", slide_dir)
            continue
        bag_paths = sorted(slide_dir.glob(bag_pattern))
        if not bag_paths:
            log.warning(
                "No bags matching %r under %s; skipping",
                bag_pattern,
                slide_dir,
            )
            continue
        for bag_path in bag_paths:
            try:
                _process_one_bag(bag_path, model, device, batch_size)
            except Exception:
                log.exception("Failed to process bag %s; continuing", bag_path)


if __name__ == "__main__":
    CLI(main, as_positional=False)
