# claude/labelconf/find_pairs.py
"""Stages 2-4: rank cross-label pairs by cosine distance, build two lists, show them.

List A: globally closest (label-0, label-1) pairs, no filtering.
List B: same, but excluding spatially adjacent same-slide pairs (boundary slop).

Outputs (under out_dir): pairs.csv, list_a.png, list_b.png.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pair_utils import (  # noqa: E402
    PairCandidate,
    is_adjacent_same_slide,
    parse_patch_filename,
    select_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _topk_cross_label(
    feats0: torch.Tensor,
    feats1: torch.Tensor,
    topk: int,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each label-0 row, return its top-k nearest label-1 indices + distances.

    Both inputs must already be L2-normalized. Cosine distance = 1 - cosine sim.

    Returns:
        (row0_idx, col1_idx, dist) flattened arrays of length ``n0 * topk``,
        where indices are into the label-0 / label-1 blocks respectively.
    """
    f1 = feats1.to(device)
    n0 = feats0.shape[0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for start in range(0, n0, chunk_size):
        chunk = feats0[start : start + chunk_size].to(device)
        sims = chunk @ f1.T  # (chunk, n1) cosine similarity
        top = torch.topk(sims, k=topk, dim=1, largest=True)
        col1 = top.indices.cpu().numpy()
        dist = (1.0 - top.values).cpu().numpy()
        row0 = np.repeat(np.arange(start, start + chunk.shape[0]), topk)
        rows.append(row0)
        cols.append(col1.reshape(-1))
        dists.append(dist.reshape(-1))
        logger.info("Ranked %d / %d label-0 patches", min(start + chunk_size, n0), n0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(dists)


def _coord_lookup(annotations_parquet: Path) -> dict[tuple, tuple[int, int]]:
    """Map (slide, fov_r, fov_c, patch_r, patch_c) -> (top_left_px_x, top_left_px_y)."""
    ann = pd.read_parquet(annotations_parquet)
    keys = list(
        zip(ann["slide"], ann["fov_r"], ann["fov_c"], ann["patch_r"], ann["patch_c"])
    )
    vals = list(zip(ann["top_left_px_x"].astype(int), ann["top_left_px_y"].astype(int)))
    return dict(zip(keys, vals))


def _px(coord: dict[tuple, tuple[int, int]], cand_key) -> tuple[int, int]:
    """Look up top-left pixel coords for a PatchKey; (-1, -1) if missing."""
    k = (
        cand_key.slide,
        cand_key.fov_r,
        cand_key.fov_c,
        cand_key.patch_r,
        cand_key.patch_c,
    )
    return coord.get(k, (-1, -1))


def _rows_for_csv(
    pairs: list[PairCandidate],
    list_name: str,
    coord: dict[tuple, tuple[int, int]],
) -> list[dict]:
    """Flatten selected pairs into CSV row dicts with full metadata."""
    out: list[dict] = []
    for rank, p in enumerate(pairs):
        x0, y0 = _px(coord, p.key0)
        x1, y1 = _px(coord, p.key1)
        out.append(
            {
                "list": list_name,
                "rank": rank,
                "distance": float(p.dist),
                "fp0": p.fp0,
                "slide0": p.key0.slide,
                "fov_r0": p.key0.fov_r,
                "fov_c0": p.key0.fov_c,
                "patch_r0": p.key0.patch_r,
                "patch_c0": p.key0.patch_c,
                "px_x0": x0,
                "px_y0": y0,
                "fp1": p.fp1,
                "slide1": p.key1.slide,
                "fov_r1": p.key1.fov_r,
                "fov_c1": p.key1.fov_c,
                "patch_r1": p.key1.patch_r,
                "patch_c1": p.key1.patch_c,
                "px_x1": x1,
                "px_y1": y1,
            }
        )
    return out


def _montage(
    pairs: list[PairCandidate],
    patch_dir: Path,
    coord: dict[tuple, tuple[int, int]],
    title: str,
    out_png: Path,
) -> None:
    """Write a montage of [label-0 | label-1] rows with captions."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle(title, fontsize=12)
    for r, p in enumerate(pairs):
        for c, (fp, key, lbl) in enumerate(
            [(p.fp0, p.key0, "label 0"), (p.fp1, p.key1, "label 1")]
        ):
            ax = axes[r, c]
            with Image.open(patch_dir / fp) as im:
                ax.imshow(np.asarray(im.convert("RGB")))
            x, y = _px(coord, key)
            ax.set_title(
                f"{lbl} | {key.slide}\nfov({key.fov_r},{key.fov_c}) "
                f"px({x},{y}) | d={p.dist:.3f}",
                fontsize=7,
            )
            ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(
    embeddings_npz: Path = Path("claude/labelconf/uni2_embeddings.npz"),
    annotations_parquet: Path = Path("annotations_backup_20260604_194852.parquet"),
    patch_dir: Path = Path(
        "/media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches"
    ),
    out_dir: Path = Path("claude/labelconf"),
    n_pairs: int = 10,
    topk: int = 5,
    adj_tiles: int = 2,
    exclude_same_slide: bool = False,
    chunk_size: int = 4096,
    device: str = "cuda",
) -> None:
    """Rank cross-label pairs and write CSV + two montages.

    Args:
        embeddings_npz: Cached UNI2 embeddings from embed_patches.py.
        annotations_parquet: Source of top-left pixel coordinates.
        patch_dir: Directory holding the patch JPEGs (for montages).
        out_dir: Output directory for pairs.csv / list_a.png / list_b.png.
        n_pairs: Pairs to surface per list.
        topk: Nearest label-1 neighbors kept per label-0 patch when building
            candidates.
        adj_tiles: List B excludes same-slide pairs within this Chebyshev tile
            distance.
        exclude_same_slide: If True, List B drops ALL same-slide pairs.
        chunk_size: Label-0 rows processed per GPU block.
        device: Torch device.
    """
    out_dir = Path(out_dir)
    data = np.load(embeddings_npz, allow_pickle=True)
    feats = torch.from_numpy(data["features"].astype(np.float32))
    feats = torch.nn.functional.normalize(feats, dim=1)
    filepaths = [str(s) for s in data["filepath"]]
    labels = data["label"]

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    feats0, feats1 = feats[idx0], feats[idx1]

    row0, col1, dist = _topk_cross_label(feats0, feats1, topk, chunk_size, device)

    keys = [parse_patch_filename(fp) for fp in filepaths]
    candidates = [
        PairCandidate(
            fp0=filepaths[idx0[r]],
            fp1=filepaths[idx1[c]],
            dist=float(d),
            key0=keys[idx0[r]],
            key1=keys[idx1[c]],
        )
        for r, c, d in zip(row0, col1, dist)
    ]

    coord = _coord_lookup(annotations_parquet)

    list_a = select_pairs(candidates, n=n_pairs)

    def accept_b(p: PairCandidate) -> bool:
        if exclude_same_slide:
            return p.key0.slide != p.key1.slide
        return not is_adjacent_same_slide(p.key0, p.key1, adj_tiles)

    list_b = select_pairs(candidates, n=n_pairs, accept=accept_b)

    rows = _rows_for_csv(list_a, "A", coord) + _rows_for_csv(list_b, "B", coord)
    csv_path = out_dir / "pairs.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(rows))

    _montage(
        list_a,
        Path(patch_dir),
        coord,
        "List A: closest cross-label pairs",
        out_dir / "list_a.png",
    )
    _montage(
        list_b,
        Path(patch_dir),
        coord,
        "List B: closest, adjacent same-slide removed",
        out_dir / "list_b.png",
    )
    logger.info("Wrote montages to %s", out_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
