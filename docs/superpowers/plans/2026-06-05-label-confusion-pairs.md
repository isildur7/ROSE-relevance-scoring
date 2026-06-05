# Label-Confusion Pair Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and showcase 10 pairs (per list) of a `label=0` patch and a `label=1` patch that are close in absolute UNI2 embedding distance — genuine visual look-alikes across the annotation boundary — annotated with slide name and coordinates.

**Architecture:** Two CLI scripts plus a pure-logic helper module. `embed_patches.py` runs UNI2 over all 40,266 JPEG patches once and caches embeddings to an `.npz`. `find_pairs.py` loads the cache, ranks cross-label pairs by cosine distance, builds two lists (A = unfiltered, B = adjacent-same-slide pairs removed), and writes a CSV + two montage PNGs. `pair_utils.py` holds the unit-tested pure functions (filename parsing, tile-grid distance, greedy pair selection).

**Tech Stack:** Python 3.12, PyTorch + timm (UNI2-h from `ROSE-processing-v3/feature_extractor`), pandas/pyarrow, matplotlib, jsonargparse, pytest. Conda env: `pytorch_latest`.

---

## Key facts (verified)

- Patches parquet: `/media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches/patches_annotations.parquet` — columns `filepath`, `label`; 40,266 rows (28,177 label-0, 12,089 label-1).
- Patch JPEGs live in the same dir; filenames look like `CF14-003233D-4__fovr1-fovc9__pr0-pc1.jpg`.
- Each FOV is an 8×8 grid of 256px patches: `patch_r`, `patch_c` ∈ 0..7. Global tile row = `fov_r*8 + patch_r`, global tile col = `fov_c*8 + patch_c`.
- Annotation parquet (`annotations_backup_20260604_194852.parquet`) maps `(slide, fov_r, fov_c, patch_r, patch_c)` → `top_left_px_x`, `top_left_px_y`.
- UNI2-h checkpoint: `/media/data1/amey/backbone_weights/uni2/pytorch_model.bin`.
- UNI2 backbone (`feature_extractor.backbones.UNI2`): `input_size=224`, ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)`, `embed_dim=1536`, `forward(x)->(B,1536)`.
- Feature-extractor repo root: `/home/amey/ROSE-processing-v3` (added to `sys.path` to import `feature_extractor`).

All scripts use `jsonargparse.CLI`, `pathlib.Path`, full type hints and docstrings. No `os.path`, no `argparse`. Run everything with `conda run -n pytorch_latest`. Work on branch `label-confusion-pairs` — do **not** merge to `main`.

---

## File Structure

- Create `claude/labelconf/pair_utils.py` — pure functions: parse patch filename, global tile coords, adjacency test, greedy pair selection. No torch/IO.
- Create `claude/labelconf/test_pair_utils.py` — pytest unit tests for `pair_utils`.
- Create `claude/labelconf/embed_patches.py` — Stage 1 CLI: UNI2 embeddings → `uni2_embeddings.npz`.
- Create `claude/labelconf/find_pairs.py` — Stages 2–4 CLI: ranking, two lists, CSV + montages.
- Outputs (git-ignored, not committed): `claude/labelconf/uni2_embeddings.npz`, `claude/labelconf/pairs.csv`, `claude/labelconf/list_a.png`, `claude/labelconf/list_b.png`.

---

## Task 1: Pure helper functions + tests (`pair_utils.py`)

**Files:**
- Create: `claude/labelconf/pair_utils.py`
- Test: `claude/labelconf/test_pair_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# claude/labelconf/test_pair_utils.py
"""Unit tests for the pure label-confusion pair helpers."""

from __future__ import annotations

import pytest

from pair_utils import (
    PairCandidate,
    PatchKey,
    global_tile_rc,
    is_adjacent_same_slide,
    parse_patch_filename,
    select_pairs,
)


def test_parse_patch_filename_basic() -> None:
    key = parse_patch_filename("CF14-003233D-4__fovr1-fovc9__pr0-pc1.jpg")
    assert key == PatchKey(
        slide="CF14-003233D-4", fov_r=1, fov_c=9, patch_r=0, patch_c=1
    )


def test_parse_patch_filename_rejects_bad_name() -> None:
    with pytest.raises(ValueError):
        parse_patch_filename("not_a_valid_name.jpg")


def test_global_tile_rc() -> None:
    key = PatchKey(slide="s", fov_r=2, fov_c=3, patch_r=4, patch_c=5)
    assert global_tile_rc(key) == (2 * 8 + 4, 3 * 8 + 5)


def test_is_adjacent_same_slide_true_within_radius() -> None:
    k0 = PatchKey("s", 0, 0, 0, 0)  # tile (0, 0)
    k1 = PatchKey("s", 0, 0, 0, 2)  # tile (0, 2) -> cheb dist 2
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is True


def test_is_adjacent_same_slide_false_when_far() -> None:
    k0 = PatchKey("s", 0, 0, 0, 0)  # tile (0, 0)
    k1 = PatchKey("s", 0, 0, 0, 4)  # tile (0, 4) -> cheb dist 4
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is False


def test_is_adjacent_same_slide_false_for_different_slide() -> None:
    k0 = PatchKey("a", 0, 0, 0, 0)
    k1 = PatchKey("b", 0, 0, 0, 0)
    assert is_adjacent_same_slide(k0, k1, adj_tiles=2) is False


def _cand(fp0: str, fp1: str, dist: float) -> PairCandidate:
    return PairCandidate(
        fp0=fp0,
        fp1=fp1,
        dist=dist,
        key0=parse_patch_filename(fp0),
        key1=parse_patch_filename(fp1),
    )


def test_select_pairs_sorts_and_dedupes() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    # Reuses the label-0 patch from `a`; must be skipped by dedupe.
    b = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-3__fovr0-fovc0__pr0-pc0.jpg", 0.20)
    c = _cand("CF-4__fovr0-fovc0__pr0-pc0.jpg", "CF-5__fovr0-fovc0__pr0-pc0.jpg", 0.30)
    out = select_pairs([c, a, b], n=2)  # unsorted input on purpose
    assert [p.dist for p in out] == [0.10, 0.30]


def test_select_pairs_applies_accept_filter() -> None:
    a = _cand("CF-1__fovr0-fovc0__pr0-pc0.jpg", "CF-2__fovr0-fovc0__pr0-pc0.jpg", 0.10)
    c = _cand("CF-4__fovr0-fovc0__pr0-pc0.jpg", "CF-5__fovr0-fovc0__pr0-pc0.jpg", 0.30)
    out = select_pairs([a, c], n=2, accept=lambda p: p.dist > 0.2)
    assert [p.dist for p in out] == [0.30]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n pytorch_latest python -m pytest claude/labelconf/test_pair_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pair_utils'`.

- [ ] **Step 3: Implement `pair_utils.py`**

```python
# claude/labelconf/pair_utils.py
"""Pure helpers for the label-confusion pair finder.

No torch / IO here so the ranking and filtering logic stays unit-testable.
A "pair" is one label-0 patch (side 0) and one label-1 patch (side 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

#: Patches are an 8x8 grid of 256px tiles per FOV.
PATCHES_PER_FOV_EDGE: int = 8

_NAME_RE = re.compile(
    r"^(?P<slide>.+)__fovr(?P<fov_r>\d+)-fovc(?P<fov_c>\d+)"
    r"__pr(?P<patch_r>\d+)-pc(?P<patch_c>\d+)\.jpg$"
)


@dataclass(frozen=True)
class PatchKey:
    """Identity of one patch, parsed from its JPEG filename."""

    slide: str
    fov_r: int
    fov_c: int
    patch_r: int
    patch_c: int


@dataclass(frozen=True)
class PairCandidate:
    """A candidate (label-0, label-1) pair and its cosine distance."""

    fp0: str
    fp1: str
    dist: float
    key0: PatchKey
    key1: PatchKey


def parse_patch_filename(filename: str) -> PatchKey:
    """Parse a patch JPEG filename into a :class:`PatchKey`.

    Args:
        filename: e.g. ``"CF14-003233D-4__fovr1-fovc9__pr0-pc1.jpg"``.

    Returns:
        The parsed :class:`PatchKey`.

    Raises:
        ValueError: If ``filename`` does not match the expected pattern.
    """
    m = _NAME_RE.match(filename)
    if m is None:
        raise ValueError(f"Unparseable patch filename: {filename!r}")
    return PatchKey(
        slide=m.group("slide"),
        fov_r=int(m.group("fov_r")),
        fov_c=int(m.group("fov_c")),
        patch_r=int(m.group("patch_r")),
        patch_c=int(m.group("patch_c")),
    )


def global_tile_rc(key: PatchKey) -> tuple[int, int]:
    """Return the patch's (row, col) on the slide-wide tile grid."""
    row = key.fov_r * PATCHES_PER_FOV_EDGE + key.patch_r
    col = key.fov_c * PATCHES_PER_FOV_EDGE + key.patch_c
    return row, col


def is_adjacent_same_slide(k0: PatchKey, k1: PatchKey, adj_tiles: int) -> bool:
    """True if both patches are on the same slide within ``adj_tiles`` (Chebyshev)."""
    if k0.slide != k1.slide:
        return False
    r0, c0 = global_tile_rc(k0)
    r1, c1 = global_tile_rc(k1)
    return max(abs(r0 - r1), abs(c0 - c1)) <= adj_tiles


def select_pairs(
    candidates: list[PairCandidate],
    n: int,
    accept: Optional[Callable[[PairCandidate], bool]] = None,
) -> list[PairCandidate]:
    """Greedily pick the ``n`` closest pairs, deduping by patch.

    Candidates are sorted ascending by distance. A patch (either side) appears
    in at most one returned pair, so the result is ``n`` distinct examples.

    Args:
        candidates: Candidate pairs (any order).
        n: Number of pairs to return.
        accept: Optional predicate; a candidate is kept only if it returns True.

    Returns:
        Up to ``n`` pairs, ascending by distance.
    """
    used: set[str] = set()
    chosen: list[PairCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.dist):
        if cand.fp0 in used or cand.fp1 in used:
            continue
        if accept is not None and not accept(cand):
            continue
        chosen.append(cand)
        used.add(cand.fp0)
        used.add(cand.fp1)
        if len(chosen) == n:
            break
    return chosen
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n pytorch_latest python -m pytest claude/labelconf/test_pair_utils.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add claude/labelconf/pair_utils.py claude/labelconf/test_pair_utils.py
git commit -m "Add pure helpers for label-confusion pair finding"
```

---

## Task 2: UNI2 embedding script (`embed_patches.py`)

This script needs the GPU, the model, and the real patch dir, so it is verified by a smoke run on a small `--limit`, not by pytest.

**Files:**
- Create: `claude/labelconf/embed_patches.py`

- [ ] **Step 1: Implement `embed_patches.py`**

```python
# claude/labelconf/embed_patches.py
"""Stage 1: embed every patch JPEG with UNI2-h and cache to an .npz.

Reuses the UNI2 backbone from the ROSE-processing-v3 feature_extractor package.
Output npz keys: ``features`` (N, 1536) float16, ``filepath`` (N,) str,
``label`` (N,) int64.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PatchJpegDataset(Dataset):
    """Yield ``(normalized_chw_tensor, index)`` for each patch JPEG.

    Mirrors the preprocessing in ``feature_extractor.dataset.BagH5Dataset``:
    uint8 -> float32/255 -> bilinear-antialias resize to ``input_size`` ->
    per-channel normalize.
    """

    def __init__(
        self,
        filepaths: list[str],
        patch_dir: Path,
        input_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.filepaths = filepaths
        self.patch_dir = Path(patch_dir)
        self.input_size = int(input_size)
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.patch_dir / self.filepaths[index]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)  # HWC
        img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        img = img.to(dtype=torch.float32).div_(255.0)
        if img.shape[-1] != self.input_size or img.shape[-2] != self.input_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        img.sub_(self._mean).div_(self._std)
        return img, index


def main(
    parquet_path: Path,
    patch_dir: Path,
    checkpoint_path: Path = Path("/media/data1/amey/backbone_weights/uni2/pytorch_model.bin"),
    output_path: Path = Path("claude/labelconf/uni2_embeddings.npz"),
    processing_repo: Path = Path("/home/amey/ROSE-processing-v3"),
    batch_size: int = 256,
    num_workers: int = 8,
    limit: int = 0,
    device: str = "cuda",
) -> None:
    """Embed all patches with UNI2-h and write a cached npz.

    Args:
        parquet_path: ``patches_annotations.parquet`` (columns filepath, label).
        patch_dir: Directory holding the patch JPEGs.
        checkpoint_path: Local UNI2-h state-dict ``.bin``.
        output_path: Destination npz.
        processing_repo: Repo root added to sys.path to import feature_extractor.
        batch_size: Forward-pass batch size.
        num_workers: DataLoader workers.
        limit: If > 0, only embed the first ``limit`` patches (smoke test).
        device: Torch device.
    """
    output_path = Path(output_path)
    df = pd.read_parquet(parquet_path)
    if limit > 0:
        df = df.iloc[:limit].reset_index(drop=True)
    filepaths = df["filepath"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    if output_path.exists() and limit == 0:
        cached = np.load(output_path, allow_pickle=True)
        if cached["features"].shape[0] == len(filepaths):
            logger.info("Embeddings already complete at %s; skipping.", output_path)
            return
        logger.info("Existing npz row count mismatch; recomputing.")

    sys.path.insert(0, str(processing_repo))
    from feature_extractor.backbones import build_backbone

    backbone = build_backbone("uni2", checkpoint_path=Path(checkpoint_path))
    backbone = backbone.to(device).eval()

    dataset = PatchJpegDataset(
        filepaths=filepaths,
        patch_dir=Path(patch_dir),
        input_size=backbone.input_size,
        mean=backbone.mean,
        std=backbone.std,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats = np.empty((len(filepaths), backbone.embed_dim), dtype=np.float16)
    with torch.inference_mode():
        for imgs, idx in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = backbone(imgs)
            feats[idx.numpy()] = out.detach().to(torch.float32).cpu().numpy().astype(np.float16)
            logger.info("Embedded %d / %d", int(idx.max()) + 1, len(filepaths))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=feats,
        filepath=np.array(filepaths, dtype=object),
        label=labels,
    )
    logger.info("Wrote %s features=%s", output_path, feats.shape)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
```

- [ ] **Step 2: Smoke-test on 256 patches**

Run:
```bash
conda run -n pytorch_latest python claude/labelconf/embed_patches.py \
  /media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches/patches_annotations.parquet \
  /media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches \
  --output_path claude/labelconf/_smoke.npz \
  --limit 256
```
Expected: logs an "UNI2 state_dict mismatch" warning only if weights differ (should NOT appear), then `Wrote claude/labelconf/_smoke.npz features=(256, 1536)`.

- [ ] **Step 3: Verify the smoke npz shape and dtype**

Run:
```bash
conda run -n pytorch_latest python -c "
import numpy as np
d = np.load('claude/labelconf/_smoke.npz', allow_pickle=True)
print(d['features'].shape, d['features'].dtype)
print(d['filepath'].shape, d['label'].shape)
assert d['features'].shape == (256, 1536)
assert np.isfinite(d['features'].astype('float32')).all()
print('OK')
"
```
Expected: `(256, 1536) float16`, `(256,) (256,)`, `OK`.

- [ ] **Step 4: Remove the smoke file and commit the script**

```bash
rm -f claude/labelconf/_smoke.npz
git add claude/labelconf/embed_patches.py
git commit -m "Add UNI2 patch embedding script"
```

---

## Task 3: Full embedding run

**Files:** none (produces `claude/labelconf/uni2_embeddings.npz`, git-ignored).

- [ ] **Step 1: Run the full embedding pass**

Run:
```bash
conda run -n pytorch_latest python claude/labelconf/embed_patches.py \
  /media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches/patches_annotations.parquet \
  /media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches
```
Expected: finishes in a few minutes; `Wrote claude/labelconf/uni2_embeddings.npz features=(40266, 1536)`.

- [ ] **Step 2: Verify row count matches the parquet and labels line up**

Run:
```bash
conda run -n pytorch_latest python -c "
import numpy as np, pandas as pd
d = np.load('claude/labelconf/uni2_embeddings.npz', allow_pickle=True)
df = pd.read_parquet('/media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches/patches_annotations.parquet')
assert d['features'].shape == (len(df), 1536), d['features'].shape
assert (d['label'] == df['label'].to_numpy()).all()
assert list(d['filepath']) == df['filepath'].astype(str).tolist()
print('counts', np.bincount(d['label']))
print('OK')
"
```
Expected: `counts [28177 12089]`, `OK`. (No commit — the npz is an artifact.)

---

## Task 4: Pair-ranking + showcase script (`find_pairs.py`)

Depends on `pair_utils` (Task 1) and the embeddings (Task 3). Verified by running it and checking the outputs.

**Files:**
- Create: `claude/labelconf/find_pairs.py`

- [ ] **Step 1: Implement `find_pairs.py`**

```python
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from pair_utils import (
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
        zip(
            ann["slide"], ann["fov_r"], ann["fov_c"], ann["patch_r"], ann["patch_c"]
        )
    )
    vals = list(zip(ann["top_left_px_x"].astype(int), ann["top_left_px_y"].astype(int)))
    return dict(zip(keys, vals))


def _px(coord: dict[tuple, tuple[int, int]], cand_key) -> tuple[int, int]:
    """Look up top-left pixel coords for a PatchKey; (-1, -1) if missing."""
    k = (cand_key.slide, cand_key.fov_r, cand_key.fov_c, cand_key.patch_r, cand_key.patch_c)
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
        list_a, Path(patch_dir), coord, "List A: closest cross-label pairs",
        out_dir / "list_a.png",
    )
    _montage(
        list_b, Path(patch_dir), coord,
        "List B: closest, adjacent same-slide removed", out_dir / "list_b.png",
    )
    logger.info("Wrote montages to %s", out_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
```

- [ ] **Step 2: Run it**

Run:
```bash
conda run -n pytorch_latest python claude/labelconf/find_pairs.py
```
Expected: ranking logs, then `Wrote claude/labelconf/pairs.csv (20 rows)` and `Wrote montages to claude/labelconf`.

- [ ] **Step 3: Verify the CSV and the List A / List B distinction**

Run:
```bash
conda run -n pytorch_latest python -c "
import pandas as pd
df = pd.read_csv('claude/labelconf/pairs.csv')
assert set(df['list']) == {'A', 'B'}
assert (df['list'] == 'A').sum() == 10 and (df['list'] == 'B').sum() == 10
# Every List B pair must NOT be same-slide-within-2-tiles.
b = df[df['list'] == 'B']
def cheb(r):
    if r.slide0 != r.slide1:
        return 99
    return max(abs((r.fov_r0*8+r.patch_r0)-(r.fov_r1*8+r.patch_r1)),
              abs((r.fov_c0*8+r.patch_c0)-(r.fov_c1*8+r.patch_c1)))
assert (b.apply(cheb, axis=1) > 2).all(), 'List B leaked adjacent same-slide pairs'
print('distances A:', df[df.list=='A']['distance'].round(3).tolist())
print('distances B:', df[df.list=='B']['distance'].round(3).tolist())
print('OK')
"
```
Expected: 10 rows each, List B has no adjacent same-slide pairs, `OK`. Distances ascending within each list.

- [ ] **Step 4: Confirm montages exist and are non-empty**

Run: `ls -la claude/labelconf/list_a.png claude/labelconf/list_b.png`
Expected: both files present, non-zero size. (Open them to eyeball the look-alikes.)

- [ ] **Step 5: Commit the script**

```bash
git add claude/labelconf/find_pairs.py
git commit -m "Add cross-label pair ranking and showcase script"
```

---

## Task 5: Ignore artifacts + document the scripts

**Files:**
- Modify: `.gitignore`
- Modify: `README.local.md`

- [ ] **Step 1: Ignore the generated artifacts**

Add to `.gitignore`:
```
claude/labelconf/*.npz
claude/labelconf/*.csv
claude/labelconf/*.png
```

- [ ] **Step 2: Document both scripts in `README.local.md`**

Append a section describing `embed_patches.py` and `find_pairs.py` with the exact run commands from Tasks 3 and 4 and their arguments (match the existing README's table style).

- [ ] **Step 3: Verify the artifacts are ignored**

Run: `git status --porcelain claude/labelconf/`
Expected: only `pair_utils.py`, `test_pair_utils.py`, `embed_patches.py`, `find_pairs.py` show as tracked; no `.npz/.csv/.png`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.local.md
git commit -m "Ignore labelconf artifacts and document scripts"
```

---

## Self-Review notes

- **Spec coverage:** Stage 1 → Task 2/3; Stage 2 (`_topk_cross_label`, cosine, top-k) → Task 4; Stage 3 (coord join, greedy dedupe, List A/B, adjacency filter) → Tasks 1+4; Stage 4 (montages + CSV) → Task 4. UNI2 checkpoint, 10 pairs, two lists, cosine distance, `adj_tiles` knob, `exclude_same_slide` fallback all present.
- **No placeholders:** every code step is complete and runnable.
- **Type consistency:** `PatchKey`, `PairCandidate`, `parse_patch_filename`, `global_tile_rc`, `is_adjacent_same_slide`, `select_pairs` are defined in Task 1 and used with the same signatures in Task 4. `backbone.input_size/mean/std/embed_dim` match the verified `feature_extractor.backbones` API.
- **Imports note:** `find_pairs.py` and `test_pair_utils.py` import `pair_utils` as a top-level module, so they must be run from `claude/labelconf/` OR that dir must be on `sys.path`. Run commands invoke scripts by path from repo root; pytest is invoked on the test file directly. If an import error occurs, run from inside `claude/labelconf/` (e.g. prefix with `PYTHONPATH=claude/labelconf`).
```
