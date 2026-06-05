# Label-Confusion Pair Finder — Design

**Date:** 2026-06-05
**Branch:** `label-confusion-pairs` (do **not** merge to `main`)
**Status:** Approved design, pending spec review

## Problem

We suspect some patches annotated `label=0` (negative) are visually very similar to
patches annotated `label=1` (positive). We want to find and showcase **5–10 example
pairs** (10 per list) where a label-0 patch and a label-1 patch are genuine
look-alikes, each annotated with its slide name and coordinates, so they can be
eyeballed for possible mislabeling / annotation ambiguity.

## Key design decisions (settled during brainstorming)

1. **Similarity signal:** pathology foundation-model embeddings (UNI2), *not* the
   trained classifier. Embeddings capture raw visual similarity and avoid the
   circularity that these patches were in the classifier's training set.
2. **"Close" means absolute, not relative.** We do *not* want each label-0 patch's
   nearest label-1 neighbor (two points can be mutually nearest yet far apart).
   We rank **cross-label pairs by absolute cosine distance** and surface the
   globally smallest distances.
3. **Two separated lists:**
   - **List A** — globally closest cross-label pairs, no filtering. Expected to
     include spatially adjacent same-slide tiles straddling an annotation
     boundary (boundary slop).
   - **List B** — globally closest cross-label pairs *after excluding spatially
     adjacent same-slide pairs*, surfacing genuine cross-region / cross-slide
     look-alikes.
4. **Embedding model:** UNI2-h (Mahmood lab, ViT-H/14, 1536-dim).
   Checkpoint: `/media/data1/amey/backbone_weights/uni2/pytorch_model.bin`.
5. **Count:** 10 pairs per list.

## Inputs

- `patches_annotations.parquet`
  (`/media/Wednesday/Temporary/amey/Informative_classification_data_v2/claude/patches/patches_annotations.parquet`)
  — columns `filepath`, `label`; 40,266 rows (28,177 label-0, 12,089 label-1).
- Patch JPEG dir (same directory as above). Filenames encode
  `slide__fovrR-fovcC__prR-pcC.jpg`.
- Annotation parquet (`annotations_backup_20260604_194852.parquet`) — used to
  recover per-patch `slide`, `fov_r/c`, `patch_r/c`, `top_left_px_x/y` via join on
  the filename-encoded keys.
- UNI2 backbone from `ROSE-processing-v3/feature_extractor` (`backbones.UNI2`).

All paths are CLI/config arguments (jsonargparse), no hardcoding.

## Pipeline

Four stages. Stages 1 and 2–4 are split so the expensive embedding pass is cached
and reused. All outputs under `claude/labelconf/`.

### Stage 1 — `claude/labelconf/embed_patches.py`

- Add `ROSE-processing-v3` to `sys.path` (or import via installed package) to use
  `feature_extractor.backbones.UNI2`.
- New lightweight `torch.utils.data.Dataset` that reads each JPEG from the patch
  dir, applies UNI2's transform (resize to `input_size`=224, ImageNet
  mean/std from the backbone), returns `(tensor, index)`.
- Batched inference on GPU with bf16 autocast (mirror `FeatureExtractor`:
  `batch_size≈256`, `num_workers≈8`, `torch.inference_mode`).
- Output: `claude/labelconf/uni2_embeddings.npz` with
  `features (N,1536) float16`, `filepath (N,)`, `label (N,)`.
- Idempotent: if the npz exists and row count matches the parquet, skip.

### Stage 2 — closest cross-label pairs (in `find_pairs.py`)

- Load embeddings, cast to float32, **L2-normalize** (cosine distance = `1 - dot`).
- Split into label-0 block (`A`, ~28k) and label-1 block (`B`, ~12k).
- Block-wise on GPU: for each row of `A`, compute cosine similarity against all of
  `B`, keep **top-k=5** most-similar label-1 indices + distances. (Process `A` in
  chunks of a few thousand rows to bound memory.)
- Flatten to a candidate list of (`i0`, `i1`, `distance`) — ~140k candidates —
  and sort ascending by distance.

### Stage 3 — build the two lists (in `find_pairs.py`)

- Recover coords: parse `slide/fov/patch` from each `filepath` and join to the
  annotation parquet for `top_left_px_x/y`.
- **Greedy dedupe** while walking the sorted candidates: a patch (either side) may
  appear in **at most one** surfaced pair, so the 10 pairs are distinct examples.
- **List A:** first 10 deduped pairs.
- **List B:** same walk, but skip any pair that is *spatially adjacent same-slide*
  — same `slide` **and** Chebyshev tile-grid distance
  `max(|Δfov_r·8 + Δpatch_r|, |Δfov_c·8 + Δpatch_c|) <= adj_tiles` (default
  `adj_tiles=2`, i.e. within ~2 tiles ≈ 512 px). Take first 10 deduped survivors.
  (`adj_tiles` is a CLI knob. A stricter `--exclude_same_slide` flag is available
  if List B still looks boundary-ish.)

Note on tile-grid distance: each FOV is an 8×8 grid of 256 px patches
(`patch_r/c ∈ 0..7`), so a patch's global tile-row is `fov_r*8 + patch_r` and
tile-col is `fov_c*8 + patch_c`. Chebyshev distance on those is the tile spacing.

### Stage 4 — showcase output (in `find_pairs.py`)

- One montage PNG per list (`list_a.png`, `list_b.png`): 10 rows, each row
  `[label-0 patch | label-1 patch]`, captioned with `slide`, `(top_left_px_x,
  top_left_px_y)`, and cosine distance.
- `pairs.csv`: one row per surfaced pair with full metadata for both patches
  (`list`, `rank`, `distance`, and for each side: `filepath`, `slide`, `fov_r`,
  `fov_c`, `patch_r`, `patch_c`, `top_left_px_x`, `top_left_px_y`).

## Outputs (all under `claude/labelconf/`)

- `uni2_embeddings.npz` — cached embeddings.
- `pairs.csv` — both lists, full metadata.
- `list_a.png`, `list_b.png` — montages.

## Feasibility / compute

- Embedding 40,266 patches with UNI2-h: a few minutes on one GPU (prior runs did
  ~8–20k tiles in 60–160 s/bag).
- Similarity: 28,177 × 12,089 cosine via blocked matmul on GPU — seconds.
- Memory bounded by chunking `A`; never materializes the full 340M-entry matrix.

## Non-goals

- Not retraining or relabeling anything — this only *surfaces* candidates.
- Not a full mislabel-audit tool; just a focused showcase of the closest pairs.
- Not comparing multiple backbones (UNI2 only, per decision).

## Testing / verification

- Sanity-check Stage 1 on `--limit` (e.g. 256 patches) before the full run.
- Verify `uni2_embeddings.npz` row count equals the parquet length and labels
  match.
- Spot-check that List A actually surfaces adjacent same-slide pairs and List B
  does not (validates the filter).
- Final check: open the two montages and confirm captions/coords match the CSV.
