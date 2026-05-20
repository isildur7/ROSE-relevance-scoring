"""PatchDataset, balanced batch sampler, slide-level splits, and image transforms."""

import random
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

PINNED_TRAIN_SLIDES: frozenset[str] = frozenset({"CF14-003066A-1"})
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def make_splits(
    parquet_path: Path,
    patch_dir: Path,
    seed: int = 42,
    split_mode: Literal["slide", "random"] = "slide",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the patch annotation table into train / val / test sets.

    Two modes are supported:

    * ``"slide"`` (default): slide-level stratified split.  ``CF14-003066A-1``
      is pinned to train; remaining slides are split ~60 / 20 / 20, stratified
      by each slide's positive-label fraction.
    * ``"random"``: plain patch-level random split (~60 / 20 / 20), stratified
      by patch label.  Slide identity is ignored entirely.

    Args:
        parquet_path: Path to ``patches_annotations.parquet`` with columns
            ``filepath`` (filename only) and ``label`` (0 or 1).
        patch_dir: Directory containing the JPEG patches; prepended to each filepath.
        seed: Random seed for reproducibility.
        split_mode: ``"slide"`` for slide-level grouping; ``"random"`` for a
            raw patch-level split with no slide consideration.

    Returns:
        Tuple of (train_df, val_df, test_df), each with columns
        ``filepath`` (absolute path) and ``label`` (plus ``slide`` in slide mode).
    """
    df = pd.read_parquet(parquet_path)
    df["filepath"] = df["filepath"].apply(lambda f: str(patch_dir / f))

    if split_mode == "random":
        train_df, temp_df = train_test_split(
            df, test_size=0.40, stratify=df["label"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    df["slide"] = df["filepath"].apply(lambda f: Path(f).stem.split("__fovr")[0])

    pinned = df[df["slide"].isin(PINNED_TRAIN_SLIDES)]
    rest = df[~df["slide"].isin(PINNED_TRAIN_SLIDES)]

    slide_stats = rest.groupby("slide")["label"].mean().reset_index()
    slide_stats.columns = ["slide", "pos_frac"]
    slide_stats["bin"] = (
        slide_stats["pos_frac"] >= slide_stats["pos_frac"].median()
    ).astype(int)

    slides = slide_stats["slide"].tolist()
    bins = slide_stats["bin"].tolist()

    train_slides, temp_slides, _, temp_bins = train_test_split(
        slides, bins, test_size=0.40, stratify=bins, random_state=seed
    )
    val_slides, test_slides = train_test_split(
        temp_slides, test_size=0.50, stratify=temp_bins, random_state=seed
    )

    train_df = pd.concat(
        [pinned, rest[rest["slide"].isin(set(train_slides))]],
        ignore_index=True,
    )
    val_df = rest[rest["slide"].isin(set(val_slides))].reset_index(drop=True)
    test_df = rest[rest["slide"].isin(set(test_slides))].reset_index(drop=True)

    return train_df, val_df, test_df


class BalancedBatchSampler(Sampler):
    """Batch sampler yielding exactly 50 % positive / 50 % negative indices per batch.

    Designed to be passed as ``batch_sampler=`` to ``DataLoader``.

    * ``undersample``: one epoch exhausts all positives exactly once; negatives are
      sampled with replacement to match. Epoch length ≈ ``2 * n_positives``.
    * ``oversample``: one epoch exhausts all negatives exactly once; positives are
      sampled with replacement. Epoch length ≈ ``2 * n_negatives``.

    Args:
        labels: Integer label (0 or 1) for each sample in the dataset.
        mode: ``"undersample"`` or ``"oversample"``.
        batch_size: Total batch size; must be even.
        seed: Base random seed; incremented each call to ``__iter__`` for epoch variety.
    """

    def __init__(
        self,
        labels: list[int],
        mode: Literal["undersample", "oversample"],
        batch_size: int,
        seed: int = 42,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed
        self._iter_count = 0

        self.pos_idx: list[int] = [i for i, lbl in enumerate(labels) if lbl == 1]
        self.neg_idx: list[int] = [i for i, lbl in enumerate(labels) if lbl == 0]

        half = batch_size // 2
        anchor = self.pos_idx if mode == "undersample" else self.neg_idx
        self._n_batches: int = len(anchor) // half

    def __iter__(self):
        """Yield one batch (list of indices) at a time."""
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1
        half = self.batch_size // 2

        if self.mode == "undersample":
            anchors = self.pos_idx.copy()
            pool = self.neg_idx
        else:
            anchors = self.neg_idx.copy()
            pool = self.pos_idx

        rng.shuffle(anchors)
        for i in range(self._n_batches):
            anchor_half = anchors[i * half : (i + 1) * half]
            pool_half = rng.choices(pool, k=half)
            batch = anchor_half + pool_half
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._n_batches


class PatchDataset(Dataset):
    """Dataset for 256×256 JPEG image patches.

    Args:
        df: DataFrame with columns ``filepath`` (absolute path) and ``label`` (int).
        transform: ``torchvision.transforms.v2`` pipeline applied to the uint8 CHW tensor.
    """

    def __init__(self, df: pd.DataFrame, transform: v2.Compose) -> None:
        self.paths: list[str] = df["filepath"].tolist()
        self.labels: list[int] = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = read_image(self.paths[index], mode=ImageReadMode.RGB)  # (3, H, W) uint8
        img = self.transform(img)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return img, label


def make_train_transform(
    aug_strength: Literal["mild", "strong"] = "mild",
) -> v2.Compose:
    """Training augmentation pipeline.

    Uses ``torchvision.transforms.v2`` throughout so transforms operate directly on
    uint8 CHW tensors from ``read_image`` with no PIL round-trips.
    Rotation is applied as exactly 90° with probability 0.5 to avoid interpolation
    artifacts from fractional angles.

    Color jitter values are conservative by design: in DiffQuik cytology staining,
    color is diagnostically meaningful (pink cytoplasm, blue/purple nuclei), so
    large color perturbations would destroy signal.

    Args:
        aug_strength: ``"mild"`` uses subtle jitter (brightness/contrast/saturation=0.05,
            hue=0.02). ``"strong"`` uses more aggressive jitter (brightness=0.2,
            contrast=0.15, saturation=0.1, hue=0.05).
    """
    if aug_strength == "strong":
        jitter = v2.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05)
    else:
        jitter = v2.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02
        )
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation((90, 90))], p=0.5),
            jitter,
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )


def make_eval_transform() -> v2.Compose:
    """Minimal eval/test pipeline — no augmentation."""
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )
