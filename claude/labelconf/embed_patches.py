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
    checkpoint_path: Path = Path(
        "/media/data1/amey/backbone_weights/uni2/pytorch_model.bin"
    ),
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
            feats[idx.numpy()] = (
                out.detach().to(torch.float32).cpu().numpy().astype(np.float16)
            )
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
