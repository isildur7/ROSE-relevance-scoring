"""Evaluate a trained patch classifier checkpoint on the held-out test split."""

import logging
from pathlib import Path
from typing import Literal

import torch
from jsonargparse import CLI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import PatchDataset, make_eval_transform, make_splits
from train import PatchClassifier

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(
    checkpoint_path: Path,
    parquet_path: Path,
    patch_dir: Path,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 8,
    gpu: int = 0,
    split_mode: Literal["slide", "random"] = "slide",
) -> None:
    """Run the test split through a trained checkpoint and print metrics.

    The ``seed`` and ``split_mode`` must match the values used during training
    so the split is identical.

    Args:
        checkpoint_path: Path to a ``.ckpt`` checkpoint file.
        parquet_path: Path to ``patches_annotations.parquet``.
        patch_dir: Directory containing the JPEG patches.
        seed: Random seed — must match the training run to reproduce the same split.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        gpu: GPU index to use (0-based). Ignored if CUDA is unavailable.
        split_mode: ``"slide"`` or ``"random"`` — must match the training run.
    """
    _, _, test_df = make_splits(parquet_path, patch_dir, seed, split_mode)
    log.info(
        "Test split: %d patches (%d positive, %d negative)",
        len(test_df),
        test_df["label"].sum(),
        (test_df["label"] == 0).sum(),
    )

    nw = num_workers
    loader_kwargs: dict = dict(
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0
    )
    if nw > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataset = PatchDataset(test_df, make_eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = PatchClassifier.load_from_checkpoint(checkpoint_path).to(device).eval()

    criterion = torch.nn.BCELoss()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            probs = model(imgs).squeeze(1)
            total_loss += criterion(probs, labels).item()
            n_batches += 1
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    scores = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / n_batches
    preds = (scores >= 0.5).astype(int)

    print("\n=== Test Metrics ===")
    print(f"Loss:      {loss:.4f}")
    print(f"AUC-ROC:   {roc_auc_score(labels, scores):.4f}")
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"F1:        {f1_score(labels, preds, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            labels, preds, target_names=["Non-informative", "Informative"]
        )
    )
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    CLI(main)
