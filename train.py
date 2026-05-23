"""Train a binary patch classifier (ResNet18) with PyTorch Lightning."""

import logging
import pickle
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
import wandb
import yaml
from jsonargparse import CLI
from lightning import seed_everything
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall

from dataset import (
    BalancedBatchSampler,
    PatchDataset,
    make_eval_transform,
    make_splits,
    make_train_transform,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PatchClassifier(LightningModule):
    """Binary patch classifier backed by ResNet-18 or EfficientNet-B0.

    Outputs a single logit; sigmoid is applied at inference time to obtain a score in [0, 1].
    Balanced batch sampling makes ``pos_weight`` in BCEWithLogitsLoss unnecessary.

    Args:
        pretrained: Use ImageNet pretrained weights. Default: False.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Total training epochs; sets ``T_max`` for CosineAnnealingLR.
        lr_schedule: ``"cosine"`` for CosineAnnealingLR, ``"constant"`` for fixed LR.
        architecture: ``"resnet18"`` or ``"efficientnet_b0"``.
        dropout_rate: Dropout probability inserted before the classification head.
            ``0.0`` disables dropout.
    """

    def __init__(
        self,
        pretrained: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        lr_schedule: Literal["cosine", "constant"] = "cosine",
        architecture: Literal["resnet18", "efficientnet_b0"] = "resnet18",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._lr: float = lr
        self._weight_decay: float = weight_decay
        self._max_epochs: int = max_epochs
        self._lr_schedule: Literal["cosine", "constant"] = lr_schedule

        if architecture == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tvm.efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(1280, 1),
            )
        else:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tvm.resnet18(weights=weights)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, 1),
            )

        self.criterion = nn.BCELoss()

        self.train_auroc = BinaryAUROC()
        self.train_acc = BinaryAccuracy()
        self.train_recall = BinaryRecall()
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_recall = BinaryRecall()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_auroc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_recall = BinaryRecall()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone.

        Args:
            x: Float32 image tensor of shape (B, 3, H, W).

        Returns:
            Sigmoid probability tensor of shape (B, 1).
        """
        return torch.sigmoid(self.backbone(x))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        imgs, labels = batch
        probs = self(imgs).squeeze(1)
        loss = self.criterion(probs, labels)
        self.train_loss.update(loss)
        self.train_auroc.update(probs, labels.int())
        self.train_acc.update(probs, labels.int())
        self.train_recall.update(probs, labels.int())
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/loss", self.train_loss.compute(), prog_bar=True, on_epoch=True)
        self.log(
            "train/auroc", self.train_auroc.compute(), prog_bar=True, on_epoch=True
        )
        self.log("train/acc", self.train_acc.compute(), prog_bar=True, on_epoch=True)
        self.log(
            "train/recall", self.train_recall.compute(), prog_bar=True, on_epoch=True
        )
        self.train_loss.reset()
        self.train_auroc.reset()
        self.train_acc.reset()
        self.train_recall.reset()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        imgs, labels = batch
        probs = self(imgs).squeeze(1)
        loss = self.criterion(probs, labels)
        self.val_loss.update(loss)
        self.val_auroc.update(probs, labels.int())
        self.val_acc.update(probs, labels.int())
        self.val_recall.update(probs, labels.int())

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss.compute(), prog_bar=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True, on_epoch=True)
        self.log("val/acc", self.val_acc.compute(), prog_bar=True, on_epoch=True)
        self.log("val/recall", self.val_recall.compute(), prog_bar=True, on_epoch=True)
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_acc.reset()
        self.val_recall.reset()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Accumulate test metrics for one batch.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.
        """
        imgs, labels = batch
        probs = self(imgs).squeeze(1)
        loss = self.criterion(probs, labels)
        self.test_loss.update(loss)
        self.test_auroc.update(probs, labels.int())
        self.test_acc.update(probs, labels.int())
        self.test_recall.update(probs, labels.int())

    def on_test_epoch_end(self) -> None:
        self.log("test/loss", self.test_loss.compute(), prog_bar=True, on_epoch=True)
        self.log("test/auroc", self.test_auroc.compute(), prog_bar=True, on_epoch=True)
        self.log("test/acc", self.test_acc.compute(), prog_bar=True, on_epoch=True)
        self.log(
            "test/recall", self.test_recall.compute(), prog_bar=True, on_epoch=True
        )
        self.test_loss.reset()
        self.test_auroc.reset()
        self.test_acc.reset()
        self.test_recall.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        if self._lr_schedule == "constant":
            return optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=self._max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class PatchDataModule(LightningDataModule):
    """Lightning DataModule for the patch classification pipeline.

    Args:
        parquet_path: Path to ``patches_annotations.parquet``.
        patch_dir: Directory containing the JPEG patches.
        batch_size: Samples per batch (must be even for the balanced sampler).
        num_workers: DataLoader workers per split.
        seed: Random seed for splits and balanced sampler.
        sampling_mode: ``"undersample"`` or ``"oversample"``.
        split_mode: ``"slide"`` for slide-level splits; ``"random"`` for patch-level.
        aug_strength: ``"mild"`` or ``"strong"`` — passed to ``make_train_transform``.
    """

    def __init__(
        self,
        parquet_path: Path,
        patch_dir: Path,
        batch_size: int = 64,
        num_workers: int = 8,
        seed: int = 42,
        split_seed: int = 9,
        sampling_mode: Literal["undersample", "oversample"] = "undersample",
        split_mode: Literal["slide", "random"] = "slide",
        aug_strength: Literal["mild", "strong"] = "mild",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._parquet_path: Path = parquet_path
        self._patch_dir: Path = patch_dir
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        self._seed: int = seed
        self._split_seed: int = split_seed
        self._sampling_mode: Literal["undersample", "oversample"] = sampling_mode
        self._split_mode: Literal["slide", "random"] = split_mode
        self._aug_strength: Literal["mild", "strong"] = aug_strength
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_df is not None:
            return
        self.train_df, self.val_df, self.test_df = make_splits(
            self._parquet_path,
            self._patch_dir,
            self._split_seed,
            self._split_mode,
        )
        log.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(self.train_df),
            len(self.val_df),
            len(self.test_df),
        )

    def _loader_kwargs(self) -> dict:
        nw = self._num_workers
        kwargs: dict = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
        if nw > 0:
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self.train_df is not None, (
            "setup() must be called before train_dataloader()"
        )
        dataset = PatchDataset(self.train_df, make_train_transform(self._aug_strength))
        sampler = BalancedBatchSampler(
            dataset.labels,
            self._sampling_mode,
            self._batch_size,
            self._seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader:
        assert self.val_df is not None, "setup() must be called before val_dataloader()"
        dataset = PatchDataset(self.val_df, make_eval_transform())
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_df is not None, (
            "setup() must be called before test_dataloader()"
        )
        dataset = PatchDataset(self.test_df, make_eval_transform())
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def full_train_dataloader(self) -> DataLoader:
        """Full training set without balanced sampling, for post-training diagnostics."""
        assert self.train_df is not None, (
            "setup() must be called before full_train_dataloader()"
        )
        dataset = PatchDataset(self.train_df, make_eval_transform())
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )


# ---------------------------------------------------------------------------
# Experiment naming
# ---------------------------------------------------------------------------


def _next_exp_name(
    experiments_dir: Path,
    model: str,
    weights_tag: str,
    sampling: str,
) -> str:
    """Return the next available experiment name with a zero-padded 3-digit counter.

    Args:
        experiments_dir: Root directory containing all experiment subdirectories.
        model: Model name string (e.g. ``"resnet18"``).
        weights_tag: ``"pretrained"`` or ``"scratch"``.
        sampling: ``"undersample"`` or ``"oversample"``.

    Returns:
        Experiment name string such as ``"resnet18_scratch_undersample_002"``.
    """
    prefix = f"{model}_{weights_tag}_{sampling}_"
    max_n = 0
    if experiments_dir.exists():
        for d in experiments_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                try:
                    max_n = max(max_n, int(d.name[len(prefix) :]))
                except ValueError:
                    pass
    return f"{prefix}{max_n + 1:03d}"


# ---------------------------------------------------------------------------
# Post-training diagnostics
# ---------------------------------------------------------------------------


def _infer(
    model: PatchClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader.

    Args:
        model: Trained classifier (already moved to ``device``).
        loader: DataLoader returning ``(images, labels)`` batches.
        device: Device to run inference on.

    Returns:
        Tuple of (probs, labels) as float32 numpy arrays of shape (N,).
    """
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = model(imgs.to(device)).squeeze(1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _save_confusion_matrix(
    labels: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Save a confusion matrix PNG at decision threshold 0.5.

    Args:
        labels: Ground-truth binary labels.
        scores: Predicted probabilities in [0, 1].
        path: Output PNG path.
        title: Figure title.
    """
    preds = (scores >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-informative", "Informative"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Save a ROC curve PNG.

    Args:
        labels: Ground-truth binary labels.
        scores: Predicted probabilities in [0, 1].
        path: Output PNG path.
        title: Figure title (AUC is appended automatically).
    """
    auc_val = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    ax.set_title(f"{title} (AUC={auc_val:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute classification metrics at threshold 0.5.

    Args:
        labels: Ground-truth binary labels.
        scores: Predicted probabilities in [0, 1].

    Returns:
        Dict with keys ``auc``, ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    preds = (scores >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def _run_diagnostics(
    ckpt_path: Path,
    datamodule: PatchDataModule,
    exp_dir: Path,
    device: torch.device,
) -> dict:
    """Load the best checkpoint and compute diagnostics for all three splits.

    Saves confusion matrix and ROC curve PNGs to ``exp_dir/figures/`` and
    writes ``exp_dir/results.pkl``.

    Args:
        ckpt_path: Path to the best-auroc checkpoint.
        datamodule: Fitted datamodule (``setup()`` already called).
        exp_dir: Experiment directory.
        device: Device to run inference on.

    Returns:
        Nested dict ``{split: {logits, labels, scores, auc, accuracy, ...}}``.
    """
    model = PatchClassifier.load_from_checkpoint(ckpt_path, weights_only=False).to(
        device
    )

    loaders = {
        "train": datamodule.full_train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }
    figures_dir = exp_dir / "figures"
    results: dict = {}

    for split, loader in loaders.items():
        scores, labels = _infer(model, loader, device)
        metrics = _compute_metrics(labels, scores)

        _save_confusion_matrix(
            labels,
            scores,
            figures_dir / f"{split}_confusion.png",
            f"{split.capitalize()} — Confusion Matrix",
        )
        _save_roc_curve(
            labels,
            scores,
            figures_dir / f"{split}_roc.png",
            f"{split.capitalize()} — ROC",
        )

        results[split] = {
            "scores": scores,
            "labels": labels,
            **metrics,
        }
        log.info(
            "[%s] AUC=%.4f  Acc=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
            split,
            metrics["auc"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    parquet_path: Path,
    patch_dir: Path,
    output_dir: Path,
    seed: int = 42,
    split_seed: int = 42,
    pretrained: bool = False,
    sampling_mode: Literal["undersample", "oversample"] = "undersample",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    num_workers: int = 8,
    max_epochs: int = 50,
    gpu: int = 0,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    split_mode: Literal["slide", "random"] = "slide",
    architecture: Literal["resnet18", "efficientnet_b0"] = "resnet18",
    dropout_rate: float = 0.0,
    aug_strength: Literal["mild", "strong"] = "mild",
) -> None:
    """Train a binary patch classifier.

    Experiment outputs are saved to ``output_dir/experiments/{name}/``.
    The experiment name encodes the model, weight init, sampling strategy,
    and a unique 3-digit counter (e.g. ``resnet18_scratch_undersample_001``).

    Args:
        parquet_path: Path to ``patches_annotations.parquet``.
        patch_dir: Directory containing the JPEG patches.
        output_dir: Root output directory; ``experiments/`` is created inside.
        seed: Random seed for the batch sampler and model initialisation (training randomness).
        split_seed: Random seed for the train/val/test split. Kept separate from ``seed``
            so the data partition can be held fixed while varying training randomness.
        pretrained: Initialise from ImageNet weights (default: train from scratch).
        sampling_mode: ``"undersample"`` exhausts positives once per epoch;
            ``"oversample"`` exhausts negatives once per epoch.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Must be even (half drawn from each class per batch).
        num_workers: DataLoader workers per split.
        max_epochs: Number of training epochs.
        gpu: GPU index to use (0-based). Ignored if CUDA is unavailable.
        lr_schedule: ``"cosine"`` decays LR with cosine annealing; ``"constant"`` holds LR fixed.
        split_mode: ``"slide"`` for slide-level stratified splits; ``"random"`` for a
            raw patch-level split that ignores slide identity.
        architecture: Backbone architecture — ``"resnet18"`` or ``"efficientnet_b0"``.
        dropout_rate: Dropout probability before the classification head; ``0.0`` disables.
        aug_strength: Training augmentation intensity — ``"mild"`` or ``"strong"``.
    """
    seed_everything(seed, workers=True)

    weights_tag = "pretrained" if pretrained else "scratch"
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    exp_name = _next_exp_name(experiments_dir, architecture, weights_tag, sampling_mode)
    exp_dir = experiments_dir / exp_name
    (exp_dir / "figures").mkdir(parents=True)
    (exp_dir / "checkpoints").mkdir()
    log.info("Experiment: %s", exp_name)

    use_gpu = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if use_gpu else "cpu")
    log.info("Device: %s", device)

    config: dict = {
        "exp_name": exp_name,
        "parquet_path": str(parquet_path),
        "patch_dir": str(patch_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "split_seed": split_seed,
        "pretrained": pretrained,
        "sampling_mode": sampling_mode,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_epochs": max_epochs,
        "gpu": gpu,
        "lr_schedule": lr_schedule,
        "split_mode": split_mode,
        "architecture": architecture,
        "dropout_rate": dropout_rate,
        "aug_strength": aug_strength,
    }
    with open(exp_dir / "config.yml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False)

    datamodule = PatchDataModule(
        parquet_path=parquet_path,
        patch_dir=patch_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        split_seed=split_seed,
        sampling_mode=sampling_mode,
        split_mode=split_mode,
        aug_strength=aug_strength,
    )
    datamodule.setup()

    assert datamodule.train_df is not None
    assert datamodule.val_df is not None
    assert datamodule.test_df is not None
    splits_rows = [
        datamodule.train_df.assign(split="train"),
        datamodule.val_df.assign(split="val"),
        datamodule.test_df.assign(split="test"),
    ]
    pd.concat(splits_rows, ignore_index=True).to_parquet(
        exp_dir / "splits.parquet", index=False
    )

    model = PatchClassifier(
        pretrained=pretrained,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        lr_schedule=lr_schedule,
        architecture=architecture,
        dropout_rate=dropout_rate,
    )

    wandb_logger = WandbLogger(
        project="ROSE-relevance",
        name=exp_name,
        group=f"{architecture}_{weights_tag}_{sampling_mode}",
        save_dir=str(exp_dir),
    )
    wandb_logger.log_hyperparams(config)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="best-auroc",
        monitor="val/auroc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, LearningRateMonitor("epoch")],
        log_every_n_steps=10,
        accelerator="gpu" if use_gpu else "cpu",
        devices=[gpu] if use_gpu else 1,
    )
    trainer.fit(model, datamodule=datamodule)

    best_ckpt = Path(checkpoint_cb.best_model_path)
    log.info("Best checkpoint: %s", best_ckpt)

    model_for_test = PatchClassifier.load_from_checkpoint(
        str(best_ckpt), weights_only=False
    )
    trainer.test(model_for_test, datamodule=datamodule)

    results = _run_diagnostics(best_ckpt, datamodule, exp_dir, device)

    with open(exp_dir / "results.pkl", "wb") as fh:
        pickle.dump(results, fh)

    wandb.finish()
    log.info("Experiment complete. Outputs at %s", exp_dir)


if __name__ == "__main__":
    CLI(main)
