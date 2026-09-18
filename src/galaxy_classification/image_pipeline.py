"""Dataset contract and small PyTorch baseline for galaxy images.

This module intentionally contains no portfolio metrics. A result is only valid
after a traceable image dataset has been added and evaluated on an object-level
holdout defined by a versioned manifest.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


REQUIRED_MANIFEST_COLUMNS = (
    "object_id",
    "label",
    "split",
    "image_path",
    "sha256",
    "source",
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


class ImageManifestError(ValueError):
    """Raised when an image manifest violates the dataset contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_manifest(
    manifest_path: Path,
    *,
    verify_files: bool = True,
) -> pd.DataFrame:
    """Load and validate a versioned, object-level image manifest."""

    manifest_path = manifest_path.resolve()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED_MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        raise ImageManifestError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ImageManifestError("Manifest contains no images.")

    for column in REQUIRED_MANIFEST_COLUMNS:
        if manifest[column].str.strip().eq("").any():
            raise ImageManifestError(f"Manifest column {column!r} contains blanks.")

    unexpected_splits = set(manifest["split"]).difference(ALLOWED_SPLITS)
    if unexpected_splits:
        raise ImageManifestError(
            f"Manifest contains unsupported splits: {sorted(unexpected_splits)}"
        )
    missing_splits = ALLOWED_SPLITS.difference(manifest["split"])
    if missing_splits:
        raise ImageManifestError(
            f"Manifest must contain train, validation, and test rows; missing: "
            f"{sorted(missing_splits)}"
        )

    split_counts = manifest.groupby("object_id")["split"].nunique()
    leaking_objects = split_counts[split_counts.gt(1)].index.tolist()
    if leaking_objects:
        preview = leaking_objects[:5]
        raise ImageManifestError(
            "Objects may not cross dataset splits; examples: " f"{preview}"
        )

    label_counts = manifest.groupby("object_id")["label"].nunique()
    conflicting_labels = label_counts[label_counts.gt(1)].index.tolist()
    if conflicting_labels:
        preview = conflicting_labels[:5]
        raise ImageManifestError(
            "Each object must have one consistent label; examples: " f"{preview}"
        )

    duplicate_paths = manifest["image_path"].duplicated(keep=False)
    if duplicate_paths.any():
        raise ImageManifestError("Each image_path must appear exactly once.")

    resolved_paths = [
        (manifest_path.parent / relative_path).resolve()
        for relative_path in manifest["image_path"]
    ]
    manifest = manifest.copy()
    manifest["resolved_path"] = [str(path) for path in resolved_paths]

    if verify_files:
        for row, image_path in zip(manifest.itertuples(index=False), resolved_paths):
            if not image_path.is_file():
                raise ImageManifestError(f"Image does not exist: {image_path}")
            actual_checksum = sha256_file(image_path)
            if actual_checksum.lower() != row.sha256.lower():
                raise ImageManifestError(
                    f"Checksum mismatch for {image_path.name}: expected "
                    f"{row.sha256}, received {actual_checksum}."
                )
    return manifest


class ManifestImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load RGB images from one declared manifest split."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        class_names: Iterable[str],
        image_size: int = 128,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.rows = manifest.loc[manifest["split"].eq(split)].reset_index(drop=True)
        if self.rows.empty:
            raise ImageManifestError(f"Split {split!r} contains no images.")
        self.class_to_index = {
            label: index for index, label in enumerate(tuple(class_names))
        }
        unknown = set(self.rows["label"]).difference(self.class_to_index)
        if unknown:
            raise ImageManifestError(f"Unknown labels in {split}: {sorted(unknown)}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        with Image.open(row["resolved_path"]) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1)
        target = torch.tensor(self.class_to_index[row["label"]], dtype=torch.long)
        return tensor, target


class SmallGalaxyCNN(nn.Module):
    """Compact baseline that accepts RGB images of any practical size."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        if number_of_classes < 2:
            raise ValueError("At least two classes are required.")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 32
    image_size: int = 128
    random_state: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data_loaders(
    manifest: pd.DataFrame,
    class_names: Iterable[str],
    config: TrainingConfig,
) -> dict[str, DataLoader]:
    """Create loaders without mixing the three manifest-defined splits."""

    class_names = tuple(class_names)
    generator = torch.Generator().manual_seed(config.random_state)
    return {
        split: DataLoader(
            ManifestImageDataset(
                manifest,
                split=split,
                class_names=class_names,
                image_size=config.image_size,
            ),
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split in ("train", "validation", "test")
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one train or evaluation epoch and return unpersisted metrics."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += len(targets)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
