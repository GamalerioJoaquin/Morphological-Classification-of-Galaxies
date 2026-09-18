"""Leakage-safe PyTorch utilities for Galaxy10 DECaLS."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


from .galaxy10_data import (
    GALAXY10_CLASS_NAMES,
    GALAXY10_FILENAME,
    GALAXY10_SHAPE,
    SPLITS,
    Galaxy10DataError,
    build_split_manifest,
    sha256_file,
    validate_arrays,
    validate_manifest,
)


class Galaxy10H5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read a declared split lazily from the cached Galaxy10 HDF5 file."""

    def __init__(
        self,
        h5_path: Path,
        manifest: pd.DataFrame,
        split: str,
        *,
        augment: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.h5_path = Path(h5_path)
        rows = manifest.loc[manifest["split"].eq(split)].copy()
        if rows.empty:
            raise Galaxy10DataError(f"Split {split!r} is empty.")
        self.indices = rows["sample_index"].astype(int).to_numpy()
        self.labels = rows["label"].astype(int).to_numpy()
        self.augment = augment
        self._file: h5py.File | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _images(self) -> h5py.Dataset:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file["images"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(self._images()[self.indices[item]], dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        if self.augment:
            image = augment_train_image(image)
        target = torch.tensor(self.labels[item], dtype=torch.long)
        return image, target


class Galaxy10NpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read individual images efficiently from a contiguous NumPy array."""

    def __init__(
        self,
        array_path: Path,
        manifest: pd.DataFrame,
        split: str,
        *,
        augment: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        rows = manifest.loc[manifest["split"].eq(split)].copy()
        if rows.empty:
            raise Galaxy10DataError(f"Split {split!r} is empty.")
        self.array_path = Path(array_path)
        self.indices = rows["sample_index"].astype(int).to_numpy()
        self.labels = rows["label"].astype(int).to_numpy()
        self.augment = augment
        self._images: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def _array(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.array_path, mmap_mode="r")
            if self._images.ndim != 4 or self._images.shape[-1] != 3:
                raise Galaxy10DataError(
                    f"Unexpected NumPy image shape: {self._images.shape}."
                )
            if int(self.indices.max()) >= len(self._images):
                raise Galaxy10DataError("Manifest index exceeds the image array.")
        return self._images

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self._array()[self.indices[item]], dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).div_(255.0)
        if self.augment:
            image = augment_train_image(image)
        target = torch.tensor(self.labels[item], dtype=torch.long)
        return image, target


class Galaxy10CNN(nn.Module):
    """Moderate four-block CNN for ten-class morphology classification."""

    def __init__(self, base_channels: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], base_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, len(GALAXY10_CLASS_NAMES)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass(frozen=True)
class Galaxy10TrainingConfig:
    learning_rate: float = 1e-3
    base_channels: int = 48
    dropout: float = 0.2
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 10
    random_state: int = 42


def augment_train_image(image: torch.Tensor) -> torch.Tensor:
    """Apply inexpensive orientation-preserving training augmentation."""
    image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(1, 2))
    if bool(torch.randint(0, 2, ()).item()):
        image = torch.flip(image, dims=(1,))
    if bool(torch.randint(0, 2, ()).item()):
        image = torch.flip(image, dims=(2,))
    return image


def augment_training_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Apply per-image right-angle rotations and flips on the training device."""
    if inputs.ndim != 4:
        raise ValueError(f"Expected NCHW batch, received shape {tuple(inputs.shape)}.")

    rotations = torch.randint(0, 4, (len(inputs),), device=inputs.device)
    augmented = torch.empty_like(inputs)
    for turns in range(4):
        mask = rotations.eq(turns)
        if mask.any():
            augmented[mask] = torch.rot90(inputs[mask], turns, dims=(2, 3))

    horizontal_flips = torch.rand(len(inputs), device=inputs.device).lt(0.5)
    if horizontal_flips.any():
        augmented[horizontal_flips] = torch.flip(
            augmented[horizontal_flips], dims=(3,)
        )
    vertical_flips = torch.rand(len(inputs), device=inputs.device).lt(0.5)
    if vertical_flips.any():
        augmented[vertical_flips] = torch.flip(augmented[vertical_flips], dims=(2,))
    return augmented


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    array_path: Path,
    manifest: pd.DataFrame,
    config: Galaxy10TrainingConfig,
    *,
    number_of_workers: int = 0,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(config.random_state)
    return {
        split: DataLoader(
            Galaxy10NpyDataset(
                array_path,
                manifest,
                split,
                augment=False,
            ),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=number_of_workers,
            generator=generator if split == "train" else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=number_of_workers > 0,
        )
        for split in SPLITS
    }


def class_weights(manifest: pd.DataFrame) -> torch.Tensor:
    train_labels = manifest.loc[manifest["split"].eq("train"), "label"].astype(int)
    counts = np.bincount(train_labels, minlength=10)
    weights = len(train_labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            inputs = augment_training_batch(inputs)
            optimizer.zero_grad()
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.bfloat16 if use_amp else None,
        ):
            logits = model(inputs)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }
