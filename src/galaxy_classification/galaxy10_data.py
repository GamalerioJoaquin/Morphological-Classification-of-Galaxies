"""Torch-free dataset validation and split utilities for Galaxy10 DECaLS."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


GALAXY10_FILENAME = "Galaxy10_DECals.h5"
GALAXY10_SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"
GALAXY10_SHAPE = (17_736, 256, 256, 3)
GALAXY10_CLASS_NAMES = (
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
)
GALAXY10_CLASS_COUNTS = (1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873)
SPLITS = ("train", "validation", "test")


class Galaxy10DataError(ValueError):
    """Raised when the cached dataset or split manifest is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_arrays(images: np.ndarray, labels: np.ndarray) -> None:
    if images.shape != GALAXY10_SHAPE:
        raise Galaxy10DataError(
            f"Expected image shape {GALAXY10_SHAPE}, received {images.shape}."
        )
    if labels.shape != (GALAXY10_SHAPE[0],):
        raise Galaxy10DataError(f"Unexpected label shape: {labels.shape}.")
    counts = np.bincount(labels.astype(np.int64), minlength=10)
    if tuple(counts) != GALAXY10_CLASS_COUNTS:
        raise Galaxy10DataError(
            f"Unexpected class counts: {tuple(int(value) for value in counts)}."
        )


def build_split_manifest(
    labels: np.ndarray,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create deterministic stratified 70/15/15 partitions by dataset row."""

    indices = np.arange(len(labels))
    train_indices, remaining_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        stratify=labels[remaining_indices],
        random_state=random_state,
    )
    split_by_index = np.empty(len(labels), dtype=object)
    split_by_index[train_indices] = "train"
    split_by_index[validation_indices] = "validation"
    split_by_index[test_indices] = "test"
    return pd.DataFrame(
        {
            "sample_index": indices,
            "object_id": [f"galaxy10-decals-{index:05d}" for index in indices],
            "label": labels.astype(np.int64),
            "class_name": [GALAXY10_CLASS_NAMES[int(label)] for label in labels],
            "split": split_by_index,
            "source": "Galaxy10 DECaLS via astroNN 1.1.0",
        }
    )


def validate_manifest(manifest: pd.DataFrame, number_of_samples: int) -> None:
    required = {"sample_index", "object_id", "label", "class_name", "split", "source"}
    missing = required.difference(manifest.columns)
    if missing:
        raise Galaxy10DataError(f"Split manifest is missing: {sorted(missing)}")
    if len(manifest) != number_of_samples:
        raise Galaxy10DataError(
            f"Manifest has {len(manifest)} rows; expected {number_of_samples}."
        )
    indices = pd.to_numeric(manifest["sample_index"], errors="raise").astype(int)
    if set(indices) != set(range(number_of_samples)):
        raise Galaxy10DataError("sample_index must cover every dataset row exactly once.")
    if manifest["object_id"].duplicated().any():
        raise Galaxy10DataError("object_id values must be unique.")
    if set(manifest["split"]) != set(SPLITS):
        raise Galaxy10DataError("Manifest must contain train, validation, and test.")
    labels = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    if not labels.between(0, 9).all():
        raise Galaxy10DataError("Labels must be integers from 0 through 9.")
