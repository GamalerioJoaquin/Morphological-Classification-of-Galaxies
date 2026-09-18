from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from galaxy_classification.galaxy10 import (
    Galaxy10CNN,
    Galaxy10DataError,
    Galaxy10H5Dataset,
    Galaxy10NpyDataset,
    augment_training_batch,
    build_split_manifest,
    validate_manifest,
)


def balanced_labels() -> np.ndarray:
    return np.repeat(np.arange(10), 20)


def test_split_manifest_is_reproducible_stratified_and_disjoint() -> None:
    labels = balanced_labels()
    first = build_split_manifest(labels)
    second = build_split_manifest(labels)

    pd.testing.assert_frame_equal(first, second)
    assert first["object_id"].is_unique
    split_counts = first.groupby(["split", "label"]).size()
    for split, expected in (("train", 14), ("validation", 3), ("test", 3)):
        assert split_counts.loc[split].eq(expected).all()
    validate_manifest(first, len(labels))


def test_manifest_rejects_duplicate_objects() -> None:
    manifest = build_split_manifest(balanced_labels())
    manifest.loc[1, "object_id"] = manifest.loc[0, "object_id"]

    with pytest.raises(Galaxy10DataError, match="unique"):
        validate_manifest(manifest, len(manifest))


def test_hdf5_dataset_converts_nhwc_uint8_to_nchw_float(tmp_path: Path) -> None:
    h5_path = tmp_path / "sample.h5"
    images = np.full((6, 8, 8, 3), 255, dtype=np.uint8)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("images", data=images)
    manifest = pd.DataFrame(
        {
            "sample_index": range(6),
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    )
    dataset = Galaxy10H5Dataset(h5_path, manifest, "validation")

    image, label = dataset[0]

    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert image.min() == image.max() == 1
    assert label.dtype == torch.long


def test_cnn_returns_ten_raw_logits() -> None:
    model = Galaxy10CNN(base_channels=4)

    logits = model(torch.zeros(2, 3, 64, 64))

    assert logits.shape == (2, 10)


def test_batch_augmentation_preserves_nchw_shape_and_dtype() -> None:
    inputs = torch.rand(4, 3, 8, 8)

    augmented = augment_training_batch(inputs)

    assert augmented.shape == inputs.shape
    assert augmented.dtype == inputs.dtype


def test_numpy_dataset_reads_contiguous_training_cache(tmp_path: Path) -> None:
    array_path = tmp_path / "images.npy"
    images = np.full((6, 8, 8, 3), 128, dtype=np.uint8)
    np.save(array_path, images, allow_pickle=False)
    manifest = pd.DataFrame(
        {
            "sample_index": range(6),
            "label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    )
    dataset = Galaxy10NpyDataset(array_path, manifest, "test")

    image, label = dataset[0]

    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert image.mean() == pytest.approx(128 / 255)
    assert label.item() == 0
