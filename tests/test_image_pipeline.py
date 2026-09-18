from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from galaxy_classification.image_pipeline import (
    ImageManifestError,
    ManifestImageDataset,
    SmallGalaxyCNN,
    load_image_manifest,
    sha256_file,
)


def write_manifest(tmp_path: Path) -> Path:
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        image_path = tmp_path / f"galaxy-{index}.png"
        Image.new("RGB", (24, 24), color=(20 * index, 40, 80)).save(image_path)
        rows.append(
            {
                "object_id": f"object-{index}",
                "label": "elliptical" if index % 2 == 0 else "spiral",
                "split": split,
                "image_path": image_path.name,
                "sha256": sha256_file(image_path),
                "source": "synthetic-test-fixture",
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_manifest_verifies_files_and_checksums(tmp_path: Path) -> None:
    manifest = load_image_manifest(write_manifest(tmp_path))

    assert len(manifest) == 3
    assert set(manifest["split"]) == {"train", "validation", "test"}


def test_manifest_rejects_object_leakage(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = pd.read_csv(manifest_path)
    manifest.loc[1, "object_id"] = manifest.loc[0, "object_id"]
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ImageManifestError, match="cross dataset splits"):
        load_image_manifest(manifest_path)


def test_manifest_rejects_conflicting_labels_for_one_object(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = pd.read_csv(manifest_path)
    duplicate = manifest.iloc[[0]].copy()
    duplicate["image_path"] = "second-view.png"
    Image.new("RGB", (24, 24), color=(1, 2, 3)).save(tmp_path / "second-view.png")
    duplicate["sha256"] = sha256_file(tmp_path / "second-view.png")
    duplicate["label"] = "spiral"
    pd.concat([manifest, duplicate], ignore_index=True).to_csv(
        manifest_path, index=False
    )

    with pytest.raises(ImageManifestError, match="one consistent label"):
        load_image_manifest(manifest_path)


def test_dataset_returns_normalized_rgb_tensor(tmp_path: Path) -> None:
    manifest = load_image_manifest(write_manifest(tmp_path))
    dataset = ManifestImageDataset(
        manifest,
        split="train",
        class_names=("elliptical", "spiral"),
        image_size=32,
    )

    image, target = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert target.dtype == torch.long


def test_cnn_outputs_one_logit_per_class() -> None:
    model = SmallGalaxyCNN(number_of_classes=3)

    logits = model(torch.zeros(2, 3, 64, 64))

    assert logits.shape == (2, 3)
    assert not torch.allclose(logits.softmax(dim=1), logits)
