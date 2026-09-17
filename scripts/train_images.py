"""Train the provisional PyTorch CNN from a verified image manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.image_pipeline import (  # noqa: E402
    ImageManifestError,
    SmallGalaxyCNN,
    TrainingConfig,
    load_image_manifest,
    make_data_loaders,
    run_epoch,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a provisional CNN; results are printed, not committed."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
    )
    try:
        manifest = load_image_manifest(args.manifest)
    except (FileNotFoundError, ImageManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    class_names = tuple(sorted(manifest["label"].unique()))
    if len(class_names) < 2:
        print("ERROR: The manifest must contain at least two classes.", file=sys.stderr)
        return 1

    seed_everything(config.random_state)
    loaders = make_data_loaders(manifest, class_names, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallGalaxyCNN(len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        train_metrics = run_epoch(
            model, loaders["train"], loss_function, device, optimizer
        )
        validation_metrics = run_epoch(
            model, loaders["validation"], loss_function, device
        )
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f}"
        )

    test_metrics = run_epoch(model, loaders["test"], loss_function, device)
    print(
        "Provisional test metrics (not portfolio evidence): "
        f"loss={test_metrics['loss']:.4f}, accuracy={test_metrics['accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
