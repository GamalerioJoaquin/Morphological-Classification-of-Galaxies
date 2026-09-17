# Image-classification pipeline

## Current status

The image classifier is a **dataset-blocked prototype**, not a completed
experiment. The repository does not currently contain a traceable image
dataset, so it reports no image-classification performance and stores no
trained weights.

The historical Keras notebook experiment used too few images and an invalid
evaluation design. Its saved outputs have been removed and it is not used as
portfolio evidence. The code remains only as historical prototype material.

## Dataset contract

Before training, a versioned CSV manifest must provide one row per image with:

| Column | Meaning |
|---|---|
| `object_id` | Stable identity used to prevent cross-split leakage |
| `label` | Source morphology label |
| `split` | `train`, `validation`, or `test` |
| `image_path` | Path relative to the manifest |
| `sha256` | Image checksum |
| `source` | Dataset/survey provenance |

Additional provenance columns such as sky coordinates, survey release, bands,
cutout size, label source, and license should be included when the dataset is
selected. The loader rejects missing files, checksum mismatches, duplicate
paths, incomplete splits, and any object that appears across multiple splits.

## Provisional model

`SmallGalaxyCNN` is a compact RGB PyTorch baseline with three convolutional
blocks, batch normalization, adaptive pooling, and raw class logits for
`CrossEntropyLoss`. Its input size and number of classes are configurable.

Once an appropriate dataset is selected, a local training run can use:

```bash
python scripts/train_images.py path/to/manifest.csv
```

Printed metrics are explicitly provisional. Promoting an image result to the
portfolio additionally requires documented dataset provenance and license,
adequate class support, object-level split isolation, training-only model
selection, class-wise metrics, uncertainty, and error analysis.
