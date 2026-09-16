# Morphological Classification of Galaxies

#### Work in Progress.

Welcome to the Morphological Classification of Galaxies project repository. In this repository, I am sharing a data science project that I completed last year. The primary objective of this project is to develop classifiers for galaxy classification. The methodology involves utilizing classical machine learning models such as decision trees and neural networks for image analysis. Additionally, the project explores unsupervised learning techniques.

## Data Source

The local dataset is a legacy course asset described as being derived from the
Sloan Digital Sky Survey (SDSS). Its original SDSS release and extraction query
are unavailable, so the project documents and verifies the supplied file
without claiming reproducible upstream provenance.

## Current State

As of the latest update, the repository now includes three jupyter notebooks. The initial notebook offers a comprehensive exploration of the dataset, encompassing data curation and visualization. Additionally, there is an unsupervised learning analysis focused on the dataset's tabular aspects, excluding image-related components.

Also, a new addition to the repository is the supervised learning notebook. This notebook features the implementation of a neural network using Keras, along with various strategies employing random forests. It provides insights into different supervised learning techniques applied to the morphological classification of galaxies.

Please note that this project is a work in progress, and updates will be provided as new developments unfold. Your feedback and contributions are highly welcome. Thank you for your interest in the Morphological Classification of Galaxies project!

## Environment setup

The project is tested with Python 3.10. The original `requirements.txt` was a
platform-specific Conda export and could not be installed with `pip`; it has
been replaced by portable direct dependencies.

Using Conda or Miniforge:

```bash
conda env create -f environment.yml
conda activate galaxy-classification
python scripts/validate_environment.py
```

For an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
python scripts/validate_environment.py
```

The current notebooks are historical artifacts and are not yet guaranteed to
execute cleanly from top to bottom. Environment validation confirms dependency
availability; notebook and data-pipeline corrections are tracked separately.

## Data integrity workflow

`galaxias_1.csv` is retained as an immutable legacy course asset. Its stored
`objID` values have lost precision and must not be used as unique identifiers.
The replacement workflow preserves every source row, adds deterministic lineage
and coordinate-group identifiers, and keeps `uncertain` distinct from any
irregular morphology class.

```bash
python scripts/audit_data.py
python scripts/build_dataset.py
pytest
```

The generated prepared dataset is intentionally not committed. See the
[data card](docs/data-card.md) for provenance, identity, label, and
transformation policies.

## Reproducible tabular baseline

The primary supervised task compares confident elliptical and spiral labels.
Uncertain rows and coordinate groups with conflicting labels are excluded from
model fitting, while repeated coordinate groups are kept entirely within one
partition to prevent train/test leakage.

```bash
python scripts/train_tabular.py
```

Model selection uses group-aware cross-validation on the development set. The
held-out test partition is used once after selection. Preprocessing, including
quantile clipping and median imputation, is fitted inside each training fold.
See the [verified results](docs/results.md) for the complete evaluation and
limitations.

## License

The repository's original code is available under the [MIT License](LICENSE).
SDSS data and imagery remain subject to their own attribution and usage terms.







![Sample Image (5 channels)](/assets/img/galaxias.jpeg)
