# Morphological Classification of Galaxies

#### Work in Progress.

Welcome to the Morphological Classification of Galaxies project repository. In this repository, I am sharing a data science project that I completed last year. The primary objective of this project is to develop classifiers for galaxy classification. The methodology involves utilizing classical machine learning models such as decision trees and neural networks for image analysis. Additionally, the project explores unsupervised learning techniques.

## Data Source

The dataset used in this project is sourced from the Sloan Digital Sky Survey (SDSS). The SDSS is a significant multi-spectral imaging and spectroscopic redshift survey conducted using a dedicated 2.5-m wide-angle optical telescope located at the Apache Point Observatory in New Mexico, United States.

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

## License

The repository's original code is available under the [MIT License](LICENSE).
SDSS data and imagery remain subject to their own attribution and usage terms.







![Sample Image (5 channels)](/assets/img/galaxias.jpeg)
