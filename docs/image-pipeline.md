# Image-classification pipeline

## Current status

Galaxy10 DECaLS is the selected dataset. The repository does not commit its
2.54 GB HDF5 file, trained weights, or image metrics. Download, inspection, and
training code are implemented but must still be run and reviewed before any
performance result can be reported.

The historical Keras notebook experiment used too few images and an invalid
evaluation design. Its saved outputs have been removed and it is not used as
portfolio evidence. The code remains only as historical prototype material.

## Galaxy10 DECaLS

The official astroNN distribution contains 17,736 images with shape
`(256, 256, 3)` and ten morphology classes. Images originate from the DESI
Legacy Imaging Surveys and labels are based on Galaxy Zoo classifications. The
HDF5 checksum expected by both astroNN and this project is:

```text
19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571
```

Prepare the local cache and deterministic split manifest with:

```bash
python scripts/prepare_galaxy10.py
```

The preparation script downloads the official HDF5 file directly from the
astroNN Galaxy10 release URL and stores it under `~/.astroNN/datasets/` for
compatibility with astroNN's conventional cache layout. It does not import
astroNN, TensorFlow, or PyTorch. Its runtime dependencies are limited to the
standard library plus the project requirements used to read and validate the
dataset (`numpy`, `h5py`, `pandas`, and `scikit-learn`).

The notebook stack is also pinned as one compatible set. Avoid installing
Jupyter or Matplotlib piecemeal with Conda inside this environment: combining
Notebook 6 with Jupyter Server 2, or the pinned Matplotlib with a newer inline
backend, breaks notebook startup and inline figures.

### NVIDIA GPU installation

The normal Python package index can install a CPU-only PyTorch build on
Windows. NVIDIA Blackwell GPUs, including the RTX 5080, require a recent CUDA
wheel. After the base requirements, install the project's CUDA 12.8 pins:

```bash
python -m pip install --force-reinstall -r requirements-gpu.txt
python scripts/check_torch_gpu.py
```

Restart the Jupyter kernel after changing PyTorch. The training notebook now
fails immediately when CUDA is unavailable instead of silently training on
the CPU.

The script prints the image and label shapes, verifies the HDF5 checksum before
creating the training cache, verifies the known class counts, and writes a
stratified 70/15/15 manifest. It also writes an ignored contiguous NumPy cache
under `data/processed/`. This additional local copy is necessary because the
official gzip-compressed HDF5 chunks span 555 images and make shuffled,
image-by-image training severely CPU/I/O bound. The NumPy cache is
memory-mapped, so workers do not load a separate 3.5 GB copy into RAM.

The two clean, unexecuted notebooks are:

- `notebooks/04-galaxy10-inspection.ipynb`: structure, split balance,
  representative images, and channel statistics;
- `notebooks/05-galaxy10-pytorch-training.ipynb`: class-weighted PyTorch CNN,
  two-candidate training-only search, validation macro-F1 selection, and a
  single held-out test evaluation.

## Generic image-manifest contract

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

## Optional legacy SDSS recovery (not the selected training dataset)

The source CSV's `objID` values contain only about 15 significant digits, but
its `ra` and `dec` values retain enough precision to cross-match positions. A
live check of the first source row against the DR18 `PhotoPrimary` catalog
recovered exact object ID `1237651192424890568`, at approximately 0.000037
arcseconds from the stored position. That exact ID rounds to the damaged value
in the CSV.

`scripts/download_sdss_images.py` applies the same check to every selected
coordinate group. It:

1. verifies the checksum of the legacy CSV;
2. keeps one image per exact coordinate and excludes the 19 label-conflicting
   coordinate groups;
3. queries the nearest DR18 primary photometric object within one arcsecond in
   batches;
4. accepts a match only when its exact `objID` rounds back to the stored legacy
   value;
5. downloads a plain RGB JPEG cutout, verifies its dimensions and format, and
   records its SHA-256 checksum;
6. writes the full recovered ID, match distance, imaging field identifiers,
   five catalog magnitudes, source URL, and deterministic object-level split to
   `manifest.csv`.

Start with a small end-to-end sample:

```bash
python scripts/download_sdss_images.py --limit 100
```

Then download all confident elliptical and spiral targets:

```bash
python scripts/download_sdss_images.py
```

The default location is `data/images/sdss-dr18-jpeg/`, which is ignored by Git.
Existing valid JPEGs are reused, so rerunning the command resumes the image
stage. Four concurrent downloads are used by default; keep that conservative
setting unless the SDSS service explicitly permits a higher request rate.

The script deliberately does not accept unmatched or incompatible IDs. Its
`catalog-matches.csv`, `rejected-id-matches.csv`, and `failures.json` outputs
make coverage and service failures auditable rather than silently substituting
objects.

## RGB cutouts versus five-band images

The SkyServer cutout endpoint returns a display-ready color JPEG. It is the
practical first dataset for an RGB classifier, but it is not five independent
scientific channels and JPEG values should not be interpreted as calibrated
flux.

SDSS also provides calibrated FITS imaging in the five `u`, `g`, `r`, `i`, and
`z` filters. These are distributed as full field frames identified by
`run/rerun/camcol/field`, not as five small cutouts from the JPEG service. The
downloader already records those four identifiers in the manifest. A later
FITS phase should therefore download each unique field/band once, use its WCS
to crop every contained target, align the bands, preserve calibration and mask
metadata, and save a five-channel array (for example FITS or compressed NumPy),
instead of redownloading a complete field for every galaxy.

This distinction follows the official access paths:

- SDSS DR18 imaging overview: https://www.sdss.org/dr18/imaging/
- SkyServer cutout and search API: https://skyserver.sdss.org/dr12/en/help/docs/api.aspx
- SDSS frame-file access pattern: https://sdss.org/dr18/data_access/get_data/
- Astroquery warning that image requests retrieve full frames that must be
  cropped locally: https://astroquery.readthedocs.io/en/stable/api/astroquery.sdss.SDSSClass.html

## Provisional model

`Galaxy10CNN` is a moderate PyTorch baseline with four convolutional blocks,
batch normalization, adaptive pooling, a hidden classifier layer, dropout, and
ten raw class logits for `CrossEntropyLoss`. Training uses class weights
derived only from the training partition and rotation/flip augmentation only
for that partition. Six bounded candidates compare two learning rates and
dropout values 0.3, 0.5, and 0.7, with early stopping on validation macro-F1.

The selected Galaxy10 experiment is run interactively from the training
notebook after preparation:

```bash
jupyter lab notebooks/05-galaxy10-pytorch-training.ipynb
```

Printed metrics are explicitly provisional. Promoting an image result to the
portfolio additionally requires documented dataset provenance and license,
adequate class support, object-level split isolation, training-only model
selection, class-wise metrics, uncertainty, and error analysis.
