# Verified Results

## Status

The tabular elliptical-versus-spiral baseline is the first result regenerated
under the repaired workflow. Historical notebook metrics are not used as
portfolio evidence because they relied on damaged identity, globally fitted
preprocessing, or resampling before the holdout split.

No image-classification result is currently reported. The former notebook CNN
run was an unsuccessful, non-reproducible prototype and its saved output is not
portfolio evidence. The replacement PyTorch pipeline remains blocked on the
selection of a traceable image dataset; see [image-pipeline.md](image-pipeline.md).

## Evaluation design

- Source: checksum-verified legacy course asset.
- Target: confident `elliptical` versus `spiral` source labels.
- Excluded: 53,799 uncertain rows and 22 confident rows belonging to coordinate
  groups with conflicting labels.
- Eligible sample: 38,281 rows in 34,813 coordinate groups.
- Holdout: 7,672 rows in 6,963 coordinate groups (20%).
- Development set: 30,609 rows in 27,850 coordinate groups (80%).
- Model selection: four-fold stratified group cross-validation on development
  data, optimizing macro F1.
- Leakage control: no coordinate group appears in both partitions. The nearest
  test-to-development coordinate separation is 5.52 arcseconds; none are within
  5 arcseconds and three are within 10 arcseconds.
- Preprocessing: invalid magnitude artifacts (`abs(value) > 100`) become
  missing, followed by 1st/99th percentile clipping, median imputation, and
  scaling where applicable. Learned steps are fitted inside the relevant
  training fold.

The evaluated features are r-band magnitude, four adjacent-band color
differences, Petrosian radius, and redshift. Identifiers, row numbers, targets,
the damaged `objID`, and assigned labels are not model inputs.

## Model selection

| Model | Development CV macro F1 | Notes |
|---|---:|---|
| Prior dummy | 0.439 ± 0.002 | Predicts the majority class |
| Balanced logistic regression | 0.770 ± 0.014 | Linear baseline |
| Balanced random forest | **0.890 ± 0.005** | Selected by development CV |

The selected forest uses 200 trees, unlimited depth, and a minimum of five
samples per leaf. The small search did not inspect the held-out test set.

## Held-out test results

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Prior dummy | 0.790 | 0.500 | 0.441 |
| Balanced logistic regression | 0.826 | 0.856 | 0.783 |
| Balanced random forest | **0.933** | **0.921** | **0.903** |

The group-bootstrap 95% interval for the selected model's macro F1 is
0.895–0.911.

Selected-model class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Elliptical | 0.802 | 0.902 | 0.849 | 1,614 |
| Spiral | 0.973 | 0.941 | 0.957 | 6,058 |

![Held-out test confusion matrix](../reports/figures/tabular_confusion_matrix.png)

The complete machine-readable result is available in
`reports/results/tabular_metrics.json` and can be regenerated with:

```bash
python scripts/train_tabular.py
```

## Limitations

- The upstream SDSS query and label-generation process remain unavailable.
- Coordinate grouping prevents exact-position leakage but cannot prove that all
  repeated physical objects have been identified.
- Metrics are row-level, so coordinate groups with multiple observations have
  greater weight.
- The result covers only confident elliptical/spiral labels and says nothing
  about irregular galaxies.
- The test set comes from the same legacy course asset; there is no external
  survey or independently labeled validation set.
- Feature clipping is a pragmatic training-only robustness step, not a
  scientifically validated replacement for missing-value flags.

## Unsupervised analysis

The repaired unsupervised workflow operates on 83,806 exact-coordinate groups.
Repeated measurements at the same coordinates are aggregated by their median,
so frequently repeated positions do not receive extra weight. Input features
are the same seven photometric measurements used by the supervised baseline.
Identifiers, row indices, source labels, and cluster assignments are excluded.
Corrupt magnitude artifacts with absolute values above 100 become missing
before coordinate aggregation and are imputed after aggregation.

### Cluster selection

The number of clusters is selected without using morphology labels. Each
candidate is fitted with three seeds and 20 K-means initializations per seed.
Selection maximizes mean silhouette among candidates with mean pairwise
stability ARI of at least 0.90.

| Clusters | Silhouette | Stability ARI |
|---:|---:|---:|
| 2 | 0.316 | 0.999 |
| 3 | **0.319** | **0.997** |
| 4 | 0.222 | 0.978 |
| 5 | 0.224 | 0.983 |
| 6 | 0.215 | 0.961 |

`k=3` is selected, although its silhouette advantage over `k=2` is small. This
supports using three groups as a compact exploratory segmentation, not as proof
that the data contain three physical morphology classes.

![Cluster selection](../reports/figures/cluster_selection.png)

### Feature-only projections

PCA is fitted to preprocessed features before cluster IDs are attached. The
first component explains 51.5% of variance and the second 28.5%, for 80.0%
combined. Seeded t-SNE is fitted independently to a fixed 5,000-coordinate
sample, also without cluster or source labels as inputs.

![PCA clusters](../reports/figures/pca_clusters.png)

![t-SNE clusters](../reports/figures/tsne_clusters.png)

The three regions primarily describe photometric gradients:

- Cluster 0: brighter, larger, redder, and lower-redshift coordinate groups.
- Cluster 1: intermediate brightness, red colors, smaller radius, and higher
  median redshift.
- Cluster 2: fainter, bluer coordinate groups with smaller median radius.

Cluster numbers are arbitrary and do not imply an ordering.

### External morphology comparison

After clustering was complete, the assignments were compared with the 34,813
coordinate groups carrying consistent elliptical or spiral labels:

- adjusted Rand index: 0.040;
- normalized mutual information: 0.098;
- homogeneity: 0.152;
- completeness: 0.072.

This weak agreement shows that the clusters do not recover the supervised
elliptical/spiral target. Source labels were not used for cluster selection or
embedding, and `uncertain` is not interpreted as an irregular morphology class.

The complete result is stored in
`reports/results/unsupervised_metrics.json` and can be regenerated with:

```bash
python scripts/run_unsupervised.py
```
