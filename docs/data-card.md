# Data Card

## Source status

`galaxias_1.csv` is a legacy course asset derived from Sloan Digital Sky Survey
(SDSS) data. The original SDSS release, query, label-generation process, and
download date are unavailable. The file is therefore treated as an immutable
local source artifact, not as a reproducible extract from SDSS.

The expected source contract is recorded in
`data/source-manifest.json`, including its SHA-256 checksum. A checksum mismatch
stops the preparation pipeline instead of silently processing a different
file.

## Identity policy

The source `objID` values are precision-damaged scientific-notation strings.
They are not valid unique SDSS object identifiers and are renamed
`legacy_objID` in prepared data.

The local pipeline uses two explicit identifiers:

- `source_row_id`: the source checksum prefix plus the one-based data-row
  number. This provides deterministic lineage back to the immutable course
  asset.
- `coordinate_id`: a stable hash of the exact numeric right ascension and
  declination pair. This identifies groups that must remain together in future
  train/validation/test splits.

Coordinates are a local grouping key, not a claim that the original SDSS
identifier has been recovered. Repeated coordinates are preserved because
6,190 repeated coordinate groups contain different feature vectors. The
pipeline reports and flags them rather than choosing a row without evidence.

## Labels

The source has three mutually exclusive flags:

| Source flag | Prepared `source_label` | Interpretation |
|---|---|---|
| `elliptical` | `elliptical` | Confident morphology class |
| `spiral` | `spiral` | Confident morphology class |
| `uncertain` | `uncertain` | Uncertain source classification; not irregular |

The primary future morphology model will use elliptical and spiral rows. Rows
marked `uncertain` are retained for separate analysis and are not relabeled as
irregular. The prepared Boolean `is_confident_morphology` makes this distinction
explicit.

Nineteen exact coordinate groups contain more than one source label. These
groups are flagged with `coordinate_has_label_conflict`; they should be
quarantined from supervised evaluation until their meaning can be resolved.

## Features and transformations

The integrity pipeline preserves source measurements and adds metadata. It
does not scale, impute, remove outliers, or learn any transformation from the
complete dataset. Those operations belong inside future training-only model
pipelines.

The historical `Color` column is renamed `legacy_color`. A transparent
`color_r_minus_u` feature is calculated as `modelMag_r - modelMag_u`, together
with `legacy_color_delta`, so disagreements remain observable rather than being
silently overwritten.

Extreme and sentinel-like values remain present. Their scientific meaning
cannot be established from the course asset alone, so any later replacement
policy must be explicit, tested, and fitted without using evaluation data.

For modeling, absolute magnitude values above 100 are treated as invalid rather
than clipped as meaningful observations. This fixed guard captures `-9999` and
multi-thousand source artifacts while remaining far outside the plausible
range of the supplied sample. The resulting missing feature values are imputed
inside training folds; the immutable source and prepared lineage dataset remain
unchanged.

## Reproducible commands

Audit the immutable source without writing outputs:

```bash
python scripts/audit_data.py
```

Build the prepared dataset and machine-readable report:

```bash
python scripts/build_dataset.py
```

Generated files are written to `data/processed/` and `reports/generated/` and
are intentionally ignored by Git. They can always be regenerated from the
tracked source asset, manifest, and code.

## Known limitations

- Upstream SDSS release, query, and labeling provenance are unavailable.
- Full-precision SDSS `objID` values cannot be recovered from this file.
- Coordinate equality is used only as a local grouping rule.
- The `uncertain` flag has no more detailed definition in the supplied asset.
- Selection effects and label quality cannot be audited against the original
  catalog.
