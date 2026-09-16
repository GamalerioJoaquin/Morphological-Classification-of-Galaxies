"""Validate the legacy course asset and add explicit local lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_COLUMNS = [
    "objID",
    "ra",
    "dec",
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
    "petroR90_r",
    "z",
    "Color",
    "elliptical",
    "spiral",
    "uncertain",
]
LABEL_COLUMNS = ["elliptical", "spiral", "uncertain"]
NUMERIC_COLUMNS = [column for column in SOURCE_COLUMNS if column != "objID"]
MEASUREMENT_COLUMNS = [
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
    "petroR90_r",
    "z",
    "Color",
]


class DataIntegrityError(ValueError):
    """Raised when an input violates the documented source contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the tracked source manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"asset", "sha256", "rows", "columns"}
    missing = required.difference(manifest)
    if missing:
        raise DataIntegrityError(
            f"Source manifest is missing required keys: {sorted(missing)}"
        )
    return manifest


def validate_source(frame: pd.DataFrame) -> None:
    """Validate schema, numeric fields, coordinates, and one-hot source labels."""

    if list(frame.columns) != SOURCE_COLUMNS:
        raise DataIntegrityError(
            "Unexpected source schema. "
            f"Expected {SOURCE_COLUMNS}, received {list(frame.columns)}."
        )

    if frame["objID"].isna().any():
        raise DataIntegrityError("Source contains missing legacy objID values.")

    non_numeric = [
        column
        for column in NUMERIC_COLUMNS
        if pd.to_numeric(frame[column], errors="coerce").isna().any()
    ]
    if non_numeric:
        raise DataIntegrityError(f"Non-numeric values found in: {non_numeric}")

    label_values = frame[LABEL_COLUMNS]
    if not label_values.isin([0, 1]).all().all():
        raise DataIntegrityError("Source label flags must contain only 0 or 1.")

    invalid_one_hot = int(label_values.sum(axis=1).ne(1).sum())
    if invalid_one_hot:
        raise DataIntegrityError(
            f"{invalid_one_hot} rows do not have exactly one source label."
        )

    if frame[["ra", "dec"]].isna().any().any():
        raise DataIntegrityError("Source coordinates must not be missing.")


def load_and_validate_source(path: Path) -> pd.DataFrame:
    """Read the course asset while preserving its damaged objID as text."""

    frame = pd.read_csv(path, dtype={"objID": "string"})
    validate_source(frame)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric)
    return frame


def derive_source_label(frame: pd.DataFrame) -> pd.Series:
    """Map one-hot source flags without reinterpreting uncertain as irregular."""

    validate_source(frame)
    return frame[LABEL_COLUMNS].idxmax(axis=1).rename("source_label")


def _coordinate_key(ra: float, dec: float) -> str:
    """Build a stable representation of an exact parsed coordinate pair."""

    return f"{float(ra):.17g}|{float(dec):.17g}"


def _coordinate_id(key: str) -> str:
    return "coord_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def audit_source(frame: pd.DataFrame) -> dict[str, Any]:
    """Return machine-readable evidence about identity and label integrity."""

    validate_source(frame)
    source_label = derive_source_label(frame)
    coordinates = ["ra", "dec"]

    coordinate_sizes = frame.groupby(coordinates, sort=False, dropna=False).size()
    coordinate_label_counts = (
        frame.assign(_source_label=source_label)
        .groupby(coordinates, sort=False, dropna=False)["_source_label"]
        .nunique()
    )
    positions_per_legacy_id = (
        frame.drop_duplicates(["objID", *coordinates])
        .groupby("objID", sort=False, dropna=False)
        .size()
    )

    feature_vectors_per_coordinate = (
        frame.drop_duplicates([*coordinates, *MEASUREMENT_COLUMNS])
        .groupby(coordinates, sort=False, dropna=False)
        .size()
    )

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "distinct_legacy_objids": int(frame["objID"].nunique(dropna=False)),
        "distinct_coordinate_groups": int(len(coordinate_sizes)),
        "repeated_coordinate_groups": int(coordinate_sizes.gt(1).sum()),
        "rows_in_repeated_coordinate_groups": int(
            coordinate_sizes[coordinate_sizes.gt(1)].sum()
        ),
        "coordinate_groups_with_different_measurements": int(
            feature_vectors_per_coordinate.gt(1).sum()
        ),
        "coordinate_groups_with_label_conflicts": int(
            coordinate_label_counts.gt(1).sum()
        ),
        "rows_in_coordinate_label_conflicts": int(
            coordinate_sizes[coordinate_label_counts.gt(1)].sum()
        ),
        "legacy_objid_collision_groups": int(positions_per_legacy_id.gt(1).sum()),
        "exact_duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "label_counts": {
            str(label): int(count)
            for label, count in source_label.value_counts().sort_index().items()
        },
    }


def prepare_dataset(frame: pd.DataFrame, source_sha256: str) -> pd.DataFrame:
    """Add lineage and integrity flags without deleting or learning from rows."""

    validate_source(frame)
    prepared = frame.copy().reset_index(drop=True)
    prepared = prepared.rename(columns={"objID": "legacy_objID", "Color": "legacy_color"})

    row_numbers = np.arange(1, len(prepared) + 1)
    prepared.insert(0, "source_row_number", row_numbers)
    prepared.insert(
        0,
        "source_row_id",
        [f"{source_sha256[:12]}:{number:06d}" for number in row_numbers],
    )

    coordinate_keys = [
        _coordinate_key(ra, dec)
        for ra, dec in zip(prepared["ra"], prepared["dec"])
    ]
    coordinate_lookup = {
        key: _coordinate_id(key) for key in dict.fromkeys(coordinate_keys)
    }
    prepared.insert(
        2, "coordinate_id", [coordinate_lookup[key] for key in coordinate_keys]
    )

    prepared["source_label"] = derive_source_label(frame).to_numpy()
    prepared["is_confident_morphology"] = prepared["source_label"].isin(
        ["elliptical", "spiral"]
    )

    coordinate_groups = prepared.groupby("coordinate_id", sort=False)
    prepared["coordinate_group_size"] = coordinate_groups["source_row_id"].transform(
        "size"
    )
    prepared["coordinate_label_count"] = coordinate_groups["source_label"].transform(
        "nunique"
    )
    prepared["coordinate_has_label_conflict"] = prepared[
        "coordinate_label_count"
    ].gt(1)

    unique_positions = prepared.drop_duplicates(["legacy_objID", "coordinate_id"])
    positions_per_legacy_id = unique_positions.groupby("legacy_objID").size()
    prepared["legacy_objid_position_count"] = prepared["legacy_objID"].map(
        positions_per_legacy_id
    )
    prepared["legacy_objid_collision"] = prepared[
        "legacy_objid_position_count"
    ].gt(1)

    prepared["color_r_minus_u"] = prepared["modelMag_r"] - prepared["modelMag_u"]
    prepared["legacy_color_delta"] = (
        prepared["legacy_color"] - prepared["color_r_minus_u"]
    )

    if len(prepared) != len(frame):
        raise DataIntegrityError("Preparation changed the number of source rows.")
    if not prepared["source_row_id"].is_unique:
        raise DataIntegrityError("Generated source_row_id values are not unique.")

    return prepared


def build_prepared_dataset(
    source_path: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Verify the immutable source, then write prepared data and an audit report."""

    manifest = load_manifest(manifest_path)
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != manifest["sha256"]:
        raise DataIntegrityError(
            "Source checksum mismatch: "
            f"expected {manifest['sha256']}, received {actual_sha256}."
        )

    frame = load_and_validate_source(source_path)
    if len(frame) != manifest["rows"] or frame.shape[1] != manifest["columns"]:
        raise DataIntegrityError(
            "Source shape does not match manifest: "
            f"expected {manifest['rows']}x{manifest['columns']}, "
            f"received {len(frame)}x{frame.shape[1]}."
        )

    report = audit_source(frame)
    report["source_sha256"] = actual_sha256
    report["identity_policy"] = manifest["identity_policy"]

    prepared = prepare_dataset(frame, actual_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
