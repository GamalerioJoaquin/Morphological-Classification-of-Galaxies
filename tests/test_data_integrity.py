from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from galaxy_classification.data_integrity import (
    SOURCE_COLUMNS,
    DataIntegrityError,
    audit_source,
    build_prepared_dataset,
    load_and_validate_source,
    prepare_dataset,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = (
    "47ee7559205b221ca5d7ff4407b6e4839a5aa693e9a14269a0a3657c0023f298"
)


def sample_frame() -> pd.DataFrame:
    rows = [
        ["damaged-a", 10.0, -1.0, 18.0, 17.0, 16.0, 15.5, 15.0, 3.0, 0.03, -2.0, 1, 0, 0],
        ["damaged-a", 11.0, -2.0, 19.0, 18.0, 17.0, 16.5, 16.0, 4.0, 0.04, -2.0, 0, 1, 0],
        ["damaged-b", 11.0, -2.0, 19.1, 18.1, 17.1, 16.6, 16.1, 4.1, 0.04, -2.0, 0, 0, 1],
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_preparation_preserves_rows_and_exposes_collisions() -> None:
    frame = sample_frame()
    prepared = prepare_dataset(frame, "a" * 64)

    assert len(prepared) == len(frame)
    assert prepared["source_row_id"].is_unique
    assert prepared.loc[0, "coordinate_id"] != prepared.loc[1, "coordinate_id"]
    assert prepared.loc[1, "coordinate_id"] == prepared.loc[2, "coordinate_id"]
    assert prepared["legacy_objid_collision"].tolist() == [True, True, False]
    assert prepared["coordinate_has_label_conflict"].tolist() == [False, True, True]


def test_uncertain_is_preserved_and_not_reinterpreted() -> None:
    prepared = prepare_dataset(sample_frame(), "b" * 64)

    assert prepared["source_label"].tolist() == [
        "elliptical",
        "spiral",
        "uncertain",
    ]
    assert prepared["is_confident_morphology"].tolist() == [True, True, False]
    assert "irregular" not in set(prepared["source_label"])


def test_invalid_one_hot_labels_fail() -> None:
    frame = sample_frame()
    frame.loc[0, ["elliptical", "spiral"]] = 1

    with pytest.raises(DataIntegrityError, match="exactly one source label"):
        prepare_dataset(frame, "c" * 64)


def test_course_asset_contract_and_audit() -> None:
    source_path = ROOT / "galaxias_1.csv"
    frame = load_and_validate_source(source_path)
    report = audit_source(frame)

    assert sha256_file(source_path) == EXPECTED_SOURCE_SHA256
    assert report["rows"] == 92_102
    assert report["distinct_legacy_objids"] == 57_681
    assert report["distinct_coordinate_groups"] == 83_806
    assert report["legacy_objid_collision_groups"] == 15_341
    assert report["coordinate_groups_with_label_conflicts"] == 19
    assert report["rows_in_coordinate_label_conflicts"] == 42
    assert report["label_counts"] == {
        "elliptical": 8_257,
        "spiral": 30_046,
        "uncertain": 53_799,
    }


def test_build_is_reproducible_and_manifest_guarded(tmp_path: Path) -> None:
    source_path = ROOT / "galaxias_1.csv"
    manifest_path = ROOT / "data" / "source-manifest.json"
    output_path = tmp_path / "galaxies_prepared.csv"
    report_path = tmp_path / "report.json"

    report = build_prepared_dataset(
        source_path, manifest_path, output_path, report_path
    )

    prepared = pd.read_csv(output_path)
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(prepared) == 92_102
    assert prepared["source_row_id"].is_unique
    assert report == written_report

    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["sha256"] = "0" * 64
    bad_manifest_path = tmp_path / "bad-manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="checksum mismatch"):
        build_prepared_dataset(
            source_path,
            bad_manifest_path,
            tmp_path / "unused.csv",
            tmp_path / "unused.json",
        )
