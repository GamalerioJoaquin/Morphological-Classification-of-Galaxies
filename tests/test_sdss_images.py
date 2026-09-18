from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from galaxy_classification.sdss_images import (
    CutoutConfig,
    build_match_query,
    build_targets,
    download_cutout,
    legacy_objid_is_compatible,
    parse_skyserver_csv,
    stable_split,
)


def test_targets_are_unique_and_label_conflicts_are_excluded() -> None:
    prepared = pd.DataFrame(
        {
            "coordinate_id": ["coord_a", "coord_a", "coord_b", "coord_c"],
            "legacy_objID": ["1E+18"] * 4,
            "ra": [1.0, 1.0, 2.0, 3.0],
            "dec": [4.0, 4.0, 5.0, 6.0],
            "source_label": ["spiral", "spiral", "elliptical", "uncertain"],
            "coordinate_group_size": [2, 2, 1, 1],
            "coordinate_has_label_conflict": [False, False, True, False],
        }
    )

    targets = build_targets(prepared, ("spiral", "elliptical"))

    assert targets["coordinate_id"].tolist() == ["coord_a"]


def test_precision_damaged_objid_can_be_checked_against_exact_id() -> None:
    assert legacy_objid_is_compatible(
        "1,23765119242489E+018", "1237651192424890568"
    )
    assert legacy_objid_is_compatible(
        "1,23765149575578E+018", "1237651495755776233"
    )
    assert not legacy_objid_is_compatible(
        "1,23765119242489E+018", "1237651495755776233"
    )


def test_match_query_uses_nearest_object_and_arcminute_radius() -> None:
    targets = pd.DataFrame(
        [{"coordinate_id": "coord_a", "ra": 116.5, "dec": 39.8}]
    )

    query = build_match_query(targets, radius_arcsec=1.0)

    assert "fGetNearestObjEq" in query
    assert "coord_a" in query
    assert str(float(Decimal(1) / Decimal(60)))[:8] in query


def test_skyserver_comment_line_is_ignored() -> None:
    payload = (
        b"#Table1\ncoordinate_id,sdss_objid,sdss_ra\n"
        b"coord_a,1237651192424890568,116.5\n"
    )

    parsed = parse_skyserver_csv(payload)

    assert parsed.iloc[0].to_dict() == {
        "coordinate_id": "coord_a",
        "sdss_objid": "1237651192424890568",
        "sdss_ra": "116.5",
    }


def test_download_cutout_writes_verified_jpeg_and_manifest_fields(
    tmp_path: Path,
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buffer, format="JPEG")
    calls = []

    def reader(url: str) -> bytes:
        calls.append(url)
        return buffer.getvalue()

    row = {
        "coordinate_id": "coord_a",
        "sdss_objid": "1237651192424890568",
        "ra": 116.5,
        "dec": 39.8,
    }
    result = download_cutout(
        row,
        tmp_path,
        CutoutConfig(width=32, height=32),
        reader=reader,
    )

    assert (tmp_path / "coord_a.jpg").is_file()
    assert result["object_id"] == "sdss_dr18_1237651192424890568"
    assert result["split"] == stable_split(result["object_id"])
    assert len(result["sha256"]) == 64
    assert len(calls) == 1
