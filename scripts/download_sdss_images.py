"""Recover exact SDSS IDs and download manifest-backed galaxy cutouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.data_integrity import (  # noqa: E402
    load_and_validate_source,
    load_manifest,
    prepare_dataset,
    sha256_file,
)
from galaxy_classification.sdss_images import (  # noqa: E402
    CutoutConfig,
    SdssImageError,
    build_targets,
    download_cutouts,
    legacy_objid_is_compatible,
    match_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover full SDSS DR18 objIDs by coordinate and download RGB JPEG cutouts."
        )
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=("elliptical", "spiral", "uncertain"),
        default=("elliptical", "spiral"),
    )
    parser.add_argument("--limit", type=int, help="Download only the first N targets.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.396)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--match-radius", type=float, default=1.0)
    parser.add_argument("--match-batch-size", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "images" / "sdss-dr18-jpeg"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = ROOT / "galaxias_1.csv"
    source_manifest = load_manifest(ROOT / "data" / "source-manifest.json")
    actual_sha = sha256_file(source_path)
    if actual_sha != source_manifest["sha256"]:
        print("ERROR: source checksum differs from data/source-manifest.json", file=sys.stderr)
        return 1

    prepared = prepare_dataset(load_and_validate_source(source_path), actual_sha)
    targets = build_targets(prepared, args.labels)
    if args.limit is not None:
        if args.limit < 1:
            print("ERROR: --limit must be positive", file=sys.stderr)
            return 1
        targets = targets.head(args.limit).copy()

    print(f"Matching {len(targets):,} unique, non-conflicting coordinates in SDSS DR18...")
    try:
        matches = match_catalog(
            targets,
            radius_arcsec=args.match_radius,
            batch_size=args.match_batch_size,
        )
    except (OSError, SdssImageError, ValueError) as exc:
        print(f"ERROR: catalog match failed: {exc}", file=sys.stderr)
        return 1

    matched = targets.merge(matches, on="coordinate_id", how="inner", validate="one_to_one")
    matched["legacy_objid_compatible"] = [
        legacy_objid_is_compatible(legacy, recovered)
        for legacy, recovered in zip(matched["legacy_objID"], matched["sdss_objid"])
    ]
    safe = matched.loc[matched["legacy_objid_compatible"]].copy()
    rejected = matched.loc[~matched["legacy_objid_compatible"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = args.output_dir / "catalog-matches.csv"
    matched.to_csv(matches_path, index=False)
    if not rejected.empty:
        rejected.to_csv(args.output_dir / "rejected-id-matches.csv", index=False)

    config = CutoutConfig(
        width=args.width,
        height=args.height,
        scale_arcsec_per_pixel=args.scale,
        workers=args.workers,
    )
    print(
        f"Downloading {len(safe):,} verified RGB cutouts "
        f"({len(targets) - len(matches):,} unmatched, {len(rejected):,} ID mismatches)..."
    )
    images, failures = download_cutouts(safe, args.output_dir, config)

    metadata_columns = [
        "coordinate_id",
        "legacy_objID",
        "sdss_objid",
        "ra",
        "dec",
        "sdss_ra",
        "sdss_dec",
        "match_distance_arcsec",
        "source_label",
        "coordinate_group_size",
        "run",
        "rerun",
        "camcol",
        "field",
        "sdss_modelMag_u",
        "sdss_modelMag_g",
        "sdss_modelMag_r",
        "sdss_modelMag_i",
        "sdss_modelMag_z",
    ]
    manifest = safe[metadata_columns].merge(
        images, on="coordinate_id", how="inner", validate="one_to_one"
    )
    manifest = manifest.rename(columns={"source_label": "label"})
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_counts = manifest["split"].value_counts().to_dict() if not manifest.empty else {}
    print(
        json.dumps(
            {
                "requested_targets": len(targets),
                "catalog_matches": len(matches),
                "legacy_id_compatible": len(safe),
                "downloaded_or_verified": len(manifest),
                "download_failures": len(failures),
                "splits": split_counts,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
