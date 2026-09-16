"""Print a read-only identity and label audit for the legacy course asset."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from a repository checkout without installing a package.
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.data_integrity import (  # noqa: E402
    audit_source,
    load_and_validate_source,
    load_manifest,
    sha256_file,
)


def main() -> int:
    source_path = ROOT / "galaxias_1.csv"
    manifest = load_manifest(ROOT / "data" / "source-manifest.json")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != manifest["sha256"]:
        print("ERROR: source checksum does not match the tracked manifest.", file=sys.stderr)
        return 1

    report = audit_source(load_and_validate_source(source_path))
    report["source_sha256"] = actual_sha256
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
