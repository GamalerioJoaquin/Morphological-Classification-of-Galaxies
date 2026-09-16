"""Build the lineage-preserving dataset used by future analysis branches."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from a repository checkout without installing a package.
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.data_integrity import (  # noqa: E402
    DataIntegrityError,
    build_prepared_dataset,
)


def main() -> int:
    try:
        report = build_prepared_dataset(
            source_path=ROOT / "galaxias_1.csv",
            manifest_path=ROOT / "data" / "source-manifest.json",
            output_path=ROOT / "data" / "processed" / "galaxies_prepared.csv",
            report_path=ROOT / "reports" / "generated" / "data_integrity_report.json",
        )
    except DataIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
