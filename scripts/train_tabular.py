"""Run the leakage-safe tabular baseline and write verified result artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from a repository checkout without installing a package.
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.data_integrity import DataIntegrityError  # noqa: E402
from galaxy_classification.tabular_evaluation import run_evaluation  # noqa: E402


def main() -> int:
    try:
        results = run_evaluation(
            source_path=ROOT / "galaxias_1.csv",
            manifest_path=ROOT / "data" / "source-manifest.json",
            results_path=ROOT / "reports" / "results" / "tabular_metrics.json",
            confusion_matrix_path=(
                ROOT / "reports" / "figures" / "tabular_confusion_matrix.png"
            ),
        )
    except DataIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    selected = results["selected_model"]
    metrics = results["held_out_test"][selected]
    print(
        json.dumps(
            {
                "selected_model": selected,
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
