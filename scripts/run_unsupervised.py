"""Run reproducible clustering and feature-only dimensionality reduction."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution from a repository checkout without installing a package.
sys.path.insert(0, str(ROOT / "src"))

from galaxy_classification.data_integrity import DataIntegrityError  # noqa: E402
from galaxy_classification.unsupervised_analysis import (  # noqa: E402
    run_unsupervised_analysis,
)


def main() -> int:
    try:
        results = run_unsupervised_analysis(
            source_path=ROOT / "galaxias_1.csv",
            manifest_path=ROOT / "data" / "source-manifest.json",
            results_path=ROOT / "reports" / "results" / "unsupervised_metrics.json",
            selection_figure_path=(
                ROOT / "reports" / "figures" / "cluster_selection.png"
            ),
            pca_figure_path=ROOT / "reports" / "figures" / "pca_clusters.png",
            tsne_figure_path=ROOT / "reports" / "figures" / "tsne_clusters.png",
        )
    except DataIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    selected = next(
        candidate
        for candidate in results["candidates"]
        if candidate["k"] == results["selected_k"]
    )
    print(
        json.dumps(
            {
                "selected_k": results["selected_k"],
                "silhouette_mean": selected["silhouette_mean"],
                "stability_ari_mean": selected["stability_ari_mean"],
                "pca_total_explained_variance": results["pca"][
                    "total_explained_variance"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
