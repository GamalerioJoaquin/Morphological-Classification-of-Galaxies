"""Leakage-safe clustering and embeddings for coordinate-level observations."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .data_integrity import (
    DataIntegrityError,
    load_and_validate_source,
    load_manifest,
    prepare_dataset,
    sha256_file,
)
from .tabular_evaluation import (
    FEATURE_NAMES,
    INVALID_MAGNITUDE_ABS_LIMIT,
    QuantileClipper,
    build_feature_matrix,
)


RANDOM_STATE = 42
DEFAULT_K_VALUES = range(2, 7)
DEFAULT_SEEDS = (42, 43, 44)
STABILITY_THRESHOLD = 0.90


def build_coordinate_dataset(prepared: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated measurements without using labels as model features."""

    features = build_feature_matrix(prepared)
    features["coordinate_id"] = prepared["coordinate_id"].to_numpy()
    coordinate_features = features.groupby("coordinate_id", sort=False).median()

    grouped = prepared.groupby("coordinate_id", sort=False)
    source_labels = grouped["source_label"].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else "conflict"
    )
    observation_counts = grouped.size().rename("observation_count")

    coordinate_data = coordinate_features.join(source_labels).join(observation_counts)
    coordinate_data.index.name = "coordinate_id"
    return coordinate_data.reset_index()


def unsupervised_features(coordinate_data: pd.DataFrame) -> pd.DataFrame:
    """Return only scientific numeric inputs, excluding IDs, labels, and clusters."""

    missing = set(FEATURE_NAMES).difference(coordinate_data.columns)
    if missing:
        raise ValueError(f"Coordinate data is missing features: {sorted(missing)}")
    return coordinate_data[FEATURE_NAMES].copy()


def make_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("clip", QuantileClipper(lower=0.01, upper=0.99)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )


def evaluate_cluster_candidates(
    transformed_features: np.ndarray,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    silhouette_sample_size: int = 5_000,
) -> list[dict[str, Any]]:
    """Measure label-independent separation and stability across random seeds."""

    seed_values = tuple(seeds)
    if len(seed_values) < 2:
        raise ValueError("At least two seeds are required for stability analysis.")

    results = []
    for number_of_clusters in k_values:
        assignments = []
        silhouette_scores = []
        for seed in seed_values:
            labels = KMeans(
                n_clusters=number_of_clusters,
                n_init=20,
                random_state=seed,
            ).fit_predict(transformed_features)
            assignments.append(labels)
            silhouette_scores.append(
                silhouette_score(
                    transformed_features,
                    labels,
                    sample_size=min(silhouette_sample_size, len(labels)),
                    random_state=RANDOM_STATE,
                )
            )

        stability_scores = [
            adjusted_rand_score(first, second)
            for first, second in itertools.combinations(assignments, 2)
        ]
        results.append(
            {
                "k": int(number_of_clusters),
                "silhouette_mean": float(np.mean(silhouette_scores)),
                "silhouette_std": float(np.std(silhouette_scores)),
                "stability_ari_mean": float(np.mean(stability_scores)),
                "stability_ari_min": float(np.min(stability_scores)),
            }
        )
    return results


def select_cluster_count(candidate_results: list[dict[str, Any]]) -> int:
    """Select the best silhouette among solutions meeting the stability gate."""

    stable = [
        result
        for result in candidate_results
        if result["stability_ari_mean"] >= STABILITY_THRESHOLD
    ]
    if not stable:
        raise ValueError(
            f"No candidate meets the stability threshold {STABILITY_THRESHOLD:.2f}."
        )
    return int(max(stable, key=lambda result: result["silhouette_mean"])["k"])


def fit_pca(transformed_features: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Fit PCA on features only; cluster assignments are not accepted as input."""

    model = PCA(n_components=2, random_state=RANDOM_STATE)
    return model, model.fit_transform(transformed_features)


def fit_tsne_sample(
    transformed_features: np.ndarray,
    sample_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit seeded t-SNE to a feature-only sample selected before visualization."""

    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(
        rng.choice(
            len(transformed_features),
            size=min(sample_size, len(transformed_features)),
            replace=False,
        )
    )
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        n_iter=1_000,
        random_state=RANDOM_STATE,
    ).fit_transform(transformed_features[sample_indices])
    return sample_indices, embedding


def _external_label_comparison(
    coordinate_data: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> dict[str, Any]:
    """Compare with confident E/S labels only after clustering is complete."""

    confident_mask = coordinate_data["source_label"].isin(["elliptical", "spiral"])
    source_labels = coordinate_data.loc[confident_mask, "source_label"]
    confident_clusters = cluster_labels[confident_mask.to_numpy()]
    return {
        "rows": int(confident_mask.sum()),
        "adjusted_rand_index": float(
            adjusted_rand_score(source_labels, confident_clusters)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(source_labels, confident_clusters)
        ),
        "homogeneity": float(homogeneity_score(source_labels, confident_clusters)),
        "completeness": float(completeness_score(source_labels, confident_clusters)),
    }


def _cluster_profiles(
    coordinate_data: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> list[dict[str, Any]]:
    profiled = coordinate_data.copy()
    profiled["cluster"] = cluster_labels
    profiles = []
    for cluster, group in profiled.groupby("cluster", sort=True):
        label_counts = group["source_label"].value_counts()
        profiles.append(
            {
                "cluster": int(cluster),
                "coordinate_groups": int(len(group)),
                "proportion": float(len(group) / len(profiled)),
                "feature_medians": {
                    feature: float(group[feature].median()) for feature in FEATURE_NAMES
                },
                "source_label_counts_for_interpretation_only": {
                    str(label): int(count)
                    for label, count in label_counts.sort_index().items()
                },
            }
        )
    return profiles


def _save_selection_figure(
    candidates: list[dict[str, Any]],
    selected_k: int,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = [candidate["k"] for candidate in candidates]
    silhouettes = [candidate["silhouette_mean"] for candidate in candidates]
    silhouette_errors = [candidate["silhouette_std"] for candidate in candidates]
    stability = [candidate["stability_ari_mean"] for candidate in candidates]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(k_values, silhouettes, yerr=silhouette_errors, marker="o")
    axes[0].axvline(selected_k, color="tab:red", linestyle="--", label="Selected k")
    axes[0].set(xlabel="Number of clusters", ylabel="Silhouette", title="Separation")
    axes[0].legend()
    axes[1].plot(k_values, stability, marker="o", color="tab:green")
    axes[1].axhline(STABILITY_THRESHOLD, color="tab:red", linestyle="--")
    axes[1].set(xlabel="Number of clusters", ylabel="Mean pairwise ARI", title="Stability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_embedding_figure(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
    title: str,
    axis_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    number_of_clusters = int(np.max(cluster_labels)) + 1
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap=plt.get_cmap("tab10", number_of_clusters),
        vmin=-0.5,
        vmax=number_of_clusters - 0.5,
        s=5,
        alpha=0.5,
        rasterized=True,
    )
    axis.set(
        xlabel=f"{axis_prefix} 1",
        ylabel=f"{axis_prefix} 2",
        title=title,
    )
    figure.colorbar(
        scatter,
        ax=axis,
        label="K-means cluster",
        ticks=range(number_of_clusters),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_unsupervised_analysis(
    source_path: Path,
    manifest_path: Path,
    results_path: Path,
    selection_figure_path: Path,
    pca_figure_path: Path,
    tsne_figure_path: Path,
) -> dict[str, Any]:
    """Run clustering, feature-only embeddings, and post-hoc label comparison."""

    source_sha256 = sha256_file(source_path)
    manifest = load_manifest(manifest_path)
    if source_sha256 != manifest["sha256"]:
        raise DataIntegrityError(
            "Source checksum mismatch: "
            f"expected {manifest['sha256']}, received {source_sha256}."
        )

    prepared = prepare_dataset(
        load_and_validate_source(source_path), source_sha256
    )
    coordinate_data = build_coordinate_dataset(prepared)
    features = unsupervised_features(coordinate_data)
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(features)

    candidate_results = evaluate_cluster_candidates(transformed)
    selected_k = select_cluster_count(candidate_results)
    cluster_model = KMeans(
        n_clusters=selected_k,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    cluster_labels = cluster_model.fit_predict(transformed)

    pca_model, pca_embedding = fit_pca(transformed)
    rng = np.random.default_rng(RANDOM_STATE)
    pca_sample_indices = np.sort(
        rng.choice(len(transformed), size=min(20_000, len(transformed)), replace=False)
    )
    tsne_sample_indices, tsne_embedding = fit_tsne_sample(transformed)

    _save_selection_figure(candidate_results, selected_k, selection_figure_path)
    _save_embedding_figure(
        pca_embedding[pca_sample_indices],
        cluster_labels[pca_sample_indices],
        pca_figure_path,
        title="PCA of photometric features (colored after projection)",
        axis_prefix="Principal component",
    )
    _save_embedding_figure(
        tsne_embedding,
        cluster_labels[tsne_sample_indices],
        tsne_figure_path,
        title="t-SNE of photometric features (colored after embedding)",
        axis_prefix="t-SNE dimension",
    )

    results: dict[str, Any] = {
        "source_sha256": source_sha256,
        "random_state": RANDOM_STATE,
        "analysis_unit": "exact_coordinate_group",
        "coordinate_groups": int(len(coordinate_data)),
        "input_features": FEATURE_NAMES,
        "excluded_from_features": [
            "source_row_id",
            "source_row_number",
            "coordinate_id",
            "legacy_objID",
            "source_label",
            "cluster",
        ],
        "preprocessing": {
            "invalid_magnitude_abs_limit": INVALID_MAGNITUDE_ABS_LIMIT,
            "quantile_clipping": [0.01, 0.99],
            "imputation": "median",
            "scaling": "robust",
        },
        "selection_rule": (
            "highest mean silhouette among candidates with mean pairwise "
            f"stability ARI >= {STABILITY_THRESHOLD:.2f}"
        ),
        "candidates": candidate_results,
        "selected_k": selected_k,
        "pca": {
            "explained_variance_ratio": [
                float(value) for value in pca_model.explained_variance_ratio_
            ],
            "total_explained_variance": float(
                pca_model.explained_variance_ratio_.sum()
            ),
            "fit_input": "preprocessed features only",
        },
        "tsne": {
            "sample_size": int(len(tsne_sample_indices)),
            "perplexity": 30,
            "fit_input": "preprocessed features only",
        },
        "external_label_comparison_not_used_for_selection": (
            _external_label_comparison(coordinate_data, cluster_labels)
        ),
        "cluster_profiles": _cluster_profiles(coordinate_data, cluster_labels),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results
