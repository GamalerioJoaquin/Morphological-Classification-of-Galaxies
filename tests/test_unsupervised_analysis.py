from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxy_classification.data_integrity import SOURCE_COLUMNS, prepare_dataset
from galaxy_classification.tabular_evaluation import FEATURE_NAMES
from galaxy_classification.unsupervised_analysis import (
    build_coordinate_dataset,
    fit_pca,
    select_cluster_count,
    unsupervised_features,
)


def sample_prepared_data() -> pd.DataFrame:
    rows = [
        ["legacy-a", 10.0, -1.0, 18.0, 17.0, 16.0, 15.5, 15.0, 3.0, 0.03, -2.0, 1, 0, 0],
        ["legacy-b", 10.0, -1.0, 18.2, 17.2, 16.2, 15.7, 15.2, 3.2, 0.03, -2.0, 1, 0, 0],
        ["legacy-c", 20.0, -2.0, 20.0, 19.0, 18.0, 17.5, 17.0, 5.0, 0.04, -2.0, 0, 1, 0],
        ["legacy-d", 30.0, -3.0, 19.0, 18.0, 17.0, 16.5, 16.0, 4.0, 0.04, -2.0, 0, 0, 1],
    ]
    return prepare_dataset(pd.DataFrame(rows, columns=SOURCE_COLUMNS), "e" * 64)


def test_coordinate_dataset_aggregates_repeated_measurements() -> None:
    coordinate_data = build_coordinate_dataset(sample_prepared_data())

    assert len(coordinate_data) == 3
    repeated = coordinate_data.loc[coordinate_data["observation_count"].eq(2)].iloc[0]
    assert repeated["modelMag_r"] == pytest.approx(16.1)
    assert repeated["source_label"] == "elliptical"


def test_unsupervised_features_exclude_identity_labels_and_clusters() -> None:
    coordinate_data = build_coordinate_dataset(sample_prepared_data())
    coordinate_data["cluster"] = [0, 1, 1]
    features = unsupervised_features(coordinate_data)

    assert list(features.columns) == FEATURE_NAMES
    assert "coordinate_id" not in features
    assert "source_label" not in features
    assert "cluster" not in features


def test_pca_accepts_features_only_and_is_deterministic() -> None:
    values = np.arange(70, dtype=float).reshape(10, 7)
    first_model, first_embedding = fit_pca(values)
    second_model, second_embedding = fit_pca(values)

    assert first_embedding.shape == (10, 2)
    assert np.allclose(first_embedding, second_embedding)
    assert np.allclose(
        first_model.explained_variance_ratio_,
        second_model.explained_variance_ratio_,
    )


def test_cluster_selection_requires_stability_then_maximizes_silhouette() -> None:
    candidates = [
        {"k": 2, "silhouette_mean": 0.25, "stability_ari_mean": 0.99},
        {"k": 3, "silhouette_mean": 0.35, "stability_ari_mean": 0.70},
        {"k": 4, "silhouette_mean": 0.30, "stability_ari_mean": 0.95},
    ]

    assert select_cluster_count(candidates) == 4


def test_cluster_selection_fails_when_every_candidate_is_unstable() -> None:
    candidates = [
        {"k": 2, "silhouette_mean": 0.25, "stability_ari_mean": 0.80},
        {"k": 3, "silhouette_mean": 0.35, "stability_ari_mean": 0.70},
    ]

    with pytest.raises(ValueError, match="No candidate meets"):
        select_cluster_count(candidates)
