from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy_classification.data_integrity import SOURCE_COLUMNS, prepare_dataset
from galaxy_classification.tabular_evaluation import (
    FEATURE_NAMES,
    QuantileClipper,
    build_feature_matrix,
    make_holdout_split,
    select_modeling_rows,
)


def sample_prepared_data() -> pd.DataFrame:
    rows = []
    for index in range(40):
        label = (index // 2) % 2
        rows.append(
            [
                f"legacy-{index // 2}",
                float(index // 2),
                float(index // 2 + 1),
                18.0 + label,
                17.0 + label,
                16.0 + label,
                15.5 + label,
                15.0 + label,
                5.0 + label,
                0.03,
                -2.0,
                1 - label,
                label,
                0,
            ]
        )
    rows.append(
        [
            "uncertain",
            100.0,
            101.0,
            18.0,
            17.0,
            16.0,
            15.5,
            15.0,
            5.0,
            0.03,
            -2.0,
            0,
            0,
            1,
        ]
    )
    return prepare_dataset(pd.DataFrame(rows, columns=SOURCE_COLUMNS), "d" * 64)


def test_modeling_selection_excludes_uncertain_and_conflicts() -> None:
    prepared = sample_prepared_data()
    prepared.loc[0, "coordinate_has_label_conflict"] = True
    selected = select_modeling_rows(prepared)

    assert set(selected["source_label"]) == {"elliptical", "spiral"}
    assert not selected["coordinate_has_label_conflict"].any()
    assert len(selected) == len(prepared) - 2


def test_feature_matrix_contains_no_identifiers_or_targets() -> None:
    selected = select_modeling_rows(sample_prepared_data())
    features = build_feature_matrix(selected)

    assert list(features.columns) == FEATURE_NAMES
    assert not any("id" in column.lower() for column in features.columns)
    assert np.isfinite(features.to_numpy()).all()


def test_quantile_clipper_uses_only_fitted_values() -> None:
    training = np.array([[0.0], [1.0], [2.0], [100.0]])
    clipper = QuantileClipper(lower=0.0, upper=0.75).fit(training)

    transformed = clipper.transform(np.array([[-1_000.0], [10_000.0]]))

    assert transformed[0, 0] == clipper.lower_bounds_[0]
    assert transformed[1, 0] == clipper.upper_bounds_[0]
    assert clipper.upper_bounds_[0] < 100.0


def test_holdout_split_is_deterministic_and_group_disjoint() -> None:
    selected = select_modeling_rows(sample_prepared_data())
    first_development, first_test = make_holdout_split(selected)
    second_development, second_test = make_holdout_split(selected)

    assert np.array_equal(first_development, second_development)
    assert np.array_equal(first_test, second_test)
    development_groups = set(selected.iloc[first_development]["coordinate_id"])
    test_groups = set(selected.iloc[first_test]["coordinate_id"])
    assert development_groups.isdisjoint(test_groups)
