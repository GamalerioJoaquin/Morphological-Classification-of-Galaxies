"""Leakage-safe supervised evaluation for confident galaxy morphologies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_array, check_is_fitted

from .data_integrity import (
    DataIntegrityError,
    load_and_validate_source,
    load_manifest,
    prepare_dataset,
    sha256_file,
)


RANDOM_STATE = 42
CLASS_LABELS = ["elliptical", "spiral"]
FEATURE_NAMES = [
    "modelMag_r",
    "color_u_minus_g",
    "color_g_minus_r",
    "color_r_minus_i",
    "color_i_minus_z",
    "petroR90_r",
    "z",
]
MAGNITUDE_COLUMNS = [
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
]
INVALID_MAGNITUDE_ABS_LIMIT = 100.0


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to quantiles learned only from the fitted partition."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> "QuantileClipper":
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
        values = check_array(X, dtype=float, force_all_finite="allow-nan")
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        values = check_array(X, dtype=float, force_all_finite="allow-nan")
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {values.shape[1]}."
            )
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def select_modeling_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    """Select confident E/S rows and quarantine coordinate-label conflicts."""

    required = {
        "is_confident_morphology",
        "coordinate_has_label_conflict",
        "source_label",
        "coordinate_id",
    }
    missing = required.difference(prepared.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

    eligible = prepared.loc[
        prepared["is_confident_morphology"]
        & ~prepared["coordinate_has_label_conflict"]
    ].copy()
    if not set(eligible["source_label"]).issubset(CLASS_LABELS):
        raise ValueError("Modeling rows contain a label outside elliptical/spiral.")
    return eligible.reset_index(drop=True)


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable photometric colors plus size and redshift."""

    magnitudes = frame[MAGNITUDE_COLUMNS].mask(
        frame[MAGNITUDE_COLUMNS].abs().gt(INVALID_MAGNITUDE_ABS_LIMIT)
    )
    features = pd.DataFrame(index=frame.index)
    features["modelMag_r"] = magnitudes["modelMag_r"]
    features["color_u_minus_g"] = magnitudes["modelMag_u"] - magnitudes["modelMag_g"]
    features["color_g_minus_r"] = magnitudes["modelMag_g"] - magnitudes["modelMag_r"]
    features["color_r_minus_i"] = magnitudes["modelMag_r"] - magnitudes["modelMag_i"]
    features["color_i_minus_z"] = magnitudes["modelMag_i"] - magnitudes["modelMag_z"]
    features["petroR90_r"] = frame["petroR90_r"]
    features["z"] = frame["z"]
    return features[FEATURE_NAMES]


def make_holdout_split(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic 80/20 split with no coordinate-group overlap."""

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    features = build_feature_matrix(frame)
    labels = frame["source_label"]
    groups = frame["coordinate_id"]
    development_indices, test_indices = next(
        splitter.split(features, labels, groups)
    )
    return development_indices, test_indices


def _preprocessing_steps(scale: bool) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [
        ("clip", QuantileClipper(lower=0.01, upper=0.99)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", RobustScaler()))
    return steps


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            *_preprocessing_steps(scale=True),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            *_preprocessing_steps(scale=False),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _cross_validation_summary(scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        metric.removeprefix("test_") + "_mean": float(np.mean(values))
        for metric, values in scores.items()
        if metric.startswith("test_")
    } | {
        metric.removeprefix("test_") + "_std": float(np.std(values))
        for metric, values in scores.items()
        if metric.startswith("test_")
    }


def _test_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CLASS_LABELS
        ).tolist(),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASS_LABELS
        },
    }


def _group_bootstrap_macro_f1(
    y_true: pd.Series,
    y_pred: np.ndarray,
    groups: pd.Series,
    iterations: int = 500,
) -> dict[str, float]:
    """Estimate uncertainty while resampling entire coordinate groups."""

    rng = np.random.default_rng(RANDOM_STATE)
    group_values = groups.to_numpy()
    unique_groups = pd.unique(group_values)
    indices_by_group = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    true_values = y_true.to_numpy()
    scores = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [indices_by_group[group] for group in sampled_groups]
        )
        scores.append(
            f1_score(
                true_values[sampled_indices],
                y_pred[sampled_indices],
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        )
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return {"lower_95": float(lower), "upper_95": float(upper)}


def _cross_split_proximity_audit(
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Check for near-coordinate leakage beyond exact coordinate grouping."""

    development_coordinates = np.radians(
        np.column_stack([development["dec"], development["ra"]])
    )
    test_coordinates = np.radians(np.column_stack([test["dec"], test["ra"]]))
    tree = BallTree(development_coordinates, metric="haversine")
    distances, _ = tree.query(test_coordinates, k=1)
    arcseconds = distances[:, 0] * (180 / np.pi) * 3_600
    return {
        "minimum_cross_split_separation_arcsec": float(arcseconds.min()),
        "test_rows_with_development_neighbor_within_arcsec": {
            str(threshold): int(np.count_nonzero(arcseconds <= threshold))
            for threshold in [1, 5, 10, 30]
        },
    }


def _save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        display_labels=["Elliptical", "Spiral"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Held-out test confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)


def run_evaluation(
    source_path: Path,
    manifest_path: Path,
    results_path: Path,
    confusion_matrix_path: Path,
) -> dict[str, Any]:
    """Train, select on group CV, and evaluate once on a held-out group split."""

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
    modeling = select_modeling_rows(prepared)
    features = build_feature_matrix(modeling)
    labels = modeling["source_label"]
    groups = modeling["coordinate_id"]

    development_indices, test_indices = make_holdout_split(modeling)
    X_development = features.iloc[development_indices]
    X_test = features.iloc[test_indices]
    y_development = labels.iloc[development_indices]
    y_test = labels.iloc[test_indices]
    groups_development = groups.iloc[development_indices]
    groups_test = groups.iloc[test_indices]

    if set(groups_development).intersection(groups_test):
        raise RuntimeError("Coordinate groups overlap development and test sets.")

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    cv_splits = list(
        cv.split(X_development, y_development, groups_development)
    )
    scoring = {
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }

    dummy = DummyClassifier(strategy="prior")
    dummy_cv = cross_validate(
        dummy, X_development, y_development, cv=cv_splits, scoring=scoring
    )
    dummy.fit(X_development, y_development)

    logistic = make_logistic_pipeline()
    logistic_cv = cross_validate(
        logistic, X_development, y_development, cv=cv_splits, scoring=scoring
    )
    logistic.fit(X_development, y_development)

    forest_search = GridSearchCV(
        make_random_forest_pipeline(),
        param_grid={
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 5, 20],
        },
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    forest_search.fit(X_development, y_development)

    models = {
        "dummy_prior": dummy,
        "logistic_regression": logistic,
        "random_forest": forest_search.best_estimator_,
    }
    cv_results = {
        "dummy_prior": _cross_validation_summary(dummy_cv),
        "logistic_regression": _cross_validation_summary(logistic_cv),
        "random_forest": {
            "macro_f1_mean": float(forest_search.best_score_),
            "macro_f1_std": float(
                forest_search.cv_results_["std_test_score"][forest_search.best_index_]
            ),
            "best_parameters": forest_search.best_params_,
        },
    }
    selected_model = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: cv_results[name]["macro_f1_mean"],
    )

    test_results: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        test_predictions[name] = predictions
        test_results[name] = _test_metrics(y_test, predictions)

    selected_predictions = test_predictions[selected_model]
    test_results[selected_model]["macro_f1_group_bootstrap_interval"] = (
        _group_bootstrap_macro_f1(y_test, selected_predictions, groups_test)
    )
    _save_confusion_matrix(y_test, selected_predictions, confusion_matrix_path)

    results: dict[str, Any] = {
        "source_sha256": source_sha256,
        "random_state": RANDOM_STATE,
        "target": "elliptical_vs_spiral",
        "excluded": {
            "uncertain_rows": int((~prepared["is_confident_morphology"]).sum()),
            "coordinate_label_conflict_rows": int(
                (
                    prepared["is_confident_morphology"]
                    & prepared["coordinate_has_label_conflict"]
                ).sum()
            ),
        },
        "features": FEATURE_NAMES,
        "invalid_magnitude_policy": (
            f"absolute magnitude values above {INVALID_MAGNITUDE_ABS_LIMIT:g} "
            "are treated as missing before fold-local imputation"
        ),
        "split": {
            "development_rows": int(len(development_indices)),
            "test_rows": int(len(test_indices)),
            "development_coordinate_groups": int(groups_development.nunique()),
            "test_coordinate_groups": int(groups_test.nunique()),
            "coordinate_group_overlap": 0,
            "development_class_counts": {
                str(label): int(count)
                for label, count in y_development.value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(label): int(count)
                for label, count in y_test.value_counts().sort_index().items()
            },
            "proximity_audit": _cross_split_proximity_audit(
                modeling.iloc[development_indices], modeling.iloc[test_indices]
            ),
        },
        "selection_metric": "four_fold_group_cv_macro_f1",
        "selected_model": selected_model,
        "cross_validation": cv_results,
        "held_out_test": test_results,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results
