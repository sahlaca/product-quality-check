"""Train and package the SECOM quality classifier without test-set leakage."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "secom.data"
LABELS_PATH = ROOT / "secom_labels.data"
MODEL_PATH = ROOT / "secom_quality_model.pkl"
METRICS_PATH = ROOT / "training_metrics.json"
DEMO_TEST_DIR = ROOT / "demo_test_data"
RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 5
N_ESTIMATORS = 500


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Read the local SECOM files and map 1 to fail (0), -1 to pass (1)."""
    features = pd.read_csv(FEATURES_PATH, sep=r"\s+", header=None)
    labels = pd.read_csv(LABELS_PATH, sep=r"\s+", header=None, usecols=[0])[0]
    labels = labels.map({1: 0, -1: 1})
    if labels.isna().any():
        raise ValueError("The label file contains values other than 1 and -1.")
    if len(features) != len(labels):
        raise ValueError("The feature and label files have different row counts.")
    return features, labels.astype(int)


def prepare_split(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.Series, list[int], StandardScaler]:
    """Fit imputing, constant-column filtering, and scaling on training rows only."""
    medians = train.median().fillna(0.0)
    train_filled = train.fillna(medians)
    test_filled = test.fillna(medians)
    selected = [int(i) for i in train_filled.columns if train_filled[i].nunique() > 1]
    if not selected:
        raise ValueError("No non-constant sensor features remain in the training data.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled[selected])
    test_scaled = scaler.transform(test_filled[selected])
    return train_scaled, test_scaled, medians, selected, scaler


def evaluate_repeated_stratified_cv(
    features: pd.DataFrame, labels: pd.Series
) -> dict:
    """Score repeated folds, fitting every preprocessing step within each fold."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    fold_scores: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "fail_recall": [],
        "pass_recall": [],
        "roc_auc_for_fail": [],
    }
    aggregate_confusion = np.zeros((2, 2), dtype=np.int64)
    test_failure_counts = []

    for fold_number, (train_indices, test_indices) in enumerate(
        cv.split(features, labels), start=1
    ):
        x_train = features.iloc[train_indices]
        x_test = features.iloc[test_indices]
        y_train = labels.iloc[train_indices]
        y_test = labels.iloc[test_indices]

        train_scaled, test_scaled, _, _, _ = prepare_split(x_train, x_test)
        train_balanced, labels_balanced = SMOTE(
            random_state=RANDOM_STATE + fold_number
        ).fit_resample(train_scaled, y_train)
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train_balanced, labels_balanced)
        predictions = model.predict(test_scaled)
        probabilities = model.predict_proba(test_scaled)
        class_to_column = {int(label): i for i, label in enumerate(model.classes_)}
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        aggregate_confusion += matrix
        test_failure_counts.append(int((y_test == 0).sum()))
        fold_scores["accuracy"].append(float(accuracy_score(y_test, predictions)))
        fold_scores["balanced_accuracy"].append(
            float(balanced_accuracy_score(y_test, predictions))
        )
        fold_scores["fail_recall"].append(
            float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
        )
        fold_scores["pass_recall"].append(
            float(matrix[1, 1] / matrix[1].sum()) if matrix[1].sum() else 0.0
        )
        fold_scores["roc_auc_for_fail"].append(
            float(
                roc_auc_score(
                    (y_test == 0).astype(int),
                    probabilities[:, class_to_column[0]],
                )
            )
        )
        if fold_number % CV_SPLITS == 0:
            print(f"Completed CV fold {fold_number}/{CV_SPLITS * CV_REPEATS}", flush=True)

    mean = {name: float(np.mean(scores)) for name, scores in fold_scores.items()}
    std = {name: float(np.std(scores, ddof=1)) for name, scores in fold_scores.items()}
    return {
        "method": f"{CV_SPLITS}-fold stratified CV repeated {CV_REPEATS} times",
        "fold_count": CV_SPLITS * CV_REPEATS,
        "rows_used": int(len(features)),
        "mean": mean,
        "standard_deviation": std,
        "aggregate_confusion_matrix_labels_0_fail_1_pass": aggregate_confusion.tolist(),
        "aggregate_predictions": int(len(features) * CV_REPEATS),
        "test_failures_per_fold_min": min(test_failure_counts),
        "test_failures_per_fold_max": max(test_failure_counts),
    }


def main() -> None:
    features, labels = load_data()

    train_indices, test_indices = train_test_split(
        np.arange(len(features)),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    print(
        f"Running {CV_SPLITS}-fold stratified CV × {CV_REPEATS} repeats "
        f"within the training split only "
        f"({CV_SPLITS * CV_REPEATS} fits)...",
        flush=True,
    )
    repeated_cv_metrics = evaluate_repeated_stratified_cv(x_train, y_train)

    train_scaled, test_scaled, train_medians, selected_for_eval, train_scaler = prepare_split(
        x_train, x_test
    )
    sampler = SMOTE(random_state=RANDOM_STATE)
    train_balanced, labels_balanced = sampler.fit_resample(train_scaled, y_train)

    evaluation_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    evaluation_model.fit(train_balanced, labels_balanced)
    predictions = evaluation_model.predict(test_scaled)
    probabilities = evaluation_model.predict_proba(test_scaled)
    class_to_column = {int(label): index for index, label in enumerate(evaluation_model.classes_)}

    metrics = {
        "evaluation": "stratified 80/20 holdout; preprocessing and SMOTE fitted on training rows only",
        "rows": int(len(features)),
        "input_feature_count": int(features.shape[1]),
        "selected_feature_count": len(selected_for_eval),
        "model_training_rows": int(len(x_train)),
        "test_rows": int(len(y_test)),
        "test_failures": int((y_test == 0).sum()),
        "test_passes": int((y_test == 1).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc_for_fail": float(
            roc_auc_score(
                (y_test == 0).astype(int),
                probabilities[:, class_to_column[0]],
            )
        ),
        "roc_auc_positive_class": "Fail (mapped class 0)",
        "confusion_matrix_labels_0_fail_1_pass": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["fail", "pass"],
            output_dict=True,
            zero_division=0,
        ),
        "repeated_stratified_cv": repeated_cv_metrics,
    }

    # Preserve the untouched holdout, in source row order, for a fair demo.
    # The app model below is trained only on x_train and can score these rows.
    DEMO_TEST_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_TEST_DIR / "README.txt").write_text(
        "This folder contains the untouched 20% holdout used for the demo evaluation.\n"
        "secom_test_features.data rows are space-separated and align by row position\n"
        "with secom_test_labels.csv. Example one-row CSVs are in examples/.\n"
        "The saved app model was fitted only on the other 80% training split.\n",
        encoding="utf-8",
    )
    x_test.to_csv(
        DEMO_TEST_DIR / "secom_test_features.data",
        sep=" ",
        header=False,
        index=False,
        na_rep="NaN",
    )
    original_labels = y_test.map({0: 1, 1: -1})
    pd.DataFrame(
        {
            "record_number_in_source": test_indices + 1,
            "actual_label": y_test.map({0: "Fail", 1: "Pass"}).to_numpy(),
            "original_secom_label": original_labels.to_numpy(),
        }
    ).to_csv(DEMO_TEST_DIR / "secom_test_labels.csv", index=False)

    # Small one-record CSVs let the current UI demonstrate an unseen detected
    # failure, a missed failure, and a correctly predicted pass when available.
    examples_dir = DEMO_TEST_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    feature_headers = [f"Sensor_{i}" for i in range(features.shape[1])]
    example_manifest = []
    example_specs = (
        ("fail_detected_example", (y_test.to_numpy() == 0) & (predictions == 0)),
        ("fail_missed_example", (y_test.to_numpy() == 0) & (predictions == 1)),
        ("pass_example", (y_test.to_numpy() == 1) & (predictions == 1)),
    )
    for example_name, selection in example_specs:
        candidates = np.flatnonzero(selection)
        if not len(candidates):
            continue
        test_position = int(candidates[0])
        example_path = examples_dir / f"{example_name}.csv"
        pd.DataFrame(
            [x_test.iloc[test_position].to_numpy()], columns=feature_headers
        ).to_csv(example_path, index=False)
        example_manifest.append(
            {
                "file": example_path.name,
                "record_number_in_source": int(test_indices[test_position]) + 1,
                "actual_label": "Fail" if int(y_test.iloc[test_position]) == 0 else "Pass",
                "model_prediction": "Fail" if int(predictions[test_position]) == 0 else "Pass",
            }
        )
    (DEMO_TEST_DIR / "examples" / "README.txt").write_text(
        "These one-row CSVs come from the held-out test set and can be uploaded to the app.\n"
        "The detected-failure and missed-failure examples show both outcomes of the model.\n"
        "The app model was fitted only on the training split. Compare its result with\n"
        "the actual_label in demo_test_data/secom_test_labels.csv. A prediction can be wrong.\n",
        encoding="utf-8",
    )
    pd.DataFrame(example_manifest).to_csv(
        DEMO_TEST_DIR / "examples" / "example_labels.csv", index=False
    )

    # Keep this demo artifact trained only on the training split so it has never
    # seen the holdout rows saved above.
    bundle = {
        "model": evaluation_model,
        "scaler": train_scaler,
        "median": train_medians,
        "feature_indices": selected_for_eval,
        "input_feature_count": int(features.shape[1]),
        "class_labels": {0: "Fail", 1: "Pass"},
        "training_rows": int(len(x_train)),
        "random_state": RANDOM_STATE,
        "bundle_purpose": "demo model fitted on the 80% training split; holdout rows remain unseen",
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved training-only demo model: {MODEL_PATH}")
    print(f"Saved held-out demo rows and labels: {DEMO_TEST_DIR}")
    print(f"Saved holdout and CV metrics: {METRICS_PATH}")
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print("Holdout classification report:")
    print(json.dumps(metrics["classification_report"], indent=2))


if __name__ == "__main__":
    main()
