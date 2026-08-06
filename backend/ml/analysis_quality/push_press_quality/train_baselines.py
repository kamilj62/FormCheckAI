import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


BASE = Path("ml/analysis_quality/push_press_quality/results")
FEATURES_PATH = BASE / "push_press_features.csv"
OUTPUT_DIR = BASE / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


META_COLUMNS = {
    "video_id",
    "video_path",
    "target",
    "label",
    "split",
    "window_start_seconds",
    "window_end_seconds",
    "source_interval_index",
    "sampling_type",
    "source_fps",
    "target_fps",
}


def select_features(df, target):
    numeric = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in META_COLUMNS
    ]

    common_prefixes = (
        "pose_",
        "mean_visibility",
        "processed_frames",
        "shoulder_y_",
        "hip_y_",
        "torso_lean_",
    )

    if target == "elbow_error":
        target_prefixes = (
            "left_elbow_angle_",
            "right_elbow_angle_",
            "elbow_angle_",
            "wrist_y_",
            "wrist_x_",
            "wrist_above_shoulder_",
            "wrist_shoulder_offset_x_",
        )
    else:
        target_prefixes = (
            "left_knee_angle_",
            "right_knee_angle_",
            "knee_angle_",
            "hip_y_",
            "torso_lean_",
        )

    selected = [
        column
        for column in numeric
        if column.startswith(common_prefixes + target_prefixes)
    ]

    # Remove exact raw ranges where robust alternatives exist.
    selected = [
        column
        for column in selected
        if not (
            column.endswith("_range")
            and not column.endswith("_robust_range")
        )
    ]

    return sorted(set(selected))


def inverse_video_weights(frame):
    counts = frame.groupby("video_id")["video_id"].transform("count")
    return 1.0 / counts.to_numpy(dtype=float)


def choose_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 181)

    best = None

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, predictions)

        if best is None or score > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(score),
            }

    return best


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_target(all_features, target):
    frame = all_features[
        (all_features["target"] == target)
        & (all_features["pose_frames"] >= 6)
        & (all_features["pose_coverage"] >= 0.40)
    ].copy()

    feature_columns = select_features(frame, target)

    train = frame[frame["split"] == "train"].copy()
    val = frame[frame["split"] == "val"].copy()
    test = frame[frame["split"] == "test"].copy()

    x_train = train[feature_columns]
    y_train = train["label"].astype(int).to_numpy()

    x_val = val[feature_columns]
    y_val = val["label"].astype(int).to_numpy()

    x_test = test[feature_columns]
    y_test = test["label"].astype(int).to_numpy()

    sample_weight = inverse_video_weights(train)

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])

    model.fit(
        x_train,
        y_train,
        classifier__sample_weight=sample_weight,
    )

    val_probabilities = model.predict_proba(x_val)[:, 1]
    threshold_result = choose_threshold(y_val, val_probabilities)
    threshold = threshold_result["threshold"]

    test_probabilities = model.predict_proba(x_test)[:, 1]

    val_metrics = evaluate(y_val, val_probabilities, threshold)
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    package = {
        "target": target,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "model": model,
    }

    model_path = OUTPUT_DIR / f"{target}_baseline.joblib"
    report_path = OUTPUT_DIR / f"{target}_baseline_report.json"

    joblib.dump(package, model_path)

    report = {
        "target": target,
        "feature_count": len(feature_columns),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "validation": val_metrics,
        "test": test_metrics,
    }

    report_path.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 70)
    print(target)
    print("=" * 70)
    print("features:", len(feature_columns))
    print("train / val / test:", len(train), len(val), len(test))
    print("selected threshold:", round(threshold, 4))
    print()
    print("VALIDATION")
    print("confusion:", val_metrics["confusion_matrix"])
    print(
        "balanced accuracy:",
        round(val_metrics["balanced_accuracy"], 4),
    )
    print("ROC-AUC:", round(val_metrics["roc_auc"], 4))
    print("precision:", round(val_metrics["precision"], 4))
    print("recall:", round(val_metrics["recall"], 4))
    print("F1:", round(val_metrics["f1"], 4))
    print()
    print("TEST")
    print("confusion:", test_metrics["confusion_matrix"])
    print(
        "balanced accuracy:",
        round(test_metrics["balanced_accuracy"], 4),
    )
    print("ROC-AUC:", round(test_metrics["roc_auc"], 4))
    print("precision:", round(test_metrics["precision"], 4))
    print("recall:", round(test_metrics["recall"], 4))
    print("F1:", round(test_metrics["f1"], 4))
    print()
    print("saved:", model_path)
    print("report:", report_path)


def main():
    features = pd.read_csv(FEATURES_PATH)

    for target in ("elbow_error", "knee_error"):
        train_target(features, target)


if __name__ == "__main__":
    main()
