from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


BASE = Path("ml/analysis_quality/push_press_quality/results")
FEATURES_PATH = BASE / "push_press_features.csv"
MODELS_DIR = BASE / "models"


def aggregate_video_predictions(features, target, split):
    package = joblib.load(
        MODELS_DIR / f"{target}_baseline.joblib"
    )

    frame = features[
        (features["target"] == target)
        & (features["split"] == split)
        & (features["pose_frames"] >= 6)
        & (features["pose_coverage"] >= 0.40)
    ].copy()

    frame["probability"] = package["model"].predict_proba(
        frame[package["feature_columns"]]
    )[:, 1]

    return (
        frame.groupby("video_id", as_index=False)
        .agg(
            label=("label", "max"),
            probability=("probability", "max"),
        )
    )


def metrics(video, threshold):
    y_true = video["label"].astype(int).to_numpy()
    probabilities = video["probability"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "confusion": matrix.tolist(),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def select_for_specificity(video, minimum_specificity):
    candidates = []

    for threshold in np.linspace(0.05, 0.95, 181):
        result = metrics(video, threshold)

        if result["specificity"] >= minimum_specificity:
            candidates.append(result)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda result: (
            result["balanced_accuracy"],
            result["recall"],
        ),
    )


def print_result(title, result):
    print(title)
    print("threshold:", round(result["threshold"], 4))
    print("confusion:", result["confusion"])
    print(
        "balanced accuracy:",
        round(result["balanced_accuracy"], 4),
    )
    print("precision:", round(result["precision"], 4))
    print("recall:", round(result["recall"], 4))
    print("specificity:", round(result["specificity"], 4))
    print("F1:", round(result["f1"], 4))


def main():
    features = pd.read_csv(FEATURES_PATH)

    elbow_val = aggregate_video_predictions(
        features,
        "elbow_error",
        "val",
    )
    elbow_test = aggregate_video_predictions(
        features,
        "elbow_error",
        "test",
    )

    for minimum_specificity in (0.85, 0.88, 0.90, 0.92):
        selected = select_for_specificity(
            elbow_val,
            minimum_specificity,
        )

        print()
        print("=" * 70)
        print(
            "ELBOW TARGET SPECIFICITY",
            minimum_specificity,
        )
        print("=" * 70)

        if selected is None:
            print("No threshold found")
            continue

        print_result("VALIDATION", selected)

        test_result = metrics(
            elbow_test,
            selected["threshold"],
        )

        print()
        print_result("TEST", test_result)


if __name__ == "__main__":
    main()
