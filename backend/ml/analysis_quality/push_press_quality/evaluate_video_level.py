from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


BASE = Path("ml/analysis_quality/push_press_quality/results")
FEATURES_PATH = BASE / "push_press_features.csv"
MODELS_DIR = BASE / "models"


def evaluate_target(features, target):
    package = joblib.load(
        MODELS_DIR / f"{target}_baseline.joblib"
    )

    frame = features[
        (features["target"] == target)
        & (features["split"] == "test")
        & (features["pose_frames"] >= 6)
        & (features["pose_coverage"] >= 0.40)
    ].copy()

    columns = package["feature_columns"]
    model = package["model"]
    threshold = float(package["threshold"])

    frame["probability"] = model.predict_proba(
        frame[columns]
    )[:, 1]

    # Any annotated interval makes the source video positive.
    # Maximum probability asks whether any analyzed window detected it.
    video = (
        frame.groupby("video_id", as_index=False)
        .agg(
            label=("label", "max"),
            probability=("probability", "max"),
            window_count=("probability", "size"),
        )
    )

    y_true = video["label"].astype(int).to_numpy()
    probabilities = video["probability"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / max(tn + fp, 1)

    print()
    print("=" * 70)
    print(target)
    print("=" * 70)
    print("test videos:", len(video))
    print("threshold:", round(threshold, 4))
    print("confusion:", matrix.tolist())
    print(
        "balanced accuracy:",
        round(balanced_accuracy_score(y_true, predictions), 4),
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_true, probabilities), 4),
    )
    print("precision:", round(float(precision), 4))
    print("recall:", round(float(recall), 4))
    print("specificity:", round(float(specificity), 4))
    print("F1:", round(float(f1), 4))
    print(
        "windows per video:",
        video["window_count"].describe().round(2).to_dict(),
    )

    output = (
        BASE
        / "models"
        / f"{target}_video_level_test_predictions.csv"
    )
    video.to_csv(output, index=False)
    print("saved:", output)


def main():
    features = pd.read_csv(FEATURES_PATH)

    for target in ("elbow_error", "knee_error"):
        evaluate_target(features, target)


if __name__ == "__main__":
    main()
