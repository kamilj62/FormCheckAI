import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HARD_NEGATIVE_MULTIPLIERS = [1.0, 2.0, 4.0, 6.0, 8.0]


def load_split(split):
    path = BASE / f"knee_v2_{split}.jsonl"

    features = []
    forward_labels = []
    inward_labels = []
    video_ids = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)

            vector = np.asarray(
                row["features"],
                dtype=np.float32,
            )

            if not np.all(np.isfinite(vector)):
                continue

            features.append(vector)
            forward_labels.append(
                int(row["labels"]["knees_forward"])
            )
            inward_labels.append(
                int(row["labels"]["knees_inward"])
            )
            video_ids.append(str(row["video_id"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(forward_labels, dtype=np.int64),
        np.asarray(inward_labels, dtype=np.int64),
        np.asarray(video_ids),
    )


def build_weights(
    forward_labels,
    inward_labels,
    video_ids,
    hard_negative_multiplier,
):
    video_counts = Counter(video_ids.tolist())
    class_counts = Counter(forward_labels.tolist())
    total = len(forward_labels)

    class_weights = {
        label: total / (2.0 * count)
        for label, count in class_counts.items()
    }

    weights = []

    for forward, inward, video_id in zip(
        forward_labels,
        inward_labels,
        video_ids,
    ):
        weight = (
            class_weights[int(forward)]
            / video_counts[str(video_id)]
        )

        if int(forward) == 0 and int(inward) == 1:
            weight *= hard_negative_multiplier

        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float64)
    return weights / np.mean(weights)


def candidate_thresholds(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return np.asarray([0.5])

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    best_f1_threshold = thresholds[
        int(np.nanargmax(f1_values))
    ]

    grid = np.linspace(0.25, 0.80, 112)

    return np.unique(
        np.concatenate([
            thresholds,
            grid,
            [best_f1_threshold],
        ])
    )


def measure(
    forward_labels,
    inward_labels,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    inward_only = (
        (forward_labels == 0)
        & (inward_labels == 1)
    )

    neither = (
        (forward_labels == 0)
        & (inward_labels == 0)
    )

    forward_only = (
        (forward_labels == 1)
        & (inward_labels == 0)
    )

    metrics = {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                forward_labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                forward_labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                forward_labels,
                predictions,
                zero_division=0,
            )
        ),
        "inward_only_false_forward": (
            float(predictions[inward_only].mean())
            if inward_only.any()
            else 0.0
        ),
        "neither_false_forward": (
            float(predictions[neither].mean())
            if neither.any()
            else 0.0
        ),
        "forward_only_recall": (
            float(predictions[forward_only].mean())
            if forward_only.any()
            else 0.0
        ),
    }

    metrics["selection_score"] = (
        metrics["f1"]
        + 0.20 * metrics["forward_only_recall"]
        - 0.45 * metrics["inward_only_false_forward"]
        - 0.20 * metrics["neither_false_forward"]
    )

    return metrics, predictions


def choose_threshold(
    forward_labels,
    inward_labels,
    probabilities,
):
    candidates = []

    for threshold in candidate_thresholds(
        forward_labels,
        probabilities,
    ):
        metrics, _ = measure(
            forward_labels,
            inward_labels,
            probabilities,
            threshold,
        )

        if metrics["recall"] < 0.65:
            continue

        candidates.append(metrics)

    if not candidates:
        raise RuntimeError(
            "No validation threshold retained at least 0.65 recall"
        )

    return max(
        candidates,
        key=lambda row: row["selection_score"],
    )


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("threshold:", round(metrics["threshold"], 4))
    print("precision:", round(metrics["precision"], 4))
    print("recall:", round(metrics["recall"], 4))
    print("f1:", round(metrics["f1"], 4))
    print(
        "forward-only recall:",
        round(metrics["forward_only_recall"], 4),
    )
    print(
        "false-forward on inward-only:",
        round(metrics["inward_only_false_forward"], 4),
    )
    print(
        "false-forward on neither:",
        round(metrics["neither_false_forward"], 4),
    )
    print(
        "selection score:",
        round(metrics["selection_score"], 4),
    )


def main():
    (
        X_train,
        y_forward_train,
        y_inward_train,
        train_video_ids,
    ) = load_split("train")

    (
        X_validation,
        y_forward_validation,
        y_inward_validation,
        _,
    ) = load_split("validation")

    (
        X_test,
        y_forward_test,
        y_inward_test,
        _,
    ) = load_split("test")

    metadata = json.loads(
        (
            BASE / "knee_v2_train_metadata.json"
        ).read_text()
    )
    feature_names = metadata["feature_names"]

    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)
    print("Feature count:", len(feature_names))

    candidates = []

    for multiplier in HARD_NEGATIVE_MULTIPLIERS:
        print("\n" + "=" * 70)
        print("HARD-NEGATIVE MULTIPLIER:", multiplier)
        print("=" * 70)

        sample_weights = build_weights(
            y_forward_train,
            y_inward_train,
            train_video_ids,
            multiplier,
        )

        model = RandomForestClassifier(
            n_estimators=700,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

        model.fit(
            X_train,
            y_forward_train,
            sample_weight=sample_weights,
        )

        validation_probabilities = (
            model.predict_proba(X_validation)[:, 1]
        )

        threshold_metrics = choose_threshold(
            y_forward_validation,
            y_inward_validation,
            validation_probabilities,
        )

        print_metrics(
            "Validation",
            threshold_metrics,
        )

        candidates.append({
            "multiplier": multiplier,
            "model": model,
            "threshold_metrics": threshold_metrics,
        })

    best = max(
        candidates,
        key=lambda row: row[
            "threshold_metrics"
        ]["selection_score"],
    )

    model = best["model"]
    threshold = best[
        "threshold_metrics"
    ]["threshold"]

    print("\n" + "=" * 70)
    print("SELECTED V3 CANDIDATE")
    print("=" * 70)
    print(
        "hard-negative multiplier:",
        best["multiplier"],
    )
    print_metrics(
        "Selected validation metrics",
        best["threshold_metrics"],
    )

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    test_metrics, test_predictions = measure(
        y_forward_test,
        y_inward_test,
        test_probabilities,
        threshold,
    )

    print_metrics(
        "Held-out test metrics",
        test_metrics,
    )

    print(
        "average precision:",
        round(
            average_precision_score(
                y_forward_test,
                test_probabilities,
            ),
            4,
        ),
    )
    print(
        "roc auc:",
        round(
            roc_auc_score(
                y_forward_test,
                test_probabilities,
            ),
            4,
        ),
    )
    print("confusion matrix:")
    print(
        confusion_matrix(
            y_forward_test,
            test_predictions,
        )
    )

    bundle = {
        "version": "v3_hard_negative",
        "target": "knees_forward",
        "model": model,
        "threshold": threshold,
        "hard_negative_multiplier": best["multiplier"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "validation_metrics": best["threshold_metrics"],
        "test_metrics": test_metrics,
        "random_state": 42,
    }

    output = MODEL_DIR / "knees_forward_rf_v3.joblib"
    joblib.dump(bundle, output)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
