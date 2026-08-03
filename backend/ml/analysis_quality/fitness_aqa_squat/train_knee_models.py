import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BIOMECHANICS_KEYS = [
    "knee_angle",
    "hip_angle",
    "torso_angle",
    "hip_y",
    "knee_y",
    "shoulder_y",
    "hip_x",
    "knee_x",
    "shoulder_x",
    "shoulder_width_x",
    "hip_width_x",
    "knee_width_x",
    "ankle_width_x",
    "shoulder_hip_distance",
    "hip_knee_distance",
]


def load_split(split_name, target):
    path = BASE / f"knee_pose_{split_name}.jsonl"

    features = []
    labels = []
    video_ids = []

    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)

            raw_features = row.get("features", [])
            biomechanics = row.get("biomechanics", {})

            if len(raw_features) != 68:
                print(
                    f"Skipping {path.name}:{line_number}: "
                    f"expected 68 features, got {len(raw_features)}"
                )
                continue

            vector = [float(value) for value in raw_features]

            vector.extend(
                float(biomechanics.get(key, 0.0) or 0.0)
                for key in BIOMECHANICS_KEYS
            )

            vector = np.asarray(vector, dtype=np.float32)

            if not np.all(np.isfinite(vector)):
                continue

            features.append(vector)
            labels.append(int(row["labels"][target]))
            video_ids.append(str(row["video_id"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(video_ids),
    )


def build_sample_weights(labels, video_ids):
    """
    Give each video approximately equal total weight, then compensate
    for class imbalance without allowing long videos to dominate.
    """
    video_counts = Counter(video_ids.tolist())
    class_counts = Counter(labels.tolist())

    total = len(labels)
    class_weights = {
        label: total / (2.0 * count)
        for label, count in class_counts.items()
    }

    weights = np.asarray(
        [
            class_weights[int(label)] / video_counts[str(video_id)]
            for label, video_id in zip(labels, video_ids)
        ],
        dtype=np.float64,
    )

    return weights / np.mean(weights)


def choose_threshold(labels, probabilities):
    """
    Select the threshold with the best validation F1 score.
    Test data is never used for threshold selection.
    """
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def evaluate(name, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)

    print(f"\n{name}")
    print("threshold:", round(threshold, 4))
    print(
        "balanced_accuracy:",
        round(balanced_accuracy_score(labels, predictions), 4),
    )
    print("f1:", round(f1_score(labels, predictions), 4))
    print(
        "average_precision:",
        round(average_precision_score(labels, probabilities), 4),
    )

    if len(np.unique(labels)) == 2:
        print(
            "roc_auc:",
            round(roc_auc_score(labels, probabilities), 4),
        )

    print("confusion_matrix:")
    print(confusion_matrix(labels, predictions))

    print("classification_report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )


def train_target(target):
    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    X_train, y_train, train_video_ids = load_split("train", target)
    X_validation, y_validation, _ = load_split("validation", target)
    X_test, y_test, _ = load_split("test", target)

    print("Feature count:", X_train.shape[1])
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)
    print("Test shape:", X_test.shape)
    print("Train labels:", dict(Counter(y_train.tolist())))
    print("Validation labels:", dict(Counter(y_validation.tolist())))
    print("Test labels:", dict(Counter(y_test.tolist())))

    sample_weights = build_sample_weights(
        y_train,
        train_video_ids,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = choose_threshold(
        y_validation,
        validation_probabilities,
    )

    evaluate(
        f"{target} validation",
        y_validation,
        validation_probabilities,
        threshold,
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]

    evaluate(
        f"{target} test",
        y_test,
        test_probabilities,
        threshold,
    )

    bundle = {
        "target": target,
        "model": model,
        "threshold": threshold,
        "raw_feature_count": 68,
        "biomechanics_keys": BIOMECHANICS_KEYS,
        "total_feature_count": 68 + len(BIOMECHANICS_KEYS),
        "random_state": 42,
    }

    output_path = MODEL_DIR / f"{target}_rf_v1.joblib"
    joblib.dump(bundle, output_path)

    print("Saved:", output_path)


def main():
    train_target("knees_forward")
    train_target("knees_inward")


if __name__ == "__main__":
    main()
