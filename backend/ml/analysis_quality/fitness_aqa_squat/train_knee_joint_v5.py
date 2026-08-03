import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATE_NAMES = [
    "neither",
    "forward_only",
    "inward_only",
    "both",
]


def encode_state(forward, inward):
    forward = int(forward)
    inward = int(inward)

    if forward == 0 and inward == 0:
        return 0
    if forward == 1 and inward == 0:
        return 1
    if forward == 0 and inward == 1:
        return 2
    return 3


def decode_forward(states):
    states = np.asarray(states)
    return np.isin(states, [1, 3]).astype(np.int64)


def decode_inward(states):
    states = np.asarray(states)
    return np.isin(states, [2, 3]).astype(np.int64)


def load_split(split):
    path = BASE / f"knee_v5_{split}.jsonl"

    features = []
    states = []
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

            forward = int(
                row["labels"]["knees_forward"]
            )
            inward = int(
                row["labels"]["knees_inward"]
            )

            features.append(vector)
            states.append(
                encode_state(forward, inward)
            )
            video_ids.append(
                str(row["video_id"])
            )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(states, dtype=np.int64),
        np.asarray(video_ids),
    )


def build_sample_weights(states, video_ids):
    state_counts = Counter(states.tolist())
    video_counts = Counter(video_ids.tolist())
    total = len(states)
    class_count = len(state_counts)

    state_weights = {
        state: total / (class_count * count)
        for state, count in state_counts.items()
    }

    weights = np.asarray(
        [
            state_weights[int(state)]
            / video_counts[str(video_id)]
            for state, video_id in zip(
                states,
                video_ids,
            )
        ],
        dtype=np.float64,
    )

    return weights / np.mean(weights)


def binary_metrics(true_labels, predicted_labels):
    return {
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
    }


def evaluate(name, true_states, predicted_states):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    balanced_accuracy = balanced_accuracy_score(
        true_states,
        predicted_states,
    )

    macro_f1 = f1_score(
        true_states,
        predicted_states,
        average="macro",
        zero_division=0,
    )

    print(
        "balanced accuracy:",
        round(float(balanced_accuracy), 4),
    )
    print(
        "macro F1:",
        round(float(macro_f1), 4),
    )

    matrix = confusion_matrix(
        true_states,
        predicted_states,
        labels=[0, 1, 2, 3],
    )

    print("\nConfusion matrix")
    print("rows=true, columns=predicted")
    print("order:", STATE_NAMES)
    print(matrix)

    print("\nClassification report")
    print(
        classification_report(
            true_states,
            predicted_states,
            labels=[0, 1, 2, 3],
            target_names=STATE_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    true_forward = decode_forward(true_states)
    predicted_forward = decode_forward(
        predicted_states
    )

    true_inward = decode_inward(true_states)
    predicted_inward = decode_inward(
        predicted_states
    )

    forward_metrics = binary_metrics(
        true_forward,
        predicted_forward,
    )

    inward_metrics = binary_metrics(
        true_inward,
        predicted_inward,
    )

    print("\nDerived knees-forward metrics")
    for key, value in forward_metrics.items():
        print(key + ":", round(value, 4))

    print("\nDerived knees-inward metrics")
    for key, value in inward_metrics.items():
        print(key + ":", round(value, 4))

    inward_only = true_states == 2
    neither = true_states == 0
    forward_only = true_states == 1
    both = true_states == 3

    metrics = {
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(macro_f1),
        "forward_precision": (
            forward_metrics["precision"]
        ),
        "forward_recall": (
            forward_metrics["recall"]
        ),
        "forward_f1": forward_metrics["f1"],
        "inward_precision": (
            inward_metrics["precision"]
        ),
        "inward_recall": (
            inward_metrics["recall"]
        ),
        "inward_f1": inward_metrics["f1"],
        "false_forward_on_inward_only": (
            float(
                predicted_forward[
                    inward_only
                ].mean()
            )
            if inward_only.any()
            else 0.0
        ),
        "correct_inward_only_state": (
            float(
                (
                    predicted_states[
                        inward_only
                    ] == 2
                ).mean()
            )
            if inward_only.any()
            else 0.0
        ),
        "false_forward_on_neither": (
            float(
                predicted_forward[
                    neither
                ].mean()
            )
            if neither.any()
            else 0.0
        ),
        "correct_forward_only_state": (
            float(
                (
                    predicted_states[
                        forward_only
                    ] == 1
                ).mean()
            )
            if forward_only.any()
            else 0.0
        ),
        "correct_both_state": (
            float(
                (
                    predicted_states[
                        both
                    ] == 3
                ).mean()
            )
            if both.any()
            else 0.0
        ),
    }

    print("\nCritical joint-state metrics")

    for key in [
        "false_forward_on_inward_only",
        "correct_inward_only_state",
        "false_forward_on_neither",
        "correct_forward_only_state",
        "correct_both_state",
    ]:
        print(
            key.replace("_", " ") + ":",
            round(metrics[key], 4),
        )

    return metrics


def main():
    (
        X_train,
        y_train,
        train_video_ids,
    ) = load_split("train")

    (
        X_validation,
        y_validation,
        _,
    ) = load_split("validation")

    (
        X_test,
        y_test,
        _,
    ) = load_split("test")

    metadata = json.loads(
        (
            BASE / "knee_v5_train_metadata.json"
        ).read_text()
    )

    feature_names = metadata["feature_names"]

    assert X_train.shape[1] == len(
        feature_names
    )
    assert X_validation.shape[1] == len(
        feature_names
    )
    assert X_test.shape[1] == len(
        feature_names
    )

    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)
    print("Feature count:", len(feature_names))

    print("\nTrain states:")
    for state, count in sorted(
        Counter(y_train.tolist()).items()
    ):
        print(STATE_NAMES[state], count)

    sample_weights = build_sample_weights(
        y_train,
        train_video_ids,
    )

    model = RandomForestClassifier(
        n_estimators=900,
        max_depth=18,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining V5 joint model...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    validation_predictions = model.predict(
        X_validation
    )

    validation_metrics = evaluate(
        "V5 VALIDATION",
        y_validation,
        validation_predictions,
    )

    test_predictions = model.predict(X_test)

    test_metrics = evaluate(
        "V5 HELD-OUT TEST",
        y_test,
        test_predictions,
    )

    bundle = {
        "version": "v5_joint_robust_geometry",
        "model": model,
        "state_names": STATE_NAMES,
        "state_encoding": {
            "neither": 0,
            "forward_only": 1,
            "inward_only": 2,
            "both": 3,
        },
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": test_metrics,
        "random_state": 42,
    }

    output = (
        MODEL_DIR
        / "knee_joint_robust_geometry_rf_v5.joblib"
    )

    joblib.dump(bundle, output)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
