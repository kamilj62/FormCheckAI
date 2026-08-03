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


def load_split(split):
    path = BASE / f"knee_v2_{split}.jsonl"

    X = []
    labels = {
        "knees_forward": [],
        "knees_inward": [],
    }
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

            X.append(vector)
            video_ids.append(str(row["video_id"]))

            for target in labels:
                labels[target].append(
                    int(row["labels"][target])
                )

    return (
        np.asarray(X, dtype=np.float32),
        {
            target: np.asarray(values, dtype=np.int64)
            for target, values in labels.items()
        },
        np.asarray(video_ids),
    )


def build_sample_weights(labels, video_ids):
    video_counts = Counter(video_ids.tolist())
    class_counts = Counter(labels.tolist())
    total = len(labels)

    class_weights = {
        label: total / (2.0 * count)
        for label, count in class_counts.items()
    }

    weights = np.asarray(
        [
            class_weights[int(label)]
            / video_counts[str(video_id)]
            for label, video_id in zip(
                labels,
                video_ids,
            )
        ],
        dtype=np.float64,
    )

    return weights / np.mean(weights)


def choose_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    return float(
        thresholds[int(np.nanargmax(f1_values))]
    )


def evaluate(name, labels, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    print(f"\n{name}")
    print("threshold:", round(threshold, 4))
    print(
        "balanced_accuracy:",
        round(
            balanced_accuracy_score(
                labels,
                predictions,
            ),
            4,
        ),
    )
    print(
        "f1:",
        round(
            f1_score(labels, predictions),
            4,
        ),
    )
    print(
        "average_precision:",
        round(
            average_precision_score(
                labels,
                probabilities,
            ),
            4,
        ),
    )
    print(
        "roc_auc:",
        round(
            roc_auc_score(
                labels,
                probabilities,
            ),
            4,
        ),
    )
    print("confusion_matrix:")
    print(confusion_matrix(labels, predictions))
    print("classification_report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "negative",
                "positive",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return predictions


def train_target(
    target,
    X_train,
    y_train,
    train_video_ids,
    X_validation,
    y_validation,
    X_test,
    y_test,
    feature_names,
):
    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)
    print(
        "Train labels:",
        dict(Counter(y_train.tolist())),
    )

    sample_weights = build_sample_weights(
        y_train,
        train_video_ids,
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=16,
        min_samples_leaf=5,
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

    validation_probabilities = (
        model.predict_proba(X_validation)[:, 1]
    )

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

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    test_predictions = evaluate(
        f"{target} test",
        y_test,
        test_probabilities,
        threshold,
    )

    bundle = {
        "version": "v2_geometry_temporal",
        "target": target,
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "random_state": 42,
    }

    output = MODEL_DIR / f"{target}_rf_v2.joblib"
    joblib.dump(bundle, output)

    print("Saved:", output)

    return {
        "probabilities": test_probabilities,
        "predictions": test_predictions,
        "threshold": threshold,
    }


def main():
    X_train, train_labels, train_video_ids = (
        load_split("train")
    )
    X_validation, validation_labels, _ = (
        load_split("validation")
    )
    X_test, test_labels, _ = load_split("test")

    metadata = json.loads(
        (
            BASE / "knee_v2_train_metadata.json"
        ).read_text()
    )
    feature_names = metadata["feature_names"]

    assert X_train.shape[1] == len(feature_names)
    assert X_validation.shape[1] == len(feature_names)
    assert X_test.shape[1] == len(feature_names)

    results = {}

    for target in [
        "knees_forward",
        "knees_inward",
    ]:
        results[target] = train_target(
            target=target,
            X_train=X_train,
            y_train=train_labels[target],
            train_video_ids=train_video_ids,
            X_validation=X_validation,
            y_validation=validation_labels[target],
            X_test=X_test,
            y_test=test_labels[target],
            feature_names=feature_names,
        )

    true_forward = test_labels["knees_forward"]
    true_inward = test_labels["knees_inward"]

    pred_forward = results[
        "knees_forward"
    ]["predictions"]
    pred_inward = results[
        "knees_inward"
    ]["predictions"]

    inward_only = (
        (true_inward == 1)
        & (true_forward == 0)
    )

    neither = (
        (true_inward == 0)
        & (true_forward == 0)
    )

    print("\n" + "=" * 70)
    print("CROSS-LABEL CONFUSION")
    print("=" * 70)

    print(
        "inward-only rows:",
        int(inward_only.sum()),
    )

    if inward_only.any():
        print(
            "false-forward ratio on inward-only:",
            round(
                float(
                    pred_forward[inward_only].mean()
                ),
                4,
            ),
        )
        print(
            "correct-inward ratio on inward-only:",
            round(
                float(
                    pred_inward[inward_only].mean()
                ),
                4,
            ),
        )

    print(
        "neither rows:",
        int(neither.sum()),
    )

    if neither.any():
        print(
            "false-forward ratio on neither:",
            round(
                float(
                    pred_forward[neither].mean()
                ),
                4,
            ),
        )
        print(
            "false-inward ratio on neither:",
            round(
                float(
                    pred_inward[neither].mean()
                ),
                4,
            ),
        )


if __name__ == "__main__":
    main()
