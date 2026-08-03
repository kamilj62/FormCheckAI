import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = [
    "forward_fraction",
    "inward_fraction",
]


def load_split(split):
    path = BASE / f"knee_v6_interval_{split}.jsonl"

    features = []
    forward_targets = []
    inward_targets = []
    video_ids = []
    records = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)

            vector = np.asarray(
                row["features"],
                dtype=np.float32,
            )

            if not np.all(np.isfinite(vector)):
                continue

            targets = row["targets"]

            features.append(vector)
            forward_targets.append(
                float(targets["forward_fraction"])
            )
            inward_targets.append(
                float(targets["inward_fraction"])
            )
            video_ids.append(str(row["video_id"]))
            records.append({
                "video_id": str(row["video_id"]),
                "segment_index": int(
                    row["segment_index"]
                ),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "frame_count": int(row["frame_count"]),
            })

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(
            forward_targets,
            dtype=np.float64,
        ),
        np.asarray(
            inward_targets,
            dtype=np.float64,
        ),
        np.asarray(video_ids),
        records,
    )


def build_sample_weights(targets, video_ids):
    """
    Give each source video approximately equal influence and
    increase the influence of intervals containing positive labels.

    The positive multiplier is derived from the training prevalence
    and capped to avoid extreme weighting.
    """
    video_counts = Counter(video_ids.tolist())

    prevalence = float(np.mean(targets))

    if prevalence <= 0.0:
        positive_multiplier = 1.0
    else:
        positive_multiplier = min(
            (1.0 - prevalence) / prevalence,
            12.0,
        )

    weights = []

    for target, video_id in zip(
        targets,
        video_ids,
    ):
        video_weight = (
            1.0 / video_counts[str(video_id)]
        )

        target_weight = (
            1.0
            + positive_multiplier * float(target)
        )

        weights.append(
            video_weight * target_weight
        )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    weights /= np.mean(weights)

    return weights, positive_multiplier


def safe_roc_auc(true_binary, scores):
    if len(np.unique(true_binary)) < 2:
        return None

    return float(
        roc_auc_score(true_binary, scores)
    )


def safe_average_precision(true_binary, scores):
    if np.sum(true_binary) == 0:
        return None

    return float(
        average_precision_score(
            true_binary,
            scores,
        )
    )


def threshold_metrics(
    true_binary,
    scores,
    threshold,
):
    predicted = (
        np.asarray(scores) >= threshold
    ).astype(np.int64)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                true_binary,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_binary,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_binary,
                predicted,
                zero_division=0,
            )
        ),
        "predicted_positive_count": int(
            np.sum(predicted)
        ),
        "true_positive_count": int(
            np.sum(true_binary)
        ),
    }


def select_threshold(true_binary, scores):
    """
    Select on validation only.

    Optimize F1, then precision, then prefer the higher threshold
    when metrics tie.
    """
    best = None

    for threshold in np.linspace(
        0.02,
        0.98,
        193,
    ):
        metrics = threshold_metrics(
            true_binary,
            scores,
            float(threshold),
        )

        key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["threshold"],
        )

        if best is None or key > best["key"]:
            best = {
                "key": key,
                "metrics": metrics,
            }

    return best["metrics"]


def print_regression_metrics(
    name,
    true_values,
    predictions,
):
    predictions = np.clip(
        predictions,
        0.0,
        1.0,
    )

    print(f"\n{name} regression")
    print(
        "MAE:",
        round(
            mean_absolute_error(
                true_values,
                predictions,
            ),
            4,
        ),
    )
    print(
        "RMSE:",
        round(
            mean_squared_error(
                true_values,
                predictions,
            ) ** 0.5,
            4,
        ),
    )
    print(
        "R2:",
        round(
            r2_score(
                true_values,
                predictions,
            ),
            4,
        ),
    )

    return {
        "mae": float(
            mean_absolute_error(
                true_values,
                predictions,
            )
        ),
        "rmse": float(
            mean_squared_error(
                true_values,
                predictions,
            ) ** 0.5
        ),
        "r2": float(
            r2_score(
                true_values,
                predictions,
            )
        ),
    }


def print_binary_evaluation(
    name,
    true_fraction,
    predictions,
    threshold,
):
    true_majority = (
        true_fraction >= 0.5
    ).astype(np.int64)

    metrics = threshold_metrics(
        true_majority,
        predictions,
        threshold,
    )

    auc = safe_roc_auc(
        true_majority,
        predictions,
    )

    average_precision = safe_average_precision(
        true_majority,
        predictions,
    )

    predicted = (
        predictions >= threshold
    ).astype(np.int64)

    print(f"\n{name} majority classification")
    print(
        "threshold:",
        round(threshold, 4),
    )
    print(
        "true positives:",
        int(np.sum(true_majority)),
    )
    print(
        "predicted positives:",
        int(np.sum(predicted)),
    )
    print(
        "precision:",
        round(metrics["precision"], 4),
    )
    print(
        "recall:",
        round(metrics["recall"], 4),
    )
    print(
        "F1:",
        round(metrics["f1"], 4),
    )
    print(
        "ROC-AUC:",
        (
            round(auc, 4)
            if auc is not None
            else "undefined"
        ),
    )
    print(
        "average precision:",
        (
            round(average_precision, 4)
            if average_precision is not None
            else "undefined"
        ),
    )
    print("confusion matrix:")
    print(
        confusion_matrix(
            true_majority,
            predicted,
            labels=[0, 1],
        )
    )

    metrics["roc_auc"] = auc
    metrics["average_precision"] = (
        average_precision
    )

    return metrics


def joint_state(
    forward_binary,
    inward_binary,
):
    forward_binary = np.asarray(
        forward_binary,
        dtype=np.int64,
    )
    inward_binary = np.asarray(
        inward_binary,
        dtype=np.int64,
    )

    return (
        forward_binary
        + 2 * inward_binary
    )


def evaluate_joint_confusion(
    true_forward_fraction,
    true_inward_fraction,
    predicted_forward_fraction,
    predicted_inward_fraction,
    forward_threshold,
    inward_threshold,
):
    true_forward = (
        true_forward_fraction >= 0.5
    ).astype(np.int64)

    true_inward = (
        true_inward_fraction >= 0.5
    ).astype(np.int64)

    predicted_forward = (
        predicted_forward_fraction
        >= forward_threshold
    ).astype(np.int64)

    predicted_inward = (
        predicted_inward_fraction
        >= inward_threshold
    ).astype(np.int64)

    true_states = joint_state(
        true_forward,
        true_inward,
    )
    predicted_states = joint_state(
        predicted_forward,
        predicted_inward,
    )

    state_names = [
        "neither",
        "forward_only",
        "inward_only",
        "both",
    ]

    print("\nJoint majority-state confusion")
    print("rows=true, columns=predicted")
    print("order:", state_names)

    matrix = confusion_matrix(
        true_states,
        predicted_states,
        labels=[0, 1, 2, 3],
    )

    print(matrix)

    inward_only = true_states == 2
    neither = true_states == 0
    forward_only = true_states == 1
    both = true_states == 3

    results = {
        "confusion_matrix": matrix.tolist(),
        "false_forward_on_inward_only": (
            float(
                predicted_forward[
                    inward_only
                ].mean()
            )
            if inward_only.any()
            else None
        ),
        "false_forward_on_neither": (
            float(
                predicted_forward[
                    neither
                ].mean()
            )
            if neither.any()
            else None
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
            else None
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
            else None
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
            else None
        ),
        "true_state_counts": {
            state_names[state]: int(
                np.sum(true_states == state)
            )
            for state in range(4)
        },
    }

    print("\nCritical joint metrics")

    for key in [
        "false_forward_on_inward_only",
        "false_forward_on_neither",
        "correct_inward_only_state",
        "correct_forward_only_state",
        "correct_both_state",
    ]:
        value = results[key]

        print(
            key.replace("_", " ") + ":",
            (
                round(value, 4)
                if value is not None
                else "undefined"
            ),
        )

    print(
        "true state counts:",
        results["true_state_counts"],
    )

    return results


def train_model(
    name,
    X_train,
    y_train,
    train_video_ids,
):
    sample_weights, multiplier = (
        build_sample_weights(
            y_train,
            train_video_ids,
        )
    )

    print(f"\nTraining {name}")
    print(
        "target mean:",
        round(float(np.mean(y_train)), 4),
    )
    print(
        "positive weight multiplier:",
        round(multiplier, 4),
    )

    model = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=18,
        min_samples_leaf=4,
        max_features=0.65,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    return model, multiplier


def main():
    (
        X_train,
        y_forward_train,
        y_inward_train,
        train_video_ids,
        _,
    ) = load_split("train")

    (
        X_validation,
        y_forward_validation,
        y_inward_validation,
        _,
        _,
    ) = load_split("validation")

    (
        X_test,
        y_forward_test,
        y_inward_test,
        _,
        test_records,
    ) = load_split("test")

    metadata = json.loads(
        (
            BASE
            / "knee_v6_interval_train_metadata.json"
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

    print("\nMajority-label counts")

    for split_name, forward, inward in [
        (
            "train",
            y_forward_train,
            y_inward_train,
        ),
        (
            "validation",
            y_forward_validation,
            y_inward_validation,
        ),
        (
            "test",
            y_forward_test,
            y_inward_test,
        ),
    ]:
        print(
            split_name,
            {
                "forward_majority": int(
                    np.sum(forward >= 0.5)
                ),
                "inward_majority": int(
                    np.sum(inward >= 0.5)
                ),
            },
        )

    forward_model, forward_multiplier = (
        train_model(
            "forward-fraction model",
            X_train,
            y_forward_train,
            train_video_ids,
        )
    )

    inward_model, inward_multiplier = (
        train_model(
            "inward-fraction model",
            X_train,
            y_inward_train,
            train_video_ids,
        )
    )

    forward_validation_predictions = np.clip(
        forward_model.predict(X_validation),
        0.0,
        1.0,
    )

    inward_validation_predictions = np.clip(
        inward_model.predict(X_validation),
        0.0,
        1.0,
    )

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    forward_validation_regression = (
        print_regression_metrics(
            "Forward",
            y_forward_validation,
            forward_validation_predictions,
        )
    )

    inward_validation_regression = (
        print_regression_metrics(
            "Inward",
            y_inward_validation,
            inward_validation_predictions,
        )
    )

    forward_threshold_metrics = (
        select_threshold(
            (
                y_forward_validation >= 0.5
            ).astype(np.int64),
            forward_validation_predictions,
        )
    )

    inward_threshold_metrics = (
        select_threshold(
            (
                y_inward_validation >= 0.5
            ).astype(np.int64),
            inward_validation_predictions,
        )
    )

    forward_threshold = (
        forward_threshold_metrics["threshold"]
    )
    inward_threshold = (
        inward_threshold_metrics["threshold"]
    )

    print("\nSelected validation thresholds")
    print(
        "forward threshold:",
        round(forward_threshold, 4),
        forward_threshold_metrics,
    )
    print(
        "inward threshold:",
        round(inward_threshold, 4),
        inward_threshold_metrics,
    )

    forward_validation_classification = (
        print_binary_evaluation(
            "Forward validation",
            y_forward_validation,
            forward_validation_predictions,
            forward_threshold,
        )
    )

    inward_validation_classification = (
        print_binary_evaluation(
            "Inward validation",
            y_inward_validation,
            inward_validation_predictions,
            inward_threshold,
        )
    )

    validation_joint = evaluate_joint_confusion(
        y_forward_validation,
        y_inward_validation,
        forward_validation_predictions,
        inward_validation_predictions,
        forward_threshold,
        inward_threshold,
    )

    # Test is evaluated only after validation thresholds are fixed.
    forward_test_predictions = np.clip(
        forward_model.predict(X_test),
        0.0,
        1.0,
    )

    inward_test_predictions = np.clip(
        inward_model.predict(X_test),
        0.0,
        1.0,
    )

    print("\n" + "=" * 70)
    print("LOCKED TEST")
    print("=" * 70)

    forward_test_regression = (
        print_regression_metrics(
            "Forward",
            y_forward_test,
            forward_test_predictions,
        )
    )

    inward_test_regression = (
        print_regression_metrics(
            "Inward",
            y_inward_test,
            inward_test_predictions,
        )
    )

    forward_test_classification = (
        print_binary_evaluation(
            "Forward test",
            y_forward_test,
            forward_test_predictions,
            forward_threshold,
        )
    )

    inward_test_classification = (
        print_binary_evaluation(
            "Inward test",
            y_inward_test,
            inward_test_predictions,
            inward_threshold,
        )
    )

    test_joint = evaluate_joint_confusion(
        y_forward_test,
        y_inward_test,
        forward_test_predictions,
        inward_test_predictions,
        forward_threshold,
        inward_threshold,
    )

    bundle = {
        "version": "v6_interval_soft_target_regression",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "forward_model": forward_model,
        "inward_model": inward_model,
        "forward_threshold": forward_threshold,
        "inward_threshold": inward_threshold,
        "forward_weight_multiplier": (
            forward_multiplier
        ),
        "inward_weight_multiplier": (
            inward_multiplier
        ),
        "validation": {
            "forward_regression": (
                forward_validation_regression
            ),
            "inward_regression": (
                inward_validation_regression
            ),
            "forward_classification": (
                forward_validation_classification
            ),
            "inward_classification": (
                inward_validation_classification
            ),
            "joint": validation_joint,
        },
        "test": {
            "forward_regression": (
                forward_test_regression
            ),
            "inward_regression": (
                inward_test_regression
            ),
            "forward_classification": (
                forward_test_classification
            ),
            "inward_classification": (
                inward_test_classification
            ),
            "joint": test_joint,
        },
    }

    model_path = (
        MODEL_DIR
        / "knee_interval_soft_target_v6.joblib"
    )

    joblib.dump(bundle, model_path)

    prediction_path = (
        BASE / "knee_interval_v6_test_predictions.jsonl"
    )

    with prediction_path.open("w") as out:
        for (
            record,
            true_forward,
            true_inward,
            predicted_forward,
            predicted_inward,
        ) in zip(
            test_records,
            y_forward_test,
            y_inward_test,
            forward_test_predictions,
            inward_test_predictions,
        ):
            output = {
                **record,
                "true_forward_fraction": float(
                    true_forward
                ),
                "true_inward_fraction": float(
                    true_inward
                ),
                "predicted_forward_fraction": float(
                    predicted_forward
                ),
                "predicted_inward_fraction": float(
                    predicted_inward
                ),
                "predicted_forward_majority": int(
                    predicted_forward
                    >= forward_threshold
                ),
                "predicted_inward_majority": int(
                    predicted_inward
                    >= inward_threshold
                ),
            }

            out.write(json.dumps(output) + "\n")

    print("\nSaved model:", model_path)
    print(
        "Saved test predictions:",
        prediction_path,
    )


if __name__ == "__main__":
    main()
