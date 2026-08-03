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

REMOVED_EXACT = {
    "segment_duration_seconds",
    "segment_frame_count",
}

REMOVED_SUFFIXES = (
    "__slope",
    "__delta",
)


def load_split(split):
    path = BASE / f"knee_v6_interval_{split}.jsonl"

    X = []
    forward = []
    inward = []
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
            forward.append(
                float(
                    row["targets"][
                        "forward_fraction"
                    ]
                )
            )
            inward.append(
                float(
                    row["targets"][
                        "inward_fraction"
                    ]
                )
            )
            video_ids.append(
                str(row["video_id"])
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(forward, dtype=np.float64),
        np.asarray(inward, dtype=np.float64),
        np.asarray(video_ids),
    )


def build_feature_mask(feature_names):
    kept_indices = []
    kept_names = []
    removed_names = []

    for index, name in enumerate(feature_names):
        remove = (
            name in REMOVED_EXACT
            or name.endswith(REMOVED_SUFFIXES)
        )

        if remove:
            removed_names.append(name)
        else:
            kept_indices.append(index)
            kept_names.append(name)

    return (
        np.asarray(kept_indices, dtype=np.int64),
        kept_names,
        removed_names,
    )


def build_sample_weights(targets, video_ids):
    video_counts = Counter(video_ids.tolist())
    prevalence = float(np.mean(targets))

    multiplier = (
        min(
            (1.0 - prevalence)
            / max(prevalence, 1e-8),
            12.0,
        )
        if prevalence > 0.0
        else 1.0
    )

    weights = np.asarray(
        [
            (
                1.0
                / video_counts[str(video_id)]
            )
            * (
                1.0
                + multiplier * float(target)
            )
            for target, video_id in zip(
                targets,
                video_ids,
            )
        ],
        dtype=np.float64,
    )

    weights /= np.mean(weights)

    return weights, multiplier


def train_model(X, targets, video_ids):
    weights, multiplier = build_sample_weights(
        targets,
        video_ids,
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
        X,
        targets,
        sample_weight=weights,
    )

    return model, multiplier


def threshold_metrics(binary_true, scores, threshold):
    predicted = (
        scores >= threshold
    ).astype(np.int64)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                binary_true,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                binary_true,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                binary_true,
                predicted,
                zero_division=0,
            )
        ),
    }


def select_threshold(binary_true, scores):
    best = None

    for threshold in np.linspace(
        0.02,
        0.98,
        193,
    ):
        metrics = threshold_metrics(
            binary_true,
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


def evaluate_target(
    name,
    true_fraction,
    scores,
    threshold,
):
    scores = np.clip(scores, 0.0, 1.0)

    binary_true = (
        true_fraction >= 0.5
    ).astype(np.int64)

    predicted = (
        scores >= threshold
    ).astype(np.int64)

    metrics = threshold_metrics(
        binary_true,
        scores,
        threshold,
    )

    metrics.update({
        "mae": float(
            mean_absolute_error(
                true_fraction,
                scores,
            )
        ),
        "rmse": float(
            mean_squared_error(
                true_fraction,
                scores,
            ) ** 0.5
        ),
        "r2": float(
            r2_score(
                true_fraction,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                binary_true,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                binary_true,
                scores,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                binary_true,
                predicted,
                labels=[0, 1],
            ).tolist()
        ),
    })

    print(f"\n{name}")
    print(
        "threshold:",
        round(threshold, 4),
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
        round(metrics["roc_auc"], 4),
    )
    print(
        "average precision:",
        round(
            metrics["average_precision"],
            4,
        ),
    )
    print(
        "MAE:",
        round(metrics["mae"], 4),
    )
    print(
        "R2:",
        round(metrics["r2"], 4),
    )
    print("confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))

    return metrics


def evaluate_joint(
    true_forward_fraction,
    true_inward_fraction,
    forward_scores,
    inward_scores,
    forward_threshold,
    inward_threshold,
):
    true_forward = (
        true_forward_fraction >= 0.5
    )
    true_inward = (
        true_inward_fraction >= 0.5
    )

    predicted_forward = (
        forward_scores >= forward_threshold
    )
    predicted_inward = (
        inward_scores >= inward_threshold
    )

    true_states = (
        true_forward.astype(np.int64)
        + 2 * true_inward.astype(np.int64)
    )

    predicted_states = (
        predicted_forward.astype(np.int64)
        + 2 * predicted_inward.astype(np.int64)
    )

    matrix = confusion_matrix(
        true_states,
        predicted_states,
        labels=[0, 1, 2, 3],
    )

    neither = true_states == 0
    inward_only = true_states == 2
    forward_only = true_states == 1
    both = true_states == 3

    metrics = {
        "confusion_matrix": matrix.tolist(),
        "false_forward_on_neither": (
            float(
                predicted_forward[
                    neither
                ].mean()
            )
            if neither.any()
            else None
        ),
        "false_forward_on_inward_only": (
            float(
                predicted_forward[
                    inward_only
                ].mean()
            )
            if inward_only.any()
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
                    predicted_states[both]
                    == 3
                ).mean()
            )
            if both.any()
            else None
        ),
    }

    print("\nJoint state confusion")
    print(matrix)

    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue

        print(
            key + ":",
            (
                round(value, 4)
                if value is not None
                else "undefined"
            ),
        )

    return metrics


def main():
    metadata = json.loads(
        (
            BASE
            / "knee_v6_interval_train_metadata.json"
        ).read_text()
    )

    original_names = metadata["feature_names"]

    (
        kept_indices,
        kept_names,
        removed_names,
    ) = build_feature_mask(original_names)

    (
        X_train,
        forward_train,
        inward_train,
        train_video_ids,
    ) = load_split("train")

    (
        X_validation,
        forward_validation,
        inward_validation,
        _,
    ) = load_split("validation")

    (
        X_test,
        forward_test,
        inward_test,
        _,
    ) = load_split("test")

    X_train = X_train[:, kept_indices]
    X_validation = X_validation[:, kept_indices]
    X_test = X_test[:, kept_indices]

    print(
        "Original features:",
        len(original_names),
    )
    print(
        "Removed features:",
        len(removed_names),
    )
    print(
        "Kept features:",
        len(kept_names),
    )

    print("\nRemoved examples:")
    for name in removed_names[:30]:
        print(name)

    forward_model, forward_multiplier = (
        train_model(
            X_train,
            forward_train,
            train_video_ids,
        )
    )

    inward_model, inward_multiplier = (
        train_model(
            X_train,
            inward_train,
            train_video_ids,
        )
    )

    forward_validation_scores = np.clip(
        forward_model.predict(X_validation),
        0.0,
        1.0,
    )

    inward_validation_scores = np.clip(
        inward_model.predict(X_validation),
        0.0,
        1.0,
    )

    forward_threshold_result = select_threshold(
        (
            forward_validation >= 0.5
        ).astype(np.int64),
        forward_validation_scores,
    )

    inward_threshold_result = select_threshold(
        (
            inward_validation >= 0.5
        ).astype(np.int64),
        inward_validation_scores,
    )

    forward_threshold = (
        forward_threshold_result["threshold"]
    )
    inward_threshold = (
        inward_threshold_result["threshold"]
    )

    print("\nSelected validation thresholds")
    print(
        "forward:",
        forward_threshold_result,
    )
    print(
        "inward:",
        inward_threshold_result,
    )

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_forward_metrics = (
        evaluate_target(
            "Forward validation",
            forward_validation,
            forward_validation_scores,
            forward_threshold,
        )
    )

    validation_inward_metrics = (
        evaluate_target(
            "Inward validation",
            inward_validation,
            inward_validation_scores,
            inward_threshold,
        )
    )

    validation_joint = evaluate_joint(
        forward_validation,
        inward_validation,
        forward_validation_scores,
        inward_validation_scores,
        forward_threshold,
        inward_threshold,
    )

    forward_test_scores = np.clip(
        forward_model.predict(X_test),
        0.0,
        1.0,
    )

    inward_test_scores = np.clip(
        inward_model.predict(X_test),
        0.0,
        1.0,
    )

    print("\n" + "=" * 70)
    print("EXPLORATORY TEST")
    print("=" * 70)

    test_forward_metrics = evaluate_target(
        "Forward test",
        forward_test,
        forward_test_scores,
        forward_threshold,
    )

    test_inward_metrics = evaluate_target(
        "Inward test",
        inward_test,
        inward_test_scores,
        inward_threshold,
    )

    test_joint = evaluate_joint(
        forward_test,
        inward_test,
        forward_test_scores,
        inward_test_scores,
        forward_threshold,
        inward_threshold,
    )

    bundle = {
        "version": "v7_geometry_ablation",
        "feature_names": kept_names,
        "removed_features": removed_names,
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
            "forward": (
                validation_forward_metrics
            ),
            "inward": (
                validation_inward_metrics
            ),
            "joint": validation_joint,
        },
        "test_exploratory": {
            "forward": test_forward_metrics,
            "inward": test_inward_metrics,
            "joint": test_joint,
        },
    }

    output = (
        MODEL_DIR
        / "knee_interval_geometry_ablation_v7.joblib"
    )

    joblib.dump(bundle, output)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
