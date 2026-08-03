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


def load_split(split):
    data_path = BASE / f"knee_v10_rep_{split}.jsonl"
    metadata_path = BASE / f"knee_v10_rep_{split}_metadata.json"

    metadata = json.loads(metadata_path.read_text())

    rows = [
        json.loads(line)
        for line in data_path.open()
    ]

    X = np.asarray(
        [row["features"] for row in rows],
        dtype=np.float32,
    )

    forward = np.asarray(
        [
            row["targets"]["ascent_forward_fraction"]
            for row in rows
        ],
        dtype=np.float64,
    )

    inward = np.asarray(
        [
            row["targets"]["ascent_inward_fraction"]
            for row in rows
        ],
        dtype=np.float64,
    )

    video_ids = np.asarray(
        [str(row["video_id"]) for row in rows]
    )

    records = [
        {
            "video_id": str(row["video_id"]),
            "rep_index": int(row["rep_index"]),
            "start_frame": int(row["start_frame"]),
            "bottom_frame": int(row["bottom_frame"]),
            "end_frame": int(row["end_frame"]),
        }
        for row in rows
    ]

    if X.shape[1] != len(metadata["feature_names"]):
        raise RuntimeError(
            f"{split} feature count mismatch: "
            f"{X.shape[1]} != {len(metadata['feature_names'])}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"{split} contains non-finite features")

    return {
        "X": X,
        "forward": forward,
        "inward": inward,
        "video_ids": video_ids,
        "records": records,
        "feature_names": metadata["feature_names"],
    }


def build_sample_weights(targets, video_ids):
    counts = Counter(video_ids.tolist())
    prevalence = float(np.mean(targets))

    positive_multiplier = (
        min(
            (1.0 - prevalence) / max(prevalence, 1e-8),
            12.0,
        )
        if prevalence > 0.0
        else 1.0
    )

    weights = np.asarray(
        [
            (
                1.0 / counts[str(video_id)]
            )
            * (
                1.0
                + positive_multiplier * float(target)
            )
            for target, video_id in zip(targets, video_ids)
        ],
        dtype=np.float64,
    )

    weights /= np.mean(weights)

    return weights, positive_multiplier


def train_model(X, targets, video_ids):
    weights, multiplier = build_sample_weights(
        targets,
        video_ids,
    )

    model = ExtraTreesRegressor(
        n_estimators=1200,
        max_depth=20,
        min_samples_leaf=4,
        max_features=0.55,
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


def binary_metrics(labels, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
    }


def select_threshold(labels, scores):
    best = None

    for threshold in np.linspace(0.02, 0.98, 193):
        metrics = binary_metrics(
            labels,
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
    title,
    fractions,
    scores,
    threshold,
):
    scores = np.clip(scores, 0.0, 1.0)

    labels = (
        fractions >= 0.5
    ).astype(np.int64)

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    metrics = binary_metrics(
        labels,
        scores,
        threshold,
    )

    metrics.update({
        "mae": float(
            mean_absolute_error(
                fractions,
                scores,
            )
        ),
        "rmse": float(
            mean_squared_error(
                fractions,
                scores,
            ) ** 0.5
        ),
        "r2": float(
            r2_score(
                fractions,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    })

    print(f"\n{title}")
    print("threshold:", round(threshold, 4))
    print("precision:", round(metrics["precision"], 4))
    print("recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print(
        "average precision:",
        round(metrics["average_precision"], 4),
    )
    print("MAE:", round(metrics["mae"], 4))
    print("R2:", round(metrics["r2"], 4))
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
    true_forward = true_forward_fraction >= 0.5
    true_inward = true_inward_fraction >= 0.5

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
    forward_only = true_states == 1
    inward_only = true_states == 2
    both = true_states == 3

    metrics = {
        "confusion_matrix": matrix.tolist(),
        "false_forward_on_neither": (
            float(predicted_forward[neither].mean())
            if neither.any()
            else None
        ),
        "false_forward_on_inward_only": (
            float(predicted_forward[inward_only].mean())
            if inward_only.any()
            else None
        ),
        "correct_neither_state": (
            float((predicted_states[neither] == 0).mean())
            if neither.any()
            else None
        ),
        "correct_forward_only_state": (
            float((predicted_states[forward_only] == 1).mean())
            if forward_only.any()
            else None
        ),
        "correct_inward_only_state": (
            float((predicted_states[inward_only] == 2).mean())
            if inward_only.any()
            else None
        ),
        "correct_both_state": (
            float((predicted_states[both] == 3).mean())
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
            round(value, 4)
            if value is not None
            else "undefined",
        )

    return metrics


def write_predictions(
    path,
    records,
    true_forward,
    true_inward,
    forward_scores,
    inward_scores,
    forward_threshold,
    inward_threshold,
):
    with path.open("w") as output:
        for (
            record,
            forward_target,
            inward_target,
            forward_score,
            inward_score,
        ) in zip(
            records,
            true_forward,
            true_inward,
            forward_scores,
            inward_scores,
        ):
            output.write(
                json.dumps({
                    **record,
                    "true_ascent_forward_fraction": float(
                        forward_target
                    ),
                    "true_ascent_inward_fraction": float(
                        inward_target
                    ),
                    "predicted_ascent_forward_fraction": float(
                        forward_score
                    ),
                    "predicted_ascent_inward_fraction": float(
                        inward_score
                    ),
                    "predicted_forward": int(
                        forward_score >= forward_threshold
                    ),
                    "predicted_inward": int(
                        inward_score >= inward_threshold
                    ),
                })
                + "\n"
            )


def main():
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    if not (
        train["feature_names"]
        == validation["feature_names"]
        == test["feature_names"]
    ):
        raise RuntimeError(
            "V10 feature ordering differs across splits"
        )

    print("Train shape:", train["X"].shape)
    print("Validation shape:", validation["X"].shape)
    print("Test shape:", test["X"].shape)

    forward_model, forward_multiplier = train_model(
        train["X"],
        train["forward"],
        train["video_ids"],
    )

    inward_model, inward_multiplier = train_model(
        train["X"],
        train["inward"],
        train["video_ids"],
    )

    validation_forward_scores = np.clip(
        forward_model.predict(validation["X"]),
        0.0,
        1.0,
    )

    validation_inward_scores = np.clip(
        inward_model.predict(validation["X"]),
        0.0,
        1.0,
    )

    forward_threshold_result = select_threshold(
        (validation["forward"] >= 0.5).astype(np.int64),
        validation_forward_scores,
    )

    inward_threshold_result = select_threshold(
        (validation["inward"] >= 0.5).astype(np.int64),
        validation_inward_scores,
    )

    forward_threshold = float(
        forward_threshold_result["threshold"]
    )

    inward_threshold = float(
        inward_threshold_result["threshold"]
    )

    print("\nSelected validation thresholds")
    print("forward:", forward_threshold_result)
    print("inward:", inward_threshold_result)

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_forward_metrics = evaluate_target(
        "Forward validation",
        validation["forward"],
        validation_forward_scores,
        forward_threshold,
    )

    validation_inward_metrics = evaluate_target(
        "Inward validation",
        validation["inward"],
        validation_inward_scores,
        inward_threshold,
    )

    validation_joint = evaluate_joint(
        validation["forward"],
        validation["inward"],
        validation_forward_scores,
        validation_inward_scores,
        forward_threshold,
        inward_threshold,
    )

    test_forward_scores = np.clip(
        forward_model.predict(test["X"]),
        0.0,
        1.0,
    )

    test_inward_scores = np.clip(
        inward_model.predict(test["X"]),
        0.0,
        1.0,
    )

    print("\n" + "=" * 70)
    print("EXPLORATORY TEST")
    print("=" * 70)

    test_forward_metrics = evaluate_target(
        "Forward test",
        test["forward"],
        test_forward_scores,
        forward_threshold,
    )

    test_inward_metrics = evaluate_target(
        "Inward test",
        test["inward"],
        test_inward_scores,
        inward_threshold,
    )

    test_joint = evaluate_joint(
        test["forward"],
        test["inward"],
        test_forward_scores,
        test_inward_scores,
        forward_threshold,
        inward_threshold,
    )

    bundle = {
        "version": "v10_fixed_phase_resampled_ascent_targets",
        "feature_names": train["feature_names"],
        "forward_model": forward_model,
        "inward_model": inward_model,
        "forward_threshold": forward_threshold,
        "inward_threshold": inward_threshold,
        "forward_weight_multiplier": forward_multiplier,
        "inward_weight_multiplier": inward_multiplier,
        "validation": {
            "forward": validation_forward_metrics,
            "inward": validation_inward_metrics,
            "joint": validation_joint,
        },
        "test_exploratory": {
            "forward": test_forward_metrics,
            "inward": test_inward_metrics,
            "joint": test_joint,
        },
    }

    model_path = (
        MODEL_DIR / "knee_complete_rep_v10.joblib"
    )

    joblib.dump(bundle, model_path)

    write_predictions(
        BASE / "knee_rep_v10_validation_predictions.jsonl",
        validation["records"],
        validation["forward"],
        validation["inward"],
        validation_forward_scores,
        validation_inward_scores,
        forward_threshold,
        inward_threshold,
    )

    write_predictions(
        BASE / "knee_rep_v10_test_predictions.jsonl",
        test["records"],
        test["forward"],
        test["inward"],
        test_forward_scores,
        test_inward_scores,
        forward_threshold,
        inward_threshold,
    )

    print("\nSaved model:", model_path)


if __name__ == "__main__":
    main()
