import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

TRAIN_PATH = (
    BASE / "forward_lean_manual_v9_train.jsonl"
)

VALIDATION_PATH = (
    BASE / "forward_lean_manual_v9_validation.jsonl"
)

MODEL_PATH = (
    BASE / "forward_lean_manual_v9_extratrees.joblib"
)

REPORT_PATH = (
    BASE / "forward_lean_manual_v9_report.json"
)

RANDOM_STATE = 42


def load_jsonl(path):
    rows = []

    with path.open() as file:
        for line in file:
            rows.append(json.loads(line))

    x = np.asarray(
        [row["features"] for row in rows],
        dtype=np.float64,
    )

    y = np.asarray(
        [row["target"] for row in rows],
        dtype=np.int64,
    )

    return rows, x, y


def metrics(y_true, y_pred, y_score):
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            "clear": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "excessive_forward_lean": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": (
            classification_report(
                y_true,
                y_pred,
                target_names=[
                    "clear",
                    "excessive_forward_lean",
                ],
                zero_division=0,
                output_dict=True,
            )
        ),
    }


def main():
    train_rows, x_train, y_train = load_jsonl(
        TRAIN_PATH
    )

    validation_rows, x_validation, y_validation = (
        load_jsonl(VALIDATION_PATH)
    )

    if x_train.shape[1] != 709:
        raise RuntimeError(
            f"Expected 709 train features, "
            f"found {x_train.shape[1]}"
        )

    if x_validation.shape[1] != 709:
        raise RuntimeError(
            f"Expected 709 validation features, "
            f"found {x_validation.shape[1]}"
        )

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    train_score = model.predict_proba(
        x_train
    )[:, 1]

    validation_score = model.predict_proba(
        x_validation
    )[:, 1]

    threshold = 0.50

    train_pred = (
        train_score >= threshold
    ).astype(np.int64)

    validation_pred = (
        validation_score >= threshold
    ).astype(np.int64)

    majority_class = int(
        np.bincount(y_train).argmax()
    )

    majority_pred = np.full_like(
        y_validation,
        majority_class,
    )

    majority_balanced_accuracy = float(
        balanced_accuracy_score(
            y_validation,
            majority_pred,
        )
    )

    report = {
        "version": "forward_lean_manual_v9_extratrees_v1",
        "status": "shadow_only",
        "test_split_used": False,
        "threshold": threshold,
        "feature_count": int(x_train.shape[1]),
        "train_rows": len(train_rows),
        "validation_rows": len(
            validation_rows
        ),
        "train": metrics(
            y_train,
            train_pred,
            train_score,
        ),
        "validation": metrics(
            y_validation,
            validation_pred,
            validation_score,
        ),
        "majority_baseline": {
            "predicted_class": majority_class,
            "balanced_accuracy": (
                majority_balanced_accuracy
            ),
        },
        "validation_predictions": [
            {
                "candidate_number": row[
                    "candidate_number"
                ],
                "video_id": row["video_id"],
                "review_label": row[
                    "review_label"
                ],
                "review_confidence": row[
                    "review_confidence"
                ],
                "score": float(score),
                "prediction": (
                    "excessive_forward_lean"
                    if prediction == 1
                    else "clear"
                ),
                "correct": bool(
                    prediction == target
                ),
            }
            for (
                row,
                score,
                prediction,
                target,
            ) in zip(
                validation_rows,
                validation_score,
                validation_pred,
                y_validation,
            )
        ],
    }

    joblib.dump(
        model,
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print("Train shape:", x_train.shape)
    print(
        "Validation shape:",
        x_validation.shape,
    )
    print(
        "Train balanced accuracy:",
        round(
            report["train"][
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "Validation balanced accuracy:",
        round(
            report["validation"][
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "Validation ROC-AUC:",
        round(
            report["validation"][
                "roc_auc"
            ],
            4,
        ),
    )
    print(
        "Validation confusion matrix:",
        report["validation"][
            "confusion_matrix"
        ],
    )
    print(
        "Validation excessive precision:",
        round(
            report["validation"][
                "per_class"
            ][
                "excessive_forward_lean"
            ][
                "precision"
            ],
            4,
        ),
    )
    print(
        "Validation excessive recall:",
        round(
            report["validation"][
                "per_class"
            ][
                "excessive_forward_lean"
            ][
                "recall"
            ],
            4,
        ),
    )
    print(
        "Majority balanced accuracy:",
        round(
            majority_balanced_accuracy,
            4,
        ),
    )
    print("Test split used: False")
    print("Model:", MODEL_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
