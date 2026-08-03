import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

TRAIN_PATH = (
    BASE / "forward_lean_manual_v9_train.jsonl"
)

VALIDATION_PATH = (
    BASE / "forward_lean_manual_v9_validation.jsonl"
)

METADATA_PATH = (
    BASE / "knee_v9_rep_train_metadata.json"
)

MODEL_PATH = (
    BASE / "forward_lean_manual_v9_compact_extratrees.joblib"
)

REPORT_PATH = (
    BASE / "forward_lean_manual_v9_compact_report.json"
)

FEATURE_NAMES = [
    "torso_angle__setup_mean",
    "torso_angle__descent_mean",
    "torso_angle__bottom_mean",
    "torso_angle__bottom_maximum",
    "torso_angle__ascent_mean",
    "torso_angle__ascent_maximum",
    "torso_angle__finish_mean",
    "torso_angle__setup_to_bottom",
    "torso_angle__bottom_to_finish",
    "torso_angle__setup_to_finish",
    "torso_angle__descent_range",
    "torso_angle__ascent_range",
    "torso_angle__bottom_vs_rep_mean",
    "torso_angle__ascent_vs_descent_mean",

    "hip_y__setup_to_bottom",
    "hip_y__bottom_to_finish",
    "hip_y__ascent_range",
    "hip_y__ascent_vs_descent_mean",

    "shoulder_y__setup_to_bottom",
    "shoulder_y__bottom_to_finish",
    "shoulder_y__ascent_range",
    "shoulder_y__ascent_vs_descent_mean",

    "hip_angle__bottom_mean",
    "knee_angle__bottom_mean",
]

RANDOM_STATE = 42


def load_rows(path):
    rows = []

    with path.open() as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def make_matrix(rows, indices):
    x = np.asarray(
        [
            [
                float(row["features"][index])
                for index in indices
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [int(row["target"]) for row in rows],
        dtype=np.int64,
    )

    return x, y


def summarize(y_true, scores, threshold):
    pred = (
        scores >= threshold
    ).astype(np.int64)

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            pred,
            labels=[0, 1],
        ).tolist(),
    }


def main():
    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    all_feature_names = metadata[
        "feature_names"
    ]

    missing = [
        name
        for name in FEATURE_NAMES
        if name not in all_feature_names
    ]

    if missing:
        raise RuntimeError(
            "Missing compact features:\n"
            + "\n".join(missing)
        )

    indices = [
        all_feature_names.index(name)
        for name in FEATURE_NAMES
    ]

    train_rows = load_rows(TRAIN_PATH)
    validation_rows = load_rows(
        VALIDATION_PATH
    )

    x_train, y_train = make_matrix(
        train_rows,
        indices,
    )

    x_validation, y_validation = (
        make_matrix(
            validation_rows,
            indices,
        )
    )

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=3,
        max_features=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    train_scores = model.predict_proba(
        x_train
    )[:, 1]

    validation_scores = model.predict_proba(
        x_validation
    )[:, 1]

    threshold = 0.50

    train_metrics = summarize(
        y_train,
        train_scores,
        threshold,
    )

    validation_metrics = summarize(
        y_validation,
        validation_scores,
        threshold,
    )

    report = {
        "version": (
            "forward_lean_manual_v9_compact_v1"
        ),
        "status": "shadow_only",
        "test_split_used": False,
        "threshold": threshold,
        "feature_count": len(
            FEATURE_NAMES
        ),
        "feature_names": FEATURE_NAMES,
        "train_rows": len(train_rows),
        "validation_rows": len(
            validation_rows
        ),
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_predictions": [
            {
                "candidate_number": row[
                    "candidate_number"
                ],
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
                validation_scores,
                (
                    validation_scores
                    >= threshold
                ).astype(np.int64),
                y_validation,
            )
        ],
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_indices": indices,
            "threshold": threshold,
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "Train shape:",
        x_train.shape,
    )
    print(
        "Validation shape:",
        x_validation.shape,
    )
    print(
        "Train balanced accuracy:",
        round(
            train_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "Validation balanced accuracy:",
        round(
            validation_metrics[
                "balanced_accuracy"
            ],
            4,
        ),
    )
    print(
        "Validation ROC-AUC:",
        round(
            validation_metrics[
                "roc_auc"
            ],
            4,
        ),
    )
    print(
        "Validation confusion matrix:",
        validation_metrics[
            "confusion_matrix"
        ],
    )
    print(
        "Validation precision:",
        round(
            validation_metrics[
                "precision"
            ],
            4,
        ),
    )
    print(
        "Validation recall:",
        round(
            validation_metrics[
                "recall"
            ],
            4,
        ),
    )
    print("Feature count:", len(FEATURE_NAMES))
    print("Test split used: False")
    print("Model:", MODEL_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
