from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.feature_engine.movement_video_features_v3 import FEATURE_NAMES


DATA_PATH = Path(
    "ml/router_rebuild_v1/features/router_features_v3.csv"
)

MODEL_OUT = Path(
    "app/models/candidates/"
    "olympic_router_hierarchical_v1_temporal.joblib"
)

REPORT_OUT = Path(
    "ml/router_rebuild_v1/reports/"
    "olympic_router_hierarchical_v1_metrics.json"
)

PREDICTIONS_OUT = Path(
    "ml/router_rebuild_v1/reports/"
    "olympic_router_hierarchical_v1_predictions.csv"
)

OLY_LABELS = [
    "clean",
    "clean_and_jerk",
    "snatch",
    "split_jerk",
]

RANDOM_STATE = 42


def make_model():
    return RandomForestClassifier(
        n_estimators=1500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_binary(frame, target):
    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = target(frame).astype(str).to_numpy()

    model = make_model()
    model.fit(X, y)
    return model


def predict_hierarchy(models, frame):
    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)

    stage1 = models["snatch_gate"].predict(X)

    predictions = []
    confidences = []
    routes = []

    for row_index, first_decision in enumerate(stage1):
        x = X[row_index : row_index + 1]

        if first_decision == "snatch":
            probabilities = models["snatch_gate"].predict_proba(x)[0]
            classes = list(models["snatch_gate"].classes_)
            confidence = float(
                probabilities[classes.index("snatch")]
            )

            predictions.append("snatch")
            confidences.append(confidence)
            routes.append("snatch_gate")
            continue

        second_decision = models["clean_gate"].predict(x)[0]

        if second_decision == "clean":
            probabilities = models["clean_gate"].predict_proba(x)[0]
            classes = list(models["clean_gate"].classes_)
            confidence = float(
                probabilities[classes.index("clean")]
            )

            predictions.append("clean")
            confidences.append(confidence)
            routes.append("rack_to_clean")
            continue

        final_decision = models["jerk_router"].predict(x)[0]
        probabilities = models["jerk_router"].predict_proba(x)[0]
        classes = list(models["jerk_router"].classes_)

        confidence = float(
            probabilities[classes.index(final_decision)]
        )

        predictions.append(str(final_decision))
        confidences.append(confidence)
        routes.append("cj_vs_split_jerk")

    return (
        np.asarray(predictions),
        np.asarray(confidences, dtype=float),
        np.asarray(routes),
    )


def evaluate(models, frame, split_name):
    y = frame["reviewed_label"].astype(str).to_numpy()

    pred, confidence, route = predict_hierarchy(
        models,
        frame,
    )

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(
        y,
        pred,
        labels=OLY_LABELS,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        pred,
        labels=OLY_LABELS,
    )

    print("\n" + "=" * 80)
    print(split_name.upper())
    print("=" * 80)
    print("rows:", len(frame))
    print("accuracy:", round(accuracy, 4))
    print("macro F1:", round(macro_f1, 4))

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y,
            pred,
            labels=OLY_LABELS,
            zero_division=0,
        )
    )

    result = frame[
        [
            "path",
            "filename",
            "reviewed_label",
            "source_id",
            "split",
        ]
    ].copy()

    result["predicted_label"] = pred
    result["confidence"] = confidence
    result["hierarchy_route"] = route
    result["correct"] = (
        result["reviewed_label"]
        == result["predicted_label"]
    )

    mistakes = result[~result["correct"]]

    print("\nMistakes:")
    if len(mistakes):
        print(
            mistakes[
                [
                    "reviewed_label",
                    "predicted_label",
                    "confidence",
                    "hierarchy_route",
                    "filename",
                ]
            ]
            .sort_values(
                ["reviewed_label", "confidence"],
                ascending=[True, False],
            )
            .to_string(index=False)
        )
    else:
        print("None")

    metrics = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
    }

    return metrics, result


df = pd.read_csv(DATA_PATH)
df = df[df["reviewed_label"].isin(OLY_LABELS)].copy()

train = df[df["split"] == "train"].copy()
dev = df[df["split"] == "dev"].copy()
test = df[df["split"] == "test"].copy()

snatch_gate = train_binary(
    train,
    lambda frame: frame["reviewed_label"].apply(
        lambda label: (
            "snatch"
            if label == "snatch"
            else "rack_based"
        )
    ),
)

rack_train = train[
    train["reviewed_label"] != "snatch"
].copy()

clean_gate = train_binary(
    rack_train,
    lambda frame: frame["reviewed_label"].apply(
        lambda label: (
            "clean"
            if label == "clean"
            else "overhead_finish"
        )
    ),
)

jerk_train = train[
    train["reviewed_label"].isin(
        ["clean_and_jerk", "split_jerk"]
    )
].copy()

jerk_router = train_binary(
    jerk_train,
    lambda frame: frame["reviewed_label"],
)

models = {
    "snatch_gate": snatch_gate,
    "clean_gate": clean_gate,
    "jerk_router": jerk_router,
}

dev_metrics, dev_predictions = evaluate(
    models,
    dev,
    "dev",
)

test_metrics, test_predictions = evaluate(
    models,
    test,
    "test",
)

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

bundle = {
    "models": models,
    "feature_names": FEATURE_NAMES,
    "architecture": (
        "snatch_vs_rack_then_clean_vs_overhead_"
        "then_cj_vs_split_jerk"
    ),
    "training_data": str(DATA_PATH),
    "random_state": RANDOM_STATE,
}

joblib.dump(bundle, MODEL_OUT)

metrics = {
    "model_path": str(MODEL_OUT),
    "training_rows": int(len(train)),
    "dev": dev_metrics,
    "test": test_metrics,
}

REPORT_OUT.write_text(
    json.dumps(metrics, indent=2)
)

pd.concat(
    [dev_predictions, test_predictions],
    ignore_index=True,
).to_csv(PREDICTIONS_OUT, index=False)

print("\nSaved model:", MODEL_OUT)
print("Saved metrics:", REPORT_OUT)
print("Saved predictions:", PREDICTIONS_OUT)
