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

from app.feature_engine.movement_video_features_v4 import FEATURE_NAMES


DATA_PATH = Path(
    "ml/router_rebuild_v1/features/router_features_v4_grouped_cj.csv"
)
MODEL_OUT = Path(
    "app/models/candidates/olympic_router_stage2_v7_grouped_cj.joblib"
)
REPORT_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_router_stage2_v7_grouped_cj_metrics.json"
)
PREDICTIONS_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_router_stage2_v7_grouped_cj_predictions.csv"
)

OLY_LABELS = [
    "clean",
    "clean_and_jerk",
    "snatch",
    "split_jerk",
]

RANDOM_STATE = 42


def evaluate(model, frame, split_name):
    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = frame["reviewed_label"].astype(str).to_numpy()

    pred = model.predict(X)
    probs = model.predict_proba(X)

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(
        y,
        pred,
        labels=OLY_LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print(split_name.upper())
    print("=" * 80)
    print("rows:", len(frame))
    print("accuracy:", round(accuracy, 4))
    print("macro F1:", round(macro_f1, 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, pred, labels=OLY_LABELS))

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
    result["confidence"] = probs.max(axis=1)
    result["correct"] = result["reviewed_label"] == result["predicted_label"]

    for class_index, label in enumerate(model.classes_):
        result[f"prob_{label}"] = probs[:, class_index]

    metrics = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=OLY_LABELS,
        ).tolist(),
    }

    return metrics, result


df = pd.read_csv(DATA_PATH)

df = df[
    df["reviewed_label"].isin(OLY_LABELS)
].copy()

train = df[df["split"] == "train"].copy()
dev = df[df["split"] == "dev"].copy()
test = df[df["split"] == "test"].copy()

print("Olympic rows:", len(df))
print("\nRows by split:")
print(df["split"].value_counts().to_string())

print("\nRows by class and split:")
print(
    pd.crosstab(
        df["reviewed_label"],
        df["split"],
    ).to_string()
)

X_train = train[FEATURE_NAMES].to_numpy(dtype=np.float32)
y_train = train["reviewed_label"].astype(str).to_numpy()

model = RandomForestClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)

dev_metrics, dev_predictions = evaluate(model, dev, "dev")
test_metrics, test_predictions = evaluate(model, test, "test")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

bundle = {
    "model": model,
    "feature_names": FEATURE_NAMES,
    "classes": list(model.classes_),
    "training_data": str(DATA_PATH),
    "random_state": RANDOM_STATE,
    "architecture": "stage_2_olympic_router_v7_grouped_cj",
}

joblib.dump(bundle, MODEL_OUT)

metrics = {
    "model_path": str(MODEL_OUT),
    "training_rows": int(len(train)),
    "dev": dev_metrics,
    "test": test_metrics,
}

REPORT_OUT.write_text(json.dumps(metrics, indent=2))

predictions = pd.concat(
    [dev_predictions, test_predictions],
    ignore_index=True,
)
predictions.to_csv(PREDICTIONS_OUT, index=False)

print("\nSaved model:", MODEL_OUT)
print("Saved metrics:", REPORT_OUT)
print("Saved predictions:", PREDICTIONS_OUT)

print("\nMisclassified test rows:")
mistakes = test_predictions[
    ~test_predictions["correct"]
]

if len(mistakes):
    print(
        mistakes[
            [
                "reviewed_label",
                "predicted_label",
                "confidence",
                "filename",
                "path",
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
