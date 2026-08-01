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
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.feature_engine.movement_video_features import FEATURE_NAMES


DATA_PATH = Path(
    "ml/router_rebuild_v1/features/router_features_v2_gate_hardneg_v1.csv"
)
MODEL_OUT = Path(
    "app/models/candidates/olympic_gate_hardneg_v1.joblib"
)
REPORT_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_gate_hardneg_v1_metrics.json"
)
PREDICTIONS_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_gate_hardneg_v1_predictions.csv"
)

RANDOM_STATE = 42


def evaluate(model, frame, split_name):
    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = frame["olympic_gate_label"].astype(str).to_numpy()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    class_to_index = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    olympic_index = class_to_index["olympic"]
    olympic_probability = probabilities[:, olympic_index]

    accuracy = accuracy_score(y, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predictions,
        labels=["non_olympic", "olympic"],
        average=None,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            (y == "olympic").astype(int),
            olympic_probability,
        )
    except ValueError:
        auc = None

    print("\n" + "=" * 80)
    print(split_name.upper())
    print("=" * 80)
    print("rows:", len(frame))
    print("accuracy:", round(accuracy, 4))
    print("ROC-AUC:", None if auc is None else round(auc, 4))
    print(
        confusion_matrix(
            y,
            predictions,
            labels=["non_olympic", "olympic"],
        )
    )
    print(
        classification_report(
            y,
            predictions,
            labels=["non_olympic", "olympic"],
            zero_division=0,
        )
    )

    result = frame[
        [
            "path",
            "filename",
            "reviewed_label",
            "olympic_gate_label",
            "source_id",
            "split",
        ]
    ].copy()

    result["predicted_gate"] = predictions
    result["olympic_probability"] = olympic_probability
    result["correct"] = (
        result["olympic_gate_label"]
        == result["predicted_gate"]
    )

    metrics = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy),
        "roc_auc": None if auc is None else float(auc),
        "non_olympic": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "olympic": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }

    return metrics, result


df = pd.read_csv(DATA_PATH)

required = {
    "split",
    "source_id",
    "olympic_gate_label",
    *FEATURE_NAMES,
}
missing = required - set(df.columns)

if missing:
    raise SystemExit(f"Missing columns: {sorted(missing)}")

train = df[df["split"] == "train"].copy()
dev = df[df["split"] == "dev"].copy()
test = df[df["split"] == "test"].copy()

X_train = train[FEATURE_NAMES].to_numpy(dtype=np.float32)
y_train = train["olympic_gate_label"].astype(str).to_numpy()

model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)

dev_metrics, dev_predictions = evaluate(
    model,
    dev,
    "dev",
)
test_metrics, test_predictions = evaluate(
    model,
    test,
    "test",
)

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

bundle = {
    "model": model,
    "feature_names": FEATURE_NAMES,
    "classes": list(model.classes_),
    "training_data": str(DATA_PATH),
    "random_state": RANDOM_STATE,
    "architecture": "stage_1_olympic_gate",
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
                "filename",
                "olympic_gate_label",
                "predicted_gate",
                "olympic_probability",
                "path",
            ]
        ]
        .sort_values("olympic_probability")
        .to_string(index=False)
    )
else:
    print("None")
