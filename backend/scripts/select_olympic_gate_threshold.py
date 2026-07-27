from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PREDICTIONS_PATH = Path(
    "ml/router_rebuild_v1/reports/olympic_gate_v1_predictions.csv"
)
METRICS_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_gate_v1_thresholds.csv"
)
SELECTED_OUT = Path(
    "ml/router_rebuild_v1/reports/olympic_gate_v1_selected_threshold.json"
)

df = pd.read_csv(PREDICTIONS_PATH)

dev = df[df["split"] == "dev"].copy()
test = df[df["split"] == "test"].copy()

rows = []

for threshold_int in range(30, 81):
    threshold = threshold_int / 100

    y_true = dev["olympic_gate_label"].astype(str)
    y_pred = dev["olympic_probability"].apply(
        lambda p: "olympic" if p >= threshold else "non_olympic"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["non_olympic", "olympic"],
    )

    tn, fp, fn, tp = matrix.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["non_olympic", "olympic"],
        average=None,
        zero_division=0,
    )

    rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "non_olympic_precision": precision[0],
        "non_olympic_recall": recall[0],
        "non_olympic_f1": f1[0],
        "olympic_precision": precision[1],
        "olympic_recall": recall[1],
        "olympic_f1": f1[1],
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })

results = pd.DataFrame(rows)

# Gate policy:
# 1. Preserve at least 90% Olympic recall on dev.
# 2. Among those thresholds, maximize Olympic F1.
# 3. Break ties using fewer false positives, then higher threshold.
eligible = results[
    results["olympic_recall"] >= 0.90
].copy()

if eligible.empty:
    raise SystemExit("No threshold achieved >= 0.90 Olympic recall on dev")

selected = (
    eligible.sort_values(
        [
            "olympic_f1",
            "false_positives",
            "threshold",
        ],
        ascending=[False, True, False],
    )
    .iloc[0]
)

threshold = float(selected["threshold"])

def evaluate_at_threshold(frame, split_name):
    y_true = frame["olympic_gate_label"].astype(str)

    y_pred = frame["olympic_probability"].apply(
        lambda p: "olympic" if p >= threshold else "non_olympic"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["non_olympic", "olympic"],
    )

    tn, fp, fn, tp = matrix.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["non_olympic", "olympic"],
        average=None,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} AT THRESHOLD {threshold:.2f}")
    print("=" * 80)
    print(matrix)
    print("accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("non_olympic precision:", round(float(precision[0]), 4))
    print("non_olympic recall:", round(float(recall[0]), 4))
    print("olympic precision:", round(float(precision[1]), 4))
    print("olympic recall:", round(float(recall[1]), 4))
    print("olympic f1:", round(float(f1[1]), 4))
    print("false positives:", int(fp))
    print("false negatives:", int(fn))

    mistakes = frame[y_true != y_pred].copy()
    mistakes["threshold_prediction"] = y_pred[y_true != y_pred]

    if len(mistakes):
        print("\nMistakes:")
        print(
            mistakes[
                [
                    "reviewed_label",
                    "filename",
                    "olympic_gate_label",
                    "threshold_prediction",
                    "olympic_probability",
                    "path",
                ]
            ]
            .sort_values("olympic_probability")
            .to_string(index=False)
        )
    else:
        print("\nMistakes: none")

    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "non_olympic_precision": float(precision[0]),
        "non_olympic_recall": float(recall[0]),
        "olympic_precision": float(precision[1]),
        "olympic_recall": float(recall[1]),
        "olympic_f1": float(f1[1]),
    }

results.to_csv(METRICS_OUT, index=False)

print("Selected threshold from dev:", threshold)
print("\nTop dev candidates:")
print(
    eligible.sort_values(
        [
            "olympic_f1",
            "false_positives",
            "threshold",
        ],
        ascending=[False, True, False],
    )
    .head(12)
    .to_string(index=False)
)

dev_metrics = evaluate_at_threshold(dev, "dev")
test_metrics = evaluate_at_threshold(test, "test")

payload = {
    "selection_split": "dev",
    "selection_policy": (
        "olympic_recall >= 0.90, maximize olympic_f1, "
        "then minimize false positives, then prefer higher threshold"
    ),
    "threshold": threshold,
    "dev": dev_metrics,
    "test": test_metrics,
}

SELECTED_OUT.write_text(json.dumps(payload, indent=2))

print("\nCreated:", METRICS_OUT)
print("Created:", SELECTED_OUT)
