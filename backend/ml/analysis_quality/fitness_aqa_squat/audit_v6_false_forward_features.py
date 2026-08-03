import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_PATH = (
    BASE / "models/knee_interval_soft_target_v6.joblib"
)
TEST_PATH = BASE / "knee_v6_interval_test.jsonl"


def load_test():
    features = []
    forward_targets = []
    inward_targets = []
    records = []

    with TEST_PATH.open() as f:
        for line in f:
            row = json.loads(line)

            features.append(
                np.asarray(
                    row["features"],
                    dtype=np.float32,
                )
            )

            forward_targets.append(
                float(
                    row["targets"][
                        "forward_fraction"
                    ]
                )
            )

            inward_targets.append(
                float(
                    row["targets"][
                        "inward_fraction"
                    ]
                )
            )

            records.append({
                "video_id": str(row["video_id"]),
                "segment_index": int(
                    row["segment_index"]
                ),
                "start_frame": int(
                    row["start_frame"]
                ),
                "end_frame": int(
                    row["end_frame"]
                ),
            })

    return (
        np.asarray(features),
        np.asarray(forward_targets),
        np.asarray(inward_targets),
        records,
    )


bundle = joblib.load(MODEL_PATH)

model = bundle["forward_model"]
threshold = float(bundle["forward_threshold"])
feature_names = bundle["feature_names"]

X, y_forward, y_inward, records = load_test()

scores = np.clip(
    model.predict(X),
    0.0,
    1.0,
)

true_forward = y_forward >= 0.5
predicted_forward = scores >= threshold

false_positive = (
    (~true_forward)
    & predicted_forward
)

true_negative = (
    (~true_forward)
    & (~predicted_forward)
)

true_positive = (
    true_forward
    & predicted_forward
)

false_negative = (
    true_forward
    & (~predicted_forward)
)

print("Threshold:", threshold)
print("False positives:", int(false_positive.sum()))
print("True negatives:", int(true_negative.sum()))
print("True positives:", int(true_positive.sum()))
print("False negatives:", int(false_negative.sum()))

print("\nMean predicted forward score")
for name, mask in [
    ("false_positive", false_positive),
    ("true_negative", true_negative),
    ("true_positive", true_positive),
    ("false_negative", false_negative),
]:
    print(
        name,
        round(float(scores[mask].mean()), 4)
        if mask.any()
        else "none",
    )

# Global model importance.
global_rows = [
    {
        "feature": feature_names[index],
        "importance": float(importance),
    }
    for index, importance in enumerate(
        model.feature_importances_
    )
]

global_rows.sort(
    key=lambda row: row["importance"],
    reverse=True,
)

print("\nTop model feature importances")
for row in global_rows[:35]:
    print(
        f"{row['feature']:55s} "
        f"{row['importance']:.6f}"
    )

# Effect size comparing false positives with true negatives.
comparison_rows = []

for index, name in enumerate(feature_names):
    fp_values = X[false_positive, index]
    tn_values = X[true_negative, index]

    if len(fp_values) < 2 or len(tn_values) < 2:
        continue

    fp_mean = float(np.mean(fp_values))
    tn_mean = float(np.mean(tn_values))

    pooled_variance = (
        float(np.var(fp_values, ddof=1))
        + float(np.var(tn_values, ddof=1))
    ) / 2.0

    pooled_std = max(
        pooled_variance ** 0.5,
        1e-8,
    )

    effect = (
        fp_mean - tn_mean
    ) / pooled_std

    comparison_rows.append({
        "feature": name,
        "effect": float(effect),
        "absolute_effect": abs(float(effect)),
        "false_positive_mean": fp_mean,
        "true_negative_mean": tn_mean,
        "model_importance": float(
            model.feature_importances_[index]
        ),
    })

comparison_rows.sort(
    key=lambda row: (
        row["absolute_effect"],
        row["model_importance"],
    ),
    reverse=True,
)

print(
    "\nLargest feature differences: "
    "false positives versus true negatives"
)

for row in comparison_rows[:40]:
    print(
        f"{row['feature']:55s} "
        f"effect={row['effect']:8.3f} "
        f"fp={row['false_positive_mean']:10.4f} "
        f"tn={row['true_negative_mean']:10.4f} "
        f"importance={row['model_importance']:.6f}"
    )

# Permutation importance on the binary forward task.
binary_target = true_forward.astype(np.int64)

def regressor_auc_scorer(estimator, features, labels):
    predictions = np.clip(
        estimator.predict(features),
        0.0,
        1.0,
    )

    return roc_auc_score(
        labels,
        predictions,
    )


permutation = permutation_importance(
    model,
    X,
    binary_target,
    scoring=regressor_auc_scorer,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_rows = [
    {
        "feature": feature_names[index],
        "importance": float(
            permutation.importances_mean[index]
        ),
        "std": float(
            permutation.importances_std[index]
        ),
    }
    for index in range(len(feature_names))
]

permutation_rows.sort(
    key=lambda row: row["importance"],
    reverse=True,
)

print("\nTop test permutation importances")
for row in permutation_rows[:35]:
    print(
        f"{row['feature']:55s} "
        f"importance={row['importance']:9.5f} "
        f"std={row['std']:9.5f}"
    )

output = {
    "threshold": threshold,
    "counts": {
        "false_positive": int(false_positive.sum()),
        "true_negative": int(true_negative.sum()),
        "true_positive": int(true_positive.sum()),
        "false_negative": int(false_negative.sum()),
    },
    "global_importance": global_rows,
    "false_positive_vs_true_negative": (
        comparison_rows
    ),
    "permutation_importance": permutation_rows,
}

output_path = (
    BASE / "v6_false_forward_feature_audit.json"
)

output_path.write_text(
    json.dumps(output, indent=2)
)

print("\nSaved:", output_path)
