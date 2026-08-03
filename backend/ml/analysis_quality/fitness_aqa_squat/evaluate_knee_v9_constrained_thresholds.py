import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")


def load_predictions(split):
    path = (
        BASE
        / f"knee_rep_v9_{split}_predictions.jsonl"
    )

    with path.open() as file:
        return [
            json.loads(line)
            for line in file
        ]


def arrays(rows):
    labels = np.asarray(
        [
            float(
                row[
                    "true_ascent_forward_fraction"
                ]
            ) >= 0.5
            for row in rows
        ],
        dtype=np.int64,
    )

    scores = np.asarray(
        [
            float(
                row[
                    "predicted_ascent_forward_fraction"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    return labels, scores


def metrics(labels, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    balanced_accuracy = (
        (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )
        + specificity
    ) / 2.0

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
        "false_positive_rate": float(fpr),
        "specificity": float(specificity),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def select_constrained(
    labels,
    scores,
    maximum_fpr,
):
    candidates = []

    for threshold in np.linspace(
        0.02,
        0.98,
        193,
    ):
        result = metrics(
            labels,
            scores,
            float(threshold),
        )

        if (
            result["false_positive_rate"]
            <= maximum_fpr
        ):
            candidates.append(result)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["balanced_accuracy"],
            item["precision"],
            item["threshold"],
        ),
    )


validation_rows = load_predictions(
    "validation"
)
test_rows = load_predictions("test")

validation_labels, validation_scores = arrays(
    validation_rows
)

test_labels, test_scores = arrays(
    test_rows
)

results = {}

print("=" * 72)
print("VALIDATION-CONSTRAINED FORWARD THRESHOLDS")
print("=" * 72)

for maximum_fpr in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]:
    selected = select_constrained(
        validation_labels,
        validation_scores,
        maximum_fpr,
    )

    if selected is None:
        print(
            f"\nMaximum validation FPR "
            f"{maximum_fpr:.2f}: no threshold"
        )
        continue

    test_result = metrics(
        test_labels,
        test_scores,
        selected["threshold"],
    )

    results[str(maximum_fpr)] = {
        "validation": selected,
        "test_exploratory": test_result,
    }

    print(
        f"\nMaximum validation FPR: "
        f"{maximum_fpr:.2f}"
    )

    print("Selected threshold:")
    print(
        round(
            selected["threshold"],
            4,
        )
    )

    print("Validation:")
    print(
        json.dumps(
            selected,
            indent=2,
        )
    )

    print("Exploratory test:")
    print(
        json.dumps(
            test_result,
            indent=2,
        )
    )


# Baseline: always predict the majority positive class.
always_positive_validation = metrics(
    validation_labels,
    np.ones_like(
        validation_scores
    ),
    0.5,
)

always_positive_test = metrics(
    test_labels,
    np.ones_like(test_scores),
    0.5,
)

print("\n" + "=" * 72)
print("ALWAYS-POSITIVE BASELINE")
print("=" * 72)

print("\nValidation:")
print(
    json.dumps(
        always_positive_validation,
        indent=2,
    )
)

print("\nTest:")
print(
    json.dumps(
        always_positive_test,
        indent=2,
    )
)

output_path = (
    BASE
    / "knee_v9_constrained_thresholds.json"
)

output_path.write_text(
    json.dumps(
        {
            "selection_source": "validation",
            "results": results,
            "always_positive_validation": (
                always_positive_validation
            ),
            "always_positive_test": (
                always_positive_test
            ),
        },
        indent=2,
    )
)

print("\nSaved:", output_path)
