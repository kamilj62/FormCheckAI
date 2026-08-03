import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

GLOBAL_FORWARD_THRESHOLD = 0.38
INWARD_THRESHOLD = 0.225

INTERVAL_TYPES = [
    "complete",
    "descent_only",
    "ascent_only",
    "short_or_ambiguous",
]


def load_predictions(split):
    path = (
        BASE
        / f"knee_interval_v8_{split}_predictions.jsonl"
    )

    with path.open() as file:
        return [
            json.loads(line)
            for line in file
        ]


def binary_metrics(labels, predictions):
    return {
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


def select_threshold(rows):
    labels = np.asarray(
        [
            float(
                row["true_forward_fraction"]
            ) >= 0.5
            for row in rows
        ],
        dtype=np.int64,
    )

    scores = np.asarray(
        [
            float(
                row[
                    "predicted_forward_fraction"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    best = None

    for threshold in np.linspace(
        0.02,
        0.98,
        193,
    ):
        predictions = (
            scores >= threshold
        ).astype(np.int64)

        metrics = binary_metrics(
            labels,
            predictions,
        )

        key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            float(threshold),
        )

        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                **metrics,
            }

    return {
        key: value
        for key, value in best.items()
        if key != "key"
    }


validation_rows = load_predictions(
    "validation"
)
test_rows = load_predictions("test")

validation_by_type = defaultdict(list)

for row in validation_rows:
    validation_by_type[
        row["interval_type"]
    ].append(row)

thresholds = {}

print("=" * 72)
print("VALIDATION-DERIVED THRESHOLDS")
print("=" * 72)

for interval_type in INTERVAL_TYPES:
    result = select_threshold(
        validation_by_type[interval_type]
    )

    thresholds[interval_type] = (
        result["threshold"]
    )

    positive_count = sum(
        float(
            row["true_forward_fraction"]
        ) >= 0.5
        for row in validation_by_type[
            interval_type
        ]
    )

    print(
        interval_type,
        {
            "rows": len(
                validation_by_type[
                    interval_type
                ]
            ),
            "positives": int(
                positive_count
            ),
            **result,
        },
    )


def evaluate(rows, use_type_thresholds):
    true_forward = np.asarray(
        [
            float(
                row["true_forward_fraction"]
            ) >= 0.5
            for row in rows
        ],
        dtype=np.int64,
    )

    true_inward = np.asarray(
        [
            float(
                row["true_inward_fraction"]
            ) >= 0.5
            for row in rows
        ],
        dtype=np.int64,
    )

    forward_scores = np.asarray(
        [
            float(
                row[
                    "predicted_forward_fraction"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    inward_scores = np.asarray(
        [
            float(
                row[
                    "predicted_inward_fraction"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    if use_type_thresholds:
        forward_predictions = np.asarray(
            [
                int(
                    score
                    >= thresholds[
                        row["interval_type"]
                    ]
                )
                for row, score in zip(
                    rows,
                    forward_scores,
                )
            ],
            dtype=np.int64,
        )
    else:
        forward_predictions = (
            forward_scores
            >= GLOBAL_FORWARD_THRESHOLD
        ).astype(np.int64)

    inward_predictions = (
        inward_scores >= INWARD_THRESHOLD
    ).astype(np.int64)

    true_states = (
        true_forward
        + 2 * true_inward
    )

    predicted_states = (
        forward_predictions
        + 2 * inward_predictions
    )

    neither = true_states == 0
    forward_only = true_states == 1
    inward_only = true_states == 2
    both = true_states == 3

    result = {
        "forward": binary_metrics(
            true_forward,
            forward_predictions,
        ),
        "joint_confusion": (
            confusion_matrix(
                true_states,
                predicted_states,
                labels=[0, 1, 2, 3],
            ).tolist()
        ),
        "false_forward_on_neither": (
            float(
                forward_predictions[
                    neither
                ].mean()
            )
            if neither.any()
            else None
        ),
        "false_forward_on_inward_only": (
            float(
                forward_predictions[
                    inward_only
                ].mean()
            )
            if inward_only.any()
            else None
        ),
        "correct_neither_state": (
            float(
                (
                    predicted_states[
                        neither
                    ] == 0
                ).mean()
            )
            if neither.any()
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
        "correct_both_state": (
            float(
                (
                    predicted_states[
                        both
                    ] == 3
                ).mean()
            )
            if both.any()
            else None
        ),
    }

    return result


print("\n" + "=" * 72)
print("EXPLORATORY TEST COMPARISON")
print("=" * 72)

global_result = evaluate(
    test_rows,
    use_type_thresholds=False,
)

type_result = evaluate(
    test_rows,
    use_type_thresholds=True,
)

print("\nGlobal threshold:")
print(
    json.dumps(
        global_result,
        indent=2,
    )
)

print("\nValidation-derived type thresholds:")
print(
    json.dumps(
        type_result,
        indent=2,
    )
)

output = {
    "global_forward_threshold": (
        GLOBAL_FORWARD_THRESHOLD
    ),
    "validation_type_thresholds": (
        thresholds
    ),
    "global_test_result": (
        global_result
    ),
    "type_threshold_test_result": (
        type_result
    ),
}

output_path = (
    BASE
    / "v8_type_threshold_evaluation.json"
)

output_path.write_text(
    json.dumps(output, indent=2)
)

print("\nSaved:", output_path)
