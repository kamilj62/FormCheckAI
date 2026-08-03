import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "p25": round(
            float(np.percentile(values, 25)),
            4,
        ),
        "p75": round(
            float(np.percentile(values, 75)),
            4,
        ),
    }


def best_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return None

    precision, recall, thresholds = (
        precision_recall_curve(labels, scores)
    )

    best = None

    for index, threshold in enumerate(thresholds):
        p = float(precision[index])
        r = float(recall[index])

        f1 = (
            2.0 * p * r / (p + r)
            if p + r > 0.0
            else 0.0
        )

        key = (f1, p, r, float(threshold))

        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                "precision": p,
                "recall": r,
                "f1": f1,
            }

    return {
        key: value
        for key, value in best.items()
        if key != "key"
    }


for split in ["validation", "test"]:
    path = (
        BASE
        / f"knee_interval_v8_{split}_predictions.jsonl"
    )

    groups = defaultdict(list)

    with path.open() as file:
        for line in file:
            row = json.loads(line)
            groups[row["interval_type"]].append(row)

    print("\n" + "=" * 76)
    print(split.upper())
    print("=" * 76)

    for interval_type in [
        "complete",
        "descent_only",
        "ascent_only",
        "short_or_ambiguous",
    ]:
        rows = groups[interval_type]

        labels = np.asarray(
            [
                float(row["true_forward_fraction"]) >= 0.5
                for row in rows
            ],
            dtype=np.int64,
        )

        scores = np.asarray(
            [
                float(
                    row["predicted_forward_fraction"]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        print(f"\n{interval_type}")
        print("rows:", len(rows))
        print("positives:", int(labels.sum()))
        print(
            "positive scores:",
            summarize(positive_scores),
        )
        print(
            "negative scores:",
            summarize(negative_scores),
        )

        if len(np.unique(labels)) >= 2:
            print(
                "ROC-AUC:",
                round(
                    float(
                        roc_auc_score(labels, scores)
                    ),
                    4,
                ),
            )
            print(
                "average precision:",
                round(
                    float(
                        average_precision_score(
                            labels,
                            scores,
                        )
                    ),
                    4,
                ),
            )
            print(
                "best local threshold:",
                best_threshold(labels, scores),
            )
        else:
            print(
                "ROC-AUC/AP unavailable: "
                "only one class present"
            )
