#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TRUTH = Path("backend/ml/benchmark/config/rep_truth_manifest.csv")
DEFAULT_RESULTS = [
    Path("backend/evaluation/results/capstone_validation_v17/results.csv"),
    Path("backend/ml/benchmark/results/random_spotcheck_latest.csv"),
    Path("backend/ml/benchmark/results/gold_candidate_eval_latest.csv"),
]
DEFAULT_OUT = Path("backend/ml/benchmark/results/rep_truth_comparison.csv")
CAPSTONE_ROOT = Path("/Users/josephkamil/Desktop/Capstone")


def parse_int(value):
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def relative_video_key(video):
    path = Path(str(video or ""))

    try:
        return str(path.relative_to(CAPSTONE_ROOT))
    except Exception:
        return str(video or "")


def result_keys(row):
    keys = set()
    for value in (
        row.get("relative_path"),
        row.get("video"),
        row.get("name"),
    ):
        if value:
            keys.add(str(value))
            keys.add(Path(str(value)).name)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH))
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        help=(
            "Saved analyzer result CSV. Can be passed multiple times. "
            "Defaults to the latest saved validation and spotcheck files."
        ),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    truth_path = Path(args.truth)
    result_paths = [
        Path(path)
        for path in (args.results or [str(path) for path in DEFAULT_RESULTS])
    ]
    out_path = Path(args.out)

    truth_rows = list(csv.DictReader(truth_path.open()))
    result_rows = []

    for results_path in result_paths:
        if not results_path.exists():
            continue
        result_rows.extend(csv.DictReader(results_path.open()))

    index = {}
    for row in result_rows:
        for key in result_keys(row):
            index.setdefault(key, row)

    comparison_rows = []

    for truth in truth_rows:
        expected = parse_int(truth.get("expected_reps"))
        relative_key = relative_video_key(truth.get("video"))
        name_key = truth.get("name") or Path(relative_key).name

        result = (
            index.get(relative_key)
            or index.get(name_key)
            or {}
        )

        detected = parse_int(
            result.get("rep_count")
            or result.get("detected_reps")
        )

        if detected is None:
            verdict = "missing_result"
            delta = ""
        else:
            delta_value = detected - int(expected)
            delta = delta_value
            if delta_value == 0:
                verdict = "pass"
            elif delta_value < 0:
                verdict = "undercount"
            else:
                verdict = "overcount"

        comparison_rows.append({
            "label": truth.get("label"),
            "name": name_key,
            "relative_path": relative_key,
            "expected_reps": expected,
            "detected_reps": detected if detected is not None else "",
            "delta": delta,
            "verdict": verdict,
            "review_status": truth.get("review_status"),
            "predicted_label": result.get("predicted_label", ""),
            "analysis_mode": result.get("analysis_mode", ""),
            "notes": truth.get("notes"),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "name",
        "relative_path",
        "expected_reps",
        "detected_reps",
        "delta",
        "verdict",
        "review_status",
        "predicted_label",
        "analysis_mode",
        "notes",
    ]

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Rep truth comparison: {out_path}")

    for row in comparison_rows:
        print(
            f"{row['label']} {row['name']}: "
            f"expected={row['expected_reps']} "
            f"detected={row['detected_reps']} "
            f"{row['verdict']}"
        )

    failed = [
        row
        for row in comparison_rows
        if row["verdict"] != "pass"
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
