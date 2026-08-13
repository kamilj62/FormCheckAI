#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_IN = Path("backend/ml/benchmark/results/analyzer_audit_latest.csv")
DEFAULT_OUT = Path("backend/ml/benchmark/results/rep_detection_summary.csv")


def parse_int(value):
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def parse_bool(value):
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def label_for(row):
    return (
        row.get("expected_api_label")
        or row.get("expected_label")
        or row.get("label")
        or "unknown"
    )


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[label_for(row)].append(row)

    summary_rows = []

    for label in sorted(groups):
        label_rows = groups[label]
        detected = [
            count
            for row in label_rows
            if (count := parse_int(
                row.get("detected_reps")
                or row.get("rep_count")
            )) is not None
        ]
        expected_pairs = [
            (
                parse_int(
                    row.get("detected_reps")
                    or row.get("rep_count")
                ),
                parse_int(row.get("expected_reps")),
            )
            for row in label_rows
            if parse_int(row.get("expected_reps")) is not None
        ]
        expected_pairs = [
            pair
            for pair in expected_pairs
            if pair[0] is not None
        ]

        phase_complete = [
            value
            for row in label_rows
            if (value := parse_bool(row.get("rep_phase_complete")))
            is not None
        ]
        phase_ordered = [
            value
            for row in label_rows
            if (value := parse_bool(row.get("rep_phase_ordered")))
            is not None
        ]

        exact = sum(got == expected for got, expected in expected_pairs)
        under = sum(got < expected for got, expected in expected_pairs)
        over = sum(got > expected for got, expected in expected_pairs)

        summary_rows.append({
            "label": label,
            "videos": len(label_rows),
            "rep_counts_known": len(detected),
            "zero_rep_videos": sum(count == 0 for count in detected),
            "min_reps": min(detected) if detected else "",
            "max_reps": max(detected) if detected else "",
            "avg_reps": (
                round(sum(detected) / len(detected), 3)
                if detected
                else ""
            ),
            "expected_reps_known": len(expected_pairs),
            "rep_count_exact": exact,
            "rep_count_under": under,
            "rep_count_over": over,
            "rep_count_accuracy": (
                round(exact / len(expected_pairs), 4)
                if expected_pairs
                else ""
            ),
            "phase_complete_known": len(phase_complete),
            "phase_complete_rate": (
                round(sum(phase_complete) / len(phase_complete), 4)
                if phase_complete
                else ""
            ),
            "phase_ordered_known": len(phase_ordered),
            "phase_ordered_rate": (
                round(sum(phase_ordered) / len(phase_ordered), 4)
                if phase_ordered
                else ""
            ),
        })

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.out)

    rows = list(csv.DictReader(source.open()))
    summary_rows = summarize(rows)

    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label",
        "videos",
        "rep_counts_known",
        "zero_rep_videos",
        "min_reps",
        "max_reps",
        "avg_reps",
        "expected_reps_known",
        "rep_count_exact",
        "rep_count_under",
        "rep_count_over",
        "rep_count_accuracy",
        "phase_complete_known",
        "phase_complete_rate",
        "phase_ordered_known",
        "phase_ordered_rate",
    ]

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Rep detection summary: {output}")
    print(f"Labels: {len(summary_rows)}")

    for row in summary_rows:
        print(
            f"{row['label']}: videos={row['videos']} "
            f"zero={row['zero_rep_videos']} "
            f"avg={row['avg_reps']} "
            f"expected_known={row['expected_reps_known']} "
            f"phase_complete={row['phase_complete_rate']}"
        )


if __name__ == "__main__":
    main()
