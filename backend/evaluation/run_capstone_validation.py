from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = Path.home() / "Desktop" / "Capstone"

MANIFEST = (
    BACKEND_ROOT
    / "evaluation"
    / "manifests"
    / "capstone_eval_balanced_v1.csv"
)

RESULT_DIR = (
    BACKEND_ROOT
    / "evaluation"
    / "results"
    / "capstone_validation_v6"
)

RAW_DIR = RESULT_DIR / "raw"
RESULT_CSV = RESULT_DIR / "results.csv"
SUMMARY_JSON = RESULT_DIR / "summary.json"

API_URL = "http://127.0.0.1:8000/analyze"
TIMEOUT_SECONDS = 600

# Manifest names -> production API labels.
LABEL_ALIASES = {
    "back_squat": "squat_back",
    "front_squat": "squat_front",
    "overhead_squat": "overhead_squat",
    "bench_press": "bench_press",
    "clean": "clean",
    "clean_and_jerk": "clean_and_jerk",
    "deadlift": "deadlift",
    "handstand_push_up": "handstand_push_up",
    "pull_up": "pull_up",
    "push_press": "push_press",
    "push_up": "push_up",
    "snatch": "snatch",
    "split_jerk": "split_jerk",
    "strict_press": "strict_press",
    "thruster": "thruster",
}

FIELDS = [
    "index",
    "expected_label",
    "expected_api_label",
    "predicted_label",
    "label_ok",
    "confidence",
    "analysis_mode",
    "rep_count",
    "runtime_seconds",
    "status",
    "error",
    "source_group",
    "relative_path",
    "raw_json",
]


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_existing() -> dict[str, dict[str, str]]:
    if not RESULT_CSV.exists():
        return {}

    with RESULT_CSV.open(newline="") as handle:
        return {
            row["relative_path"]: row
            for row in csv.DictReader(handle)
        }


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def raw_filename(relative_path: str) -> str:
    digest = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:16]

    return f"{digest}.json"


def analyze_video(video_path: Path) -> tuple[dict[str, Any], float]:
    started = time.monotonic()

    with video_path.open("rb") as video:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    video_path.name,
                    video,
                    "application/octet-stream",
                )
            },
            timeout=TIMEOUT_SECONDS,
        )

    runtime = time.monotonic() - started
    response.raise_for_status()

    return response.json(), runtime


def main() -> int:
    if not MANIFEST.exists():
        print(f"Missing manifest: {MANIFEST}", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open(newline="") as handle:
        manifest_rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "validation"
        ]

    existing = load_existing()
    results: list[dict[str, Any]] = []

    # Preserve completed results so the run can resume.
    for row in manifest_rows:
        prior = existing.get(row["relative_path"])

        if prior and prior.get("status") == "ok":
            results.append(prior)

    completed_paths = {
        row["relative_path"]
        for row in results
    }

    print(f"Validation videos: {len(manifest_rows)}")
    print(f"Already completed: {len(completed_paths)}")
    print(f"Remaining: {len(manifest_rows) - len(completed_paths)}")
    print()

    for index, manifest_row in enumerate(manifest_rows, start=1):
        relative_path = manifest_row["relative_path"]

        if relative_path in completed_paths:
            continue

        video_path = CAPSTONE_ROOT / relative_path
        expected = manifest_row["expected_label"]
        expected_api = LABEL_ALIASES.get(expected, expected)
        raw_path = RAW_DIR / raw_filename(relative_path)

        print(
            f"[{index}/{len(manifest_rows)}] "
            f"{expected_api} — {relative_path}",
            flush=True,
        )

        result_row: dict[str, Any] = {
            "index": index,
            "expected_label": expected,
            "expected_api_label": expected_api,
            "predicted_label": "",
            "label_ok": False,
            "confidence": "",
            "analysis_mode": "",
            "rep_count": 0,
            "runtime_seconds": "",
            "status": "error",
            "error": "",
            "source_group": manifest_row["source_group"],
            "relative_path": relative_path,
            "raw_json": str(raw_path.relative_to(BACKEND_ROOT)),
        }

        if not video_path.exists():
            result_row["error"] = "video_not_found"
            results.append(result_row)
            write_results(results)
            print("  ERROR: video not found")
            continue

        try:
            payload, runtime = analyze_video(video_path)

            predicted = str(payload.get("exercise_label") or "")
            confidence = safe_float(payload.get("confidence"))
            rep_feedback = payload.get("rep_feedback") or []

            raw_path.write_text(
                json.dumps(payload, indent=2, default=str)
            )

            result_row.update({
                "predicted_label": predicted,
                "label_ok": predicted == expected_api,
                "confidence": (
                    round(confidence, 6)
                    if confidence is not None
                    else ""
                ),
                "analysis_mode": payload.get(
                    "analysis_mode",
                    "",
                ),
                "rep_count": len(rep_feedback),
                "runtime_seconds": round(runtime, 3),
                "status": "ok",
                "error": "",
            })

            verdict = "PASS" if result_row["label_ok"] else "FAIL"

            print(
                f"  {verdict}: predicted={predicted or 'None'} "
                f"confidence={result_row['confidence']} "
                f"reps={result_row['rep_count']} "
                f"time={result_row['runtime_seconds']}s",
                flush=True,
            )

        except requests.Timeout:
            result_row["error"] = (
                f"timeout_after_{TIMEOUT_SECONDS}s"
            )
            print(f"  ERROR: {result_row['error']}", flush=True)

        except Exception as error:
            result_row["error"] = (
                f"{type(error).__name__}: {error}"
            )
            print(f"  ERROR: {result_row['error']}", flush=True)

        results.append(result_row)
        write_results(results)

    successful = [
        row
        for row in results
        if row["status"] == "ok"
    ]

    correct = [
        row
        for row in successful
        if str(row["label_ok"]).lower() == "true"
        or row["label_ok"] is True
    ]

    confusion = Counter(
        (
            row["expected_api_label"],
            row["predicted_label"],
        )
        for row in successful
        if not (
            str(row["label_ok"]).lower() == "true"
            or row["label_ok"] is True
        )
    )

    per_class: dict[str, dict[str, Any]] = {}

    for label in sorted({
        row["expected_api_label"]
        for row in successful
    }):
        class_rows = [
            row
            for row in successful
            if row["expected_api_label"] == label
        ]

        class_correct = sum(
            str(row["label_ok"]).lower() == "true"
            or row["label_ok"] is True
            for row in class_rows
        )

        per_class[label] = {
            "total": len(class_rows),
            "correct": class_correct,
            "accuracy": (
                round(class_correct / len(class_rows), 4)
                if class_rows
                else None
            ),
        }

    summary = {
        "manifest": str(MANIFEST),
        "split": "validation",
        "total_manifest_rows": len(manifest_rows),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "correct": len(correct),
        "accuracy": (
            round(len(correct) / len(successful), 4)
            if successful
            else None
        ),
        "per_class": per_class,
        "confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (expected, predicted), count
            in confusion.most_common()
        ],
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 64)
    print("CAPSTONE VALIDATION SUMMARY")
    print("=" * 64)
    print(f"Successful: {len(successful)}")
    print(f"Errors:     {len(results) - len(successful)}")
    print(f"Correct:    {len(correct)}")
    print(f"Accuracy:   {summary['accuracy']}")
    print(f"Results:    {RESULT_CSV}")
    print(f"Summary:    {SUMMARY_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
