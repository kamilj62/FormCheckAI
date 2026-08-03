#!/usr/bin/env python3

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
BACKEND_ROOT = HERE.parents[2]
REPO_ROOT = BACKEND_ROOT.parent
MANIFEST_PATH = HERE / "manifest.json"
API_URL = os.environ.get("FORMCHECK_API_URL", "http://127.0.0.1:8000/analyze")

FAULT_TO_BREAKDOWN = {
    "shallow_depth": "depth",
    "torso_collapse": "torso",
    "knee_valgus": "knees",
    "heel_lift": "heels",
    "neck_position": "neck",
}

FAULTS = list(FAULT_TO_BREAKDOWN)

PREDICTED_POSITIVE = {"borderline", "poor"}
EXPECTED_POSITIVE = {"mild", "moderate", "severe"}


def resolve_video_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()

    candidates = [
        path,
        BACKEND_ROOT / path,
        REPO_ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(raw_path)


def predicted_faults(rep: dict) -> dict[str, bool]:
    breakdown = rep.get("breakdown") or {}

    return {
        fault: str(breakdown.get(category, "good")).lower()
        in PREDICTED_POSITIVE
        for fault, category in FAULT_TO_BREAKDOWN.items()
    }


def expected_faults(rep: dict) -> dict[str, bool]:
    faults = rep.get("faults") or {}

    return {
        fault: str(faults.get(fault, "none")).lower()
        in EXPECTED_POSITIVE
        for fault in FAULTS
    }


def update_confusion(stats: dict, expected: bool, predicted: bool) -> None:
    if expected and predicted:
        stats["tp"] += 1
    elif not expected and predicted:
        stats["fp"] += 1
    elif expected and not predicted:
        stats["fn"] += 1
    else:
        stats["tn"] += 1


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    data = json.loads(MANIFEST_PATH.read_text())
    videos = data.get("videos", [])

    if not videos:
        print("ERROR: manifest contains no videos")
        raise SystemExit(1)

    fault_stats = {
        fault: defaultdict(int)
        for fault in FAULTS
    }

    total_videos = 0
    label_correct = 0
    rep_count_correct = 0
    total_expected_reps = 0
    total_predicted_reps = 0

    detailed_results = []

    for video in videos:
        video_id = video["id"]

        try:
            video_path = resolve_video_path(video["path"])
        except FileNotFoundError:
            print(f"[ERROR] {video_id}: file not found: {video['path']}")
            continue

        print(f"\nAnalyzing {video_id}")
        print(f"  {video_path}")

        try:
            with video_path.open("rb") as file_handle:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            video_path.name,
                            file_handle,
                            "application/octet-stream",
                        )
                    },
                    timeout=900,
                )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            print(f"[ERROR] {video_id}: {exc}")
            continue

        total_videos += 1

        predicted_label = result.get("exercise_label")
        expected_reps = video["expected_reps"]
        predicted_reps = result.get("rep_feedback") or []

        label_ok = predicted_label == "squat_back"
        reps_ok = len(predicted_reps) == expected_reps

        label_correct += int(label_ok)
        rep_count_correct += int(reps_ok)
        total_expected_reps += expected_reps
        total_predicted_reps += len(predicted_reps)

        print(
            f"  label: {predicted_label} "
            f"({'PASS' if label_ok else 'FAIL'})"
        )
        print(
            f"  reps: expected={expected_reps}, "
            f"predicted={len(predicted_reps)} "
            f"({'PASS' if reps_ok else 'FAIL'})"
        )

        expected_rep_entries = video.get("reps", [])
        max_reps = max(len(expected_rep_entries), len(predicted_reps))

        video_rep_results = []

        for index in range(max_reps):
            expected_rep = (
                expected_rep_entries[index]
                if index < len(expected_rep_entries)
                else {"faults": {}}
            )
            predicted_rep = (
                predicted_reps[index]
                if index < len(predicted_reps)
                else {"breakdown": {}}
            )

            expected_map = expected_faults(expected_rep)
            predicted_map = predicted_faults(predicted_rep)

            print(f"  rep {index + 1}:")

            for fault in FAULTS:
                expected_value = expected_map[fault]
                predicted_value = predicted_map[fault]

                update_confusion(
                    fault_stats[fault],
                    expected_value,
                    predicted_value,
                )

                marker = "PASS" if expected_value == predicted_value else "FAIL"
                print(
                    f"    {fault}: "
                    f"expected={expected_value}, "
                    f"predicted={predicted_value} "
                    f"[{marker}]"
                )

            video_rep_results.append(
                {
                    "rep": index + 1,
                    "expected": expected_map,
                    "predicted": predicted_map,
                    "breakdown": predicted_rep.get("breakdown", {}),
                    "score": predicted_rep.get("score"),
                }
            )

        detailed_results.append(
            {
                "id": video_id,
                "path": str(video_path),
                "predicted_label": predicted_label,
                "label_ok": label_ok,
                "expected_reps": expected_reps,
                "predicted_reps": len(predicted_reps),
                "rep_count_ok": reps_ok,
                "reps": video_rep_results,
            }
        )

    print("\n" + "=" * 72)
    print("BACK SQUAT ANALYSIS V2 — BASELINE RESULTS")
    print("=" * 72)

    print(
        f"Exercise-label accuracy: "
        f"{label_correct}/{total_videos} "
        f"({safe_div(label_correct, total_videos) * 100:.1f}%)"
    )

    print(
        f"Exact rep-count accuracy: "
        f"{rep_count_correct}/{total_videos} "
        f"({safe_div(rep_count_correct, total_videos) * 100:.1f}%)"
    )

    print(
        f"Total reps: expected={total_expected_reps}, "
        f"predicted={total_predicted_reps}"
    )

    print("\nFault metrics:")

    summary = {}

    for fault in FAULTS:
        stats = fault_stats[fault]

        precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        specificity = safe_div(stats["tn"], stats["tn"] + stats["fp"])
        accuracy = safe_div(
            stats["tp"] + stats["tn"],
            stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"],
        )

        summary[fault] = {
            **dict(stats),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
        }

        print(
            f"  {fault:18} "
            f"precision={precision:.2f} "
            f"recall={recall:.2f} "
            f"specificity={specificity:.2f} "
            f"accuracy={accuracy:.2f} "
            f"TP={stats['tp']} FP={stats['fp']} "
            f"FN={stats['fn']} TN={stats['tn']}"
        )

    output = {
        "api_url": API_URL,
        "videos_tested": total_videos,
        "label_accuracy": safe_div(label_correct, total_videos),
        "rep_count_accuracy": safe_div(rep_count_correct, total_videos),
        "fault_metrics": summary,
        "results": detailed_results,
    }

    output_path = HERE / "baseline_results.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
