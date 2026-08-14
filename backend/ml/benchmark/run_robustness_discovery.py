#!/usr/bin/env python3

import csv
import json
import subprocess
from pathlib import Path

MANIFEST = Path("ml/benchmark/config/robustness_discovery_manifest.csv")
OUT = Path("ml/benchmark/results/robustness_discovery_latest.csv")
API = "http://127.0.0.1:8000/analyze"

rows = list(csv.DictReader(MANIFEST.open()))
results = []

for i, row in enumerate(rows, start=1):
    label = row["label"]
    video = row["video"]
    name = row["name"]

    print(f"[{i}/{len(rows)}] {label} :: {name}", flush=True)

    cmd = [
        "curl",
        "--max-time", "300",
        "-s",
        "-X", "POST",
        "-F", f"file=@{video}",
        API,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=330,
        )

        data = json.loads(proc.stdout or "{}")

        predicted = data.get("exercise_label")
        reps = len(data.get("rep_feedback") or [])
        confidence = data.get("confidence")
        mode = data.get("analysis_mode")

        debug = data.get("debug") or {}

        result = {
            "expected_label": label,
            "name": name,
            "video": video,
            "predicted_label": predicted,
            "confidence": confidence,
            "analysis_mode": mode,
            "detected_reps": reps,
            "label_match": predicted == label,
            "raw_label": debug.get("raw_label"),
            "bodyweight_router_label": debug.get("bodyweight_router_label"),
            "protected_label": debug.get("protected_label"),
            "protected_reason": debug.get("protected_reason"),
            "error": data.get("error"),
        }

    except Exception as e:
        result = {
            "expected_label": label,
            "name": name,
            "video": video,
            "predicted_label": "",
            "confidence": "",
            "analysis_mode": "",
            "detected_reps": "",
            "label_match": False,
            "raw_label": "",
            "bodyweight_router_label": "",
            "protected_label": "",
            "protected_reason": "",
            "error": repr(e),
        }

    results.append(result)

    print(
        f"    -> {result['predicted_label']} "
        f"reps={result['detected_reps']} "
        f"{'MATCH' if result['label_match'] else 'MISMATCH'}"
    )

OUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = list(results[0].keys())

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print()
print(f"Saved: {OUT}")
