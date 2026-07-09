#!/usr/bin/env python3

import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CANDIDATES = Path("ml/benchmark/config/gold_candidates.csv")
OUT_DIR = Path("ml/benchmark/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_LABEL = 10


def analyze(video):
    cmd = [
        "curl", "--max-time", "300", "-s", "-X", "POST",
        "-F", f"file=@{video}",
        "http://localhost:8000/analyze",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        return {"error": r.stderr}

    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": "invalid_json", "raw": r.stdout[:300]}


rows = list(csv.DictReader(CANDIDATES.open()))
groups = defaultdict(list)

for row in rows:
    groups[row["label"]].append(row)

results = []

for label in sorted(groups):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    for row in groups[label][:MAX_PER_LABEL]:
        video = row["video"]
        name = Path(video).name

        print(f"Analyzing {name}...")

        data = analyze(video)

        got_label = data.get("exercise_label")
        got_reps = len(data.get("rep_feedback") or [])
        confidence = data.get("confidence")
        analysis_mode = data.get("analysis_mode")
        ok_label = got_label == label

        print(f"  expected={label} got={got_label} reps={got_reps} conf={confidence} {'PASS' if ok_label else 'FAIL'}")

        results.append({
            "expected_label": label,
            "video": video,
            "name": name,
            "got_label": got_label,
            "got_reps": got_reps,
            "confidence": confidence,
            "analysis_mode": analysis_mode,
            "label_ok": ok_label,
            "error": data.get("error"),
        })

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"gold_candidate_eval_{ts}.csv"
latest = OUT_DIR / "gold_candidate_eval_latest.csv"

fieldnames = [
    "expected_label",
    "video",
    "name",
    "got_label",
    "got_reps",
    "confidence",
    "analysis_mode",
    "label_ok",
    "error",
]

for path in [out, latest]:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

passed = sum(1 for r in results if r["label_ok"])
total = len(results)

print()
print("=" * 70)
print("AUTO CURATE SUMMARY")
print("=" * 70)
print(f"Passed labels: {passed}/{total}")
print(f"Saved: {out}")
print(f"Latest: {latest}")
