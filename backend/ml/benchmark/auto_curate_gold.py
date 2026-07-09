#!/usr/bin/env python3

import csv
import json
import subprocess
from collections import defaultdict, Counter
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
verdict_counts = Counter()

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

        v7_label = data.get("exercise_label")
        v7_reps = len(data.get("rep_feedback") or [])
        v7_conf = data.get("confidence")
        analysis_mode = data.get("analysis_mode")

        router_v8 = ((data.get("debug") or {}).get("router_v8") or {})
        v8_label = router_v8.get("winner") or v7_label
        v8_conf = router_v8.get("winner_confidence")

        v7_correct = v7_label == label
        v8_correct = v8_label == label

        if v7_correct and v8_correct:
            verdict = "BOTH_PASS"
        elif (not v7_correct) and v8_correct:
            verdict = "V8_FIXED"
        elif v7_correct and (not v8_correct):
            verdict = "V8_REGRESSION"
        else:
            verdict = "BOTH_FAIL"

        verdict_counts[verdict] += 1

        print(
            f"  expected={label} "
            f"V7={v7_label} "
            f"V8={v8_label} "
            f"reps={v7_reps} "
            f"conf={v7_conf} "
            f"{verdict}"
        )

        results.append({
            "expected_label": label,
            "video": video,
            "name": name,
            "v7_label": v7_label,
            "v7_reps": v7_reps,
            "v7_confidence": v7_conf,
            "v7_analysis_mode": analysis_mode,
            "v7_label_ok": v7_correct,
            "v8_label": v8_label,
            "v8_confidence": v8_conf,
            "v8_label_ok": v8_correct,
            "verdict": verdict,
            "error": data.get("error"),
        })

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"gold_candidate_eval_{ts}.csv"
latest = OUT_DIR / "gold_candidate_eval_latest.csv"

fieldnames = [
    "expected_label",
    "video",
    "name",
    "v7_label",
    "v7_reps",
    "v7_confidence",
    "v7_analysis_mode",
    "v7_label_ok",
    "v8_label",
    "v8_confidence",
    "v8_label_ok",
    "verdict",
    "error",
]

for path in [out, latest]:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

v7_passed = sum(1 for r in results if r["v7_label_ok"])
v8_passed = sum(1 for r in results if r["v8_label_ok"])
total = len(results)

print()
print("=" * 70)
print("AUTO CURATE V7 vs V8 SUMMARY")
print("=" * 70)
print(f"V7 passed: {v7_passed}/{total} ({v7_passed / total:.1%})")
print(f"V8 passed: {v8_passed}/{total} ({v8_passed / total:.1%})")
print()
for k in ["BOTH_PASS", "V8_FIXED", "V8_REGRESSION", "BOTH_FAIL"]:
    print(f"{k:15} {verdict_counts[k]}")
print()
print(f"Saved: {out}")
print(f"Latest: {latest}")
