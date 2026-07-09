#!/usr/bin/env python3

import csv
import json
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

RESULTS = Path("ml/benchmark/results/gold_candidate_eval_latest.csv")


def analyze(video):
    cmd = [
        "curl", "--max-time", "300", "-s", "-X", "POST",
        "-F", f"file=@{video}",
        "http://localhost:8000/analyze",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": "invalid_json", "raw": r.stdout[:300]}


rows = list(csv.DictReader(RESULTS.open()))

failures = [
    r for r in rows
    if str(r.get("label_ok")).lower() not in {"true", "1", "yes"}
]

pair_counts = Counter(
    (r["expected_label"], r.get("got_label") or "None")
    for r in failures
)

representatives = OrderedDict()
for r in failures:
    pair = (r["expected_label"], r.get("got_label") or "None")
    if pair not in representatives:
        representatives[pair] = r

winner_counts = Counter()

print("=" * 80)
print("REPRESENTATIVE ROUTER FAILURE REPORT")
print("=" * 80)
print(f"Total benchmark rows: {len(rows)}")
print(f"Failures: {len(failures)}")
print(f"Unique confusion pairs: {len(representatives)}")
print()

for pair, r in representatives.items():
    expected, got = pair
    video = r["video"]

    data = analyze(video)
    debug = data.get("debug", {}) or {}

    winner = (
        debug.get("analysis_path")
        or debug.get("protected_reason")
        or data.get("analysis_mode")
        or "unknown"
    )
    winner_counts[winner] += pair_counts[pair]

    print("=" * 80)
    print(Path(video).name)
    print(f"Pair count : {pair_counts[pair]}")
    print(f"Expected   : {expected}")
    print(f"Predicted  : {data.get('exercise_label')}")
    print(f"Mode       : {data.get('analysis_mode')}")
    print(f"Confidence : {data.get('confidence')}")
    print()

    print("Base       :", debug.get("raw_label"), debug.get("base_conf"))
    print("Biomech    :", debug.get("bio_label"), debug.get("bio_conf"))
    print("Squat      :", debug.get("squat_label"), debug.get("squat_conf"))
    print("Olympic    :", debug.get("olympic_pred"), debug.get("olympic_conf"))
    print("Bodyweight :", debug.get("bodyweight_router_label"), debug.get("bodyweight_router_conf"))

    rv5 = debug.get("router_v5") or {}
    print("Router V5  :", rv5.get("decision") if isinstance(rv5, dict) else None)
    print("Winner     :", winner)
    print()

print("=" * 80)
print("FAILURES BY WINNING ROUTER / MODE")
print("=" * 80)
for k, v in winner_counts.most_common():
    print(f"{k:35} {v}")

print()
print("=" * 80)
print("MOST COMMON CONFUSION PAIRS")
print("=" * 80)
for (expected, got), count in pair_counts.most_common(30):
    print(f"{expected:20} -> {got:20} {count}")
