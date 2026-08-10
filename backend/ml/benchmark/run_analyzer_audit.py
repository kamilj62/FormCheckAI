#!/usr/bin/env python3

import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CANDIDATES = Path("ml/benchmark/config/analyzer_audit_small.csv")
OUT_DIR = Path("ml/benchmark/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze(video):
    cmd = [
        "curl",
        "--max-time", "300",
        "-s",
        "-X", "POST",
        "-F", f"file=@{video}",
        "http://127.0.0.1:8000/analyze",
    ]

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if r.returncode != 0:
        return {
            "error": r.stderr.strip() or f"curl_exit_{r.returncode}"
        }

    try:
        return json.loads(r.stdout)
    except Exception:
        return {
            "error": "invalid_json",
            "raw": r.stdout[:500],
        }

rows = list(csv.DictReader(CANDIDATES.open()))

results = []
label_stats = defaultdict(Counter)
confusions = Counter()

latest = OUT_DIR / "analyzer_audit_latest.csv"

fieldnames = [
    "expected_label",
    "video",
    "name",
    "predicted_label",
    "label_ok",
    "confidence",
    "detected_reps",
    "expected_reps",
    "rep_count_ok",
    "analysis_mode",
    "protected_reason",
    "raw_label",
    "bio_label",
    "squat_label",
    "olympic_label",
    "router_v8_winner",
    "error",
]

completed = {}

if latest.exists():
    try:
        for prior in csv.DictReader(latest.open()):
            video = prior.get("video")
            error = (prior.get("error") or "").strip()

            predicted = (prior.get("predicted_label") or "").strip()

            # Keep only genuine completed analyses.
            # Retry timeouts / empty analyzer responses.
            if video and not error and predicted:
                completed[video] = prior

        print(f"Resume: keeping {len(completed)} valid completed rows")
    except Exception:
        completed = {}

def process_row(index, row):
    expected = row["label"]
    video = row["video"]
    name = row["name"]

    if not Path(video).exists():
        return {
            "expected_label": expected,
            "video": video,
            "name": name,
            "predicted_label": "",
            "label_ok": False,
            "confidence": "",
            "detected_reps": "",
            "expected_reps": row.get("expected_reps") or "",
            "rep_count_ok": "",
            "analysis_mode": "",
            "protected_reason": "",
            "raw_label": "",
            "bio_label": "",
            "squat_label": "",
            "olympic_label": "",
            "router_v8_winner": "",
            "error": "missing_file",
        }

    data = analyze(video)

    predicted = data.get("exercise_label")
    confidence = data.get("confidence")
    reps = len(data.get("rep_feedback") or [])
    analysis_mode = data.get("analysis_mode")
    debug = data.get("debug") or {}

    expected_reps_raw = (row.get("expected_reps") or "").strip()
    rep_count_ok = ""

    if expected_reps_raw:
        try:
            expected_reps = int(expected_reps_raw)
            rep_count_ok = reps == expected_reps
        except Exception:
            expected_reps = expected_reps_raw
    else:
        expected_reps = ""

    router_v8 = debug.get("router_v8") or {}

    return {
        "expected_label": expected,
        "video": video,
        "name": name,
        "predicted_label": predicted,
        "label_ok": predicted == expected,
        "confidence": confidence,
        "detected_reps": reps,
        "expected_reps": expected_reps,
        "rep_count_ok": rep_count_ok,
        "analysis_mode": analysis_mode,
        "protected_reason": debug.get("protected_reason"),
        "raw_label": debug.get("raw_label"),
        "bio_label": debug.get("bio_label"),
        "squat_label": debug.get("squat_label"),
        "olympic_label": debug.get("olympic_pred"),
        "router_v8_winner": (
            router_v8.get("winner")
            if isinstance(router_v8, dict)
            else None
        ),
        "error": data.get("error"),
    }

pending = [
    (i, row)
    for i, row in enumerate(rows, 1)
    if row["video"] not in completed
]

print(f"Total candidates: {len(rows)}")
print(f"Already completed: {len(completed)}")
print(f"Pending: {len(pending)}")
print("Workers: 1")

all_results = list(completed.values())

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {
        executor.submit(process_row, i, row): (i, row)
        for i, row in pending
    }

    for future in as_completed(futures):
        i, row = futures[future]

        try:
            result = future.result()
        except Exception as exc:
            result = {
                "expected_label": row["label"],
                "video": row["video"],
                "name": row["name"],
                "predicted_label": "",
                "label_ok": False,
                "confidence": "",
                "detected_reps": "",
                "expected_reps": row.get("expected_reps") or "",
                "rep_count_ok": "",
                "analysis_mode": "",
                "protected_reason": "",
                "raw_label": "",
                "bio_label": "",
                "squat_label": "",
                "olympic_label": "",
                "router_v8_winner": "",
                "error": f"exception:{exc}",
            }

        all_results.append(result)

        print(
            f"[{len(all_results)}/{len(rows)}] "
            f"{result['expected_label']} :: {result['name']} "
            f"-> {result['predicted_label']} "
            f"reps={result['detected_reps']} "
            f"{'PASS' if result['label_ok'] else 'FAIL'}"
        )

        with latest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

results = all_results

for r in results:
    expected = r["expected_label"]

    if r.get("error") == "missing_file":
        label_stats[expected]["missing"] += 1
        continue

    if str(r.get("label_ok")).lower() in {"true", "1", "yes"}:
        label_stats[expected]["pass"] += 1
    else:
        label_stats[expected]["fail"] += 1
        confusions[
            (
                expected,
                r.get("predicted_label") or "None",
            )
        ] += 1

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

out = OUT_DIR / f"analyzer_audit_{ts}.csv"
latest = OUT_DIR / "analyzer_audit_latest.csv"

total = len(results)
passed = sum(1 for r in results if r["label_ok"])
failed = total - passed

print()
print("=" * 72)
print("ANALYZER AUDIT SUMMARY")
print("=" * 72)
print(f"Total:  {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if total:
    print(f"Label accuracy: {passed / total:.1%}")

print()
print("BY MOVEMENT")
print("-" * 72)

for label in sorted(label_stats):
    p = label_stats[label]["pass"]
    f = label_stats[label]["fail"]
    m = label_stats[label]["missing"]
    denom = p + f

    accuracy = (
        f"{p / denom:.1%}"
        if denom
        else "n/a"
    )

    print(
        f"{label:24} "
        f"pass={p:3} "
        f"fail={f:3} "
        f"missing={m:3} "
        f"accuracy={accuracy}"
    )

print()
print("TOP CONFUSIONS")
print("-" * 72)

for (expected, predicted), count in confusions.most_common(25):
    print(
        f"{expected:22} -> "
        f"{predicted:22} "
        f"{count}"
    )

print()
print(f"Saved:  {out}")
print(f"Latest: {latest}")
