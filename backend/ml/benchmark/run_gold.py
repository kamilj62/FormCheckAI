#!/usr/bin/env python3

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

MANIFEST = Path("ml/benchmark/config/gold_manifest.csv")
OUT_DIR = Path("ml/benchmark/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze(video):
    cmd = [
        "curl", "--max-time", "300", "-s", "-X", "POST",
        "-F", f"file=@{video}",
        "http://localhost:8000/analyze",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": result.stderr}

    try:
        return json.loads(result.stdout)
    except Exception:
        return {"error": "invalid_json", "raw": result.stdout[:500]}


def main():
    rows = list(csv.DictReader(MANIFEST.open()))

    results = []
    passed = 0

    for r in rows:
        video = r["video"]
        expected_label = r["label"]
        expected_reps = int(r["expected_reps"])

        print(f"\nAnalyzing: {Path(video).name}")

        data = analyze(video)

        got_label = data.get("exercise_label")
        got_reps = len(data.get("rep_feedback") or [])

        label_ok = got_label == expected_label
        reps_ok = got_reps == expected_reps
        ok = label_ok and reps_ok

        if ok:
            passed += 1

        print(f"Expected: {expected_label}, reps={expected_reps}")
        print(f"Got     : {got_label}, reps={got_reps}")
        print("PASS" if ok else "FAIL")

        results.append({
            **r,
            "got_label": got_label,
            "got_reps": got_reps,
            "label_ok": label_ok,
            "reps_ok": reps_ok,
            "pass": ok,
            "analysis_mode": data.get("analysis_mode"),
            "confidence": data.get("confidence"),
            "error": data.get("error"),
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT_DIR / f"gold_results_{ts}.csv"
    latest_csv = OUT_DIR / "gold_results_latest.csv"

    fieldnames = list(results[0].keys()) if results else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with latest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    acc = passed / total if total else 0

    print()
    print("=" * 60)
    print("GOLD BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {acc:.1%}")
    print(f"Saved: {out_csv}")
    print("=" * 60)

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
