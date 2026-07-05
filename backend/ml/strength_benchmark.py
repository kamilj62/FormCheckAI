from pathlib import Path
from datetime import datetime
import subprocess
import json
import os
import sys

script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

BENCHMARKS = [
    {
        "name": "bench_press",
        "folders": [
            "/Users/josephkamil/Desktop/Capstone/BenchPress",
            "/Users/josephkamil/Desktop/Capstone/bench press",
        ],
        "patterns": ["*.mp4", "*.mov", "*.avi"],
    },
    {
        "name": "deadlift",
        "folders": [
            "/Users/josephkamil/Desktop/Capstone/deadlift",
        ],
        "patterns": ["*.mp4", "*.mov", "*.avi"],
    },
]


def analyze_video(path):
    cmd = [
        "curl",
        "-s",
        "--max-time",
        "180",
        "-X",
        "POST",
        "-F",
        f"file=@{path}",
        f"{API_URL.rstrip('/')}/analyze",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=200,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "request_timeout", "analysis_mode": "error"}

    if result.returncode != 0:
        return {
            "error": f"curl_failed_{result.returncode}",
            "stderr": result.stderr[-500:],
            "analysis_mode": "error",
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "error": "invalid_json",
            "response": result.stdout[-500:],
            "analysis_mode": "error",
        }


rows = []

for bench in BENCHMARKS:
    files = []

    for folder in bench["folders"]:
        folder = Path(folder)

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for pattern in bench["patterns"]:
            files.extend(folder.glob(pattern))

    files = sorted(set(files))

    print(f"\nFound {len(files)} files for {bench['name']}")

    for f in files:
        print("Testing", bench["name"], f.name, flush=True)
        data = analyze_video(f)

        rows.append({
            "true_label": bench["name"],
            "predicted_label": data.get("exercise_label"),
            "confidence": data.get("confidence"),
            "analysis_mode": data.get("analysis_mode"),
            "error": data.get("error"),
            "valid_extraction": (
                data.get("error") is None
                and data.get("analysis_mode") not in {"insufficient_data", "error"}
                and data.get("exercise_label") not in {None, "Unknown"}
            ),
            "video": str(f),
            "rep_count": len(data.get("rep_feedback", [])),
        })


df = pd.DataFrame(rows)

Path("ml/reports").mkdir(parents=True, exist_ok=True)

csv_path = Path("ml/reports/strength_benchmark_latest.csv")
df.to_csv(csv_path, index=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_dir = Path("ml/reports/history")
history_dir.mkdir(parents=True, exist_ok=True)

history_path = history_dir / f"strength_benchmark_{timestamp}.csv"
df.to_csv(history_path, index=False)

print("\nSaved:", csv_path)
print("History:", history_path)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(pd.crosstab(df.true_label, df.predicted_label))

print("\n==============================")
print("PER-CLASS ACCURACY")
print("==============================")

for label in sorted(df.true_label.unique()):
    subset = df[df.true_label == label]
    acc = (subset.true_label == subset.predicted_label).mean()
    print(f"{label:20s} {100*acc:5.1f}%")

overall = (df.true_label == df.predicted_label).mean()
valid = df[df.valid_extraction == True]
valid_overall = (valid.true_label == valid.predicted_label).mean() if len(valid) else 0

print("\n==============================")
print(f"OVERALL RAW: {100*overall:.1f}%")
print(f"VALID ONLY: {100*valid_overall:.1f}% ({len(valid)}/{len(df)} valid)")
print("==============================")

mistakes = df[df.true_label != df.predicted_label]

print("\n==============================")
print("MISTAKES")
print("==============================")

if len(mistakes):
    print(mistakes.to_string(index=False))
else:
    print("None 🎉")
