"""
Olympic routing benchmark.

Produces:
- CSV of predictions
- Confusion matrix
- Per-class accuracy
- Overall accuracy
"""

from pathlib import Path
from datetime import datetime
import subprocess
import json
import pandas as pd

BENCHMARKS = [
    {
        "name": "clean_and_jerk",
        "folder": "/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/clean_and_jerk",
        "pattern": "*.avi",
        "limit": 20,
    },
    {
        "name": "snatch",
        "folder": "/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/snatch_mp4",
        "pattern": "*.mp4",
        "limit": 20,
    },
]


def analyze_video(path):
    cmd = [
        "curl",
        "-s",
        "--max-time",
        "300",
        "-X",
        "POST",
        "-F",
        f"file=@{path}",
        "http://localhost:8000/analyze",
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


rows = []

for bench in BENCHMARKS:
    folder = Path(bench["folder"])
    files = sorted(folder.glob(bench["pattern"]))[: bench["limit"]]

    for f in files:
        print("Testing", bench["name"], f.name)

        data = analyze_video(f)

        record = {
            "true_label": bench["name"],
            "predicted_label": data.get("exercise_label"),
            "confidence": data.get("confidence"),
            "analysis_mode": data.get("analysis_mode"),
            "frames_processed": data.get("debug", {}).get("frames_processed"),
            "valid_extraction": data.get("analysis_mode") != "insufficient_data",
            "video": f.name,
        }

        record["rep_count"] = len(data.get("rep_feedback", []))

        if record["rep_count"]:
            rep = data["rep_feedback"][0]

            phase_keys = [
                "start_frame",
                "first_pull_frame",
                "extension_frame",
                "catch_frame",
                "end_frame",
                "clean_catch_frame",
                "clean_recovery_frame",
                "jerk_dip_frame",
                "jerk_drive_frame",
                "jerk_catch_frame",
                "dip_frame",
                "drive_frame",
                "lockout_frame",
                "bottom_frame",
                "ascent_frame",
            ]

            for key in phase_keys:
                record[key] = rep.get(key)

        rows.append(record)

df = pd.DataFrame(rows)

Path("ml/reports").mkdir(parents=True, exist_ok=True)

csv_path = Path("ml/reports/olympic_benchmark_latest.csv")
df.to_csv(csv_path, index=False)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_dir = Path("ml/reports/history")
history_dir.mkdir(parents=True, exist_ok=True)
history_path = history_dir / f"olympic_benchmark_{timestamp}.csv"
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
print(f"OVERALL: {100*overall:.1f}%")
print(f"VALID ONLY: {100*valid_overall:.1f}% ({len(valid)}/{len(df)} valid)")
print("==============================")

mistakes = df[df.true_label != df.predicted_label]

# ---------------- Phase sanity checks ----------------
phase_errors = []

for _, row in df.iterrows():
    label = row.predicted_label

    if label == "clean":
        frames = [
            row.get("start_frame"),
            row.get("first_pull_frame"),
            row.get("extension_frame"),
            row.get("catch_frame"),
            row.get("end_frame"),
        ]
    elif label == "clean_and_jerk":
        frames = [
            row.get("start_frame"),
            row.get("clean_catch_frame"),
            row.get("clean_recovery_frame"),
            row.get("jerk_dip_frame"),
            row.get("jerk_drive_frame"),
            row.get("jerk_catch_frame"),
            row.get("end_frame"),
        ]
    elif label == "snatch":
        frames = [
            row.get("start_frame"),
            row.get("first_pull_frame"),
            row.get("extension_frame"),
            row.get("catch_frame"),
            row.get("end_frame"),
        ]
    else:
        continue

    frames = [f for f in frames if pd.notna(f)]

    if len(frames) >= 2 and any(frames[i] > frames[i + 1] for i in range(len(frames) - 1)):
        phase_errors.append(row.video)

print("\n==============================")
print("PHASE ORDER")
print("==============================")
print(f"Failures: {len(phase_errors)}")

for video in phase_errors:
    print(video)

print("\n==============================")
print("MISTAKES")
print("==============================")

if len(mistakes):
    print(mistakes.to_string(index=False))
else:
    print("None 🎉")
