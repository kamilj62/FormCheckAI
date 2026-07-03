"""
Olympic routing benchmark.

Produces:
- CSV of predictions
- Confusion matrix
- Per-class accuracy
- Overall accuracy
"""

from pathlib import Path
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

        rows.append(
            {
                "true_label": bench["name"],
                "predicted_label": data.get("exercise_label"),
                "confidence": data.get("confidence"),
                "video": f.name,
            }
        )

df = pd.DataFrame(rows)

Path("ml/reports").mkdir(parents=True, exist_ok=True)

csv_path = Path("ml/reports/olympic_benchmark_latest.csv")
df.to_csv(csv_path, index=False)

print("\nSaved:", csv_path)

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

print("\n==============================")
print(f"OVERALL: {100*overall:.1f}%")
print("==============================")

mistakes = df[df.true_label != df.predicted_label]

print("\n==============================")
print("MISTAKES")
print("==============================")

if len(mistakes):
    print(mistakes.to_string(index=False))
else:
    print("None 🎉")
