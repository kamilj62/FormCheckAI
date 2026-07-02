"""
Run Olympic router benchmark against fixed audit folders.
"""

from pathlib import Path
import subprocess
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
        "curl", "-s", "--max-time", "300",
        "-X", "POST",
        "-F", f"file=@{path}",
        "http://localhost:8000/analyze",
    ]
    return subprocess.check_output(cmd, text=True)


def main():
    import json

    rows = []

    for bench in BENCHMARKS:
        files = sorted(Path(bench["folder"]).glob(bench["pattern"]))[:bench["limit"]]

        for f in files:
            print("Testing", bench["name"], f.name)
            data = json.loads(analyze_video(f))

            rows.append({
                "true_label": bench["name"],
                "video": f.name,
                "predicted_label": data.get("exercise_label"),
                "confidence": data.get("confidence"),
            })

    df = pd.DataFrame(rows)
    out = Path("ml/reports/olympic_benchmark_latest.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print()
    print(pd.crosstab(df["true_label"], df["predicted_label"]))

    correct = (df["true_label"] == df["predicted_label"]).sum()
    total = len(df)
    print(f"\nOverall: {correct}/{total} ({100*correct/total:.1f}%)")

    print("\nMistakes:")
    print(df[df["true_label"] != df["predicted_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
