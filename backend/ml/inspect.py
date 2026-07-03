"""
Inspect benchmark failures with full routing debug.

Usage:
    python3 -m ml.inspect ml/reports/olympic_benchmark_latest.csv
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOTS = {
    "clean_and_jerk": Path("/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/clean_and_jerk"),
    "snatch": Path("/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/snatch_mp4"),
}


def analyze(path):
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 -m ml.inspect benchmark.csv")
        return

    df = pd.read_csv(sys.argv[1])

    mistakes = df[df.true_label != df.predicted_label]

    print(f"\nFound {len(mistakes)} mistakes\n")

    for _, row in mistakes.iterrows():

        path = ROOTS[row.true_label] / row.video

        print("=" * 70)
        print(path.name)
        print("=" * 70)

        data = analyze(path)

        dbg = data.get("debug", {})
        rv5 = dbg.get("router_v5", {})

        print("True label :", row.true_label)
        print("Predicted  :", data.get("exercise_label"))
        print("Confidence :", data.get("confidence"))
        print()

        print("Base label :", dbg.get("original_prediction"))
        print("Olympic    :", dbg.get("olympic_prediction"))
        print("Final      :", dbg.get("final_label"))
        print()

        print("Gate")
        print(json.dumps(dbg.get("oly_gate", {}), indent=2))

        print("\nRouter V5")
        print(json.dumps(rv5, indent=2))

        print()

if __name__ == "__main__":
    main()
