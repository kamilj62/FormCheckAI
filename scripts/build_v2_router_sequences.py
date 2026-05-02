import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "dataset_v2" / "keypoints" / "master_keypoints.csv"
OUT_DIR = BASE_DIR / "data" / "dataset_v2" / "processed_router"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
TARGET_LABELS = ["squat_front", "strict_press"]
IGNORE_COLS = {"label", "source_video", "frame"}


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # only the two new lifts
    df = df[df["label"].isin(TARGET_LABELS)].copy()

    feature_cols = [c for c in df.columns if c not in IGNORE_COLS]

    labels = sorted(df["label"].unique())
    label_map = {label: i for i, label in enumerate(labels)}

    X = []
    y = []

    grouped = df.groupby(["label", "source_video"])
    total_groups = len(grouped)

    print("Groups:", total_groups)
    print("Labels:", label_map)

    for i, ((label, source_video), group) in enumerate(grouped, start=1):
        group = group.sort_values("frame")
        feats = group[feature_cols].values.astype(np.float32)

        if len(feats) < SEQ_LEN:
            continue

        for start in range(0, len(feats) - SEQ_LEN + 1, SEQ_LEN):
            window = feats[start:start + SEQ_LEN]

            if len(window) == SEQ_LEN:
                X.append(window)
                y.append(label_map[label])

        if i % 25 == 0:
            print(f"{i}/{total_groups}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)

    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\nDONE")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()