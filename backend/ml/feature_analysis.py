import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pathlib import Path

DATASET = "ml/datasets/olympic_video_dataset_v2.csv"

df = pd.read_csv(DATASET)

IGNORE = {
    "video",
    "label",
    "frames_processed",
    "pose_frames",
    "total_frames",
}

feature_cols = [c for c in df.columns if c not in IGNORE]

rows = []

for col in feature_cols:
    if not is_numeric_dtype(df[col]):
        continue

    cj = df[df.label == "clean_and_jerk"][col].fillna(0)
    sn = df[df.label == "snatch"][col].fillna(0)

    diff = abs(cj.mean() - sn.mean())

    pooled = np.sqrt((cj.var() + sn.var()) / 2)

    effect = diff / pooled if pooled > 1e-9 else 0.0

    rows.append({
        "feature": col,
        "cj_mean": cj.mean(),
        "snatch_mean": sn.mean(),
        "difference": diff,
        "effect_size": effect,
    })

out = (
    pd.DataFrame(rows)
      .sort_values("effect_size", ascending=False)
)

Path("ml/reports").mkdir(exist_ok=True)

outfile = "ml/reports/feature_analysis.csv"
out.to_csv(outfile, index=False)

print("=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)
print(out.head(30).to_string(index=False))

print("\nSaved:", outfile)
