import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/josephkamil/Desktop/Capstone")
CSV_PATH = BASE_DIR / "Oly_Data/oly_keypoints.csv"
MODEL_PATH = BASE_DIR / "models/oly_router_rf.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

df = pd.read_csv(CSV_PATH)
feature_cols = bundle["feature_cols"]

# Pick one known clean_and_jerk video
sample = df[df["label"] == "not_oly"]["video"].iloc[0]

group = df[df["video"] == sample].sort_values("frame")
arr = group[feature_cols].values.astype("float32")

X = np.concatenate([
    arr.mean(axis=0),
    arr.std(axis=0),
    arr.min(axis=0),
    arr.max(axis=0),
]).reshape(1, -1)

pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]

print("Video:", sample)
print("True label:", group["label"].iloc[0])
print("Prediction:", pred)

for label, prob in zip(model.classes_, probs):
    print(f"{label}: {prob:.3f}")