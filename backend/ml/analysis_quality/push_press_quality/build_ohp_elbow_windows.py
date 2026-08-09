import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ANNOTATIONS = Path(
    "ml/analysis_quality/fitness_aqa_ohp/source_annotations/error_elbows.json"
)

VIDEO_DIR = Path(
    "/Users/josephkamil/Desktop/Capstone/videos-OHP/videos"
)

OUTPUT = Path(
    "ml/analysis_quality/push_press_quality/"
    "ohp_elbow_windows_v2.csv"
)


def video_duration(path):
    cap = cv2.VideoCapture(str(path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    cap.release()

    if fps <= 0 or frames <= 0:
        return 0.0

    return float(frames / fps)


def assign_splits(video_ids):
    ids = list(video_ids)
    random.shuffle(ids)

    n = len(ids)
    train_end = int(round(n * 0.70))
    val_end = train_end + int(round(n * 0.15))

    mapping = {}

    for video_id in ids[:train_end]:
        mapping[video_id] = "train"

    for video_id in ids[train_end:val_end]:
        mapping[video_id] = "val"

    for video_id in ids[val_end:]:
        mapping[video_id] = "test"

    return mapping


data = json.loads(ANNOTATIONS.read_text())

positive = []
negative = []

for video_id, intervals in data.items():
    path = VIDEO_DIR / f"{video_id}.mp4"

    if not path.exists():
        continue

    if intervals:
        positive.append((video_id, path, intervals))
    else:
        negative.append((video_id, path))

# One negative video per positive video.
random.shuffle(negative)
negative = negative[:len(positive)]

all_ids = (
    [x[0] for x in positive]
    + [x[0] for x in negative]
)

splits = assign_splits(all_ids)

# Typical positive-window duration.
positive_lengths = [
    float(end) - float(start)
    for _, _, intervals in positive
    for start, end in intervals
    if float(end) > float(start)
]

window_length = float(np.median(positive_lengths))

print("median positive window:", round(window_length, 3), "seconds")

rows = []

# Use one randomly selected annotated elbow interval per positive video.
for video_id, path, intervals in positive:
    start, end = random.choice(intervals)

    rows.append({
        "video_id": video_id,
        "video_path": str(path),
        "target": "elbow_error",
        "label": 1,
        "split": splits[video_id],
        "window_start_seconds": float(start),
        "window_end_seconds": float(end),
        "source_interval_index": intervals.index([start, end]),
        "sampling_type": "ohp_positive_interval",
        "source": "ohp",
        "sample_weight": 1.0,
    })


# Generate one clean window per negative video.
for video_id, path in negative:
    duration = video_duration(path)

    if duration <= 0.25:
        continue

    length = min(window_length, duration)

    if duration <= length:
        start = 0.0
    else:
        start = random.uniform(0.0, duration - length)

    end = min(duration, start + length)

    rows.append({
        "video_id": video_id,
        "video_path": str(path),
        "target": "elbow_error",
        "label": 0,
        "split": splits[video_id],
        "window_start_seconds": round(start, 4),
        "window_end_seconds": round(end, 4),
        "source_interval_index": -1,
        "sampling_type": "ohp_clean_window",
        "source": "ohp",
        "sample_weight": 1.0,
    })

random.shuffle(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "video_id",
    "video_path",
    "target",
    "label",
    "split",
    "window_start_seconds",
    "window_end_seconds",
    "source_interval_index",
    "sampling_type",
    "source",
    "sample_weight",
]

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("saved:", OUTPUT)
print("rows:", len(rows))
print("positive:", sum(int(r["label"]) == 1 for r in rows))
print("negative:", sum(int(r["label"]) == 0 for r in rows))

for split in ("train", "val", "test"):
    subset = [r for r in rows if r["split"] == split]
    print(
        split,
        "rows:", len(subset),
        "positive:", sum(int(r["label"]) == 1 for r in subset),
        "negative:", sum(int(r["label"]) == 0 for r in subset),
    )
