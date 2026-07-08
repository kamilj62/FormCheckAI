import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import extract_video_biomechanics, build_bodyweight_features

DATASETS = {
    "handstand_push_up": Path("/Users/josephkamil/Desktop/Capstone/HandstandPushups"),
    "pull_up": Path("/Users/josephkamil/Desktop/Capstone/PullUps"),
    "push_up": Path("/Users/josephkamil/Desktop/Capstone/PushUps"),
}

EXTRA = {
    "pull_up": Path("/Users/josephkamil/Desktop/Capstone/pull Up"),
    "push_up": Path("/Users/josephkamil/Desktop/Capstone/push-up"),
}

OUT = ROOT / "ml/reports/bodyweight_router_features.csv"

FEATURES = [
    "total_frames",
    "wrist_above_shoulder_ratio",
    "wrist_below_shoulder_ratio",
    "mean_wrist_minus_shoulder_y",
    "mean_hip_minus_shoulder_y",
    "mean_knee_minus_hip_y",
    "median_head_drop",
    "avg_wrist_forward",
    "wrist_y_range",
    "shoulder_y_range",
    "hip_y_range",
    "elbow_range",
    "min_elbow",
    "max_elbow",
    "avg_elbow",
    "avg_torso_angle",
]

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".m4v"}

def iter_videos(folder):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])

rows = []

for label, folder in DATASETS.items():
    videos = iter_videos(folder)
    if label in EXTRA:
        videos += iter_videos(EXTRA[label])

    print(f"\n{label}: {len(videos)} videos", flush=True)

    for video in videos:
        print("Extracting", label, video.name, flush=True)
        try:
            extracted = extract_video_biomechanics(str(video))

            # extract_video_biomechanics returns multiple values.
            # The biomechanics list is the last item.
            if isinstance(extracted, (tuple, list)) and len(extracted) >= 2:
                # extract_video_biomechanics returns:
                # features, biomechanics, video_debug
                biomech = extracted[1]
            else:
                biomech = extracted

            feats = build_bodyweight_features(biomech)

            row = {
                "label": label,
                "video": str(video),
                "valid": int(bool(biomech)),
                "error": "",
            }

            for k in FEATURES:
                row[k] = feats.get(k, 0.0)

            rows.append(row)

        except Exception as e:
            row = {
                "label": label,
                "video": str(video),
                "valid": 0,
                "error": str(e),
            }
            for k in FEATURES:
                row[k] = 0.0
            rows.append(row)

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "video", "valid", "error"] + FEATURES)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {OUT}")
print(f"Rows: {len(rows)}")
