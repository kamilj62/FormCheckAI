from pathlib import Path
from collections import defaultdict

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ROOT = Path.home() / "Desktop" / "Capstone"

LABEL_MAP = {
    "bench": "bench_press",
    "benchpress": "bench_press",
    "bench press": "bench_press",
    "deadlift": "deadlift",
    "push_press": "push_press",
    "push press": "push_press",
    "strict_press": "strict_press",
    "strict press": "strict_press",
    "handstandpushups": "handstand_push_up",
    "handstand pushups": "handstand_push_up",
    "handstand": "handstand_push_up",
    "pullups": "pull_up",
    "pullup": "pull_up",
    "pull up": "pull_up",
    "pushups": "push_up",
    "pushup": "push_up",
    "push-up": "push_up",
    "squat_back": "squat_back",
    "back squat": "squat_back",
    "squat_front": "squat_front",
    "front squat": "squat_front",
    "overhead_squat": "overhead_squat",
    "overhead squat": "overhead_squat",
    "clean_and_jerk": "clean_and_jerk",
    "cleanandjerk": "clean_and_jerk",
    "clean": "clean",
    "split_jerk": "split_jerk",
    "split jerk": "split_jerk",
    "snatch": "snatch",
    "snatching": "snatch",
    "lunge": "lunge",
    "lunges": "lunge",
    "not_oly": "not_oly",
}

def normalize_folder(name):
    raw = name.lower().replace("-", "_")
    compact = raw.replace("_", "").replace(" ", "")

    for key, label in LABEL_MAP.items():
        k = key.lower()
        if k in raw or k.replace("_", "").replace(" ", "") in compact:
            return label

    return "unknown"

counts = defaultdict(int)
folders = defaultdict(set)

for f in ROOT.rglob("*"):
    if "_Duplicate_Backups" in f.parts or "_Removed_Datasets" in f.parts:
        continue
    if f.suffix.lower() not in VIDEO_EXTS:
        continue
    label = normalize_folder(f.parent.name)
    counts[label] += 1
    folders[label].add(str(f.parent))

print("\nNormalized Dataset Audit")
print("=" * 60)
print(f"TOTAL VIDEOS: {sum(counts.values())}")
print("=" * 60)

for label, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{n:5d}   {label}")
    for folder in sorted(folders[label])[:5]:
        print(f"        {folder}")
    if len(folders[label]) > 5:
        print(f"        ... {len(folders[label]) - 5} more folders")
