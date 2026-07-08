from pathlib import Path
from collections import defaultdict
import csv

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ROOT = Path.home() / "Desktop" / "Capstone"
OUT = Path("ml/reports/dataset_integrity")
OUT.mkdir(parents=True, exist_ok=True)

IGNORE_PARTS = {"_Duplicate_Backups", "_Removed_Datasets"}

LABEL_MAP = [
    ("handstandpushups", "handstand_push_up"),
    ("handstand", "handstand_push_up"),
    ("cleanandjerk", "clean_and_jerk"),
    ("clean_and_jerk", "clean_and_jerk"),
    ("benchpress", "bench_press"),
    ("bench_press", "bench_press"),
    ("bench press", "bench_press"),
    ("pullups", "pull_up"),
    ("pullup", "pull_up"),
    ("pull up", "pull_up"),
    ("pushups", "push_up"),
    ("pushup", "push_up"),
    ("push-up", "push_up"),
    ("split_jerk", "split_jerk"),
    ("splitjerk", "split_jerk"),
    ("strict_press", "strict_press"),
    ("push_press", "push_press"),
    ("squat_back", "squat_back"),
    ("squat_front", "squat_front"),
    ("overhead_squat", "overhead_squat"),
    ("deadlift", "deadlift"),
    ("snatch", "snatch"),
    ("clean", "clean"),
    ("jerk", "split_jerk"),
    ("not_oly", "not_oly"),
]

def label_for(path: Path) -> str:
    text = " ".join(part.lower() for part in path.parts)
    compact = text.replace("_", "").replace("-", "").replace(" ", "")
    for key, label in LABEL_MAP:
        k = key.lower()
        kc = k.replace("_", "").replace("-", "").replace(" ", "")
        if k in text or kc in compact:
            return label
    return "unknown"

videos = []

for p in ROOT.rglob("*"):
    if any(part in IGNORE_PARTS for part in p.parts):
        continue
    if p.suffix.lower() not in VIDEO_EXTS:
        continue
    try:
        st = p.stat()
    except Exception:
        continue

    videos.append({
        "path": str(p),
        "name": p.name,
        "size": st.st_size,
        "parent": p.parent.name,
        "label": label_for(p),
    })

by_label = defaultdict(list)
by_name_size = defaultdict(list)

for row in videos:
    by_label[row["label"]].append(row)
    by_name_size[(row["name"].lower(), row["size"])].append(row)

duplicates = []
conflicts = []

for (name, size), rows in by_name_size.items():
    if len(rows) <= 1:
        continue

    labels = sorted({r["label"] for r in rows})
    for r in rows:
        out = dict(r)
        out["copies"] = len(rows)
        out["labels_in_group"] = ",".join(labels)
        duplicates.append(out)

    if len(labels) > 1:
        for r in rows:
            out = dict(r)
            out["copies"] = len(rows)
            out["labels_in_group"] = ",".join(labels)
            conflicts.append(out)

summary_path = OUT / "summary.csv"
dupes_path = OUT / "duplicates_by_name_size.csv"
conflicts_path = OUT / "conflicting_label_duplicates.csv"
unknown_path = OUT / "unknown_videos.csv"

with summary_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "count"])
    for label, rows in sorted(by_label.items(), key=lambda x: len(x[1]), reverse=True):
        w.writerow([label, len(rows)])

def write_rows(path, rows):
    fields = ["label", "name", "size", "copies", "labels_in_group", "parent", "path"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

write_rows(dupes_path, duplicates)
write_rows(conflicts_path, conflicts)

with unknown_path.open("w", newline="") as f:
    fields = ["label", "name", "size", "parent", "path"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in videos:
        if r["label"] == "unknown":
            w.writerow({k: r.get(k, "") for k in fields})

print("\nDataset Integrity Audit")
print("=" * 60)
print(f"Active videos: {len(videos)}")
print(f"Labels: {len(by_label)}")
print(f"Duplicate groups: {sum(1 for v in by_name_size.values() if len(v) > 1)}")
print(f"Duplicate files: {len(duplicates)}")
print(f"Conflicting-label duplicate files: {len(conflicts)}")
print(f"Unknown videos: {len(by_label.get('unknown', []))}")
print("=" * 60)
print(f"Summary:    {summary_path}")
print(f"Duplicates: {dupes_path}")
print(f"Conflicts:  {conflicts_path}")
print(f"Unknown:    {unknown_path}")
