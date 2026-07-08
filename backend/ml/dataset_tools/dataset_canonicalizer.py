from pathlib import Path
from collections import defaultdict
import csv

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

ROOT = Path.home() / "Desktop" / "Capstone"

IGNORE = {
    "_Duplicate_Backups",
    "_Removed_Datasets",
    "FormCheck_Phase_Audit",
    "__pycache__",
}

CANONICAL_ORDER = [
    "data/dataset_v2/raw",
    "Oly_Data/raw",
    "Oly_Data/segmented",
    "PushUps",
    "PullUps",
    "HandstandPushups",
    "BenchPress",
    "deadlift",
]

OUT = Path("ml/reports/dataset_integrity")
OUT.mkdir(parents=True, exist_ok=True)

videos = []

for f in ROOT.rglob("*"):
    if any(x in f.parts for x in IGNORE):
        continue
    if f.suffix.lower() not in VIDEO_EXTS:
        continue

    videos.append({
        "path": str(f),
        "name": f.name,
        "size": f.stat().st_size,
    })

groups = defaultdict(list)

for v in videos:
    groups[(v["name"].lower(), v["size"])].append(v)

rows = []

for key, files in groups.items():

    if len(files) == 1:
        continue

    canonical = None

    for preferred in CANONICAL_ORDER:
        for f in files:
            if preferred in f["path"]:
                canonical = f["path"]
                break
        if canonical:
            break

    if canonical is None:
        canonical = sorted(x["path"] for x in files)[0]

    for f in files:
        rows.append({
            "keep": "YES" if f["path"] == canonical else "",
            "canonical": canonical,
            "duplicate": f["path"],
        })

csv_path = OUT / "canonicalization_plan.csv"

with csv_path.open("w", newline="") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=[
            "keep",
            "canonical",
            "duplicate",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 60)
print("Canonicalization plan written:")
print(csv_path)
print(f"Duplicate entries: {len(rows)}")
print("=" * 60)