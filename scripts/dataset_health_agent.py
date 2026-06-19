from pathlib import Path
from collections import defaultdict

ROOTS = [
    Path("/Users/josephkamil/Desktop/Capstone/data"),
    Path("/Users/josephkamil/Desktop/Capstone/Oly_Data"),
]

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".m4v", ".MOV", ".MP4", ".AVI"}

counts = defaultdict(int)
examples = defaultdict(list)

for root in ROOTS:
    if not root.exists():
        continue

    for file in root.rglob("*"):
        if file.suffix in VIDEO_EXTS:
            movement = file.parent.name
            counts[movement] += 1
            if len(examples[movement]) < 3:
                examples[movement].append(str(file))

print("=" * 60)
print("FORMCHECK DATASET HEALTH AGENT")
print("=" * 60)

total = sum(counts.values())
print(f"\nTotal videos: {total}")
print(f"Movement classes: {len(counts)}\n")

for movement, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    status = "OK"
    if count < 20:
        status = "LOW"
    elif count < 50:
        status = "WATCH"

    print(f"{movement:35} {count:5}  {status}")

print("\nLOW-DATA CLASSES")
for movement, count in sorted(counts.items(), key=lambda x: x[1]):
    if count < 50:
        print(f"\n{movement} ({count})")
        for ex in examples[movement]:
            print(f"  - {ex}")

print("\nDone.")