from pathlib import Path
from collections import defaultdict

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

ROOT = Path.home() / "Desktop" / "Capstone"

counts = defaultdict(int)

print("\nScanning...\n")

for f in ROOT.rglob("*"):
    if f.suffix.lower() in VIDEO_EXTS:
        counts[f.parent.name] += 1

total = sum(counts.values())

print("=" * 60)
print(f"TOTAL VIDEOS: {total}")
print("=" * 60)

for folder, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{n:5d}   {folder}")