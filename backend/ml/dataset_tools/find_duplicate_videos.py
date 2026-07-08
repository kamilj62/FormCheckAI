from pathlib import Path
from collections import defaultdict
import hashlib

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ROOT = Path.home() / "Desktop" / "Capstone"

def file_hash(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

by_hash = defaultdict(list)

videos = [p for p in ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
print(f"Scanning {len(videos)} videos...")

for i, path in enumerate(videos, 1):
    try:
        by_hash[file_hash(path)].append(path)
    except Exception as e:
        print(f"ERROR hashing {path}: {e}")

dupes = {h: files for h, files in by_hash.items() if len(files) > 1}

print("\nDuplicate Video Groups")
print("=" * 60)
print(f"Duplicate groups: {len(dupes)}")
print(f"Duplicate files beyond first copy: {sum(len(v)-1 for v in dupes.values())}")

for h, files in sorted(dupes.items(), key=lambda x: len(x[1]), reverse=True)[:50]:
    print("\n---")
    print(f"Copies: {len(files)}")
    for f in files:
        print(f"  {f}")
