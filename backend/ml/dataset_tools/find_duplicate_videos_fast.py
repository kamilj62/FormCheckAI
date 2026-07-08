from pathlib import Path
from collections import defaultdict

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
ROOT = Path.home() / "Desktop" / "Capstone"

groups = defaultdict(list)

print("Scanning videos...\n")

count = 0

for p in ROOT.rglob("*"):
    if p.suffix.lower() not in VIDEO_EXTS:
        continue

    count += 1

    try:
        key = (p.name.lower(), p.stat().st_size)
        groups[key].append(p)
    except Exception:
        pass

dupes = {k: v for k, v in groups.items() if len(v) > 1}

print("=" * 70)
print(f"Videos scanned: {count}")
print(f"Duplicate groups: {len(dupes)}")
print(f"Duplicate files beyond first copy: {sum(len(v)-1 for v in dupes.values())}")
print("=" * 70)

for (name, size), files in sorted(
    dupes.items(),
    key=lambda x: len(x[1]),
    reverse=True
):
    print("\n" + "-" * 70)
    print(f"{name}")
    print(f"Size: {size:,} bytes")
    print(f"Copies: {len(files)}")

    for f in files:
        print(f"   {f}")
