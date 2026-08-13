from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/Users/josephkamil/Desktop/Capstone")
OUT = Path("/Users/josephkamil/Downloads/formcheck_main_merge/capstone_video_manifest.csv")
SUMMARY_OUT = Path("/Users/josephkamil/Downloads/formcheck_main_merge/capstone_video_summary.json")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm"}
TARGET_ROOTS = [
    "exercises-dataset/videos",
    "strict_press_clips",
    "front_squat_candidates",
    "deadlift",
    "holdout_v1",
    "videos-OHP",
    "uindy_external_validation/videos",
    "videos-squat",
    "FormCheck_Data/raw",
    "FormCheck_Phase_Audit_v2",
    "FormCheck_Phase_Audit",
    "strict_press_sources",
    "BackSquat_Audit",
    "BackSquat_Real_Audit",
    "PushUps",
    "PullUps",
    "HandstandPushups",
    "BenchPress",
    "bench press",
    "pull Up",
    "push-up",
    "fresh_validation_2026_08/videos",
    "data/raw_videos",
    "data/raw",
    "data/downloads",
    "router_challenge_v1/raw",
    "router_challenge_v1/review",
    "router_challenge_v1/regression_cases",
    "router_challenge_v1/hard_negatives",
    "internet_external_validation",
    "Oly_Data/raw",
    "Oly_Data/segmented",
    "Oly_Data/datasets_v3",
    "_Removed_Datasets/bad_video_files",
]

PRUNE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "Pods",
    "DerivedData",
    ".expo",
    ".next",
    "build",
    "dist",
}


def indexed_paths() -> list[Path]:
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", str(ROOT), "kMDItemContentTypeTree == public.movie"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return []
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def walked_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.suffix.lower() in VIDEO_EXTS:
                paths.append(p)
    return paths


def row_for(p: Path) -> dict[str, object] | None:
    try:
        st = p.stat()
        rel = p.relative_to(ROOT)
    except Exception:
        return None
    return {
        "path": str(p),
        "relative_path": str(rel),
        "name": p.name,
        "folder": str(rel.parent),
        "top_folder": rel.parts[0] if len(rel.parts) > 1 else "(root)",
        "extension": p.suffix.lower(),
        "size_mb": round(st.st_size / (1024 * 1024), 2),
        "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def main() -> None:
    seen: set[Path] = set()
    paths: list[Path] = []
    target_paths = [ROOT / rel for rel in TARGET_ROOTS if (ROOT / rel).exists()]
    sources = [indexed_paths(), *[walked_paths(target) for target in target_paths]]
    for source_paths in sources:
        for p in source_paths:
            if p.suffix.lower() not in VIDEO_EXTS or p in seen:
                continue
            seen.add(p)
            paths.append(p)

    rows = [row for p in paths if (row := row_for(p))]
    rows.sort(key=lambda r: (str(r["folder"]).lower(), str(r["name"]).lower()))

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "relative_path",
                "name",
                "folder",
                "top_folder",
                "extension",
                "size_mb",
                "modified",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_ext = collections.Counter(str(r["extension"]) for r in rows)
    by_top = collections.Counter(str(r["top_folder"]) for r in rows)
    by_folder = collections.Counter(str(r["folder"]) for r in rows)
    total_mb = sum(float(r["size_mb"]) for r in rows)
    largest = sorted(rows, key=lambda r: float(r["size_mb"]), reverse=True)[:20]
    recent = sorted(rows, key=lambda r: str(r["modified"]), reverse=True)[:20]

    summary = {
        "root": str(ROOT),
        "manifest": str(OUT),
        "count": len(rows),
        "total_gb": round(total_mb / 1024, 2),
        "by_extension": dict(by_ext.most_common()),
        "top_folders": by_top.most_common(20),
        "folders_with_most_videos": by_folder.most_common(25),
        "largest_20": largest,
        "most_recent_20": recent,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))

    print(f"Found {len(rows)} video files")
    print(f"Total size: {round(total_mb / 1024, 2)} GB")
    print(f"Manifest: {OUT}")
    print(f"Summary: {SUMMARY_OUT}")
    print("\nBy extension:")
    for ext, count in by_ext.most_common():
        print(f"  {ext}: {count}")
    print("\nTop folders:")
    for folder, count in by_top.most_common(15):
        print(f"  {folder}: {count}")
    print("\nLargest files:")
    for r in largest[:10]:
        print(f"  {float(r['size_mb']):>9.2f} MB  {r['relative_path']}")
    print("\nMost recent files:")
    for r in recent[:10]:
        print(f"  {r['modified']}  {r['relative_path']}")


if __name__ == "__main__":
    main()
