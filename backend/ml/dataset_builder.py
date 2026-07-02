"""
Build one-row-per-video feature datasets.

Usage:
python -m ml.dataset_builder \
  --input /path/to/videos \
  --label snatch \
  --output ml/datasets/snatch_video_dataset.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from app.main import extract_video_biomechanics
from app.feature_engine.movement_video_features import build_movement_video_features
from app.feature_engine.feature_names import FEATURE_NAMES


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v"}


def iter_videos(input_folder: str):
    folder = Path(input_folder).expanduser()
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def build_dataset(input_folder: str, label: str, output_csv: str):
    rows = []
    failures = []

    videos = iter_videos(input_folder)
    print(f"Found {len(videos)} videos")

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        try:
            sequence, biomechanics, debug = extract_video_biomechanics(str(video))

            if len(sequence) < 10 or len(biomechanics) < 10:
                failures.append({
                    "video": video.name,
                    "reason": "insufficient_data",
                    **debug,
                })
                continue

            features = build_movement_video_features(biomechanics)

            row = {
                "video": video.name,
                "path": str(video),
                "label": label,
                "frames_processed": debug.get("frames_processed", len(sequence)),
                "pose_frames": debug.get("pose_frames", len(biomechanics)),
                "total_frames": debug.get("total_frames", 0),
            }

            for name, value in zip(FEATURE_NAMES, features):
                row[name] = float(value)

            rows.append(row)

        except Exception as e:
            failures.append({
                "video": video.name,
                "reason": str(e),
            })

    out = Path(output_csv).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out, index=False)

    fail_out = out.with_name(out.stem + "_failures.csv")
    pd.DataFrame(failures).to_csv(fail_out, index=False)

    print()
    print("Saved:", out)
    print("Rows:", len(rows))
    print("Failures:", len(failures))
    print("Failure log:", fail_out)

    return rows, failures


def build_multiple_datasets(config):
    all_rows = []
    all_failures = []

    for label, cfg in config.items():
        rows, failures = build_dataset(
            input_folder=cfg["input_folder"],
            label=label,
            output_csv=cfg["output_csv"],
        )
        all_rows.extend(rows)
        all_failures.extend(failures)

    return all_rows, all_failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_dataset(args.input, args.label, args.output)


if __name__ == "__main__":
    main()
