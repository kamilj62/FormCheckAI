from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from app.feature_engine.extractors import extract_features_and_biomechanics
from app.feature_engine.movement_video_features_v4 import (
    FEATURE_NAMES,
    build_movement_video_features_v4,
)


def extract_video(path: Path) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    biomechanics = []
    total_frames = 0
    pose_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            _, biomech = extract_features_and_biomechanics(results)

            if biomech is None:
                continue

            biomech["frame_number"] = total_frames - 1
            biomechanics.append(biomech)
            pose_frames += 1
    finally:
        cap.release()
        pose.close()

    if total_frames == 0:
        raise RuntimeError("Video contains no readable frames")

    if pose_frames < 10:
        raise RuntimeError(
            f"Insufficient pose frames: {pose_frames}/{total_frames}"
        )

    features = build_movement_video_features_v4(biomechanics)

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )

    if not np.isfinite(features).all():
        raise RuntimeError("Feature vector contains NaN or infinity")

    metadata = {
        "total_frames": total_frames,
        "pose_frames": pose_frames,
        "pose_ratio": pose_frames / total_frames,
    }

    return features.astype(np.float32), metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="ml/router_rebuild_v1/manifests/router_training_manifest_v1.csv",
    )
    parser.add_argument(
        "--features-out",
        default="ml/router_rebuild_v1/features/router_features_v3.csv",
    )
    parser.add_argument(
        "--errors-out",
        default="ml/router_rebuild_v1/features/router_feature_errors_v3.csv",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    features_out = Path(args.features_out)
    errors_out = Path(args.errors_out)

    df = pd.read_csv(manifest_path)

    required = {
        "path",
        "reviewed_label",
        "olympic_gate_label",
        "source_id",
        "split",
        "include",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")

    df = df[df["include"].astype(str).str.lower() == "yes"].copy()

    rows = []
    errors = []

    total = len(df)

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        video_path = Path(str(row["path"]))

        print(
            f"[{position:03d}/{total:03d}] "
            f"{row['reviewed_label']}: {video_path.name}",
            flush=True,
        )

        try:
            if not video_path.exists():
                raise FileNotFoundError(str(video_path))

            features, meta = extract_video(video_path)

            output = row.to_dict()
            output.update(meta)

            for name, value in zip(FEATURE_NAMES, features):
                output[name] = float(value)

            rows.append(output)

        except Exception as exc:
            errors.append({
                "path": str(video_path),
                "filename": row.get("filename", video_path.name),
                "folder": row.get("folder", ""),
                "reviewed_label": row.get("reviewed_label", ""),
                "source_id": row.get("source_id", ""),
                "split": row.get("split", ""),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)

    features_out.parent.mkdir(parents=True, exist_ok=True)

    feature_df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)

    feature_df.to_csv(features_out, index=False)
    error_df.to_csv(errors_out, index=False)

    print("\nExtraction complete")
    print("successful:", len(feature_df))
    print("errors:", len(error_df))
    print("features:", features_out)
    print("error report:", errors_out)

    if not feature_df.empty:
        print("\nSuccessful rows by label:")
        print(feature_df["reviewed_label"].value_counts().to_string())

        print("\nSuccessful rows by split:")
        print(feature_df["split"].value_counts().to_string())

    return 0 if len(feature_df) else 1


if __name__ == "__main__":
    sys.exit(main())
