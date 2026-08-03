import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

from app.main import extract_features_and_biomechanics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="ml/analysis_quality/fitness_aqa_squat/train_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="ml/analysis_quality/fitness_aqa_squat/shallow_pose_smoke.jsonl",
    )
    parser.add_argument("--limit-videos", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(manifest_path.read_text())

    labeled_records = [
        record
        for record in records
        if record["annotations"]["shallow_depth"]["samples"]
    ]

    if args.limit_videos:
        labeled_records = labeled_records[: args.limit_videos]

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    written = 0
    failed = 0

    with output_path.open("w") as out:
        for record in labeled_records:
            video_id = record["video_id"]
            video_path = Path(record["video_path"])
            samples = record["annotations"]["shallow_depth"]["samples"]

            print(
                f"Processing {video_id}: "
                f"{len(samples)} labeled frames",
                flush=True,
            )

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"  VIDEO_OPEN_FAILED: {video_path}", flush=True)
                failed += len(samples)
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

            for sample in samples:
                frame_number = int(sample["frame"])
                label = int(sample["label"])

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = cap.read()

                if not ok:
                    print(
                        f"  FRAME_READ_FAILED: {frame_number}",
                        flush=True,
                    )
                    failed += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    print(
                        f"  POSE_NOT_FOUND: {frame_number}",
                        flush=True,
                    )
                    failed += 1
                    continue

                features, biomechanics = (
                    extract_features_and_biomechanics(results)
                )

                if features is None or biomechanics is None:
                    print(
                        f"  FEATURE_EXTRACTION_FAILED: {frame_number}",
                        flush=True,
                    )
                    failed += 1
                    continue

                row = {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "frame_number": frame_number,
                    "timestamp_seconds": (
                        round(frame_number / fps, 6)
                        if fps > 0
                        else None
                    ),
                    "label": label,
                    "features": [
                        float(value)
                        for value in features
                    ],
                    "biomechanics": {
                        key: (
                            float(value)
                            if isinstance(value, (int, float))
                            else value
                        )
                        for key, value in biomechanics.items()
                        if key != "full_features"
                    },
                }

                out.write(json.dumps(row) + "\n")
                written += 1

            cap.release()

    pose.close()

    print()
    print("Output:", output_path)
    print("Videos processed:", len(labeled_records))
    print("Samples written:", written)
    print("Samples failed:", failed)


if __name__ == "__main__":
    main()
