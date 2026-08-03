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
        default="ml/analysis_quality/fitness_aqa_squat/knee_pose_smoke.jsonl",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=5,
    )

    return parser.parse_args()


def timestamp_in_intervals(timestamp, intervals):
    return int(
        any(
            float(start) <= timestamp <= float(end)
            for start, end in intervals
        )
    )


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(manifest_path.read_text())

    if args.limit_videos > 0:
        records = records[:args.limit_videos]

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    written = 0
    failed = 0

    forward_positive = 0
    inward_positive = 0

    with output_path.open("w") as out:
        for index, record in enumerate(records, start=1):
            video_id = record["video_id"]
            video_path = Path(record["video_path"])

            forward_intervals = record["annotations"][
                "knees_forward"
            ]["intervals"]

            inward_intervals = record["annotations"][
                "knees_inward"
            ]["intervals"]

            print(
                f"[{index}/{len(records)}] {video_id}",
                flush=True,
            )

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"  VIDEO_OPEN_FAILED: {video_path}", flush=True)
                failed += 1
                continue

            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if source_fps <= 0 or frame_count <= 0:
                print("  INVALID_VIDEO_METADATA", flush=True)
                cap.release()
                failed += 1
                continue

            frame_step = max(
                1,
                round(source_fps / args.sample_fps),
            )

            for frame_number in range(0, frame_count, frame_step):
                timestamp = frame_number / source_fps

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = cap.read()

                if not ok:
                    failed += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    failed += 1
                    continue

                features, biomechanics = (
                    extract_features_and_biomechanics(results)
                )

                if features is None or biomechanics is None:
                    failed += 1
                    continue

                knees_forward = timestamp_in_intervals(
                    timestamp,
                    forward_intervals,
                )

                knees_inward = timestamp_in_intervals(
                    timestamp,
                    inward_intervals,
                )

                forward_positive += knees_forward
                inward_positive += knees_inward

                row = {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 6),
                    "labels": {
                        "knees_forward": knees_forward,
                        "knees_inward": knees_inward,
                    },
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
    print("Videos attempted:", len(records))
    print("Samples written:", written)
    print("Samples failed:", failed)
    print("Knees-forward positive samples:", forward_positive)
    print("Knees-inward positive samples:", inward_positive)


if __name__ == "__main__":
    main()
