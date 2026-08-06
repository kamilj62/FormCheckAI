import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LANDMARK_NAMES = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]


def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def angle(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-8:
        return 0.0

    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def midpoint(left, right):
    return (
        (left[0] + right[0]) / 2.0,
        (left[1] + right[1]) / 2.0,
    )


def summarize(values, prefix):
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_delta": 0.0,
        }

    p10, p25, p50, p75, p90 = np.percentile(
        array,
        [10, 25, 50, 75, 90],
    )

    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_range": float(np.max(array) - np.min(array)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p25": float(p25),
        f"{prefix}_median": float(p50),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_robust_range": float(p90 - p10),
        f"{prefix}_start": float(array[0]),
        f"{prefix}_end": float(array[-1]),
        f"{prefix}_delta": float(array[-1] - array[0]),
    }


def extract_window(
    video_path,
    start_seconds,
    end_seconds,
    target,
    target_fps=10.0,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, "video_not_opened"

    source_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
    if source_fps <= 0:
        source_fps = 30.0

    sample_every = max(1, int(round(source_fps / target_fps)))
    start_frame = max(0, int(round(start_seconds * source_fps)))
    end_frame = max(start_frame, int(round(end_seconds * source_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    series = {
        "left_elbow_angle": [],
        "right_elbow_angle": [],
        "elbow_angle": [],
        "left_knee_angle": [],
        "right_knee_angle": [],
        "knee_angle": [],
        "wrist_y": [],
        "wrist_x": [],
        "shoulder_y": [],
        "hip_y": [],
        "torso_lean": [],
        "wrist_above_shoulder": [],
        "wrist_shoulder_offset_x": [],
    }

    visibility_values = []
    processed_frames = 0
    pose_frames = 0
    frame_number = start_frame

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while frame_number <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame_number - start_frame) % sample_every != 0:
                frame_number += 1
                continue

            processed_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                frame_number += 1
                continue

            landmarks = result.pose_landmarks.landmark
            points = {}

            frame_visibility = {}

            for name in LANDMARK_NAMES:
                index = mp.solutions.pose.PoseLandmark[name].value
                landmark = landmarks[index]
                points[name] = (landmark.x, landmark.y)
                frame_visibility[name] = float(landmark.visibility)

            if target == "elbow_error":
                required_landmarks = [
                    "LEFT_SHOULDER",
                    "RIGHT_SHOULDER",
                    "LEFT_ELBOW",
                    "RIGHT_ELBOW",
                    "LEFT_WRIST",
                    "RIGHT_WRIST",
                    "LEFT_HIP",
                    "RIGHT_HIP",
                ]
            else:
                required_landmarks = [
                    "LEFT_SHOULDER",
                    "RIGHT_SHOULDER",
                    "LEFT_HIP",
                    "RIGHT_HIP",
                    "LEFT_KNEE",
                    "RIGHT_KNEE",
                    "LEFT_ANKLE",
                    "RIGHT_ANKLE",
                ]

            required_visibility = [
                frame_visibility[name]
                for name in required_landmarks
            ]

            # Preserve the pose frame and retain visibility as a model
            # quality feature. Window-level filtering is safer than deleting
            # individual frames based on an unrelated landmark.
            visibility_values.extend(required_visibility)

            left_elbow = angle(
                points["LEFT_SHOULDER"],
                points["LEFT_ELBOW"],
                points["LEFT_WRIST"],
            )
            right_elbow = angle(
                points["RIGHT_SHOULDER"],
                points["RIGHT_ELBOW"],
                points["RIGHT_WRIST"],
            )

            left_knee = angle(
                points["LEFT_HIP"],
                points["LEFT_KNEE"],
                points["LEFT_ANKLE"],
            )
            right_knee = angle(
                points["RIGHT_HIP"],
                points["RIGHT_KNEE"],
                points["RIGHT_ANKLE"],
            )

            shoulder = midpoint(
                points["LEFT_SHOULDER"],
                points["RIGHT_SHOULDER"],
            )
            hip = midpoint(
                points["LEFT_HIP"],
                points["RIGHT_HIP"],
            )
            wrist = midpoint(
                points["LEFT_WRIST"],
                points["RIGHT_WRIST"],
            )

            torso_dx = shoulder[0] - hip[0]
            torso_dy = hip[1] - shoulder[1]
            torso_lean = float(
                np.degrees(np.arctan2(abs(torso_dx), max(abs(torso_dy), 1e-6)))
            )

            elbow_average = (left_elbow + right_elbow) / 2.0
            knee_average = (left_knee + right_knee) / 2.0

            series["left_elbow_angle"].append(left_elbow)
            series["right_elbow_angle"].append(right_elbow)
            series["elbow_angle"].append(elbow_average)
            series["left_knee_angle"].append(left_knee)
            series["right_knee_angle"].append(right_knee)
            series["knee_angle"].append(knee_average)
            series["wrist_y"].append(wrist[1])
            series["wrist_x"].append(wrist[0])
            series["shoulder_y"].append(shoulder[1])
            series["hip_y"].append(hip[1])
            series["torso_lean"].append(torso_lean)
            series["wrist_above_shoulder"].append(shoulder[1] - wrist[1])
            series["wrist_shoulder_offset_x"].append(abs(wrist[0] - shoulder[0]))

            pose_frames += 1
            frame_number += 1

    cap.release()

    if pose_frames < 4:
        return None, "insufficient_pose_frames"

    features = {
        "source_fps": source_fps,
        "target_fps": target_fps,
        "processed_frames": processed_frames,
        "pose_frames": pose_frames,
        "pose_coverage": pose_frames / max(processed_frames, 1),
        "mean_visibility": (
            float(np.mean(visibility_values))
            if visibility_values
            else 0.0
        ),
    }

    for name, values in series.items():
        features.update(summarize(values, name))

    return features, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    failures_path = Path(args.failures)

    with input_path.open() as f:
        rows = list(csv.DictReader(f))

    if args.limit > 0:
        rows = rows[:args.limit]

    extracted = []
    failures = []

    for index, row in enumerate(rows, 1):
        print(
            f"[{index}/{len(rows)}] "
            f"{row['video_id']} {row['target']} label={row['label']}",
            flush=True,
        )

        features, error = extract_window(
            video_path=Path(row["video_path"]),
            start_seconds=float(row["window_start_seconds"]),
            end_seconds=float(row["window_end_seconds"]),
            target=row["target"],
            target_fps=args.target_fps,
        )

        if error:
            failures.append({
                **row,
                "error": error,
            })
            continue

        extracted.append({
            **row,
            **features,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if extracted:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(extracted[0].keys()),
            )
            writer.writeheader()
            writer.writerows(extracted)

    failure_fields = (
        list(failures[0].keys())
        if failures
        else [
            "video_id",
            "video_path",
            "target",
            "label",
            "split",
            "window_start_seconds",
            "window_end_seconds",
            "source_interval_index",
            "sampling_type",
            "error",
        ]
    )

    with failures_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failures)

    print()
    print("extracted:", len(extracted))
    print("failures:", len(failures))
    print("saved:", output_path)
    print("failure log:", failures_path)


if __name__ == "__main__":
    main()
