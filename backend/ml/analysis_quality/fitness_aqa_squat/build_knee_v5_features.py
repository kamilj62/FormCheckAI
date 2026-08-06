import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

LANDMARKS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
]

LANDMARK_INDEX = {
    name: index for index, name in enumerate(LANDMARKS)
}

TEMPORAL_KEYS = [
    "knee_angle",
    "hip_angle",
    "torso_angle",
    "hip_y",
    "knee_y",
    "forward_knee_offset",
    "absolute_knee_offset",
    "knee_minus_ankle_width",
    "knee_minus_hip_width",
    "left_knee_line_offset",
    "right_knee_line_offset",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def landmark(features, name):
    start = LANDMARK_INDEX[name] * 4

    return {
        "x": float(features[start]),
        "y": float(features[start + 1]),
        "z": float(features[start + 2]),
        "visibility": float(features[start + 3]),
    }


def midpoint(a, b):
    return {
        "x": (a["x"] + b["x"]) / 2.0,
        "y": (a["y"] + b["y"]) / 2.0,
    }


def normalized_angle(angle):
    angle = abs(float(angle)) % 360.0

    if angle > 180.0:
        angle = 360.0 - angle

    return min(angle, abs(180.0 - angle))


def safe_normalize(value, scale):
    return float(value) / max(float(scale), 1e-4)


def clip(value, low=-3.0, high=3.0):
    return float(np.clip(float(value), low, high))


def geometry(row):
    features = row["features"]
    bio = row["biomechanics"]

    nose = landmark(features, "nose")

    left_shoulder = landmark(features, "left_shoulder")
    right_shoulder = landmark(features, "right_shoulder")

    left_hip = landmark(features, "left_hip")
    right_hip = landmark(features, "right_hip")

    left_knee = landmark(features, "left_knee")
    right_knee = landmark(features, "right_knee")

    left_ankle = landmark(features, "left_ankle")
    right_ankle = landmark(features, "right_ankle")

    left_heel = landmark(features, "left_heel")
    right_heel = landmark(features, "right_heel")

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    ankle_mid = midpoint(left_ankle, right_ankle)

    torso_length = max(
        math.dist(
            (shoulder_mid["x"], shoulder_mid["y"]),
            (hip_mid["x"], hip_mid["y"]),
        ),
        float(
            bio.get("shoulder_hip_distance", 0.0)
            or 0.0
        ),
        1e-4,
    )

    shoulder_width = abs(
        right_shoulder["x"] - left_shoulder["x"]
    )
    hip_width = abs(
        right_hip["x"] - left_hip["x"]
    )
    knee_width = abs(
        right_knee["x"] - left_knee["x"]
    )
    ankle_width = abs(
        right_ankle["x"] - left_ankle["x"]
    )
    heel_width = abs(
        right_heel["x"] - left_heel["x"]
    )

    shoulder_width_n = safe_normalize(
        shoulder_width,
        torso_length,
    )
    hip_width_n = safe_normalize(
        hip_width,
        torso_length,
    )
    knee_width_n = safe_normalize(
        knee_width,
        torso_length,
    )
    ankle_width_n = safe_normalize(
        ankle_width,
        torso_length,
    )
    heel_width_n = safe_normalize(
        heel_width,
        torso_length,
    )

    facing_delta = nose["x"] - shoulder_mid["x"]

    if abs(facing_delta) < 0.015:
        facing_sign = 0.0
    else:
        facing_sign = (
            1.0 if facing_delta > 0.0 else -1.0
        )

    raw_knee_offset = safe_normalize(
        knee_mid["x"] - ankle_mid["x"],
        torso_length,
    )

    forward_knee_offset = (
        facing_sign * raw_knee_offset
    )

    left_leg_line_x = (
        left_hip["x"] + left_ankle["x"]
    ) / 2.0

    right_leg_line_x = (
        right_hip["x"] + right_ankle["x"]
    ) / 2.0

    left_knee_line_offset = safe_normalize(
        left_knee["x"] - left_leg_line_x,
        torso_length,
    )

    right_knee_line_offset = safe_normalize(
        right_knee["x"] - right_leg_line_x,
        torso_length,
    )

    projected_width_quality = min(
        shoulder_width_n / 0.25,
        hip_width_n / 0.16,
        1.0,
    )

    frontal_view_confidence = float(
        np.clip(projected_width_quality, 0.0, 1.0)
    )

    side_view_confidence = float(
        1.0 - frontal_view_confidence
    )

    minimum_visibility = min(
        left_shoulder["visibility"],
        right_shoulder["visibility"],
        left_hip["visibility"],
        right_hip["visibility"],
        left_knee["visibility"],
        right_knee["visibility"],
        left_ankle["visibility"],
        right_ankle["visibility"],
    )

    return {
        "knee_angle": float(
            bio.get("knee_angle", 0.0) or 0.0
        ),
        "hip_angle": float(
            bio.get("hip_angle", 0.0) or 0.0
        ),
        "torso_angle": normalized_angle(
            bio.get("torso_angle", 0.0) or 0.0
        ),
        "hip_y": float(
            bio.get("hip_y", 0.0) or 0.0
        ),
        "knee_y": float(
            bio.get("knee_y", 0.0) or 0.0
        ),
        "shoulder_y": float(
            bio.get("shoulder_y", 0.0) or 0.0
        ),

        # Sagittal-plane feature.
        "forward_knee_offset": clip(
            forward_knee_offset
        ),
        "absolute_knee_offset": clip(
            abs(raw_knee_offset)
        ),
        "facing_known": float(
            facing_sign != 0.0
        ),

        # Robust frontal-plane features.
        "shoulder_width_n": clip(
            shoulder_width_n
        ),
        "hip_width_n": clip(
            hip_width_n
        ),
        "knee_width_n": clip(
            knee_width_n
        ),
        "ankle_width_n": clip(
            ankle_width_n
        ),
        "heel_width_n": clip(
            heel_width_n
        ),
        "knee_minus_ankle_width": clip(
            knee_width_n - ankle_width_n
        ),
        "knee_minus_hip_width": clip(
            knee_width_n - hip_width_n
        ),
        "knee_minus_heel_width": clip(
            knee_width_n - heel_width_n
        ),
        "left_knee_line_offset": clip(
            left_knee_line_offset
        ),
        "right_knee_line_offset": clip(
            right_knee_line_offset
        ),
        "mean_knee_line_offset": clip(
            (
                left_knee_line_offset
                + right_knee_line_offset
            ) / 2.0
        ),
        "knee_line_asymmetry": clip(
            abs(
                left_knee_line_offset
                - right_knee_line_offset
            )
        ),

        # Explicit camera reliability.
        "frontal_view_confidence": (
            frontal_view_confidence
        ),
        "side_view_confidence": (
            side_view_confidence
        ),
        "projected_hip_width_reliable": float(
            hip_width_n >= 0.16
        ),
        "projected_shoulder_width_reliable": float(
            shoulder_width_n >= 0.25
        ),
        "minimum_visibility": float(
            minimum_visibility
        ),
    }


def build_temporal_vector(
    previous,
    current,
    following,
):
    names = []
    values = []

    for key, value in current.items():
        names.append(key)
        values.append(float(value))

    for key in TEMPORAL_KEYS:
        previous_value = float(previous[key])
        current_value = float(current[key])
        following_value = float(following[key])

        names.extend([
            f"{key}_delta_previous",
            f"{key}_delta_next",
            f"{key}_central_velocity",
            f"{key}_local_range",
        ])

        values.extend([
            current_value - previous_value,
            following_value - current_value,
            (
                following_value - previous_value
            ) / 2.0,
            max(
                previous_value,
                current_value,
                following_value,
            )
            - min(
                previous_value,
                current_value,
                following_value,
            ),
        ])

    return names, values


def main():
    args = parse_args()

    source = BASE / f"knee_pose_{args.split}.jsonl"
    output = BASE / f"knee_v5_{args.split}.jsonl"

    rows_by_video = defaultdict(list)

    with source.open() as f:
        for line in f:
            row = json.loads(line)
            row["_geometry"] = geometry(row)
            rows_by_video[row["video_id"]].append(row)

    written = 0
    rejected_phase = 0
    rejected_quality = 0
    feature_names = None

    with output.open("w") as out:
        for video_id, rows in rows_by_video.items():
            rows.sort(
                key=lambda row: row["frame_number"]
            )

            for index, row in enumerate(rows):
                current = row["_geometry"]

                if not (
                    60.0
                    <= current["knee_angle"]
                    < 155.0
                ):
                    rejected_phase += 1
                    continue

                if current["minimum_visibility"] < 0.35:
                    rejected_quality += 1
                    continue

                previous = rows[
                    max(0, index - 1)
                ]["_geometry"]

                following = rows[
                    min(len(rows) - 1, index + 1)
                ]["_geometry"]

                names, values = build_temporal_vector(
                    previous,
                    current,
                    following,
                )

                vector = np.asarray(
                    values,
                    dtype=np.float32,
                )

                if not np.all(np.isfinite(vector)):
                    rejected_quality += 1
                    continue

                if feature_names is None:
                    feature_names = names

                out.write(json.dumps({
                    "video_id": video_id,
                    "frame_number": row["frame_number"],
                    "timestamp_seconds": (
                        row["timestamp_seconds"]
                    ),
                    "labels": row["labels"],
                    "features": vector.tolist(),
                }) + "\n")

                written += 1

    metadata = {
        "version": "v5_robust_geometry",
        "split": args.split,
        "source": str(source),
        "output": str(output),
        "feature_count": len(
            feature_names or []
        ),
        "feature_names": feature_names or [],
        "rows_written": written,
        "rejected_phase": rejected_phase,
        "rejected_quality": rejected_quality,
    }

    metadata_path = (
        BASE / f"knee_v5_{args.split}_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print("Output:", output)
    print("Rows written:", written)
    print(
        "Rejected by phase gate:",
        rejected_phase,
    )
    print(
        "Rejected by quality gate:",
        rejected_quality,
    )
    print(
        "Feature count:",
        len(feature_names or []),
    )
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
