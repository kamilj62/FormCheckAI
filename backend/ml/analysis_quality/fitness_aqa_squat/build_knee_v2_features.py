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
    "knee_over_ankle",
    "knee_width_to_hip",
    "knee_width_to_ankle",
    "knee_collapse_vs_ankle",
    "knee_collapse_vs_hip",
    "hip_y",
    "knee_y",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        required=True,
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


def safe_ratio(numerator, denominator):
    return float(numerator) / max(abs(float(denominator)), 1e-6)


def normalized_angle(angle):
    angle = abs(float(angle)) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle
    return min(angle, abs(180.0 - angle))


def base_geometry(row):
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

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    ankle_mid = midpoint(left_ankle, right_ankle)

    torso_length = max(
        float(bio.get("shoulder_hip_distance", 0.0) or 0.0),
        math.dist(
            (shoulder_mid["x"], shoulder_mid["y"]),
            (hip_mid["x"], hip_mid["y"]),
        ),
        1e-6,
    )

    shoulder_width = abs(
        right_shoulder["x"] - left_shoulder["x"]
    )
    hip_width = abs(right_hip["x"] - left_hip["x"])
    knee_width = abs(right_knee["x"] - left_knee["x"])
    ankle_width = abs(
        right_ankle["x"] - left_ankle["x"]
    )

    facing_delta = nose["x"] - shoulder_mid["x"]

    if abs(facing_delta) < 0.015:
        facing_sign = 0.0
    else:
        facing_sign = 1.0 if facing_delta > 0 else -1.0

    knee_over_ankle = (
        facing_sign
        * safe_ratio(
            knee_mid["x"] - ankle_mid["x"],
            torso_length,
        )
    )

    left_line_mid_x = (
        left_hip["x"] + left_ankle["x"]
    ) / 2.0
    right_line_mid_x = (
        right_hip["x"] + right_ankle["x"]
    ) / 2.0

    knee_line_deviation = (
        abs(left_knee["x"] - left_line_mid_x)
        + abs(right_knee["x"] - right_line_mid_x)
    ) / (2.0 * torso_length)

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
        "hip_y": float(bio.get("hip_y", 0.0) or 0.0),
        "knee_y": float(bio.get("knee_y", 0.0) or 0.0),
        "shoulder_y": float(
            bio.get("shoulder_y", 0.0) or 0.0
        ),
        "knee_over_ankle": knee_over_ankle,
        "absolute_knee_over_ankle": abs(
            safe_ratio(
                knee_mid["x"] - ankle_mid["x"],
                torso_length,
            )
        ),
        "knee_width_to_hip": safe_ratio(
            knee_width,
            hip_width,
        ),
        "knee_width_to_ankle": safe_ratio(
            knee_width,
            ankle_width,
        ),
        "knee_collapse_vs_ankle": safe_ratio(
            ankle_width - knee_width,
            torso_length,
        ),
        "knee_collapse_vs_hip": safe_ratio(
            hip_width - knee_width,
            torso_length,
        ),
        "knee_line_deviation": knee_line_deviation,
        "shoulder_width_ratio": safe_ratio(
            shoulder_width,
            torso_length,
        ),
        "hip_width_ratio": safe_ratio(
            hip_width,
            torso_length,
        ),
        "knee_width_ratio": safe_ratio(
            knee_width,
            torso_length,
        ),
        "ankle_width_ratio": safe_ratio(
            ankle_width,
            torso_length,
        ),
        "minimum_visibility": minimum_visibility,
        "facing_known": float(facing_sign != 0.0),
    }


def temporal_vector(previous, current, following):
    values = []
    names = []

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
            (following_value - previous_value) / 2.0,
            max(previous_value, current_value, following_value)
            - min(previous_value, current_value, following_value),
        ])

    return names, values


def main():
    args = parse_args()

    source = BASE / f"knee_pose_{args.split}.jsonl"
    output = BASE / f"knee_v2_{args.split}.jsonl"

    by_video = defaultdict(list)

    with source.open() as f:
        for line in f:
            row = json.loads(line)
            row["_geometry"] = base_geometry(row)
            by_video[row["video_id"]].append(row)

    written = 0
    rejected_phase = 0
    rejected_quality = 0
    feature_names = None

    with output.open("w") as out:
        for video_id, rows in by_video.items():
            rows.sort(key=lambda row: row["frame_number"])

            for index, row in enumerate(rows):
                geometry = row["_geometry"]
                knee_angle = geometry["knee_angle"]

                # Remove standing frames and obvious pose artifacts.
                if not 60.0 <= knee_angle < 155.0:
                    rejected_phase += 1
                    continue

                if geometry["minimum_visibility"] < 0.35:
                    rejected_quality += 1
                    continue

                previous = rows[max(index - 1, 0)]["_geometry"]
                following = rows[
                    min(index + 1, len(rows) - 1)
                ]["_geometry"]

                names, values = temporal_vector(
                    previous,
                    geometry,
                    following,
                )

                vector = np.asarray(values, dtype=np.float32)

                if not np.all(np.isfinite(vector)):
                    rejected_quality += 1
                    continue

                if feature_names is None:
                    feature_names = names

                out.write(json.dumps({
                    "video_id": video_id,
                    "frame_number": row["frame_number"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "labels": row["labels"],
                    "features": vector.tolist(),
                }) + "\n")

                written += 1

    metadata = {
        "split": args.split,
        "source": str(source),
        "output": str(output),
        "feature_count": len(feature_names or []),
        "feature_names": feature_names or [],
        "rows_written": written,
        "rejected_phase": rejected_phase,
        "rejected_quality": rejected_quality,
    }

    metadata_path = (
        BASE / f"knee_v2_{args.split}_metadata.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Output:", output)
    print("Rows written:", written)
    print("Rejected by phase gate:", rejected_phase)
    print("Rejected by quality gate:", rejected_quality)
    print("Feature count:", len(feature_names or []))
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
