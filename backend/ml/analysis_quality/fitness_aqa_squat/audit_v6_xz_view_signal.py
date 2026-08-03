import json
from collections import Counter, defaultdict
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

INDEX = {
    name: index
    for index, name in enumerate(LANDMARKS)
}


def point(vector, name):
    start = INDEX[name] * 4

    return {
        "x": float(vector[start]),
        "y": float(vector[start + 1]),
        "z": float(vector[start + 2]),
        "visibility": float(vector[start + 3]),
    }


def safe_ratio(numerator, denominator):
    return float(numerator) / max(
        float(denominator),
        1e-4,
    )


def true_state(row):
    forward = (
        float(row["true_forward_fraction"])
        >= 0.5
    )
    inward = (
        float(row["true_inward_fraction"])
        >= 0.5
    )

    if not forward and not inward:
        return "neither"
    if forward and not inward:
        return "forward_only"
    if not forward and inward:
        return "inward_only"
    return "both"


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "p10": round(
            float(np.percentile(values, 10)),
            4,
        ),
        "p25": round(
            float(np.percentile(values, 25)),
            4,
        ),
        "median": round(
            float(np.median(values)),
            4,
        ),
        "p75": round(
            float(np.percentile(values, 75)),
            4,
        ),
        "p90": round(
            float(np.percentile(values, 90)),
            4,
        ),
    }


def interval_view_metrics(rows):
    shoulder_ratios = []
    hip_ratios = []
    knee_ratios = []
    ankle_ratios = []
    visibility_asymmetry = []

    for row in rows:
        vector = row["features"]

        ls = point(vector, "left_shoulder")
        rs = point(vector, "right_shoulder")
        lh = point(vector, "left_hip")
        rh = point(vector, "right_hip")
        lk = point(vector, "left_knee")
        rk = point(vector, "right_knee")
        la = point(vector, "left_ankle")
        ra = point(vector, "right_ankle")

        shoulder_x = abs(ls["x"] - rs["x"])
        hip_x = abs(lh["x"] - rh["x"])
        knee_x = abs(lk["x"] - rk["x"])
        ankle_x = abs(la["x"] - ra["x"])

        shoulder_z = abs(ls["z"] - rs["z"])
        hip_z = abs(lh["z"] - rh["z"])
        knee_z = abs(lk["z"] - rk["z"])
        ankle_z = abs(la["z"] - ra["z"])

        shoulder_ratios.append(
            safe_ratio(shoulder_z, shoulder_x)
        )
        hip_ratios.append(
            safe_ratio(hip_z, hip_x)
        )
        knee_ratios.append(
            safe_ratio(knee_z, knee_x)
        )
        ankle_ratios.append(
            safe_ratio(ankle_z, ankle_x)
        )

        left_visibility = np.mean([
            ls["visibility"],
            lh["visibility"],
            lk["visibility"],
            la["visibility"],
        ])

        right_visibility = np.mean([
            rs["visibility"],
            rh["visibility"],
            rk["visibility"],
            ra["visibility"],
        ])

        visibility_asymmetry.append(
            abs(
                left_visibility
                - right_visibility
            )
        )

    # Use torso landmarks for the primary view score.
    torso_ratio = np.median([
        np.median(shoulder_ratios),
        np.median(hip_ratios),
    ])

    lower_body_ratio = np.median([
        np.median(knee_ratios),
        np.median(ankle_ratios),
    ])

    return {
        "torso_z_to_x": float(torso_ratio),
        "lower_body_z_to_x": float(
            lower_body_ratio
        ),
        "shoulder_z_to_x": float(
            np.median(shoulder_ratios)
        ),
        "hip_z_to_x": float(
            np.median(hip_ratios)
        ),
        "knee_z_to_x": float(
            np.median(knee_ratios)
        ),
        "ankle_z_to_x": float(
            np.median(ankle_ratios)
        ),
        "visibility_asymmetry": float(
            np.median(visibility_asymmetry)
        ),
    }


for split in ["validation", "test"]:
    pose_path = BASE / f"knee_pose_{split}.jsonl"
    prediction_path = (
        BASE
        / (
            "knee_interval_v6_test_predictions.jsonl"
            if split == "test"
            else "knee_v6_interval_validation.jsonl"
        )
    )

    pose_by_video = defaultdict(list)

    with pose_path.open() as f:
        for line in f:
            row = json.loads(line)
            pose_by_video[
                str(row["video_id"])
            ].append(row)

    # Validation interval file contains targets but no predictions.
    if split == "validation":
        intervals = []

        with prediction_path.open() as f:
            for line in f:
                row = json.loads(line)

                intervals.append({
                    "video_id": row["video_id"],
                    "segment_index": (
                        row["segment_index"]
                    ),
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "true_forward_fraction": (
                        row["targets"][
                            "forward_fraction"
                        ]
                    ),
                    "true_inward_fraction": (
                        row["targets"][
                            "inward_fraction"
                        ]
                    ),
                    "predicted_forward_majority": None,
                })
    else:
        with prediction_path.open() as f:
            intervals = [
                json.loads(line)
                for line in f
            ]

    records = []

    for interval in intervals:
        video_id = str(interval["video_id"])
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])

        rows = [
            row
            for row in pose_by_video[video_id]
            if (
                start
                <= int(row["frame_number"])
                <= end
            )
        ]

        if not rows:
            continue

        metrics = interval_view_metrics(rows)

        records.append({
            **interval,
            **metrics,
            "true_state": true_state(interval),
        })

    print("\n" + "=" * 76)
    print(split.upper())
    print("=" * 76)
    print("intervals:", len(records))

    for state in [
        "neither",
        "forward_only",
        "inward_only",
        "both",
    ]:
        state_rows = [
            row
            for row in records
            if row["true_state"] == state
        ]

        print(f"\n{state}")
        print(
            "torso z/x:",
            summarize([
                row["torso_z_to_x"]
                for row in state_rows
            ]),
        )
        print(
            "lower-body z/x:",
            summarize([
                row["lower_body_z_to_x"]
                for row in state_rows
            ]),
        )
        print(
            "visibility asymmetry:",
            summarize([
                row["visibility_asymmetry"]
                for row in state_rows
            ]),
        )

    if split == "test":
        print("\n" + "-" * 76)
        print("TEST FALSE-FORWARD RATE BY TORSO Z/X BIN")
        print("-" * 76)

        bins = [
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
            (4.0, float("inf")),
        ]

        for low, high in bins:
            bin_rows = [
                row
                for row in records
                if (
                    low
                    <= row["torso_z_to_x"]
                    < high
                )
            ]

            negative_rows = [
                row
                for row in bin_rows
                if (
                    float(
                        row[
                            "true_forward_fraction"
                        ]
                    )
                    < 0.5
                )
            ]

            false_forward = [
                row
                for row in negative_rows
                if int(
                    row[
                        "predicted_forward_majority"
                    ]
                ) == 1
            ]

            rate = (
                len(false_forward)
                / len(negative_rows)
                if negative_rows
                else 0.0
            )

            print({
                "bin": f"{low}-{high}",
                "all_intervals": len(bin_rows),
                "forward_negative": len(
                    negative_rows
                ),
                "false_forward": len(
                    false_forward
                ),
                "false_forward_rate": round(
                    rate,
                    4,
                ),
            })

    output = (
        BASE
        / f"v6_xz_view_audit_{split}.json"
    )

    output.write_text(
        json.dumps(records, indent=2)
    )

    print("\nSaved:", output)
