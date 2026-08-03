import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

MAX_GAP_SECONDS = 0.6
MIN_SEGMENT_FRAMES = 3

CORE_FEATURES = [
    "knee_angle",
    "hip_angle",
    "torso_angle",
    "hip_y",
    "knee_y",
    "shoulder_y",
    "forward_knee_offset",
    "absolute_knee_offset",
    "facing_known",
    "shoulder_width_n",
    "hip_width_n",
    "knee_width_n",
    "ankle_width_n",
    "heel_width_n",
    "knee_minus_ankle_width",
    "knee_minus_hip_width",
    "knee_minus_heel_width",
    "left_knee_line_offset",
    "right_knee_line_offset",
    "mean_knee_line_offset",
    "knee_line_asymmetry",
    "frontal_view_confidence",
    "side_view_confidence",
    "projected_hip_width_reliable",
    "projected_shoulder_width_reliable",
    "minimum_visibility",
]

SUMMARY_STATS = [
    "mean",
    "std",
    "min",
    "max",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "range",
    "delta",
    "slope",
    "bottom_mean",
    "bottom_min",
    "bottom_max",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def find_segments(rows):
    rows = sorted(
        rows,
        key=lambda row: float(
            row["timestamp_seconds"]
        ),
    )

    segments = []
    current = []

    for row in rows:
        if not current:
            current = [row]
            continue

        gap = (
            float(row["timestamp_seconds"])
            - float(current[-1]["timestamp_seconds"])
        )

        if gap <= MAX_GAP_SECONDS:
            current.append(row)
        else:
            if len(current) >= MIN_SEGMENT_FRAMES:
                segments.append(current)

            current = [row]

    if len(current) >= MIN_SEGMENT_FRAMES:
        segments.append(current)

    return segments


def safe_slope(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=np.float64)

    slope = np.polyfit(x, values, 1)[0]

    if not np.isfinite(slope):
        return 0.0

    return float(slope)


def summarize_values(values, bottom_mask):
    values = np.asarray(values, dtype=np.float64)

    bottom_values = values[bottom_mask]

    if len(bottom_values) == 0:
        bottom_values = values

    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
        float(np.percentile(values, 90)),
        float(np.max(values) - np.min(values)),
        float(values[-1] - values[0]),
        safe_slope(values),
        float(np.mean(bottom_values)),
        float(np.min(bottom_values)),
        float(np.max(bottom_values)),
    ]


def build_interval_features(
    segment,
    feature_names,
):
    name_to_index = {
        name: index
        for index, name in enumerate(feature_names)
    }

    missing = [
        name
        for name in CORE_FEATURES
        if name not in name_to_index
    ]

    if missing:
        raise RuntimeError(
            f"Missing V5 features: {missing}"
        )

    matrix = np.asarray(
        [row["features"] for row in segment],
        dtype=np.float64,
    )

    knee_angles = matrix[
        :, name_to_index["knee_angle"]
    ]

    # Bottom phase = deepest 40% of sampled interval frames.
    bottom_threshold = np.percentile(
        knee_angles,
        40,
    )
    bottom_mask = knee_angles <= bottom_threshold

    output_names = [
        "segment_frame_count",
        "segment_duration_seconds",
    ]

    forward_labels = np.asarray(
        [
            int(row["labels"]["knees_forward"])
            for row in segment
        ],
        dtype=np.float64,
    )

    inward_labels = np.asarray(
        [
            int(row["labels"]["knees_inward"])
            for row in segment
        ],
        dtype=np.float64,
    )

    duration = (
        float(segment[-1]["timestamp_seconds"])
        - float(segment[0]["timestamp_seconds"])
    )

    output_values = [
        float(len(segment)),
        float(duration),
    ]

    for name in CORE_FEATURES:
        values = matrix[:, name_to_index[name]]

        feature_values = summarize_values(
            values,
            bottom_mask,
        )

        for statistic in SUMMARY_STATS:
            output_names.append(
                f"{name}__{statistic}"
            )

        output_values.extend(feature_values)

    vector = np.asarray(
        output_values,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(vector)):
        raise RuntimeError(
            "Non-finite V6 interval feature detected"
        )

    return output_names, vector


def main():
    args = parse_args()

    metadata_path = (
        BASE / f"knee_v5_{args.split}_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    feature_names = metadata["feature_names"]

    source = BASE / f"knee_v5_{args.split}.jsonl"
    output = (
        BASE / f"knee_v6_interval_{args.split}.jsonl"
    )

    by_video = defaultdict(list)

    with source.open() as f:
        for line in f:
            row = json.loads(line)
            by_video[str(row["video_id"])].append(row)

    written = 0
    output_feature_names = None

    forward_fraction_values = []
    inward_fraction_values = []
    mixed_interval_count = 0

    with output.open("w") as out:
        for video_id, rows in by_video.items():
            segments = find_segments(rows)

            for segment_index, segment in enumerate(
                segments
            ):
                names, vector = build_interval_features(
                    segment,
                    feature_names,
                )

                if output_feature_names is None:
                    output_feature_names = names
                elif names != output_feature_names:
                    raise RuntimeError(
                        "V6 feature ordering changed"
                    )

                forward_labels = np.asarray(
                    [
                        int(
                            row["labels"][
                                "knees_forward"
                            ]
                        )
                        for row in segment
                    ],
                    dtype=np.float64,
                )

                inward_labels = np.asarray(
                    [
                        int(
                            row["labels"][
                                "knees_inward"
                            ]
                        )
                        for row in segment
                    ],
                    dtype=np.float64,
                )

                forward_fraction = float(
                    np.mean(forward_labels)
                )
                inward_fraction = float(
                    np.mean(inward_labels)
                )

                if (
                    0.0 < forward_fraction < 1.0
                    or 0.0 < inward_fraction < 1.0
                ):
                    mixed_interval_count += 1

                record = {
                    "video_id": video_id,
                    "segment_index": segment_index,
                    "start_frame": int(
                        segment[0]["frame_number"]
                    ),
                    "end_frame": int(
                        segment[-1]["frame_number"]
                    ),
                    "start_time": float(
                        segment[0]["timestamp_seconds"]
                    ),
                    "end_time": float(
                        segment[-1]["timestamp_seconds"]
                    ),
                    "frame_count": len(segment),
                    "targets": {
                        "forward_fraction": (
                            forward_fraction
                        ),
                        "inward_fraction": (
                            inward_fraction
                        ),
                        "forward_any": int(
                            np.any(forward_labels == 1)
                        ),
                        "inward_any": int(
                            np.any(inward_labels == 1)
                        ),
                        "forward_majority": int(
                            forward_fraction >= 0.5
                        ),
                        "inward_majority": int(
                            inward_fraction >= 0.5
                        ),
                    },
                    "features": vector.tolist(),
                }

                out.write(
                    json.dumps(record) + "\n"
                )

                written += 1
                forward_fraction_values.append(
                    forward_fraction
                )
                inward_fraction_values.append(
                    inward_fraction
                )

    output_metadata = {
        "version": "v6_interval_soft_targets",
        "split": args.split,
        "source": str(source),
        "output": str(output),
        "segment_gap_seconds": MAX_GAP_SECONDS,
        "minimum_segment_frames": (
            MIN_SEGMENT_FRAMES
        ),
        "core_feature_count": len(CORE_FEATURES),
        "feature_count": len(
            output_feature_names or []
        ),
        "feature_names": (
            output_feature_names or []
        ),
        "intervals_written": written,
        "mixed_intervals": mixed_interval_count,
        "forward_fraction_mean": (
            float(np.mean(forward_fraction_values))
            if forward_fraction_values
            else 0.0
        ),
        "inward_fraction_mean": (
            float(np.mean(inward_fraction_values))
            if inward_fraction_values
            else 0.0
        ),
    }

    output_metadata_path = (
        BASE
        / f"knee_v6_interval_{args.split}_metadata.json"
    )

    output_metadata_path.write_text(
        json.dumps(output_metadata, indent=2)
    )

    print("Output:", output)
    print("Intervals written:", written)
    print(
        "Mixed-label intervals:",
        mixed_interval_count,
    )
    print(
        "Feature count:",
        output_metadata["feature_count"],
    )
    print(
        "Forward fraction mean:",
        round(
            output_metadata[
                "forward_fraction_mean"
            ],
            4,
        ),
    )
    print(
        "Inward fraction mean:",
        round(
            output_metadata[
                "inward_fraction_mean"
            ],
            4,
        ),
    )
    print("Metadata:", output_metadata_path)


if __name__ == "__main__":
    main()
