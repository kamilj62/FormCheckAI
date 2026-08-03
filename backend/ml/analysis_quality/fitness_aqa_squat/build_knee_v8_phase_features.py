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

PHASE_STATS = [
    "mean",
    "min",
    "max",
    "std",
]

CHANGE_STATS = [
    "setup_to_bottom",
    "bottom_to_finish",
    "setup_to_finish",
    "descent_range",
    "ascent_range",
    "peak_positive_from_setup",
    "peak_negative_from_setup",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "train",
            "validation",
            "test",
        ],
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
            - float(
                current[-1]["timestamp_seconds"]
            )
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


def bounded_window(
    center,
    radius,
    length,
):
    start = max(0, center - radius)
    end = min(length, center + radius + 1)

    return np.arange(
        start,
        end,
        dtype=np.int64,
    )


def evenly_spaced_window(
    start,
    end,
    fraction,
):
    length = max(1, end - start)

    count = max(
        1,
        int(np.ceil(length * fraction)),
    )

    return np.arange(
        start,
        min(end, start + count),
        dtype=np.int64,
    )


def ending_window(
    start,
    end,
    fraction,
):
    length = max(1, end - start)

    count = max(
        1,
        int(np.ceil(length * fraction)),
    )

    return np.arange(
        max(start, end - count),
        end,
        dtype=np.int64,
    )


def phase_indices(knee_angles):
    frame_count = len(knee_angles)

    bottom_index = int(
        np.argmin(knee_angles)
    )

    bottom_radius = max(
        0,
        int(round(frame_count * 0.10)),
    )

    bottom = bounded_window(
        bottom_index,
        bottom_radius,
        frame_count,
    )

    descent_end = bottom_index + 1
    ascent_start = bottom_index

    setup = evenly_spaced_window(
        0,
        max(1, descent_end),
        0.30,
    )

    finish = ending_window(
        min(ascent_start, frame_count - 1),
        frame_count,
        0.30,
    )

    descent = np.arange(
        0,
        max(1, descent_end),
        dtype=np.int64,
    )

    ascent = np.arange(
        min(ascent_start, frame_count - 1),
        frame_count,
        dtype=np.int64,
    )

    return {
        "setup": setup,
        "descent": descent,
        "bottom": bottom,
        "ascent": ascent,
        "finish": finish,
        "bottom_index": bottom_index,
    }


def safe_values(values, indices):
    selected = values[indices]

    if len(selected) == 0:
        return values

    return selected


def summarize_phase(values):
    return [
        float(np.mean(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.std(values)),
    ]


def build_feature_values(
    values,
    phases,
):
    setup = safe_values(
        values,
        phases["setup"],
    )
    descent = safe_values(
        values,
        phases["descent"],
    )
    bottom = safe_values(
        values,
        phases["bottom"],
    )
    ascent = safe_values(
        values,
        phases["ascent"],
    )
    finish = safe_values(
        values,
        phases["finish"],
    )

    setup_mean = float(np.mean(setup))
    bottom_mean = float(np.mean(bottom))
    finish_mean = float(np.mean(finish))

    phase_values = []

    for selected in [
        setup,
        descent,
        bottom,
        ascent,
        finish,
    ]:
        phase_values.extend(
            summarize_phase(selected)
        )

    change_values = [
        bottom_mean - setup_mean,
        finish_mean - bottom_mean,
        finish_mean - setup_mean,
        float(
            np.max(descent)
            - np.min(descent)
        ),
        float(
            np.max(ascent)
            - np.min(ascent)
        ),
        float(
            np.max(values - setup_mean)
        ),
        float(
            np.min(values - setup_mean)
        ),
    ]

    return phase_values + change_values


def output_feature_names():
    names = [
        "segment_frame_count",
        "segment_duration_seconds",
        "bottom_frame_fraction",
        "descent_frame_fraction",
        "ascent_frame_fraction",
    ]

    for feature in CORE_FEATURES:
        for phase in [
            "setup",
            "descent",
            "bottom",
            "ascent",
            "finish",
        ]:
            for statistic in PHASE_STATS:
                names.append(
                    f"{feature}__"
                    f"{phase}_{statistic}"
                )

        for statistic in CHANGE_STATS:
            names.append(
                f"{feature}__{statistic}"
            )

    return names


def build_interval_features(
    segment,
    source_feature_names,
):
    name_to_index = {
        name: index
        for index, name
        in enumerate(source_feature_names)
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
        [
            row["features"]
            for row in segment
        ],
        dtype=np.float64,
    )

    knee_angles = matrix[
        :,
        name_to_index["knee_angle"],
    ]

    phases = phase_indices(knee_angles)

    duration = (
        float(
            segment[-1][
                "timestamp_seconds"
            ]
        )
        - float(
            segment[0][
                "timestamp_seconds"
            ]
        )
    )

    frame_count = len(segment)
    denominator = max(
        frame_count - 1,
        1,
    )

    bottom_fraction = (
        phases["bottom_index"]
        / denominator
    )

    values = [
        float(frame_count),
        float(duration),
        float(bottom_fraction),
        float(
            len(phases["descent"])
            / frame_count
        ),
        float(
            len(phases["ascent"])
            / frame_count
        ),
    ]

    for feature in CORE_FEATURES:
        feature_values = matrix[
            :,
            name_to_index[feature],
        ]

        values.extend(
            build_feature_values(
                feature_values,
                phases,
            )
        )

    vector = np.asarray(
        values,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(vector)):
        raise RuntimeError(
            "Non-finite V8 feature detected"
        )

    names = output_feature_names()

    if len(names) != len(vector):
        raise RuntimeError(
            "V8 feature-name/vector mismatch: "
            f"{len(names)} names versus "
            f"{len(vector)} values"
        )

    return names, vector, phases


def main():
    args = parse_args()

    metadata_path = (
        BASE
        / f"knee_v5_{args.split}_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text()
    )

    source_feature_names = (
        metadata["feature_names"]
    )

    source = (
        BASE
        / f"knee_v5_{args.split}.jsonl"
    )

    output = (
        BASE
        / f"knee_v8_phase_{args.split}.jsonl"
    )

    output_metadata_path = (
        BASE
        / (
            f"knee_v8_phase_{args.split}"
            "_metadata.json"
        )
    )

    by_video = defaultdict(list)

    with source.open() as file:
        for line in file:
            row = json.loads(line)

            by_video[
                str(row["video_id"])
            ].append(row)

    expected_names = None
    written = 0
    mixed_intervals = 0
    forward_fractions = []
    inward_fractions = []
    bottom_fractions = []

    with output.open("w") as out:
        for video_id, rows in by_video.items():
            segments = find_segments(rows)

            for segment_index, segment in enumerate(
                segments
            ):
                (
                    names,
                    vector,
                    phases,
                ) = build_interval_features(
                    segment,
                    source_feature_names,
                )

                if expected_names is None:
                    expected_names = names
                elif names != expected_names:
                    raise RuntimeError(
                        "V8 feature order changed"
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
                    mixed_intervals += 1

                frame_count = len(segment)
                denominator = max(
                    frame_count - 1,
                    1,
                )

                bottom_fraction = float(
                    phases["bottom_index"]
                    / denominator
                )

                record = {
                    "video_id": video_id,
                    "segment_index": (
                        segment_index
                    ),
                    "start_frame": int(
                        segment[0]["frame_number"]
                    ),
                    "end_frame": int(
                        segment[-1]["frame_number"]
                    ),
                    "start_time": float(
                        segment[0][
                            "timestamp_seconds"
                        ]
                    ),
                    "end_time": float(
                        segment[-1][
                            "timestamp_seconds"
                        ]
                    ),
                    "frame_count": frame_count,
                    "bottom_index": int(
                        phases["bottom_index"]
                    ),
                    "bottom_frame": int(
                        segment[
                            phases["bottom_index"]
                        ]["frame_number"]
                    ),
                    "bottom_frame_fraction": (
                        bottom_fraction
                    ),
                    "targets": {
                        "forward_fraction": (
                            forward_fraction
                        ),
                        "inward_fraction": (
                            inward_fraction
                        ),
                        "forward_any": int(
                            np.any(
                                forward_labels == 1
                            )
                        ),
                        "inward_any": int(
                            np.any(
                                inward_labels == 1
                            )
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

                forward_fractions.append(
                    forward_fraction
                )
                inward_fractions.append(
                    inward_fraction
                )
                bottom_fractions.append(
                    bottom_fraction
                )

    output_metadata = {
        "version": (
            "v8_phase_normalized_temporal"
        ),
        "split": args.split,
        "source": str(source),
        "output": str(output),
        "segment_gap_seconds": (
            MAX_GAP_SECONDS
        ),
        "minimum_segment_frames": (
            MIN_SEGMENT_FRAMES
        ),
        "core_feature_count": len(
            CORE_FEATURES
        ),
        "feature_count": len(
            expected_names or []
        ),
        "feature_names": (
            expected_names or []
        ),
        "intervals_written": written,
        "mixed_intervals": mixed_intervals,
        "forward_fraction_mean": (
            float(np.mean(forward_fractions))
            if forward_fractions
            else 0.0
        ),
        "inward_fraction_mean": (
            float(np.mean(inward_fractions))
            if inward_fractions
            else 0.0
        ),
        "bottom_frame_fraction": {
            "mean": (
                float(
                    np.mean(bottom_fractions)
                )
                if bottom_fractions
                else 0.0
            ),
            "median": (
                float(
                    np.median(
                        bottom_fractions
                    )
                )
                if bottom_fractions
                else 0.0
            ),
            "p10": (
                float(
                    np.percentile(
                        bottom_fractions,
                        10,
                    )
                )
                if bottom_fractions
                else 0.0
            ),
            "p90": (
                float(
                    np.percentile(
                        bottom_fractions,
                        90,
                    )
                )
                if bottom_fractions
                else 0.0
            ),
        },
    }

    output_metadata_path.write_text(
        json.dumps(
            output_metadata,
            indent=2,
        )
    )

    print("Split:", args.split)
    print("Output:", output)
    print("Intervals:", written)
    print(
        "Mixed intervals:",
        mixed_intervals,
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
    print(
        "Bottom frame fraction:",
        output_metadata[
            "bottom_frame_fraction"
        ],
    )
    print(
        "Metadata:",
        output_metadata_path,
    )


if __name__ == "__main__":
    main()
