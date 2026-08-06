import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.analysis_quality.fitness_aqa_squat.build_knee_v5_features import (
    geometry,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

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

PHASES = [
    "setup",
    "descent",
    "bottom",
    "ascent",
    "finish",
]

STATS = [
    "mean",
    "minimum",
    "maximum",
    "std",
]

BOTTOM_RADIUS = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    return parser.parse_args()


def safe_stats(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {
            "mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "std": 0.0,
        }

    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "std": float(np.std(values)),
    }


def label_fraction(rows, label):
    if not rows:
        return 0.0

    return float(
        np.mean([
            int(row["labels"][label])
            for row in rows
        ])
    )


def phase_windows(rows, bottom_position):
    descent = rows[:bottom_position + 1]
    ascent = rows[bottom_position:]

    setup_size = max(
        1,
        int(round(len(descent) * 0.25)),
    )

    finish_size = max(
        1,
        int(round(len(ascent) * 0.25)),
    )

    bottom_start = max(
        0,
        bottom_position - BOTTOM_RADIUS,
    )

    bottom_end = min(
        len(rows),
        bottom_position + BOTTOM_RADIUS + 1,
    )

    return {
        "setup": descent[:setup_size],
        "descent": descent,
        "bottom": rows[bottom_start:bottom_end],
        "ascent": ascent,
        "finish": ascent[-finish_size:],
    }


def build_feature_names():
    names = [
        "rep_row_count",
        "source_frame_span",
        "descent_row_count",
        "ascent_row_count",
        "bottom_position_fraction",
        "descent_fraction",
        "ascent_fraction",
    ]

    for feature in CORE_FEATURES:
        for phase in PHASES:
            for statistic in STATS:
                names.append(
                    f"{feature}__{phase}_{statistic}"
                )

        names.extend([
            f"{feature}__setup_to_bottom",
            f"{feature}__bottom_to_finish",
            f"{feature}__setup_to_finish",
            f"{feature}__descent_range",
            f"{feature}__ascent_range",
            f"{feature}__bottom_vs_rep_mean",
            f"{feature}__ascent_vs_descent_mean",
        ])

    return names


def build_vector(
    rep_rows,
    phase_rows,
    geometry_rows,
    bottom_position,
):
    frame_numbers = np.asarray(
        [
            int(row["frame_number"])
            for row in rep_rows
        ],
        dtype=np.int64,
    )

    row_index = {
        id(row): index
        for index, row in enumerate(rep_rows)
    }

    phase_geometry = {}

    for phase_name, rows in phase_rows.items():
        phase_geometry[phase_name] = [
            geometry_rows[row_index[id(row)]]
            for row in rows
        ]

    descent_count = len(
        phase_rows["descent"]
    )

    ascent_count = len(
        phase_rows["ascent"]
    )

    denominator = max(
        len(rep_rows) - 1,
        1,
    )

    vector = [
        float(len(rep_rows)),
        float(frame_numbers[-1] - frame_numbers[0]),
        float(descent_count),
        float(ascent_count),
        float(bottom_position / denominator),
        float(descent_count / len(rep_rows)),
        float(ascent_count / len(rep_rows)),
    ]

    for feature in CORE_FEATURES:
        stats_by_phase = {}

        for phase_name in PHASES:
            values = [
                float(item[feature])
                for item in phase_geometry[
                    phase_name
                ]
            ]

            statistics = safe_stats(values)
            stats_by_phase[phase_name] = statistics

            for statistic in STATS:
                vector.append(
                    statistics[statistic]
                )

        rep_values = np.asarray(
            [
                float(item[feature])
                for item in geometry_rows
            ],
            dtype=np.float64,
        )

        descent_values = np.asarray(
            [
                float(item[feature])
                for item in phase_geometry[
                    "descent"
                ]
            ],
            dtype=np.float64,
        )

        ascent_values = np.asarray(
            [
                float(item[feature])
                for item in phase_geometry[
                    "ascent"
                ]
            ],
            dtype=np.float64,
        )

        setup_mean = stats_by_phase[
            "setup"
        ]["mean"]

        bottom_mean = stats_by_phase[
            "bottom"
        ]["mean"]

        finish_mean = stats_by_phase[
            "finish"
        ]["mean"]

        descent_range = float(
            np.max(descent_values)
            - np.min(descent_values)
        )

        ascent_range = float(
            np.max(ascent_values)
            - np.min(ascent_values)
        )

        vector.extend([
            bottom_mean - setup_mean,
            finish_mean - bottom_mean,
            finish_mean - setup_mean,
            descent_range,
            ascent_range,
            bottom_mean
            - float(np.mean(rep_values)),
            float(np.mean(ascent_values))
            - float(np.mean(descent_values)),
        ])

    return np.asarray(
        vector,
        dtype=np.float32,
    )


def state_name(forward, inward):
    if not forward and not inward:
        return "neither"

    if forward and not inward:
        return "forward_only"

    if not forward and inward:
        return "inward_only"

    return "both"


def main():
    args = parse_args()

    pose_path = (
        BASE / f"knee_pose_{args.split}.jsonl"
    )

    audit_path = (
        BASE
        / f"knee_v9_raw_pose_rep_audit_{args.split}.json"
    )

    output_path = (
        BASE / f"knee_v9_rep_{args.split}.jsonl"
    )

    metadata_path = (
        BASE
        / f"knee_v9_rep_{args.split}_metadata.json"
    )

    rows_by_video = defaultdict(list)

    with pose_path.open() as file:
        for line in file:
            row = json.loads(line)

            rows_by_video[
                str(row["video_id"])
            ].append(row)

    for rows in rows_by_video.values():
        rows.sort(
            key=lambda row: int(
                row["frame_number"]
            )
        )

    audit = json.loads(
        audit_path.read_text()
    )

    feature_names = build_feature_names()
    state_counts = Counter()

    written = 0
    rejected_missing = 0
    rejected_geometry = 0

    forward_targets = []
    inward_targets = []

    with output_path.open("w") as output:
        for rep in audit["reps"]:
            video_id = str(
                rep["video_id"]
            )

            start_frame = int(
                rep["start_frame"]
            )

            bottom_frame = int(
                rep["bottom_frame"]
            )

            end_frame = int(
                rep["end_frame"]
            )

            rep_rows = [
                row
                for row in rows_by_video.get(
                    video_id,
                    [],
                )
                if (
                    start_frame
                    <= int(row["frame_number"])
                    <= end_frame
                )
            ]

            if len(rep_rows) < 9:
                rejected_missing += 1
                continue

            bottom_position = min(
                range(len(rep_rows)),
                key=lambda index: abs(
                    int(
                        rep_rows[index][
                            "frame_number"
                        ]
                    )
                    - bottom_frame
                ),
            )

            if (
                bottom_position < 4
                or len(rep_rows)
                - bottom_position
                - 1 < 4
            ):
                rejected_missing += 1
                continue

            try:
                geometry_rows = [
                    geometry(row)
                    for row in rep_rows
                ]
            except Exception:
                rejected_geometry += 1
                continue

            missing_core = any(
                feature not in geometry_rows[0]
                for feature in CORE_FEATURES
            )

            if missing_core:
                raise RuntimeError(
                    "V5 geometry does not contain "
                    "all V9 core features"
                )

            windows = phase_windows(
                rep_rows,
                bottom_position,
            )

            vector = build_vector(
                rep_rows,
                windows,
                geometry_rows,
                bottom_position,
            )

            if len(vector) != len(
                feature_names
            ):
                raise RuntimeError(
                    "V9 feature count mismatch: "
                    f"{len(vector)} != "
                    f"{len(feature_names)}"
                )

            if not np.all(
                np.isfinite(vector)
            ):
                rejected_geometry += 1
                continue

            targets = {
                "rep_forward_fraction": (
                    label_fraction(
                        rep_rows,
                        "knees_forward",
                    )
                ),
                "rep_inward_fraction": (
                    label_fraction(
                        rep_rows,
                        "knees_inward",
                    )
                ),
                "bottom_forward_fraction": (
                    label_fraction(
                        windows["bottom"],
                        "knees_forward",
                    )
                ),
                "bottom_inward_fraction": (
                    label_fraction(
                        windows["bottom"],
                        "knees_inward",
                    )
                ),
                "ascent_forward_fraction": (
                    label_fraction(
                        windows["ascent"],
                        "knees_forward",
                    )
                ),
                "ascent_inward_fraction": (
                    label_fraction(
                        windows["ascent"],
                        "knees_inward",
                    )
                ),
            }

            forward_target = targets[
                "ascent_forward_fraction"
            ]

            inward_target = targets[
                "ascent_inward_fraction"
            ]

            forward_targets.append(
                forward_target
            )

            inward_targets.append(
                inward_target
            )

            state_counts[
                state_name(
                    forward_target >= 0.5,
                    inward_target >= 0.5,
                )
            ] += 1

            output_row = {
                "video_id": video_id,
                "rep_index": int(
                    rep["rep_index"]
                ),
                "start_frame": start_frame,
                "bottom_frame": bottom_frame,
                "end_frame": end_frame,
                "rep_row_count": len(rep_rows),
                "targets": targets,
                "features": vector.tolist(),
            }

            output.write(
                json.dumps(output_row)
                + "\n"
            )

            written += 1

    metadata = {
        "version": "v9_complete_rep_phase_features",
        "split": args.split,
        "source_pose": str(pose_path),
        "source_rep_audit": str(audit_path),
        "output": str(output_path),
        "target_definition": {
            "forward": "ascent_forward_fraction",
            "inward": "ascent_inward_fraction",
            "binary_reporting_threshold": 0.5,
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "rows_written": written,
        "rejected_missing": rejected_missing,
        "rejected_geometry": rejected_geometry,
        "state_counts_at_0_5": dict(
            state_counts
        ),
        "forward_target_mean": float(
            np.mean(forward_targets)
        ),
        "inward_target_mean": float(
            np.mean(inward_targets)
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("Split:", args.split)
    print("Output:", output_path)
    print("Rows:", written)
    print(
        "Feature count:",
        len(feature_names),
    )
    print(
        "States at 0.5:",
        dict(state_counts),
    )
    print(
        "Forward target mean:",
        round(
            metadata[
                "forward_target_mean"
            ],
            4,
        ),
    )
    print(
        "Inward target mean:",
        round(
            metadata[
                "inward_target_mean"
            ],
            4,
        ),
    )
    print(
        "Rejected missing:",
        rejected_missing,
    )
    print(
        "Rejected geometry:",
        rejected_geometry,
    )
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
