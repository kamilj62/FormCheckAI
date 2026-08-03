import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.analysis_quality.fitness_aqa_squat.build_knee_v5_features import (
    geometry,
)
from ml.analysis_quality.fitness_aqa_squat.build_knee_v9_rep_features import (
    CORE_FEATURES,
    label_fraction,
    phase_windows,
    state_name,
)


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

PHASE_POINT_COUNTS = {
    "setup": 4,
    "descent": 8,
    "bottom": 5,
    "ascent": 8,
    "finish": 4,
}

PHASES = [
    "setup",
    "descent",
    "bottom",
    "ascent",
    "finish",
]

REP_METADATA_FEATURES = [
    "rep_row_count",
    "source_frame_span",
    "descent_row_count",
    "ascent_row_count",
    "bottom_position_fraction",
    "descent_fraction",
    "ascent_fraction",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    return parser.parse_args()


def phase_positions(row_count, bottom_position):
    last_position = float(row_count - 1)
    bottom = float(bottom_position)

    descent_span = max(bottom, 1.0)
    ascent_span = max(last_position - bottom, 1.0)

    setup_end = min(
        bottom,
        0.25 * descent_span,
    )

    bottom_start = max(
        0.0,
        bottom - 2.0,
    )

    bottom_end = min(
        last_position,
        bottom + 2.0,
    )

    finish_start = max(
        bottom,
        last_position - 0.25 * ascent_span,
    )

    return {
        "setup": np.linspace(
            0.0,
            setup_end,
            PHASE_POINT_COUNTS["setup"],
            dtype=np.float64,
        ),
        "descent": np.linspace(
            0.0,
            bottom,
            PHASE_POINT_COUNTS["descent"],
            dtype=np.float64,
        ),
        "bottom": np.linspace(
            bottom_start,
            bottom_end,
            PHASE_POINT_COUNTS["bottom"],
            dtype=np.float64,
        ),
        "ascent": np.linspace(
            bottom,
            last_position,
            PHASE_POINT_COUNTS["ascent"],
            dtype=np.float64,
        ),
        "finish": np.linspace(
            finish_start,
            last_position,
            PHASE_POINT_COUNTS["finish"],
            dtype=np.float64,
        ),
    }


def interpolate_feature(values, positions):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    source_positions = np.arange(
        len(values),
        dtype=np.float64,
    )

    return np.interp(
        positions,
        source_positions,
        values,
    )


def build_feature_names():
    names = list(REP_METADATA_FEATURES)

    for feature in CORE_FEATURES:
        for phase in PHASES:
            count = PHASE_POINT_COUNTS[phase]

            for point_index in range(count):
                names.append(
                    f"{feature}__{phase}_p{point_index:02d}"
                )

    return names


def build_vector(
    rep_rows,
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

    row_count = len(rep_rows)
    descent_count = bottom_position + 1
    ascent_count = row_count - bottom_position

    denominator = max(row_count - 1, 1)

    vector = [
        float(row_count),
        float(frame_numbers[-1] - frame_numbers[0]),
        float(descent_count),
        float(ascent_count),
        float(bottom_position / denominator),
        float(descent_count / row_count),
        float(ascent_count / row_count),
    ]

    positions_by_phase = phase_positions(
        row_count,
        bottom_position,
    )

    for feature in CORE_FEATURES:
        values = [
            float(item[feature])
            for item in geometry_rows
        ]

        for phase in PHASES:
            resampled = interpolate_feature(
                values,
                positions_by_phase[phase],
            )

            vector.extend(
                float(value)
                for value in resampled
            )

    return np.asarray(
        vector,
        dtype=np.float32,
    )


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
        BASE / f"knee_v10_rep_{args.split}.jsonl"
    )

    metadata_path = (
        BASE
        / f"knee_v10_rep_{args.split}_metadata.json"
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

    expected_feature_count = (
        len(REP_METADATA_FEATURES)
        + len(CORE_FEATURES)
        * sum(PHASE_POINT_COUNTS.values())
    )

    if len(feature_names) != expected_feature_count:
        raise RuntimeError(
            "V10 feature-name count mismatch: "
            f"{len(feature_names)} != "
            f"{expected_feature_count}"
        )

    state_counts = Counter()

    written = 0
    rejected_missing = 0
    rejected_geometry = 0

    forward_targets = []
    inward_targets = []

    with output_path.open("w") as output:
        for rep in audit["reps"]:
            video_id = str(rep["video_id"])

            start_frame = int(rep["start_frame"])
            bottom_frame = int(rep["bottom_frame"])
            end_frame = int(rep["end_frame"])

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

            # Preserve the exact V9 acceptance rule.
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

            # Preserve the exact V9 phase-coverage rule.
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
                    "all V10 core features"
                )

            vector = build_vector(
                rep_rows,
                geometry_rows,
                bottom_position,
            )

            if len(vector) != len(feature_names):
                raise RuntimeError(
                    "V10 feature count mismatch: "
                    f"{len(vector)} != "
                    f"{len(feature_names)}"
                )

            if not np.all(np.isfinite(vector)):
                rejected_geometry += 1
                continue

            # Preserve V9's original row-based phase windows
            # for target calculation.
            target_windows = phase_windows(
                rep_rows,
                bottom_position,
            )

            targets = {
                "rep_forward_fraction": label_fraction(
                    rep_rows,
                    "knees_forward",
                ),
                "rep_inward_fraction": label_fraction(
                    rep_rows,
                    "knees_inward",
                ),
                "bottom_forward_fraction": label_fraction(
                    target_windows["bottom"],
                    "knees_forward",
                ),
                "bottom_inward_fraction": label_fraction(
                    target_windows["bottom"],
                    "knees_inward",
                ),
                "ascent_forward_fraction": label_fraction(
                    target_windows["ascent"],
                    "knees_forward",
                ),
                "ascent_inward_fraction": label_fraction(
                    target_windows["ascent"],
                    "knees_inward",
                ),
            }

            forward_target = targets[
                "ascent_forward_fraction"
            ]

            inward_target = targets[
                "ascent_inward_fraction"
            ]

            forward_targets.append(forward_target)
            inward_targets.append(inward_target)

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
                json.dumps(output_row) + "\n"
            )

            written += 1

    metadata = {
        "version": (
            "v10_fixed_phase_resampled_features"
        ),
        "split": args.split,
        "source_pose": str(pose_path),
        "source_rep_audit": str(audit_path),
        "source_detector_version": "v9",
        "output": str(output_path),
        "phase_point_counts": PHASE_POINT_COUNTS,
        "total_phase_points": int(
            sum(PHASE_POINT_COUNTS.values())
        ),
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
    print("Feature count:", len(feature_names))
    print(
        "Phase points:",
        sum(PHASE_POINT_COUNTS.values()),
    )
    print(
        "States at 0.5:",
        dict(state_counts),
    )
    print(
        "Forward target mean:",
        round(
            metadata["forward_target_mean"],
            4,
        ),
    )
    print(
        "Inward target mean:",
        round(
            metadata["inward_target_mean"],
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
