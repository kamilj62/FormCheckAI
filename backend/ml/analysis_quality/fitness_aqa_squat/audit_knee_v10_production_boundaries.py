import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

PHASE_POINT_COUNTS = {
    "setup": 4,
    "descent": 8,
    "bottom": 5,
    "ascent": 8,
    "finish": 4,
}

TOTAL_RESAMPLED_POINTS = sum(
    PHASE_POINT_COUNTS.values()
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )

    parser.add_argument(
        "--pose-jsonl",
        default=None,
        help="Optional pose JSONL override for sampling-rate audits.",
    )

    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix for the audit output filename.",
    )

    return parser.parse_args()


def split_on_frame_gaps(rows, maximum_gap=30):
    if not rows:
        return []

    chunks = []
    current = [rows[0]]

    for previous, row in zip(rows, rows[1:]):
        previous_frame = int(
            previous["frame_number"]
        )
        current_frame = int(
            row["frame_number"]
        )

        if current_frame - previous_frame > maximum_gap:
            chunks.append(current)
            current = [row]
        else:
            current.append(row)

    if current:
        chunks.append(current)

    return chunks


def production_back_squat_boundaries(chunk):
    if len(chunk) < 9:
        return [], Counter({
            "chunk_too_short": 1,
        })

    frame_numbers = np.asarray(
        [
            int(row["frame_number"])
            for row in chunk
        ],
        dtype=np.int64,
    )

    knee_angles = np.asarray(
        [
            float(
                row["biomechanics"].get(
                    "knee_angle",
                    180.0,
                )
                or 180.0
            )
            for row in chunk
        ],
        dtype=np.float64,
    )

    torso_angles = np.asarray(
        [
            float(
                row["biomechanics"].get(
                    "torso_angle",
                    0.0,
                )
                or 0.0
            )
            for row in chunk
        ],
        dtype=np.float64,
    )

    # Exact back-squat production threshold.
    threshold = float(
        np.percentile(knee_angles, 35)
    )

    reps = []
    rejections = Counter()

    in_rep = False
    start = 0

    for index, knee in enumerate(knee_angles):
        if not in_rep and knee < threshold:
            in_rep = True
            start = index
            continue

        if not in_rep or knee < threshold:
            continue

        end = index
        in_rep = False

        if end - start < 5:
            rejections[
                "threshold_window_lt_5"
            ] += 1
            continue

        candidate_bottom = (
            start
            + int(
                np.argmin(
                    knee_angles[start:end + 1]
                )
            )
        )

        descent_span = (
            candidate_bottom - start
        )
        ascent_span = (
            end - candidate_bottom
        )

        source_span = int(
            frame_numbers[end]
            - frame_numbers[start]
        )

        starts_near_video_open = (
            int(frame_numbers[start])
            <= int(frame_numbers[0]) + 30
        )

        opening_candidate_knee = (
            knee_angles[start:end + 1]
        )

        opening_candidate_torso = (
            torso_angles[start:end + 1]
        )

        opening_pose_artifact = (
            starts_near_video_open
            and (
                float(
                    np.percentile(
                        np.clip(
                            opening_candidate_knee,
                            45,
                            180,
                        ),
                        20,
                    )
                )
                <= 45.5
                or float(
                    np.percentile(
                        np.clip(
                            opening_candidate_torso,
                            0,
                            90,
                        ),
                        75,
                    )
                )
                >= 89.5
            )
        )

        reasons = []

        if descent_span < 4:
            reasons.append(
                "descent_span_lt_4"
            )

        if ascent_span < 4:
            reasons.append(
                "ascent_span_lt_4"
            )

        if source_span < 20:
            reasons.append(
                "source_span_lt_20"
            )

        if (
            starts_near_video_open
            and source_span < 30
        ):
            reasons.append(
                "opening_source_span_lt_30"
            )

        if opening_pose_artifact:
            reasons.append(
                "opening_pose_artifact"
            )

        if reasons:
            rejections.update(reasons)
            continue

        reps.append({
            "start_index": int(start),
            "bottom_index": int(
                candidate_bottom
            ),
            "end_index": int(end),
            "start_frame": int(
                frame_numbers[start]
            ),
            "bottom_frame": int(
                frame_numbers[candidate_bottom]
            ),
            "end_frame": int(
                frame_numbers[end]
            ),
            "sampled_rows": int(
                end - start + 1
            ),
            "descent_rows": int(
                descent_span + 1
            ),
            "ascent_rows": int(
                ascent_span + 1
            ),
            "source_span": source_span,
            "source_descent_span": int(
                frame_numbers[candidate_bottom]
                - frame_numbers[start]
            ),
            "source_ascent_span": int(
                frame_numbers[end]
                - frame_numbers[candidate_bottom]
            ),
            "threshold": threshold,
            "bottom_knee": float(
                knee_angles[candidate_bottom]
            ),
        })

    return reps, rejections


def interpolate_rows(rows, start, end, count):
    if count <= 0:
        return []

    if end < start:
        start, end = end, start

    positions = np.linspace(
        float(start),
        float(end),
        count,
    )

    return positions.tolist()


def fixed_phase_positions(rep):
    start = float(rep["start_index"])
    bottom = float(rep["bottom_index"])
    end = float(rep["end_index"])

    descent_span = max(
        bottom - start,
        1.0,
    )

    ascent_span = max(
        end - bottom,
        1.0,
    )

    setup_end = (
        start + 0.25 * descent_span
    )

    finish_start = (
        end - 0.25 * ascent_span
    )

    bottom_start = max(
        start,
        bottom - 0.15 * descent_span,
    )

    bottom_end = min(
        end,
        bottom + 0.15 * ascent_span,
    )

    return {
        "setup": interpolate_rows(
            None,
            start,
            setup_end,
            PHASE_POINT_COUNTS["setup"],
        ),
        "descent": interpolate_rows(
            None,
            start,
            bottom,
            PHASE_POINT_COUNTS["descent"],
        ),
        "bottom": interpolate_rows(
            None,
            bottom_start,
            bottom_end,
            PHASE_POINT_COUNTS["bottom"],
        ),
        "ascent": interpolate_rows(
            None,
            bottom,
            end,
            PHASE_POINT_COUNTS["ascent"],
        ),
        "finish": interpolate_rows(
            None,
            finish_start,
            end,
            PHASE_POINT_COUNTS["finish"],
        ),
    }


def label_fraction(
    chunk,
    start_index,
    end_index,
    label,
):
    rows = chunk[
        start_index:end_index + 1
    ]

    if not rows:
        return 0.0

    return float(
        np.mean([
            int(row["labels"][label])
            for row in rows
        ])
    )


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "min": round(
            float(np.min(values)),
            3,
        ),
        "p10": round(
            float(np.percentile(values, 10)),
            3,
        ),
        "median": round(
            float(np.median(values)),
            3,
        ),
        "p90": round(
            float(np.percentile(values, 90)),
            3,
        ),
        "max": round(
            float(np.max(values)),
            3,
        ),
    }


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
        Path(args.pose_jsonl)
        if args.pose_jsonl
        else BASE / f"knee_pose_{args.split}.jsonl"
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

    all_reps = []
    rejection_counts = Counter()

    chunks_examined = 0
    videos_with_reps = 0

    for video_id, rows in rows_by_video.items():
        video_rep_count = 0

        for chunk_index, chunk in enumerate(
            split_on_frame_gaps(rows),
            start=1,
        ):
            chunks_examined += 1

            reps, rejections = (
                production_back_squat_boundaries(
                    chunk
                )
            )

            rejection_counts.update(
                rejections
            )

            for local_index, rep in enumerate(
                reps,
                start=1,
            ):
                phase_positions = (
                    fixed_phase_positions(rep)
                )

                flat_positions = [
                    position
                    for phase in [
                        "setup",
                        "descent",
                        "bottom",
                        "ascent",
                        "finish",
                    ]
                    for position in phase_positions[
                        phase
                    ]
                ]

                if (
                    len(flat_positions)
                    != TOTAL_RESAMPLED_POINTS
                ):
                    raise RuntimeError(
                        "Unexpected resampled point count"
                    )

                ascent_forward_fraction = (
                    label_fraction(
                        chunk,
                        rep["bottom_index"],
                        rep["end_index"],
                        "knees_forward",
                    )
                )

                ascent_inward_fraction = (
                    label_fraction(
                        chunk,
                        rep["bottom_index"],
                        rep["end_index"],
                        "knees_inward",
                    )
                )

                all_reps.append({
                    **rep,
                    "video_id": video_id,
                    "chunk_index": chunk_index,
                    "rep_index": local_index,
                    "phase_positions": (
                        phase_positions
                    ),
                    "resampled_point_count": (
                        len(flat_positions)
                    ),
                    "ascent_forward_fraction": (
                        ascent_forward_fraction
                    ),
                    "ascent_inward_fraction": (
                        ascent_inward_fraction
                    ),
                })

                video_rep_count += 1

        if video_rep_count > 0:
            videos_with_reps += 1

    states = Counter()

    for rep in all_reps:
        states[
            state_name(
                rep[
                    "ascent_forward_fraction"
                ] >= 0.5,
                rep[
                    "ascent_inward_fraction"
                ] >= 0.5,
            )
        ] += 1

    print("=" * 76)
    print(args.split.upper())
    print("=" * 76)

    print("videos:", len(rows_by_video))
    print("chunks examined:", chunks_examined)
    print("videos with reps:", videos_with_reps)
    print("detected reps:", len(all_reps))
    print(
        "fixed points per rep:",
        TOTAL_RESAMPLED_POINTS,
    )
    print("states at 0.5:", dict(states))

    print(
        "sampled rows:",
        summarize([
            rep["sampled_rows"]
            for rep in all_reps
        ]),
    )

    print(
        "source span:",
        summarize([
            rep["source_span"]
            for rep in all_reps
        ]),
    )

    print(
        "source descent span:",
        summarize([
            rep["source_descent_span"]
            for rep in all_reps
        ]),
    )

    print(
        "source ascent span:",
        summarize([
            rep["source_ascent_span"]
            for rep in all_reps
        ]),
    )

    print(
        "bottom knee:",
        summarize([
            rep["bottom_knee"]
            for rep in all_reps
        ]),
    )

    print(
        "forward ascent fractions:",
        summarize([
            rep[
                "ascent_forward_fraction"
            ]
            for rep in all_reps
        ]),
    )

    print(
        "inward ascent fractions:",
        summarize([
            rep[
                "ascent_inward_fraction"
            ]
            for rep in all_reps
        ]),
    )

    print("\nRejections:")

    for reason, count in (
        rejection_counts.most_common()
    ):
        print(reason, count)

    output_suffix = (
        f"_{args.output_tag}"
        if args.output_tag
        else ""
    )

    output_path = (
        BASE
        / (
            f"knee_v10_boundary_audit_"
            f"{args.split}{output_suffix}.json"
        )
    )

    output_path.write_text(
        json.dumps(
            {
                "version": (
                    "v10_production_boundary_audit"
                ),
                "split": args.split,
                "phase_point_counts": (
                    PHASE_POINT_COUNTS
                ),
                "total_resampled_points": (
                    TOTAL_RESAMPLED_POINTS
                ),
                "videos": len(rows_by_video),
                "chunks_examined": (
                    chunks_examined
                ),
                "videos_with_reps": (
                    videos_with_reps
                ),
                "detected_reps": len(all_reps),
                "state_counts_at_0_5": (
                    dict(states)
                ),
                "rejection_counts": dict(
                    rejection_counts
                ),
                "reps": all_reps,
            },
            indent=2,
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
