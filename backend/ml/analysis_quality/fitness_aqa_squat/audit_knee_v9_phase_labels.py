import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
BOTTOM_RADIUS = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "median": round(float(np.median(values)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "max": round(float(np.max(values)), 4),
    }


def state_name(forward, inward):
    if not forward and not inward:
        return "neither"
    if forward and not inward:
        return "forward_only"
    if not forward and inward:
        return "inward_only"
    return "both"


def fraction(rows, label):
    if not rows:
        return 0.0

    return float(
        np.mean([
            int(row["labels"][label])
            for row in rows
        ])
    )


def threshold_counts(records, prefix):
    results = {}

    for threshold in [0.1, 0.2, 0.25, 0.3, 0.5]:
        forward_key = f"{prefix}_forward_fraction"
        inward_key = f"{prefix}_inward_fraction"

        states = Counter()

        for record in records:
            forward = (
                float(record[forward_key])
                >= threshold
            )
            inward = (
                float(record[inward_key])
                >= threshold
            )

            states[
                state_name(forward, inward)
            ] += 1

        results[str(threshold)] = dict(states)

    return results


def main():
    args = parse_args()

    pose_path = BASE / f"knee_pose_{args.split}.jsonl"
    audit_path = (
        BASE
        / f"knee_v9_raw_pose_rep_audit_{args.split}.json"
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
            key=lambda row: int(row["frame_number"])
        )

    audit = json.loads(audit_path.read_text())
    phase_records = []
    missing_reps = 0

    for rep in audit["reps"]:
        video_id = str(rep["video_id"])
        rows = rows_by_video.get(video_id, [])

        start_frame = int(rep["start_frame"])
        bottom_frame = int(rep["bottom_frame"])
        end_frame = int(rep["end_frame"])

        rep_rows = [
            row
            for row in rows
            if (
                start_frame
                <= int(row["frame_number"])
                <= end_frame
            )
        ]

        if not rep_rows:
            missing_reps += 1
            continue

        bottom_position = min(
            range(len(rep_rows)),
            key=lambda index: abs(
                int(rep_rows[index]["frame_number"])
                - bottom_frame
            ),
        )

        bottom_start = max(
            0,
            bottom_position - BOTTOM_RADIUS,
        )
        bottom_end = min(
            len(rep_rows),
            bottom_position + BOTTOM_RADIUS + 1,
        )

        descent_rows = rep_rows[
            :bottom_position + 1
        ]

        bottom_rows = rep_rows[
            bottom_start:bottom_end
        ]

        ascent_rows = rep_rows[
            bottom_position:
        ]

        record = {
            "video_id": video_id,
            "start_frame": start_frame,
            "bottom_frame": bottom_frame,
            "end_frame": end_frame,
            "rep_row_count": len(rep_rows),
            "bottom_row_count": len(bottom_rows),
        }

        for phase_name, phase_rows in [
            ("rep", rep_rows),
            ("descent", descent_rows),
            ("bottom", bottom_rows),
            ("ascent", ascent_rows),
        ]:
            record[
                f"{phase_name}_forward_fraction"
            ] = fraction(
                phase_rows,
                "knees_forward",
            )

            record[
                f"{phase_name}_inward_fraction"
            ] = fraction(
                phase_rows,
                "knees_inward",
            )

        phase_records.append(record)

    print("=" * 76)
    print(args.split.upper())
    print("=" * 76)
    print("detected reps:", len(audit["reps"]))
    print("phase records:", len(phase_records))
    print("missing reps:", missing_reps)

    for phase_name in [
        "rep",
        "descent",
        "bottom",
        "ascent",
    ]:
        forward_values = [
            row[
                f"{phase_name}_forward_fraction"
            ]
            for row in phase_records
        ]

        inward_values = [
            row[
                f"{phase_name}_inward_fraction"
            ]
            for row in phase_records
        ]

        print("\n" + phase_name.upper())
        print(
            "forward fractions:",
            summarize(forward_values),
        )
        print(
            "inward fractions:",
            summarize(inward_values),
        )

        print("states by threshold:")

        counts = threshold_counts(
            phase_records,
            phase_name,
        )

        for threshold, states in counts.items():
            print(threshold, states)

    output_path = (
        BASE
        / f"knee_v9_phase_label_audit_{args.split}.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "version": "v9_phase_label_audit",
                "split": args.split,
                "bottom_radius": BOTTOM_RADIUS,
                "records": phase_records,
            },
            indent=2,
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
