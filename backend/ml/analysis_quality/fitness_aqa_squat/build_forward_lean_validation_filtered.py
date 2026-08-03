import csv
import json
import random
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

OUTPUT_JSON = (
    BASE / "forward_lean_validation_filtered.json"
)

OUTPUT_CSV = (
    BASE / "forward_lean_validation_filtered.csv"
)

RANDOM_SEED = 42

GROUP_LIMITS = {
    "high": 10,
    "middle": 5,
    "low": 5,
}

TORSO_FEATURES = [
    "torso_angle__bottom_mean",
    "torso_angle__ascent_mean",
    "torso_angle__ascent_maximum",
    "torso_angle__setup_to_bottom",
]

CAMERA_FEATURES = [
    "shoulder_width_n__setup_mean",
    "shoulder_width_n__bottom_mean",
    "shoulder_width_n__ascent_mean",
    "hip_width_n__setup_mean",
    "hip_width_n__bottom_mean",
    "hip_width_n__ascent_mean",
]


def load_manifest():
    records = json.loads(
        (
            BASE / "validation_manifest.json"
        ).read_text()
    )

    return {
        str(record["video_id"]): str(
            record["video_path"]
        )
        for record in records
    }


def load_validation_reps():
    metadata = json.loads(
        (
            BASE
            / "knee_v9_rep_validation_metadata.json"
        ).read_text()
    )

    feature_names = metadata["feature_names"]

    needed = TORSO_FEATURES + CAMERA_FEATURES

    missing = [
        name
        for name in needed
        if name not in feature_names
    ]

    if missing:
        raise RuntimeError(
            "Missing required features:\n"
            + "\n".join(missing)
        )

    indices = {
        name: feature_names.index(name)
        for name in needed
    }

    records = []

    with (
        BASE / "knee_v9_rep_validation.jsonl"
    ).open() as file:
        for line in file:
            row = json.loads(line)

            values = {
                name: float(
                    row["features"][index]
                )
                for name, index in indices.items()
            }

            shoulder_median = float(
                np.median([
                    values[
                        "shoulder_width_n__setup_mean"
                    ],
                    values[
                        "shoulder_width_n__bottom_mean"
                    ],
                    values[
                        "shoulder_width_n__ascent_mean"
                    ],
                ])
            )

            hip_median = float(
                np.median([
                    values[
                        "hip_width_n__setup_mean"
                    ],
                    values[
                        "hip_width_n__bottom_mean"
                    ],
                    values[
                        "hip_width_n__ascent_mean"
                    ],
                ])
            )

            passes_camera_filter = (
                shoulder_median < 0.60
                and hip_median < 0.40
            )

            ranking_score = (
                0.50
                * values[
                    "torso_angle__ascent_maximum"
                ]
                + 0.25
                * values[
                    "torso_angle__bottom_mean"
                ]
                + 0.20
                * values[
                    "torso_angle__ascent_mean"
                ]
                + 0.05
                * values[
                    "torso_angle__setup_to_bottom"
                ]
            )

            records.append({
                "split": "validation",
                "video_id": str(row["video_id"]),
                "rep_index": int(row["rep_index"]),
                "start_frame": int(
                    row["start_frame"]
                ),
                "bottom_frame": int(
                    row["bottom_frame"]
                ),
                "end_frame": int(
                    row["end_frame"]
                ),
                "rep_row_count": int(
                    row["rep_row_count"]
                ),
                "ranking_score": float(
                    ranking_score
                ),
                "camera_shoulder_median": (
                    shoulder_median
                ),
                "camera_hip_median": hip_median,
                "passes_camera_filter": (
                    passes_camera_filter
                ),
                **values,
            })

    return records


def best_rep_per_video(rows):
    best = {}

    for row in rows:
        video_id = row["video_id"]

        if (
            video_id not in best
            or row["ranking_score"]
            > best[video_id]["ranking_score"]
        ):
            best[video_id] = row

    return list(best.values())


def sample_rows(rows, count, rng):
    rows = list(rows)

    if len(rows) <= count:
        return rows

    return rng.sample(rows, count)


def main():
    rng = random.Random(RANDOM_SEED)
    manifest = load_manifest()

    all_reps = load_validation_reps()

    filtered_reps = [
        row
        for row in all_reps
        if row["passes_camera_filter"]
    ]

    filtered_reps = best_rep_per_video(
        filtered_reps
    )

    if len(filtered_reps) < sum(
        GROUP_LIMITS.values()
    ):
        raise RuntimeError(
            "Not enough filtered validation videos: "
            f"{len(filtered_reps)}"
        )

    scores = np.asarray(
        [
            row["ranking_score"]
            for row in filtered_reps
        ],
        dtype=np.float64,
    )

    low_cutoff = float(
        np.quantile(scores, 0.25)
    )

    high_cutoff = float(
        np.quantile(scores, 0.75)
    )

    groups = {
        "low": [
            row
            for row in filtered_reps
            if row["ranking_score"] <= low_cutoff
        ],
        "middle": [
            row
            for row in filtered_reps
            if (
                low_cutoff
                < row["ranking_score"]
                < high_cutoff
            )
        ],
        "high": [
            row
            for row in filtered_reps
            if row["ranking_score"] >= high_cutoff
        ],
    }

    selected = []

    for group_name in [
        "high",
        "middle",
        "low",
    ]:
        group_rows = sample_rows(
            groups[group_name],
            GROUP_LIMITS[group_name],
            rng,
        )

        for row in group_rows:
            selected.append({
                **row,
                "candidate_group": group_name,
                "video_path": manifest.get(
                    row["video_id"],
                    "",
                ),
                "review_label": "",
                "review_confidence": "",
                "review_notes": "",
            })

    selected.sort(
        key=lambda row: (
            {
                "high": 0,
                "middle": 1,
                "low": 2,
            }[row["candidate_group"]],
            -row["ranking_score"],
        )
    )

    for number, row in enumerate(
        selected,
        start=61,
    ):
        row["candidate_number"] = number

    payload = {
        "version": (
            "forward_lean_validation_filtered_v1"
        ),
        "selection_source": "validation",
        "validation_labels_used": False,
        "test_split_used": False,
        "camera_filter": {
            "derived_from": (
                "reviewed training candidates only"
            ),
            "shoulder_median_max": 0.60,
            "hip_median_max": 0.40,
            "training_usable_recall": 0.9118,
            "training_unusable_rejection": 0.6316,
        },
        "available_reps_before_filter": len(
            all_reps
        ),
        "available_videos_after_filter": len(
            filtered_reps
        ),
        "score_cutoffs": {
            "low": low_cutoff,
            "high": high_cutoff,
        },
        "selected_by_group": {
            group: sum(
                row["candidate_group"] == group
                for row in selected
            )
            for group in [
                "high",
                "middle",
                "low",
            ]
        },
        "candidates": selected,
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2)
    )

    fieldnames = [
        "candidate_number",
        "split",
        "candidate_group",
        "video_id",
        "video_path",
        "rep_index",
        "start_frame",
        "bottom_frame",
        "end_frame",
        "rep_row_count",
        "ranking_score",
        "camera_shoulder_median",
        "camera_hip_median",
        "review_label",
        "review_confidence",
        "review_notes",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(selected)

    print(
        "Available reps before filter:",
        len(all_reps),
    )

    print(
        "Available videos after filter:",
        len(filtered_reps),
    )

    print(
        "Selected:",
        len(selected),
    )

    print(
        "Selected by group:",
        payload["selected_by_group"],
    )

    print(
        "Score cutoffs:",
        round(low_cutoff, 4),
        round(high_cutoff, 4),
    )

    print("Validation labels used: False")
    print("Test split used: False")
    print("JSON:", OUTPUT_JSON)
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
