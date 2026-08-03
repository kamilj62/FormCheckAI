import csv
import json
import random
from pathlib import Path

import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
OUTPUT_JSON = BASE / "forward_lean_annotation_pilot.json"
OUTPUT_CSV = BASE / "forward_lean_annotation_pilot.csv"

RANDOM_SEED = 42

SPLIT_LIMITS = {
    "train": {
        "high": 30,
        "middle": 15,
        "low": 15,
    },
    "validation": {
        "high": 10,
        "middle": 5,
        "low": 5,
    },
}

FEATURES = [
    "torso_angle__bottom_mean",
    "torso_angle__ascent_mean",
    "torso_angle__ascent_maximum",
    "torso_angle__setup_to_bottom",
]


def load_manifest(split):
    path = BASE / f"{split}_manifest.json"

    records = json.loads(path.read_text())

    return {
        str(record["video_id"]): str(record["video_path"])
        for record in records
    }


def load_reps(split):
    metadata = json.loads(
        (
            BASE
            / f"knee_v9_rep_{split}_metadata.json"
        ).read_text()
    )

    feature_names = metadata["feature_names"]

    missing = [
        name
        for name in FEATURES
        if name not in feature_names
    ]

    if missing:
        raise RuntimeError(
            "Missing required V9 torso features: "
            + ", ".join(missing)
        )

    feature_indices = {
        name: feature_names.index(name)
        for name in FEATURES
    }

    rows = [
        json.loads(line)
        for line in (
            BASE / f"knee_v9_rep_{split}.jsonl"
        ).open()
    ]

    reps = []

    for row in rows:
        features = row["features"]

        values = {
            name: float(features[index])
            for name, index in feature_indices.items()
        }

        # Candidate-ranking score only.
        # This is not a label or coaching threshold.
        ranking_score = (
            0.50 * values[
                "torso_angle__ascent_maximum"
            ]
            + 0.25 * values[
                "torso_angle__bottom_mean"
            ]
            + 0.20 * values[
                "torso_angle__ascent_mean"
            ]
            + 0.05 * values[
                "torso_angle__setup_to_bottom"
            ]
        )

        reps.append({
            "split": split,
            "video_id": str(row["video_id"]),
            "rep_index": int(row["rep_index"]),
            "start_frame": int(row["start_frame"]),
            "bottom_frame": int(row["bottom_frame"]),
            "end_frame": int(row["end_frame"]),
            "ranking_score": float(ranking_score),
            **values,
        })

    return reps


def best_rep_per_video(reps):
    best = {}

    for rep in reps:
        video_id = rep["video_id"]

        if (
            video_id not in best
            or rep["ranking_score"]
            > best[video_id]["ranking_score"]
        ):
            best[video_id] = rep

    return list(best.values())


def sample_group(rows, count, rng):
    rows = list(rows)

    if len(rows) <= count:
        return rows

    return rng.sample(rows, count)


def build_split_candidates(split, rng):
    manifest = load_manifest(split)
    reps = best_rep_per_video(load_reps(split))

    scores = np.asarray(
        [rep["ranking_score"] for rep in reps],
        dtype=np.float64,
    )

    low_cutoff = float(np.quantile(scores, 0.25))
    high_cutoff = float(np.quantile(scores, 0.75))

    groups = {
        "low": [
            rep
            for rep in reps
            if rep["ranking_score"] <= low_cutoff
        ],
        "middle": [
            rep
            for rep in reps
            if (
                low_cutoff
                < rep["ranking_score"]
                < high_cutoff
            )
        ],
        "high": [
            rep
            for rep in reps
            if rep["ranking_score"] >= high_cutoff
        ],
    }

    selected = []

    for group_name in ["high", "middle", "low"]:
        group_rows = sample_group(
            groups[group_name],
            SPLIT_LIMITS[split][group_name],
            rng,
        )

        for rep in group_rows:
            selected.append({
                **rep,
                "candidate_group": group_name,
                "video_path": manifest.get(
                    rep["video_id"],
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

    return selected, {
        "available_videos": len(reps),
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "selected": len(selected),
        "selected_by_group": {
            group: sum(
                row["candidate_group"] == group
                for row in selected
            )
            for group in ["high", "middle", "low"]
        },
    }


def main():
    rng = random.Random(RANDOM_SEED)

    all_candidates = []
    split_summaries = {}

    for split in ["train", "validation"]:
        candidates, summary = build_split_candidates(
            split,
            rng,
        )

        all_candidates.extend(candidates)
        split_summaries[split] = summary

    payload = {
        "version": "forward_lean_annotation_pilot_v1",
        "selection_purpose": (
            "Diverse manual-review candidate selection only. "
            "Ranking scores are not labels or thresholds."
        ),
        "test_split_used": False,
        "allowed_review_labels": [
            "clear",
            "excessive_forward_lean",
            "uncertain",
            "unusable",
        ],
        "split_summaries": split_summaries,
        "candidates": all_candidates,
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2)
    )

    fieldnames = [
        "split",
        "candidate_group",
        "video_id",
        "video_path",
        "rep_index",
        "start_frame",
        "bottom_frame",
        "end_frame",
        "ranking_score",
        "torso_angle__bottom_mean",
        "torso_angle__ascent_mean",
        "torso_angle__ascent_maximum",
        "torso_angle__setup_to_bottom",
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
        )

        writer.writeheader()
        writer.writerows(all_candidates)

    print("Candidates:", len(all_candidates))
    print("Test split used: False")

    for split, summary in split_summaries.items():
        print()
        print(split.upper())
        print(
            "available videos:",
            summary["available_videos"],
        )
        print(
            "score cutoffs:",
            round(summary["low_cutoff"], 4),
            round(summary["high_cutoff"], 4),
        )
        print(
            "selected:",
            summary["selected_by_group"],
        )

    print()
    print("JSON:", OUTPUT_JSON)
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
