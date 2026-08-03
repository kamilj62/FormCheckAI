import csv
import json
from collections import Counter
from pathlib import Path


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

ANNOTATIONS = (
    BASE / "forward_lean_annotations_consolidated.csv"
)

OUTPUT_TRAIN = (
    BASE / "forward_lean_manual_v9_train.jsonl"
)

OUTPUT_VALIDATION = (
    BASE / "forward_lean_manual_v9_validation.jsonl"
)

OUTPUT_METADATA = (
    BASE / "forward_lean_manual_v9_metadata.json"
)

SOURCE_FILES = {
    "train": BASE / "knee_v9_rep_train.jsonl",
    "validation": BASE / "knee_v9_rep_validation.jsonl",
}

USABLE_LABELS = {
    "clear": 0,
    "excessive_forward_lean": 1,
}


def load_annotations():
    annotations = {}

    with ANNOTATIONS.open(newline="") as file:
        for row in csv.DictReader(file):
            label = row["review_label"]

            if label not in USABLE_LABELS:
                continue

            split = row["split"]

            if split not in SOURCE_FILES:
                continue

            key = (
                split,
                str(row["video_id"]),
                int(row["rep_index"]),
            )

            if key in annotations:
                raise RuntimeError(
                    f"Duplicate annotation key: {key}"
                )

            annotations[key] = {
                "candidate_number": int(
                    row["candidate_number"]
                ),
                "review_label": label,
                "target": USABLE_LABELS[label],
                "review_confidence": row[
                    "review_confidence"
                ],
                "review_notes": row[
                    "review_notes"
                ],
            }

    return annotations


def load_source_rows(path):
    rows = {}

    with path.open() as file:
        for line in file:
            row = json.loads(line)

            key = (
                str(row["video_id"]),
                int(row["rep_index"]),
            )

            if key in rows:
                raise RuntimeError(
                    f"Duplicate source key: {key}"
                )

            rows[key] = row

    return rows


def build_split(split, annotations):
    source_rows = load_source_rows(
        SOURCE_FILES[split]
    )

    output_rows = []

    split_annotations = {
        (video_id, rep_index): annotation
        for (
            annotation_split,
            video_id,
            rep_index,
        ), annotation in annotations.items()
        if annotation_split == split
    }

    missing = []

    for key, annotation in split_annotations.items():
        source = source_rows.get(key)

        if source is None:
            missing.append(key)
            continue

        output_rows.append({
            "candidate_number": annotation[
                "candidate_number"
            ],
            "split": split,
            "video_id": str(source["video_id"]),
            "rep_index": int(source["rep_index"]),
            "start_frame": int(
                source["start_frame"]
            ),
            "bottom_frame": int(
                source["bottom_frame"]
            ),
            "end_frame": int(
                source["end_frame"]
            ),
            "rep_row_count": int(
                source["rep_row_count"]
            ),
            "features": source["features"],
            "review_label": annotation[
                "review_label"
            ],
            "target": annotation["target"],
            "review_confidence": annotation[
                "review_confidence"
            ],
            "review_notes": annotation[
                "review_notes"
            ],
        })

    if missing:
        raise RuntimeError(
            f"Missing {split} source rows: {missing}"
        )

    output_rows.sort(
        key=lambda row: row["candidate_number"]
    )

    return output_rows


def write_jsonl(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(
                json.dumps(row) + "\n"
            )


def main():
    annotations = load_annotations()

    train_rows = build_split(
        "train",
        annotations,
    )

    validation_rows = build_split(
        "validation",
        annotations,
    )

    if len(train_rows) != 34:
        raise RuntimeError(
            f"Expected 34 train rows, "
            f"found {len(train_rows)}"
        )

    if len(validation_rows) != 15:
        raise RuntimeError(
            f"Expected 15 validation rows, "
            f"found {len(validation_rows)}"
        )

    train_feature_lengths = {
        len(row["features"])
        for row in train_rows
    }

    validation_feature_lengths = {
        len(row["features"])
        for row in validation_rows
    }

    if len(train_feature_lengths) != 1:
        raise RuntimeError(
            "Train feature lengths are inconsistent"
        )

    if len(validation_feature_lengths) != 1:
        raise RuntimeError(
            "Validation feature lengths are inconsistent"
        )

    if (
        train_feature_lengths
        != validation_feature_lengths
    ):
        raise RuntimeError(
            "Train and validation feature lengths differ"
        )

    write_jsonl(
        OUTPUT_TRAIN,
        train_rows,
    )

    write_jsonl(
        OUTPUT_VALIDATION,
        validation_rows,
    )

    feature_count = next(
        iter(train_feature_lengths)
    )

    metadata = {
        "version": "forward_lean_manual_v9_v1",
        "source_features": "knee_v9_rep",
        "feature_count": feature_count,
        "label_mapping": {
            "clear": 0,
            "excessive_forward_lean": 1,
        },
        "excluded_labels": [
            "uncertain",
            "unusable",
        ],
        "test_split_used": False,
        "train": {
            "rows": len(train_rows),
            "labels": dict(
                Counter(
                    row["review_label"]
                    for row in train_rows
                )
            ),
            "confidence": dict(
                Counter(
                    row["review_confidence"]
                    for row in train_rows
                )
            ),
        },
        "validation": {
            "rows": len(validation_rows),
            "labels": dict(
                Counter(
                    row["review_label"]
                    for row in validation_rows
                )
            ),
            "confidence": dict(
                Counter(
                    row["review_confidence"]
                    for row in validation_rows
                )
            ),
        },
    }

    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2)
    )

    print("Train rows:", len(train_rows))
    print(
        "Train labels:",
        metadata["train"]["labels"],
    )
    print(
        "Train confidence:",
        metadata["train"]["confidence"],
    )

    print(
        "Validation rows:",
        len(validation_rows),
    )
    print(
        "Validation labels:",
        metadata["validation"]["labels"],
    )
    print(
        "Validation confidence:",
        metadata["validation"]["confidence"],
    )

    print("Feature count:", feature_count)
    print("Test split used: False")
    print("Train output:", OUTPUT_TRAIN)
    print(
        "Validation output:",
        OUTPUT_VALIDATION,
    )
    print("Metadata:", OUTPUT_METADATA)


if __name__ == "__main__":
    main()
