import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

TRAIN_DATA = BASE / "knee_v9_rep_train.jsonl"
TRAIN_MANIFEST = BASE / "train_manifest.json"
METADATA_PATH = BASE / "knee_v9_rep_train_metadata.json"

ANNOTATIONS = BASE / "forward_lean_annotations_consolidated.csv"

BATCH_08_REVIEW = (
    BASE
    / "forward_lean_review"
    / "batch_08_motion"
    / "batch_08_review.csv"
)

MODEL_PATH = (
    BASE / "forward_lean_manual_v9_extratrees.joblib"
)

OUTPUT_DIR = (
    BASE
    / "forward_lean_review"
    / "batch_09_motion"
)

OUTPUT_CSV = OUTPUT_DIR / "batch_09_review.csv"
OUTPUT_JSON = OUTPUT_DIR / "batch_09_selection.json"

START_CANDIDATE_NUMBER = 93
BATCH_SIZE = 12
MIN_SCORE = 0.52
MAX_SCORE = 0.70
TARGET_SCORE = 0.60
MARGIN_FRAMES = 15


def load_manifest():
    payload = json.loads(TRAIN_MANIFEST.read_text())

    if isinstance(payload, list):
        records = payload
    else:
        records = (
            payload.get("records")
            or payload.get("videos")
            or payload.get("items")
            or []
        )

    if not isinstance(records, list):
        raise RuntimeError(
            "Could not locate manifest record list"
        )

    return {
        str(record["video_id"]): str(record["video_path"])
        for record in records
    }


def load_excluded():
    excluded = set()

    with ANNOTATIONS.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == "train":
                excluded.add((
                    str(row["video_id"]),
                    int(row["rep_index"]),
                ))

    if BATCH_08_REVIEW.exists():
        with BATCH_08_REVIEW.open(newline="") as file:
            for row in csv.DictReader(file):
                excluded.add((
                    str(row["video_id"]),
                    int(row["rep_index"]),
                ))

    return excluded


def safe_fps(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0 or fps > 240:
        return 30.0

    return fps


def extract_clip(
    source,
    output,
    candidate_number,
    start_frame,
    bottom_frame,
    end_frame,
):
    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open: {source}")

    fps = safe_fps(capture)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_start = max(0, start_frame - MARGIN_FRAMES)
    clip_end = end_frame + MARGIN_FRAMES

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create: {output}")

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        clip_start,
    )

    frame_number = clip_start
    written = 0

    while frame_number <= clip_end:
        ok, frame = capture.read()

        if not ok:
            break

        phase = ""

        if frame_number == start_frame:
            phase = " START"
        elif frame_number == bottom_frame:
            phase = " BOTTOM"
        elif frame_number == end_frame:
            phase = " END"

        cv2.putText(
            frame,
            (
                f"Candidate {candidate_number} | "
                f"Frame {frame_number}{phase}"
            ),
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_number += 1
        written += 1

    writer.release()
    capture.release()

    if written == 0:
        raise RuntimeError(
            f"No frames written for Candidate "
            f"{candidate_number}"
        )

    return written


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    excluded = load_excluded()
    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    names = metadata["feature_names"]

    camera_names = [
        "shoulder_width_n__setup_mean",
        "shoulder_width_n__bottom_mean",
        "shoulder_width_n__ascent_mean",
        "hip_width_n__setup_mean",
        "hip_width_n__bottom_mean",
        "hip_width_n__ascent_mean",
    ]

    indices = {
        name: names.index(name)
        for name in camera_names
    }

    candidates = []

    with TRAIN_DATA.open() as file:
        for line in file:
            row = json.loads(line)

            key = (
                str(row["video_id"]),
                int(row["rep_index"]),
            )

            if key in excluded:
                continue

            features = np.asarray(
                row["features"],
                dtype=np.float64,
            )

            shoulder_median = float(np.median([
                features[
                    indices[
                        "shoulder_width_n__setup_mean"
                    ]
                ],
                features[
                    indices[
                        "shoulder_width_n__bottom_mean"
                    ]
                ],
                features[
                    indices[
                        "shoulder_width_n__ascent_mean"
                    ]
                ],
            ]))

            hip_median = float(np.median([
                features[
                    indices[
                        "hip_width_n__setup_mean"
                    ]
                ],
                features[
                    indices[
                        "hip_width_n__bottom_mean"
                    ]
                ],
                features[
                    indices[
                        "hip_width_n__ascent_mean"
                    ]
                ],
            ]))

            if (
                shoulder_median >= 0.60
                or hip_median >= 0.40
            ):
                continue

            score = float(
                model.predict_proba(
                    features.reshape(1, -1)
                )[0, 1]
            )

            if not (
                MIN_SCORE <= score <= MAX_SCORE
            ):
                continue

            video_id = key[0]
            video_path = manifest.get(video_id)

            if not video_path:
                continue

            candidates.append({
                "video_id": video_id,
                "video_path": video_path,
                "rep_index": key[1],
                "start_frame": int(row["start_frame"]),
                "bottom_frame": int(row["bottom_frame"]),
                "end_frame": int(row["end_frame"]),
                "rep_row_count": int(
                    row["rep_row_count"]
                ),
                "model_score": score,
                "camera_shoulder_median": (
                    shoulder_median
                ),
                "camera_hip_median": hip_median,
            })

    # Retain the rep closest to the target score
    # when a video contains multiple reps.
    best_by_video = {}

    for row in candidates:
        video_id = row["video_id"]
        current = best_by_video.get(video_id)

        if (
            current is None
            or abs(row["model_score"] - TARGET_SCORE)
            < abs(
                current["model_score"] - TARGET_SCORE
            )
        ):
            best_by_video[video_id] = row

    candidates = sorted(
        best_by_video.values(),
        key=lambda row: (
            abs(row["model_score"] - TARGET_SCORE),
            row["video_id"],
        ),
    )

    selected = candidates[:BATCH_SIZE]

    if len(selected) != BATCH_SIZE:
        raise RuntimeError(
            f"Expected {BATCH_SIZE} candidates, "
            f"found {len(selected)}"
        )

    output_rows = []

    for offset, row in enumerate(selected):
        candidate_number = (
            START_CANDIDATE_NUMBER + offset
        )

        source = Path(row["video_path"])

        if not source.exists():
            raise RuntimeError(
                f"Missing Candidate "
                f"{candidate_number} video: {source}"
            )

        output_clip = (
            OUTPUT_DIR
            / f"candidate_{candidate_number:02d}_motion.mp4"
        )

        written = extract_clip(
            source=source,
            output=output_clip,
            candidate_number=candidate_number,
            start_frame=row["start_frame"],
            bottom_frame=row["bottom_frame"],
            end_frame=row["end_frame"],
        )

        output_rows.append({
            "candidate_number": candidate_number,
            "split": "train",
            "selection_group": (
                "targeted_false_positive"
            ),
            "video_id": row["video_id"],
            "video_path": row["video_path"],
            "rep_index": row["rep_index"],
            "start_frame": row["start_frame"],
            "bottom_frame": row["bottom_frame"],
            "end_frame": row["end_frame"],
            "rep_row_count": row["rep_row_count"],
            "model_score": row["model_score"],
            "camera_shoulder_median": row[
                "camera_shoulder_median"
            ],
            "camera_hip_median": row[
                "camera_hip_median"
            ],
            "motion_clip": str(output_clip),
            "review_label": "",
            "review_confidence": "",
            "review_notes": "",
        })

        print(
            f"Candidate {candidate_number}:",
            "score=",
            round(row["model_score"], 4),
            "frames=",
            (
                f"{row['start_frame']}->"
                f"{row['bottom_frame']}->"
                f"{row['end_frame']}"
            ),
            "written=",
            written,
        )

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    payload = {
        "version": "forward_lean_motion_batch_09_v1",
        "split": "train",
        "purpose": (
            "target current model false positives"
        ),
        "validation_used_for_selection": False,
        "test_used": False,
        "score_range": [
            MIN_SCORE,
            MAX_SCORE,
        ],
        "target_score": TARGET_SCORE,
        "candidate_count": len(output_rows),
        "candidates": output_rows,
    }

    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2)
    )

    print()
    print("Selected:", len(output_rows))
    print("Validation used: False")
    print("Test used: False")
    print("Review CSV:", OUTPUT_CSV)
    print("Selection JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
