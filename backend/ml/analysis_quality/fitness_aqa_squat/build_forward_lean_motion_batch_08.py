import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")

TRAIN_DATA = BASE / "knee_v9_rep_train.jsonl"
TRAIN_MANIFEST = BASE / "train_manifest.json"

ANNOTATIONS = (
    BASE / "forward_lean_annotations_consolidated.csv"
)

MODEL_PATH = (
    BASE / "forward_lean_manual_v9_extratrees.joblib"
)

OUTPUT_DIR = (
    BASE
    / "forward_lean_review"
    / "batch_08_motion"
)

OUTPUT_CSV = OUTPUT_DIR / "batch_08_review.csv"
OUTPUT_JSON = OUTPUT_DIR / "batch_08_selection.json"

START_CANDIDATE_NUMBER = 81

CLEAR_LIKE_COUNT = 5
BOUNDARY_COUNT = 5
HIGH_SCORE_COUNT = 2

MARGIN_FRAMES = 15


def load_manifest():
    payload = json.loads(
        TRAIN_MANIFEST.read_text()
    )

    if isinstance(payload, dict):
        records = payload.get(
            "records",
            payload.get(
                "videos",
                payload.get(
                    "items",
                    [],
                ),
            ),
        )
    else:
        records = payload

    if not isinstance(records, list):
        raise RuntimeError(
            "Could not locate the manifest record list"
        )

    mapping = {}

    for record in records:
        video_id = str(record["video_id"])
        video_path = str(record["video_path"])
        mapping[video_id] = video_path

    return mapping


def safe_fps(capture):
    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0 or fps > 240:
        return 30.0

    return fps


def extract_motion_clip(
    source,
    output,
    candidate_number,
    start_frame,
    bottom_frame,
    end_frame,
):
    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        return False, "could_not_open_video"

    fps = safe_fps(capture)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    clip_start = max(
        0,
        start_frame - MARGIN_FRAMES,
    )

    clip_end = end_frame + MARGIN_FRAMES

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        return False, "could_not_create_output"

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

        text = (
            f"Candidate {candidate_number} | "
            f"Frame {frame_number}{phase}"
        )

        cv2.putText(
            frame,
            text,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        written += 1
        frame_number += 1

    writer.release()
    capture.release()

    if written == 0:
        return False, "no_frames_written"

    return True, written


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model: {MODEL_PATH}"
        )

    reviewed = set()

    with ANNOTATIONS.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] != "train":
                continue

            reviewed.add((
                str(row["video_id"]),
                int(row["rep_index"]),
            ))

    manifest = load_manifest()

    metadata = json.loads(
        (
            BASE
            / "knee_v9_rep_train_metadata.json"
        ).read_text()
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

    model = joblib.load(MODEL_PATH)

    candidates = []

    with TRAIN_DATA.open() as file:
        for line in file:
            row = json.loads(line)

            key = (
                str(row["video_id"]),
                int(row["rep_index"]),
            )

            if key in reviewed:
                continue

            features = np.asarray(
                row["features"],
                dtype=np.float64,
            )

            shoulder_median = float(
                np.median([
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
                ])
            )

            hip_median = float(
                np.median([
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
                ])
            )

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

            video_id = key[0]
            video_path = manifest.get(video_id)

            if not video_path:
                continue

            candidates.append({
                "video_id": video_id,
                "video_path": video_path,
                "rep_index": key[1],
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
                "model_score": score,
                "camera_shoulder_median": (
                    shoulder_median
                ),
                "camera_hip_median": hip_median,
            })

    # One candidate per video.
    best_by_video = {}

    for row in candidates:
        video_id = row["video_id"]

        current = best_by_video.get(video_id)

        if (
            current is None
            or abs(row["model_score"] - 0.50)
            < abs(current["model_score"] - 0.50)
        ):
            best_by_video[video_id] = row

    candidates = list(
        best_by_video.values()
    )

    clear_like_pool = [
        row
        for row in candidates
        if row["model_score"] < 0.45
    ]

    boundary_pool = [
        row
        for row in candidates
        if 0.45 <= row["model_score"] < 0.55
    ]

    high_score_pool = [
        row
        for row in candidates
        if row["model_score"] >= 0.70
    ]

    clear_like = sorted(
        clear_like_pool,
        key=lambda row: row["model_score"],
    )[:CLEAR_LIKE_COUNT]

    boundary = sorted(
        boundary_pool,
        key=lambda row: abs(
            row["model_score"] - 0.50
        ),
    )[:BOUNDARY_COUNT]

    high_score = sorted(
        high_score_pool,
        key=lambda row: row["model_score"],
        reverse=True,
    )[:HIGH_SCORE_COUNT]

    selected = []

    for group, rows in [
        ("clear_like", clear_like),
        ("boundary", boundary),
        ("high_score", high_score),
    ]:
        for row in rows:
            selected.append({
                **row,
                "selection_group": group,
            })

    expected = (
        CLEAR_LIKE_COUNT
        + BOUNDARY_COUNT
        + HIGH_SCORE_COUNT
    )

    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} candidates, "
            f"selected {len(selected)}"
        )

    output_rows = []

    for offset, row in enumerate(selected):
        candidate_number = (
            START_CANDIDATE_NUMBER + offset
        )

        source = Path(row["video_path"])

        output_clip = (
            OUTPUT_DIR
            / f"candidate_{candidate_number:02d}_motion.mp4"
        )

        if not source.exists():
            raise RuntimeError(
                f"Missing video for Candidate "
                f"{candidate_number}: {source}"
            )

        ok, result = extract_motion_clip(
            source=source,
            output=output_clip,
            candidate_number=candidate_number,
            start_frame=row["start_frame"],
            bottom_frame=row["bottom_frame"],
            end_frame=row["end_frame"],
        )

        if not ok:
            raise RuntimeError(
                f"Candidate {candidate_number}: {result}"
            )

        output_rows.append({
            "candidate_number": candidate_number,
            "split": "train",
            "selection_group": row[
                "selection_group"
            ],
            "video_id": row["video_id"],
            "video_path": row["video_path"],
            "rep_index": row["rep_index"],
            "start_frame": row["start_frame"],
            "bottom_frame": row[
                "bottom_frame"
            ],
            "end_frame": row["end_frame"],
            "rep_row_count": row[
                "rep_row_count"
            ],
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
            row["selection_group"],
            "score=",
            round(row["model_score"], 4),
            "frames=",
            f"{row['start_frame']}->"
            f"{row['bottom_frame']}->"
            f"{row['end_frame']}",
            "written=",
            result,
        )

    fieldnames = list(
        output_rows[0].keys()
    )

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
        "version": (
            "forward_lean_motion_batch_08_v1"
        ),
        "split": "train",
        "selection_method": (
            "active learning from current "
            "709-feature shadow model"
        ),
        "validation_used_for_selection": False,
        "test_used": False,
        "camera_filter": {
            "shoulder_median_max": 0.60,
            "hip_median_max": 0.40,
        },
        "counts": {
            "clear_like": len(clear_like),
            "boundary": len(boundary),
            "high_score": len(high_score),
        },
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
    print("Motion directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
