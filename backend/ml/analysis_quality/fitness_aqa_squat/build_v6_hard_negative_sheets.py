import json
from pathlib import Path

import cv2
import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
VIDEO_ROOT = Path(
    "/Users/josephkamil/Desktop/Capstone/videos/videos"
)
PREDICTIONS = BASE / "knee_interval_v6_test_predictions.jsonl"
OUTPUT_DIR = BASE / "v6_hard_negative_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_video(video_id):
    candidates = [
        VIDEO_ROOT / f"{video_id}.mp4",
        VIDEO_ROOT / f"{video_id}.avi",
        VIDEO_ROOT / f"{video_id}.mov",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(VIDEO_ROOT.rglob(f"{video_id}.*"))

    for match in matches:
        if match.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }:
            return match

    return None


def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()

    if not ok or frame is None:
        return None

    return frame


def resize_panel(frame, width=420, height=420):
    h, w = frame.shape[:2]

    scale = min(width / w, height / h)
    resized = cv2.resize(
        frame,
        (
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        ),
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2

    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1],
    ] = resized

    return canvas


def label_panel(panel, lines):
    output = panel.copy()

    y = 28

    for line in lines:
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return output


rows = []

with PREDICTIONS.open() as f:
    for line in f:
        row = json.loads(line)

        true_inward = (
            float(row["true_inward_fraction"]) >= 0.5
        )
        true_forward = (
            float(row["true_forward_fraction"]) >= 0.5
        )
        predicted_forward = int(
            row["predicted_forward_majority"]
        ) == 1

        # Primary V6 failure:
        # true inward-only interval predicted as forward.
        if (
            true_inward
            and not true_forward
            and predicted_forward
        ):
            rows.append(row)

rows.sort(
    key=lambda row: (
        float(row["predicted_forward_fraction"]),
        float(row["true_inward_fraction"]),
    ),
    reverse=True,
)

print("Hard-negative intervals:", len(rows))

manifest = []

for rank, row in enumerate(rows, start=1):
    video_id = str(row["video_id"])
    video_path = find_video(video_id)

    if video_path is None:
        print("Missing video:", video_id)
        continue

    start = int(row["start_frame"])
    end = int(row["end_frame"])
    midpoint = (start + end) // 2

    frame_numbers = [start, midpoint, end]

    cap = cv2.VideoCapture(str(video_path))

    panels = []

    for position, frame_number in zip(
        ["start", "middle", "end"],
        frame_numbers,
    ):
        frame = read_frame(cap, frame_number)

        if frame is None:
            frame = np.zeros(
                (420, 420, 3),
                dtype=np.uint8,
            )
        else:
            frame = resize_panel(frame)

        frame = label_panel(
            frame,
            [
                f"{video_id} | {position}",
                f"frame {frame_number}",
                (
                    "true forward="
                    f"{row['true_forward_fraction']:.3f}"
                ),
                (
                    "true inward="
                    f"{row['true_inward_fraction']:.3f}"
                ),
                (
                    "pred forward="
                    f"{row['predicted_forward_fraction']:.3f}"
                ),
                (
                    "pred inward="
                    f"{row['predicted_inward_fraction']:.3f}"
                ),
            ],
        )

        panels.append(frame)

    cap.release()

    sheet = np.hstack(panels)

    output_path = (
        OUTPUT_DIR
        / f"{rank:02d}_{video_id}_segment_{row['segment_index']}.jpg"
    )

    cv2.imwrite(str(output_path), sheet)

    manifest.append({
        "rank": rank,
        "video_id": video_id,
        "segment_index": row["segment_index"],
        "video_path": str(video_path),
        "sheet_path": str(output_path),
        "start_frame": start,
        "end_frame": end,
        "true_forward_fraction": (
            row["true_forward_fraction"]
        ),
        "true_inward_fraction": (
            row["true_inward_fraction"]
        ),
        "predicted_forward_fraction": (
            row["predicted_forward_fraction"]
        ),
        "predicted_inward_fraction": (
            row["predicted_inward_fraction"]
        ),
    })

manifest_path = (
    OUTPUT_DIR / "manifest.json"
)

manifest_path.write_text(
    json.dumps(manifest, indent=2)
)

print("Sheets written:", len(manifest))
print("Output directory:", OUTPUT_DIR)
print("Manifest:", manifest_path)
