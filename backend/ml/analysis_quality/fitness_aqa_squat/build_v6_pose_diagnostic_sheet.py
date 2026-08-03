import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
VIDEO_ROOT = Path(
    "/Users/josephkamil/Desktop/Capstone/videos/videos"
)

POSE_PATH = BASE / "knee_pose_test.jsonl"
PREDICTION_PATH = (
    BASE / "knee_interval_v6_test_predictions.jsonl"
)

OUTPUT_DIR = BASE / "v6_pose_diagnostic_sheets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_PATH = (
    OUTPUT_DIR / "all_pose_diagnostics_contact_sheet.jpg"
)
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

LANDMARKS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
]

LANDMARK_INDEX = {
    name: index
    for index, name in enumerate(LANDMARKS)
}

SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
]

PANEL_WIDTH = 340
PANEL_HEIGHT = 430
HEADER_HEIGHT = 145
CONTACT_COLUMNS = 1
CONTACT_MARGIN = 24


def find_video(video_id):
    for extension in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    ]:
        candidate = VIDEO_ROOT / (
            video_id + extension
        )

        if candidate.exists():
            return candidate

    for match in VIDEO_ROOT.rglob(
        video_id + ".*"
    ):
        if match.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }:
            return match

    return None


def landmark(vector, name):
    start = LANDMARK_INDEX[name] * 4

    return {
        "x": float(vector[start]),
        "y": float(vector[start + 1]),
        "z": float(vector[start + 2]),
        "visibility": float(vector[start + 3]),
    }


def to_pixel(point, width, height):
    return (
        int(np.clip(point["x"], 0.0, 1.0) * width),
        int(np.clip(point["y"], 0.0, 1.0) * height),
    )


def midpoint(first, second):
    return {
        "x": (
            first["x"] + second["x"]
        ) / 2.0,
        "y": (
            first["y"] + second["y"]
        ) / 2.0,
    }


def view_metrics(vector):
    left_shoulder = landmark(
        vector,
        "left_shoulder",
    )
    right_shoulder = landmark(
        vector,
        "right_shoulder",
    )
    left_hip = landmark(vector, "left_hip")
    right_hip = landmark(vector, "right_hip")

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder,
    )
    hip_mid = midpoint(
        left_hip,
        right_hip,
    )

    torso_length = max(
        math.dist(
            (
                shoulder_mid["x"],
                shoulder_mid["y"],
            ),
            (
                hip_mid["x"],
                hip_mid["y"],
            ),
        ),
        1e-4,
    )

    shoulder_width = abs(
        right_shoulder["x"]
        - left_shoulder["x"]
    ) / torso_length

    hip_width = abs(
        right_hip["x"]
        - left_hip["x"]
    ) / torso_length

    frontal_confidence = float(
        np.clip(
            min(
                shoulder_width / 0.25,
                hip_width / 0.16,
                1.0,
            ),
            0.0,
            1.0,
        )
    )

    return {
        "shoulder_width_n": shoulder_width,
        "hip_width_n": hip_width,
        "frontal_confidence": (
            frontal_confidence
        ),
        "side_confidence": (
            1.0 - frontal_confidence
        ),
    }


def resize_with_padding(frame):
    height, width = frame.shape[:2]

    scale = min(
        PANEL_WIDTH / width,
        PANEL_HEIGHT / height,
    )

    resized = cv2.resize(
        frame,
        (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            PANEL_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        PANEL_WIDTH - resized.shape[1]
    ) // 2
    y_offset = (
        PANEL_HEIGHT - resized.shape[0]
    ) // 2

    canvas[
        y_offset:
        y_offset + resized.shape[0],
        x_offset:
        x_offset + resized.shape[1],
    ] = resized

    return canvas


def draw_pose(frame, vector):
    height, width = frame.shape[:2]

    points = {
        name: landmark(vector, name)
        for name in LANDMARKS
    }

    for first_name, second_name in SKELETON:
        first = points[first_name]
        second = points[second_name]

        if min(
            first["visibility"],
            second["visibility"],
        ) < 0.30:
            continue

        cv2.line(
            frame,
            to_pixel(first, width, height),
            to_pixel(second, width, height),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    # Knee-to-ankle vertical reference lines.
    for side in ["left", "right"]:
        knee = points[f"{side}_knee"]
        ankle = points[f"{side}_ankle"]
        hip = points[f"{side}_hip"]

        if min(
            knee["visibility"],
            ankle["visibility"],
        ) >= 0.30:
            ankle_pixel = to_pixel(
                ankle,
                width,
                height,
            )

            cv2.line(
                frame,
                (
                    ankle_pixel[0],
                    max(0, ankle_pixel[1] - 160),
                ),
                (
                    ankle_pixel[0],
                    min(
                        height - 1,
                        ankle_pixel[1] + 30,
                    ),
                ),
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

        if min(
            hip["visibility"],
            knee["visibility"],
            ankle["visibility"],
        ) >= 0.30:
            cv2.line(
                frame,
                to_pixel(hip, width, height),
                to_pixel(knee, width, height),
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

            cv2.line(
                frame,
                to_pixel(knee, width, height),
                to_pixel(ankle, width, height),
                (255, 255, 255),
                5,
                cv2.LINE_AA,
            )

    for name, point in points.items():
        if point["visibility"] < 0.30:
            continue

        radius = (
            8
            if "knee" in name
            else 6
        )

        cv2.circle(
            frame,
            to_pixel(point, width, height),
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            to_pixel(point, width, height),
            radius,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def read_frame(cap, frame_number):
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number),
    )

    success, frame = cap.read()

    if not success or frame is None:
        return None

    return frame


def add_text(panel, lines):
    y = 24

    for line in lines:
        cv2.putText(
            panel,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 21

    return panel


def select_phase_rows(rows):
    rows = sorted(
        rows,
        key=lambda row: int(
            row["frame_number"]
        ),
    )

    if not rows:
        return []

    knee_angles = np.asarray(
        [
            float(
                row["biomechanics"].get(
                    "knee_angle",
                    180.0,
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    bottom_index = int(
        np.argmin(knee_angles)
    )

    phase_indices = [
        0,
        max(
            0,
            int(round(bottom_index * 0.5)),
        ),
        bottom_index,
        min(
            len(rows) - 1,
            bottom_index
            + int(
                round(
                    (
                        len(rows)
                        - 1
                        - bottom_index
                    )
                    * 0.5
                )
            ),
        ),
        len(rows) - 1,
    ]

    phase_names = [
        "setup",
        "descent",
        "bottom",
        "ascent",
        "finish",
    ]

    return [
        (name, rows[index])
        for name, index in zip(
            phase_names,
            phase_indices,
        )
    ]


pose_by_video = defaultdict(list)

with POSE_PATH.open() as file:
    for line in file:
        row = json.loads(line)
        pose_by_video[
            str(row["video_id"])
        ].append(row)

hard_negatives = []

with PREDICTION_PATH.open() as file:
    for line in file:
        row = json.loads(line)

        true_forward = (
            float(
                row["true_forward_fraction"]
            )
            >= 0.5
        )
        true_inward = (
            float(
                row["true_inward_fraction"]
            )
            >= 0.5
        )
        predicted_forward = (
            int(
                row[
                    "predicted_forward_majority"
                ]
            )
            == 1
        )

        if (
            true_inward
            and not true_forward
            and predicted_forward
        ):
            hard_negatives.append(row)

hard_negatives.sort(
    key=lambda row: float(
        row["predicted_forward_fraction"]
    ),
    reverse=True,
)

print(
    "Hard-negative intervals:",
    len(hard_negatives),
)

manifest = []
individual_sheets = []

for rank, prediction in enumerate(
    hard_negatives,
    start=1,
):
    video_id = str(
        prediction["video_id"]
    )
    start_frame = int(
        prediction["start_frame"]
    )
    end_frame = int(
        prediction["end_frame"]
    )

    interval_rows = [
        row
        for row in pose_by_video[video_id]
        if (
            start_frame
            <= int(row["frame_number"])
            <= end_frame
        )
    ]

    selected = select_phase_rows(
        interval_rows
    )

    if not selected:
        print(
            "No pose rows:",
            video_id,
        )
        continue

    video_path = find_video(video_id)

    if video_path is None:
        print(
            "Missing video:",
            video_id,
        )
        continue

    capture = cv2.VideoCapture(
        str(video_path)
    )

    panels = []

    for phase_name, pose_row in selected:
        frame_number = int(
            pose_row["frame_number"]
        )

        frame = read_frame(
            capture,
            frame_number,
        )

        if frame is None:
            frame = np.zeros(
                (
                    PANEL_HEIGHT,
                    PANEL_WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )
        else:
            frame = draw_pose(
                frame,
                pose_row["features"],
            )
            frame = resize_with_padding(frame)

        metrics = view_metrics(
            pose_row["features"]
        )

        knee_angle = float(
            pose_row["biomechanics"].get(
                "knee_angle",
                0.0,
            )
        )

        frame = add_text(
            frame,
            [
                (
                    f"{video_id} | "
                    f"{phase_name}"
                ),
                f"frame {frame_number}",
                (
                    f"knee angle "
                    f"{knee_angle:.1f}"
                ),
                (
                    "front conf "
                    f"{metrics['frontal_confidence']:.2f}"
                ),
                (
                    "side conf "
                    f"{metrics['side_confidence']:.2f}"
                ),
            ],
        )

        panels.append(frame)

    capture.release()

    while len(panels) < 5:
        panels.append(
            panels[-1].copy()
        )

    strip = np.hstack(panels[:5])

    header = np.zeros(
        (
            HEADER_HEIGHT,
            strip.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    title_lines = [
        (
            f"Rank {rank} | {video_id} | "
            f"segment {prediction['segment_index']} | "
            f"frames {start_frame}-{end_frame}"
        ),
        (
            "true forward="
            f"{prediction['true_forward_fraction']:.3f} | "
            "true inward="
            f"{prediction['true_inward_fraction']:.3f}"
        ),
        (
            "pred forward="
            f"{prediction['predicted_forward_fraction']:.3f} | "
            "pred inward="
            f"{prediction['predicted_inward_fraction']:.3f}"
        ),
        (
            "vertical gray line = ankle reference; "
            "white lines = MediaPipe leg tracking"
        ),
    ]

    header = add_text(
        header,
        title_lines,
    )

    sheet = np.vstack(
        [header, strip]
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{rank:02d}_{video_id}"
            f"_segment_{prediction['segment_index']}.jpg"
        )
    )

    cv2.imwrite(
        str(output_path),
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            94,
        ],
    )

    individual_sheets.append(sheet)

    manifest.append({
        "rank": rank,
        "video_id": video_id,
        "segment_index": int(
            prediction["segment_index"]
        ),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sheet_path": str(output_path),
        "true_forward_fraction": float(
            prediction[
                "true_forward_fraction"
            ]
        ),
        "true_inward_fraction": float(
            prediction[
                "true_inward_fraction"
            ]
        ),
        "predicted_forward_fraction": float(
            prediction[
                "predicted_forward_fraction"
            ]
        ),
        "predicted_inward_fraction": float(
            prediction[
                "predicted_inward_fraction"
            ]
        ),
    })

MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2)
)

if not individual_sheets:
    raise SystemExit(
        "No diagnostic sheets generated"
    )

tile_width = max(
    sheet.shape[1]
    for sheet in individual_sheets
)
tile_height = max(
    sheet.shape[0]
    for sheet in individual_sheets
)

contact_rows = math.ceil(
    len(individual_sheets)
    / CONTACT_COLUMNS
)

canvas_width = (
    CONTACT_COLUMNS * tile_width
    + (
        CONTACT_COLUMNS + 1
    ) * CONTACT_MARGIN
)

canvas_height = (
    contact_rows * tile_height
    + (
        contact_rows + 1
    ) * CONTACT_MARGIN
)

canvas = np.full(
    (
        canvas_height,
        canvas_width,
        3,
    ),
    245,
    dtype=np.uint8,
)

for index, sheet in enumerate(
    individual_sheets
):
    row = index // CONTACT_COLUMNS
    column = index % CONTACT_COLUMNS

    x = (
        CONTACT_MARGIN
        + column
        * (
            tile_width
            + CONTACT_MARGIN
        )
    )
    y = (
        CONTACT_MARGIN
        + row
        * (
            tile_height
            + CONTACT_MARGIN
        )
    )

    canvas[
        y:y + sheet.shape[0],
        x:x + sheet.shape[1],
    ] = sheet

cv2.imwrite(
    str(CONTACT_PATH),
    canvas,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        92,
    ],
)

print(
    "Individual sheets:",
    len(individual_sheets),
)
print("Manifest:", MANIFEST_PATH)
print("Contact sheet:", CONTACT_PATH)
print(
    "Contact size:",
    canvas_width,
    "x",
    canvas_height,
)
