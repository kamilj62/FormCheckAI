import json
import math
from pathlib import Path

import cv2
import numpy as np


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
SOURCE_DIR = BASE / "v6_hard_negative_sheets"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"
OUTPUT_PATH = SOURCE_DIR / "all_hard_negatives_contact_sheet.jpg"

COLUMNS = 2
THUMBNAIL_WIDTH = 1200
MARGIN = 24
HEADER_HEIGHT = 70


def resize_to_width(image, width):
    height, current_width = image.shape[:2]

    scale = width / current_width

    return cv2.resize(
        image,
        (
            width,
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


manifest = json.loads(MANIFEST_PATH.read_text())

if not manifest:
    raise SystemExit("Manifest is empty")

tiles = []

for item in manifest:
    sheet_path = Path(item["sheet_path"])
    image = cv2.imread(str(sheet_path))

    if image is None:
        print("Skipping unreadable sheet:", sheet_path)
        continue

    image = resize_to_width(
        image,
        THUMBNAIL_WIDTH,
    )

    header = np.zeros(
        (
            HEADER_HEIGHT,
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    title = (
        f"Rank {item['rank']} | "
        f"{item['video_id']} | "
        f"segment {item['segment_index']} | "
        f"frames {item['start_frame']}-{item['end_frame']}"
    )

    scores = (
        f"true forward={item['true_forward_fraction']:.3f} | "
        f"true inward={item['true_inward_fraction']:.3f} | "
        f"pred forward={item['predicted_forward_fraction']:.3f} | "
        f"pred inward={item['predicted_inward_fraction']:.3f}"
    )

    cv2.putText(
        header,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        scores,
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    tile = np.vstack([header, image])
    tiles.append(tile)

if not tiles:
    raise SystemExit("No sheets could be loaded")

tile_width = max(tile.shape[1] for tile in tiles)
tile_height = max(tile.shape[0] for tile in tiles)

rows = math.ceil(len(tiles) / COLUMNS)

canvas_width = (
    COLUMNS * tile_width
    + (COLUMNS + 1) * MARGIN
)

canvas_height = (
    rows * tile_height
    + (rows + 1) * MARGIN
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

for index, tile in enumerate(tiles):
    row = index // COLUMNS
    column = index % COLUMNS

    y = MARGIN + row * (
        tile_height + MARGIN
    )
    x = MARGIN + column * (
        tile_width + MARGIN
    )

    canvas[
        y:y + tile.shape[0],
        x:x + tile.shape[1],
    ] = tile

cv2.imwrite(
    str(OUTPUT_PATH),
    canvas,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        92,
    ],
)

print("Items:", len(tiles))
print("Grid:", rows, "rows x", COLUMNS, "columns")
print("Size:", canvas_width, "x", canvas_height)
print("Saved:", OUTPUT_PATH)
