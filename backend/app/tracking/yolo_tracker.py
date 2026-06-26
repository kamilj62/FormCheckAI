from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass
class CropResult:
    crop: object
    box: tuple[int, int, int, int]
    target_id: int | None


class YOLOTracker:
    def __init__(self, model_path="models/yolov8n.pt", pad=100, initial_target_id=None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.pad = pad
        self.target_id = initial_target_id
        self.last_box = None

    def get_crop(self, frame):
        h, w = frame.shape[:2]

        result = self.model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],  # person
        )[0]

        boxes = result.boxes
        candidates = []

        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(track_id.item())

                area = max(1, (x2 - x1) * (y2 - y1))
                cx = (x1 + x2) / 2
                bottom = y2

                score = area + bottom * 250 - abs(cx - w / 2) * 2
                candidates.append((score, tid, x1, y1, x2, y2))

        if not candidates:
            return CropResult(crop=frame, box=(0, 0, w, h), target_id=None)

        candidates.sort(reverse=True)

        if self.target_id is None:
            _, self.target_id, x1, y1, x2, y2 = candidates[0]
        else:
            match = [c for c in candidates if c[1] == self.target_id]
            if match:
                _, _, x1, y1, x2, y2 = match[0]
            else:
                _, self.target_id, x1, y1, x2, y2 = candidates[0]

        x1 = max(0, x1 - self.pad)
        y1 = max(0, y1 - self.pad)
        x2 = min(w, x2 + self.pad)
        y2 = min(h, y2 + self.pad)

        self.last_box = (x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]

        return CropResult(
            crop=crop,
            box=(x1, y1, x2, y2),
            target_id=self.target_id,
        )