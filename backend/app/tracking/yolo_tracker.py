from dataclasses import dataclass
from pathlib import Path

import cv2

from ultralytics import YOLO


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter + 1e-6)



@dataclass
class CropResult:
    crop: object
    box: tuple[int, int, int, int]
    target_id: int | None


class YOLOTracker:
    def __init__(
        self,
        model_path="models/yolov8n.pt",
        pad=220,
        initial_target_id=None,
        smooth_alpha=0.20,
        max_missed_frames=30,
        detection_confidence=0.25,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.pad = pad
        self.target_id = initial_target_id
        self.last_box = None
        self.missed_frames = 0
        self.max_missed_frames = max(1, int(max_missed_frames))
        self.detection_confidence = max(
            0.01,
            min(1.0, float(detection_confidence)),
        )
        self.smooth_alpha = max(0.0, min(1.0, float(smooth_alpha)))

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
            self.missed_frames += 1

            # Keep the previous athlete crop during brief detector dropouts.
            if (
                self.last_box is not None
                and self.missed_frames <= self.max_missed_frames
            ):
                x1, y1, x2, y2 = self.last_box
                crop = frame[y1:y2, x1:x2]

                return CropResult(
                    crop=crop,
                    box=self.last_box,
                    target_id=self.target_id,
                )

            # Never fall back to the full busy frame in YOLO mode.
            # Returning no crop lets the analysis loop skip this frame.
            return CropResult(
                crop=None,
                box=self.last_box or (0, 0, w, h),
                target_id=self.target_id,
            )

        candidates.sort(reverse=True)

        if self.target_id is None:
            _, self.target_id, x1, y1, x2, y2 = candidates[0]
            self.missed_frames = 0
        else:
            match = [c for c in candidates if c[1] == self.target_id]

            if match:
                _, _, x1, y1, x2, y2 = match[0]
                self.missed_frames = 0
            else:
                self.missed_frames += 1

                # Do not immediately switch to another athlete.
                # Reuse the last crop while the tracked ID is briefly missing.
                if self.last_box is not None and self.missed_frames <= self.max_missed_frames:
                    x1, y1, x2, y2 = self.last_box
                    return CropResult(
                        crop=frame[y1:y2, x1:x2],
                        box=self.last_box,
                        target_id=self.target_id,
                    )

                # Try to stay on the previous athlete using box continuity.
                if self.last_box is not None:
                    lx1, ly1, lx2, ly2 = self.last_box
                    lcx = (lx1 + lx2) / 2
                    lcy = (ly1 + ly2) / 2

                    best = None
                    best_score = -1e9

                    for _, tid, cx1, cy1, cx2, cy2 in candidates:
                        candidate_box = (cx1, cy1, cx2, cy2)

                        ccx = (cx1 + cx2) / 2
                        ccy = (cy1 + cy2) / 2
                        dist = ((ccx - lcx) ** 2 + (ccy - lcy) ** 2) ** 0.5

                        iou = box_iou(self.last_box, candidate_box)

                        # Prefer overlap first, then center continuity.
                        score = (iou * 1000) - dist

                        if score > best_score:
                            best_score = score
                            best = (tid, cx1, cy1, cx2, cy2, iou, dist)

                    if best is not None:
                        tid, bx1, by1, bx2, by2, iou, dist = best

                        if iou >= 0.15 or dist < 180:
                            self.target_id = tid
                            self.missed_frames = 0
                            x1, y1, x2, y2 = bx1, by1, bx2, by2
                        else:
                            # No candidate is close enough to the previous
                            # athlete. Reuse the previous crop instead of
                            # silently switching people under the old ID.
                            x1, y1, x2, y2 = self.last_box
                    else:
                        x1, y1, x2, y2 = self.last_box
                else:
                    _, self.target_id, x1, y1, x2, y2 = candidates[0]
                    self.missed_frames = 0

        x1 = max(0, x1 - self.pad)
        y1 = max(0, y1 - self.pad)
        x2 = min(w, x2 + self.pad)
        y2 = min(h, y2 + self.pad)

        # Smooth crop movement and scale changes. Raw YOLO boxes can jitter,
        # which becomes false landmark movement after coordinate remapping.
        if self.last_box is not None:
            alpha = self.smooth_alpha
            lx1, ly1, lx2, ly2 = self.last_box

            x1 = int(round(lx1 * (1.0 - alpha) + x1 * alpha))
            y1 = int(round(ly1 * (1.0 - alpha) + y1 * alpha))
            x2 = int(round(lx2 * (1.0 - alpha) + x2 * alpha))
            y2 = int(round(ly2 * (1.0 - alpha) + y2 * alpha))

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

        self.last_box = (x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]

        return CropResult(
            crop=crop,
            box=(x1, y1, x2, y2),
            target_id=self.target_id,
        )