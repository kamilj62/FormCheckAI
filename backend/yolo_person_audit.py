import sys
import json
import cv2
from ultralytics import YOLO

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "regression_tests/videos/thruster.mp4"
MODEL = "models/yolov8n.pt"
MAX_FRAMES = 300

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

frame_idx = 0
person_counts = []
multi_person_frames = 0
zero_person_frames = 0

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False, classes=[0])[0]
    boxes = result.boxes

    count = 0
    meaningful_count = 0

    h, w = frame.shape[:2]
    frame_area = max(1, h * w)

    if boxes is not None and boxes.xyxy is not None:
        count = len(boxes.xyxy)

        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(float, box)
            area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / frame_area)

            # Ignore tiny/background people.
            if area_ratio >= 0.04:
                meaningful_count += 1

    person_counts.append(meaningful_count)

    if count == 0:
        zero_person_frames += 1
    if count >= 2:
        multi_person_frames += 1

    frame_idx += 1

cap.release()

summary = {
    "video": VIDEO,
    "frames_checked": len(person_counts),
    "avg_people": round(sum(person_counts) / max(1, len(person_counts)), 2),
    "max_people": max(person_counts) if person_counts else 0,
    "zero_person_frames": zero_person_frames,
    "multi_person_frames": multi_person_frames,
    "multi_person_ratio": round(multi_person_frames / max(1, len(person_counts)), 3),
    "multi_person_risk": (
        max(person_counts) >= 2
        and multi_person_frames / max(1, len(person_counts)) >= 0.05
    ),
}

print(json.dumps(summary, indent=2))
