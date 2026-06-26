import cv2
import mediapipe as mp
from ultralytics import YOLO

VIDEO = "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4"
OUTPUT = "/tmp/yolo_track_crop.mp4"

model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

target_id = None

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False, classes=[0])[0]

        boxes = results.boxes
        candidates = []

        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(track_id.item())

                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) / 2
                bottom = y2

                score = area + bottom * 250 - abs(cx - w / 2) * 2
                candidates.append((score, tid, x1, y1, x2, y2))

        if candidates:
            candidates.sort(reverse=True)

            if target_id is None:
                _, target_id, x1, y1, x2, y2 = candidates[0]
            else:
                match = [c for c in candidates if c[1] == target_id]
                if match:
                    _, _, x1, y1, x2, y2 = match[0]
                else:
                    _, target_id, x1, y1, x2, y2 = candidates[0]

            pad = 100
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = frame[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"ID {target_id}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if pose_result.pose_landmarks:
                for lm in pose_result.pose_landmarks.landmark:
                    px = x1 + int(lm.x * cw)
                    py = y1 + int(lm.y * ch)
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        writer.write(frame)

cap.release()
writer.release()

print(f"Saved: {OUTPUT}")