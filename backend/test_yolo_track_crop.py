import cv2
import mediapipe as mp

from app.tracking import YOLOTracker

VIDEO = "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4"
OUTPUT = "/tmp/yolo_track_crop.mp4"

tracker = YOLOTracker("models/yolov8n.pt", pad=100)
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

        crop_result = tracker.get_crop(frame)
        crop = crop_result.crop
        if crop is None:
            writer.write(frame)
            continue

        x1, y1, x2, y2 = crop_result.box
        ch, cw = crop.shape[:2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if crop_result.target_id is not None:
            cv2.putText(
                frame,
                f"ID {crop_result.target_id}",
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
