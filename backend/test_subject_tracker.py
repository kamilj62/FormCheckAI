import cv2
from app.pose_tracking import SubjectTracker

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

VIDEO = "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4"
OUTPUT = "/tmp/tracked_subject.mp4"

# ------------------------------------------------------------
# LOAD TRACKER
# ------------------------------------------------------------

tracker = SubjectTracker("models/pose_landmarker_full.task")

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_idx = 0

print("Tracking...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp_ms = int((frame_idx / fps) * 1000)

    landmarks = tracker.process(frame, timestamp_ms)

    if landmarks is not None:
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    writer.write(frame)

    frame_idx += 1

cap.release()
writer.release()
tracker.close()

print(f"\nSaved tracked video to:\n{OUTPUT}")