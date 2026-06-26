import cv2
import mediapipe as mp

VIDEO = "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4"
OUTPUT = "/tmp/crop_pose_tracker.mp4"

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

x1 = int(width * 0.18)
x2 = int(width * 0.72)
y1 = int(height * 0.05)
y2 = height

subject_area = None
subject_center = None

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

        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if results.pose_landmarks:
            abs_xs = []
            abs_ys = []

            for lm in results.pose_landmarks.landmark:
                px = x1 + int(lm.x * crop_w)
                py = y1 + int(lm.y * crop_h)
                abs_xs.append(px)
                abs_ys.append(py)

            bx1, by1 = min(abs_xs), min(abs_ys)
            bx2, by2 = max(abs_xs), max(abs_ys)

            area = max(1, (bx2 - bx1) * (by2 - by1))
            center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

            accept = True

            if subject_area is not None and subject_center is not None:
                area_ratio = area / max(subject_area, 1)
                jump_y = center[1] - subject_center[1]
                jump_x = abs(center[0] - subject_center[0])

                # Background lifter usually appears smaller and higher when front lifter squats.
                if area_ratio < 0.55:
                    accept = False
                if jump_y < -height * 0.12:
                    accept = False
                if jump_x > width * 0.18:
                    accept = False

            if accept:
                subject_area = area if subject_area is None else subject_area * 0.9 + area * 0.1
                subject_center = center if subject_center is None else (
                    subject_center[0] * 0.9 + center[0] * 0.1,
                    subject_center[1] * 0.9 + center[1] * 0.1,
                )

                for px, py in zip(abs_xs, abs_ys):
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                pad_x = 120
                pad_y = 140

                nx1 = max(0, bx1 - pad_x)
                ny1 = max(0, by1 - pad_y)
                nx2 = min(width, bx2 + pad_x)
                ny2 = min(height, by2 + pad_y)

                min_w = int(width * 0.45)
                min_h = int(height * 0.75)

                if nx2 - nx1 < min_w:
                    extra = (min_w - (nx2 - nx1)) // 2
                    nx1 = max(0, nx1 - extra)
                    nx2 = min(width, nx2 + extra)

                if ny2 - ny1 < min_h:
                    extra = (min_h - (ny2 - ny1)) // 2
                    ny1 = max(0, ny1 - extra)
                    ny2 = min(height, ny2 + extra)

                alpha = 0.08
                x1 = int(x1 * (1 - alpha) + nx1 * alpha)
                y1 = int(y1 * (1 - alpha) + ny1 * alpha)
                x2 = int(x2 * (1 - alpha) + nx2 * alpha)
                y2 = int(y2 * (1 - alpha) + ny2 * alpha)

        writer.write(frame)

cap.release()
writer.release()

print(f"Saved: {OUTPUT}")
