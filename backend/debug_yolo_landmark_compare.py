import cv2
import json
import math
import mediapipe as mp

from app.tracking import YOLOTracker

VIDEO = "regression_tests/videos/front_squat.mov"
MAX_FRAMES = 120

mp_pose = mp.solutions.pose


def angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - cy)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)

    if n1 == 0 or n2 == 0:
        return None

    cosv = max(-1, min(1, dot / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def get_points(result, w, h, offset=(0, 0)):
    if not result.pose_landmarks:
        return None

    ox, oy = offset
    lm = result.pose_landmarks.landmark

    def pt(name):
        idx = getattr(mp_pose.PoseLandmark, name).value
        return (ox + lm[idx].x * w, oy + lm[idx].y * h)

    return {
        "l_shoulder": pt("LEFT_SHOULDER"),
        "r_shoulder": pt("RIGHT_SHOULDER"),
        "l_hip": pt("LEFT_HIP"),
        "r_hip": pt("RIGHT_HIP"),
        "l_knee": pt("LEFT_KNEE"),
        "r_knee": pt("RIGHT_KNEE"),
        "l_ankle": pt("LEFT_ANKLE"),
        "r_ankle": pt("RIGHT_ANKLE"),
        "l_wrist": pt("LEFT_WRIST"),
        "r_wrist": pt("RIGHT_WRIST"),
    }


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


cap = cv2.VideoCapture(VIDEO)
tracker = YOLOTracker("models/yolov8n.pt", pad=220)

rows = []

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose_full, mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose_crop:

    frame_idx = 0

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_result = pose_full.process(full_rgb)
        full_pts = get_points(full_result, w, h)

        crop_result = tracker.get_crop(frame)
        crop = crop_result.crop
        crop_pts = None

        if crop is not None:
            x1, y1, x2, y2 = crop_result.box
            ch, cw = crop.shape[:2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pose = pose_crop.process(crop_rgb)
            crop_pts = get_points(crop_pose, cw, ch, offset=(x1, y1))

        row = {
            "frame": frame_idx,
            "full_pose": full_pts is not None,
            "crop_pose": crop_pts is not None,
            "box": crop_result.box if crop_result else None,
            "target_id": crop_result.target_id if crop_result else None,
        }

        if full_pts and crop_pts:
            keys = ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_wrist", "r_wrist"]
            errors = [dist(full_pts[k], crop_pts[k]) for k in keys]
            row["avg_landmark_px_error"] = sum(errors) / len(errors)
            row["max_landmark_px_error"] = max(errors)

        rows.append(row)
        frame_idx += 1

cap.release()

summary = {
    "video": VIDEO,
    "frames": len(rows),
    "full_pose_frames": sum(r["full_pose"] for r in rows),
    "crop_pose_frames": sum(r["crop_pose"] for r in rows),
    "both_pose_frames": sum(r["full_pose"] and r["crop_pose"] for r in rows),
}

errs = [r["avg_landmark_px_error"] for r in rows if "avg_landmark_px_error" in r]
if errs:
    summary["avg_px_error_when_both"] = sum(errs) / len(errs)
    summary["max_avg_px_error_frame"] = max(errs)

print(json.dumps(summary, indent=2))
print("\nSample rows:")
print(json.dumps(rows[:10], indent=2))
