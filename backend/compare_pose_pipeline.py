import os
import math
import cv2
import mediapipe as mp

from app.tracking import YOLOTracker, remap_crop_landmarks_to_full_frame

VIDEO = "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4"
OUT_DIR = "/tmp/pose_compare"
MODEL_PATH = "models/yolov8n.pt"

os.makedirs(OUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

KEY_JOINTS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def draw_points(frame, landmarks, color):
    h, w = frame.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, color, -1)


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    tracker = YOLOTracker(MODEL_PATH, pad=220)

    frame_idx = 0
    compared = 0
    total_avg_error = 0.0
    worst_frame = None
    worst_avg = -1
    joint_totals = {name: 0.0 for name in KEY_JOINTS.values()}
    joint_counts = {name: 0 for name in KEY_JOINTS.values()}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as full_pose, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as crop_pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Full-frame MediaPipe
            full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_results = full_pose.process(full_rgb)

            # YOLO crop -> MediaPipe -> remap
            crop_result = tracker.get_crop(frame)
            crop = crop_result.crop

            if crop is None:
                frame_idx += 1
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_results = crop_pose.process(crop_rgb)

            if not full_results.pose_landmarks or not crop_results.pose_landmarks:
                frame_idx += 1
                continue

            crop_results = remap_crop_landmarks_to_full_frame(
                crop_results,
                crop_result.box,
                w,
                h,
            )

            full_lm = full_results.pose_landmarks.landmark
            crop_lm = crop_results.pose_landmarks.landmark

            errors = [dist(a, b) for a, b in zip(full_lm, crop_lm)]
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)

            compared += 1
            total_avg_error += avg_error

            if avg_error > worst_avg:
                worst_avg = avg_error
                worst_frame = frame_idx

            for idx, name in KEY_JOINTS.items():
                e = errors[idx]
                joint_totals[name] += e
                joint_counts[name] += 1

            if frame_idx % 20 == 0:
                print(
                    f"frame={frame_idx} avg_error={avg_error:.4f} "
                    f"max_error={max_error:.4f} box={crop_result.box} id={crop_result.target_id}"
                )

            # Save debug image when pipelines disagree a lot
            if avg_error > 0.03:
                debug = frame.copy()

                # Green = full-frame pose
                draw_points(debug, full_lm, (0, 255, 0))

                # Red = YOLO-crop/remapped pose
                draw_points(debug, crop_lm, (0, 0, 255))

                x1, y1, x2, y2 = crop_result.box
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 3)

                cv2.putText(
                    debug,
                    f"frame {frame_idx} avg={avg_error:.4f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cv2.imwrite(os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.jpg"), debug)

            frame_idx += 1

    cap.release()

    print("\nSUMMARY")
    print("Frames compared:", compared)

    if compared == 0:
        print("No comparable frames found.")
        return

    print("Average landmark error:", round(total_avg_error / compared, 4))
    print("Worst frame:", worst_frame)
    print("Worst avg error:", round(worst_avg, 4))

    print("\nKey joint average errors:")
    for name in joint_totals:
        if joint_counts[name]:
            print(f"{name}: {joint_totals[name] / joint_counts[name]:.4f}")

    print(f"\nDebug images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()