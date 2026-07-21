import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from app.tracking import YOLOTracker, remap_crop_landmarks_to_full_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="/tmp/yolo_pose_debug.mp4")
    parser.add_argument("--pad", type=int, default=40)
    parser.add_argument("--model", default="models/yolov8n.pt")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    tracker = YOLOTracker(args.model, pad=args.pad)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_number = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            crop_result = tracker.get_crop(frame)

            analysis_frame = crop_result.crop
            if analysis_frame is None or analysis_frame.size == 0:
                analysis_frame = frame
                crop_result = None

            rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if crop_result is not None:
                x1, y1, x2, y2 = crop_result.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

                cv2.putText(
                    frame,
                    f"YOLO target={crop_result.target_id} pad={args.pad}",
                    (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            if results.pose_landmarks:
                if crop_result is not None:
                    results = remap_crop_landmarks_to_full_frame(
                        results,
                        crop_result.box,
                        width,
                        height,
                    )

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            else:
                cv2.putText(
                    frame,
                    "NO POSE",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            label = f"FRAME {frame_number}"

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                4,
            )

            text_x = max(20, (width - text_size[0]) // 2)

            cv2.rectangle(
                frame,
                (text_x - 15, 10),
                (text_x + text_size[0] + 15, 65),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (text_x, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                4,
            )

            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
