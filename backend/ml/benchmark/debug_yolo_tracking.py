import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="/tmp/yolo_tracking_debug.mp4")
    parser.add_argument("--model", default="models/yolov8n.pt")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
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

    selected_id = None
    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1

        result = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],
        )[0]

        candidates = []

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, confidence in zip(
                result.boxes.xyxy,
                result.boxes.id,
                result.boxes.conf,
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                tid = int(track_id.item())
                conf = float(confidence.item())

                area = max(1, (x2 - x1) * (y2 - y1))
                center_x = (x1 + x2) / 2
                bottom = y2

                score = area + bottom * 250 - abs(center_x - width / 2) * 2
                candidates.append((score, tid, x1, y1, x2, y2, conf))

        candidates.sort(reverse=True)

        if selected_id is None and candidates:
            selected_id = candidates[0][1]

        for score, tid, x1, y1, x2, y2, conf in candidates:
            thickness = 5 if tid == selected_id else 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness,
            )

            label = (
                f"ID {tid} conf={conf:.2f} "
                f"score={score / 1_000_000:.2f}M"
            )

            if tid == selected_id:
                label += " SELECTED"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"frame={frame_number} selected_id={selected_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Saved: {args.output}")
    print(f"Initial selected target ID: {selected_id}")


if __name__ == "__main__":
    main()
