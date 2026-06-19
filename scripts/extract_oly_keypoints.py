import csv
import cv2
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path("/Users/josephkamil/Desktop/Capstone")

DATASETS = {
    "clean_and_jerk": BASE_DIR / "Oly_Data/raw/clean_and_jerk",
    "snatch": BASE_DIR / "Oly_Data/raw/snatch_mp4",
    "not_oly": BASE_DIR / "Oly_Data/raw/not_oly",
}

OUTPUT_CSV = BASE_DIR / "Oly_Data/oly_keypoints.csv"

mp_pose = mp.solutions.pose


def extract_video(video_path, label, writer):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, total_frames // 120)

    rows = 0
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % sample_every != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            row = {
                "video": video_path.name,
                "label": label,
                "frame": frame_idx,
            }

            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f"x_{i}"] = lm.x
                row[f"y_{i}"] = lm.y
                row[f"z_{i}"] = lm.z
                row[f"v_{i}"] = lm.visibility

            writer.writerow(row)
            rows += 1

    cap.release()
    return rows


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["video", "label", "frame"]
    for i in range(33):
        fieldnames += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]

    total_videos = 0
    total_rows = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, input_dir in DATASETS.items():
            videos = sorted(input_dir.glob("*.avi")) + sorted(input_dir.glob("*.mp4"))

            print("\n==============================")
            print(f"Label: {label}")
            print(f"Folder: {input_dir}")
            print(f"Videos found: {len(videos)}")
            print("==============================")

            if not videos:
                continue

            for idx, video in enumerate(videos, start=1):
                print(f"[{label}] [{idx}/{len(videos)}] {video.name}")
                rows = extract_video(video, label, writer)
                print(f"  rows: {rows}")

                total_videos += 1
                total_rows += rows

    print("\nDONE")
    print(f"Videos processed: {total_videos}")
    print(f"Rows written: {total_rows}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()