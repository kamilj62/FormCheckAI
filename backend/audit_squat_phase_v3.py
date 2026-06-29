from pathlib import Path
import os
import cv2
from app.phase_engine.squat_v3 import pick_squat_visual_phases

OUTPUT_DIR = Path("outputs/squat_phase_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    ("knee_valgus", "/Users/josephkamil/Desktop/Capstone/Back Squat- knee valgus.mov", 257, 298, 360),
    ("heel_rise", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- heel rise.mov", 160, 189, 207),
    ("bar_drift", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- bar drift.mov", 152, 167, 171),
    ("yt_001", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/yt_001.mp4", 1, 52, 67),
]

def extract_frame(video_path, frame_idx, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out_path), frame)

for name, path, start, bottom, end in TESTS:
    print("\n==", name, "==")

    phases = pick_squat_visual_phases(path, start, bottom, end, sample_step=2)

    if not phases or "error" in phases:
        print("skip:", phases)
        continue

    video_dir = OUTPUT_DIR / name
    video_dir.mkdir(exist_ok=True)

    for phase, frame in phases.items():
        out_file = video_dir / f"{phase}_{frame}.jpg"
        extract_frame(path, frame, out_file)

    print("saved:", video_dir)
