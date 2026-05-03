import joblib
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/josephkamil/Desktop/Capstone")
MODEL_PATH = BASE_DIR / "models/oly_router_rf.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

mp_pose = mp.solutions.pose


def extract_features(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")

    rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            feats = []

            for lm in results.pose_landmarks.landmark:
                feats.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility,
                ])

            rows.append(feats)

    cap.release()

    if not rows:
        raise ValueError("No pose landmarks found")

    arr = np.array(rows, dtype=np.float32)

    features = np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
    ])

    return features.reshape(1, -1)


def main():
    # change this path to test
    video = BASE_DIR / "Oly_Data/raw/clean_and_jerk/v_CleanAndJerk_g02_c02.avi"

    X = extract_features(video)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    print("\nPrediction:", pred)

    for label, prob in zip(model.classes_, probs):
        print(f"{label}: {prob:.3f}")


if __name__ == "__main__":
    main()