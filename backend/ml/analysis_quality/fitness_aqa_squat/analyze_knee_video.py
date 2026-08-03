import argparse
import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from app.main import extract_features_and_biomechanics


BASE = Path("ml/analysis_quality/fitness_aqa_squat")
MODEL_DIR = BASE / "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="ml/analysis_quality/fitness_aqa_squat/knee_video_analysis.json",
    )
    return parser.parse_args()


def build_vector(features, biomechanics, biomechanics_keys):
    vector = [float(value) for value in features]

    vector.extend(
        float(biomechanics.get(key, 0.0) or 0.0)
        for key in biomechanics_keys
    )

    vector = np.asarray(vector, dtype=np.float32)

    if not np.all(np.isfinite(vector)):
        return None

    return vector


def summarize_frames(frames, target, threshold):
    if not frames:
        return {
            "target": target,
            "threshold": threshold,
            "frames_analyzed": 0,
            "positive_frames": 0,
            "positive_ratio": 0.0,
            "max_probability": None,
            "mean_probability": None,
        }

    probabilities = np.asarray(
        [frame["probabilities"][target] for frame in frames],
        dtype=float,
    )

    positive = probabilities >= threshold

    return {
        "target": target,
        "threshold": round(float(threshold), 4),
        "frames_analyzed": len(frames),
        "positive_frames": int(positive.sum()),
        "positive_ratio": round(float(positive.mean()), 4),
        "max_probability": round(float(probabilities.max()), 4),
        "mean_probability": round(float(probabilities.mean()), 4),
    }


def main():
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    bundles = {
        "knees_forward": joblib.load(
            MODEL_DIR / "knees_forward_rf_v1.joblib"
        ),
        "knees_inward": joblib.load(
            MODEL_DIR / "knees_inward_rf_v1.joblib"
        ),
    }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if source_fps <= 0 or frame_count <= 0:
        raise SystemExit("Invalid video metadata")

    frame_step = max(1, round(source_fps / args.sample_fps))

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    frames = []
    failed_frames = 0

    for frame_number in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            failed_frames += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            failed_frames += 1
            continue

        features, biomechanics = extract_features_and_biomechanics(results)

        if features is None or biomechanics is None:
            failed_frames += 1
            continue

        probabilities = {}

        for target, bundle in bundles.items():
            vector = build_vector(
                features,
                biomechanics,
                bundle["biomechanics_keys"],
            )

            if vector is None:
                probabilities = {}
                break

            expected = int(bundle["total_feature_count"])

            if len(vector) != expected:
                raise ValueError(
                    f"{target}: expected {expected} features, "
                    f"got {len(vector)}"
                )

            probability = bundle["model"].predict_proba(
                vector.reshape(1, -1)
            )[0, 1]

            probabilities[target] = round(float(probability), 6)

        if not probabilities:
            failed_frames += 1
            continue

        frames.append({
            "frame_number": frame_number,
            "timestamp_seconds": round(frame_number / source_fps, 6),
            "probabilities": probabilities,
            "predictions": {
                target: int(
                    probabilities[target] >= bundles[target]["threshold"]
                )
                for target in bundles
            },
            "biomechanics": {
                "knee_angle": float(
                    biomechanics.get("knee_angle", 0.0) or 0.0
                ),
                "hip_angle": float(
                    biomechanics.get("hip_angle", 0.0) or 0.0
                ),
                "torso_angle": float(
                    biomechanics.get("torso_angle", 0.0) or 0.0
                ),
            },
        })

    cap.release()
    pose.close()

    result = {
        "video": str(video_path),
        "source_fps": source_fps,
        "sample_fps": args.sample_fps,
        "source_frames": frame_count,
        "frames_analyzed": len(frames),
        "failed_frames": failed_frames,
        "summary": {
            target: summarize_frames(
                frames,
                target,
                bundle["threshold"],
            )
            for target, bundle in bundles.items()
        },
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result["summary"], indent=2))
    print()
    print("Full analysis:", output_path)


if __name__ == "__main__":
    main()
