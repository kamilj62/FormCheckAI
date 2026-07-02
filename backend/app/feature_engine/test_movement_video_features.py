import numpy as np

from app.feature_engine.movement_video_features import (
    build_movement_video_features,
)
from app.feature_engine.feature_names import FEATURE_NAMES

# Fake biomechanics sequence
biomechanics = []

for i in range(120):
    biomechanics.append({
        "knee_angle": 180 - i * 0.7,
        "hip_angle": 180 - i * 0.5,
        "elbow_angle": 90 + i * 0.2,
        "shoulder_angle": 100 + i * 0.1,
        "hip_y": 0.5 + i * 0.001,
        "shoulder_y": 0.3,
        "wrist_y": 0.45 - i * 0.002,
        "wrist_shoulder_distance": 0.25,
    })

features = build_movement_video_features(biomechanics)

print("Feature length:", len(features))
print()

for name, value in zip(FEATURE_NAMES, features):
    print(f"{name:35} {value:8.4f}")

assert len(features) == len(FEATURE_NAMES)
assert np.isfinite(features).all()

print("\nPASS")
