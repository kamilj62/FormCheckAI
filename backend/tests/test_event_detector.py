import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.movement.event_detector import detect_movement_events

def test_simple_squat():
    bio = []

    for i in range(100):
        if i < 40:
            knee = 180 - i * 1.2
        elif i < 60:
            knee = 132
        else:
            knee = 132 + (i - 60) * 1.5

        bio.append({
            "knee_angle": knee,
            "hip_angle": i,
            "wrist_y": 1.0 - i * 0.01,
        })

    events = detect_movement_events(bio, "squat")

    print(events)

    assert 35 <= events["bottom"] <= 60