import os
import json
from app.main import analyze_video

VIDEOS = [
    "regression_tests/videos/pull_up.mov",
    "regression_tests/videos/ring_muscle_up.mov",
    "regression_tests/videos/handstand_push_up.mov",
    "regression_tests/videos/thruster.mp4",
    "regression_tests/videos/front_squat.mov",
]

for video in VIDEOS:
    print("\n==============================")
    print(video)
    print("==============================")

    r = analyze_video(video, make_visuals=False, make_overlay=False)

    print(json.dumps({
        "label": r.get("exercise_label"),
        "confidence": r.get("confidence"),
        "rep_count": len(r.get("rep_feedback") or []),
        "debug": r.get("debug"),
    }, indent=2))
