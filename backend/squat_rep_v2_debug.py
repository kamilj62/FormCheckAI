import json, subprocess

VIDEOS = [
("yt_001", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/yt_001.mp4"),
("yt_003", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/yt_003.mp4"),
("knee_valgus", "/Users/josephkamil/Desktop/Capstone/Back Squat- knee valgus.mov"),
]

for name, path in VIDEOS:
    print("\n==", name, "==")

    r = subprocess.run([
        "curl", "--max-time", "300", "-s",
        "-H", "Expect:",
        "-F", f"file=@{path}",
        "http://127.0.0.1:8000/analyze",
    ], capture_output=True, text=True)

    data = json.loads(r.stdout)

    print("label:", data.get("exercise_label"), "conf:", data.get("confidence"))
    for rep in data.get("rep_feedback", []):
        print({
            "rep": rep.get("rep"),
            "start": rep.get("start_frame"),
            "descent": rep.get("descent_frame"),
            "bottom": rep.get("bottom_frame"),
            "ascent": rep.get("ascent_frame"),
            "end": rep.get("end_frame"),
            "score": rep.get("score"),
        })
