from app.phase_engine.squat_v3 import pick_squat_visual_phases

tests = [
    ("knee_valgus", "/Users/josephkamil/Desktop/Capstone/Back Squat- knee valgus.mov", 257, 298, 360),
    ("heel_rise", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- heel rise.mov", 160, 189, 207),
    ("bar_drift", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/Back squats- bar drift.mov", 152, 167, 171),
    ("yt_001", "/Users/josephkamil/Desktop/Capstone/data/dataset_v2/raw/squat_back/yt_001.mp4", 1, 52, 67),
]

for name, path, start, bottom, end in tests:
    print(f"\n== {name} ==")
    print(
        pick_squat_visual_phases(
            path,
            start,
            bottom,
            end,
            sample_step=2,
        )
    )
