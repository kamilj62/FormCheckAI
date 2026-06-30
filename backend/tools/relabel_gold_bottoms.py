import json
from pathlib import Path
from app.phase_engine.squat_v3 import extract_pose_records
from app.phase_engine.squat_v4 import find_bottom_v4

GOLD_PATH = Path("gold_squat/gold_bottoms_v1.json")
data = json.loads(GOLD_PATH.read_text())

def get_window(video, start, end):
    return extract_pose_records(
        f"gold_squat/reps/{video}",
        start,
        end,
        sample_step=1
    )

updated = []

for item in data:
    video = item["video"]
    rep = item["rep"]

    start = item["bottom_start"]
    end = item["bottom_end"]

    records = get_window(video, start, end)

    try:
        pred = find_bottom_v4(records, approx_bottom=item["bottom_center"])
    except Exception as e:
        print("SKIP:", item["video"], item["rep"], str(e))
        continue

    corrected = item.copy()

    corrected["bottom_center"] = pred["frame"]

    updated.append(corrected)

    print(f"{video} rep {rep}: {item['bottom_center']} -> {pred['frame']}")

Path("gold_squat/gold_bottoms_v1_relabeled.json").write_text(
    json.dumps(updated, indent=2)
)

print("\n✅ RELABEL COMPLETE")