from app.phase_engine.squat_v3 import extract_pose_records
from app.phase_engine.bottom_detector import find_bottom_v1


def evaluate():

    dataset = [
        {
            "video": "gold_squat/reps/squat_25.mp4",
            "window_start": 24,
            "window_end": 96,
            "bottom_center": 54
        },
    ]

    total = 0
    passed = 0

    for item in dataset:

        video = item["video"]

        # -------------------------------------------------------
        # 1. Extract pose (KEEP ABSOLUTE FRAME SPACE)
        # -------------------------------------------------------
        records = extract_pose_records(
            video,
            item["window_start"],
            item["window_end"],
            sample_step=1
        )

        if len(records) == 0:
            continue

        # -------------------------------------------------------
        # 2. DO NOT SHIFT ANYTHING (IMPORTANT)
        # -------------------------------------------------------
        gold_center = item["bottom_center"]

        # -------------------------------------------------------
        # 3. Run model
        # -------------------------------------------------------
        pred = find_bottom_v1(
            records,
            approx_bottom=gold_center
        )

        # -------------------------------------------------------
        # 4. Evaluate
        # -------------------------------------------------------
        error = abs(pred["frame"] - gold_center)

        print(
            f"{item['video']} "
            f"gold: {gold_center} "
            f"pred: {pred['frame']} "
            f"error: {error}"
        )

        total += 1

        if error <= 5:
            passed += 1

        print("FIRST FRAME:", records[0]["frame"])
        print("GOLD:", gold_center)
        print("PRED:", pred["frame"])
        print("-" * 40)

    print("\n--- SUMMARY ---")
    print(f"Pass rate: {passed} / {total}")


if __name__ == "__main__":
    evaluate()