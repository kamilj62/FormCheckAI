import requests

API = "http://formcheck-ai-api-v3.eba-pvfk7qtv.us-west-2.elasticbeanstalk.com"

MIN_SCORES = {
    "Bench Press": 8.0,
    "Push Press": 8.0,
    "Squat": 7.0,
    "Thruster": 8.5,
}

TESTS = [
    (
        "Bench Press",
        "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/bench_press/Bench press- correct.mov",
    ),
    (
        "Push Press",
        "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/push_press/Push Press- correct.mov",
    ),
    (
        "Squat",
        "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/squat_back/Backsquat- correct.mov",
    ),
    (
        "Thruster",
        "/Users/josephkamil/Desktop/Capstone/thruster-correct.mov",
    ),
]

print("=" * 50)
print("FORMCHECK AI CORE CLASSIFIER CHECK")
print("=" * 50)

for expected, path in TESTS:
    print(f"\nTesting {expected}")

    with open(path, "rb") as f:
        response = requests.post(
            f"{API}/analyze",
            files={"file": f},
            timeout=300,
        )

    data = response.json()

    got = data.get("exercise_label")
    reps = data.get("rep_feedback", [])
    score = max((r.get("score", 0) for r in reps), default=None)
    reason = data.get("debug", {}).get("classification_reason")
    original = data.get("debug", {}).get("original_prediction")

    status = (
        "PASS"
        if got == expected
        or (expected == "Squat" and got in ["Squat", "Back Squat"])
        else "FAIL"
    )

    print(f"{status}: expected={expected} got={got}")
    print(f"original={original}")
    print(f"reason={reason}")
    print(f"score={score}")

    min_score = MIN_SCORES.get(expected)

    if (
        score is not None
        and min_score is not None
        and score < min_score
    ):
        print(
            f"WARN: low score for correct sample: "
            f"{score} (expected >= {min_score})"
        )

print("\nDone.")