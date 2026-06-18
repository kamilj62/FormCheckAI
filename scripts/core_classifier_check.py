import requests

API = "http://formcheck-ai-api-v3.eba-pvfk7qtv.us-west-2.elasticbeanstalk.com"

TESTS = [
    ("Bench Press", "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/bench_press/Bench press- correct.mov"),
    ("Push Press", "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/push_press/Push Press- correct.mov"),
    ("Squat", "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/squat_back/Backsquat- correct.mov"),
    ("Thruster", "/Users/josephkamil/Desktop/Capstone/thruster-correct.mov"),
]

for expected, path in TESTS:
    print(f"\nTesting {expected}")
    with open(path, "rb") as f:
        r = requests.post(f"{API}/analyze", files={"file": f}, timeout=300)

    data = r.json()
    got = data.get("exercise_label")
    score = data.get("rep_feedback", [{}])[0].get("score")
    reason = data.get("debug", {}).get("classification_reason")
    original = data.get("debug", {}).get("original_prediction")

    status = "PASS" if got == expected or (expected == "Squat" and got in ["Squat", "Back Squat"]) else "FAIL"

    print(f"{status}: expected={expected} got={got}")
    print(f"original={original} reason={reason} score={score}")