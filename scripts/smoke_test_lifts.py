import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

TEST_VIDEOS = [
    {
        "label": "Bench Press",
        "path": "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/bench_press/Bench press- correct.mov",
        "expected": "Bench Press",
        "required_phases": ["setup", "descent", "bottom", "press", "lockout"],
    },
    {
        "label": "Deadlift",
        "path": "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/deadlift/Deadlift.mov",
        "expected": "Deadlift",
        "required_phases": ["setup", "pull", "mid", "finish", "lockout"],
    },
    {
        "label": "Push Press",
        "path": "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/push_press/Push Press- correct.mov",
        "expected": "Push Press",
        "required_phases": ["setup", "dip", "drive", "catch", "lockout"],
    },
    {
        "label": "Squat",
        "path": "/Users/josephkamil/Desktop/Capstone/data/dataset_a/raw/squat_back/Backsquat- correct.mov",
        "expected": "Squat",
        "required_phases": ["setup", "descent", "bottom", "ascent", "lockout"],
    },
]


def assert_url_exists(url_path):
    if not url_path:
        return False

    res = requests.get(f"{API_URL}{url_path}", timeout=15)
    return res.status_code == 200


def test_lift(video):
    print(f"\nTesting {video['label']}")

    path = Path(video["path"])

    if not path.exists():
        print(f"SKIP: file not found: {path}")
        return False

    with open(path, "rb") as f:
        res = requests.post(
            f"{API_URL}/analyze",
            files={"file": (path.name, f, "video/quicktime")},
            timeout=180,
        )

    if res.status_code != 200:
        print(f"FAIL: /analyze returned {res.status_code}")
        print(res.text[:500])
        return False

    data = res.json()

    exercise = data.get("exercise_label")
    reps = data.get("rep_feedback") or []
    overlay = data.get("overlay_video_url")
    phases = data.get("phase_images") or {}

    passed = True

    if exercise != video["expected"]:
        print(f"FAIL: expected {video['expected']}, got {exercise}")
        passed = False
    else:
        print(f"PASS: detected {exercise}")

    if not reps:
        print("FAIL: no reps detected")
        passed = False
    else:
        print(f"PASS: detected {len(reps)} reps")

    if not overlay or not assert_url_exists(overlay):
        print(f"FAIL: overlay missing or unreachable: {overlay}")
        passed = False
    else:
        print(f"PASS: overlay exists: {overlay}")

    for phase in video["required_phases"]:
        phase_url = phases.get(phase)

        if not phase_url or not assert_url_exists(phase_url):
            print(f"FAIL: missing phase image: {phase}")
            passed = False
        else:
            print(f"PASS: {phase} image exists")

    if passed:
        print(f"✅ {video['label']} passed")
    else:
        print(f"❌ {video['label']} failed")

    return passed


def main():
    print("Running FormCheck AI smoke tests...")

    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        health.raise_for_status()
        print("PASS: backend health check")
    except Exception as e:
        print("FAIL: backend is not running")
        print(e)
        return

    results = [test_lift(video) for video in TEST_VIDEOS]

    print("\n====================")
    print("SUMMARY")
    print("====================")

    passed = sum(results)
    total = len(results)

    print(f"{passed}/{total} lift tests passed")

    if passed == total:
        print("✅ All core lift smoke tests passed")
    else:
        print("❌ Some lift tests failed")


if __name__ == "__main__":
    main()