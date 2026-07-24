import json
import time
from pathlib import Path

import requests


API_BASE = "http://127.0.0.1:8000"

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
EXPECTED_PATH = BASE / "expected_results.json"
OUTPUT_PATH = BASE / "results" / "family_router_shadow_results.json"

expected = json.loads(EXPECTED_PATH.read_text())


LABEL_TO_FAMILY = {
    "squat": "squat",
    "squat_back": "squat",
    "squat_front": "squat",
    "overhead_squat": "squat",

    "clean": "olympic",
    "clean_and_jerk": "olympic",
    "snatch": "olympic",
    "split_jerk": "olympic",

    "bench_press": "press",
    "strict_press": "press",
    "push_press": "press",
    "thruster": "press",

    "pull_up": "bodyweight",
    "push_up": "bodyweight",
    "handstand_push_up": "bodyweight",
    "burpee": "bodyweight",
    "muscle_up": "bodyweight",

    "deadlift": "hinge",
}


def normalize(label):
    return str(label or "").strip().lower().replace(" ", "_")


def expected_family(label):
    return LABEL_TO_FAMILY.get(normalize(label), "unknown")


def analyze_video(video_path):
    with video_path.open("rb") as handle:
        response = requests.post(
            f"{API_BASE}/analyze",
            files={
                "file": (
                    video_path.name,
                    handle,
                    "video/quicktime",
                )
            },
            timeout=300,
        )

    response.raise_for_status()
    return response.json()


results = []
start = time.time()

print()
print("=" * 72)
print("FAMILY ROUTER SHADOW REGRESSION")
print(f"Testing {len(expected)} exercises")
print("=" * 72)

for filename, checks in expected.items():
    video_path = VIDEO_DIR / filename
    expected_label = normalize(checks["label"])
    target_family = expected_family(expected_label)

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"  MISSING: {video_path}")
        results.append({
            "file": filename,
            "expected_label": expected_label,
            "expected_family": target_family,
            "error": "missing_video",
        })
        continue

    try:
        data = analyze_video(video_path)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append({
            "file": filename,
            "expected_label": expected_label,
            "expected_family": target_family,
            "error": str(exc),
        })
        continue

    debug = data.get("debug") or {}
    family_shadow = debug.get("family_router_shadow") or {}

    predicted_family = normalize(family_shadow.get("family"))
    passed = predicted_family == target_family

    print(f"  Expected label : {expected_label}")
    print(f"  Expected family: {target_family}")
    print(f"  Current label  : {data.get('exercise_label')}")
    print(
        f"  Shadow family  : {predicted_family} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"  Score          : {family_shadow.get('score')}")
    print(f"  Margin         : {family_shadow.get('margin')}")
    print(
        "  Runner-up      : "
        f"{(family_shadow.get('runner_up') or {}).get('family')}"
    )

    results.append({
        "file": filename,
        "expected_label": expected_label,
        "expected_family": target_family,
        "current_label": data.get("exercise_label"),
        "family_shadow": family_shadow,
        "routing_candidates": debug.get("routing_candidates"),
        "pass": passed,
    })


valid = [item for item in results if "pass" in item]
passed = sum(1 for item in valid if item["pass"])

summary = {
    "total_expected": len(expected),
    "total_analyzed": len(valid),
    "passed": passed,
    "failed": len(valid) - passed,
    "accuracy": (
        round(passed / len(valid), 4)
        if valid
        else 0.0
    ),
    "runtime_seconds": round(time.time() - start, 2),
}

payload = {
    "summary": summary,
    "results": results,
}

OUTPUT_PATH.parent.mkdir(exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

print()
print("=" * 72)
print("FAMILY ROUTER SUMMARY")
print("=" * 72)
print(
    f"Passed: {passed}/{len(valid)} "
    f"({summary['accuracy']:.1%})"
)
print(f"Runtime: {summary['runtime_seconds']} seconds")
print(f"Saved: {OUTPUT_PATH}")
print("=" * 72)

raise SystemExit(0 if passed == len(valid) else 1)
