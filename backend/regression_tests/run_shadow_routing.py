import json
import time
from pathlib import Path

import requests


API_BASE = "http://127.0.0.1:8000"

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
EXPECTED_PATH = BASE / "expected_results.json"
OUTPUT_PATH = BASE / "results" / "central_router_shadow_results.json"

expected = json.loads(EXPECTED_PATH.read_text())


def normalize_label(label):
    return str(label or "").strip().lower().replace(" ", "_")


def analyze_video(video_path):
    with video_path.open("rb") as file_handle:
        response = requests.post(
            f"{API_BASE}/analyze",
            files={
                "file": (
                    video_path.name,
                    file_handle,
                    "video/quicktime",
                )
            },
            timeout=300,
        )

    response.raise_for_status()
    return response.json()


results = []
start_time = time.time()

print()
print("=" * 72)
print("CENTRAL ROUTER SHADOW REGRESSION")
print(f"Testing {len(expected)} exercises")
print("=" * 72)

for filename, checks in expected.items():
    video_path = VIDEO_DIR / filename
    expected_label = checks["label"]

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"  MISSING: {video_path}")

        results.append(
            {
                "file": filename,
                "expected": expected_label,
                "error": "missing_video",
            }
        )
        continue

    try:
        data = analyze_video(video_path)
    except Exception as exc:
        print(f"  ERROR: {exc}")

        results.append(
            {
                "file": filename,
                "expected": expected_label,
                "error": str(exc),
            }
        )
        continue

    debug = data.get("debug") or {}
    shadow = debug.get("central_router_shadow") or {}

    current_label = data.get("exercise_label")
    shadow_label = shadow.get("label")

    current_ok = (
        normalize_label(current_label)
        == normalize_label(expected_label)
    )

    shadow_ok = (
        normalize_label(shadow_label)
        == normalize_label(expected_label)
    )

    marker = "PASS" if shadow_ok else "FAIL"

    print(
        f"  Expected: {expected_label}\n"
        f"  Current : {current_label} "
        f"{'PASS' if current_ok else 'FAIL'}\n"
        f"  Shadow  : {shadow_label} {marker}\n"
        f"  Score   : {shadow.get('score')}\n"
        f"  Margin  : {shadow.get('margin')}\n"
        f"  Runner  : "
        f"{(shadow.get('runner_up') or {}).get('label')}"
    )

    results.append(
        {
            "file": filename,
            "expected": expected_label,
            "current": {
                "label": current_label,
                "confidence": data.get("confidence"),
                "mode": data.get("analysis_mode"),
                "reps": len(data.get("rep_feedback") or []),
                "pass": current_ok,
            },
            "shadow": {
                "label": shadow_label,
                "score": shadow.get("score"),
                "margin": shadow.get("margin"),
                "runner_up": shadow.get("runner_up"),
                "scores": shadow.get("scores"),
                "eligibility": shadow.get("eligibility"),
                "signals": shadow.get("signals"),
                "pass": shadow_ok,
            },
            "candidates": debug.get("routing_candidates"),
        }
    )


valid_results = [
    result
    for result in results
    if "shadow" in result
]

current_passed = sum(
    1
    for result in valid_results
    if result["current"]["pass"]
)

shadow_passed = sum(
    1
    for result in valid_results
    if result["shadow"]["pass"]
)

summary = {
    "total_expected": len(expected),
    "total_analyzed": len(valid_results),
    "current_passed": current_passed,
    "shadow_passed": shadow_passed,
    "current_accuracy": (
        round(current_passed / len(valid_results), 4)
        if valid_results
        else 0.0
    ),
    "shadow_accuracy": (
        round(shadow_passed / len(valid_results), 4)
        if valid_results
        else 0.0
    ),
    "runtime_seconds": round(time.time() - start_time, 2),
}

payload = {
    "summary": summary,
    "results": results,
}

OUTPUT_PATH.parent.mkdir(exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

print()
print("=" * 72)
print("SHADOW ROUTING SUMMARY")
print("=" * 72)
print(
    f"Current: {current_passed}/{len(valid_results)} "
    f"({summary['current_accuracy']:.1%})"
)
print(
    f"Shadow : {shadow_passed}/{len(valid_results)} "
    f"({summary['shadow_accuracy']:.1%})"
)
print(f"Runtime: {summary['runtime_seconds']} seconds")
print(f"Saved  : {OUTPUT_PATH}")
print("=" * 72)

raise SystemExit(0 if shadow_passed == len(valid_results) else 1)
