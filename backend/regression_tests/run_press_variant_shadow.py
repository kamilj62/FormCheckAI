import json
import time
from pathlib import Path

import requests


API_BASE = "http://127.0.0.1:8000"

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
OUTPUT_PATH = (
    BASE
    / "results"
    / "press_variant_shadow_results.json"
)

EXPECTED = {
    "bench_press.mov": "bench_press",
    "strict_press.mov": "strict_press",
    "push_press.mov": "push_press",
    "thruster.mp4": "thruster",
}


def analyze_video(video_path: Path) -> dict:
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
started = time.time()

print()
print("=" * 72)
print("PRESS VARIANT SHADOW REGRESSION")
print(f"Testing {len(EXPECTED)} press variants")
print("=" * 72)

for filename, expected_label in EXPECTED.items():
    video_path = VIDEO_DIR / filename

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"  MISSING: {video_path}")
        results.append({
            "file": filename,
            "expected": expected_label,
            "error": "missing_video",
            "pass": False,
        })
        continue

    try:
        data = analyze_video(video_path)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append({
            "file": filename,
            "expected": expected_label,
            "error": str(exc),
            "pass": False,
        })
        continue

    debug = data.get("debug") or {}
    family_shadow = debug.get("family_router_shadow") or {}
    press_shadow = debug.get("press_variant_shadow") or {}

    family = family_shadow.get("family")
    shadow_label = press_shadow.get("label")

    passed = (
        family == "press"
        and shadow_label == expected_label
    )

    print(f"  Expected: {expected_label}")
    print(f"  Current : {data.get('exercise_label')}")
    print(f"  Family  : {family}")
    print(
        f"  Shadow  : {shadow_label} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"  Score   : {press_shadow.get('score')}")
    print(f"  Margin  : {press_shadow.get('margin')}")
    print(
        "  Runner  : "
        f"{(press_shadow.get('runner_up') or {}).get('label')}"
    )

    results.append({
        "file": filename,
        "expected": expected_label,
        "current_label": data.get("exercise_label"),
        "family_shadow": family_shadow,
        "press_variant_shadow": press_shadow,
        "pass": passed,
    })


passed_count = sum(
    1 for item in results
    if item.get("pass")
)

summary = {
    "total": len(EXPECTED),
    "passed": passed_count,
    "failed": len(EXPECTED) - passed_count,
    "accuracy": round(
        passed_count / len(EXPECTED),
        4,
    ),
    "runtime_seconds": round(
        time.time() - started,
        2,
    ),
}

OUTPUT_PATH.parent.mkdir(exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(
        {
            "summary": summary,
            "results": results,
        },
        indent=2,
    )
)

print()
print("=" * 72)
print("PRESS VARIANT SHADOW SUMMARY")
print("=" * 72)
print(
    f"Passed: {passed_count}/{len(EXPECTED)} "
    f"({summary['accuracy']:.1%})"
)
print(f"Runtime: {summary['runtime_seconds']} seconds")
print(f"Saved: {OUTPUT_PATH}")
print("=" * 72)

raise SystemExit(
    0 if passed_count == len(EXPECTED) else 1
)
