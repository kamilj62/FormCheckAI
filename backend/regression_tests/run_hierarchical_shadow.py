import json
import time
from pathlib import Path

import requests


API_BASE = "http://127.0.0.1:8000"

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
EXPECTED_PATH = BASE / "expected_results.json"
OUTPUT_PATH = (
    BASE
    / "results"
    / "hierarchical_router_shadow_results.json"
)

expected = json.loads(EXPECTED_PATH.read_text())


def normalize(label):
    return str(label or "").strip().lower().replace(" ", "_")


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
print("HIERARCHICAL ROUTER SHADOW REGRESSION")
print(f"Testing {len(expected)} exercises")
print("=" * 72)

for filename, checks in expected.items():
    video_path = VIDEO_DIR / filename
    expected_label = normalize(checks["label"])

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
    hierarchical = (
        debug.get("hierarchical_router_shadow") or {}
    )

    production_label = normalize(data.get("exercise_label"))
    shadow_label = normalize(hierarchical.get("label"))

    production_pass = production_label == expected_label
    shadow_pass = shadow_label == expected_label

    print(f"  Expected  : {expected_label}")
    print(
        f"  Production: {production_label} "
        f"{'PASS' if production_pass else 'FAIL'}"
    )
    print(f"  Family    : {family_shadow.get('family')}")
    print(
        f"  Shadow    : {shadow_label} "
        f"{'PASS' if shadow_pass else 'FAIL'}"
    )
    print(f"  Source    : {hierarchical.get('source')}")
    print(f"  Confidence: {hierarchical.get('confidence')}")
    print(f"  Fam margin: {hierarchical.get('family_margin')}")

    results.append({
        "file": filename,
        "expected": expected_label,
        "production": {
            "label": production_label,
            "pass": production_pass,
        },
        "family_shadow": family_shadow,
        "hierarchical_shadow": hierarchical,
        "shadow_pass": shadow_pass,
    })


valid = [
    item
    for item in results
    if "shadow_pass" in item
]

production_passed = sum(
    1
    for item in valid
    if item["production"]["pass"]
)

shadow_passed = sum(
    1
    for item in valid
    if item["shadow_pass"]
)

summary = {
    "total_expected": len(expected),
    "total_analyzed": len(valid),
    "production_passed": production_passed,
    "shadow_passed": shadow_passed,
    "production_accuracy": (
        round(production_passed / len(valid), 4)
        if valid
        else 0.0
    ),
    "shadow_accuracy": (
        round(shadow_passed / len(valid), 4)
        if valid
        else 0.0
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
print("HIERARCHICAL SHADOW SUMMARY")
print("=" * 72)
print(
    f"Production: {production_passed}/{len(valid)} "
    f"({summary['production_accuracy']:.1%})"
)
print(
    f"Shadow    : {shadow_passed}/{len(valid)} "
    f"({summary['shadow_accuracy']:.1%})"
)
print(f"Runtime   : {summary['runtime_seconds']} seconds")
print(f"Saved     : {OUTPUT_PATH}")
print("=" * 72)

raise SystemExit(
    0 if shadow_passed == len(valid) else 1
)
