import json
import time
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
EXPECTED_PATH = BASE / "expected_results.json"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

expected = json.loads(EXPECTED_PATH.read_text())

summary = {
    "classification": {"passed": 0, "failed": 0},
    "analysis": {"passed": 0, "failed": 0},
    "visuals": {"passed": 0, "failed": 0},
    "overlay": {"passed": 0, "failed": 0},
}

def pass_stage(stage):
    summary[stage]["passed"] += 1

def fail_stage(stage):
    summary[stage]["failed"] += 1

def normalize_label(label):
    return str(label or "").strip().lower().replace("_", " ")

def post_video(endpoint, video_path, extra_data=None, timeout=300):
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/quicktime")}
        return requests.post(
            f"{API_BASE}{endpoint}",
            files=files,
            data=extra_data or {},
            timeout=timeout,
        )

start = time.time()

print("\n========================================")
print("FORMCHECK AI REGRESSION SUITE")
print("========================================")

for filename, checks in expected.items():
    video_path = VIDEO_DIR / filename

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"❌ Missing video: {video_path}")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        continue

    # Stage 1: /analyze classification
    try:
        r = post_video("/analyze", video_path)
    except Exception as e:
        print(f"❌ /analyze request failed: {e}")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        continue

    if r.status_code != 200:
        print(f"❌ /analyze HTTP {r.status_code}: {r.text[:300]}")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        continue

    data = r.json()
    (RESULTS_DIR / f"{filename}.analyze.json").write_text(json.dumps(data, indent=2))

    actual_label = data.get("exercise_label")
    confidence = data.get("confidence") or 0
    reps = data.get("rep_feedback") or []
    analysis_mode = data.get("analysis_mode")

    expected_label = checks["label"]
    min_confidence = checks.get("min_confidence", 0)
    min_reps = checks.get("min_reps", 0)
    expected_mode = checks.get("analysis_mode")

    label_ok = normalize_label(actual_label) == normalize_label(expected_label)
    conf_ok = confidence >= min_confidence
    reps_ok = len(reps) >= min_reps

    if label_ok and conf_ok and reps_ok:
        print(f"✅ Classification: {actual_label} confidence={confidence} reps={len(reps)}")
        pass_stage("classification")
    else:
        print(f"❌ Classification failed")
        print(f"   expected label: {expected_label}")
        print(f"   actual label:   {actual_label}")
        print(f"   confidence:     {confidence} minimum={min_confidence}")
        print(f"   reps:           {len(reps)} minimum={min_reps}")
        fail_stage("classification")

    # Stage 2: analysis shape
    # Your API usually stores useful coaching inside rep_feedback and set_summary,
    # not always in top-level feedback.
    feedback_ok = isinstance(data.get("feedback"), list) or isinstance(data.get("rep_feedback"), list)
    zones_ok = "coaching_zones" in data
    summary_ok = "set_summary" in data
    mode_ok = True if not expected_mode else analysis_mode == expected_mode

    if feedback_ok and zones_ok and summary_ok and mode_ok:
        print("✅ Analysis fields present")
        pass_stage("analysis")
    else:
        print("❌ Analysis fields failed")
        print(f"   feedback list:   {feedback_ok}")
        print(f"   coaching_zones:  {zones_ok}")
        print(f"   set_summary:     {summary_ok}")
        print(f"   analysis_mode:   {analysis_mode} expected={expected_mode}")
        fail_stage("analysis")

    # Pick first rep for visual/overlay calls
    rep_json = reps[0] if reps else {}

    # Stage 3: /generate_visuals
    try:
        vr = post_video(
            "/generate_visuals",
            video_path,
            extra_data={
                "exercise_label": actual_label or expected_label,
                "rep_json": json.dumps(rep_json),
            },
        )

        if vr.status_code == 200:
            visuals = vr.json()
            (RESULTS_DIR / f"{filename}.visuals.json").write_text(json.dumps(visuals, indent=2))

            phase_images = visuals.get("phase_images")
            if isinstance(phase_images, dict) and len(phase_images) > 0:
                print(f"✅ Visuals generated: {len(phase_images)} images")
                pass_stage("visuals")
            else:
                print("❌ Visuals missing phase_images")
                fail_stage("visuals")
        else:
            print(f"❌ /generate_visuals HTTP {vr.status_code}: {vr.text[:300]}")
            fail_stage("visuals")

    except Exception as e:
        print(f"❌ /generate_visuals failed: {e}")
        fail_stage("visuals")

    # Stage 4: /generate_overlay
    try:
        orr = post_video(
            "/generate_overlay",
            video_path,
            extra_data={
                "exercise_label": actual_label or expected_label,
                "rep_json": json.dumps(rep_json),
            },
            timeout=300,
        )

        if orr.status_code == 200:
            overlay = orr.json()
            (RESULTS_DIR / f"{filename}.overlay.json").write_text(json.dumps(overlay, indent=2))

            if overlay.get("overlay_video_url"):
                print(f"✅ Overlay generated: {overlay.get('overlay_video_url')}")
                pass_stage("overlay")
            else:
                print("❌ Overlay missing overlay_video_url")
                print(f"   overlay_error={overlay.get('overlay_error')}")
                fail_stage("overlay")
        else:
            print(f"❌ /generate_overlay HTTP {orr.status_code}: {orr.text[:300]}")
            fail_stage("overlay")

    except Exception as e:
        print(f"❌ /generate_overlay failed: {e}")
        fail_stage("overlay")

elapsed = time.time() - start

print("\n========================================")
print("FORMCHECK AI REGRESSION REPORT")
print("========================================")

total_failed = 0

for stage, counts in summary.items():
    passed = counts["passed"]
    failed = counts["failed"]
    total_failed += failed
    print(f"{stage.title():15} passed={passed} failed={failed}")

print(f"\nRuntime: {elapsed:.1f}s")

if total_failed:
    print("\nOVERALL STATUS: FAIL")
    raise SystemExit(1)

print("\nOVERALL STATUS: PASS")
