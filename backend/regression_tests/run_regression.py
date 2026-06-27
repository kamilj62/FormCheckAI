import json
import time
from datetime import datetime
from pathlib import Path

import requests

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

results = []


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


suite_start = time.perf_counter()

print("\n========================================")
print("FORMCHECK AI REGRESSION SUITE")
print("========================================")

for filename, checks in expected.items():
    test_start = time.perf_counter()

    video_path = VIDEO_DIR / filename
    expected_label = checks["label"]

    row = {
        "exercise": filename,
        "classification": False,
        "analysis": False,
        "visuals": False,
        "overlay": False,
        "confidence": 0.0,
        "time": 0.0,
        "expected": expected_label,
        "actual": None,
        "notes": [],
    }

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"❌ Missing video: {video_path}")
        row["notes"].append("Missing video")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        row["time"] = round(time.perf_counter() - test_start, 2)
        results.append(row)
        continue

    # Stage 1: /analyze classification
    try:
        r = post_video("/analyze", video_path)
    except Exception as e:
        print(f"❌ /analyze request failed: {e}")
        row["notes"].append(f"/analyze request failed: {e}")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        row["time"] = round(time.perf_counter() - test_start, 2)
        results.append(row)
        continue

    if r.status_code != 200:
        print(f"❌ /analyze HTTP {r.status_code}: {r.text[:300]}")
        row["notes"].append(f"/analyze HTTP {r.status_code}")
        fail_stage("classification")
        fail_stage("analysis")
        fail_stage("visuals")
        fail_stage("overlay")
        row["time"] = round(time.perf_counter() - test_start, 2)
        results.append(row)
        continue

    data = r.json()
    (RESULTS_DIR / f"{filename}.analyze.json").write_text(json.dumps(data, indent=2))

    actual_label = data.get("exercise_label")
    confidence = data.get("confidence") or 0
    reps = data.get("rep_feedback") or []
    analysis_mode = data.get("analysis_mode")

    row["actual"] = actual_label
    row["confidence"] = float(confidence or 0)

    min_confidence = checks.get("min_confidence", 0)
    min_reps = checks.get("min_reps", 0)
    expected_mode = checks.get("analysis_mode")

    label_ok = normalize_label(actual_label) == normalize_label(expected_label)
    conf_ok = confidence >= min_confidence
    reps_ok = len(reps) >= min_reps

    class_ok = label_ok and conf_ok and reps_ok

    if class_ok:
        print(f"✅ Classification: {actual_label} confidence={confidence} reps={len(reps)}")
        pass_stage("classification")
        row["classification"] = True
    else:
        print("❌ Classification failed")
        print(f"   expected label: {expected_label}")
        print(f"   actual label:   {actual_label}")
        print(f"   confidence:     {confidence} minimum={min_confidence}")
        print(f"   reps:           {len(reps)} minimum={min_reps}")
        fail_stage("classification")
        row["notes"].append(
            f"Classification expected={expected_label}, actual={actual_label}, "
            f"confidence={confidence}, reps={len(reps)}"
        )

    # Stage 2: analysis shape
    feedback_ok = isinstance(data.get("feedback"), list) or isinstance(data.get("rep_feedback"), list)
    zones_ok = "coaching_zones" in data
    summary_ok = "set_summary" in data
    mode_ok = True if not expected_mode else analysis_mode == expected_mode

    analysis_ok = feedback_ok and zones_ok and summary_ok and mode_ok

    if analysis_ok:
        print("✅ Analysis fields present")
        pass_stage("analysis")
        row["analysis"] = True
    else:
        print("❌ Analysis fields failed")
        print(f"   feedback list:   {feedback_ok}")
        print(f"   coaching_zones:  {zones_ok}")
        print(f"   set_summary:     {summary_ok}")
        print(f"   analysis_mode:   {analysis_mode} expected={expected_mode}")
        fail_stage("analysis")
        row["notes"].append("Analysis fields failed")

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
                row["visuals"] = True
            else:
                print("❌ Visuals missing phase_images")
                fail_stage("visuals")
                row["notes"].append("Visuals missing phase_images")
        else:
            print(f"❌ /generate_visuals HTTP {vr.status_code}: {vr.text[:300]}")
            fail_stage("visuals")
            row["notes"].append(f"/generate_visuals HTTP {vr.status_code}")

    except Exception as e:
        print(f"❌ /generate_visuals failed: {e}")
        fail_stage("visuals")
        row["notes"].append(f"/generate_visuals failed: {e}")

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
                row["overlay"] = True
            else:
                print("❌ Overlay missing overlay_video_url")
                print(f"   overlay_error={overlay.get('overlay_error')}")
                fail_stage("overlay")
                row["notes"].append("Overlay missing overlay_video_url")
        else:
            print(f"❌ /generate_overlay HTTP {orr.status_code}: {orr.text[:300]}")
            fail_stage("overlay")
            row["notes"].append(f"/generate_overlay HTTP {orr.status_code}")

    except Exception as e:
        print(f"❌ /generate_overlay failed: {e}")
        fail_stage("overlay")
        row["notes"].append(f"/generate_overlay failed: {e}")

    row["time"] = round(time.perf_counter() - test_start, 2)
    results.append(row)


suite_time = round(time.perf_counter() - suite_start, 2)
total_failed = sum(counts["failed"] for counts in summary.values())

print("\n")
print("=" * 78)
print("                    FORMCHECK AI REGRESSION REPORT")
print("=" * 78)

print(
    f'{"Exercise":24}'
    f'{"Class":8}'
    f'{"Analysis":10}'
    f'{"Visual":8}'
    f'{"Overlay":9}'
    f'{"Conf":8}'
    f'{"Time"}'
)
print("-" * 78)

for r in results:
    print(
        f'{r["exercise"][:22]:24}'
        f'{"✅" if r["classification"] else "❌":8}'
        f'{"✅" if r["analysis"] else "❌":10}'
        f'{"✅" if r["visuals"] else "❌":8}'
        f'{"✅" if r["overlay"] else "❌":9}'
        f'{r["confidence"]:.2f}    '
        f'{r["time"]:.2f}s'
    )

print("-" * 78)

for stage, counts in summary.items():
    passed = counts["passed"]
    failed = counts["failed"]
    total = passed + failed
    print(f"{stage.title():15} {passed}/{total} passed, {failed} failed")

print(f"\nRuntime: {suite_time:.2f}s")

print("\nSlowest Analyses")
print("-" * 30)
for r in sorted(results, key=lambda x: x["time"], reverse=True)[:3]:
    print(f'{r["exercise"]:<24} {r["time"]:.2f}s')

failed_rows = [
    r for r in results
    if not (r["classification"] and r["analysis"] and r["visuals"] and r["overlay"])
]

print("\nFailed Tests")
print("-" * 30)

if not failed_rows:
    print("None 🎉")
else:
    for r in failed_rows:
        print(f'\n{r["exercise"]}')
        print(f'  Expected: {r["expected"]}')
        print(f'  Actual:   {r["actual"]}')
        for note in r["notes"]:
            print(f"  - {note}")

# Markdown report
report_path = RESULTS_DIR / "Regression_Report.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

md = []
md.append("# FormCheck AI Regression Report\n")
md.append(f"Generated: {timestamp}\n")
md.append(f"Overall Status: {'PASS' if total_failed == 0 else 'FAIL'}\n")
md.append(f"Runtime: {suite_time:.2f}s\n")

md.append("## Summary\n")
for stage, counts in summary.items():
    passed = counts["passed"]
    failed = counts["failed"]
    total = passed + failed
    md.append(f"- **{stage.title()}**: {passed}/{total} passed, {failed} failed")
md.append("")

md.append("## Results\n")
md.append("| Exercise | Class | Analysis | Visuals | Overlay | Confidence | Time | Expected | Actual |")
md.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

for r in results:
    md.append(
        f'| {r["exercise"]} '
        f'| {"PASS" if r["classification"] else "FAIL"} '
        f'| {"PASS" if r["analysis"] else "FAIL"} '
        f'| {"PASS" if r["visuals"] else "FAIL"} '
        f'| {"PASS" if r["overlay"] else "FAIL"} '
        f'| {r["confidence"]:.2f} '
        f'| {r["time"]:.2f}s '
        f'| {r["expected"]} '
        f'| {r["actual"]} |'
    )

md.append("\n## Failed Tests\n")
if not failed_rows:
    md.append("None 🎉")
else:
    for r in failed_rows:
        md.append(f"### {r['exercise']}")
        md.append(f"- Expected: {r['expected']}")
        md.append(f"- Actual: {r['actual']}")
        for note in r["notes"]:
            md.append(f"- {note}")
        md.append("")

report_path.write_text("\n".join(md))
print(f"\nMarkdown report written to: {report_path}")

if total_failed:
    print("\nOVERALL STATUS: FAIL")
    raise SystemExit(1)

print("\nOVERALL STATUS: PASS")
