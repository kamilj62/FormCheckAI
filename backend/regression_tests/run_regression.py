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
    "analysis":       {"passed": 0, "failed": 0},
    "visuals":        {"passed": 0, "failed": 0},
    "overlay":        {"passed": 0, "failed": 0},
}

def pass_stage(stage): summary[stage]["passed"] += 1
def fail_stage(stage): summary[stage]["failed"] += 1

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

def get_best_rep(reps):
    """Match the frontend getBestRep — highest score wins."""
    if not reps:
        return None
    return max(reps, key=lambda r: float(r.get("score") or 0))

start = time.time()

print("\n========================================")
print("FORMCHECK AI REGRESSION SUITE")
print(f"Testing {len(expected)} exercises")
print("========================================")

for filename, checks in expected.items():
    video_path = VIDEO_DIR / filename

    print(f"\n--- {filename} ---")

    if not video_path.exists():
        print(f"  ❌ Missing video: {video_path}")
        for stage in summary:
            fail_stage(stage)
        continue

    # ── Stage 1: /analyze — classification ───────────────────────────────────
    try:
        r = post_video("/analyze", video_path)
    except Exception as e:
        print(f"  ❌ /analyze request failed: {e}")
        for stage in summary:
            fail_stage(stage)
        continue

    if r.status_code != 200:
        print(f"  ❌ /analyze HTTP {r.status_code}: {r.text[:300]}")
        for stage in summary:
            fail_stage(stage)
        continue

    data = r.json()
    (RESULTS_DIR / f"{filename}.analyze.json").write_text(json.dumps(data, indent=2))

    actual_label  = data.get("exercise_label", "")
    confidence    = float(data.get("confidence") or 0)
    reps          = data.get("rep_feedback") or []
    analysis_mode = data.get("analysis_mode", "")

    expected_label   = checks["label"]
    min_confidence   = checks.get("min_confidence", 0)
    min_reps         = checks.get("min_reps", 0)
    expected_mode    = checks.get("analysis_mode", "any")

    label_ok = normalize_label(actual_label) == normalize_label(expected_label)
    conf_ok  = confidence >= min_confidence
    reps_ok  = len(reps) >= min_reps

    if label_ok and conf_ok and reps_ok:
        print(f"  ✅ Classification: {actual_label}  conf={confidence:.0%}  reps={len(reps)}")
        pass_stage("classification")
    else:
        print(f"  ❌ Classification failed")
        if not label_ok:
            print(f"     label:      expected={expected_label}  got={actual_label}")
        if not conf_ok:
            print(f"     confidence: expected>={min_confidence:.0%}  got={confidence:.0%}")
        if not reps_ok:
            print(f"     reps:       expected>={min_reps}  got={len(reps)}")
        fail_stage("classification")

    # ── Stage 2: analysis shape + coaching validation ─────────────────────────
    allowed_modes = {
        "detailed_rep_analysis",
        "router_v5",
        "biomechanics_override",
        "shape_override",
        "olympic_locked",
        "insufficient_signal",
    }

    mode_ok = (
        True if expected_mode == "any"
        else analysis_mode in allowed_modes if expected_mode == "detailed_rep_analysis"
        else analysis_mode == expected_mode
    )

    rep_list_ok = isinstance(reps, list)
    zones_ok    = "coaching_zones" in data
    summary_ok  = isinstance(data.get("set_summary"), dict)

    # breakdown keys — best rep must have expected keys
    best_rep = get_best_rep(reps)
    expect_breakdown_keys = checks.get("expect_breakdown_keys", [])
    breakdown = best_rep.get("breakdown", {}) if best_rep else {}
    missing_keys = [k for k in expect_breakdown_keys if k not in breakdown]
    breakdown_ok = len(missing_keys) == 0

    # coaching object — Olympic lifts must have rep.coaching.sections
    expect_coaching  = checks.get("expect_coaching", False)
    expect_sections  = checks.get("expect_coaching_sections", [])
    coaching         = best_rep.get("coaching") if best_rep else None
    coaching_ok      = True
    missing_sections = []

    if expect_coaching:
        if not coaching:
            coaching_ok = False
        else:
            actual_titles = [s.get("title", "") for s in (coaching.get("sections") or [])]
            missing_sections = [
                s for s in expect_sections
                if not any(s.lower() in t.lower() for t in actual_titles)
            ]
            coaching_ok = len(missing_sections) == 0

    analysis_ok = all([rep_list_ok, zones_ok, summary_ok, mode_ok, breakdown_ok, coaching_ok])

    if analysis_ok:
        coaching_note = f"  coaching={len(coaching.get('sections', []))} sections" if coaching else ""
        print(f"  ✅ Analysis fields valid  mode={analysis_mode}{coaching_note}")
        pass_stage("analysis")
    else:
        print(f"  ❌ Analysis fields failed")
        if not rep_list_ok:  print(f"     rep_feedback not a list")
        if not zones_ok:     print(f"     coaching_zones missing from response")
        if not summary_ok:   print(f"     set_summary missing or not a dict")
        if not mode_ok:      print(f"     analysis_mode: expected={expected_mode}  got={analysis_mode}")
        if not breakdown_ok: print(f"     breakdown missing keys: {missing_keys}  got: {list(breakdown.keys())}")
        if not coaching_ok:
            if not coaching:
                print(f"     coaching object missing from best rep")
            else:
                print(f"     coaching sections missing: {missing_sections}")
                print(f"     got sections: {[s.get('title') for s in coaching.get('sections', [])]}")
        fail_stage("analysis")

    # ── Stage 3: /generate_visuals ────────────────────────────────────────────
    # Visual generation is frame-sensitive. The first detected rep is
    # typically the most reliable because later reps can end near EOF.
    rep_for_visuals = reps[0] if reps else {}

    try:
        vr = post_video(
            "/generate_visuals",
            video_path,
            extra_data={
                "exercise_label": actual_label or expected_label,
                "rep_json": json.dumps(rep_for_visuals),
            },
        )

        if vr.status_code == 200:
            visuals = vr.json()
            (RESULTS_DIR / f"{filename}.visuals.json").write_text(json.dumps(visuals, indent=2))

            phase_images = visuals.get("phase_images")
            if isinstance(phase_images, dict) and len(phase_images) > 0:
                print(f"  ✅ Visuals: {len(phase_images)} phase images")
                pass_stage("visuals")
            else:
                print(f"  ❌ Visuals: phase_images empty or missing")
                print(f"     visuals_error={visuals.get('visuals_error')}")
                fail_stage("visuals")
        else:
            print(f"  ❌ /generate_visuals HTTP {vr.status_code}: {vr.text[:300]}")
            fail_stage("visuals")

    except Exception as e:
        print(f"  ❌ /generate_visuals failed: {e}")
        fail_stage("visuals")

    # ── Stage 4: /generate_overlay ────────────────────────────────────────────
    try:
        orr = post_video(
            "/generate_overlay",
            video_path,
            extra_data={
                "exercise_label": actual_label or expected_label,
                "rep_json": json.dumps(rep_for_visuals),
            },
            timeout=300,
        )

        if orr.status_code == 200:
            overlay = orr.json()
            (RESULTS_DIR / f"{filename}.overlay.json").write_text(json.dumps(overlay, indent=2))

            if overlay.get("overlay_video_url"):
                print(f"  ✅ Overlay: {overlay['overlay_video_url']}")
                pass_stage("overlay")
            else:
                print(f"  ❌ Overlay: overlay_video_url missing")
                print(f"     overlay_error={overlay.get('overlay_error')}")
                fail_stage("overlay")
        else:
            print(f"  ❌ /generate_overlay HTTP {orr.status_code}: {orr.text[:300]}")
            fail_stage("overlay")

    except Exception as e:
        print(f"  ❌ /generate_overlay failed: {e}")
        fail_stage("overlay")

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - start

print("\n========================================")
print("FORMCHECK AI REGRESSION REPORT")
print("========================================")

total_passed = 0
total_failed = 0

for stage, counts in summary.items():
    p = counts["passed"]
    f = counts["failed"]
    total_passed += p
    total_failed += f
    bar = "✅" if f == 0 else "❌"
    print(f"  {bar} {stage.title():15}  passed={p}  failed={f}")

total = total_passed + total_failed
pct = int(100 * total_passed / total) if total else 0
print(f"\n  Total: {total_passed}/{total} ({pct}%)")
print(f"  Runtime: {elapsed:.1f}s  ({elapsed/len(expected):.1f}s avg per video)")

if total_failed:
    print("\n  OVERALL STATUS: ❌ FAIL")
    raise SystemExit(1)

print("\n  OVERALL STATUS: ✅ PASS")