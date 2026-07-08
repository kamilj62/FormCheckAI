import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "agents/baselines/regression_baseline.json"
REPORT_PATH = ROOT / "agents/reports/latest_regression_report.json"

API_URL = "http://localhost:8000/analyze"

TESTS = [
    {
        "name": "Back Squat",
        "video": "/Users/josephkamil/Desktop/Capstone/backsquat_short.mov",
        "expected_labels": ["squat_back", "back_squat", "Back Squat"],
        "min_reps": 1,
    },
    {
        "name": "Overhead Squat",
        "video": "/Users/josephkamil/Desktop/Capstone/OverheadSquat- correct5.mov",
        "expected_labels": ["overhead_squat", "Overhead Squat"],
        "min_reps": 1,
    },

    {
        "name": "Deadlift",
        "video": "/Users/josephkamil/Desktop/Capstone/deadlift/deadlift_1.mp4",
        "expected_labels": ["deadlift", "Deadlift"],
        "min_reps": 1,
    },
    {
        "name": "Bench Press",
        "video": "/Users/josephkamil/Desktop/Capstone/bench_short.mov",
        "expected_labels": ["bench_press", "Bench Press"],
        "min_reps": 1,
    },
    {
        "name": "Push Press",
        "video": "/Users/josephkamil/Desktop/Capstone/idealPushPress.mov",
        "expected_labels": ["push_press", "Push Press"],
        "min_reps": 1,
    },
    {
        "name": "Clean",
        "video": "/Users/josephkamil/Desktop/Capstone/clean-correct.mov",
        "expected_labels": ["clean", "Clean"],
        "min_reps": 1,
    },
    {
        "name": "Snatch",
        "video": "/Users/josephkamil/Desktop/Capstone/snatch- correct.mov",
        "expected_labels": ["snatch", "Snatch"],
        "min_reps": 0,
    },
    {
        "name": "Push-up",
        "video": "/Users/josephkamil/Desktop/Capstone/push-up/push-up_1.mp4",
        "expected_labels": ["push_up", "Push-up", "Push Up"],
        "min_reps": 1,
    },
    {
        "name": "Handstand Push-up",
        "video": "/Users/josephkamil/Desktop/Capstone/Handstand Push-up.mov",
        "expected_labels": ["handstand_push_up", "Handstand Push-up"],
        "min_reps": 1,
    },

    {
        "name": "Front Squat",
        "video": "/Users/josephkamil/Desktop/Capstone/data/datasets/google_drive_export/FormCheck_Data/raw/squat_front__correct/Front squat- correct.mov",
        "expected_labels": ["squat_front", "front_squat", "Front Squat"],
        "min_reps": 1,
    },
]


def call_analyze(video):
    cmd = [
        "curl", "--max-time", "300", "-s",
        "-X", "POST",
        "-F", f"file=@{video}",
        API_URL,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr or "curl failed")

    return json.loads(result.stdout)


def summarize_response(data):
    reps = data.get("rep_feedback") or []
    scores = []

    for r in reps:
        scores.append({
            "rep": r.get("rep"),
            "score": r.get("score"),
            "start_frame": r.get("start_frame"),
            "end_frame": r.get("end_frame"),
        })

    debug = data.get("debug") or {}

    return {
        "label": data.get("exercise_label"),
        "confidence": data.get("confidence"),
        "analysis_mode": data.get("analysis_mode"),
        "rep_count": len(reps),
        "scores": scores,
        "debug": {
            "raw_label": debug.get("raw_label"),
            "raw_confidence": debug.get("raw_confidence"),
            "bio_label": debug.get("bio_label"),
            "bio_conf": debug.get("bio_conf"),
            "squat_label": debug.get("squat_label"),
            "squat_conf": debug.get("squat_conf"),
            "bodyweight_router_label": debug.get("bodyweight_router_label"),
            "bodyweight_router_conf": debug.get("bodyweight_router_conf"),
            "olympic_pred": debug.get("olympic_pred"),
            "olympic_conf": debug.get("olympic_conf"),
            "router_v5_label": debug.get("router_v5_label"),
            "router_v5_conf": debug.get("router_v5_conf"),
            "protected_label": debug.get("protected_label"),
            "protected_reason": debug.get("protected_reason"),
        },
    }


def compare_to_expectations(test, summary):
    failures = []

    if summary["label"] not in test["expected_labels"]:
        failures.append(
            f"Expected label in {test['expected_labels']}, got {summary['label']}"
        )

    if summary["rep_count"] < test.get("min_reps", 0):
        failures.append(
            f"Expected at least {test.get('min_reps', 0)} reps, got {summary['rep_count']}"
        )

    return failures


def run_test(test):
    video = Path(test["video"])

    if not video.exists():
        return {
            "name": test["name"],
            "status": "SKIP",
            "reason": f"Missing video: {video}",
        }

    try:
        data = call_analyze(str(video))
        summary = summarize_response(data)
        failures = compare_to_expectations(test, summary)

        return {
            "name": test["name"],
            "status": "FAIL" if failures else "PASS",
            "failures": failures,
            "summary": summary,
        }

    except Exception as e:
        return {
            "name": test["name"],
            "status": "FAIL",
            "failures": [str(e)],
        }


def save_baseline(results):
    baseline = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": {
            r["name"]: r.get("summary")
            for r in results
            if r["status"] == "PASS" and r.get("summary")
        },
    }

    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"\nSaved baseline: {BASELINE_PATH}")


def print_result(r):
    print(f"\n[{r['status']}] {r['name']}")

    if r.get("failures"):
        for f in r["failures"]:
            print("Failure:", f)

    summary = r.get("summary")
    if not summary:
        return

    print("Label:", summary["label"])
    print("Reps:", summary["rep_count"])
    print("Scores:", summary["scores"])

    debug = summary.get("debug") or {}
    print("Debug:")
    for k, v in debug.items():
        if v is not None:
            print(f"  {k}: {v}")


def main():
    import sys

    save_mode = "--save-baseline" in sys.argv

    print("\nFormCheck AI Regression Agent")
    print("=" * 40)

    results = [run_test(t) for t in TESTS]

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    for r in results:
        print_result(r)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\nSummary")
    print("=" * 40)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Report: {REPORT_PATH}")

    if save_mode:
        if failed == 0:
            save_baseline(results)
        else:
            print("\nBaseline not saved because tests failed.")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
