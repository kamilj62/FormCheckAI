"""
CoachBench: validates coaching output against gold-standard expectations.

Usage:
    python3 -m ml.coachbench
"""

import json
import subprocess
from pathlib import Path


BENCH_DIR = Path("ml/coachbench")


def analyze(video_path):
    cmd = [
        "curl",
        "-s",
        "--max-time",
        "300",
        "-X",
        "POST",
        "-F",
        f"file=@{video_path}",
        "http://localhost:8000/analyze",
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def normalize_text(x):
    return str(x or "").lower()


def main():
    files = sorted(BENCH_DIR.glob("*.json"))

    total = 0
    passed = 0

    for f in files:
        total += 1
        spec = json.loads(f.read_text())

        print("=" * 70)
        print(f.name)
        print("=" * 70)

        data = analyze(spec["video"])
        reps = data.get("rep_feedback", [])
        rep = reps[0] if reps else {}

        ok = True

        label = data.get("exercise_label")
        if label != spec["expected_label"]:
            print(f"FAIL label: expected {spec['expected_label']}, got {label}")
            ok = False

        score = rep.get("score", 0)
        if not (spec["score_min"] <= score <= spec["score_max"]):
            print(f"FAIL score: expected {spec['score_min']}–{spec['score_max']}, got {score}")
            ok = False

        breakdown = rep.get("breakdown", {})
        for key, expected in spec.get("required_breakdown", {}).items():
            actual = breakdown.get(key)
            if actual != expected:
                print(f"FAIL breakdown.{key}: expected {expected}, got {actual}")
                ok = False

        issues_text = " ".join(normalize_text(i) for i in rep.get("issues", []))
        feedback_text = " ".join(normalize_text(i) for i in rep.get("feedback", []))
        combined = issues_text + " " + feedback_text

        for forbidden in spec.get("forbidden_issues", []):
            if normalize_text(forbidden) in combined:
                print(f"FAIL forbidden issue/text found: {forbidden}")
                ok = False

        if ok:
            passed += 1
            print("PASS")
        else:
            print("FAIL")

        print("Label:", label)
        print("Score:", score)
        print("Feedback:", rep.get("feedback", []))
        print()

    print("=" * 70)
    print(f"CoachBench: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
