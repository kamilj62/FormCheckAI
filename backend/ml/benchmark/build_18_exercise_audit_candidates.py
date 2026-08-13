#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml.movement_signatures import MOVEMENT_SIGNATURES


ROOT = Path("/Users/josephkamil/Desktop/Capstone")
OUT = Path("backend/ml/benchmark/config/analyzer_audit_18_exercises.csv")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v"}
PER_LABEL = 4
CANONICAL_LABELS = tuple(MOVEMENT_SIGNATURES)

LABEL_PATTERNS = [
    ("clean_and_jerk", ["clean_and_jerk", "cleanandjerk", "clean and jerk"]),
    ("handstand_push_up", ["handstand_push_up", "handstand push", "handstand-push", "hspu"]),
    ("ring_muscle_up", ["ring_muscle_up", "ring muscle"]),
    ("bar_muscle_up", ["bar_muscle_up", "bar muscle"]),
    ("overhead_squat", ["overhead_squat", "overhead squat", "ohs"]),
    ("split_jerk", ["split_jerk", "splitjerk", "split jerk"]),
    ("strict_press", ["strict_press", "strict press", "shoulder_press", "shoulder press"]),
    ("front_squat", ["front_squat", "frontsquat", "front squat"]),
    ("bench_press", ["bench_press", "bench press", "bench"]),
    ("push_press", ["push_press", "push press"]),
    ("push_up", ["push_up", "push-up", "push up", "pushups", "pushups"]),
    ("pull_up", ["pull_up", "pull up", "pull-up", "pullups", "strictpullup"]),
    ("back_squat", ["back_squat", "backsquat", "back squat", "squat_back"]),
    ("deadlift", ["deadlift"]),
    ("thruster", ["thruster"]),
    ("burpee", ["burpee"]),
    ("snatch", ["snatch"]),
    ("clean", ["clean"]),
]

SEARCH_ROOTS = [
    ROOT / "FormCheck_Phase_Audit",
    ROOT / "FormCheck_Phase_Audit_v2",
    ROOT / "holdout_v1",
    ROOT / "router_challenge_v1" / "raw",
    ROOT / "router_challenge_v1" / "regression_cases",
    ROOT / "strict_press_sources",
    ROOT / "strict_press_clips",
    ROOT / "front_squat_candidates",
    ROOT / "BackSquat_Audit",
    ROOT / "BackSquat_Real_Audit",
    ROOT / "fresh_validation_2026_08" / "videos",
    ROOT / "data" / "raw",
    ROOT / "data" / "raw_videos",
    ROOT / "Oly_Data" / "raw",
    ROOT / "uindy_external_validation" / "videos",
    ROOT / "PushUps",
    ROOT / "PullUps",
    ROOT / "HandstandPushups",
    ROOT / "BenchPress",
    ROOT / "deadlift",
    ROOT / "bench press",
    ROOT / "pull Up",
    ROOT / "push-up",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\\-]+", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text


def infer_label(path: Path) -> str | None:
    haystack = normalize(" ".join(path.relative_to(ROOT).parts))
    for label, needles in LABEL_PATTERNS:
        if any(normalize(needle) in haystack for needle in needles):
            return label
    return None


def sort_key(path: Path) -> tuple[int, int, str]:
    preferred = 0
    parts = set(path.relative_to(ROOT).parts)
    if "FormCheck_Phase_Audit" in parts or "FormCheck_Phase_Audit_v2" in parts:
        preferred = -2
    elif "holdout_v1" in parts or "router_challenge_v1" in parts:
        preferred = -1
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return preferred, size, str(path)


def main() -> None:
    by_label: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            if path in seen:
                continue
            seen.add(path)
            label = infer_label(path)
            if label:
                by_label[label].append(path)

    rows = []
    for label, paths in sorted(by_label.items()):
        for path in sorted(paths, key=sort_key)[:PER_LABEL]:
            rows.append({
                "label": label,
                "video": str(path),
                "name": path.name,
                "expected_reps": "",
                "notes": "",
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "video", "name", "expected_reps", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidates to {OUT}")
    print()
    print("COVERAGE")
    print("-" * 72)
    for label in CANONICAL_LABELS:
        found = len(by_label[label])
        selected = min(found, PER_LABEL)
        status = "ok" if selected else "missing"
        if 0 < selected < PER_LABEL:
            status = "thin"
        print(f"{label:20} selected={selected:2} found={found:3} {status}")

    missing = [label for label in CANONICAL_LABELS if not by_label[label]]
    if missing:
        print()
        print("Missing canonical exercises:", ", ".join(missing))


if __name__ == "__main__":
    main()
