#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml.movement_signatures import (
    LABEL_SIGNATURES,
    MOVEMENT_SIGNATURES,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--candidates",
    default="backend/ml/benchmark/config/analyzer_audit_18_exercises.csv",
)
parser.add_argument(
    "--out",
    default="backend/ml/benchmark/results/analyzer_audit_latest.csv",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=120,
)
parser.add_argument(
    "--limit",
    type=int,
    default=0,
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Ignore any existing output file and analyze every candidate again.",
)
parser.add_argument(
    "--save-responses",
    default="",
    help="Optional directory for saving each raw analyzer JSON response.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit non-zero when any analyzed row fails or is blocked.",
)
args = parser.parse_args()

def resolve_workspace_path(value):
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_label(label):
    signature = LABEL_SIGNATURES.get(str(label or ""))
    if signature:
        return signature.internal_label
    return str(label or "")


def label_family(label):
    signature = LABEL_SIGNATURES.get(str(label or ""))
    if signature:
        return signature.family
    return ""


def canonical_coverage_label(label):
    """
    Map manifest aliases to their canonical external movement name.

    Examples:
      squat_back  -> back_squat
      squat_front -> front_squat

    Internal/alias labels such as muscle_up are therefore not reported as
    separate missing coverage when canonical movement variants already exist.
    """
    label = str(label or "")

    if label in MOVEMENT_SIGNATURES:
        return label

    for canonical_label, signature in MOVEMENT_SIGNATURES.items():
        if label in signature.aliases:
            return canonical_label

    return label


def truthy(value):
    return str(value).lower() in {"true", "1", "yes"}


def is_blank(value):
    return value is None or str(value).strip() == ""


def is_runtime_blocked(row):
    mode = str(row.get("analysis_mode") or "")
    error = str(row.get("error") or "")
    return (
        mode == "pose_runtime_error"
        or error.startswith("curl_exit_")
        or "connection refused" in error.lower()
        or "kGpuService" in error
        or "NSOpenGLPixelFormat" in error
        or "ImageToTensorCalculator" in error
    )


def row_passes(row):
    if is_runtime_blocked(row):
        return False

    if not truthy(row.get("label_ok")):
        return False

    rep_count_ok = row.get("rep_count_ok")
    if is_blank(rep_count_ok):
        return True

    return truthy(rep_count_ok)


def normalize_result_row(row):
    expected = row.get("expected_label") or row.get("label")
    predicted = row.get("predicted_label")
    expected_family = label_family(expected)
    predicted_family = label_family(predicted)

    row["expected_label"] = expected
    row["expected_family"] = expected_family
    row["predicted_family"] = predicted_family
    row["label_ok"] = normalize_label(predicted) == normalize_label(expected)
    row["family_ok"] = bool(expected_family and predicted_family) and (
        expected_family == predicted_family
    )
    row["overall_ok"] = row_passes(row)
    return row


CANDIDATES = resolve_workspace_path(args.candidates)
latest = resolve_workspace_path(args.out)
OUT_DIR = latest.parent
latest.parent.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR = (
    resolve_workspace_path(args.save_responses)
    if args.save_responses
    else None
)
if RESPONSES_DIR is not None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

def analyze(video):
    cmd = [
        "curl",
        "--max-time", str(max(1, int(args.timeout))),
        "-s",
        "-X", "POST",
        "-F", f"file=@{video}",
        "http://127.0.0.1:8000/analyze",
    ]

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if r.returncode != 0:
        return {
            "error": r.stderr.strip() or f"curl_exit_{r.returncode}"
        }

    try:
        return json.loads(r.stdout)
    except Exception:
        return {
            "error": "invalid_json",
            "raw": r.stdout[:500],
        }

rows = list(csv.DictReader(CANDIDATES.open()))
if args.limit > 0:
    rows = rows[: args.limit]

results = []
label_stats = defaultdict(Counter)
confusions = Counter()

fieldnames = [
    "expected_label",
    "expected_family",
    "video",
    "name",
    "predicted_label",
    "predicted_family",
    "label_ok",
    "family_ok",
    "overall_ok",
    "confidence",
    "detected_reps",
    "expected_reps",
    "rep_delta",
    "rep_verdict",
    "rep_scores",
    "rep_frames",
    "rep_count_ok",
    "rep_phase_complete",
    "rep_phase_ordered",
    "missing_phase_fields",
    "analysis_mode",
    "protected_reason",
    "raw_label",
    "bio_label",
    "squat_label",
    "olympic_label",
    "router_v8_winner",
    "family_shadow",
    "family_margin",
    "learned_family",
    "learned_family_confidence",
    "learned_family_trusted",
    "learned_press",
    "learned_press_confidence",
    "learned_press_trusted",
    "hierarchical_label",
    "hierarchical_family",
    "hierarchical_source",
    "error",
]

completed = {}

if latest.exists() and not args.fresh:
    try:
        reader = csv.DictReader(latest.open())
        existing_fields = set(reader.fieldnames or [])
        required_resume_fields = {
            "video",
            "predicted_label",
            "rep_count_ok",
            "rep_delta",
            "rep_verdict",
        }

        if not required_resume_fields.issubset(existing_fields):
            print("Resume: ignoring stale result schema")
        else:
            for prior in reader:
                video = prior.get("video")
                error = (prior.get("error") or "").strip()

                predicted = (prior.get("predicted_label") or "").strip()

                # Keep only genuine completed analyses.
                # Retry timeouts / empty analyzer responses.
                if video and not error and predicted:
                    completed[video] = normalize_result_row(prior)

            print(f"Resume: keeping {len(completed)} valid completed rows")
    except Exception:
        completed = {}
elif args.fresh:
    print("Resume: fresh run requested; ignoring existing rows")

def process_row(index, row):
    expected = row["label"]
    expected_normalized = normalize_label(expected)
    expected_family = label_family(expected)
    video = row["video"]
    name = row["name"]

    if not Path(video).exists():
        return {
            "expected_label": expected,
            "expected_family": expected_family,
            "video": video,
            "name": name,
            "predicted_label": "",
            "predicted_family": "",
            "label_ok": False,
            "family_ok": False,
            "overall_ok": False,
            "confidence": "",
            "detected_reps": "",
            "expected_reps": row.get("expected_reps") or "",
            "rep_delta": "",
            "rep_verdict": "",
            "rep_scores": "",
            "rep_frames": "",
            "rep_count_ok": "",
            "rep_phase_complete": "",
            "rep_phase_ordered": "",
            "missing_phase_fields": "",
            "analysis_mode": "",
            "protected_reason": "",
            "raw_label": "",
            "bio_label": "",
            "squat_label": "",
            "olympic_label": "",
            "router_v8_winner": "",
            "error": "missing_file",
        }

    data = analyze(video)
    if RESPONSES_DIR is not None:
        safe_name = (
            f"{expected}_{name}"
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        response_path = RESPONSES_DIR / f"{safe_name}.json"
        response_path.write_text(
            json.dumps(data, indent=2, sort_keys=True)
        )

    if data.get("error") and not data.get("exercise_label"):
        return {
            "expected_label": expected,
            "expected_family": expected_family,
            "video": video,
            "name": name,
            "predicted_label": "",
            "predicted_family": "",
            "label_ok": False,
            "family_ok": False,
            "overall_ok": False,
            "confidence": "",
            "detected_reps": "",
            "expected_reps": row.get("expected_reps") or "",
            "rep_delta": "",
            "rep_verdict": "",
            "rep_scores": "",
            "rep_frames": "",
            "rep_count_ok": "",
            "rep_phase_complete": "",
            "rep_phase_ordered": "",
            "missing_phase_fields": "",
            "analysis_mode": "",
            "protected_reason": "",
            "raw_label": "",
            "bio_label": "",
            "squat_label": "",
            "olympic_label": "",
            "router_v8_winner": "",
            "family_shadow": "",
            "family_margin": "",
            "learned_family": "",
            "learned_family_confidence": "",
            "learned_family_trusted": "",
            "learned_press": "",
            "learned_press_confidence": "",
            "learned_press_trusted": "",
            "hierarchical_label": "",
            "hierarchical_family": "",
            "hierarchical_source": "",
            "error": data.get("error"),
        }

    predicted = data.get("exercise_label")
    predicted_normalized = normalize_label(predicted)
    predicted_family = label_family(predicted)
    confidence = data.get("confidence")
    rep_feedback = data.get("rep_feedback") or []
    reps = len(rep_feedback)
    analysis_mode = data.get("analysis_mode")
    debug = data.get("debug") or {}

    expected_reps_raw = (row.get("expected_reps") or "").strip()
    rep_count_ok = ""
    rep_delta = ""
    rep_verdict = ""

    if expected_reps_raw:
        try:
            expected_reps = int(expected_reps_raw)
            rep_count_ok = reps == expected_reps
            rep_delta = reps - expected_reps
            if rep_delta == 0:
                rep_verdict = "exact"
            elif rep_delta < 0:
                rep_verdict = "undercount"
            else:
                rep_verdict = "overcount"
        except Exception:
            expected_reps = expected_reps_raw
    else:
        expected_reps = ""

    router_v8 = debug.get("router_v8") or {}
    family_shadow = debug.get("family_router_shadow") or {}
    hierarchical_shadow = debug.get("hierarchical_router_shadow") or {}
    rep_detector = debug.get("rep_detector") or {}
    rep_phase_rows = (
        rep_detector.get("reps")
        if isinstance(rep_detector, dict)
        else []
    )
    missing_phase_fields = []
    rep_scores = []
    rep_frames = []

    if isinstance(rep_feedback, list):
        for rep in rep_feedback:
            if not isinstance(rep, dict):
                continue

            if rep.get("score") is not None:
                rep_scores.append(str(rep.get("score")))

            frames = [
                str(rep.get(key))
                for key in (
                    "start_frame",
                    "dip_frame",
                    "drive_frame",
                    "catch_frame",
                    "lockout_frame",
                    "end_frame",
                )
                if rep.get(key) is not None
            ]
            if frames:
                rep_frames.append(":".join(frames))

    if isinstance(rep_phase_rows, list):
        for rep_row in rep_phase_rows:
            if not isinstance(rep_row, dict):
                continue

            missing_phase_fields.extend(
                str(field)
                for field in rep_row.get("missing_fields", [])
            )

    return {
        "expected_label": expected,
        "expected_family": expected_family,
        "video": video,
        "name": name,
        "predicted_label": predicted,
        "predicted_family": predicted_family,
        "label_ok": predicted_normalized == expected_normalized,
        "family_ok": bool(expected_family and predicted_family)
        and predicted_family == expected_family,
        "overall_ok": (
            predicted_normalized == expected_normalized
            and (rep_count_ok == "" or rep_count_ok is True)
        ),
        "confidence": confidence,
        "detected_reps": reps,
        "expected_reps": expected_reps,
        "rep_delta": rep_delta,
        "rep_verdict": rep_verdict,
        "rep_scores": ",".join(rep_scores),
        "rep_frames": "|".join(rep_frames),
        "rep_count_ok": rep_count_ok,
        "rep_phase_complete": (
            rep_detector.get("phase_complete")
            if isinstance(rep_detector, dict)
            else ""
        ),
        "rep_phase_ordered": (
            rep_detector.get("phase_ordered")
            if isinstance(rep_detector, dict)
            else ""
        ),
        "missing_phase_fields": ",".join(sorted(set(missing_phase_fields))),
        "analysis_mode": analysis_mode,
        "protected_reason": debug.get("protected_reason"),
        "raw_label": debug.get("raw_label"),
        "bio_label": debug.get("bio_label"),
        "squat_label": debug.get("squat_label"),
        "olympic_label": debug.get("olympic_pred"),
        "router_v8_winner": (
            router_v8.get("winner")
            if isinstance(router_v8, dict)
            else None
        ),
        "family_shadow": (
            family_shadow.get("family")
            if isinstance(family_shadow, dict)
            else None
        ),
        "family_margin": (
            family_shadow.get("margin")
            if isinstance(family_shadow, dict)
            else None
        ),
        "learned_family": debug.get(
            "learned_family_shadow_label"
        ),
        "learned_family_confidence": debug.get(
            "learned_family_shadow_confidence"
        ),
        "learned_family_trusted": debug.get(
            "learned_family_shadow_trusted"
        ),
        "learned_press": debug.get(
            "learned_press_shadow_label"
        ),
        "learned_press_confidence": debug.get(
            "learned_press_shadow_confidence"
        ),
        "learned_press_trusted": debug.get(
            "learned_press_shadow_trusted"
        ),
        "hierarchical_label": (
            hierarchical_shadow.get("label")
            if isinstance(hierarchical_shadow, dict)
            else None
        ),
        "hierarchical_family": (
            hierarchical_shadow.get("family")
            if isinstance(hierarchical_shadow, dict)
            else None
        ),
        "hierarchical_source": (
            hierarchical_shadow.get("source")
            if isinstance(hierarchical_shadow, dict)
            else None
        ),
        "error": data.get("error"),
    }

pending = [
    (i, row)
    for i, row in enumerate(rows, 1)
    if row["video"] not in completed
]

print(f"Total candidates: {len(rows)}", flush=True)
print(f"Already completed: {len(completed)}", flush=True)
print(f"Pending: {len(pending)}", flush=True)
print("Workers: 1", flush=True)

all_results = list(completed.values())

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {
        executor.submit(process_row, i, row): (i, row)
        for i, row in pending
    }

    for future in as_completed(futures):
        i, row = futures[future]

        try:
            result = future.result()
        except Exception as exc:
            result = {
                "expected_label": row["label"],
                "expected_family": label_family(row["label"]),
                "video": row["video"],
                "name": row["name"],
                "predicted_label": "",
                "predicted_family": "",
                "label_ok": False,
                "family_ok": False,
                "overall_ok": False,
                "confidence": "",
                "detected_reps": "",
                "expected_reps": row.get("expected_reps") or "",
                "rep_delta": "",
                "rep_verdict": "",
                "rep_scores": "",
                "rep_frames": "",
                "rep_count_ok": "",
                "rep_phase_complete": "",
                "rep_phase_ordered": "",
                "missing_phase_fields": "",
                "analysis_mode": "",
                "protected_reason": "",
                "raw_label": "",
                "bio_label": "",
                "squat_label": "",
                "olympic_label": "",
                "router_v8_winner": "",
                "error": f"exception:{exc}",
            }

        all_results.append(result)

        print(
            f"[{len(all_results)}/{len(rows)}] "
            f"{result['expected_label']} :: {result['name']} "
            f"-> {result['predicted_label']} "
            f"reps={result['detected_reps']}/{result['expected_reps']} "
            f"rep={result.get('rep_verdict') or 'n/a'} "
            f"{'BLOCKED' if is_runtime_blocked(result) else 'PASS' if row_passes(result) else 'FAIL'}",
            flush=True,
        )

        with latest.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(all_results)

results = all_results

coverage_stats = Counter(
    canonical_coverage_label(row["label"])
    for row in rows
)
family_stats = defaultdict(Counter)

for r in results:
    expected = r["expected_label"]
    family = r.get("expected_family") or label_family(expected)

    if r.get("error") == "missing_file":
        label_stats[expected]["missing"] += 1
        family_stats[family]["missing"] += 1
        continue

    if is_runtime_blocked(r):
        label_stats[expected]["blocked"] += 1
        family_stats[family]["blocked"] += 1
        continue

    if row_passes(r):
        label_stats[expected]["pass"] += 1
    else:
        label_stats[expected]["fail"] += 1
        confusions[
            (
                expected,
                r.get("predicted_label") or "None",
            )
        ] += 1

    if truthy(r.get("family_ok")) and (
        is_blank(r.get("rep_count_ok"))
        or truthy(r.get("rep_count_ok"))
    ):
        family_stats[family]["pass"] += 1
    else:
        family_stats[family]["fail"] += 1

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

timestamped = OUT_DIR / f"analyzer_audit_{ts}.csv"

for path in (latest, timestamped):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

total = len(results)
passed = sum(1 for r in results if row_passes(r))
blocked = sum(1 for r in results if is_runtime_blocked(r))
failed = total - passed - blocked

print()
print("=" * 72)
print("ANALYZER AUDIT SUMMARY")
print("=" * 72)
print(f"Total:  {total}")
print(f"Passed: {passed}")
print(f"Blocked: {blocked}")
print(f"Failed: {failed}")

scored_total = total - blocked
if scored_total:
    print(f"Overall accuracy: {passed / scored_total:.1%}")
elif total:
    print("Overall accuracy: n/a (all rows blocked)")

print()
print("COVERAGE")
print("-" * 72)
for label in sorted(MOVEMENT_SIGNATURES):
    count = coverage_stats[label]
    status = "ok" if count else "missing"
    if 0 < count < 4:
        status = "thin"
    print(f"{label:24} candidates={count:3} {status}")

print()
print("BY FAMILY")
print("-" * 72)

for family in sorted(family_stats):
    p = family_stats[family]["pass"]
    f = family_stats[family]["fail"]
    m = family_stats[family]["missing"]
    b = family_stats[family]["blocked"]
    denom = p + f

    accuracy = f"{p / denom:.1%}" if denom else "n/a"

    print(
        f"{family:24} "
        f"pass={p:3} "
        f"fail={f:3} "
        f"blocked={b:3} "
        f"missing={m:3} "
        f"accuracy={accuracy}"
    )

print()
print("BY MOVEMENT")
print("-" * 72)

for label in sorted(label_stats):
    p = label_stats[label]["pass"]
    f = label_stats[label]["fail"]
    m = label_stats[label]["missing"]
    b = label_stats[label]["blocked"]
    denom = p + f

    accuracy = (
        f"{p / denom:.1%}"
        if denom
        else "n/a"
    )

    print(
        f"{label:24} "
        f"pass={p:3} "
        f"fail={f:3} "
        f"blocked={b:3} "
        f"missing={m:3} "
        f"accuracy={accuracy}"
    )

print()
print("TOP CONFUSIONS")
print("-" * 72)

for (expected, predicted), count in confusions.most_common(25):
    print(
        f"{expected:22} -> "
        f"{predicted:22} "
        f"{count}"
    )

print()
print(f"Saved:  {timestamped}")
print(f"Latest: {latest}")

if args.strict and (failed or blocked):
    raise SystemExit(1)
