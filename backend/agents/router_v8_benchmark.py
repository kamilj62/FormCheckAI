#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".avi", ".m4v", ".mpeg", ".mpg"
}

LABEL_ALIASES = {
    "back_squat": "squat_back",
    "backsquat": "squat_back",
    "squat_back": "squat_back",

    "front_squat": "squat_front",
    "frontsquat": "squat_front",
    "squat_front": "squat_front",

    "overhead_squat": "overhead_squat",
    "overheadsquat": "overhead_squat",

    "bench": "bench_press",
    "bench_press": "bench_press",

    "deadlift": "deadlift",

    "push_press": "push_press",
    "pushpress": "push_press",

    "strict_press": "strict_press",
    "strictpress": "strict_press",

    "thruster": "thruster",

    "clean": "clean",
    "real_clean": "clean",

    "clean_and_jerk": "clean_and_jerk",
    "cleanandjerk": "clean_and_jerk",

    "snatch": "snatch",
    "real_snatch": "snatch",
    "snatch_mp4": "snatch",
    "snatching_videos": "snatch",

    "split_jerk": "split_jerk",
    "splitjerk": "split_jerk",

    "pull_up": "pull_up",
    "pullup": "pull_up",

    "push_up": "push_up",
    "pushup": "push_up",

    "handstand_push_up": "handstand_push_up",
    "handstandpushup": "handstand_push_up",

    "burpee": "burpee",

    "bar_muscle_up": "muscle_up",
    "bar_muscleup": "muscle_up",
    "ring_muscle_up": "muscle_up",
    "ring_muscleup": "muscle_up",
    "muscle_up": "muscle_up",
    "muscleup": "muscle_up",
}


def normalize_name(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("&", "and")
    value = value.replace("-", "_").replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def infer_expected_label(path: Path) -> str | None:
    for parent in path.parents:
        normalized = normalize_name(parent.name)

        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]

        for alias, label in sorted(
            LABEL_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias in normalized:
                return label

    return None


def discover_videos(
    roots: list[Path],
) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for root in roots:
        root = root.expanduser().resolve()

        if not root.exists():
            print(
                f"Warning: root does not exist: {root}",
                file=sys.stderr,
            )
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            expected = infer_expected_label(path)

            if expected is None:
                continue

            seen.add(resolved)
            records.append((resolved, expected))

    return sorted(
        records,
        key=lambda item: (item[1], str(item[0])),
    )


def analyze_video(
    path: Path,
    api_url: str,
    timeout: int,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(timeout),
            "-X",
            "POST",
            "-F",
            f"file=@{path}",
            api_url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "error": (
                result.stderr.strip()
                or f"curl exited {result.returncode}"
            )
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "error": f"Invalid JSON: {exc}",
            "response_preview": result.stdout[:500],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare production routing against "
            "Router V8 shadow predictions."
        )
    )

    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Labeled dataset roots.",
    )

    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/analyze",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "agents/reports/router_v8_benchmark"
        ),
    )

    args = parser.parse_args()

    discovered = discover_videos(args.roots)

    # Remove exact duplicate files that appear under conflicting labels.
    hash_groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)

    unreadable_files: list[tuple[Path, str, str]] = []

    for path, expected in discovered:
        digest = hashlib.sha256()

        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            unreadable_files.append((path, expected, str(exc)))
            print(
                f"Skipping unreadable file: {path} ({exc})",
                file=sys.stderr,
            )
            continue

        hash_groups[digest.hexdigest()].append((path, expected))

    conflicting_hashes = {
        digest
        for digest, items in hash_groups.items()
        if len({expected for _, expected in items}) > 1
    }

    filtered: list[tuple[Path, str]] = []

    for digest, items in hash_groups.items():
        if digest in conflicting_hashes:
            continue

        # Keep one copy of same-label duplicates.
        filtered.append(items[0])

    print(
        f"Excluded {len(conflicting_hashes)} conflicting duplicate groups"
    )
    print(
        f"Skipped {len(unreadable_files)} unreadable files"
    )

    by_class: dict[str, list[Path]] = defaultdict(list)

    for path, expected in filtered:
        by_class[expected].append(path)

    selected: list[tuple[Path, str]] = []

    for expected in sorted(by_class):
        for path in by_class[expected][:args.max_per_class]:
            selected.append((path, expected))

    if not selected:
        print(
            "No labeled videos discovered.",
            file=sys.stderr,
        )
        return 1

    print(f"Discovered {len(discovered)} labeled clips")
    print(
        f"Selected {len(selected)} clips "
        f"across {len(by_class)} classes"
    )
    print()

    rows: list[dict[str, Any]] = []
    started = time.time()

    for index, (path, expected) in enumerate(
        selected,
        start=1,
    ):
        print(
            f"[{index:>3}/{len(selected)}] "
            f"{expected:<22} {path.name}",
            flush=True,
        )

        payload = analyze_video(
            path=path,
            api_url=args.api_url,
            timeout=args.timeout,
        )

        if payload.get("error"):
            row = {
                "video": str(path),
                "expected": expected,
                "production": None,
                "production_confidence": None,
                "v8": None,
                "v8_confidence": None,
                "production_correct": False,
                "v8_correct": False,
                "agreement": False,
                "decision": "error",
                "family": None,
                "reason": None,
                "error": payload["error"],
            }

            rows.append(row)
            print(f"    ERROR: {payload['error']}")
            continue

        production = payload.get("exercise_label")
        production_conf = payload.get("confidence")

        router_v8 = (
            (payload.get("debug") or {})
            .get("router_v8")
            or {}
        )

        winner = router_v8.get("winner")
        selected_lock = (
            router_v8.get("selected_lock")
            or {}
        )

        row = {
            "video": str(path),
            "expected": expected,
            "production": production,
            "production_confidence": production_conf,
            "v8": winner,
            "v8_confidence": router_v8.get(
                "winner_confidence"
            ),
            "snapshot": {
                "predictions": router_v8.get("predictions") or [],
                "state": router_v8.get("state") or {},
            },
            "production_correct": production == expected,
            "v8_correct": winner == expected,
            "agreement": production == winner,
            "decision": router_v8.get("decision"),
            "family": router_v8.get(
                "winning_family"
            ),
            "reason": selected_lock.get("reason"),
            "error": router_v8.get("error"),
        }

        rows.append(row)

        print(
            f"    expected={expected} "
            f"production={production} "
            f"v8={winner}"
        )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    report = {
        "roots": [
            str(root.expanduser().resolve())
            for root in args.roots
        ],
        "api_url": args.api_url,
        "max_per_class": args.max_per_class,
        "runtime_seconds": round(
            time.time() - started,
            2,
        ),
        "rows": rows,
    }

    json_path.write_text(
        json.dumps(report, indent=2)
    )

    fieldnames = [
        "video",
        "expected",
        "production",
        "production_confidence",
        "v8",
        "v8_confidence",
        "production_correct",
        "v8_correct",
        "agreement",
        "decision",
        "family",
        "reason",
        "error",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key in fieldnames
            }
            for row in rows
        )

    total = len(rows)

    production_correct = sum(
        bool(row["production_correct"])
        for row in rows
    )

    v8_correct = sum(
        bool(row["v8_correct"])
        for row in rows
    )

    agreements = sum(
        bool(row["agreement"])
        for row in rows
    )

    errors = sum(
        bool(row["error"])
        for row in rows
    )

    print()
    print("=" * 72)
    print("ROUTER V8 BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Videos tested:       {total}")
    print(
        f"Production correct:  "
        f"{production_correct}/{total}"
    )
    print(
        f"Router V8 correct:   "
        f"{v8_correct}/{total}"
    )
    print(
        f"Router agreement:    "
        f"{agreements}/{total}"
    )
    print(f"Errors:              {errors}")
    print(f"JSON report:         {json_path}")
    print(f"CSV report:          {csv_path}")

    production_errors = Counter(
        (row["expected"], row["production"])
        for row in rows
        if not row["production_correct"]
        and not row["error"]
    )

    v8_errors = Counter(
        (row["expected"], row["v8"])
        for row in rows
        if not row["v8_correct"]
        and not row["error"]
    )

    if production_errors:
        print()
        print("Production confusion pairs:")

        for (
            expected,
            predicted,
        ), count in production_errors.most_common():
            print(
                f"  {expected:<22} "
                f"-> {str(predicted):<22} {count}"
            )

    if v8_errors:
        print()
        print("Router V8 confusion pairs:")

        for (
            expected,
            predicted,
        ), count in v8_errors.most_common():
            print(
                f"  {expected:<22} "
                f"-> {str(predicted):<22} {count}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
