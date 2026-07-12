#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import requests


BACKEND = Path(__file__).resolve().parents[1]
VIDEO_DIR = BACKEND / "regression_tests" / "videos"
EXPECTED_PATH = BACKEND / "regression_tests" / "expected_results.json"
OUTPUT_PATH = BACKEND / "agents" / "reports" / "router_v8_snapshot_golden.json"
API_URL = "http://localhost:8000/analyze"


def normalize_label(value):
    return str(value or "").strip().lower().replace(" ", "_")


def main():
    expected = json.loads(EXPECTED_PATH.read_text())
    rows = []

    print("=" * 72)
    print("ROUTER V8 GOLDEN SNAPSHOT")
    print("=" * 72)
    print(f"Fixtures: {len(expected)}")
    print(f"Output:   {OUTPUT_PATH.relative_to(BACKEND)}")
    print()

    for index, (filename, checks) in enumerate(expected.items(), start=1):
        video_path = VIDEO_DIR / filename
        expected_label = normalize_label(checks.get("label"))

        print(
            f"[{index:>2}/{len(expected)}] "
            f"{expected_label:<22} {filename}"
        )

        row = {
            "video": str(video_path),
            "expected": expected_label,
            "production": None,
            "previous_v8": None,
            "snapshot": {
                "predictions": [],
                "state": {},
            },
            "error": None,
        }

        if not video_path.exists():
            row["error"] = f"Missing video: {video_path}"
            rows.append(row)
            print(f"    ERROR: {row['error']}")
            continue

        try:
            with video_path.open("rb") as handle:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            video_path.name,
                            handle,
                            "application/octet-stream",
                        )
                    },
                    timeout=300,
                )

            if response.status_code != 200:
                row["error"] = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                rows.append(row)
                print(f"    ERROR: {row['error']}")
                continue

            data = response.json()
            router_v8 = (data.get("debug") or {}).get("router_v8") or {}

            row["production"] = normalize_label(
                data.get("exercise_label")
            )
            row["previous_v8"] = normalize_label(
                router_v8.get("final_label")
                or router_v8.get("label")
                or router_v8.get("prediction")
            ) or None
            row["snapshot"] = {
                "predictions": router_v8.get("predictions") or [],
                "state": router_v8.get("state") or {},
            }

            if not row["snapshot"]["predictions"]:
                row["error"] = "Router V8 predictions missing"
            elif not row["snapshot"]["state"]:
                row["error"] = "Router V8 state missing"

            print(
                f"    production={row['production']} "
                f"shadow={row['previous_v8']} "
                f"predictions={len(row['snapshot']['predictions'])} "
                f"state={'yes' if row['snapshot']['state'] else 'no'}"
            )

        except Exception as exc:
            row["error"] = str(exc)
            print(f"    ERROR: {exc}")

        rows.append(row)

    valid = sum(
        1
        for row in rows
        if not row["error"]
        and row["snapshot"]["predictions"]
        and row["snapshot"]["state"]
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "regression_tests/expected_results.json",
        "total": len(rows),
        "valid": valid,
        "errors": len(rows) - valid,
        "rows": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print()
    print("=" * 72)
    print(f"Valid snapshots: {valid}/{len(rows)}")
    print(f"Errors:          {len(rows) - valid}")
    print(f"Saved:           {OUTPUT_PATH.relative_to(BACKEND)}")

    return 0 if valid == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
