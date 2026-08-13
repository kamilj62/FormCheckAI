#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOTS = [
    Path("/Users/josephkamil/Desktop/Capstone/FormCheck_Phase_Audit"),
    Path("/Users/josephkamil/Desktop/Capstone/FormCheck_Phase_Audit_v2"),
]
OUT = Path("backend/ml/benchmark/results/saved_phase_audit_summary.csv")


ALIASES = {
    "back_squat": {"backsquat", "back_squat", "squat_back"},
    "bench_press": {"bench"},
    "burpee": {"burpee"},
    "clean": {"clean-correct", "clean_mov"},
    "clean_and_jerk": {"cleanandjerk", "clean_and_jerk"},
    "deadlift": {"deadlift"},
    "front_squat": {"frontsquat", "front_squat"},
    "handstand_push_up": {"handstand", "handstand_push_up"},
    "overhead_squat": {"overheadsquat", "overhead_squat"},
    "pull_up": {"pullup", "pull_up", "strictpullup"},
    "push_press": {"pushpress", "push_press"},
    "push_up": {"push_up"},
    "ring_muscle_up": {"ring_muscle_up"},
    "snatch": {"snatch"},
    "split_jerk": {"splitjerk", "split_jerk"},
    "strict_press": {"strict_press"},
    "thruster": {"thruster"},
}


def compact(value: str) -> str:
    return (
        value.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
    )


def infer_expected(folder: Path) -> str | None:
    name = compact(folder.name)
    for label, aliases in ALIASES.items():
        if any(compact(alias) in name for alias in aliases):
            return label
    return None


def load_json(folder: Path) -> dict:
    for name in ("analyze.json", "response.json"):
        path = folder / name
        if path.exists():
            try:
                print(f"Reading {path}", flush=True)
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                return {"error": f"json_error:{exc}"}
    return {}


def main() -> None:
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            expected = infer_expected(folder)
            data = load_json(folder)
            if not expected or not data:
                continue
            debug = data.get("debug") or {}
            rows.append({
                "expected_label": expected,
                "audit_folder": str(folder),
                "predicted_label": data.get("exercise_label"),
                "label_ok": data.get("exercise_label") == expected,
                "confidence": data.get("confidence"),
                "analysis_mode": data.get("analysis_mode"),
                "rep_count": len(data.get("rep_feedback") or []),
                "protected_reason": debug.get("protected_reason"),
                "raw_label": debug.get("raw_label"),
                "bio_label": debug.get("bio_label"),
                "squat_label": debug.get("squat_label"),
                "olympic_label": debug.get("olympic_pred"),
                "router_v8_winner": (
                    (debug.get("router_v8") or {}).get("winner")
                    if isinstance(debug.get("router_v8"), dict)
                    else None
                ),
                "error": data.get("error") or debug.get("error"),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} saved audit rows to {OUT}")
    wrong = [r for r in rows if not r["label_ok"]]
    print(f"Wrong labels: {len(wrong)} / {len(rows)}")
    for r in wrong:
        print(
            f"{r['expected_label']} -> {r['predicted_label']} "
            f"mode={r['analysis_mode']} folder={Path(r['audit_folder']).name}"
        )


if __name__ == "__main__":
    main()
