#!/usr/bin/env python3

import json
from pathlib import Path

RESULTS_DIR = Path("regression_tests/results")
OUTPUT_JSON = RESULTS_DIR / "router_v8_shadow_report.json"

rows = []

for path in sorted(RESULTS_DIR.glob("*.analyze.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        rows.append({
            "file": path.name,
            "error": f"Could not read JSON: {exc}",
        })
        continue

    debug = data.get("debug") or {}
    v8 = debug.get("router_v8") or {}

    production_label = data.get("exercise_label")
    v8_winner = v8.get("winner")
    winner_confidence = v8.get("winner_confidence")
    scores = v8.get("scores") or {}
    locks = v8.get("locks") or []

    ranked = sorted(
        (
            (str(label), float(score))
            for label, score in scores.items()
            if score is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    top_score = ranked[0][1] if ranked else None
    second_score = ranked[1][1] if len(ranked) > 1 else None
    margin = (
        round(top_score - second_score, 3)
        if top_score is not None and second_score is not None
        else None
    )

    row = {
        "video": path.name.removesuffix(".analyze.json"),
        "production_label": production_label,
        "production_confidence": data.get("confidence"),
        "production_mode": data.get("analysis_mode"),
        "v8_winner": v8_winner,
        "v8_winner_confidence": winner_confidence,
        "v8_top_score": round(top_score, 3) if top_score is not None else None,
        "v8_margin": margin,
        "v8_locks": locks,
        "match": production_label == v8_winner,
    }

    if not v8:
        row["error"] = "router_v8 debug data missing"

    rows.append(row)

valid_rows = [row for row in rows if "error" not in row]
matches = sum(bool(row.get("match")) for row in valid_rows)
mismatches = len(valid_rows) - matches

report = {
    "summary": {
        "videos": len(rows),
        "valid_v8_results": len(valid_rows),
        "matches": matches,
        "mismatches": mismatches,
        "match_rate": (
            round(matches / len(valid_rows), 4)
            if valid_rows
            else 0.0
        ),
    },
    "results": rows,
}

OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

print("=" * 108)
print("ROUTER V8 SHADOW COMPARISON")
print("=" * 108)
print(
    f"{'VIDEO':27} "
    f"{'PRODUCTION':18} "
    f"{'V8 WINNER':18} "
    f"{'MARGIN':8} "
    f"{'LOCK':22} "
    f"RESULT"
)
print("-" * 108)

for row in rows:
    if "error" in row:
        print(f"{row['file'][:27]:27} ERROR: {row['error']}")
        continue

    locks = row.get("v8_locks") or []
    lock_text = ", ".join(
        str(lock.get("label", "unknown"))
        for lock in locks
        if isinstance(lock, dict)
    ) or "-"

    result = "PASS" if row["match"] else "MISMATCH"

    print(
        f"{row['video'][:27]:27} "
        f"{str(row['production_label'])[:18]:18} "
        f"{str(row['v8_winner'])[:18]:18} "
        f"{str(row['v8_margin']):8} "
        f"{lock_text[:22]:22} "
        f"{result}"
    )

print("-" * 108)
print(
    f"Matches: {matches}/{len(valid_rows)}"
    f"  |  Mismatches: {mismatches}"
    f"  |  Match rate: {report['summary']['match_rate']:.1%}"
)
print(f"Report: {OUTPUT_JSON.resolve()}")
