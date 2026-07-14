#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

from app.ml.router_v8.models import RouterPrediction
from app.ml.router_v8.state import RouterState


def load_fusion(module_name: str):
    module = importlib.import_module(module_name)

    if not hasattr(module, "fuse_predictions"):
        raise RuntimeError(
            f"{module_name} does not define fuse_predictions"
        )

    return module.fuse_predictions


def load_predictions(items: list[dict]) -> list[RouterPrediction]:
    return [
        RouterPrediction(
            router=item.get("router", ""),
            label=item.get("label"),
            confidence=float(item.get("confidence") or 0.0),
            reason=item.get("reason") or "",
            lock=bool(item.get("lock", False)),
            metadata=item.get("metadata") or {},
        )
        for item in items
    ]


def load_state(data: dict) -> RouterState:
    allowed = set(RouterState.__dataclass_fields__)

    filtered = {
        key: value
        for key, value in data.items()
        if key in allowed
    }

    return RouterState(**filtered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay Router V8 fusion from saved snapshots."
    )

    parser.add_argument(
        "snapshot_file",
        type=Path,
    )

    parser.add_argument(
        "--fusion-module",
        default="app.ml.router_v8.fusion",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "agents/reports/router_v8_replay.json"
        ),
    )

    args = parser.parse_args()

    payload = json.loads(
        args.snapshot_file.read_text()
    )

    rows = payload.get("rows", payload)
    fuse_predictions = load_fusion(
        args.fusion_module
    )

    replay_rows = []

    for row in rows:
        snapshot = row.get("snapshot") or {}
        predictions_data = (
            snapshot.get("predictions") or []
        )
        state_data = snapshot.get("state") or {}

        if not predictions_data or not state_data:
            replay_rows.append({
                "video": row.get("video"),
                "expected": row.get("expected"),
                "production": row.get("production"),
                "previous_v8": row.get("v8"),
                "replay_v8": None,
                "replay_correct": False,
                "decision": None,
                "winning_family": None,
                "family_scores": {},
                "scores": {},
                "replay_error": (
                    "missing predictions or state"
                ),
            })
            continue

        predictions = load_predictions(
            predictions_data
        )
        state = load_state(state_data)

        try:
            result = fuse_predictions(
                predictions,
                state=state,
            )

            winner = result.get("label")
            error = None

        except Exception as exc:
            winner = None
            result = {}
            error = str(exc)

        expected = row.get("expected")

        replay_rows.append({
            "video": row.get("video"),
            "expected": expected,
            "production": row.get("production"),
            "previous_v8": row.get("v8"),
            "replay_v8": winner,
            "replay_correct": winner == expected,
            "decision": result.get("decision"),
            "winning_family": result.get(
                "winning_family"
            ),
            "family_scores": result.get(
                "family_scores",
                {},
            ),
            "scores": result.get(
                "scores",
                {},
            ),
            "replay_error": error,
        })

    valid = [
        row
        for row in replay_rows
        if not row.get("replay_error")
    ]

    correct = sum(
        bool(row["replay_correct"])
        for row in valid
    )

    confusions = Counter(
        (
            row["expected"],
            row["replay_v8"],
        )
        for row in valid
        if not row["replay_correct"]
    )

    report = {
        "fusion_module": args.fusion_module,
        "total": len(replay_rows),
        "valid": len(valid),
        "correct": correct,
        "accuracy": (
            round(correct / len(valid), 4)
            if valid
            else 0.0
        ),
        "confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (
                expected,
                predicted,
            ), count in confusions.most_common()
        ],
        "rows": replay_rows,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2)
    )

    print("=" * 70)
    print("ROUTER V8 OFFLINE REPLAY")
    print("=" * 70)
    print(f"Fusion module: {args.fusion_module}")
    print(f"Total rows:    {len(replay_rows)}")
    print(f"Valid rows:    {len(valid)}")
    print(f"Correct:       {correct}/{len(valid)}")
    print(
        f"Accuracy:      "
        f"{report['accuracy']:.1%}"
    )
    print(f"Report:        {args.output}")

    if confusions:
        print()
        print("Top confusion pairs:")

        for (
            expected,
            predicted,
        ), count in confusions.most_common(15):
            print(
                f"  {str(expected):<22} "
                f"-> {str(predicted):<22} "
                f"{count}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
