from pathlib import Path
import json

import pandas as pd

PREDICTIONS = Path(
    "ml/router_rebuild_v1/reports/"
    "olympic_router_stage2_v2_predictions.csv"
)

OUT_CSV = Path(
    "ml/router_rebuild_v1/reports/"
    "olympic_router_stage2_v2_confidence_policy.csv"
)

OUT_JSON = Path(
    "ml/router_rebuild_v1/reports/"
    "olympic_router_stage2_v2_selected_policy.json"
)

df = pd.read_csv(PREDICTIONS)

prob_cols = [
    column for column in df.columns
    if column.startswith("prob_")
]

df["second_probability"] = (
    df[prob_cols]
    .apply(
        lambda row: sorted(row, reverse=True)[1],
        axis=1,
    )
)

df["margin"] = (
    df["confidence"] - df["second_probability"]
)

dev = df[df["split"] == "dev"].copy()
test = df[df["split"] == "test"].copy()

rows = []

for confidence_int in range(30, 81, 5):
    confidence_threshold = confidence_int / 100

    for margin_int in range(0, 41, 5):
        margin_threshold = margin_int / 100

        accepted = dev[
            (dev["confidence"] >= confidence_threshold)
            & (dev["margin"] >= margin_threshold)
        ]

        if len(accepted) == 0:
            continue

        rows.append({
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "accepted": int(len(accepted)),
            "coverage": float(len(accepted) / len(dev)),
            "accepted_accuracy": float(
                accepted["correct"].mean()
            ),
            "errors": int((~accepted["correct"]).sum()),
        })

results = pd.DataFrame(rows)

eligible = results[
    (results["accepted_accuracy"] >= 0.90)
    & (results["coverage"] >= 0.50)
].copy()

if eligible.empty:
    print(
        "No dev policy reached both 90% accepted accuracy "
        "and 50% coverage."
    )

    selected = (
        results.sort_values(
            [
                "accepted_accuracy",
                "coverage",
                "confidence_threshold",
                "margin_threshold",
            ],
            ascending=[False, False, True, True],
        )
        .iloc[0]
    )
else:
    selected = (
        eligible.sort_values(
            [
                "coverage",
                "accepted_accuracy",
                "confidence_threshold",
                "margin_threshold",
            ],
            ascending=[False, False, True, True],
        )
        .iloc[0]
    )

confidence_threshold = float(
    selected["confidence_threshold"]
)
margin_threshold = float(
    selected["margin_threshold"]
)

def evaluate(frame, split_name):
    accepted = frame[
        (frame["confidence"] >= confidence_threshold)
        & (frame["margin"] >= margin_threshold)
    ].copy()

    deferred = frame.drop(index=accepted.index)

    print("\n" + "=" * 80)
    print(split_name.upper())
    print("=" * 80)
    print("rows:", len(frame))
    print("accepted:", len(accepted))
    print("deferred:", len(deferred))
    print("coverage:", round(len(accepted) / len(frame), 4))
    print(
        "accepted accuracy:",
        round(float(accepted["correct"].mean()), 4)
        if len(accepted) else None,
    )

    if len(accepted):
        mistakes = accepted[~accepted["correct"]]

        print("accepted mistakes:", len(mistakes))

        if len(mistakes):
            print(
                mistakes[
                    [
                        "reviewed_label",
                        "predicted_label",
                        "confidence",
                        "margin",
                        "filename",
                    ]
                ]
                .to_string(index=False)
            )

    return {
        "rows": int(len(frame)),
        "accepted": int(len(accepted)),
        "deferred": int(len(deferred)),
        "coverage": float(len(accepted) / len(frame)),
        "accepted_accuracy": (
            float(accepted["correct"].mean())
            if len(accepted)
            else None
        ),
    }

results.to_csv(OUT_CSV, index=False)

print("Selected from dev:")
print(selected.to_string())

dev_result = evaluate(dev, "dev")
test_result = evaluate(test, "test")

payload = {
    "selection_split": "dev",
    "confidence_threshold": confidence_threshold,
    "margin_threshold": margin_threshold,
    "dev": dev_result,
    "test": test_result,
}

OUT_JSON.write_text(json.dumps(payload, indent=2))

print("\nCreated:", OUT_CSV)
print("Created:", OUT_JSON)
