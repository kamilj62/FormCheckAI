import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path("ml/analysis_quality/fitness_aqa_squat")


def load_split(split):
    path = BASE / f"knee_v2_{split}.jsonl"

    features = []
    labels = []
    video_ids = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)

            forward = int(row["labels"]["knees_forward"])
            inward = int(row["labels"]["knees_inward"])

            # Keep only the two states involved in the main confusion.
            if forward == 1 and inward == 0:
                label = 0  # forward_only
            elif forward == 0 and inward == 1:
                label = 1  # inward_only
            else:
                continue

            vector = np.asarray(row["features"], dtype=np.float32)

            if not np.all(np.isfinite(vector)):
                continue

            features.append(vector)
            labels.append(label)
            video_ids.append(str(row["video_id"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(video_ids),
    )


def standardized_difference(forward_values, inward_values):
    forward_mean = float(np.mean(forward_values))
    inward_mean = float(np.mean(inward_values))

    pooled_variance = (
        float(np.var(forward_values, ddof=1))
        + float(np.var(inward_values, ddof=1))
    ) / 2.0

    pooled_std = max(pooled_variance ** 0.5, 1e-8)

    return (inward_mean - forward_mean) / pooled_std


def main():
    metadata = json.loads(
        (BASE / "knee_v2_train_metadata.json").read_text()
    )
    feature_names = metadata["feature_names"]

    X_train, y_train, _ = load_split("train")
    X_validation, y_validation, _ = load_split("validation")
    X_test, y_test, _ = load_split("test")

    print("Class encoding:")
    print("0 = forward_only")
    print("1 = inward_only")

    print("\nTrain:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    print(
        "Train class counts:",
        {
            "forward_only": int((y_train == 0).sum()),
            "inward_only": int((y_train == 1).sum()),
        },
    )

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nHeld-out test confusion matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nHeld-out test classification report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["forward_only", "inward_only"],
            digits=4,
            zero_division=0,
        )
    )

    importance = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="balanced_accuracy",
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )

    rows = []

    forward_train = X_train[y_train == 0]
    inward_train = X_train[y_train == 1]

    for index, name in enumerate(feature_names):
        effect = standardized_difference(
            forward_train[:, index],
            inward_train[:, index],
        )

        rows.append({
            "feature": name,
            "permutation_importance": float(
                importance.importances_mean[index]
            ),
            "importance_std": float(
                importance.importances_std[index]
            ),
            "standardized_difference": float(effect),
            "forward_mean": float(
                np.mean(forward_train[:, index])
            ),
            "inward_mean": float(
                np.mean(inward_train[:, index])
            ),
        })

    rows.sort(
        key=lambda row: (
            row["permutation_importance"],
            abs(row["standardized_difference"]),
        ),
        reverse=True,
    )

    print("\nTop features by validation permutation importance")

    for row in rows[:30]:
        print(
            f"{row['feature']:40s} "
            f"importance={row['permutation_importance']:8.4f} "
            f"effect={row['standardized_difference']:8.3f} "
            f"forward_mean={row['forward_mean']:10.4f} "
            f"inward_mean={row['inward_mean']:10.4f}"
        )

    output = BASE / "v2_feature_separation_report.json"
    output.write_text(json.dumps(rows, indent=2))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
