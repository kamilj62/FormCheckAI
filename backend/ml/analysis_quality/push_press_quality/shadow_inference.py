from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


MODELS_DIR = Path(
    "ml/analysis_quality/push_press_quality/results/models"
)

MODEL_FILES = {
    "elbow_error": "elbow_error_candidate.joblib",
    "knee_error": "knee_error_candidate.joblib",
}


@lru_cache(maxsize=2)
def load_candidate(target):
    if target not in MODEL_FILES:
        raise ValueError(f"Unsupported target: {target}")

    path = MODELS_DIR / MODEL_FILES[target]

    if not path.exists():
        raise FileNotFoundError(f"Candidate model not found: {path}")

    package = joblib.load(path)

    required_keys = {
        "target",
        "model",
        "feature_columns",
        "threshold",
        "model_version",
    }

    missing = required_keys - set(package)

    if missing:
        raise ValueError(
            f"{path.name} missing package keys: {sorted(missing)}"
        )

    return package


def score_feature_row(feature_row, target):
    package = load_candidate(target)

    feature_columns = package["feature_columns"]

    row = {
        column: feature_row.get(column)
        for column in feature_columns
    }

    model_input = pd.DataFrame(
        [row],
        columns=feature_columns,
    )

    probability = float(
        package["model"].predict_proba(model_input)[0, 1]
    )

    threshold = float(package["threshold"])
    detected = probability >= threshold

    return {
        "target": target,
        "model_version": package["model_version"],
        "probability": probability,
        "threshold": threshold,
        "detected": bool(detected),
        "feature_count": len(feature_columns),
    }


def score_push_press_features(feature_row):
    return {
        target: score_feature_row(feature_row, target)
        for target in MODEL_FILES
    }
