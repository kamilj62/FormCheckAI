#!/usr/bin/env python3

from pathlib import Path
import sys
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.feature_engine.feature_names_v2 import FEATURE_NAMES


DATASET = Path("ml/datasets/olympic_video_dataset_v2.csv")
PRODUCTION = Path("app/models/oly_router_rf.joblib")
CANDIDATE = Path("app/models/candidates/olympic_router_v2_candidate.joblib")
REPORT_DIR = Path("ml/reports/model_comparison")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(path)

    obj = joblib.load(path)

    # Production model is stored as:
    # {"model": RandomForestClassifier, "feature_cols": [...], "classes": [...]}
    # Candidate model may be stored directly as RandomForestClassifier.
    if isinstance(obj, dict):
        if "model" not in obj:
            raise ValueError(f"Dict model missing 'model' key: {path}")
        return obj["model"]

    return obj


def predict(model, X):
    return model.predict(X)


def main():
    df = pd.read_csv(DATASET)

    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing features: {missing[:10]}")

    X = df[FEATURE_NAMES].fillna(0)
    y = df["label"]

    prod = load_model(PRODUCTION)
    cand = load_model(CANDIDATE)

    prod_pred = predict(prod, X)
    cand_pred = predict(cand, X)

    prod_acc = accuracy_score(y, prod_pred)
    cand_acc = accuracy_score(y, cand_pred)

    out = df.copy()
    out["production_pred"] = prod_pred
    out["candidate_pred"] = cand_pred
    out["production_correct"] = out["production_pred"] == y
    out["candidate_correct"] = out["candidate_pred"] == y
    out["changed"] = out["production_pred"] != out["candidate_pred"]

    changed = out[out["changed"]]
    prod_errors = int((~out["production_correct"]).sum())
    cand_errors = int((~out["candidate_correct"]).sum())

    comparison_csv = REPORT_DIR / "olympic_model_comparison_latest.csv"
    changed_csv = REPORT_DIR / "olympic_model_changed_predictions.csv"

    out.to_csv(comparison_csv, index=False)
    changed.to_csv(changed_csv, index=False)

    labels = sorted(y.unique())

    print()
    print("=" * 70)
    print("OLYMPIC ROUTER MODEL COMPARISON")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print(f"Rows: {len(df)}")
    print()
    print(f"Production: {PRODUCTION}")
    print(f"Candidate : {CANDIDATE}")
    print()
    print(f"Production accuracy: {prod_acc:.3f}")
    print(f"Candidate accuracy : {cand_acc:.3f}")
    print()
    print(f"Production errors: {prod_errors}")
    print(f"Candidate errors : {cand_errors}")
    print(f"Changed predictions: {len(changed)}")
    print()

    print("=" * 70)
    print("PRODUCTION REPORT")
    print("=" * 70)
    print(classification_report(y, prod_pred))
    print(confusion_matrix(y, prod_pred, labels=labels))

    print()
    print("=" * 70)
    print("CANDIDATE REPORT")
    print("=" * 70)
    print(classification_report(y, cand_pred))
    print(confusion_matrix(y, cand_pred, labels=labels))

    print()
    print("=" * 70)
    if cand_errors < prod_errors:
        print("PROMOTE CANDIDATE")
    elif cand_errors == prod_errors and cand_acc >= prod_acc:
        print("CANDIDATE TIES PRODUCTION")
    else:
        print("KEEP PRODUCTION")
    print("=" * 70)

    print()
    print(f"Saved comparison: {comparison_csv}")
    print(f"Saved changed predictions: {changed_csv}")


if __name__ == "__main__":
    main()
