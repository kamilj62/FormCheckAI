import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]

FEATURE_CSV = ROOT / "ml/reports/bodyweight_router_features.csv"
MODEL_OUT = ROOT / "app/models/bodyweight_router.joblib"
LABELS_OUT = ROOT / "app/models/bodyweight_router_labels.joblib"

REPORT_DIR = ROOT / "ml/reports"
IMPORTANCE_OUT = REPORT_DIR / "bodyweight_feature_importance.csv"
CONFUSION_OUT = REPORT_DIR / "bodyweight_confusion_matrix.csv"
METRICS_OUT = REPORT_DIR / "bodyweight_router_metrics.json"

DROP_COLS = {"label", "video", "valid", "error"}


def main():
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Missing feature CSV: {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)

    df = df[df["valid"] == 1].copy()
    df = df.dropna(subset=["label"])

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X = df[feature_cols].fillna(0.0)
    y_text = df["label"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    report = classification_report(
        y_test,
        preds,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in encoder.classes_],
        columns=[f"pred_{c}" for c in encoder.classes_],
    )

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_OUT)
    joblib.dump(encoder, LABELS_OUT)

    importance.to_csv(IMPORTANCE_OUT, index=False)
    cm_df.to_csv(CONFUSION_OUT)

    metrics = {
        "accuracy": acc,
        "labels": encoder.classes_.tolist(),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "feature_count": len(feature_cols),
        "classification_report": report,
        "model_path": str(MODEL_OUT),
        "labels_path": str(LABELS_OUT),
        "feature_importance_path": str(IMPORTANCE_OUT),
        "confusion_matrix_path": str(CONFUSION_OUT),
    }

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    print("\nBodyweight Router Training Complete")
    print("=" * 45)
    print(f"Rows used: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report")
    print(classification_report(
        y_test,
        preds,
        target_names=encoder.classes_,
        zero_division=0,
    ))

    print("\nConfusion Matrix")
    print(cm_df)

    print("\nTop Features")
    print(importance.head(10).to_string(index=False))

    print("\nSaved:")
    print(MODEL_OUT)
    print(LABELS_OUT)
    print(IMPORTANCE_OUT)
    print(CONFUSION_OUT)
    print(METRICS_OUT)


if __name__ == "__main__":
    main()
