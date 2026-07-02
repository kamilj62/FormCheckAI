import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.feature_engine.feature_names_v2 import FEATURE_NAMES


def train(dataset, model_out, report_out):
    df = pd.read_csv(dataset)

    X = df[FEATURE_NAMES].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds, labels=clf.classes_))
    print("Classes:", list(clf.classes_))

    model_out = Path(model_out)
    report_out = Path(report_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_out)

    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(report_out, index=False)

    print()
    print("Saved model:", model_out)
    print("Saved feature importance:", report_out)
    print()
    print("Top 20 features:")
    print(importance.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    train(args.dataset, args.model, args.report)


if __name__ == "__main__":
    main()
