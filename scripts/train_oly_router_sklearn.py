import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/Users/josephkamil/Desktop/Capstone")
CSV_PATH = BASE_DIR / "Oly_Data/oly_keypoints.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_MODEL = MODEL_DIR / "oly_router_rf.joblib"


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_", "v_"))]

    video_rows = []

    for (video, label), group in df.groupby(["video", "label"]):
        feats = group[feature_cols].values.astype("float32")

        mean_feats = feats.mean(axis=0)
        std_feats = feats.std(axis=0)
        min_feats = feats.min(axis=0)
        max_feats = feats.max(axis=0)

        row = np.concatenate([mean_feats, std_feats, min_feats, max_feats])
        video_rows.append((video, label, row))

    X = np.array([r[2] for r in video_rows])
    y = np.array([r[1] for r in video_rows])

    print("Videos:", len(y))
    print("X shape:", X.shape)
    print(pd.Series(y).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, preds))

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "classes": model.classes_,
        },
        OUTPUT_MODEL,
    )

    print("\nSaved:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()