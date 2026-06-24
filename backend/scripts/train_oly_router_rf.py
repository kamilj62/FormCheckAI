from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CLEAN_CSV = Path("/Users/josephkamil/Desktop/Capstone/Oly_Data/clean_and_jerk_keypoints.csv")
SNATCH_CSV = Path("/Users/josephkamil/Desktop/Capstone/Oly_Data/snatch_keypoints.csv")
OUT = Path("app/models/oly_router_rf.joblib")

clean = pd.read_csv(CLEAN_CSV)
snatch = pd.read_csv(SNATCH_CSV)

df = pd.concat([clean, snatch], ignore_index=True)
df = df[df["label"].isin(["clean_and_jerk", "snatch"])].copy()

feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_", "v_"))]
X = df[feature_cols].fillna(0)
y = df["label"]

print("label counts:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
print(confusion_matrix(y_test, pred, labels=["clean_and_jerk", "snatch"]))
print(classification_report(y_test, pred))

joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,
        "classes": model.classes_,
    },
    OUT,
)

print("saved", OUT)
print("classes:", model.classes_)
