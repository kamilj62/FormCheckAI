import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path("/Users/josephkamil/Desktop/Capstone")
CSV_PATH = BASE_DIR / "Oly_Data/oly_keypoints.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEQUENCE_LEN = 30


def load_sequences():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    feature_cols = [
        c for c in df.columns
        if c.startswith(("x_", "y_", "z_", "v_"))
    ]

    X = []
    y = []

    grouped = df.groupby(["video", "label"])

    for (video, label), group in grouped:
        group = group.sort_values("frame")

        feats = group[feature_cols].values.astype("float32")

        if len(feats) < SEQUENCE_LEN:
            continue

        for i in range(0, len(feats) - SEQUENCE_LEN + 1, 5):
            seq = feats[i:i + SEQUENCE_LEN]
            X.append(seq)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Sequences:", X.shape)
    return X, y


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Masking(mask_value=0.0),

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(64),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    X, y = load_sequences()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Classes:", list(le.classes_))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    model = build_model(
        input_shape=X.shape[1:],
        num_classes=len(le.classes_),
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
    )

    model.save(MODEL_DIR / "oly_router.keras")

    np.save(MODEL_DIR / "oly_router_labels.npy", le.classes_)

    print("\nDONE")
    print("Saved model:", MODEL_DIR / "oly_router.keras")


if __name__ == "__main__":
    main()