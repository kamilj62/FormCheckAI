from pathlib import Path

import numpy as np
import tensorflow as tf


LABELS = ["bench_press", "deadlift", "push_press", "squat", "squat_front"]

LABEL_DISPLAY = {
    "bench_press": "Bench Press",
    "deadlift": "Deadlift",
    "push_press": "Push Press",
    "squat": "Squat",
    "squat_front": "Front Squat",
}


class NumpyFormCheckModel:
    def __init__(self, model_dir):
        weights_path = Path(__file__).parent / "models" / "movement_classifier.weights.h5"

        print("\n====================")
        print("LOADING CLEAN WEIGHTS MODEL")
        print(weights_path)
        print("====================\n")

        self.model = self._build_model()
        self.model.load_weights(weights_path)

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(30, 68)),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    128,
                    return_sequences=True,
                    zero_output_for_mask=True,
                )
            ),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    64,
                    return_sequences=False,
                )
            ),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(5, activation="softmax"),
        ])

        return model

    def predict_proba(self, seq_142: np.ndarray) -> np.ndarray:
        seq_68 = seq_142[:, :68].astype(np.float32)
        x = np.expand_dims(seq_68, axis=0)

        preds = self.model.predict(x, verbose=0)[0]
        return preds.astype(np.float32)