import os
import numpy as np
import tensorflow as tf

BASE_DIR = "/Users/josephkamil/Desktop/Capstone"
MODEL_PATH = os.path.join(BASE_DIR, "models", "movement_router_v2.keras")
DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset_v2",
    "processed_router",
)

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# class names (must match training order)
class_names = sorted(
    [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]
)

print("\nClasses:")
for i, c in enumerate(class_names):
    print(f"{i}: {c}")

print("\nTesting...\n")


def test_class(class_name, n=5):
    class_dir = os.path.join(DATA_DIR, class_name)

    files = [
        f for f in os.listdir(class_dir)
        if f.endswith(".npy")
    ]

    if not files:
        print(f"No files found for {class_name}")
        return

    files = files[:n]

    correct = 0

    for file in files:
        path = os.path.join(class_dir, file)

        x = np.load(path)

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)[0]
        pred_idx = np.argmax(pred)
        pred_name = class_names[pred_idx]
        confidence = pred[pred_idx]

        ok = pred_name == class_name
        if ok:
            correct += 1

        mark = "✓" if ok else "✗"

        print(
            f"{mark} true={class_name:15s} "
            f"pred={pred_name:15s} "
            f"conf={confidence:.3f}"
        )

    print(
        f"\n{class_name}: {correct}/{len(files)} correct\n"
    )


test_class("squat_front")
test_class("strict_press")