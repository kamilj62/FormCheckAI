from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "models" / "bodyweight_router.joblib"
LABELS_PATH = ROOT / "app" / "models" / "bodyweight_router_labels.joblib"


def test_bodyweight_router_artifacts_exist_and_match_labels():
    assert MODEL_PATH.exists()
    assert LABELS_PATH.exists()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(LABELS_PATH)

    assert set(encoder.classes_) == {
        "handstand_push_up",
        "pull_up",
        "push_up",
    }
    assert getattr(model, "n_features_in_", None) == 16
