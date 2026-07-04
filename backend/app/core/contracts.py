import numpy as np
from dataclasses import dataclass

from app.core.feature_contracts import enforce_model_contract

def predict(seq):
    seq = enforce_model_contract(seq, "movement_lstm")
    return self.model.predict_proba(seq)

# ---------------------------
# MODEL CONTRACT REGISTRY
# ---------------------------

MODEL_CONTRACTS = {
    "movement_lstm": {
        "features": 68,
        "strict": True,
    },
    "rf_oly_router_v2": {
        "features": 54,
        "strict": False,
    },
    "rf_generic_video": {
        "features": 80,
        "strict": False,
    },
}

# ---------------------------
# EXCEPTIONS
# ---------------------------

class FeatureContractError(Exception):
    pass


# ---------------------------
# DEBUG WRAPPER
# ---------------------------

@dataclass
class ContractResult:
    X: np.ndarray
    original_shape: tuple
    final_shape: tuple
    model_name: str
    action: str  # "pass | trim | pad | reject"


# ---------------------------
# CORE CONTRACT ENFORCER
# ---------------------------

def enforce_model_contract(X: np.ndarray, model_name: str, debug=False):
    """
    HARD FEATURE CONTRACT SYSTEM

    Guarantees:
    - correct feature dimension per model
    - safe padding/trimming
    - optional strict rejection
    """

    if X is None:
        raise FeatureContractError(f"[{model_name}] Input is None")

    X = np.asarray(X, dtype=np.float32)

    if model_name not in MODEL_CONTRACTS:
        raise FeatureContractError(f"Unknown model: {model_name}")

    expected = MODEL_CONTRACTS[model_name]["features"]
    strict = MODEL_CONTRACTS[model_name]["strict"]

    original_shape = X.shape

    # ---------------------------
    # CASE 1: EXACT MATCH
    # ---------------------------
    if X.shape[-1] == expected:
        result = X
        action = "pass"

    # ---------------------------
    # CASE 2: TOO LARGE → TRIM
    # ---------------------------
    elif X.shape[-1] > expected:
        result = X[..., :expected]
        action = "trim"

    # ---------------------------
    # CASE 3: TOO SMALL → PAD
    # ---------------------------
    else:
        pad = expected - X.shape[-1]

        # pad with zeros (safe default for ML pipelines)
        padding = np.zeros((*X.shape[:-1], pad), dtype=np.float32)
        result = np.concatenate([X, padding], axis=-1)
        action = "pad"

    # ---------------------------
    # STRICT MODE CHECK
    # ---------------------------
    if strict and action != "pass":
        raise FeatureContractError(
            f"[{model_name}] strict contract violation: "
            f"expected {expected}, got {original_shape[-1]}"
        )

    # ---------------------------
    # DEBUG OUTPUT (OPTIONAL)
    # ---------------------------
    if debug:
        return ContractResult(
            X=result,
            original_shape=original_shape,
            final_shape=result.shape,
            model_name=model_name,
            action=action,
        )

    return result