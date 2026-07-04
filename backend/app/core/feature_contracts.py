import numpy as np
from dataclasses import dataclass

# ---------------------------
# CONTRACT REGISTRY
# ---------------------------

MODEL_CONTRACTS = {
    "movement_lstm": 68,
    "rf_oly_router_v2": 54,
    "rf_generic_video": 80,
}


# ---------------------------
# EXCEPTION
# ---------------------------

class FeatureContractError(Exception):
    pass


# ---------------------------
# DEBUG RESULT
# ---------------------------

@dataclass
class ContractResult:
    X: np.ndarray
    original_shape: tuple
    final_shape: tuple
    model_name: str
    action: str


# ---------------------------
# CONTRACT ENFORCER
# ---------------------------

def enforce_model_contract(X: np.ndarray, model_name: str, debug=False):
    """
    Ensures model input shape safety.
    """

    if X is None:
        raise FeatureContractError(f"[{model_name}] Input is None")

    X = np.asarray(X, dtype=np.float32)

    if model_name not in MODEL_CONTRACTS:
        raise FeatureContractError(f"Unknown model: {model_name}")

    expected = MODEL_CONTRACTS[model_name]
    original_shape = X.shape

    # EXACT
    if X.shape[-1] == expected:
        result = X
        action = "pass"

    # TRIM
    elif X.shape[-1] > expected:
        result = X[..., :expected]
        action = "trim"

    # PAD
    else:
        pad = expected - X.shape[-1]
        padding = np.zeros((*X.shape[:-1], pad), dtype=np.float32)
        result = np.concatenate([X, padding], axis=-1)
        action = "pad"

    if debug:
        return ContractResult(
            X=result,
            original_shape=original_shape,
            final_shape=result.shape,
            model_name=model_name,
            action=action,
        )

    return result