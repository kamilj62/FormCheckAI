import numpy as np

LSTM_DIM = 68
OLY_DIM = 54


class InferenceContractViolation(Exception):
    pass


def enforce_lstm_contract(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise InferenceContractViolation(f"LSTM expects (T,F), got {x.shape}")

    if x.shape[1] != LSTM_DIM:
        raise InferenceContractViolation(
            f"LSTM feature mismatch: expected {LSTM_DIM}, got {x.shape[1]}"
        )

    return x


def enforce_oly_contract(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.shape[0] != OLY_DIM:
        if x.shape[0] > OLY_DIM:
            x = x[:OLY_DIM]
        else:
            pad = np.zeros(OLY_DIM - x.shape[0], dtype=np.float32)
            x = np.concatenate([x, pad])

    return x