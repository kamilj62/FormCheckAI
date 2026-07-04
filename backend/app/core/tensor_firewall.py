import numpy as np


class TensorFirewallError(Exception):
    pass

class TensorFirewall:

    LSTM_FEATURES = 68
    OLY_FEATURES = 54
    RF_FEATURES = 54

    @staticmethod
    def check_lstm(x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2:
            raise TensorFirewallError(f"LSTM expects 2D tensor, got {x.ndim}D")

        if x.shape[1] != TensorFirewall.LSTM_FEATURES:
            raise TensorFirewallError(
                f"LSTM FEATURE VIOLATION: expected {TensorFirewall.LSTM_FEATURES}, got {x.shape[1]}"
            )

        return x

    @staticmethod
    def check_oly(x):
        x = np.asarray(x, dtype=np.float32)

        if x.size != TensorFirewall.OLY_FEATURES:
            raise TensorFirewallError(
                f"OLY FEATURE VIOLATION: expected {TensorFirewall.OLY_FEATURES}, got {x.size}"
            )

        return x

    @staticmethod
    def check_rf(x):
        x = np.asarray(x, dtype=np.float32)

        if x.size != TensorFirewall.RF_FEATURES:
            raise TensorFirewallError(
                f"RF FEATURE VIOLATION: expected {TensorFirewall.RF_FEATURES}, got {x.size}"
            )

        return x
    @staticmethod

    def safe_rf(x):
        x = np.asarray(x, dtype=np.float32)

        if x.size > 54:
            return x[:54]

        if x.size < 54:
            pad = np.zeros(54 - x.size, dtype=np.float32)
            return np.concatenate([x, pad])

        return x