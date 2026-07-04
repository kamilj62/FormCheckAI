import numpy as np

class FeatureRegistry:

    LSTM_DIM = 68
    OLY_DIM = 54

    @staticmethod
    def validate_lstm(x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[1] != FeatureRegistry.LSTM_DIM:
            raise ValueError("LSTM feature contract violated")
        return x

    @staticmethod
    def validate_oly(x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != FeatureRegistry.OLY_DIM:
            if x.shape[0] > FeatureRegistry.OLY_DIM:
                x = x[:54]
            else:
                pad = np.zeros(54 - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad])
        return x