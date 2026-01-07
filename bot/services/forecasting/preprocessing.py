import numpy as np
import pandas as pd

def train_test_split_series(y: pd.Series, test_size: int):
    if len(y) <= test_size + 10:
        raise ValueError()
    y_train = y.iloc[:-test_size].copy()
    y_test = y.iloc[-test_size:].copy()
    return y_train, y_test

def make_lag_features(y: pd.Series, lags: int = 14):
    df = pd.DataFrame({"y": y})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()
    return df

def make_supervised_windows(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X, dtype=float).reshape(-1, lookback, 1)
    y = np.array(y, dtype=float)
    return X, y
