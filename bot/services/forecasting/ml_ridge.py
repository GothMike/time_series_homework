import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocessing import make_lag_features

class RidgeLagModel:
    def __init__(self, lags: int = 14, alpha: float = 1.0):
        self.lags = lags
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=42)),
        ])
        self._last_window = None 

    def fit(self, y_train: pd.Series):
        df = make_lag_features(y_train, lags=self.lags)
        X = df.drop(columns=["y"]).values
        y = df["y"].values
        self.model.fit(X, y)

        self._last_window = y_train.iloc[-self.lags:].astype(float).values
        return self

    def predict(self, y_context: pd.Series, horizon: int):
        window = y_context.iloc[-self.lags:].astype(float).values.copy()
        preds = []
        for _ in range(horizon):
            X = window[::-1]  
            X = window[-1::-1]
            X = X.reshape(1, -1)
            yhat = float(self.model.predict(X)[0])
            preds.append(yhat)
            window = np.append(window[1:], yhat)
        return np.array(preds, dtype=float)
