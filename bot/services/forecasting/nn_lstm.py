import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from .preprocessing import make_supervised_windows

class LSTMModel:
    def __init__(self, lookback: int = 30, epochs: int = 50, batch_size: int = 32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self._train_scaled = None

    def _build(self):
        m = Sequential([
            LSTM(32, input_shape=(self.lookback, 1)),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def fit(self, y_train: pd.Series):
        y = y_train.astype(float).values.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y).reshape(-1)
        self._train_scaled = y_scaled.copy()

        X, y_sup = make_supervised_windows(y_scaled, lookback=self.lookback)

        self.model = self._build()
        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

        self.model.fit(
            X, y_sup,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es],
            shuffle=False,
        )
        return self

    def predict(self, y_context: pd.Series, horizon: int):
        if self.model is None:
            raise RuntimeError()

        context = y_context.astype(float).values.reshape(-1, 1)
        context_scaled = self.scaler.transform(context).reshape(-1)

        window = context_scaled[-self.lookback:].copy()
        preds_scaled = []

        for _ in range(horizon):
            X = window.reshape(1, self.lookback, 1)
            yhat = float(self.model.predict(X, verbose=0)[0, 0])
            preds_scaled.append(yhat)
            window = np.append(window[1:], yhat)

        preds_scaled = np.array(preds_scaled, dtype=float).reshape(-1, 1)
        preds = self.scaler.inverse_transform(preds_scaled).reshape(-1)
        return preds