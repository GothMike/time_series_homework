import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self._fit = None

    def fit(self, y_train: pd.Series):
        y = y_train.astype(float)
        model = SARIMAX(
            y,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit = model.fit(disp=False)
        return self

    def predict(self, horizon: int):
        if self._fit is None:
            raise RuntimeError()
        fc = self._fit.forecast(steps=horizon)
        return np.asarray(fc, dtype=float)
