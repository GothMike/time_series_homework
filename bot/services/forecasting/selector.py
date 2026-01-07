import numpy as np
import pandas as pd

from .metrics import rmse, mape
from .preprocessing import train_test_split_series
from .ml_ridge import RidgeLagModel
from .stat_arima import ARIMAModel
from .nn_lstm import LSTMModel

def evaluate_models(
    y: pd.Series,
    test_size: int = 60,
    horizon: int = 30,
):
    y_train, y_test = train_test_split_series(y, test_size=test_size)

    candidates = [
        ("RidgeLag", RidgeLagModel(lags=14, alpha=1.0)),
        ("ARIMA", ARIMAModel(order=(1, 1, 1))),
        ("LSTM", LSTMModel(lookback=30, epochs=50, batch_size=32)),
    ]

    results = []
    fitted = {}

    # обучаем на train и прогнозируем на длину test для оценки
    for name, model in candidates:
        try:
            model.fit(y_train)

            if name == "ARIMA":
                pred_test = model.predict(horizon=len(y_test))
            else:
                pred_test = model.predict(y_context=y_train, horizon=len(y_test))

            score = {
                "model": name,
                "rmse": rmse(y_test.values, pred_test),
                "mape": mape(y_test.values, pred_test),
            }
            results.append(score)
            fitted[name] = model

        except Exception as e:
            results.append({
                "model": name,
                "rmse": float("inf"),
                "mape": float("inf"),
                "error": str(e)[:300],
            })

    # выбор лучшей модели (по RMSE, при равенстве по MAPE)
    results_sorted = sorted(results, key=lambda d: (d.get("rmse", float("inf")), d.get("mape", float("inf"))))
    best = results_sorted[0]["model"]

    # переобучение лучшей модели на всём ряду и прогноз horizon
    y_full = y.astype(float)
    if best == "RidgeLag":
        best_model = RidgeLagModel(lags=14, alpha=1.0).fit(y_full)
        forecast = best_model.predict(y_context=y_full, horizon=horizon)
    elif best == "ARIMA":
        best_model = ARIMAModel(order=(1, 1, 1)).fit(y_full)
        forecast = best_model.predict(horizon=horizon)
    else:
        best_model = LSTMModel(lookback=30, epochs=50, batch_size=32).fit(y_full)
        forecast = best_model.predict(y_context=y_full, horizon=horizon)

    return {
        "metrics": results_sorted, # список метрик по всем моделям
        "best_model": best, # имя лучшей
        "forecast": np.asarray(forecast, dtype=float), # прогноз на 30 дней
    }