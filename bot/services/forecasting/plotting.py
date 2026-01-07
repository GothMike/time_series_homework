from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_history_and_forecast(
    history: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    out_path: str,
    history_points: int = 180,
    title: str = "Stock price forecast",
):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    h = history.dropna().astype(float)
    if len(h) > history_points:
        h = h.iloc[-history_points:]

    forecast_values = np.asarray(forecast_values, dtype=float)
    fc = pd.Series(forecast_values, index=forecast_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(h.index, h.values, label="History")
    plt.plot(fc.index, fc.values, label="Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return str(out)
