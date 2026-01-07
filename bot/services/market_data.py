import pandas as pd
import yfinance as yf


def load_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    return df[["Date", "Close"]].dropna()
