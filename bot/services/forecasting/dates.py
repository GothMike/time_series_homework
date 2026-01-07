import pandas as pd

def make_future_trading_days(last_date: pd.Timestamp, periods: int = 30):
    last_date = pd.Timestamp(last_date).normalize()
    start = last_date + pd.tseries.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=periods)