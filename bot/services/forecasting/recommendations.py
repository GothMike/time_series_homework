from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import numpy as np
import pandas as pd


Action = Literal["BUY", "SELL"]


@dataclass
class TradeSignal:
    date: pd.Timestamp
    action: Action
    price: float


@dataclass
class Trade:
    buy_date: pd.Timestamp
    buy_price: float
    sell_date: pd.Timestamp
    sell_price: float
    shares: float
    pnl: float  # profit for this trade


@dataclass
class StrategyResult:
    signals: List[TradeSignal]
    trades: List[Trade]
    final_value: float
    profit: float
    profit_pct: float


def _local_extrema(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает индексы локальных минимумов и максимумов:
    min: v[i-1] > v[i] < v[i+1]
    max: v[i-1] < v[i] > v[i+1]
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    mins = np.where((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
    maxs = np.where((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    return mins, maxs


def build_signals(
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
) -> List[TradeSignal]:
    """
    Строит сигналы BUY на локальных минимумах и SELL на локальных максимумах,
    затем приводит их к чередованию BUY -> SELL -> BUY -> SELL...
    """
    dates = pd.DatetimeIndex(forecast_dates)
    values = np.asarray(forecast_values, dtype=float)

    mins, maxs = _local_extrema(values)

    raw = []
    for i in mins:
        raw.append(TradeSignal(date=dates[i], action="BUY", price=float(values[i])))
    for i in maxs:
        raw.append(TradeSignal(date=dates[i], action="SELL", price=float(values[i])))

    # сортируем по дате
    raw.sort(key=lambda s: s.date)

    # приводим к логике чередования: начинаем с BUY, SELL только после BUY, BUY только после SELL
    signals: List[TradeSignal] = []
    expected: Action = "BUY"
    for s in raw:
        if s.action == expected:
            signals.append(s)
            expected = "SELL" if expected == "BUY" else "BUY"

    # если последний сигнал BUY без SELL — оставляем (можно закрыть в конце по последней цене при симуляции)
    return signals


def simulate_strategy(
    amount: float,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    close_at_end: bool = True,
) -> StrategyResult:
    if amount <= 0:
        raise ValueError("Количество должно быть больше > 0")

    dates = pd.DatetimeIndex(forecast_dates)
    values = np.asarray(forecast_values, dtype=float)
    if len(values) == 0:
        raise ValueError("Нет прогноза")

    signals = build_signals(dates, values)

    cash = float(amount)
    shares = 0.0
    trades: List[Trade] = []

    pending_buy: Optional[TradeSignal] = None

    for s in signals:
        if s.action == "BUY" and shares == 0.0:
            # покупаем на всю сумму
            if s.price <= 0:
                continue
            shares = cash / s.price
            cash = 0.0
            pending_buy = s

        elif s.action == "SELL" and shares > 0.0:
            # продаём всё
            cash = shares * s.price
            if pending_buy is not None:
                pnl = cash - amount if len(trades) == 0 else cash - (trades[-1].sell_price * trades[-1].shares) 
                # корректный pnl считаем по цене покупки
                pnl = (s.price - pending_buy.price) * shares
                trades.append(
                    Trade(
                        buy_date=pending_buy.date,
                        buy_price=pending_buy.price,
                        sell_date=s.date,
                        sell_price=s.price,
                        shares=shares,
                        pnl=float(pnl),
                    )
                )
            shares = 0.0
            pending_buy = None

    # закрытие позиции в конце горизонта
    if close_at_end and shares > 0.0:
        last_price = float(values[-1])
        cash = shares * last_price
        if pending_buy is not None:
            pnl = (last_price - pending_buy.price) * shares
            trades.append(
                Trade(
                    buy_date=pending_buy.date,
                    buy_price=pending_buy.price,
                    sell_date=dates[-1],
                    sell_price=last_price,
                    shares=shares,
                    pnl=float(pnl),
                )
            )
        shares = 0.0
        pending_buy = None

    final_value = cash
    profit = final_value - float(amount)
    profit_pct = (profit / float(amount)) * 100.0 if amount != 0 else 0.0

    return StrategyResult(
        signals=signals,
        trades=trades,
        final_value=float(final_value),
        profit=float(profit),
        profit_pct=float(profit_pct),
    )
