import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.types.input_file import FSInputFile

from bot.states import Form
from bot.services.validation import validate_ticker, parse_amount
from bot.services.market_data import load_history
from bot.services.forecasting.selector import evaluate_models
from bot.services.forecasting.dates import make_future_trading_days
from bot.services.forecasting.plotting import plot_history_and_forecast
from bot.services.forecasting.recommendations import simulate_strategy
from bot.utils.logger import log_json_line

from bot.texts import (
    pick,
    START_PROMPTS,
    TICKER_INVALID,
    ASK_AMOUNT,
    AMOUNT_INVALID,
    PROGRESS_LOAD,
    PROGRESS_TRAIN,
    PROGRESS_PLOT,
    PROGRESS_RECO,
    ERROR_GENERIC,
    ERROR_NO_DATA,
    FOOTER_HINT,
)


@dataclass
class SessionLog:
    user_id: int
    ts_utc: str
    ticker: str
    amount: float
    best_model: str
    metric_name: str
    metric_value: float
    profit: float


def pick_primary_metric(metrics: list[dict], best_model: str) -> Tuple[str, float]:
    best_row = next((m for m in metrics if m.get("model") == best_model), None)
    if not best_row or best_row.get("rmse") is None:
        return "RMSE", float("inf")
    return "RMSE", float(best_row["rmse"])


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


async def start_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(pick(START_PROMPTS))
    await state.set_state(Form.waiting_ticker)


async def ticker_handler(message: Message, state: FSMContext) -> None:
    ticker = validate_ticker(message.text or "")
    if not ticker:
        await message.answer(pick(TICKER_INVALID))
        return

    await state.update_data(ticker=ticker)
    await message.answer(pick(ASK_AMOUNT))
    await state.set_state(Form.waiting_amount)


async def amount_handler(message: Message, state: FSMContext, logger) -> None:
    data = await state.get_data()
    ticker = data.get("ticker")

    amount = parse_amount(message.text or "")
    if amount is None:
        await message.answer(pick(AMOUNT_INVALID))
        return

    user = message.from_user
    start_ts = time.time()

    # 1) load
    await message.answer(pick(PROGRESS_LOAD))
    try:
        df = load_history(ticker)
        if df.empty:
            await message.answer(pick(ERROR_NO_DATA) + "\n" + pick(FOOTER_HINT))
            await state.clear()
            return

        y = df.set_index("Date")["Close"]
        last_close = float(y.iloc[-1])

        # 2) train
        await message.answer(pick(PROGRESS_TRAIN))
        out = evaluate_models(y, test_size=60, horizon=30)
        best_model = out["best_model"]
        metrics = out["metrics"]
        forecast = out["forecast"]

        # 3) dates + plot
        forecast_dates = make_future_trading_days(y.index[-1], periods=30)

        await message.answer(pick(PROGRESS_PLOT))
        plots_dir = Path("plots")
        plot_path = plots_dir / f"{ticker}_{int(time.time())}.png"
        saved_path = plot_history_and_forecast(
            history=y,
            forecast_dates=forecast_dates,
            forecast_values=forecast,
            out_path=str(plot_path),
            history_points=180,
            title=f"{ticker} — 30-day forecast ({best_model})",
        )

        # 4) recommendations
        await message.answer(pick(PROGRESS_RECO))
        strat = simulate_strategy(
            amount=float(amount),
            forecast_dates=forecast_dates,
            forecast_values=forecast,
            close_at_end=True,
        )

        # 5) logging (stage 5)
        metric_name, metric_value = pick_primary_metric(metrics, best_model)
        log_json_line(
            logger,
            SessionLog(
                user_id=user.id,
                ts_utc=datetime.now(timezone.utc).isoformat(),
                ticker=str(ticker),
                amount=float(amount),
                best_model=str(best_model),
                metric_name=metric_name,
                metric_value=float(metric_value),
                profit=float(strat.profit),
            ),
        )

        # results
        await message.answer_photo(FSInputFile(saved_path))

        fc_last = float(forecast[-1])
        delta_abs = fc_last - last_close
        delta_pct = (delta_abs / last_close) * 100.0 if last_close != 0 else 0.0

        direction = "вырастет" if delta_abs >= 0 else "упадёт"
        sign = "+" if delta_abs >= 0 else ""

        await message.answer(
            f"Готово по {ticker}.\n"
            f"Лучшая модель: {best_model}\n"
            f"Метрика ({metric_name}): {metric_value:.4f}\n\n"
            f"Сейчас: {_fmt_money(last_close)}\n"
            f"Через 30 торговых дней: {_fmt_money(fc_last)}\n"
            f"Ожидаемо {direction}: {sign}{delta_abs:.2f} ({sign}{delta_pct:.2f}%)\n\n"
            f"Стратегия по экстремумам (условно): прибыль {sign}{strat.profit:.2f} ({sign}{strat.profit_pct:.2f}%)"
        )

        duration_ms = int((time.time() - start_ts) * 1000)
        _ = duration_ms

        await state.clear()

    except Exception:
        await message.answer(pick(ERROR_GENERIC))
        await state.clear()
