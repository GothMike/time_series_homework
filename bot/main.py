import logging
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.states import Form
from bot.handlers import start_handler, ticker_handler, amount_handler
from bot.utils.logger import setup_logger

load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = '' # Токен бота тг
if not BOT_TOKEN:
    raise RuntimeError()

logger = setup_logger("stockbot")
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await start_handler(message, state)


@dp.message(Form.waiting_ticker)
async def on_ticker(message: Message, state: FSMContext) -> None:
    await ticker_handler(message, state)


@dp.message(Form.waiting_amount)
async def on_amount(message: Message, state: FSMContext) -> None:
    await amount_handler(message, state, logger)


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    await dp.start_polling(bot)

@dp.message(F.text)
async def fallback_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()

    if current_state is not None:
        return

    text = (message.text or "").strip()

    # пробуем как тикер
    from bot.services.validation import validate_ticker
    ticker = validate_ticker(text)

    if ticker:
        await state.update_data(ticker=ticker)
        await message.answer(
            f"Принял тикер {ticker}.\n"
            "Теперь введите сумму для условной инвестиции (например, 1000)."
        )
        await state.set_state(Form.waiting_amount)
        return

    # иначе — мягкая подсказка
    await message.answer(
        "Чтобы начать, просто напишите тикер компании.\n"
        "Например: AAPL, MSFT, TSLA."
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
