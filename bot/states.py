from aiogram.fsm.state import State, StatesGroup


class Form(StatesGroup):
    waiting_ticker = State()
    waiting_amount = State()