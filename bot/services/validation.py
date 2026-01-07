import re
from typing import Optional

TICKER_RE = re.compile(r"^[A-Z.\-]{1,10}$")


def validate_ticker(ticker: str):
    t = (ticker or "").strip().upper()
    return t if TICKER_RE.match(t) else None


def parse_amount(text: str) -> Optional[float]:
    s = (text or "").strip().replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return None

    if value <= 0 or value > 1e9:
        return None

    return round(value, 2)