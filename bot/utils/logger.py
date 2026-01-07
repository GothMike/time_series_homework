import json
import logging
from dataclasses import asdict
from typing import Any

LOG_FILE = "logs.txt"


def setup_logger(name: str = "stockbot") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def log_json_line(logger: logging.Logger, payload: Any) -> None:
    if hasattr(payload, "__dataclass_fields__"):
        data = asdict(payload)
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError()

    logger.info(json.dumps(data, ensure_ascii=False))
