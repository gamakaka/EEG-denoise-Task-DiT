import logging
import time
from pathlib import Path


def build_logger(name, log_dir, prefix):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    current_time = time.localtime()
    log_path = Path(log_dir) / (
        f"{prefix}_{current_time.tm_mon}_{current_time.tm_mday}_{current_time.tm_hour}_{current_time.tm_min}.log"
    )

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
