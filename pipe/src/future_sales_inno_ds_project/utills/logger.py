# logger.py

import logging


def setup_logger(name: str = "sales_pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
