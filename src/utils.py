import logging
import sys
from typing import Optional


def setup_logger(name: str = "pdf_rag", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standardized application logger.
    
    Ensures safe formatting and logs to standard output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
