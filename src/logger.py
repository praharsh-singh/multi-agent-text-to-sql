"""Shared logging factory for the Text-to-SQL pipeline.

Provides a single ``get_logger`` factory that stamps every log line with a
timestamp, severity, and the originating module name.
"""

import logging
import sys
from typing import Optional

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates and returns a configured :class:`logging.Logger`.

    Attaches a single stdout handler on first call for a given ``name``;
    subsequent calls return the cached logger without duplicate handlers.

    Args:
        name:  Logger name, typically ``__name__`` of the calling module.
        level: Logging verbosity threshold. Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger: logging.Logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    return logger