"""Structured logging utilities using Rich + Python logging."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(
    name: str,
    level: str = "info",
    log_file: str | None = None,
) -> logging.Logger:
    """Return a configured logger with Rich console output."""
    logger = logging.getLogger(name)
    logger.setLevel(_LEVEL_MAP.get(level.lower(), logging.INFO))

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

    return logger


def setup_root_logger(level: str = "info") -> None:
    """Configure the root logger for the entire package."""
    logging.basicConfig(
        level=_LEVEL_MAP.get(level.lower(), logging.INFO),
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        format="%(message)s",
    )
