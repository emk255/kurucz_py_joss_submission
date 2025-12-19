"""Progress reporting helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .logging import get_logger


@contextmanager
def progress(step: str) -> Iterator[None]:
    logger = get_logger("progress")
    logger.info("%s...", step)
    try:
        yield
    finally:
        logger.info("%s done", step)
