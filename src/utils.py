"""src/utils.py - small shared helpers: retry with backoff, env access, candle validation."""
from __future__ import annotations

import functools
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from src.models import FeedError

log = logging.getLogger(__name__)

OHLCV = ("open", "high", "low", "close", "volume")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def require_env(name: str) -> str:
    """Read a mandatory secret from the environment; fail loudly if it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def retry(attempts: int = 3, base_delay: float = 0.5, max_delay: float = 8.0,
          exceptions: tuple = (Exception,), give_up_on: tuple = ()) -> Callable:
    """
    Retry with exponential backoff and jitter.
    Use ONLY for safe, repeatable calls (reads). Never wrap order placement with it:
    a blind retry can open the same trade twice.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except give_up_on:
                    raise
                except exceptions as exc:
                    if attempt == attempts:
                        raise
                    pause = min(max_delay, delay) * (0.5 + random.random() / 2)
                    log.warning("%s failed (%s). Retry %d/%d in %.1fs",
                                fn.__name__, exc, attempt, attempts - 1, pause)
                    time.sleep(pause)
                    delay *= 2
        return wrapper
    return decorator


def validate_candles(df: pd.DataFrame, source: str = "feed") -> pd.DataFrame:
    """
    Clean and sanity-check candles before any maths touches them:
    sorted, unique timestamps, no NaN, positive prices, high >= low.
    """
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise FeedError(f"{source}: missing columns {missing}")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=list(OHLCV))
    good = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["high"] >= df["low"]) & (df["volume"] >= 0)
    dropped = int((~good).sum())
    if dropped:
        log.warning("%s: dropped %d malformed candle(s)", source, dropped)
    df = df[good]
    if df.empty:
        raise FeedError(f"{source}: no valid candles")
    return df