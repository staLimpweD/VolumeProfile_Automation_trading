"""
src/data/exchange_feed.py - candles and quotes from a crypto exchange through ccxt.

Public market data needs no API keys. Crypto exchanges report REAL traded volume,
so volume-profile maths is more trustworthy here than on Forex tick volume.
Symbols use ccxt format, e.g. "SOL/USDT".
"""
from __future__ import annotations

import logging
from typing import Optional

import ccxt
import pandas as pd

from src.models import FeedError, MarketFeed, Quote, TIMEFRAMES
from src.utils import retry, utcnow, validate_candles

log = logging.getLogger(__name__)


def build_exchange(exchange_id: str, api_key: Optional[str] = None, secret: Optional[str] = None,
                   password: Optional[str] = None, sandbox: bool = False,
                   options: Optional[dict] = None):
    """Create a ccxt exchange with rate limiting on. Shared by the feed and the broker."""
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"unknown ccxt exchange id '{exchange_id}'")
    config = {"enableRateLimit": True, "timeout": 15_000, "options": options or {}}
    if api_key and secret:
        config.update(apiKey=api_key, secret=secret)
        if password:
            config["password"] = password
    exchange = getattr(ccxt, exchange_id)(config)
    if sandbox:
        exchange.set_sandbox_mode(True)
    return exchange


class ExchangeFeed(MarketFeed):
    def __init__(self, exchange_id: str = "binance", sandbox: bool = False,
                 options: Optional[dict] = None, exchange=None):
        """`exchange` lets tests inject a fake; normally leave it None."""
        self.exchange_id = exchange_id
        self.ex = exchange or build_exchange(exchange_id, sandbox=sandbox, options=options)

    def connect(self) -> None:
        self._load_markets()

    def close(self) -> None:
        closer = getattr(self.ex, "close", None)
        if callable(closer):
            closer()

    @retry(attempts=3, exceptions=(ccxt.NetworkError,))
    def _load_markets(self) -> None:
        self.ex.load_markets()

    @retry(attempts=3, exceptions=(ccxt.NetworkError,))
    def fetch_candles(self, symbol: str, timeframe: str, count: int,
                      include_live: bool = True) -> pd.DataFrame:
        if timeframe not in TIMEFRAMES:
            raise FeedError(f"unsupported timeframe '{timeframe}' (use one of {sorted(TIMEFRAMES)})")
        try:
            rows = self.ex.fetch_ohlcv(symbol, TIMEFRAMES[timeframe][1], limit=int(count))
        except ccxt.BadSymbol as exc:
            raise FeedError(f"{self.exchange_id} does not list '{symbol}'") from exc
        if not rows:
            raise FeedError(f"no candles for {symbol} {timeframe} from {self.exchange_id}")
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), unit="ms", utc=True), name="time")
        df = validate_candles(df.astype(float), f"{self.exchange_id}:{symbol}")
        return df if include_live else df.iloc[:-1]       # exchanges include the forming candle last

    @retry(attempts=3, exceptions=(ccxt.NetworkError,))
    def get_quote(self, symbol: str) -> Quote:
        t = self.ex.fetch_ticker(symbol)
        bid, ask = t.get("bid"), t.get("ask")
        if not bid or not ask:
            raise FeedError(f"no bid/ask for {symbol} on {self.exchange_id}")
        ts = pd.Timestamp(t["timestamp"], unit="ms", tz="UTC").to_pydatetime() if t.get("timestamp") else utcnow()
        return Quote(symbol, float(bid), float(ask), ts)