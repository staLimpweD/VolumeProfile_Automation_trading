"""
src/data/mt5_feed.py - candles and quotes from a MetaTrader 5 terminal (Forex / Gold).

Volume note: retail Forex/CFD feeds only provide TICK volume (number of price
changes), which is a proxy for activity, not true traded volume.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data.mt5_session import MT5Session
from src.models import FeedError, MarketFeed, Quote, TIMEFRAMES
from src.utils import retry, validate_candles

log = logging.getLogger(__name__)


class MT5Feed(MarketFeed):
    def __init__(self, session: MT5Session, volume_field: str = "tick_volume"):
        if volume_field not in ("tick_volume", "real_volume"):
            raise ValueError("volume_field must be 'tick_volume' or 'real_volume'")
        self.session = session
        self.volume_field = volume_field

    # ---- lifecycle ------------------------------------------------------------
    def connect(self) -> None:
        self.session.connect()

    def close(self) -> None:
        self.session.shutdown()

    # ---- helpers ---------------------------------------------------------------
    def _timeframe(self, name: str):
        if name not in TIMEFRAMES:
            raise FeedError(f"unsupported timeframe '{name}' (use one of {sorted(TIMEFRAMES)})")
        return getattr(self.session.mt5, f"TIMEFRAME_{name}")

    def _select(self, symbol: str) -> None:
        """Make sure the symbol is visible in Market Watch, otherwise MT5 returns no data."""
        if not self.session.mt5.symbol_select(symbol, True):
            raise FeedError(f"MT5 cannot select symbol '{symbol}' (check the exact name incl. suffix, e.g. XAUUSDm)")

    # ---- public ------------------------------------------------------------------
    @retry(attempts=3, exceptions=(FeedError,))
    def fetch_candles(self, symbol: str, timeframe: str, count: int,
                      include_live: bool = True) -> pd.DataFrame:
        s = self.session
        with s.lock:
            s.ensure()
            self._select(symbol)
            # position 0 = the LIVE (still forming) candle
            rates = s.mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 0, int(count))
            if rates is None or len(rates) == 0:
                raise FeedError(f"no candles for {symbol} {timeframe}: {s.mt5.last_error()}")

        df = pd.DataFrame(rates)
        volume = df[self.volume_field]
        if self.volume_field == "real_volume" and not (volume > 0).any():
            log.warning("%s has no real_volume; falling back to tick_volume", symbol)
            volume = df["tick_volume"]
        idx = pd.to_datetime(df["time"], unit="s", utc=True) - pd.Timedelta(hours=s.server_utc_offset_hours)
        out = pd.DataFrame({"open": df["open"], "high": df["high"], "low": df["low"],
                            "close": df["close"], "volume": volume.astype(float)})
        out.index = pd.DatetimeIndex(idx, name="time")
        out = validate_candles(out, f"mt5:{symbol}")
        return out if include_live else out.iloc[:-1]

    @retry(attempts=3, exceptions=(FeedError,))
    def get_quote(self, symbol: str) -> Quote:
        s = self.session
        with s.lock:
            s.ensure()
            self._select(symbol)
            tick = s.mt5.symbol_info_tick(symbol)
            if tick is None or tick.bid <= 0 or tick.ask <= 0:
                raise FeedError(f"no valid tick for {symbol}: {s.mt5.last_error()}")
            return Quote(symbol, float(tick.bid), float(tick.ask), s.to_utc(tick.time))