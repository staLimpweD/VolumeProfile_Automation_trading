"""
src/models.py - shared contracts for the whole bot.

Everything the data, execution, guardian and API layers need to agree on lives
here: plain data classes, exceptions, the two abstract interfaces (MarketFeed and
Broker) and the position-sizing maths.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Timeframes
# --------------------------------------------------------------------------- #
# name -> (seconds, ccxt string)
TIMEFRAMES: dict[str, tuple[int, str]] = {
    "M1": (60, "1m"), "M5": (300, "5m"), "M15": (900, "15m"), "M30": (1800, "30m"),
    "H1": (3600, "1h"), "H4": (14400, "4h"), "D1": (86400, "1d"),
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class BotError(Exception):
    """Base class for all bot errors."""


class FeedError(BotError):
    """Market data could not be fetched or failed validation."""


class BrokerError(BotError):
    """Generic broker / exchange failure."""


class ConnectionLost(BrokerError):
    """Link to the terminal or exchange is down."""


class OrderRejected(BrokerError):
    """The order was refused (by our own checks or by the venue). Safe to NOT retry."""


class UnprotectedPositionError(BrokerError):
    """
    CRITICAL: a position is open but its stop-loss could not be placed and the
    emergency close also failed. A human must act. The main loop must alert and halt.
    """


# --------------------------------------------------------------------------- #
# Enums and data classes
# --------------------------------------------------------------------------- #
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime               # UTC, timezone-aware

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2.0


@dataclass(frozen=True)
class AccountInfo:
    login: str
    currency: str
    balance: float
    equity: float
    free_margin: float
    leverage: float
    trade_allowed: bool


@dataclass(frozen=True)
class SymbolSpec:
    """
    Trading rules of one instrument.

    tick_size   : smallest price step.
    tick_value  : profit/loss, in ACCOUNT currency, of a one-tick move on ONE unit of
                  volume (1 lot on MT5, 1 coin on a USDT-quoted spot market).
    """
    symbol: str
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    digits: int = 2
    min_stop_distance: float = 0.0          # price units; closest a stop may sit to price

    def normalize_volume(self, volume: float) -> float:
        """
        Round DOWN to the volume step and clamp to the maximum.
        Returns 0.0 if the result is below the minimum. We never round UP to the
        minimum, because that would silently risk more than the Guardian approved.
        """
        if volume <= 0 or self.volume_step <= 0:
            return 0.0
        steps = math.floor(volume / self.volume_step + 1e-9)
        vol = round(steps * self.volume_step, 10)
        if vol < self.volume_min - 1e-12:
            return 0.0
        return min(vol, self.volume_max)


@dataclass(frozen=True)
class OrderRequest:
    """A market order the Guardian has already approved."""
    symbol: str
    side: Side
    volume: float
    stop_loss: Optional[float]
    take_profit: Optional[float] = None
    client_id: str = ""                      # unique, <= 24 chars, [A-Za-z0-9_-]; used to detect duplicates
    ref_price: Optional[float] = None        # price the signal was based on (slippage guard)
    max_slippage_pct: float = 0.10           # reject if the market moved more than this since ref_price
    comment: str = ""

    def __post_init__(self):
        if self.volume <= 0:
            raise ValueError("volume must be positive")


@dataclass
class OrderResult:
    ok: bool
    symbol: str
    side: Side
    order_id: str = ""
    position_id: str = ""
    fill_price: float = 0.0
    filled_volume: float = 0.0
    message: str = ""
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class Position:
    id: str
    symbol: str
    side: Side
    volume: float
    entry_price: Optional[float]             # None if unknown (e.g. spot holding after a restart)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    profit: float                            # unrealised, account currency
    opened_at: Optional[datetime] = None
    comment: str = ""


# --------------------------------------------------------------------------- #
# Abstract interfaces
# --------------------------------------------------------------------------- #
class MarketFeed(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def fetch_candles(self, symbol: str, timeframe: str, count: int,
                      include_live: bool = True) -> pd.DataFrame:
        """OHLCV DataFrame, UTC DatetimeIndex (candle open time), columns open/high/low/close/volume.
        The last row is the still-forming candle when include_live=True."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_healthy(self) -> bool: ...

    @abstractmethod
    def account_info(self) -> AccountInfo: ...

    @abstractmethod
    def get_symbol_spec(self, symbol: str) -> SymbolSpec: ...

    @abstractmethod
    def positions(self, symbol: Optional[str] = None) -> list[Position]: ...

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_sltp(self, position_id: str, stop_loss: Optional[float],
                    take_profit: Optional[float]) -> None: ...

    @abstractmethod
    def close_position(self, position_id: str) -> OrderResult: ...

    @abstractmethod
    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]: ...


# --------------------------------------------------------------------------- #
# Sizing and validation helpers
# --------------------------------------------------------------------------- #
def risk_position_size(spec: SymbolSpec, risk_amount: float, entry: float, stop: float) -> float:
    """
    Volume such that hitting the stop loses about `risk_amount` (account currency).

        loss per unit of volume = (|entry - stop| / tick_size) * tick_value
        volume                  = risk_amount / loss per unit   (rounded DOWN to the step)

    The AI never chooses the size; the Guardian calls this with its fixed risk %.
    """
    distance = abs(entry - stop)
    if distance <= 0 or risk_amount <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
        return 0.0
    loss_per_unit = (distance / spec.tick_size) * spec.tick_value
    return spec.normalize_volume(risk_amount / loss_per_unit)


def validate_levels(side: Side, price: float, stop_loss: Optional[float],
                    take_profit: Optional[float], min_distance: float = 0.0) -> None:
    """Raise OrderRejected if SL/TP sit on the wrong side of price or too close to it."""
    if stop_loss is not None:
        wrong = stop_loss >= price if side is Side.BUY else stop_loss <= price
        if wrong or abs(price - stop_loss) < min_distance:
            raise OrderRejected(f"invalid stop-loss {stop_loss} for {side.value} at {price}")
    if take_profit is not None:
        wrong = take_profit <= price if side is Side.BUY else take_profit >= price
        if wrong or abs(price - take_profit) < min_distance:
            raise OrderRejected(f"invalid take-profit {take_profit} for {side.value} at {price}")


def check_slippage(price: float, ref_price: Optional[float], max_pct: float) -> None:
    if ref_price and ref_price > 0:
        moved = abs(price - ref_price) / ref_price * 100.0
        if moved > max_pct:
            raise OrderRejected(f"price moved {moved:.3f}% since signal (limit {max_pct}%)")