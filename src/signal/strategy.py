"""
src/signals/strategy.py - turns an AAVP signal into a concrete TradeProposal.

aavp.py only FLAGS a setup ("rejection off VAL after a sweep"). A tradable idea also needs
an entry reference, a stop-loss and a take-profit. This module decides them with simple,
visible rules, and attaches the context the Guardian checks (ATR, volatility percentile,
higher-timeframe trend, asset class). It never sizes a trade and never talks to a broker.

Stop-loss   : just beyond the sweep extreme (the "invalidation"), plus a small ATR buffer.
Take-profit : the nearest volume-profile level (POC, then VAH for longs; POC, then VAL for
              shorts) that is at least `min_rr` times the risk away. If none qualifies, a
              fixed `min_rr` multiple is used.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.models import Side, TIMEFRAMES
from src.signals.aavp import AAVPResult
from src.signals.ai_reviewer import AIOpinion


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: Side
    entry: float                       # live ask for a buy, live bid for a sell
    stop_loss: float
    take_profit: Optional[float]
    signal_time: datetime              # UTC, when the trigger candle CLOSED
    atr: float
    asset_class: str                   # "forex" | "metal" | "crypto"
    level_name: str                    # POC | VAL | VAH
    level_price: float
    timeframe: str = ""
    atr_percentile: Optional[float] = None
    htf_trend: Optional[str] = None    # "up" | "down" | "flat"
    ai: Optional[AIOpinion] = None

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_distance(self) -> float:
        return abs(self.take_profit - self.entry) if self.take_profit is not None else 0.0

    @property
    def risk_reward(self) -> float:
        risk = self.risk_distance
        return self.reward_distance / risk if risk > 0 else 0.0

    def with_ai(self, opinion: AIOpinion) -> "TradeProposal":
        return dataclasses.replace(self, ai=opinion)


def _round_away(price: float, tick: float, up: bool) -> float:
    """Round to the tick grid in the direction given (stops are rounded AWAY from price)."""
    n = price / tick
    n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
    return round(n * tick, 10)


def _pick_target(side: Side, entry: float, risk: float, poc: float, vah: float, val: float,
                 min_rr: float) -> float:
    levels = [poc, vah] if side is Side.BUY else [poc, val]
    ahead = sorted((p for p in levels if (p > entry if side is Side.BUY else p < entry)),
                   key=lambda p: abs(p - entry))
    for price in ahead:                                             # nearest first
        if abs(price - entry) / risk >= min_rr:
            return price
    return entry + min_rr * risk if side is Side.BUY else entry - min_rr * risk


def build_proposal(symbol: str, result: AAVPResult, atr_value: float, *, bid: float, ask: float,
                   timeframe: str, asset_class: str, tick_size: float, min_rr: float = 1.5,
                   stop_buffer_atr: float = 0.10, atr_percentile: Optional[float] = None,
                   htf_trend: Optional[str] = None) -> Optional[TradeProposal]:
    """Return a proposal if the AAVP result carries a signal and a sane stop can be placed, else None."""
    sig = result.signal
    if not (result.ready and sig and atr_value > 0 and tick_size > 0):
        return None
    side = Side.BUY if sig["direction"] == "long" else Side.SELL
    entry = ask if side is Side.BUY else bid

    buffer = stop_buffer_atr * atr_value
    if side is Side.BUY:
        stop = _round_away(sig["invalidation"] - buffer, tick_size, up=False)
        if stop >= entry:
            return None
    else:
        stop = _round_away(sig["invalidation"] + buffer, tick_size, up=True)
        if stop <= entry:
            return None
    risk = abs(entry - stop)

    target = _pick_target(side, entry, risk, result.poc, result.vah, result.val, min_rr)
    target = _round_away(target, tick_size, up=(side is Side.SELL))   # a tick short of the level is safer

    opened = pd.Timestamp(sig["trigger_time"])
    if opened.tzinfo is None:
        opened = opened.tz_localize("UTC")
    closed_at = opened.to_pydatetime() + timedelta(seconds=TIMEFRAMES[timeframe][0])

    return TradeProposal(symbol=symbol, side=side, entry=entry, stop_loss=stop, take_profit=target,
                         signal_time=closed_at, atr=atr_value, asset_class=asset_class,
                         level_name=sig["level_name"], level_price=sig["level_price"], timeframe=timeframe,
                         atr_percentile=atr_percentile, htf_trend=htf_trend)