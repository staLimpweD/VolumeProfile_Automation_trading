"""
src/guardian/rules.py - the Guardian: plain-code gatekeeper between ideas and money.

Think of the compliance officer at a bank. The analyst (AAVP + AI) may propose a trade; the
Guardian checks it against fixed rules and can say no. It never takes orders from the AI
or from the phone API.

Principles
  * FAIL CLOSED: missing data (no quote, no baseline equity, no news calendar...) = reject
  * EVERY check runs and is recorded, so the journal shows all reasons for a rejection
  * the Guardian alone decides position SIZE (fixed % risk); nobody else may
  * some breaches HALT the bot until a human calls resume(): max drawdown, too many
    consecutive losses, an unprotected position, feed / broker outages

Rules come from config/guardian.yaml (see load_guardian_config).
"""
from __future__ import annotations

import csv
import logging
import re
import uuid
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from src.models import (AccountInfo, BotError, OrderRequest, Position, Quote, Side, SymbolSpec,
                        risk_position_size)
from src.signals.strategy import TradeProposal
from src.utils import utcnow

log = logging.getLogger(__name__)

FX_LIKE = ("forex", "metal")                       # asset classes that follow weekday / session rules


class ConfigError(BotError):
    """guardian.yaml is invalid. The bot must refuse to start."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AccountRules:
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 6.0
    max_drawdown_pct: float = 12.0
    max_open_positions: int = 2
    max_positions_per_symbol: int = 1
    max_total_risk_pct: float = 2.0
    max_consecutive_losses: int = 3
    min_free_margin_pct: float = 50.0


@dataclass(frozen=True)
class TradeRules:
    require_stop_loss: bool = True
    min_risk_reward: float = 1.5
    min_stop_atr: float = 0.5
    max_stop_atr: float = 4.0
    max_slippage_pct: float = 0.10
    signal_max_age_seconds: float = 120
    cooldown_minutes_after_loss: float = 30


@dataclass(frozen=True)
class MarketRules:
    max_spread_atr_fraction: float = 0.15
    max_quote_age_seconds: float = 15
    min_atr_percentile: float = 20
    max_atr_percentile: float = 97
    require_higher_tf_alignment: bool = False


@dataclass(frozen=True)
class SessionRules:
    enabled: bool = True
    windows: tuple = ()                            # tuple of (name, start "HH:MM", end "HH:MM")
    crypto_24_7: bool = True
    block_weekend_forex: bool = True
    friday_cutoff_utc: str = "20:00"


@dataclass(frozen=True)
class NewsRules:
    enabled: bool = True
    calendar_file: str = "data/news_calendar.csv"
    blackout_minutes_before: float = 30
    blackout_minutes_after: float = 30
    impact_levels: tuple = ("high",)
    symbol_currencies: dict = field(default_factory=dict)
    require_future_events: bool = True             # an out-of-date calendar must not look like "no news"


@dataclass(frozen=True)
class AIRules:
    require_agreement: bool = True
    min_confidence: float = 0.70


@dataclass(frozen=True)
class KillRules:
    close_positions_on_kill: bool = True
    halt_on_unprotected_position: bool = True
    halt_on_feed_failure_minutes: float = 5
    halt_on_broker_disconnect_minutes: float = 3


_SECTIONS = {"account": AccountRules, "trade": TradeRules, "market_conditions": MarketRules,
             "sessions": SessionRules, "news": NewsRules, "ai": AIRules, "kill_switch": KillRules}


def _hhmm(value: str, where: str) -> dtime:
    try:
        h, m = value.split(":")
        return dtime(int(h), int(m))
    except Exception as exc:
        raise ConfigError(f"{where}: '{value}' is not HH:MM") from exc


@dataclass(frozen=True)
class GuardianConfig:
    account: AccountRules
    trade: TradeRules
    market: MarketRules
    sessions: SessionRules
    news: NewsRules
    ai: AIRules
    kill: KillRules
    mode: str = "demo"

    def validate(self) -> "GuardianConfig":
        """Hard limits that protect you from a typo in the YAML (e.g. 10% risk per trade)."""
        a, t, m = self.account, self.trade, self.market

        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)
        need(0 < a.max_risk_per_trade_pct <= 5, "account.max_risk_per_trade_pct must be in (0, 5]")
        need(0 < a.max_daily_loss_pct <= 20, "account.max_daily_loss_pct must be in (0, 20]")
        need(a.max_daily_loss_pct <= a.max_weekly_loss_pct <= 50, "account.max_weekly_loss_pct must be >= daily and <= 50")
        need(0 < a.max_drawdown_pct <= 50, "account.max_drawdown_pct must be in (0, 50]")
        need(a.max_open_positions >= 1 and a.max_positions_per_symbol >= 1, "position limits must be >= 1")
        need(a.max_risk_per_trade_pct <= a.max_total_risk_pct <= 20, "account.max_total_risk_pct must be >= per-trade risk and <= 20")
        need(a.max_consecutive_losses >= 1, "account.max_consecutive_losses must be >= 1")
        need(0 <= a.min_free_margin_pct < 100, "account.min_free_margin_pct must be in [0, 100)")
        need(t.require_stop_loss, "trade.require_stop_loss must stay true (the bot never trades without a stop)")
        need(t.min_risk_reward >= 1.0, "trade.min_risk_reward must be >= 1")
        need(0 < t.min_stop_atr < t.max_stop_atr, "trade.min_stop_atr must be positive and below max_stop_atr")
        need(0 < t.max_slippage_pct <= 1, "trade.max_slippage_pct must be in (0, 1]")
        need(t.signal_max_age_seconds > 0 and m.max_quote_age_seconds > 0, "age limits must be positive")
        need(0 < m.max_spread_atr_fraction <= 1, "market_conditions.max_spread_atr_fraction must be in (0, 1]")
        need(0 <= m.min_atr_percentile < m.max_atr_percentile <= 100, "ATR percentile bounds are invalid")
        need(0 <= self.ai.min_confidence <= 1, "ai.min_confidence must be in [0, 1]")
        for name, start, end in self.sessions.windows:
            _hhmm(start, f"sessions.{name}.start"), _hhmm(end, f"sessions.{name}.end")
        _hhmm(self.sessions.friday_cutoff_utc, "sessions.friday_cutoff_utc")
        return self

    def with_overrides(self, overrides: dict[str, Any]) -> "GuardianConfig":
        """Apply per-symbol overrides (symbols.yaml -> guardian:). Each key must exist in exactly one section."""
        cfg = self
        attr_of = {"account": "account", "trade": "trade", "market_conditions": "market",
                   "sessions": "sessions", "news": "news", "ai": "ai", "kill_switch": "kill"}
        for key, value in (overrides or {}).items():
            homes = [s for s, cls in _SECTIONS.items() if key in {f.name for f in fields(cls)}]
            if len(homes) != 1:
                raise ConfigError(f"unknown guardian override '{key}'")
            attr = attr_of[homes[0]]
            cfg = replace(cfg, **{attr: replace(getattr(cfg, attr), **{key: value})})
        return cfg.validate()


def _build(cls, data: dict, section: str):
    data = dict(data or {})
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"guardian.yaml [{section}]: unknown keys {sorted(unknown)}")
    if cls is SessionRules and "windows" in data:
        wins = []
        for w in data["windows"] or []:
            if not {"name", "start", "end"} <= set(w):
                raise ConfigError("each session window needs name, start and end")
            wins.append((str(w["name"]), str(w["start"]), str(w["end"])))
        data["windows"] = tuple(wins)
    if cls is NewsRules and "impact_levels" in data:
        data["impact_levels"] = tuple(str(x).lower() for x in data["impact_levels"])
    return cls(**data)


def load_guardian_config(path: str | Path, mode: str = "demo") -> GuardianConfig:
    """Read guardian.yaml, apply the demo/live overlay, validate. Raises ConfigError if anything is off."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if mode not in ("demo", "live"):
        raise ConfigError("mode must be 'demo' or 'live'")
    unknown = set(raw) - set(_SECTIONS) - {"modes"}
    if unknown:
        raise ConfigError(f"guardian.yaml: unknown sections {sorted(unknown)}")

    account = dict(raw.get("account") or {})
    overlay = (raw.get("modes") or {}).get(mode) or {}
    bad = set(overlay) - {f.name for f in fields(AccountRules)}
    if bad:
        raise ConfigError(f"modes.{mode}: keys {sorted(bad)} are not account rules")
    account.update(overlay)                                   # the mode overlay wins

    built = {s: _build(cls, account if s == "account" else raw.get(s), s) for s, cls in _SECTIONS.items()}
    return GuardianConfig(built["account"], built["trade"], built["market_conditions"], built["sessions"],
                          built["news"], built["ai"], built["kill_switch"], mode).validate()


# --------------------------------------------------------------------------- #
# News calendar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    title: str


class NewsCalendar:
    """
    CSV with columns: time_utc, currency, impact, title (ISO times, e.g. 2026-11-13T13:30:00Z).
    Reloaded automatically when the file changes. You keep it updated (weekly export from an
    economic calendar). A missing or stale file makes the Guardian refuse to trade.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mtime: Optional[float] = None
        self._events: list[NewsEvent] = []

    def _load(self) -> bool:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._events, self._mtime = [], None
            return False
        if mtime == self._mtime:
            return True
        events: list[NewsEvent] = []
        with self.path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    t = datetime.fromisoformat(row["time_utc"].strip().replace("Z", "+00:00"))
                    t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
                    events.append(NewsEvent(t.astimezone(timezone.utc), row["currency"].strip().upper(),
                                            row["impact"].strip().lower(), row.get("title", "").strip()))
                except (KeyError, ValueError, AttributeError):
                    log.warning("skipping malformed news row: %s", row)
        self._events, self._mtime = events, mtime
        return True

    def available(self) -> bool:
        return self._load()

    def has_future_events(self, now: datetime) -> bool:
        return self._load() and any(e.time > now for e in self._events)

    def active(self, currencies, now: datetime, before: float, after: float, impacts) -> Optional[NewsEvent]:
        """An event of a relevant currency/impact whose blackout window contains `now`."""
        if not self._load():
            return None
        cur = {c.upper() for c in currencies}
        for e in self._events:
            if e.currency in cur and e.impact in impacts and \
                    e.time - timedelta(minutes=before) <= now <= e.time + timedelta(minutes=after):
                return e
        return None

    def upcoming(self, currencies, now: datetime, hours: float = 6, impacts=("high",)) -> list[NewsEvent]:
        """For the AI prompt: events coming up soon."""
        if not self._load():
            return []
        cur, end = {c.upper() for c in currencies}, now + timedelta(hours=hours)
        return sorted((e for e in self._events if e.currency in cur and e.impact in impacts and now <= e.time <= end),
                      key=lambda e: e.time)


# --------------------------------------------------------------------------- #
# Inputs and outputs
# --------------------------------------------------------------------------- #
@dataclass
class RiskSnapshot:
    """Numbers the journal provides (see Journal.risk_inputs). None means 'unknown' and fails closed."""
    equity_start_of_day: Optional[float]
    equity_start_of_week: Optional[float]
    equity_peak: Optional[float]
    consecutive_losses: int = 0
    last_loss_by_symbol: dict[str, datetime] = field(default_factory=dict)


@dataclass
class GuardianContext:
    account: AccountInfo
    quote: Quote
    positions: list[Position]
    specs: dict[str, SymbolSpec]                   # must contain the proposal's symbol (and open positions' symbols)
    risk: RiskSnapshot
    now: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Decision:
    approved: bool
    checks: list[CheckResult]
    request: Optional[OrderRequest] = None
    volume: float = 0.0
    risk_amount: float = 0.0
    halt_reason: Optional[str] = None

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"approved": self.approved, "volume": self.volume, "risk_amount": round(self.risk_amount, 2),
                "halt_reason": self.halt_reason, "reasons": self.reasons,
                "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks]}


# --------------------------------------------------------------------------- #
# Market helpers (compute proposal inputs; used by main.py)
# --------------------------------------------------------------------------- #
def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def atr_percentile(df: pd.DataFrame, period: int = 14, window: int = 500) -> Optional[float]:
    """Where the current ATR ranks (0-100) against the last `window` ATR values. None if not enough data."""
    atr = _atr_series(df, period).dropna().tail(window)
    if len(atr) < 30:
        return None
    return float((atr < atr.iloc[-1]).mean() * 100.0)


def higher_tf_trend(df: pd.DataFrame, ema_period: int = 50, slope_bars: int = 5) -> Optional[str]:
    """'up' / 'down' / 'flat' from price vs a rising/falling EMA on the higher timeframe. None if not enough data."""
    if len(df) < ema_period + slope_bars:
        return None
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    rising, price = ema.iloc[-1] > ema.iloc[-1 - slope_bars], df["close"].iloc[-1]
    if price > ema.iloc[-1] and rising:
        return "up"
    if price < ema.iloc[-1] and not rising:
        return "down"
    return "flat"


# --------------------------------------------------------------------------- #
# The Guardian
# --------------------------------------------------------------------------- #
class Guardian:
    def __init__(self, config: GuardianConfig, *, ai_enabled: bool = True,
                 news: Optional[NewsCalendar] = None, symbol_overrides: Optional[dict[str, dict]] = None):
        self.config = config
        self.ai_enabled = ai_enabled
        self.news = news
        self._overrides = symbol_overrides or {}
        self._cfg_cache: dict[str, GuardianConfig] = {}
        self.halted: Optional[str] = None
        self.baseline_at: Optional[datetime] = None     # after resume(): count losses / peak equity only from here
        self._feed_down_since: Optional[datetime] = None
        self._broker_down_since: Optional[datetime] = None

    # ---- halting --------------------------------------------------------------------
    def halt(self, reason: str) -> None:
        if self.halted is None:
            log.critical("GUARDIAN HALT: %s", reason)
            self.halted = reason

    def resume(self, now: Optional[datetime] = None) -> None:
        """Human decision only. Clears the halt and restarts loss streak / drawdown counting from now."""
        self.halted, self.baseline_at = None, now or utcnow()
        self._feed_down_since = self._broker_down_since = None
        log.warning("Guardian resumed by operator; baseline reset to %s", self.baseline_at)

    def on_unprotected_position(self, symbol: str) -> None:
        if self.config.kill.halt_on_unprotected_position:
            self.halt(f"unprotected position on {symbol}")

    def watchdog(self, now: datetime, feed_ok: bool, broker_ok: bool) -> Optional[str]:
        """Call every loop. Halts and returns the reason when data or the broker has been down too long."""
        k = self.config.kill
        self._feed_down_since = None if feed_ok else (self._feed_down_since or now)
        self._broker_down_since = None if broker_ok else (self._broker_down_since or now)
        if self._feed_down_since and now - self._feed_down_since >= timedelta(minutes=k.halt_on_feed_failure_minutes):
            self.halt(f"no market data for {k.halt_on_feed_failure_minutes:g} minutes")
        if self._broker_down_since and now - self._broker_down_since >= timedelta(minutes=k.halt_on_broker_disconnect_minutes):
            self.halt(f"broker disconnected for {k.halt_on_broker_disconnect_minutes:g} minutes")
        return self.halted

    # ---- sessions ---------------------------------------------------------------------
    def active_sessions(self, now: datetime) -> list[str]:
        t = now.astimezone(timezone.utc).time()
        out = []
        for name, start, end in self.config.sessions.windows:
            s, e = _hhmm(start, name), _hhmm(end, name)
            if (s <= t < e) if s <= e else (t >= s or t < e):       # a window may cross midnight
                out.append(name)
        return out

    def _cfg_for(self, symbol: str) -> GuardianConfig:
        if symbol not in self._cfg_cache:
            self._cfg_cache[symbol] = self.config.with_overrides(self._overrides.get(symbol, {}))
        return self._cfg_cache[symbol]

    # ---- evaluation ------------------------------------------------------------------
    def evaluate(self, p: TradeProposal, ctx: GuardianContext) -> Decision:
        cfg = self._cfg_for(p.symbol)
        a, t, m = cfg.account, cfg.trade, cfg.market
        now = ctx.now.astimezone(timezone.utc)
        checks: list[CheckResult] = []

        def add(name: str, ok: bool, detail: str = "") -> None:
            checks.append(CheckResult(name, bool(ok), "" if ok else detail))

        acc, risk = ctx.account, ctx.risk
        equity, balance = acc.equity, acc.balance
        basis = min(equity, balance)

        # 1. operator / safety halts
        add("not_halted", self.halted is None, f"guardian is halted ({self.halted})")
        add("account_trading_allowed", acc.trade_allowed, "trading is disabled on this account")
        add("equity_positive", equity > 0 and balance > 0, "equity or balance is not positive")

        # 2. freshness of information
        age = (now - p.signal_time).total_seconds()
        add("signal_fresh", 0 <= age <= t.signal_max_age_seconds,
            f"signal is {age:.0f}s old (limit {t.signal_max_age_seconds:g}s)")
        qage = (now - ctx.quote.time).total_seconds()
        add("quote_fresh", -5 <= qage <= m.max_quote_age_seconds, f"quote is {qage:.0f}s old (limit {m.max_quote_age_seconds:g}s)")

        # 3. stop / target geometry
        risk_dist, side_ok = p.risk_distance, (p.stop_loss < p.entry if p.side is Side.BUY else p.stop_loss > p.entry)
        add("stop_loss_valid", side_ok and risk_dist > 0, "stop-loss missing or on the wrong side of entry")
        if p.atr > 0 and risk_dist > 0:
            ratio = risk_dist / p.atr
            add("stop_distance_atr", t.min_stop_atr <= ratio <= t.max_stop_atr,
                f"stop is {ratio:.2f} ATR away (allowed {t.min_stop_atr:g}-{t.max_stop_atr:g})")
        else:
            add("stop_distance_atr", False, "ATR or stop distance unavailable")
        tp_ok = p.take_profit is not None and (p.take_profit > p.entry if p.side is Side.BUY else p.take_profit < p.entry)
        add("take_profit_valid", tp_ok, "take-profit missing or on the wrong side of entry")
        add("risk_reward", tp_ok and p.risk_reward >= t.min_risk_reward,
            f"risk:reward is {p.risk_reward:.2f} (minimum {t.min_risk_reward:g})")

        # 4. market conditions
        spread_limit = m.max_spread_atr_fraction * p.atr
        add("spread", p.atr > 0 and ctx.quote.spread <= spread_limit,
            f"spread {ctx.quote.spread:.5g} exceeds {spread_limit:.5g} ({m.max_spread_atr_fraction:g} ATR)")
        pct = p.atr_percentile
        add("volatility_regime", pct is not None and m.min_atr_percentile <= pct <= m.max_atr_percentile,
            "ATR percentile unavailable" if pct is None else
            f"ATR percentile {pct:.0f} outside {m.min_atr_percentile:g}-{m.max_atr_percentile:g}")
        if m.require_higher_tf_alignment:
            want = "up" if p.side is Side.BUY else "down"
            add("higher_tf_alignment", p.htf_trend == want, f"higher-timeframe trend is {p.htf_trend}, need {want}")

        # 5. time rules
        self._check_time(cfg, p, now, add)

        # 6. news
        self._check_news(cfg, p, now, add)

        # 7. account state
        self._check_account(cfg, p, ctx, equity, basis, now, add)

        # 8. AI second opinion
        if self.ai_enabled and cfg.ai.require_agreement:
            op = p.ai
            if op is None or not op.ok:
                add("ai_opinion", False, f"no usable AI opinion ({getattr(op, 'error', 'not requested')})")
            else:
                add("ai_agrees", op.agree, f"AI disagrees: {op.reason}")
                add("ai_confidence", op.confidence >= cfg.ai.min_confidence,
                    f"AI confidence {op.confidence:.2f} below {cfg.ai.min_confidence:g}")

        # 9. sizing (only when the geometry is valid) and the final risk limits
        volume, risk_amount = 0.0, 0.0
        spec = ctx.specs.get(p.symbol)
        if spec is None:
            add("symbol_spec", False, f"no trading rules loaded for {p.symbol}")
        elif side_ok and risk_dist > 0:
            target_risk = basis * a.max_risk_per_trade_pct / 100.0
            volume = risk_position_size(spec, target_risk, p.entry, p.stop_loss)
            risk_amount = self._risk_amount(spec, p.entry, p.stop_loss, volume)
            add("size_above_minimum", volume > 0,
                f"minimum lot would risk more than {a.max_risk_per_trade_pct:g}% of the account")
            if volume > 0:
                existing = self._open_risk(ctx, basis, a)
                total_pct = (existing + risk_amount) / basis * 100.0
                add("total_open_risk", total_pct <= a.max_total_risk_pct,
                    f"total risk would be {total_pct:.2f}% (limit {a.max_total_risk_pct:g}%)")

        approved = all(c.passed for c in checks)
        request = None
        if approved:
            request = OrderRequest(symbol=p.symbol, side=p.side, volume=volume, stop_loss=p.stop_loss,
                                   take_profit=p.take_profit, client_id=self._client_id(p.symbol), ref_price=p.entry,
                                   max_slippage_pct=t.max_slippage_pct,
                                   comment=f"{p.level_name} {p.timeframe}".strip())
        else:
            volume, risk_amount = 0.0, 0.0
        return Decision(approved, checks, request, volume, risk_amount, self.halted)

    # ---- check groups ---------------------------------------------------------------
    def _check_time(self, cfg, p, now, add) -> None:
        s = cfg.sessions
        if not s.enabled:
            return
        if p.asset_class in FX_LIKE and s.block_weekend_forex:
            wd = now.weekday()                                      # Mon=0 .. Sun=6
            cutoff = _hhmm(s.friday_cutoff_utc, "friday_cutoff_utc")
            closed = wd >= 5 or (wd == 4 and now.time() >= cutoff)
            add("weekend_block", not closed, "FX/metals market closed for new trades (weekend / Friday cutoff)")
        if p.asset_class == "crypto" and s.crypto_24_7:
            return
        add("session_window", bool(self.active_sessions(now)) or not s.windows,
            f"{now:%H:%M} UTC is outside the allowed sessions")

    def _check_news(self, cfg, p, now, add) -> None:
        n = cfg.news
        currencies = n.symbol_currencies.get(p.symbol)
        if not n.enabled or not currencies:
            return                                                  # e.g. crypto has no mapped currencies
        if self.news is None or not self.news.available():
            add("news_calendar", False, "news calendar file is missing")
            return
        if n.require_future_events and not self.news.has_future_events(now):
            add("news_calendar", False, "news calendar has no upcoming events (file out of date?)")
            return
        event = self.news.active(currencies, now, n.blackout_minutes_before, n.blackout_minutes_after, n.impact_levels)
        add("news_blackout", event is None,
            f"{event.impact} {event.currency} news '{event.title}' at {event.time:%H:%M} UTC" if event else "")

    def _check_account(self, cfg, p, ctx, equity, basis, now, add) -> None:
        a, risk = cfg.account, ctx.risk
        # drawdown and loss streak HALT the bot until a human resumes
        if risk.equity_peak and risk.equity_peak > 0:
            dd = (risk.equity_peak - equity) / risk.equity_peak * 100.0
            if dd >= a.max_drawdown_pct:
                self.halt(f"max drawdown reached ({dd:.1f}% from peak)")
            add("max_drawdown", dd < a.max_drawdown_pct, f"drawdown {dd:.1f}% (limit {a.max_drawdown_pct:g}%)")
        else:
            add("max_drawdown", False, "peak equity unknown")
        if risk.consecutive_losses >= a.max_consecutive_losses:
            self.halt(f"{risk.consecutive_losses} consecutive losses")
        add("loss_streak", risk.consecutive_losses < a.max_consecutive_losses,
            f"{risk.consecutive_losses} consecutive losses (limit {a.max_consecutive_losses})")

        for label, base, limit in (("daily_loss", risk.equity_start_of_day, a.max_daily_loss_pct),
                                   ("weekly_loss", risk.equity_start_of_week, a.max_weekly_loss_pct)):
            if not base or base <= 0:
                add(label, False, f"{label.split('_')[0]} baseline equity unknown")
                continue
            loss = (base - equity) / base * 100.0
            add(label, loss < limit, f"{loss:.2f}% down (limit {limit:g}%)")

        mine = [pos for pos in ctx.positions if pos.symbol == p.symbol]
        add("max_open_positions", len(ctx.positions) < a.max_open_positions,
            f"{len(ctx.positions)} positions open (limit {a.max_open_positions})")
        add("max_positions_per_symbol", len(mine) < a.max_positions_per_symbol,
            f"already {len(mine)} position(s) on {p.symbol}")

        last_loss = risk.last_loss_by_symbol.get(p.symbol)
        if last_loss is not None:
            mins = (now - last_loss).total_seconds() / 60.0
            add("loss_cooldown", mins >= cfg.trade.cooldown_minutes_after_loss,
                f"last loss on {p.symbol} was {mins:.0f} min ago (cooldown {cfg.trade.cooldown_minutes_after_loss:g})")
        free_pct = ctx.account.free_margin / equity * 100.0 if equity > 0 else 0.0
        add("free_margin", free_pct >= a.min_free_margin_pct, f"free margin {free_pct:.0f}% (minimum {a.min_free_margin_pct:g}%)")

    # ---- risk maths -------------------------------------------------------------------
    @staticmethod
    def _risk_amount(spec: SymbolSpec, entry: float, stop: float, volume: float) -> float:
        return abs(entry - stop) / spec.tick_size * spec.tick_value * volume

    def _open_risk(self, ctx: GuardianContext, basis: float, a: AccountRules) -> float:
        """Money at risk in open trades. A position with no stop counts as a full-size risk (fail closed)."""
        total = 0.0
        for pos in ctx.positions:
            spec = ctx.specs.get(pos.symbol)
            if spec and pos.stop_loss and pos.entry_price:
                adverse = (pos.entry_price - pos.stop_loss) if pos.side is Side.BUY else (pos.stop_loss - pos.entry_price)
                total += max(0.0, adverse) / spec.tick_size * spec.tick_value * pos.volume   # trailed into profit = 0 risk
            else:
                total += basis * a.max_risk_per_trade_pct / 100.0
        return total

    @staticmethod
    def _client_id(symbol: str) -> str:
        return f"{re.sub('[^A-Za-z0-9]', '', symbol)[:6].upper()}-{uuid.uuid4().hex[:10]}"