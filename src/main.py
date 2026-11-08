"""
src/main.py - the conductor. Ties every layer into one safe, restartable trading loop.

    python -m src.main check             # validate config, connect, show accounts, reconcile (no trading)
    python -m src.main run               # the trading loop (default)
    python -m src.main resume-guardian   # clear a Guardian halt from the server (the phone cannot do this)

The cast (see the architecture guide):
    feeds (MT5 / exchange)  ->  AAVP engine  ->  strategy (stop / target)  ->  AI second opinion
        ->  GUARDIAN (fixed rules, the only thing that sizes or approves)  ->  broker (MT5 / exchange)
        ->  journal (everything is recorded)         control API + push alerts for the phone

Safety rules enforced here
  * demo mode refuses to run on a REAL MT5 account or a non-sandbox exchange; live mode needs --confirm-live
  * one signal is traded at most once (persisted, survives restarts)
  * the AI is only asked when every other Guardian rule already passed (saves cost, limits exposure)
  * positions are re-checked every tick: missing stop-loss -> repair, else close and halt
  * halts persist across restarts and can only be cleared on the server
  * stopping the bot leaves positions protected by their broker/exchange-side stops
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import logging.handlers
import math
import os
import queue
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yaml

from src.api.server import ApiServer, create_app
from src.data.exchange_feed import ExchangeFeed
from src.data.mt5_feed import MT5Feed
from src.data.mt5_session import MT5Session
from src.execution.exchange_broker import ExchangeBroker
from src.execution.mt5_broker import MT5Broker
from src.guardian.rules import (FX_LIKE, ConfigError, Guardian, GuardianContext, NewsCalendar, RiskSnapshot,
                                atr_percentile, higher_tf_trend, load_guardian_config)
from src.journal.db import Journal, iso, parse_iso
from src.models import (TIMEFRAMES, AccountInfo, Broker, BrokerError, ConnectionLost, MarketFeed, OrderRejected,
                        OrderRequest, Position, Quote, Side, SymbolSpec, UnprotectedPositionError)
from src.signals.aavp import AAVPConfig, AAVPEngine, AnchorMode, atr
from src.signals.ai_reviewer import AIOpinion, GeminiReviewer, build_review_context
from src.signals.strategy import build_proposal
from src.utils import require_env, utcnow

log = logging.getLogger("cman.bot")

VERSION = "1.0.0"
SETTLE_RETRIES = 3                                     # ticks to wait for broker history before estimating a P&L
FUNDS_BUFFER = 0.98                                    # spot orders may use at most 98% of free USDT (fees, rounding)
PLACEHOLDER_OPINION = AIOpinion(True, True, 1.0, "pre-check placeholder")   # never logged, never traded on


# =========================================================================== #
# 1. Settings
# =========================================================================== #
@dataclass
class SymbolSettings:
    name: str                       # logical name used everywhere inside the bot, e.g. XAUUSD
    venue: str                      # "mt5" | "exchange"
    native: str                     # the venue's own symbol, e.g. "XAUUSDm" or "SOL/USDT"
    timeframe: str
    higher_timeframe: str
    asset_class: str                # forex | metal | crypto
    aavp: dict = field(default_factory=dict)
    guardian: dict = field(default_factory=dict)


@dataclass
class Settings:
    root: Path
    mode: str
    loop_interval: float
    candles: int
    magic: int
    log_level: str
    log_file: str
    equity_interval: float
    health_interval: float
    signal_delay: float
    max_internal_errors: int
    venues: dict
    aavp: dict
    ai: dict
    journal: dict
    api: dict
    alerts: dict
    management: dict
    symbols: list[SymbolSettings]


MANAGEMENT_DEFAULTS = {
    "breakeven_at_r": 0.0,               # 0 = off. e.g. 1.0 moves the stop to entry once price is 1R in profit
    "breakeven_buffer_r": 0.10,          # ...plus this fraction of R beyond entry
    "flatten_fx_friday_utc": None,       # e.g. "20:45" closes Forex/Gold positions before the weekend gap
    "close_unprotected": True,           # a position without a stop that cannot be repaired is closed
    "unprotected_ticks": 3,              # consecutive sightings before acting (avoids false alarms)
}


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def _hhmm(value: str) -> dtime:
    try:
        h, m = str(value).split(":")
        return dtime(int(h), int(m))
    except Exception as exc:
        raise ConfigError(f"'{value}' is not a valid HH:MM time") from exc


def load_settings(config_dir: str | Path = "config") -> Settings:
    cdir = Path(config_dir).resolve()
    root = cdir.parent
    raw, sym_raw = _read_yaml(cdir / "bot.yaml"), _read_yaml(cdir / "symbols.yaml")
    b = raw.get("bot") or {}
    venues, aavp = raw.get("venues") or {}, raw.get("aavp") or {}
    ai, api = raw.get("ai") or {}, raw.get("api") or {}
    mgmt = {**MANAGEMENT_DEFAULTS, **(raw.get("management") or {})}
    unknown = set(raw.get("management") or {}) - set(MANAGEMENT_DEFAULTS)
    if unknown:
        raise ConfigError(f"bot.yaml management: unknown keys {sorted(unknown)}")

    mode = b.get("mode", "demo")
    if mode not in ("demo", "live"):
        raise ConfigError("bot.mode must be 'demo' or 'live'")
    interval = float(b.get("loop_interval_seconds", 5))
    if not 1 <= interval <= 60:
        raise ConfigError("bot.loop_interval_seconds must be between 1 and 60")

    # venue sanity: the mode and the exchange sandbox flag must agree
    ex = venues.get("exchange") or {}
    if ex.get("enabled"):
        if mode == "demo" and not ex.get("sandbox", True):
            raise ConfigError("mode is 'demo' but venues.exchange.sandbox is false: refusing to risk real funds")
        if mode == "live" and ex.get("sandbox", True):
            raise ConfigError("mode is 'live' but venues.exchange.sandbox is true: set sandbox false deliberately")
    if ai.get("enabled") and str(ai.get("model", "")).startswith("SET_"):
        raise ConfigError("ai.model still has the placeholder; set a real Gemini model name (or ai.enabled: false)")
    if mgmt["flatten_fx_friday_utc"]:
        _hhmm(mgmt["flatten_fx_friday_utc"])

    candles = int(b.get("candles_per_fetch", 400))
    symbols: list[SymbolSettings] = []
    for name, s in (sym_raw.get("symbols") or {}).items():
        if not s.get("enabled", False):
            continue
        venue = s.get("venue")
        if venue not in ("mt5", "exchange"):
            raise ConfigError(f"symbol {name}: venue must be 'mt5' or 'exchange'")
        if not (venues.get(venue) or {}).get("enabled"):
            raise ConfigError(f"symbol {name} uses venue '{venue}' which is not enabled in bot.yaml")
        native = s.get("broker_symbol" if venue == "mt5" else "exchange_symbol")
        if not native:
            raise ConfigError(f"symbol {name}: missing {'broker_symbol' if venue == 'mt5' else 'exchange_symbol'}")
        tf, htf = s.get("timeframe", "M15"), s.get("higher_timeframe", "H4")
        if tf not in TIMEFRAMES or htf not in TIMEFRAMES:
            raise ConfigError(f"symbol {name}: timeframe must be one of {sorted(TIMEFRAMES)}")
        merged = {**aavp, **(s.get("aavp") or {})}
        if candles < int(merged.get("lookback", 200)) + 20:
            raise ConfigError(f"bot.candles_per_fetch ({candles}) must be at least aavp.lookback + 20 for {name}")
        symbols.append(SymbolSettings(name, venue, native, tf, htf, s.get("asset_class", "forex"),
                                      s.get("aavp") or {}, s.get("guardian") or {}))
    if not symbols:
        raise ConfigError("no enabled symbols in symbols.yaml")

    return Settings(root=root, mode=mode, loop_interval=interval, candles=candles,
                    magic=int(b.get("magic_number", 20260101)), log_level=str(b.get("log_level", "INFO")),
                    log_file=str(b.get("log_file", "logs/bot.log")),
                    equity_interval=float(b.get("equity_interval_seconds", 60)),
                    health_interval=float(b.get("health_interval_seconds", 15)),
                    signal_delay=float(b.get("signal_delay_seconds", 2)),
                    max_internal_errors=int(b.get("max_internal_errors", 20)),
                    venues=venues, aavp=aavp, ai=ai, journal=raw.get("journal") or {}, api=api,
                    alerts=raw.get("alerts") or {}, management=mgmt, symbols=symbols)


# =========================================================================== #
# 2. Logging (with secret masking)
# =========================================================================== #
class RedactFilter(logging.Filter):
    """Masks anything that looks like a configured secret before it reaches a log file."""
    NAMES = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN)", re.I)

    def __init__(self):
        super().__init__()
        self._secrets: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self._secrets = sorted({v for k, v in os.environ.items() if self.NAMES.search(k) and len(v) >= 6},
                               key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            msg = record.getMessage()
            for s in self._secrets:
                msg = msg.replace(s, "***")
            record.msg, record.args = msg, ()
        return True


def setup_logging(level: str, log_file: Optional[Path]) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
    redact = RedactFilter()
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(redact)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)         # httpx logs full URLs at INFO


# =========================================================================== #
# 3. Alerts
# =========================================================================== #
class Notifier:
    """
    Push notifications without ever blocking the trading loop: messages go on a queue and a
    worker thread sends them. Firebase Cloud Messaging is optional (pip install firebase-admin).
    A failed push is logged and dropped; it never raises into the bot.
    """

    def __init__(self, journal: Journal, *, kinds: set[str], push: bool = False,
                 credentials_file: Optional[str] = None, dedupe_seconds: float = 300.0,
                 sender: Optional[Callable[[str, str, str, str], None]] = None):
        self.journal, self.kinds, self.push = journal, kinds, push
        self._cred, self._dedupe, self._sender = credentials_file, dedupe_seconds, sender
        self._recent: dict[tuple, float] = {}
        self._fcm = None
        self._q: queue.Queue = queue.Queue(maxsize=200)
        self._thread = threading.Thread(target=self._worker, name="notifier", daemon=True)
        self._thread.start()

    def notify(self, kind: str, title: str, body: str = "", level: str = "info") -> None:
        if kind != "heartbeat" and kind not in self.kinds and level != "critical":
            return                                               # critical alerts ALWAYS go through
        key, now = (kind, title), time.monotonic()
        if kind == "error" and now - self._recent.get(key, -1e9) < self._dedupe:
            return                                               # an error loop must not spam the phone
        self._recent[key] = now
        try:
            self._q.put_nowait((kind, title, body, level))
        except queue.Full:
            log.warning("notification queue full; dropped '%s'", title)

    def flush(self, timeout: float = 5.0) -> None:
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)

    def close(self) -> None:
        self.flush(3.0)
        self._q.put(None)
        self._thread.join(2.0)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                (self._sender or self._send_fcm)(*item)
            except Exception:
                log.exception("notification failed")
            finally:
                self._q.task_done()

    def _send_fcm(self, kind: str, title: str, body: str, level: str) -> None:
        if not self.push or not self._cred:
            return
        tokens = [d["token"] for d in self.journal.devices()]
        if not tokens:
            return
        import firebase_admin                                   # lazy: only needed when push is on
        from firebase_admin import credentials, messaging
        if self._fcm is None:
            self._fcm = firebase_admin.initialize_app(credentials.Certificate(self._cred))
        msg = messaging.MulticastMessage(
            tokens=tokens, notification=messaging.Notification(title=title[:100], body=body[:300]),
            data={"kind": kind, "level": level}, android=messaging.AndroidConfig(priority="high"))
        resp = messaging.send_each_for_multicast(msg)
        for token, r in zip(tokens, resp.responses):
            if not r.success and isinstance(r.exception, messaging.UnregisteredError):
                self.journal.remove_device(token)                # the app was uninstalled


# =========================================================================== #
# 4. Runtime structures
# =========================================================================== #
@dataclass
class Venue:
    name: str
    feed: MarketFeed
    broker: Broker


@dataclass
class SymbolRuntime:
    s: SymbolSettings
    engine: Any                                  # AAVPEngine (a stub in tests)
    last_candle: Optional[datetime] = None
    next_check_at: Optional[datetime] = None


@dataclass
class Settlement:
    exit_price: Optional[float]
    pnl: float
    reason: str
    estimated: bool = False


def settle_mt5(broker: MT5Broker, position_id: str) -> Optional[Settlement]:
    """Real P&L of a closed MT5 position from the deal history (profit + commission + swap + fees)."""
    with broker.session.lock:
        m = broker.mt5
        deals = m.history_deals_get(position=int(position_id))
        if not deals:
            return None
        out_kinds = {getattr(m, n, None) for n in ("DEAL_ENTRY_OUT", "DEAL_ENTRY_INOUT", "DEAL_ENTRY_OUT_BY")}
        exits = [d for d in deals if d.entry in out_kinds]
        if not exits:
            return None                                          # still open, or history not updated yet
        last = exits[-1]
        pnl = sum(float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0.0)) for d in deals)
        reason = {getattr(m, "DEAL_REASON_SL", None): "stop_loss", getattr(m, "DEAL_REASON_TP", None): "take_profit",
                  getattr(m, "DEAL_REASON_SO", None): "stop_out"}.get(last.reason, "closed")
        return Settlement(float(last.price), round(pnl, 2), reason)


# =========================================================================== #
# 5. The Bot
# =========================================================================== #
class Bot:
    """Implements the trading loop AND the BotController contract used by the control API."""

    def __init__(self, settings: Settings, journal: Journal, guardian: Guardian, venues: dict[str, Venue],
                 symbols: list[SymbolRuntime], reviewer: Optional[GeminiReviewer] = None,
                 notifier: Optional[Notifier] = None, clock: Callable[[], datetime] = utcnow):
        self.settings, self.journal, self.guardian = settings, journal, guardian
        self.venues, self.symbols, self.reviewer = venues, symbols, reviewer
        self.notifier, self.clock = notifier, clock
        self._by_name = {rt.s.name: rt for rt in symbols}
        self._to_logical = {(rt.s.venue, rt.s.native): rt.s.name for rt in symbols}
        self._trade_lock = threading.RLock()            # one trading action at a time (loop, kill switch)
        self._params_lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = False
        self._started_at = clock()
        self._last_tick: Optional[datetime] = None
        self._specs: dict[str, SymbolSpec] = {}
        self._snapshot: Optional[list[tuple[str, Position]]] = None
        self._snapshot_at: Optional[datetime] = None
        self._pos_cache: list[dict[str, Any]] = []
        self._account: Optional[AccountInfo] = None
        self._feed_ok = {v: True for v in venues}
        self._broker_ok = {v: True for v in venues}
        self._last_profit: dict[str, float] = {}
        self._settle_attempts: dict[str, int] = {}
        self._closing: dict[str, str] = {}
        self._unprotected: dict[str, int] = {}
        self._be_attempts: dict[str, int] = {}
        self._halt_seen: Optional[str] = None
        self._errors = 0
        self._params: dict[str, Any] = {}
        self._last_equity_at = self._last_health_at = self._last_heartbeat_at = None
        self.api: Optional[ApiServer] = None

    # ---- small helpers -----------------------------------------------------------
    def alert(self, kind: str, title: str, body: str = "", level: str = "info", data: Optional[dict] = None) -> None:
        """One call = log + journal event + phone notification."""
        getattr(log, {"info": "info", "warning": "warning", "critical": "critical"}.get(level, "info"))("%s: %s %s", kind, title, body)
        try:
            self.journal.add_event(kind, f"{title} {body}".strip(), level, data, self.clock())
        except Exception:
            log.exception("could not write event")
        if self.notifier:
            self.notifier.notify(kind, title, body, level)

    def _logical(self, venue: str, p: Position) -> Position:
        return dataclasses.replace(p, symbol=self._to_logical.get((venue, p.symbol), p.symbol))

    def _state(self) -> str:
        return "halted" if self.guardian.halted else "paused" if self._paused else "running"

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    # ---- startup -------------------------------------------------------------------
    def start(self) -> None:
        """Connect, verify accounts, restore persisted state, reconcile with the brokers."""
        for name, v in self.venues.items():
            v.feed.connect()
            v.broker.connect()
            log.info("venue '%s' connected", name)
        self._assert_account_modes()
        self._restore_state()
        self._load_specs()
        self.reconcile()
        self._record_equity(self.clock(), force=True)
        self.alert("started", f"Bot started ({self.settings.mode} mode, v{VERSION})",
                   f"{len(self.symbols)} symbol(s); state={self._state()}")

    def _assert_account_modes(self) -> None:
        """Demo must never touch a real MT5 account; live must never run on a demo one."""
        v = self.venues.get("mt5")
        if v is None or not isinstance(v.broker, MT5Broker):
            return
        with v.broker.session.lock:
            m, acc = v.broker.mt5, v.broker.mt5.account_info()
            if acc is None:
                raise ConnectionLost("cannot read the MT5 account")
            real = acc.trade_mode == getattr(m, "ACCOUNT_TRADE_MODE_REAL", 2)
        if self.settings.mode == "demo" and real:
            raise ConfigError("mode is 'demo' but the MT5 account is a REAL account: refusing to start")
        if self.settings.mode == "live" and not real:
            raise ConfigError("mode is 'live' but the MT5 account is not a real account: check the login")
        if str(acc.currency).upper() not in ("USD", "USC"):
            log.warning("MT5 account currency is %s; portfolio totals assume USD-like accounts", acc.currency)

    def _restore_state(self) -> None:
        self._paused = self.journal.get_state("paused", "0") == "1"
        halted = self.journal.get_state("halted", "")
        baseline = parse_iso(self.journal.get_state("baseline_at"))
        if baseline:
            self.guardian.baseline_at = baseline
        if halted:
            self.guardian.halt(halted)
            self._halt_seen = halted
            log.warning("restored HALT from the journal: %s (clear it with: python -m src.main resume-guardian)", halted)
        saved = self.journal.get_state("aavp_params")
        if saved:
            try:
                self._apply_params(json.loads(saved), persist=False)
            except (ValueError, TypeError):
                log.exception("ignoring invalid saved AAVP parameters")

    def _load_specs(self) -> None:
        for rt in self.symbols:
            spec = self.venues[rt.s.venue].broker.get_symbol_spec(rt.s.native)
            self._specs[rt.s.name] = spec
            if hasattr(rt.engine, "cfg"):
                rt.engine.cfg.tick_size = spec.tick_size       # the live tick size beats the yaml hint

    def reconcile(self) -> dict[str, Any]:
        """Align the journal with the brokers after a start or crash. Never trades."""
        snap = self._snapshot_positions()
        if snap is None:
            raise ConnectionLost("cannot read positions from every venue; refusing to start blind")
        self._snapshot, self._snapshot_at = snap, self.clock()
        report = self.journal.reconcile([(p.id, p.symbol) for _, p in snap])
        for row in report["ghosts"]:
            log.warning("journal trade %s is open but the broker has no such position; settling", row["id"])
            self._settle(row, self.clock(), force=True)
        by_id = {p.id: (v, p) for v, p in snap}
        for stray in report["unknown"]:
            venue, p = by_id[stray["position_id"]]
            self._adopt(venue, p, self.clock())
        for o in self.journal.unresolved_orders():
            self.alert("error", f"Order {o['client_id']} has no recorded outcome",
                       f"{o['symbol']} {o['side']} status={o['status']}: check the broker manually", "warning")
        return report

    # ---- main loop -----------------------------------------------------------------
    def run(self) -> None:
        self.start()
        self._start_api()
        try:
            while not self._stop.is_set():
                began = time.monotonic()
                try:
                    self.tick()
                    self._errors = 0
                except Exception as exc:
                    self._errors += 1
                    log.exception("tick failed (%d in a row)", self._errors)
                    self.alert("error", "Internal error in trading loop", f"{type(exc).__name__}: {exc}", "warning")
                    if self._errors >= self.settings.max_internal_errors:
                        self.guardian.halt(f"{self._errors} consecutive internal errors")
                self._stop.wait(max(0.0, self.settings.loop_interval - (time.monotonic() - began)))
        finally:
            self.shutdown()

    def tick(self) -> None:
        """One pass of the loop. Public so tests (and `check`) can drive it directly."""
        now = self.clock()
        self._last_tick = now
        self._sync_halt(now)
        self._health(now)
        self.guardian.watchdog(now, all(self._feed_ok.values()), all(self._broker_ok.values()))
        self._sync_positions(now)
        self._record_equity(now)
        self._heartbeat(now)
        self._maybe_backup(now)
        self._sync_halt(now)
        for rt in self.symbols:
            try:
                self._process_symbol(rt, self.clock())
            except Exception as exc:                            # one bad symbol must not stop the others
                log.exception("processing %s failed", rt.s.name)
                self.alert("error", f"{rt.s.name}: processing failed", f"{type(exc).__name__}: {exc}", "warning")

    # ---- persistence of halts ------------------------------------------------------------
    def _persist_halt(self) -> None:
        g = self.guardian.halted
        if g and g != self._halt_seen:
            self.journal.set_state("halted", g, self.clock())
            self._halt_seen = g
            self.alert("halt", "BOT HALTED", g, "critical")

    def _sync_halt(self, now: datetime) -> None:
        """Persist new halts; honour an operator clearing the halt from the server CLI."""
        g = self.guardian.halted
        if g and g != self._halt_seen:
            self._persist_halt()
        elif g and g == self._halt_seen and not self.journal.get_state("halted", ""):
            baseline = parse_iso(self.journal.get_state("baseline_at")) or now
            self.guardian.resume(baseline)
            self._halt_seen = None
            self.alert("resumed", "Guardian resumed by operator", f"counting restarts from {iso(baseline)}", "warning")
        elif not g:
            self._halt_seen = None

    # ---- health ----------------------------------------------------------------------------
    def _health(self, now: datetime) -> None:
        if self._last_health_at and (now - self._last_health_at).total_seconds() < self.settings.health_interval:
            return
        self._last_health_at = now
        for name, v in self.venues.items():
            probe = next((rt for rt in self.symbols if rt.s.venue == name), None)
            try:
                if probe:
                    v.feed.get_quote(probe.s.native)
                self._feed_ok[name] = True
            except Exception as exc:
                self._feed_ok[name] = False
                log.warning("feed '%s' unhealthy: %s", name, exc)
            try:
                self._broker_ok[name] = bool(v.broker.is_healthy())
            except Exception:
                self._broker_ok[name] = False

    def _heartbeat(self, now: datetime) -> None:
        minutes = float(self.settings.alerts.get("heartbeat_minutes", 0) or 0)
        if not minutes or not self.settings.alerts.get("push_enabled", False):
            return
        if self._last_heartbeat_at and now - self._last_heartbeat_at < timedelta(minutes=minutes):
            return
        self._last_heartbeat_at = now
        acc = self._account
        if self.notifier:
            self.notifier.notify("heartbeat", f"Bot alive ({self._state()})",
                                 f"equity {acc.equity:.2f} | positions {len(self._pos_cache)}" if acc else "")

    def _maybe_backup(self, now: datetime) -> None:
        hours = float(self.settings.journal.get("backup_every_hours", 0) or 0)
        if hours:
            try:
                self.journal.maybe_backup(self.settings.root / "data" / "backups", hours, now)
            except Exception:
                log.exception("journal backup failed")

    # ---- accounts and equity ----------------------------------------------------------------
    def _portfolio_account(self, venue: Optional[str] = None) -> Optional[AccountInfo]:
        """All venues combined into ONE account (assumes USD-like currencies). None if any venue cannot answer."""
        accounts = []
        for name, v in self.venues.items():
            try:
                accounts.append((name, v.broker.account_info()))
            except Exception as exc:
                self._broker_ok[name] = False
                log.warning("account_info failed on '%s': %s", name, exc)
                return None
        allowed = all(a.trade_allowed for n, a in accounts if venue in (None, n))
        combined = AccountInfo("portfolio", "USD", sum(a.balance for _, a in accounts), sum(a.equity for _, a in accounts),
                               sum(a.free_margin for _, a in accounts), max(a.leverage for _, a in accounts), allowed)
        self._account = combined
        return combined

    def _record_equity(self, now: datetime, force: bool = False) -> None:
        if not force and self._last_equity_at and (now - self._last_equity_at).total_seconds() < self.settings.equity_interval:
            return
        acc = self._portfolio_account()
        if acc:
            self.journal.record_equity(acc.equity, acc.balance, now)
            self._last_equity_at = now

    # ---- positions: sync, settle, adopt, protect, manage -----------------------------------------
    def _snapshot_positions(self) -> Optional[list[tuple[str, Position]]]:
        """Positions from EVERY venue, or None if any venue failed (a partial view must never close trades)."""
        out: list[tuple[str, Position]] = []
        for name, v in self.venues.items():
            try:
                out += [(name, self._logical(name, p)) for p in v.broker.positions()]
            except Exception as exc:
                self._broker_ok[name] = False
                log.warning("positions() failed on '%s': %s", name, exc)
                return None
        return out

    def _sync_positions(self, now: datetime, force_settle: bool = False) -> None:
        with self._trade_lock:
            snap = self._snapshot_positions()
            if snap is None:
                self._snapshot = None
                return
            self._snapshot, self._snapshot_at = snap, now
            self._pos_cache = [self._position_dict(v, p) for v, p in snap]
            live = {p.id: (v, p) for v, p in snap}
            rows = {r["position_id"]: r for r in self.journal.open_trades()}
            for pid, row in rows.items():
                if pid not in live:
                    self._settle(row, now, force=force_settle)
            for pid, (venue, p) in live.items():
                if pid not in rows:
                    self._adopt(venue, p, now)
                    rows[pid] = self.journal.find_open_trade(pid) or {}
                self._watch_position(venue, p, rows.get(pid) or {}, now)

    @staticmethod
    def _position_dict(venue: str, p: Position) -> dict[str, Any]:
        return {"id": p.id, "venue": venue, "symbol": p.symbol, "side": p.side.value, "volume": p.volume,
                "entry_price": p.entry_price, "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                "profit": round(p.profit, 2), "opened_at": iso(p.opened_at) if p.opened_at else None}

    def _settle(self, row: dict[str, Any], now: datetime, force: bool = False) -> None:
        pid, broker = row["position_id"], self.venues[row["venue"]].broker
        st: Optional[Settlement] = None
        if isinstance(broker, MT5Broker):
            try:
                st = settle_mt5(broker, pid)
            except Exception:
                log.exception("could not read MT5 deal history for %s", pid)
        attempts = self._settle_attempts[pid] = self._settle_attempts.get(pid, 0) + 1
        if st is None:
            if attempts < SETTLE_RETRIES and not force:
                return                                           # give the broker's history a moment
            st = Settlement(None, round(self._last_profit.get(pid, 0.0), 2), "closed_estimated", True)
        reason = self._closing.pop(pid, None) or st.reason
        self.journal.close_trade(row["id"], exit_price=st.exit_price, pnl=st.pnl, reason=reason, exit_time=now)
        self._settle_attempts.pop(pid, None)
        self._last_profit.pop(pid, None)
        self._unprotected.pop(pid, None)
        self._be_attempts.pop(pid, None)
        note = " (P&L estimated: verify in the broker statement)" if st.estimated else ""
        self.alert("trade_closed", f"{row['symbol']} {row['side']} closed: {st.pnl:+.2f} ({reason})", note,
                   "warning" if st.pnl < 0 else "info", {"trade_id": row["id"], "pnl": st.pnl})

    def _adopt(self, venue: str, p: Position, now: datetime) -> None:
        """A position the journal does not know (crash during an order, or a manual action): start tracking it."""
        client_id, signal_id = "adopted", None
        for o in self.journal.unresolved_orders():               # was it OUR order whose outcome we lost?
            if o["client_id"] and o["client_id"] in (p.comment or "") and o["symbol"] == p.symbol:
                client_id, signal_id = o["client_id"], o["signal_id"]
                self.journal.update_order(client_id, "filled", position_id=p.id, fill_price=p.entry_price,
                                          filled_volume=p.volume, message="recovered by reconcile")
                break
        self.journal.open_trade(symbol=p.symbol, side=p.side.value, volume=p.volume, entry_price=p.entry_price or 0.0,
                                stop_loss=p.stop_loss, take_profit=p.take_profit, position_id=p.id, venue=venue,
                                client_id=client_id, signal_id=signal_id, entry_time=p.opened_at or now)
        self.alert("adopted", f"Adopted {p.symbol} {p.side.value} {p.volume}",
                   f"recovered={client_id != 'adopted'}; stop={p.stop_loss}", "warning")

    def _watch_position(self, venue: str, p: Position, row: dict[str, Any], now: datetime) -> None:
        self._last_profit[p.id] = p.profit
        m = self.settings.management
        if p.stop_loss is None:
            n = self._unprotected[p.id] = self._unprotected.get(p.id, 0) + 1
            if n >= int(m["unprotected_ticks"]):
                self._handle_unprotected(venue, p, row)
            return
        self._unprotected.pop(p.id, None)
        self._maybe_breakeven(venue, p, row)
        self._maybe_flatten(venue, p, now)

    def _handle_unprotected(self, venue: str, p: Position, row: dict[str, Any]) -> None:
        broker = self.venues[venue].broker
        stop = row.get("stop_loss")
        if stop:
            try:
                broker.modify_sltp(p.id, stop, None)
                self._unprotected.pop(p.id, None)
                self.alert("error", f"{p.symbol}: stop-loss was missing and has been restored", f"stop={stop}", "warning")
                return
            except Exception as exc:
                log.error("could not restore stop for %s: %s", p.id, exc)
        self.guardian.on_unprotected_position(p.symbol)
        self._persist_halt()
        self.alert("error", f"{p.symbol} has NO stop-loss", "repair failed; closing it" if
                   self.settings.management["close_unprotected"] else "repair failed; CLOSE IT MANUALLY", "critical")
        if self.settings.management["close_unprotected"]:
            try:
                self._closing[p.id] = "unprotected"
                broker.close_position(p.id)
            except Exception as exc:
                self.alert("error", f"{p.symbol}: emergency close FAILED", str(exc), "critical")

    def _maybe_breakeven(self, venue: str, p: Position, row: dict[str, Any]) -> None:
        m = self.settings.management
        at_r = float(m["breakeven_at_r"] or 0)
        orig_stop = row.get("stop_loss")
        if at_r <= 0 or not p.entry_price or not orig_stop or self._be_attempts.get(p.id, 0) >= 3:
            return
        long = p.side is Side.BUY
        if (p.stop_loss >= p.entry_price) if long else (p.stop_loss <= p.entry_price):
            return                                               # already at or beyond break-even
        risk = abs(p.entry_price - orig_stop)
        rt = self._by_name.get(p.symbol)
        if risk <= 0 or rt is None:
            return
        try:
            q = self.venues[venue].feed.get_quote(rt.s.native)
            price = q.bid if long else q.ask
            if ((price - p.entry_price) if long else (p.entry_price - price)) < at_r * risk:
                return
            self._be_attempts[p.id] = self._be_attempts.get(p.id, 0) + 1
            buf = float(m["breakeven_buffer_r"]) * risk
            tick = self._specs[p.symbol].tick_size
            raw = p.entry_price + buf if long else p.entry_price - buf
            steps = math.floor(raw / tick + 1e-9) if long else math.ceil(raw / tick - 1e-9)   # round AWAY from price
            new_sl = round(steps * tick, 10)
            self.venues[venue].broker.modify_sltp(p.id, new_sl, None)
            self.alert("breakeven", f"{p.symbol}: stop moved to break-even", f"new stop {new_sl}")
        except Exception as exc:
            log.warning("break-even for %s failed: %s", p.id, exc)

    def _maybe_flatten(self, venue: str, p: Position, now: datetime) -> None:
        cutoff = self.settings.management["flatten_fx_friday_utc"]
        rt = self._by_name.get(p.symbol)
        if not cutoff or rt is None or rt.s.asset_class not in FX_LIKE:
            return
        t = now.astimezone(timezone.utc)
        if t.weekday() == 4 and t.time() >= _hhmm(cutoff) and p.id not in self._closing:
            try:
                self._closing[p.id] = "weekend_flatten"
                self.venues[venue].broker.close_position(p.id)
                self.alert("flatten", f"{p.symbol}: closed before the weekend")
            except Exception as exc:
                self._closing.pop(p.id, None)
                log.warning("weekend flatten of %s failed: %s", p.id, exc)

    # ---- signal pipeline ----------------------------------------------------------------------
    def _process_symbol(self, rt: SymbolRuntime, now: datetime) -> None:
        if self._paused or self.guardian.halted:
            return
        if rt.next_check_at and now < rt.next_check_at:
            return                                               # the next candle is not due yet
        venue = self.venues[rt.s.venue]
        candles = venue.feed.fetch_candles(rt.s.native, rt.s.timeframe, self.settings.candles, include_live=True)
        last_open = candles.index[-1].to_pydatetime()
        rt.next_check_at = last_open + timedelta(seconds=TIMEFRAMES[rt.s.timeframe][0] + self.settings.signal_delay)
        if rt.last_candle is not None and last_open <= rt.last_candle:
            return                                               # no new candle yet; retry on the next tick
        rt.last_candle = last_open

        with self._params_lock:
            result = rt.engine.update(candles)
        sig = result.signal if result.ready else None
        if not sig:
            return
        key = iso(pd.Timestamp(sig["trigger_time"]).to_pydatetime())
        state_key = f"last_signal:{rt.s.name}"
        if self.journal.get_state(state_key) == key:
            return                                               # this exact signal was already handled
        self.journal.set_state(state_key, key, now)              # mark BEFORE trading: a crash may miss a trade, never double it
        self._handle_signal(rt, candles, result, now)

    def _handle_signal(self, rt: SymbolRuntime, candles: pd.DataFrame, result, now: datetime) -> None:
        venue, cfg = self.venues[rt.s.venue], rt.engine.cfg
        quote = venue.feed.get_quote(rt.s.native)
        atr_value = atr(candles, cfg.atr_period)
        pct = atr_percentile(candles, cfg.atr_period)
        trend = None
        try:
            trend = higher_tf_trend(venue.feed.fetch_candles(rt.s.native, rt.s.higher_timeframe, 200, include_live=False))
        except Exception as exc:
            log.warning("higher-timeframe data unavailable for %s: %s", rt.s.name, exc)

        proposal = build_proposal(rt.s.name, result, atr_value, bid=quote.bid, ask=quote.ask, timeframe=rt.s.timeframe,
                                  asset_class=rt.s.asset_class, tick_size=self._specs[rt.s.name].tick_size,
                                  min_rr=self.guardian.config.trade.min_risk_reward, atr_percentile=pct, htf_trend=trend)
        if proposal is None:
            self.alert("signal_skipped", f"{rt.s.name}: signal dropped", "no valid stop-loss could be placed")
            return
        signal_id = self.journal.log_signal(proposal, result.to_dict(), now)

        ctx = self._context(rt, quote, now)
        if ctx is None:
            self.journal.update_signal(signal_id, "rejected", {"reasons": ["positions or account unavailable"]})
            return

        opinion: Optional[AIOpinion] = None
        if self.guardian.ai_enabled and self.reviewer:
            # Ask the AI only if EVERY other rule already passes: a placeholder opinion stands in for the real one.
            pre = self.guardian.evaluate(proposal.with_ai(PLACEHOLDER_OPINION), ctx)
            decision = pre
            if pre.approved:
                opinion = self.reviewer.review(self._ai_context(rt, proposal, result, candles, quote, now), rt.s.name)
                quote = venue.feed.get_quote(rt.s.native)        # the AI call took time: judge on a fresh quote
                ctx = self._context(rt, quote, self.clock()) or ctx
                decision = self.guardian.evaluate(proposal.with_ai(opinion), ctx)
        else:
            decision = self.guardian.evaluate(proposal, ctx)

        self.journal.update_signal(signal_id, "approved" if decision.approved else "rejected", decision.to_dict(),
                                   opinion.to_dict() if opinion else None)
        self._persist_halt()
        if not decision.approved:
            self.alert("guardian_veto", f"{rt.s.name} {proposal.side.value} rejected", "; ".join(decision.reasons[:3]))
            return
        self._execute(rt, decision, signal_id)

    def _context(self, rt: SymbolRuntime, quote: Quote, now: datetime) -> Optional[GuardianContext]:
        fresh = self._snapshot is not None and self._snapshot_at and (now - self._snapshot_at).total_seconds() < 30
        account = self._portfolio_account(rt.s.venue)
        if not fresh or account is None:
            return None
        risk = RiskSnapshot(**self.journal.risk_inputs(now, self.guardian.baseline_at))
        return GuardianContext(account, quote, [p for _, p in self._snapshot], dict(self._specs), risk, now)

    def _ai_context(self, rt, proposal, result, candles, quote: Quote, now: datetime) -> dict[str, Any]:
        news = self.guardian.news
        currencies = self.guardian.config.news.symbol_currencies.get(rt.s.name, [])
        titles = [f"{e.currency} {e.impact}: {e.title} at {e.time:%H:%M} UTC"
                  for e in (news.upcoming(currencies, now) if news and currencies else [])]
        return build_review_context(proposal, result, candles, spread=quote.spread,
                                    sessions=self.guardian.active_sessions(now), news_titles=titles)

    # ---- execution -------------------------------------------------------------------------------
    def _fit_to_funds(self, rt: SymbolRuntime, req: OrderRequest) -> OrderRequest:
        """Spot only: never ask for more than the free USDT can buy. Shrinking only ever REDUCES risk."""
        broker = self.venues[rt.s.venue].broker
        if not (isinstance(broker, ExchangeBroker) and broker.market_type == "spot") or not req.ref_price:
            return req
        free = broker.account_info().free_margin
        affordable = self._specs[rt.s.name].normalize_volume(free * FUNDS_BUFFER / req.ref_price)
        if req.volume <= affordable:
            return req
        if affordable <= 0:
            raise OrderRejected(f"free balance {free:.2f} cannot buy the minimum size")
        log.warning("%s: size cut from %s to %s to fit free funds", rt.s.name, req.volume, affordable)
        return dataclasses.replace(req, volume=affordable)

    def _execute(self, rt: SymbolRuntime, decision, signal_id: int) -> None:
        venue, logical_req = self.venues[rt.s.venue], decision.request
        with self._trade_lock:
            if self._paused or self.guardian.halted:
                self.journal.update_signal(signal_id, "rejected", {"reasons": ["paused or halted before execution"]})
                return
            try:
                req = self._fit_to_funds(rt, dataclasses.replace(logical_req, symbol=rt.s.native))
            except OrderRejected as exc:
                self.journal.update_signal(signal_id, "error", {"reasons": [str(exc)]})
                self.alert("error", f"{rt.s.name}: order not sent", str(exc), "warning")
                return
            self.journal.log_order_intent(logical_req, signal_id, self.clock())       # write-ahead: BEFORE sending
            try:
                res = venue.broker.place_order(req)
            except UnprotectedPositionError as exc:
                self.journal.update_order(req.client_id, "error", message=str(exc))
                self.journal.update_signal(signal_id, "error", {"reasons": [str(exc)]})
                self.guardian.on_unprotected_position(rt.s.name)
                self._persist_halt()
                self.alert("error", f"{rt.s.name}: UNPROTECTED POSITION", str(exc), "critical")
                try:
                    venue.broker.close_all(rt.s.native)
                except Exception:
                    log.exception("emergency close_all failed")
                return
            except ConnectionLost as exc:
                self.journal.update_order(req.client_id, "unknown", message=str(exc))
                self.journal.update_signal(signal_id, "error", {"reasons": [str(exc)]})
                self.alert("error", f"{rt.s.name}: order outcome UNKNOWN", f"{exc}. The next sync will adopt it if it filled.", "critical")
                return
            except BrokerError as exc:
                self.journal.update_order(req.client_id, "rejected", message=str(exc))
                self.journal.update_signal(signal_id, "error", {"reasons": [str(exc)]})
                self.alert("error", f"{rt.s.name}: order rejected", str(exc), "warning")
                return

            pos_id = res.position_id or self._find_position_id(rt, req.client_id) or res.order_id
            self.journal.update_order(req.client_id, "filled", order_id=res.order_id, position_id=pos_id,
                                      fill_price=res.fill_price, filled_volume=res.filled_volume, message=res.message)
            self.journal.open_trade(symbol=rt.s.name, side=req.side.value, volume=res.filled_volume or req.volume,
                                    entry_price=res.fill_price, stop_loss=req.stop_loss, take_profit=req.take_profit,
                                    position_id=pos_id, venue=rt.s.venue, client_id=req.client_id, signal_id=signal_id,
                                    entry_time=self.clock())
            self.journal.update_signal(signal_id, "executed")
            self.alert("trade_opened", f"{rt.s.name} {req.side.value.upper()} {res.filled_volume or req.volume} @ {res.fill_price}",
                       f"SL {req.stop_loss} | TP {req.take_profit} | risk ~{decision.risk_amount:.2f}",
                       data={"signal_id": signal_id, "position_id": pos_id})
            # Refresh NOW: the next symbol in this same tick must see this position (position and risk limits).
            self._sync_positions(self.clock())

    def _find_position_id(self, rt: SymbolRuntime, client_id: str) -> str:
        try:
            for p in self.venues[rt.s.venue].broker.positions(rt.s.native):
                if client_id and client_id in (p.comment or ""):
                    return p.id
        except Exception:
            log.exception("could not look up the new position id")
        return ""

    # ---- BotController (called from the API thread) -----------------------------------------------
    def status(self) -> dict[str, Any]:
        acc = self._account
        return {"version": VERSION, "mode": self.settings.mode, "state": self._state(), "paused": self._paused,
                "halted_reason": self.guardian.halted, "started_at": iso(self._started_at),
                "uptime_seconds": int((self.clock() - self._started_at).total_seconds()),
                "last_tick": iso(self._last_tick) if self._last_tick else None,
                "venues": {n: {"feed_ok": self._feed_ok[n], "broker_ok": self._broker_ok[n]} for n in self.venues},
                "equity": round(acc.equity, 2) if acc else None, "balance": round(acc.balance, 2) if acc else None,
                "open_positions": len(self._pos_cache),
                "ai": {"enabled": bool(self.reviewer), "calls_today": self.journal.ai_calls_today(self.clock())},
                "news_calendar_ok": bool(self.guardian.news and self.guardian.news.available()),
                "symbols": [{"name": rt.s.name, "timeframe": rt.s.timeframe,
                             "last_candle": iso(rt.last_candle) if rt.last_candle else None,
                             "last_signal": self.journal.get_state(f"last_signal:{rt.s.name}")} for rt in self.symbols]}

    def positions(self) -> list[dict[str, Any]]:
        return list(self._pos_cache)

    def pnl(self) -> dict[str, Any]:
        acc = self._account
        return {**self.journal.pnl_summary(self.clock()), "unrealized": round(sum(p["profit"] for p in self._pos_cache), 2),
                "equity": round(acc.equity, 2) if acc else None}

    def recent_signals(self, limit: int) -> list[dict[str, Any]]:
        return self.journal.recent_signals(limit)

    def recent_trades(self, limit: int) -> list[dict[str, Any]]:
        return self.journal.recent_trades(limit)

    def get_params(self) -> dict[str, Any]:
        rt = self.symbols[0]
        keys = ("lookback", "anchor_mode", "value_area_pct", "atr_fraction", "lvn_ratio", "sweep_lookback", "wick_ratio", "touch_tol_atr")
        return {k: (getattr(rt.engine.cfg, k).value if k == "anchor_mode" else getattr(rt.engine.cfg, k)) for k in keys}

    def _apply_params(self, changes: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        if "lookback" in changes and changes["lookback"] > self.settings.candles - 20:
            raise ValueError(f"lookback must be at most candles_per_fetch - 20 = {self.settings.candles - 20}")
        with self._params_lock:
            for rt in self.symbols:
                for k, v in changes.items():
                    setattr(rt.engine.cfg, k, AnchorMode(v) if k == "anchor_mode" else v)
            self._params.update(changes)
            if persist:
                self.journal.set_state("aavp_params", json.dumps(self._params), self.clock())
        return self.get_params()

    def update_params(self, changes: dict[str, Any]) -> dict[str, Any]:
        applied = self._apply_params(changes)
        self.alert("params", "AAVP parameters changed from the app", json.dumps(changes), "warning")
        return applied

    def pause(self, reason: str) -> dict[str, Any]:
        self._paused = True
        self.journal.set_state("paused", "1", self.clock())
        self.alert("paused", "Trading paused", reason, "warning")
        return self.status()

    def resume(self) -> dict[str, Any]:
        if self.guardian.halted:
            raise ValueError(f"the Guardian is halted ({self.guardian.halted}); clear it on the server with: "
                             "python -m src.main resume-guardian")
        self._paused = False
        self.journal.set_state("paused", "0", self.clock())
        self.alert("resumed", "Trading resumed", "", "warning")
        return self.status()

    def kill_switch(self, reason: str) -> dict[str, Any]:
        """Stop everything NOW: pause, halt (survives restarts) and close every bot position."""
        with self._trade_lock:
            self._paused = True
            self.journal.set_state("paused", "1", self.clock())
            self.guardian.halt(f"kill switch: {reason}")
            self._persist_halt()
            closed, failed = 0, []
            if self.guardian.config.kill.close_positions_on_kill:
                for _, p in (self._snapshot_positions() or []):
                    self._closing[p.id] = "kill_switch"
                for name, v in self.venues.items():
                    try:
                        for r in v.broker.close_all():
                            closed += 1 if r.ok else 0
                            if not r.ok:
                                failed.append(f"{r.symbol}: {r.message}")
                    except Exception as exc:
                        failed.append(f"{name}: {exc}")
                self._sync_positions(self.clock(), force_settle=True)
            self.alert("kill_switch", "KILL SWITCH used", f"{reason}; closed {closed}, failed {len(failed)}", "critical")
            return {"halted": True, "closed": closed, "failed": failed}

    def register_device(self, token: str, platform: str) -> None:
        self.journal.register_device(token, platform, self.clock())

    # ---- API and shutdown ----------------------------------------------------------------------------
    def _start_api(self) -> None:
        cfg = self.settings.api
        if not cfg.get("enabled", False):
            return
        token = os.environ.get("API_TOKEN", "")
        if not token:
            log.warning("api.enabled but API_TOKEN is not set: the control API is NOT started")
            return
        app = create_app(self, token, enable_docs=bool(cfg.get("enable_docs", False)))
        self.api = ApiServer(app, host=str(cfg.get("host", "127.0.0.1")), port=int(cfg.get("port", 8000)))
        self.api.start()

    def shutdown(self) -> None:
        log.info("shutting down; open positions stay protected by their broker/exchange-side stops")
        self.alert("stopped", "Bot stopped", f"{len(self._pos_cache)} position(s) remain open with stops", "warning")
        for step in (lambda: self.api and self.api.stop(), lambda: self.notifier and self.notifier.close(),
                     *[(lambda v=v: (v.feed.close(), v.broker.close())) for v in self.venues.values()],
                     lambda: self.journal.close()):
            try:
                step()
            except Exception:
                log.exception("error during shutdown")

    def describe(self) -> list[str]:
        """Used by `check`: a human-readable readiness report."""
        acc = self._portfolio_account()
        lines = [f"mode: {self.settings.mode}   state: {self._state()}",
                 f"portfolio: equity {acc.equity:.2f} balance {acc.balance:.2f} free {acc.free_margin:.2f}" if acc else "portfolio: UNAVAILABLE"]
        for rt in self.symbols:
            sp = self._specs[rt.s.name]
            lines.append(f"{rt.s.name:<9} {rt.s.venue:<8} native={rt.s.native} tf={rt.s.timeframe}/{rt.s.higher_timeframe} "
                         f"tick={sp.tick_size} lot[{sp.volume_min}..{sp.volume_max} step {sp.volume_step}]")
        lines.append(f"open positions: {len(self._pos_cache)}   AI: {'on' if self.reviewer else 'off'}   "
                     f"news calendar: {'ok' if self.guardian.news and self.guardian.news.available() else 'MISSING'}")
        return lines


# =========================================================================== #
# 6. Wiring
# =========================================================================== #
def build_aavp_config(settings: Settings, sym: SymbolSettings) -> AAVPConfig:
    merged = {**settings.aavp, **sym.aavp}
    valid = {f.name for f in dataclasses.fields(AAVPConfig)}
    bad = set(merged) - valid
    if bad:
        raise ConfigError(f"unknown AAVP settings for {sym.name}: {sorted(bad)}")
    if "anchor_mode" in merged:
        try:
            merged["anchor_mode"] = AnchorMode(merged["anchor_mode"])
        except ValueError as exc:
            raise ConfigError(f"{sym.name}: invalid anchor_mode '{merged['anchor_mode']}'") from exc
    return AAVPConfig(**merged)


def build_bot(settings: Settings, journal: Optional[Journal] = None) -> Bot:
    """Create every real component from the settings and the environment (.env)."""
    root = settings.root
    journal = journal or Journal(root / settings.journal.get("database", "data/journal.sqlite3"))
    gcfg = load_guardian_config(root / "config" / "guardian.yaml", settings.mode)
    news = NewsCalendar(root / gcfg.news.calendar_file)
    guardian = Guardian(gcfg, ai_enabled=bool(settings.ai.get("enabled")), news=news,
                        symbol_overrides={s.name: s.guardian for s in settings.symbols if s.guardian})

    venues: dict[str, Venue] = {}
    v = settings.venues.get("mt5") or {}
    if v.get("enabled"):
        session = MT5Session(int(require_env("MT5_LOGIN")), require_env("MT5_PASSWORD"), require_env("MT5_SERVER"),
                             path=os.environ.get("MT5_PATH") or None,
                             server_utc_offset_hours=float(v.get("server_utc_offset_hours", 0)))
        venues["mt5"] = Venue("mt5", MT5Feed(session, v.get("volume_field", "tick_volume")),
                              MT5Broker(session, settings.magic, bool(v.get("require_stop_loss", True))))
    v = settings.venues.get("exchange") or {}
    if v.get("enabled"):
        mtype = v.get("market_type", "spot")
        tracked = [s.native for s in settings.symbols if s.venue == "exchange"]
        venues["exchange"] = Venue(
            "exchange",
            ExchangeFeed(v.get("id", "binance"), sandbox=bool(v.get("sandbox", True)), options={"defaultType": mtype}),
            ExchangeBroker(v.get("id", "binance"), require_env("EXCHANGE_API_KEY"), require_env("EXCHANGE_API_SECRET"),
                           os.environ.get("EXCHANGE_API_PASSWORD") or None, mtype, bool(v.get("sandbox", True)),
                           v.get("quote_currency", "USDT"), tracked, bool(v.get("require_stop_loss", True))))

    reviewer = None
    if settings.ai.get("enabled"):
        reviewer = GeminiReviewer(require_env("GEMINI_API_KEY"), settings.ai["model"],
                                  timeout_seconds=float(settings.ai.get("timeout_seconds", 20)),
                                  max_calls_per_day=int(settings.ai.get("max_calls_per_day", 200)),
                                  calls_today=journal.ai_calls_today, on_call=journal.log_ai_call)

    alerts = settings.alerts
    push = bool(alerts.get("push_enabled")) and bool(os.environ.get("FCM_CREDENTIALS_FILE"))
    if alerts.get("push_enabled") and not push:
        log.warning("alerts.push_enabled but FCM_CREDENTIALS_FILE is not set: push notifications are OFF")
    notifier = Notifier(journal, kinds=set(alerts.get("notify_on") or []), push=push,
                        credentials_file=os.environ.get("FCM_CREDENTIALS_FILE"))

    symbols = [SymbolRuntime(s, AAVPEngine(build_aavp_config(settings, s))) for s in settings.symbols]
    return Bot(settings, journal, guardian, venues, symbols, reviewer, notifier)


# =========================================================================== #
# 7. Command line
# =========================================================================== #
def cmd_resume_guardian(settings: Settings) -> int:
    """Clear a persisted Guardian halt. Run on the server; a running bot notices within one tick."""
    journal = Journal(settings.root / settings.journal.get("database", "data/journal.sqlite3"))
    try:
        halted = journal.get_state("halted", "")
        if not halted:
            print("The Guardian is not halted.")
            return 0
        print(f"Guardian is halted: {halted}")
        if input("Type RESUME to clear the halt and restart loss/drawdown counting from now: ").strip() != "RESUME":
            print("Cancelled.")
            return 1
        now = utcnow()
        journal.set_state("baseline_at", iso(now), now)
        journal.set_state("halted", "", now)
        journal.add_event("resume_requested", f"operator cleared halt: {halted}", "warning", None, now)
        print("Done. A running bot resumes within a few seconds; otherwise it starts clean.")
        return 0
    finally:
        journal.close()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.main", description="CMAN trading bot")
    p.add_argument("command", nargs="?", default="run", choices=["run", "check", "resume-guardian"])
    p.add_argument("--config-dir", default="config")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--confirm-live", action="store_true", help="required to run in live mode")
    args = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    except ImportError:
        pass
    try:
        settings = load_settings(args.config_dir)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    setup_logging(settings.log_level, settings.root / settings.log_file)

    if args.command == "resume-guardian":
        return cmd_resume_guardian(settings)
    if settings.mode == "live" and not (args.confirm_live or os.environ.get("CMAN_CONFIRM_LIVE") == "YES"):
        print("Live mode trades REAL money. Re-run with --confirm-live (or CMAN_CONFIRM_LIVE=YES) if you mean it.", file=sys.stderr)
        return 2

    try:
        bot = build_bot(settings)
    except (ConfigError, RuntimeError, ValueError) as exc:
        print(f"STARTUP ERROR: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        try:
            bot.start()
            print("\n".join(bot.describe()))
            return 0
        except (BrokerError, ConfigError) as exc:
            print(f"CHECK FAILED: {exc}", file=sys.stderr)
            return 1
        finally:
            bot.shutdown()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):             # SIGBREAK exists on Windows only
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: bot.request_stop())
    try:
        bot.run()
    except (BrokerError, ConfigError) as exc:
        log.critical("startup failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())