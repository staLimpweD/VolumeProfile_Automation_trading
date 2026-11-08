"""
src/journal/db.py - the bot's memory: an SQLite journal of everything that matters.

Stores signals, Guardian decisions, every AI call (prompt + answer), order attempts,
trades, equity snapshots, events and the registered phone tokens. Three jobs:
  1. AUDIT   - reconstruct exactly why any trade was (or was not) taken
  2. SAFETY  - risk inputs for the Guardian, and reconciliation with the broker after a restart
  3. LEARNING - the data you review to improve the strategy

Design notes
  * one connection guarded by a lock (safe to share between the trading loop and the API thread)
  * WAL mode + synchronous=FULL: a power cut must not lose a trade record
  * an order's intent is written BEFORE it is sent (write-ahead), so a crash mid-order is visible
  * all times are stored as ISO-8601 UTC text, which also sorts correctly as text
  * schema changes go through the MIGRATIONS list (tracked with PRAGMA user_version)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.utils import utcnow

log = logging.getLogger(__name__)

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT,
        direction TEXT NOT NULL, level_name TEXT, level_price REAL, entry REAL, stop_loss REAL, take_profit REAL,
        status TEXT NOT NULL DEFAULT 'new',          -- new | approved | rejected | executed | error
        reasons TEXT, aavp_json TEXT, ai_json TEXT, decision_json TEXT);
    CREATE INDEX idx_signals_ts ON signals(ts);

    CREATE TABLE ai_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT, model TEXT, ok INTEGER NOT NULL,
        latency_ms INTEGER, tokens INTEGER, error TEXT, prompt TEXT, response TEXT);
    CREATE INDEX idx_ai_calls_ts ON ai_calls(ts);

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, client_id TEXT NOT NULL UNIQUE, signal_id INTEGER,
        symbol TEXT NOT NULL, side TEXT NOT NULL, volume REAL, stop_loss REAL, take_profit REAL, ref_price REAL,
        status TEXT NOT NULL,                        -- sent | filled | rejected | error | unknown
        order_id TEXT, position_id TEXT, fill_price REAL, filled_volume REAL, message TEXT);

    CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, signal_id INTEGER, symbol TEXT NOT NULL,
        side TEXT NOT NULL, volume REAL NOT NULL, entry_price REAL, entry_time TEXT NOT NULL,
        stop_loss REAL, take_profit REAL, position_id TEXT, venue TEXT,
        status TEXT NOT NULL DEFAULT 'open',         -- open | closed
        exit_price REAL, exit_time TEXT, pnl REAL, exit_reason TEXT);
    CREATE INDEX idx_trades_status ON trades(status);
    CREATE INDEX idx_trades_exit ON trades(exit_time);

    CREATE TABLE equity (ts TEXT PRIMARY KEY, equity REAL NOT NULL, balance REAL);

    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL,
        message TEXT, data TEXT);
    CREATE INDEX idx_events_ts ON events(ts);

    CREATE TABLE devices (token TEXT PRIMARY KEY, platform TEXT, registered_at TEXT NOT NULL);

    CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT NOT NULL);
    """),
]


def iso(dt: Optional[datetime] = None) -> str:
    """UTC ISO-8601 with seconds precision, e.g. 2026-11-08T14:05:09+00:00."""
    dt = dt or utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(text: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(text) if text else None


def start_of_day(now: datetime) -> datetime:
    now = now.astimezone(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def start_of_week(now: datetime) -> datetime:
    """Monday 00:00 UTC."""
    return start_of_day(now) - timedelta(days=now.astimezone(timezone.utc).weekday())


def _dumps(obj: Any) -> Optional[str]:
    return None if obj is None else json.dumps(obj, default=str, separators=(",", ":"))


class Journal:
    def __init__(self, path: str | Path = "data/journal.sqlite3"):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)  # we manage transactions
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # ---- plumbing ---------------------------------------------------------------------
    def _migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for version, sql in MIGRATIONS:
                if version > current:
                    self._conn.executescript("BEGIN;" + sql + f"PRAGMA user_version={version};COMMIT;")
                    log.info("journal migrated to schema v%d", version)

    def _run(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(sql, tuple(params))
                self._conn.execute("COMMIT")
                return cur
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def _one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict[str, Any]]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- signals ----------------------------------------------------------------------------
    def log_signal(self, proposal, aavp: Optional[dict] = None, ts: Optional[datetime] = None) -> int:
        cur = self._run(
            "INSERT INTO signals(ts,symbol,timeframe,direction,level_name,level_price,entry,stop_loss,take_profit,aavp_json)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (iso(ts), proposal.symbol, proposal.timeframe, proposal.side.value, proposal.level_name,
             proposal.level_price, proposal.entry, proposal.stop_loss, proposal.take_profit, _dumps(aavp)))
        return int(cur.lastrowid)

    def update_signal(self, signal_id: int, status: str, decision: Optional[dict] = None,
                      ai: Optional[dict] = None) -> None:
        reasons = "; ".join(decision.get("reasons", [])) if decision else None
        self._run("UPDATE signals SET status=?, reasons=COALESCE(?,reasons), decision_json=COALESCE(?,decision_json),"
                  " ai_json=COALESCE(?,ai_json) WHERE id=?", (status, reasons, _dumps(decision), _dumps(ai), signal_id))

    def recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._all("SELECT id,ts,symbol,timeframe,direction,level_name,level_price,entry,stop_loss,take_profit,"
                         "status,reasons FROM signals ORDER BY id DESC LIMIT ?", (limit,))

    # ---- AI calls -------------------------------------------------------------------------
    def log_ai_call(self, call: dict[str, Any], ts: Optional[datetime] = None) -> None:
        """Accepts the dict GeminiReviewer passes to its on_call hook."""
        self._run("INSERT INTO ai_calls(ts,symbol,model,ok,latency_ms,tokens,error,prompt,response) VALUES(?,?,?,?,?,?,?,?,?)",
                  (iso(ts), call.get("symbol"), call.get("model"), int(bool(call.get("ok"))), call.get("latency_ms"),
                   call.get("tokens"), call.get("error"), call.get("prompt"), call.get("response")))

    def ai_calls_today(self, now: Optional[datetime] = None) -> int:
        since = iso(start_of_day(now or utcnow()))
        return int(self._one("SELECT COUNT(*) AS n FROM ai_calls WHERE ts>=?", (since,))["n"])

    def prune_ai_text(self, older_than_days: int = 30, now: Optional[datetime] = None) -> int:
        """Keep the call record but drop bulky prompt/response text from old rows."""
        cutoff = iso((now or utcnow()) - timedelta(days=older_than_days))
        return self._run("UPDATE ai_calls SET prompt=NULL, response=NULL WHERE ts<? AND (prompt IS NOT NULL OR response IS NOT NULL)",
                         (cutoff,)).rowcount

    # ---- orders (write-ahead) ----------------------------------------------------------
    def log_order_intent(self, req, signal_id: Optional[int] = None, ts: Optional[datetime] = None) -> int:
        """Call BEFORE sending the order. If the bot crashes mid-send, a row with status 'sent' remains."""
        cur = self._run(
            "INSERT INTO orders(ts,client_id,signal_id,symbol,side,volume,stop_loss,take_profit,ref_price,status)"
            " VALUES(?,?,?,?,?,?,?,?,?,'sent')",
            (iso(ts), req.client_id, signal_id, req.symbol, req.side.value, req.volume, req.stop_loss,
             req.take_profit, req.ref_price))
        return int(cur.lastrowid)

    def update_order(self, client_id: str, status: str, *, order_id: str = "", position_id: str = "",
                     fill_price: Optional[float] = None, filled_volume: Optional[float] = None, message: str = "") -> None:
        self._run("UPDATE orders SET status=?, order_id=?, position_id=?, fill_price=?, filled_volume=?, message=? "
                  "WHERE client_id=?", (status, order_id, position_id, fill_price, filled_volume, message[:500], client_id))

    def unresolved_orders(self) -> list[dict[str, Any]]:
        """Orders still marked 'sent' or 'unknown': investigate these after any crash or restart."""
        return self._all("SELECT * FROM orders WHERE status IN ('sent','unknown') ORDER BY id")

    # ---- trades -----------------------------------------------------------------------------
    def open_trade(self, *, symbol: str, side: str, volume: float, entry_price: float, stop_loss: Optional[float],
                   take_profit: Optional[float], position_id: str, venue: str, client_id: str = "",
                   signal_id: Optional[int] = None, entry_time: Optional[datetime] = None) -> int:
        cur = self._run(
            "INSERT INTO trades(client_id,signal_id,symbol,side,volume,entry_price,entry_time,stop_loss,take_profit,position_id,venue)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (client_id, signal_id, symbol, side, volume, entry_price, iso(entry_time), stop_loss, take_profit, position_id, venue))
        return int(cur.lastrowid)

    def update_trade_levels(self, trade_id: int, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        self._run("UPDATE trades SET stop_loss=COALESCE(?,stop_loss), take_profit=COALESCE(?,take_profit) WHERE id=?",
                  (stop_loss, take_profit, trade_id))

    def close_trade(self, trade_id: int, *, exit_price: Optional[float], pnl: float, reason: str,
                    exit_time: Optional[datetime] = None) -> None:
        self._run("UPDATE trades SET status='closed', exit_price=?, exit_time=?, pnl=?, exit_reason=? "
                  "WHERE id=? AND status='open'", (exit_price, iso(exit_time), pnl, reason, trade_id))

    def open_trades(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM trades WHERE status='open' ORDER BY id")

    def find_open_trade(self, position_id: str) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM trades WHERE status='open' AND position_id=? ORDER BY id DESC LIMIT 1", (position_id,))

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))

    def reconcile(self, live_positions: Iterable[tuple[str, str]]) -> dict[str, list]:
        """
        Compare the journal with what the broker really holds. `live_positions` = (position_id, symbol).
          ghosts  : journal says open, broker has nothing  -> it closed while we were away; settle it
          unknown : broker holds it, journal has no record -> manual trade or a lost record; investigate
        """
        live = {pid: sym for pid, sym in live_positions}
        open_rows = self.open_trades()
        known = {r["position_id"] for r in open_rows}
        return {"ghosts": [r for r in open_rows if r["position_id"] not in live],
                "unknown": [{"position_id": p, "symbol": s} for p, s in live.items() if p not in known]}

    # ---- equity and risk inputs -----------------------------------------------------------
    def record_equity(self, equity: float, balance: Optional[float] = None, ts: Optional[datetime] = None) -> None:
        self._run("INSERT OR REPLACE INTO equity(ts,equity,balance) VALUES(?,?,?)", (iso(ts), equity, balance))

    def equity_peak(self, since: Optional[datetime] = None) -> Optional[float]:
        row = self._one("SELECT MAX(equity) AS m FROM equity WHERE ts>=?", (iso(since) if since else "",))
        return row["m"] if row else None

    def equity_since(self, boundary: datetime) -> Optional[float]:
        """First equity snapshot at or after `boundary` (start of the day/week as seen by the bot)."""
        row = self._one("SELECT equity FROM equity WHERE ts>=? ORDER BY ts LIMIT 1", (iso(boundary),))
        return row["equity"] if row else None

    def realized_pnl_since(self, since: datetime) -> float:
        row = self._one("SELECT COALESCE(SUM(pnl),0) AS p FROM trades WHERE status='closed' AND exit_time>=?", (iso(since),))
        return float(row["p"])

    def consecutive_losses(self, since: Optional[datetime] = None) -> int:
        """Current streak of losing closed trades (a win or break-even ends it). Only trades closed at/after `since`."""
        rows = self._all("SELECT pnl FROM trades WHERE status='closed' AND exit_time>=? ORDER BY exit_time DESC, id DESC",
                         (iso(since) if since else "",))
        streak = 0
        for r in rows:
            if (r["pnl"] or 0.0) < 0:
                streak += 1
            else:
                break
        return streak

    def last_loss_times(self) -> dict[str, datetime]:
        rows = self._all("SELECT symbol, MAX(exit_time) AS t FROM trades WHERE status='closed' AND pnl<0 GROUP BY symbol")
        return {r["symbol"]: parse_iso(r["t"]) for r in rows}

    def risk_inputs(self, now: Optional[datetime] = None, baseline_at: Optional[datetime] = None) -> dict[str, Any]:
        """Everything guardian.RiskSnapshot needs:  RiskSnapshot(**journal.risk_inputs(now, guardian.baseline_at))."""
        now = now or utcnow()
        return {"equity_start_of_day": self.equity_since(start_of_day(now)),
                "equity_start_of_week": self.equity_since(start_of_week(now)),
                "equity_peak": self.equity_peak(baseline_at),
                "consecutive_losses": self.consecutive_losses(baseline_at),
                "last_loss_by_symbol": self.last_loss_times()}

    def pnl_summary(self, now: Optional[datetime] = None) -> dict[str, Any]:
        now = now or utcnow()
        closed = self._all("SELECT pnl FROM trades WHERE status='closed' AND pnl IS NOT NULL")
        wins = [r["pnl"] for r in closed if r["pnl"] > 0]
        losses = [r["pnl"] for r in closed if r["pnl"] < 0]
        n = len(closed)
        return {"today": round(self.realized_pnl_since(start_of_day(now)), 2),
                "this_week": round(self.realized_pnl_since(start_of_week(now)), 2),
                "total": round(sum(r["pnl"] for r in closed), 2), "closed_trades": n,
                "wins": len(wins), "losses": len(losses),
                "win_rate": round(len(wins) / n, 3) if n else None,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
                "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else None}

    # ---- events, state, devices ---------------------------------------------------------
    def add_event(self, kind: str, message: str = "", level: str = "info", data: Optional[dict] = None,
                  ts: Optional[datetime] = None) -> None:
        self._run("INSERT INTO events(ts,level,kind,message,data) VALUES(?,?,?,?,?)",
                  (iso(ts), level, kind, message[:1000], _dumps(data)))

    def recent_events(self, limit: int = 100, level: Optional[str] = None) -> list[dict[str, Any]]:
        if level:
            return self._all("SELECT * FROM events WHERE level=? ORDER BY id DESC LIMIT ?", (level, limit))
        return self._all("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._one("SELECT value FROM state WHERE key=?", (key,))
        return row["value"] if row else default

    def set_state(self, key: str, value: str, ts: Optional[datetime] = None) -> None:
        self._run("INSERT INTO state(key,value,updated_at) VALUES(?,?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at", (key, value, iso(ts)))

    def register_device(self, token: str, platform: str = "android", ts: Optional[datetime] = None) -> None:
        self._run("INSERT INTO devices(token,platform,registered_at) VALUES(?,?,?) "
                  "ON CONFLICT(token) DO UPDATE SET platform=excluded.platform", (token, platform, iso(ts)))

    def devices(self) -> list[dict[str, Any]]:
        return self._all("SELECT token, platform, registered_at FROM devices")

    def remove_device(self, token: str) -> None:
        self._run("DELETE FROM devices WHERE token=?", (token,))

    # ---- backups -------------------------------------------------------------------------
    def backup(self, dest_dir: str | Path, now: Optional[datetime] = None) -> Path:
        """Consistent online copy using SQLite's backup API (safe while the bot is running)."""
        now = now or utcnow()
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"journal-{now:%Y%m%d-%H%M%S}.sqlite3"
        with self._lock:
            out = sqlite3.connect(str(target))
            try:
                self._conn.backup(out)
            finally:
                out.close()
        self.set_state("last_backup", iso(now), now)
        return target

    def maybe_backup(self, dest_dir: str | Path, every_hours: float, now: Optional[datetime] = None,
                     keep: int = 14) -> Optional[Path]:
        now = now or utcnow()
        last = parse_iso(self.get_state("last_backup"))
        if last and now - last < timedelta(hours=every_hours):
            return None
        path = self.backup(dest_dir, now)
        for old in sorted(Path(dest_dir).glob("journal-*.sqlite3"))[:-keep]:      # keep the newest `keep` copies
            old.unlink(missing_ok=True)
        return path