"""
src/data/mt5_session.py - one shared connection to the MetaTrader 5 terminal.

The MetaTrader5 Python package is a process-wide singleton: one initialize() per
process, one terminal. Both the feed and the broker therefore share ONE session,
and every call into MT5 is serialised through `session.lock`.

Windows only (the package talks to a terminal installed on the same machine).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.models import ConnectionLost

log = logging.getLogger(__name__)


class MT5Session:
    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None,
                 timeout_ms: int = 60_000, server_utc_offset_hours: float = 0.0,
                 mt5_module=None):
        """
        server_utc_offset_hours: MT5 reports times in the BROKER SERVER's clock, labelled as
        if it were UTC. Set this to the server's offset (e.g. 2 or 3 for many brokers) so
        candles line up with real UTC. Check it by comparing a tick time with utcnow().
        mt5_module: inject a fake in tests; None imports the real package.
        """
        self._login, self._password, self._server = int(login), password, server
        self._path, self._timeout = path, timeout_ms
        self.server_utc_offset_hours = server_utc_offset_hours
        self._mt5 = mt5_module
        self.lock = threading.RLock()

    # ---- module access ------------------------------------------------------
    @property
    def mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5          # imported lazily so the code loads on Linux/macOS
            except ImportError as exc:
                raise ConnectionLost("MetaTrader5 package not available (Windows + `pip install MetaTrader5`)") from exc
            self._mt5 = mt5
        return self._mt5

    # ---- lifecycle ------------------------------------------------------------
    def connect(self) -> None:
        with self.lock:
            kwargs = dict(login=self._login, password=self._password, server=self._server, timeout=self._timeout)
            if self._path:
                kwargs["path"] = self._path
            if not self.mt5.initialize(**kwargs):
                raise ConnectionLost(f"MT5 initialize failed: {self.mt5.last_error()}")
            log.info("MT5 connected: login=%s server=%s", self._login, self._server)

    def shutdown(self) -> None:
        with self.lock:
            try:
                self.mt5.shutdown()
            except Exception:                       # shutting down must never raise
                log.exception("MT5 shutdown error")

    def is_connected(self) -> bool:
        with self.lock:
            info = self.mt5.terminal_info()
            return bool(info is not None and info.connected)

    def ensure(self) -> None:
        """Reconnect if the terminal link dropped. Call at the start of every operation."""
        with self.lock:
            if not self.is_connected():
                log.warning("MT5 link lost, reconnecting")
                self.connect()
                if not self.is_connected():
                    raise ConnectionLost("MT5 terminal not connected to the broker server")

    # ---- time -----------------------------------------------------------------
    def to_utc(self, epoch_seconds: float) -> datetime:
        """Convert an MT5 timestamp (server clock) to timezone-aware UTC."""
        naive = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        return naive - timedelta(hours=self.server_utc_offset_hours)