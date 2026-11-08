"""
src/api/server.py - the "remote control" for your Android app.

Design rules
  * The API NEVER trades and never touches API keys. It only calls a BotController
    (implemented by main.py), which owns the bot's state.
  * It can read status/P&L, pause, resume, hit the kill switch, tweak whitelisted AAVP
    parameters and register a push-notification token.
  * It can NOT change Guardian risk limits. Those are edited on the server only.
  * Every request needs a long secret bearer token. Repeated failures lock the caller out.
  * Bind to 127.0.0.1 and put HTTPS in front (Caddy / reverse proxy / tunnel). Never expose
    this over plain HTTP on the internet.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Literal, Optional, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("cman.api")
audit = logging.getLogger("cman.audit")

MIN_TOKEN_LENGTH = 32


# --------------------------------------------------------------------------- #
# Contract with the bot (main.py implements this)
# --------------------------------------------------------------------------- #
class BotController(Protocol):
    def status(self) -> dict[str, Any]: ...
    def positions(self) -> list[dict[str, Any]]: ...
    def pnl(self) -> dict[str, Any]: ...
    def recent_signals(self, limit: int) -> list[dict[str, Any]]: ...
    def recent_trades(self, limit: int) -> list[dict[str, Any]]: ...
    def get_params(self) -> dict[str, Any]: ...
    def update_params(self, changes: dict[str, Any]) -> dict[str, Any]: ...
    def pause(self, reason: str) -> dict[str, Any]: ...
    def resume(self) -> dict[str, Any]: ...
    def kill_switch(self, reason: str) -> dict[str, Any]: ...
    def register_device(self, token: str, platform: str) -> None: ...


# --------------------------------------------------------------------------- #
# Request models (strict: unknown fields are rejected)
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParamsUpdate(_Strict):
    """Only these AAVP knobs may be changed remotely, each within sane bounds."""
    lookback: Optional[int] = Field(None, ge=20, le=1000)
    anchor_mode: Optional[Literal["highest_high", "lowest_low", "highest_volume"]] = None
    value_area_pct: Optional[float] = Field(None, ge=0.50, le=0.95)
    atr_fraction: Optional[float] = Field(None, ge=0.02, le=0.50)
    lvn_ratio: Optional[float] = Field(None, ge=0.10, le=0.90)
    sweep_lookback: Optional[int] = Field(None, ge=5, le=100)
    wick_ratio: Optional[float] = Field(None, ge=0.20, le=0.90)
    touch_tol_atr: Optional[float] = Field(None, ge=0.05, le=1.00)


class PauseRequest(_Strict):
    reason: str = Field("manual", max_length=100)


class KillRequest(_Strict):
    confirm: Literal["KILL"]                      # typed confirmation prevents accidental taps
    reason: str = Field("manual kill switch", max_length=100)


class DeviceRegistration(_Strict):
    token: str = Field(..., min_length=20, max_length=4096)
    platform: Literal["android"] = "android"


# --------------------------------------------------------------------------- #
# Throttling
# --------------------------------------------------------------------------- #
class _Window:
    """Counts events per key inside a sliding time window."""
    def __init__(self, limit: int, seconds: float):
        self.limit, self.seconds = limit, seconds
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: deque, now: float) -> None:
        while q and now - q[0] > self.seconds:
            q.popleft()

    def hit(self, key: str) -> bool:
        """Record an event; True if still within the limit."""
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._trim(q, now)
            q.append(now)
            return len(q) <= self.limit

    def blocked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._trim(q, now)
            return len(q) >= self.limit


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(controller: BotController, api_token: str, *, enable_docs: bool = False,
               max_requests_per_minute: int = 120, max_auth_failures: int = 10) -> FastAPI:
    if len(api_token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"API token must be at least {MIN_TOKEN_LENGTH} characters "
                         "(generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
    token_bytes = api_token.encode()
    bearer = HTTPBearer(auto_error=False)
    requests_window = _Window(max_requests_per_minute, 60)
    failures_window = _Window(max_auth_failures, 15 * 60)

    app = FastAPI(
        title="CMAN Trading Bot Control API", version="1.0.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url=None, openapi_url="/openapi.json" if enable_docs else None)

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def require_auth(request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> str:
        ip = client_ip(request)
        if failures_window.blocked(ip):
            raise HTTPException(429, "too many failed attempts; try again later")
        supplied = creds.credentials.encode() if creds else b""
        if not hmac.compare_digest(supplied, token_bytes):       # constant-time comparison
            failures_window.hit(ip)
            audit.warning("auth failure from %s on %s", ip, request.url.path)
            raise HTTPException(401, "invalid credentials", headers={"WWW-Authenticate": "Bearer"})
        if not requests_window.hit(ip):
            raise HTTPException(429, "rate limit exceeded")
        return ip

    def guarded(call, *args):
        """Translate controller errors into clean HTTP errors."""
        try:
            return call(*args)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ---- public ---------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness only. Reveals nothing about the bot."""
        return {"status": "ok"}

    # ---- read-only ------------------------------------------------------------------
    @app.get("/v1/status")
    def status(_: str = Depends(require_auth)) -> dict[str, Any]:
        return guarded(controller.status)

    @app.get("/v1/positions")
    def positions(_: str = Depends(require_auth)) -> list[dict[str, Any]]:
        return guarded(controller.positions)

    @app.get("/v1/pnl")
    def pnl(_: str = Depends(require_auth)) -> dict[str, Any]:
        return guarded(controller.pnl)

    @app.get("/v1/signals")
    def signals(limit: int = Query(50, ge=1, le=500), _: str = Depends(require_auth)) -> list[dict[str, Any]]:
        return guarded(controller.recent_signals, limit)

    @app.get("/v1/trades")
    def trades(limit: int = Query(50, ge=1, le=500), _: str = Depends(require_auth)) -> list[dict[str, Any]]:
        return guarded(controller.recent_trades, limit)

    @app.get("/v1/params")
    def get_params(_: str = Depends(require_auth)) -> dict[str, Any]:
        return guarded(controller.get_params)

    # ---- control ---------------------------------------------------------------------
    @app.patch("/v1/params")
    def patch_params(body: ParamsUpdate, ip: str = Depends(require_auth)) -> dict[str, Any]:
        changes = body.model_dump(exclude_none=True)
        if not changes:
            raise HTTPException(400, "no parameters supplied")
        audit.warning("params change from %s: %s", ip, changes)
        return guarded(controller.update_params, changes)

    @app.post("/v1/control/pause")
    def pause(body: PauseRequest, ip: str = Depends(require_auth)) -> dict[str, Any]:
        audit.warning("PAUSE from %s (%s)", ip, body.reason)
        return guarded(controller.pause, body.reason)

    @app.post("/v1/control/resume")
    def resume(ip: str = Depends(require_auth)) -> dict[str, Any]:
        audit.warning("RESUME from %s", ip)
        return guarded(controller.resume)

    @app.post("/v1/control/kill")
    def kill(body: KillRequest, ip: str = Depends(require_auth)) -> dict[str, Any]:
        """Emergency stop: halts trading and closes all bot positions (implemented by the controller)."""
        audit.critical("KILL SWITCH from %s (%s)", ip, body.reason)
        return guarded(controller.kill_switch, body.reason)

    @app.post("/v1/devices")
    def register_device(body: DeviceRegistration, ip: str = Depends(require_auth)) -> dict[str, str]:
        guarded(controller.register_device, body.token, body.platform)
        audit.info("device registered from %s", ip)
        return {"status": "registered"}

    return app


# --------------------------------------------------------------------------- #
# Background runner
# --------------------------------------------------------------------------- #
class ApiServer:
    """Runs uvicorn in a background thread so the trading loop keeps the main thread."""

    def __init__(self, app: FastAPI, host: str = "127.0.0.1", port: int = 8000,
                 ssl_certfile: Optional[str] = None, ssl_keyfile: Optional[str] = None):
        config = uvicorn.Config(app, host=host, port=port, log_level="info",
                                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None      # signals belong to the main thread
        self._thread = threading.Thread(target=self._server.run, name="api-server", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("control API started")

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout)