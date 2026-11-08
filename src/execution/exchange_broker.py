"""
src/execution/exchange_broker.py - crypto execution through ccxt (Binance, Bybit, ...).

Modes
  spot : long-only. A SELL is only allowed to close an existing holding.
  swap : perpetual futures; long and short, positions come from the exchange.

Safety rules built in
  * a stop-loss is mandatory and is placed ON THE EXCHANGE right after the fill
  * if the stop cannot be placed, the position is closed immediately; if even that fails,
    UnprotectedPositionError is raised so the main loop can alert and halt
  * entry orders are never blindly retried; after a network error we look the order up by
    client id first
  * API keys must NOT have withdrawal permission (enforce that on the exchange itself)

Entry prices of spot holdings are remembered in memory only. After a restart a spot
position reports entry_price=None until the journal reconciles it.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

import ccxt

from src.data.exchange_feed import build_exchange
from src.models import (AccountInfo, Broker, BrokerError, ConnectionLost, OrderRejected,
                        OrderRequest, OrderResult, Position, Side, SymbolSpec,
                        UnprotectedPositionError, check_slippage, validate_levels)
from src.utils import retry

log = logging.getLogger(__name__)
NET = (ccxt.NetworkError,)                      # includes timeouts, rate limits, exchange-unavailable


def _step(precision, mode, ex) -> float:
    """Convert ccxt's precision value to a step size, whatever precision mode the exchange uses."""
    if not precision:
        return 0.0
    if mode == getattr(ex, "TICK_SIZE", 4):
        return float(precision)
    if mode == getattr(ex, "DECIMAL_PLACES", 2):
        return 10.0 ** (-int(precision))
    return 0.0


class ExchangeBroker(Broker):
    def __init__(self, exchange_id: str = "binance", api_key: str = "", secret: str = "",
                 password: Optional[str] = None, market_type: str = "spot", sandbox: bool = True,
                 quote_currency: str = "USDT", tracked_symbols: Optional[list[str]] = None,
                 require_stop_loss: bool = True, dust_value: float = 5.0,
                 options: Optional[dict] = None, exchange=None):
        if market_type not in ("spot", "swap"):
            raise ValueError("market_type must be 'spot' or 'swap'")
        opts = {"defaultType": market_type, **(options or {})}
        self.ex = exchange or build_exchange(exchange_id, api_key, secret, password, sandbox, opts)
        self.market_type, self.quote = market_type, quote_currency
        self.tracked = list(tracked_symbols or [])       # spot: only these coins count as bot positions
        self.require_stop_loss = require_stop_loss
        self.dust_value = dust_value
        self._entries: dict[str, float] = {}             # symbol -> entry price (memory only)
        self._levels: dict[str, tuple] = {}              # symbol -> (stop, take_profit), used to re-protect

    # ---- lifecycle ---------------------------------------------------------------
    @retry(attempts=3, exceptions=NET)
    def connect(self) -> None:
        if not getattr(self.ex, "apiKey", None) or not getattr(self.ex, "secret", None):
            raise ConnectionLost("exchange API key/secret missing")
        self.ex.load_markets()
        self.account_info()                              # fails fast on bad keys / permissions

    def close(self) -> None:
        closer = getattr(self.ex, "close", None)
        if callable(closer):
            closer()

    def is_healthy(self) -> bool:
        try:
            self.ex.fetch_time()
            return True
        except Exception:
            log.exception("exchange health check failed")
            return False

    # ---- account and symbol --------------------------------------------------------
    @retry(attempts=3, exceptions=NET)
    def account_info(self) -> AccountInfo:
        bal = self.ex.fetch_balance()
        total = float((bal.get("total") or {}).get(self.quote, 0.0) or 0.0)
        free = float((bal.get("free") or {}).get(self.quote, 0.0) or 0.0)
        equity = total
        if self.market_type == "spot":                   # add the value of coins we hold
            for p in self.positions():
                equity += p.volume * self._last(p.symbol)
        return AccountInfo(str(getattr(self.ex, "id", "exchange")), self.quote, total, equity, free, 1.0, True)

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        try:
            mk = self.ex.market(symbol)
        except ccxt.BadSymbol as exc:
            raise BrokerError(f"unknown symbol '{symbol}'") from exc
        prec, mode = mk.get("precision", {}), getattr(self.ex, "precisionMode", None)
        limits = mk.get("limits", {})
        tick = _step(prec.get("price"), mode, self.ex) or 0.01
        contract = float(mk.get("contractSize") or 1.0)
        vstep = _step(prec.get("amount"), mode, self.ex) or 1e-6
        amt = limits.get("amount") or {}
        return SymbolSpec(symbol, tick, tick * contract, contract, float(amt.get("min") or vstep),
                          float(amt.get("max") or 1e12), vstep,
                          max(0, -int(round(math.log10(tick)))), 0.0)

    def _last(self, symbol: str) -> float:
        return float(self.ex.fetch_ticker(symbol)["last"])

    # ---- positions -------------------------------------------------------------------
    def _protective(self, symbol: str) -> list[dict]:
        """Open reduce-side trigger orders (our stop-loss / take-profit) for a symbol."""
        out = []
        for o in self.ex.fetch_open_orders(symbol):
            trigger = o.get("triggerPrice") or o.get("stopPrice") or o.get("stopLossPrice") or o.get("takeProfitPrice")
            if trigger:
                out.append(o)
        return out

    def _split_levels(self, orders: list[dict], side: Side, ref: float):
        sl = tp = None
        for o in orders:
            trig = float(o.get("triggerPrice") or o.get("stopPrice") or o.get("stopLossPrice") or o.get("takeProfitPrice"))
            below = trig < ref
            if (side is Side.BUY and below) or (side is Side.SELL and not below):
                sl = trig
            else:
                tp = trig
        return sl, tp

    @retry(attempts=3, exceptions=NET)
    def positions(self, symbol: Optional[str] = None) -> list[Position]:
        out: list[Position] = []
        if self.market_type == "swap":
            for p in self.ex.fetch_positions([symbol] if symbol else (self.tracked or None)):
                size = abs(float(p.get("contracts") or 0.0))
                if size <= 0:
                    continue
                side = Side.BUY if p.get("side") == "long" else Side.SELL
                entry = float(p.get("entryPrice") or 0.0) or None
                sl, tp = self._split_levels(self._protective(p["symbol"]), side, entry or self._last(p["symbol"]))
                out.append(Position(p["symbol"], p["symbol"], side, size, entry, sl, tp,
                                    float(p.get("unrealizedPnl") or 0.0), comment="swap"))
            return out
        # spot: a "position" is a free+locked balance of a tracked base coin above dust
        bal = self.ex.fetch_balance()
        for sym in ([symbol] if symbol else self.tracked):
            base = self.ex.market(sym)["base"]
            qty = float((bal.get("total") or {}).get(base, 0.0) or 0.0)
            last = self._last(sym)
            if qty * last < self.dust_value:
                continue
            entry = self._entries.get(sym)
            sl, tp = self._split_levels(self._protective(sym), Side.BUY, entry or last)
            pnl = (last - entry) * qty if entry else 0.0
            out.append(Position(sym, sym, Side.BUY, qty, entry, sl, tp, pnl,
                                comment="" if entry else "entry unknown (restart)"))
        return out

    # ---- order helpers -----------------------------------------------------------------
    @staticmethod
    def _clean_id(client_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "", client_id)[:32]

    def _find_order(self, symbol: str, client_id: str) -> Optional[dict]:
        """After a network error: did the exchange actually receive our order?"""
        if not client_id:
            return None
        try:
            for o in self.ex.fetch_open_orders(symbol) + self.ex.fetch_closed_orders(symbol, limit=20):
                if o.get("clientOrderId") == client_id:
                    return o
        except Exception:
            log.exception("order lookup failed")
        return None

    def _free_base(self, symbol: str) -> float:
        base = self.ex.market(symbol)["base"]
        return float((self.ex.fetch_balance().get("free") or {}).get(base, 0.0) or 0.0)

    def _amount(self, symbol: str, amount: float) -> float:
        return float(self.ex.amount_to_precision(symbol, amount))       # truncates, never rounds up

    def _reduce_params(self, extra: Optional[dict] = None) -> dict:
        p = dict(extra or {})
        if self.market_type == "swap":
            p["reduceOnly"] = True
        return p

    def _place_protection(self, symbol: str, side: Side, amount: float, stop: float, take: Optional[float]) -> None:
        exit_side = side.opposite.value
        self.ex.create_order(symbol, "market", exit_side, amount, None,
                             self._reduce_params({"stopLossPrice": float(self.ex.price_to_precision(symbol, stop))}))
        if take is not None:
            try:
                self.ex.create_order(symbol, "market", exit_side, amount, None,
                                     self._reduce_params({"takeProfitPrice": float(self.ex.price_to_precision(symbol, take))}))
            except Exception:                                            # a missing TP is not an emergency
                log.exception("take-profit order failed for %s; stop-loss is in place", symbol)

    def _cancel_protective(self, symbol: str) -> None:
        for o in self._protective(symbol):
            try:
                self.ex.cancel_order(o["id"], symbol)
            except ccxt.OrderNotFound:
                pass

    # ---- trading -------------------------------------------------------------------------
    def place_order(self, req: OrderRequest) -> OrderResult:
        if self.require_stop_loss and req.stop_loss is None:
            raise OrderRejected("stop-loss is mandatory")
        if self.market_type == "spot" and req.side is Side.SELL:
            raise OrderRejected("spot is long-only; use close_position() to sell a holding")
        spec = self.get_symbol_spec(req.symbol)
        amount = self._amount(req.symbol, spec.normalize_volume(req.volume))
        if amount <= 0:
            raise OrderRejected(f"volume {req.volume} below the minimum ({spec.volume_min})")

        quote = self.ex.fetch_ticker(req.symbol)
        price = float(quote["ask"] if req.side is Side.BUY else quote["bid"])
        check_slippage(price, req.ref_price, req.max_slippage_pct)
        validate_levels(req.side, price, req.stop_loss, req.take_profit)
        min_cost = ((self.ex.market(req.symbol).get("limits") or {}).get("cost") or {}).get("min")
        if min_cost and amount * price < float(min_cost):
            raise OrderRejected(f"order value {amount * price:.2f} below exchange minimum {min_cost}")

        client_id = self._clean_id(req.client_id)
        params = {"clientOrderId": client_id} if client_id else {}
        try:
            order = self.ex.create_order(req.symbol, "market", req.side.value, amount, None, params)
        except ccxt.InsufficientFunds as exc:
            raise OrderRejected(f"insufficient funds: {exc}") from exc
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied) as exc:
            raise OrderRejected(str(exc)) from exc
        except NET as exc:                                              # unknown outcome: look, never resend
            order = self._find_order(req.symbol, client_id)
            if order is None:
                raise ConnectionLost(f"order outcome unknown and not found: {exc}") from exc
            log.warning("order recovered after network error: %s", order.get("id"))

        order = self._settle(order, req.symbol)
        filled = float(order.get("filled") or 0.0)
        fill_price = float(order.get("average") or order.get("price") or price)
        if filled <= 0:
            raise OrderRejected(f"order {order.get('id')} not filled (status {order.get('status')})")

        protect_amount = self._amount(req.symbol, min(filled, self._free_base(req.symbol))) \
            if self.market_type == "spot" else self._amount(req.symbol, filled)   # spot fees can be taken in the coin itself
        try:
            self._place_protection(req.symbol, req.side, protect_amount, req.stop_loss, req.take_profit)
        except Exception as exc:
            log.critical("stop-loss placement FAILED for %s, flattening: %s", req.symbol, exc)
            try:
                self.ex.create_order(req.symbol, "market", req.side.opposite.value, protect_amount, None, self._reduce_params())
            except Exception as exc2:
                raise UnprotectedPositionError(f"{req.symbol} is open WITHOUT a stop-loss and could not be closed: {exc2}") from exc2
            raise OrderRejected(f"stop-loss could not be placed; position closed again: {exc}") from exc

        self._entries[req.symbol] = fill_price
        self._levels[req.symbol] = (req.stop_loss, req.take_profit)
        return OrderResult(True, req.symbol, req.side, str(order.get("id")), req.symbol, fill_price, filled,
                           "filled", {"status": order.get("status")})

    @retry(attempts=3, exceptions=NET)
    def _settle(self, order: dict, symbol: str) -> dict:
        """Market orders sometimes come back without fill details; fetch the final state."""
        if order.get("filled") is None or order.get("average") is None:
            return self.ex.fetch_order(order["id"], symbol)
        return order

    def modify_sltp(self, position_id: str, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """
        Replace the protective orders. Exchanges reject duplicate reduce-only orders, so we must
        cancel first and then place; if placing fails we immediately restore the previous levels.
        None keeps the current value. The stop-loss can never be removed.
        """
        pos = next((p for p in self.positions(position_id)), None)
        if pos is None:
            raise BrokerError(f"no open position for {position_id}")
        new_sl = stop_loss if stop_loss is not None else pos.stop_loss
        new_tp = take_profit if take_profit is not None else pos.take_profit
        if self.require_stop_loss and new_sl is None:
            raise OrderRejected("cannot leave a position without a stop-loss")
        validate_levels(pos.side, self._last(pos.symbol), new_sl, new_tp)
        old = self._levels.get(pos.symbol, (pos.stop_loss, pos.take_profit))
        amount = self._amount(pos.symbol, pos.volume)
        self._cancel_protective(pos.symbol)
        try:
            self._place_protection(pos.symbol, pos.side, amount, new_sl, new_tp)
        except Exception as exc:
            log.critical("modify failed, restoring previous protection: %s", exc)
            try:
                self._place_protection(pos.symbol, pos.side, amount, old[0], old[1])
            except Exception as exc2:
                raise UnprotectedPositionError(f"{pos.symbol} lost its stop-loss during modify: {exc2}") from exc2
            raise BrokerError(f"modify failed, old levels restored: {exc}") from exc
        self._levels[pos.symbol] = (new_sl, new_tp)

    def close_position(self, position_id: str) -> OrderResult:
        pos = next((p for p in self.positions(position_id)), None)
        if pos is None:
            raise BrokerError(f"no open position for {position_id}")
        self._cancel_protective(pos.symbol)              # stale stops must not outlive the position
        amount = self._amount(pos.symbol, pos.volume)
        try:
            order = self.ex.create_order(pos.symbol, "market", pos.side.opposite.value, amount, None, self._reduce_params())
        except Exception as exc:
            lv = self._levels.get(pos.symbol)
            if lv and lv[0]:
                try:
                    self._place_protection(pos.symbol, pos.side, amount, lv[0], lv[1])      # close failed: re-protect
                except Exception:
                    log.critical("close failed AND re-protection failed for %s", pos.symbol)
            raise OrderRejected(f"close failed: {exc}") from exc
        order = self._settle(order, pos.symbol)
        self._entries.pop(pos.symbol, None)
        self._levels.pop(pos.symbol, None)
        return OrderResult(True, pos.symbol, pos.side.opposite, str(order.get("id")), pos.symbol,
                           float(order.get("average") or 0.0), float(order.get("filled") or amount), "closed")

    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]:
        results: list[OrderResult] = []
        for pos in self.positions(symbol):
            try:
                results.append(self.close_position(pos.id))
            except BrokerError as exc:
                log.error("close_all: failed to close %s: %s", pos.id, exc)
                results.append(OrderResult(False, pos.symbol, pos.side.opposite, position_id=pos.id, message=str(exc)))
        return results

    def sync_protective_orders(self) -> int:
        """Housekeeping: cancel stop/TP orders left behind by positions that no longer exist."""
        cancelled = 0
        open_syms = {p.symbol for p in self.positions()}
        for sym in self.tracked:
            if sym not in open_syms:
                for o in self._protective(sym):
                    try:
                        self.ex.cancel_order(o["id"], sym)
                        cancelled += 1
                    except ccxt.OrderNotFound:
                        pass
        return cancelled