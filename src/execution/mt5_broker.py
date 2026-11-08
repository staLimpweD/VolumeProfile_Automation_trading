"""
src/execution/mt5_broker.py - order execution on MetaTrader 5 (Exness, Deriv, ...).

Safety rules built in:
  * a stop-loss is mandatory (configurable, on by default)
  * every order is pre-validated with order_check() before order_send()
  * order_send is NEVER blindly retried: if the outcome is unknown we look for the
    position (by client id in the comment) before deciding anything
  * only positions carrying OUR magic number are ever touched
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from src.data.mt5_session import MT5Session
from src.models import (AccountInfo, Broker, BrokerError, ConnectionLost, OrderRejected,
                        OrderRequest, OrderResult, Position, Side, SymbolSpec,
                        check_slippage, validate_levels)

log = logging.getLogger(__name__)

MAX_COMMENT = 31                              # MT5 truncates comments beyond this


class MT5Broker(Broker):
    def __init__(self, session: MT5Session, magic: int, require_stop_loss: bool = True,
                 max_attempts: int = 3):
        self.session = session
        self.magic = int(magic)               # tags OUR orders so we never touch manual trades
        self.require_stop_loss = require_stop_loss
        self.max_attempts = max_attempts

    # ---- convenience ---------------------------------------------------------------
    @property
    def mt5(self):
        return self.session.mt5

    def _codes(self):
        m = self.mt5
        g = lambda n: getattr(m, n, None)
        return {
            "done": {g("TRADE_RETCODE_DONE"), g("TRADE_RETCODE_DONE_PARTIAL")},
            "retry": {g("TRADE_RETCODE_REQUOTE"), g("TRADE_RETCODE_PRICE_CHANGED"), g("TRADE_RETCODE_PRICE_OFF")},
            "unknown": {g("TRADE_RETCODE_TIMEOUT"), g("TRADE_RETCODE_CONNECTION")},
        }

    # ---- lifecycle -------------------------------------------------------------------
    def connect(self) -> None:
        self.session.connect()

    def close(self) -> None:
        self.session.shutdown()

    def is_healthy(self) -> bool:
        try:
            with self.session.lock:
                term, acc = self.mt5.terminal_info(), self.mt5.account_info()
                return bool(term and term.connected and term.trade_allowed and acc and acc.trade_allowed)
        except Exception:
            log.exception("MT5 health check failed")
            return False

    # ---- account and symbol --------------------------------------------------------
    def account_info(self) -> AccountInfo:
        with self.session.lock:
            self.session.ensure()
            a = self.mt5.account_info()
            if a is None:
                raise ConnectionLost(f"account_info failed: {self.mt5.last_error()}")
            return AccountInfo(str(a.login), a.currency, float(a.balance), float(a.equity),
                               float(a.margin_free), float(a.leverage), bool(a.trade_allowed))

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        with self.session.lock:
            self.session.ensure()
            i = self.mt5.symbol_info(symbol)
            if i is None:
                raise BrokerError(f"unknown symbol '{symbol}': {self.mt5.last_error()}")
            tick_value = getattr(i, "trade_tick_value_loss", 0) or i.trade_tick_value   # loss-side value is the safe one
            stops = max(i.trade_stops_level, getattr(i, "trade_freeze_level", 0)) * i.point
            return SymbolSpec(symbol, float(i.trade_tick_size or i.point), float(tick_value),
                              float(i.trade_contract_size), float(i.volume_min), float(i.volume_max),
                              float(i.volume_step), int(i.digits), float(stops))

    # ---- positions -------------------------------------------------------------------
    def _to_position(self, p) -> Position:
        m = self.mt5
        return Position(
            id=str(p.ticket), symbol=p.symbol,
            side=Side.BUY if p.type == m.POSITION_TYPE_BUY else Side.SELL,
            volume=float(p.volume), entry_price=float(p.price_open),
            stop_loss=float(p.sl) or None, take_profit=float(p.tp) or None,
            profit=float(p.profit) + float(getattr(p, "swap", 0.0)),
            opened_at=self.session.to_utc(p.time), comment=p.comment or "")

    def positions(self, symbol: Optional[str] = None) -> list[Position]:
        with self.session.lock:
            self.session.ensure()
            raw = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
            if raw is None:                                   # None = error, () = simply no positions
                raise ConnectionLost(f"positions_get failed: {self.mt5.last_error()}")
            return [self._to_position(p) for p in raw if p.magic == self.magic]

    def _get_position(self, position_id: str):
        raw = self.mt5.positions_get(ticket=int(position_id))
        if not raw or raw[0].magic != self.magic:
            raise BrokerError(f"position {position_id} not found (or not opened by this bot)")
        return raw[0]

    def _find_by_client_id(self, symbol: str, client_id: str):
        """Used after an uncertain send: did the trade actually happen?"""
        if not client_id:
            return None
        raw = self.mt5.positions_get(symbol=symbol) or ()
        for p in raw:
            if p.magic == self.magic and client_id in (p.comment or ""):
                return p
        return None

    # ---- order construction ------------------------------------------------------
    def _filling(self, symbol: str):
        """Pick a fill mode the symbol supports (brokers differ: Exness usually IOC)."""
        m = self.mt5
        mode = m.symbol_info(symbol).filling_mode
        if mode & 2:
            return m.ORDER_FILLING_IOC
        if mode & 1:
            return m.ORDER_FILLING_FOK
        return m.ORDER_FILLING_RETURN

    def _deviation(self, price: float, point: float, pct: float) -> int:
        return max(1, int(math.ceil(price * pct / 100.0 / point)))

    # ---- trading ---------------------------------------------------------------------
    def place_order(self, req: OrderRequest) -> OrderResult:
        if self.require_stop_loss and req.stop_loss is None:
            raise OrderRejected("stop-loss is mandatory")
        m, codes = self.mt5, self._codes()
        with self.session.lock:
            self.session.ensure()
            spec = self.get_symbol_spec(req.symbol)
            info = m.symbol_info(req.symbol)
            volume = spec.normalize_volume(req.volume)
            if volume <= 0:
                raise OrderRejected(f"volume {req.volume} is below the minimum lot ({spec.volume_min})")

            comment = (req.client_id or req.comment)[:MAX_COMMENT]
            request, last_msg = None, ""
            for attempt in range(1, self.max_attempts + 1):
                tick = m.symbol_info_tick(req.symbol)
                if tick is None:
                    raise ConnectionLost(f"no tick for {req.symbol}")
                price = float(tick.ask if req.side is Side.BUY else tick.bid)
                check_slippage(price, req.ref_price, req.max_slippage_pct)
                validate_levels(req.side, price, req.stop_loss, req.take_profit, spec.min_stop_distance)

                request = {
                    "action": m.TRADE_ACTION_DEAL, "symbol": req.symbol, "volume": volume,
                    "type": m.ORDER_TYPE_BUY if req.side is Side.BUY else m.ORDER_TYPE_SELL,
                    "price": price, "sl": req.stop_loss or 0.0, "tp": req.take_profit or 0.0,
                    "deviation": self._deviation(price, info.point, req.max_slippage_pct),
                    "magic": self.magic, "comment": comment,
                    "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(req.symbol),
                }
                check = m.order_check(request)                      # margin, stops, trading allowed?
                if check is None or check.retcode != 0:
                    raise OrderRejected(f"order_check failed: {getattr(check, 'comment', m.last_error())}")

                result = m.order_send(request)
                if result is None or result.retcode in codes["unknown"]:
                    # Outcome unknown. Do NOT resend: look for the position first.
                    found = self._find_by_client_id(req.symbol, req.client_id)
                    if found:
                        log.warning("order outcome was unclear but position exists: %s", found.ticket)
                        return self._result(req, found.price_open, found.volume, str(found.ticket), str(found.ticket), "recovered")
                    raise ConnectionLost(f"order outcome unknown and no matching position found: {m.last_error()}")
                if result.retcode in codes["done"]:
                    return self._ok(req, result)
                last_msg = f"retcode {result.retcode}: {result.comment}"
                if result.retcode in codes["retry"] and attempt < self.max_attempts:
                    log.info("requote/price change, retrying (%d/%d)", attempt, self.max_attempts)
                    continue
                raise OrderRejected(last_msg)
            raise OrderRejected(last_msg or "order not filled")

    def _ok(self, req: OrderRequest, result) -> OrderResult:
        position_id = ""
        try:
            deals = self.mt5.history_deals_get(ticket=result.deal)
            if deals:
                position_id = str(deals[0].position_id)
        except Exception:
            log.exception("could not resolve position id for deal %s", result.deal)
        return self._result(req, float(result.price), float(result.volume), str(result.order),
                            position_id, result.comment or "", {"retcode": result.retcode, "deal": result.deal})

    @staticmethod
    def _result(req: OrderRequest, price, volume, order_id, position_id, msg, raw=None) -> OrderResult:
        return OrderResult(True, req.symbol, req.side, order_id, position_id, float(price), float(volume), msg, raw or {})

    def modify_sltp(self, position_id: str, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """Change SL and/or TP. A value of None keeps the current one. The SL can never be removed."""
        m = self.mt5
        with self.session.lock:
            self.session.ensure()
            p = self._get_position(position_id)
            side = Side.BUY if p.type == m.POSITION_TYPE_BUY else Side.SELL
            new_sl = stop_loss if stop_loss is not None else (p.sl or None)
            new_tp = take_profit if take_profit is not None else (p.tp or None)
            if self.require_stop_loss and new_sl is None:
                raise OrderRejected("cannot leave a position without a stop-loss")
            tick = m.symbol_info_tick(p.symbol)
            price = float(tick.bid if side is Side.BUY else tick.ask)      # price at which it would close
            validate_levels(side, price, new_sl, new_tp, self.get_symbol_spec(p.symbol).min_stop_distance)
            result = m.order_send({"action": m.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol,
                                   "sl": new_sl or 0.0, "tp": new_tp or 0.0, "magic": self.magic})
            if result is None or result.retcode not in self._codes()["done"]:
                raise OrderRejected(f"modify failed: {getattr(result, 'comment', m.last_error())}")

    def close_position(self, position_id: str) -> OrderResult:
        m = self.mt5
        with self.session.lock:
            self.session.ensure()
            p = self._get_position(position_id)
            side = Side.BUY if p.type == m.POSITION_TYPE_BUY else Side.SELL
            tick = m.symbol_info_tick(p.symbol)
            if tick is None:
                raise ConnectionLost(f"no tick for {p.symbol}")
            price = float(tick.bid if side is Side.BUY else tick.ask)      # close a buy at the bid, a sell at the ask
            info = m.symbol_info(p.symbol)
            request = {
                "action": m.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "position": p.ticket,
                "type": m.ORDER_TYPE_SELL if side is Side.BUY else m.ORDER_TYPE_BUY, "price": price,
                "deviation": self._deviation(price, info.point, 0.2), "magic": self.magic,
                "comment": "bot close", "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(p.symbol),
            }
            result = m.order_send(request)
            if result is None or result.retcode not in self._codes()["done"]:
                raise OrderRejected(f"close failed: {getattr(result, 'comment', m.last_error())}")
            return OrderResult(True, p.symbol, side.opposite, str(result.order), str(p.ticket),
                               float(result.price), float(result.volume), "closed", {"retcode": result.retcode})

    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]:
        """Close every bot position. Keeps going if one fails; failures are returned, not hidden."""
        results: list[OrderResult] = []
        for pos in self.positions(symbol):
            try:
                results.append(self.close_position(pos.id))
            except BrokerError as exc:
                log.error("close_all: failed to close %s: %s", pos.id, exc)
                results.append(OrderResult(False, pos.symbol, pos.side.opposite, position_id=pos.id, message=str(exc)))
        return results