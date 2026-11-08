"""
aavp.py - Auto-Anchored Volume Profile (AAVP)

Finds "macro market gravity" (POC / Value Area), "air pockets" (LVNs) and a
liquidity-sweep + rejection signal, from plain OHLCV candles.

Plain-English picture
---------------------
Imagine a market as a busy street trading day:
  * Volume Profile  = a tally of HOW MUCH business happened at each price.
  * POC             = the price where the most business happened (the "town square").
  * Value Area      = the price zone holding 70% of all the business (the "town centre").
  * LVN             = a price zone where almost no business happened (an "empty road");
                      price tends to zip through it fast.
  * Anchor          = the day we start the tally. We re-start it automatically
                      whenever a new important event (big high / low / volume spike)
                      happens in the lookback window.

IMPORTANT: this module only DETECTS and REPORTS. It never places orders and never
sizes positions. A separate deterministic risk layer (the "Guardian") must approve
anything this module flags.

Input: pandas DataFrame indexed by candle OPEN time with columns
       open, high, low, close, volume.
       Forex/Gold on MT5 -> use tick_volume (real volume is not available on most
       retail FX/CFD feeds). Crypto -> use the exchange's real traded volume.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
class AnchorMode(str, Enum):
    HIGHEST_HIGH = "highest_high"        # major resistance sweep
    LOWEST_LOW = "lowest_low"            # major support sweep
    HIGHEST_VOLUME = "highest_volume"    # climax / capitulation candle


# Minimum price increment per symbol. VERIFY against your broker/exchange
# (MT5: symbol_info().trade_tick_size, Binance: PRICE_FILTER tickSize).
TICK_SIZES = {"XAUUSD": 0.01, "SOLUSDT": 0.01, "BTCUSDT": 0.01, "ETHUSDT": 0.01}


@dataclass
class AAVPConfig:
    lookback: int = 200                  # candles searched for the macro anchor
    anchor_mode: AnchorMode = AnchorMode.HIGHEST_VOLUME
    min_profile_bars: int = 5            # a profile built from fewer bars is not trusted

    # --- price rows (the "buckets" volume is dropped into) ---
    tick_size: float = 0.01              # smallest price step of the symbol
    row_size: Optional[float] = None     # fixed row height; None = auto from ATR
    atr_period: int = 14
    atr_fraction: float = 0.10           # auto row height ~= 10% of ATR ...
    max_rows: int = 400                  # ... but never more than this many rows

    # --- value area ---
    value_area_pct: float = 0.70

    # --- low volume nodes ---
    lvn_ratio: float = 0.50              # row is "low" if volume < ratio * average row volume
    lvn_smooth: int = 3                  # odd number; smooths single-row noise

    # --- signal ---
    live_candle: bool = True             # last row of df is still forming -> judge the one before it
    sweep_lookback: int = 20             # swing window whose extreme must be swept
    lvn_lookback: int = 3                # candles (incl. trigger) allowed to have touched an LVN
    wick_ratio: float = 0.50             # rejection wick must be >= 50% of candle range
    touch_tol_atr: float = 0.25          # "directly off the level" tolerance, in ATRs

    max_frozen_levels: int = 9

    @classmethod
    def for_symbol(cls, symbol: str, **overrides) -> "AAVPConfig":
        cfg = cls(tick_size=TICK_SIZES.get(symbol.upper(), 0.01))
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------- #
# 2. Small helpers
# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range = typical candle size, used to scale everything per asset."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    val = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(val)


def nice_step(x: float, tick: float) -> float:
    """
    Snap a row height UP to a 1-2-5 sequence of tick sizes (1,2,5,10,20,50 ticks...).
    Why: the ATR changes every candle. If the row height changed with it, every row
    edge would shift and the POC would jitter. Snapping keeps the grid stable.
    """
    units = max(x / tick, 1.0)
    exp = math.floor(math.log10(units))
    for m in (1, 2, 5, 10):
        cand = m * 10 ** exp
        if cand >= units - 1e-12:
            return round(cand * tick, 10)
    return round(10 ** (exp + 1) * tick, 10)


# --------------------------------------------------------------------------- #
# 3. Anchor selection
# --------------------------------------------------------------------------- #
def find_anchor(window: pd.DataFrame, mode: AnchorMode) -> int:
    """
    Position (0-based, inside `window`) of the macro event the profile starts from.
    On ties we take the MOST RECENT occurrence (newer information matters more).
    """
    col = {"highest_high": "high", "lowest_low": "low", "highest_volume": "volume"}[AnchorMode(mode).value]
    arr = window[col].to_numpy()
    best = arr.min() if mode == AnchorMode.LOWEST_LOW else arr.max()
    return int(np.flatnonzero(arr == best)[-1])


# --------------------------------------------------------------------------- #
# 4. Volume aggregation
# --------------------------------------------------------------------------- #
@dataclass
class VolumeProfile:
    edges: np.ndarray      # n_rows + 1 price boundaries (row i spans edges[i]..edges[i+1])
    volumes: np.ndarray    # n_rows volumes (the "tally" per row)
    row_size: float

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0


def build_profile(bars: pd.DataFrame, row_size: float) -> VolumeProfile:
    """
    Aggregate volume per price row, from the anchor candle to the live candle.

    We only have candles (not every tick), so we assume each candle's volume was
    spread EVENLY across its high-low range. A candle covering 40% of a row's
    height therefore puts 40% of that row's slice of the candle's volume there.
    (Feeding real tick/trade data would be more exact; the rest of the code is unchanged.)

    Array shapes (B = candles, N = rows):
        lows/highs/vols : (B, 1)      row_lo/row_hi : (1, N)
        share           : (B, N)  fraction of each candle's volume landing in each row
        volumes         : (N,)    column sums of share * vols  -> the profile
    """
    lo, hi = float(bars["low"].min()), float(bars["high"].max())

    # Rows are aligned to multiples of row_size (not to this window's low).
    # Result: row edges are identical between recalculations, so levels don't wobble.
    first_edge = math.floor(lo / row_size + 1e-9) * row_size
    n_rows = max(1, int(math.ceil((hi - first_edge) / row_size - 1e-9)))
    edges = first_edge + row_size * np.arange(n_rows + 1)

    lows = bars["low"].to_numpy()[:, None]
    highs = bars["high"].to_numpy()[:, None]
    vols = bars["volume"].to_numpy(dtype=float)[:, None]
    row_lo, row_hi = edges[:-1][None, :], edges[1:][None, :]

    # How much of each candle's range overlaps each row (clipped at 0 = no overlap).
    overlap = np.clip(np.minimum(highs, row_hi) - np.maximum(lows, row_lo), 0.0, None)
    rng = highs - lows
    share = np.zeros_like(overlap)

    nz = rng[:, 0] > 0
    share[nz] = overlap[nz] / rng[nz]          # shares of a normal candle sum to 1

    flat = ~nz                                 # high == low: all volume at that one price
    if flat.any():
        idx = np.clip(np.floor((lows[flat, 0] - first_edge) / row_size).astype(int), 0, n_rows - 1)
        share[np.flatnonzero(flat), idx] = 1.0

    return VolumeProfile(edges=edges, volumes=(share * vols).sum(axis=0), row_size=row_size)


# --------------------------------------------------------------------------- #
# 5. POC, Value Area, LVNs
# --------------------------------------------------------------------------- #
def point_of_control(volumes: np.ndarray) -> int:
    """Index of the row with the most volume. Ties -> the one nearest the profile's middle."""
    candidates = np.flatnonzero(np.isclose(volumes, volumes.max()))
    mid = (len(volumes) - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - mid))])


def value_area(volumes: np.ndarray, poc_idx: int, pct: float = 0.70) -> tuple[int, int, float]:
    """
    Standard "two-rows-at-a-time" Value Area (the method used by CME-style profiles).

    Maths:
        target = pct * total_volume
        1. Start with just the POC row:  acc = volumes[poc]
        2. Sum the next 2 rows ABOVE the current area  -> up
           Sum the next 2 rows BELOW the current area  -> down
        3. Add whichever pair is bigger (ties go up), update acc.
        4. Repeat until acc >= target (or no rows left).
    So the area grows outward from the POC toward where the volume is, and always
    ends up CONTIGUOUS. Because we add two rows at once, it can slightly overshoot
    70% (e.g. 71.3%), never undershoot.

    Returns (low_row_index, high_row_index, achieved_fraction).
    """
    n, total = len(volumes), float(volumes.sum())
    target = pct * total
    lo = hi = poc_idx
    acc = float(volumes[poc_idx])

    while acc < target and (lo > 0 or hi < n - 1):
        up = float(volumes[hi + 1: hi + 3].sum()) if hi < n - 1 else -1.0
        down = float(volumes[max(lo - 2, 0): lo].sum()) if lo > 0 else -1.0
        if up >= down:                         # go up (also covers "down exhausted")
            new_hi = min(hi + 2, n - 1)
            acc += float(volumes[hi + 1: new_hi + 1].sum())
            hi = new_hi
        else:
            new_lo = max(lo - 2, 0)
            acc += float(volumes[new_lo: lo].sum())
            lo = new_lo
    return lo, hi, (acc / total if total > 0 else 0.0)


@dataclass
class LvnZone:
    low: float
    high: float
    min_volume: float


def find_lvns(prof: VolumeProfile, ratio: float, smooth: int) -> list[LvnZone]:
    """
    Low Volume Nodes = runs of adjacent rows whose (smoothed) volume is below
    `ratio` x the average row volume. Runs touching the very top or bottom of the
    profile are discarded: those are just the thin tails of the range, not gaps
    INSIDE it, and price has not "zipped through" them.
    """
    vol = prof.volumes
    n = len(vol)
    if n < 3:
        return []
    k = max(1, smooth | 1)                                   # force odd
    sm = np.convolve(np.pad(vol, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    low_mask = sm < ratio * vol.mean()

    zones, i = [], 0
    while i < n:
        if low_mask[i]:
            j = i
            while j + 1 < n and low_mask[j + 1]:
                j += 1
            if i > 0 and j < n - 1:                          # interior runs only
                zones.append(LvnZone(float(prof.edges[i]), float(prof.edges[j + 1]), float(vol[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    return zones


# --------------------------------------------------------------------------- #
# 6. Persistent levels
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    name: str                    # "POC" | "VAH" | "VAL"
    price: float
    anchor_time: object
    frozen: bool = False
    frozen_at: object = None


# --------------------------------------------------------------------------- #
# 7. Signal: LVN entry + liquidity sweep + rejection off POC / VAL (or VAH)
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    direction: str               # "long" | "short"
    level_name: str              # which macro level price rejected from
    level_price: float
    trigger_time: object
    reference_price: float       # close of the trigger candle (NOT an order price)
    swept_price: float           # the local high/low that was taken out
    invalidation: float          # extreme of the sweep wick; idea is wrong beyond this
    lvn: Optional[dict] = None


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def detect_signal(window: pd.DataFrame, levels: dict, lvns: list[LvnZone],
                  cfg: AAVPConfig, atr_val: float, row_size: float) -> Optional[Signal]:
    """
    LONG  = (1) a recent candle traded inside an LVN
            (2) trigger candle's low swept below the lowest low of the prior N candles
                but it CLOSED back above that level (stops were taken, price refused)
            (3) rejection candle: bullish, lower wick >= wick_ratio of its range
            (4) its low sits right at the macro VAL or POC (within tolerance) and it closes above it
    SHORT = the mirror image, using VAH or POC and a sweep of the prior highs.
    The brief names POC/VAL (longs); the short side is the symmetric extension.
    """
    n = len(window)
    i = n + (-2 if cfg.live_candle else -1)          # position of the candle we judge
    if i < 6 or not lvns:
        return None

    c = window.iloc[i]
    prior = window.iloc[max(0, i - cfg.sweep_lookback): i]
    recent = window.iloc[max(0, i - cfg.lvn_lookback + 1): i + 1]
    tol = max(cfg.touch_tol_atr * atr_val, row_size)
    rng = float(c["high"] - c["low"])
    if rng <= 0:
        return None

    # (1) LVN touched by any of the last few candles (range overlaps the zone)
    hit = next((z for z in lvns
                if bool(((recent["low"] <= z.high) & (recent["high"] >= z.low)).any())), None)
    if hit is None:
        return None

    o, cl, lo, hi = float(c["open"]), float(c["close"]), float(c["low"]), float(c["high"])

    # LONG
    swept_low = float(prior["low"].min())
    lower_wick = min(o, cl) - lo
    if lo < swept_low < cl and cl > o and lower_wick / rng >= cfg.wick_ratio:
        for name in ("VAL", "POC"):
            lvl = levels.get(name)
            if lvl is not None and _near(lo, lvl, tol) and cl > lvl:
                return Signal("long", name, lvl, window.index[i], cl, swept_low, lo, asdict(hit))

    # SHORT
    swept_high = float(prior["high"].max())
    upper_wick = hi - max(o, cl)
    if hi > swept_high > cl and cl < o and upper_wick / rng >= cfg.wick_ratio:
        for name in ("VAH", "POC"):
            lvl = levels.get(name)
            if lvl is not None and _near(hi, lvl, tol) and cl < lvl:
                return Signal("short", name, lvl, window.index[i], cl, swept_high, hi, asdict(hit))
    return None


# --------------------------------------------------------------------------- #
# 8. Engine (stateful, call .update(df) on every new tick / candle)
# --------------------------------------------------------------------------- #
@dataclass
class AAVPResult:
    ready: bool
    status: str
    anchor: Optional[dict] = None
    row_size: Optional[float] = None
    n_rows: int = 0
    bars_in_profile: int = 0
    total_volume: float = 0.0
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    value_area_fraction: Optional[float] = None
    lvn_zones: list = field(default_factory=list)
    frozen_levels: list = field(default_factory=list)
    signal: Optional[dict] = None
    profile: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        """JSON payload: easy to send to an Android app or feed to the Gemini API."""
        def default(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, pd.Timestamp): return o.isoformat()
            return str(o)
        return json.dumps(self.to_dict(), default=default, **kw)


class AAVPEngine:
    def __init__(self, cfg: Optional[AAVPConfig] = None):
        self.cfg = cfg or AAVPConfig()
        self._anchor_key = None
        self._live: dict[str, Level] = {}
        self._frozen: list[Level] = []

    # ---- internals ---------------------------------------------------------
    def _resolve_row_size(self, bars: pd.DataFrame, window: pd.DataFrame, atr_val: float) -> float:
        c = self.cfg
        if c.row_size:
            rs = c.row_size
        else:
            rs = nice_step(max(atr_val * c.atr_fraction, c.tick_size), c.tick_size)
        span = float(bars["high"].max() - bars["low"].min())
        if span / rs > c.max_rows:                     # protect speed in wild markets
            rs = nice_step(span / c.max_rows, c.tick_size)
        return rs

    def _prune_frozen(self, window: pd.DataFrame) -> None:
        """A retired level stays projected until price trades through it, then it's gone."""
        last = window.iloc[-2:]
        lo, hi = float(last["low"].min()), float(last["high"].max())
        self._frozen = [l for l in self._frozen if not (lo <= l.price <= hi)]

    def _freeze_live(self, at) -> None:
        for l in self._live.values():
            l.frozen, l.frozen_at = True, at
            self._frozen.append(l)
        self._live = {}
        self._frozen = self._frozen[-self.cfg.max_frozen_levels:]

    # ---- public ------------------------------------------------------------
    def update(self, df: pd.DataFrame, include_profile: bool = False) -> AAVPResult:
        c = self.cfg
        missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            raise ValueError(f"df is missing columns: {sorted(missing)}")

        window = df.iloc[-c.lookback:]
        if len(window) < c.min_profile_bars:
            return AAVPResult(False, "not_enough_bars")

        # --- re-anchoring ---------------------------------------------------
        # Every update we re-scan the lookback window for the macro event.
        # If a NEW high/low/volume record appears (or the old one scrolls out of
        # the window), the anchor candle changes. When that happens:
        #   - the old profile's POC/VAH/VAL are FROZEN (kept as projected levels),
        #   - a fresh profile starts from the new anchor.
        pos = find_anchor(window, c.anchor_mode)
        anchor_time = window.index[pos]
        key = (AnchorMode(c.anchor_mode).value, anchor_time)

        self._prune_frozen(window)
        if key != self._anchor_key:
            self._freeze_live(at=window.index[-1])
            self._anchor_key = key

        anchor_info = {"mode": AnchorMode(c.anchor_mode).value, "time": anchor_time,
                       "price": float(window[{"highest_high": "high", "lowest_low": "low",
                                              "highest_volume": "close"}[AnchorMode(c.anchor_mode).value]].iloc[pos]),
                       "bars_ago": len(window) - 1 - pos}
        frozen = [asdict(l) for l in self._frozen]

        bars = window.iloc[pos:]
        if len(bars) < c.min_profile_bars:
            return AAVPResult(False, "anchor_pending", anchor_info, bars_in_profile=len(bars), frozen_levels=frozen)

        # --- profile --------------------------------------------------------
        atr_val = atr(window, c.atr_period)
        row_size = self._resolve_row_size(bars, window, atr_val)
        prof = build_profile(bars, row_size)

        poc_i = point_of_control(prof.volumes)
        va_lo, va_hi, va_frac = value_area(prof.volumes, poc_i, c.value_area_pct)
        poc = float(prof.centers[poc_i])               # POC = centre of the busiest row
        val_ = float(prof.edges[va_lo])                # VAL = bottom edge of lowest VA row
        vah = float(prof.edges[va_hi + 1])             # VAH = top edge of highest VA row

        # "Developing" levels: they are re-written each update while this anchor is
        # active, and projected forward (persist) as variables until the anchor changes.
        self._live = {n: Level(n, p, anchor_time) for n, p in (("POC", poc), ("VAH", vah), ("VAL", val_))}

        lvns = find_lvns(prof, c.lvn_ratio, c.lvn_smooth)
        sig = detect_signal(window, {"POC": poc, "VAH": vah, "VAL": val_}, lvns, c, atr_val, row_size)

        return AAVPResult(
            ready=True, status="ok", anchor=anchor_info, row_size=row_size,
            n_rows=len(prof.volumes), bars_in_profile=len(bars),
            total_volume=float(prof.volumes.sum()),
            poc=poc, vah=vah, val=val_, value_area_fraction=va_frac,
            lvn_zones=[asdict(z) for z in lvns], frozen_levels=frozen,
            signal=asdict(sig) if sig else None,
            profile=[{"price": float(p), "volume": float(v)} for p, v in zip(prof.centers, prof.volumes)]
            if include_profile else None,
        )


# --------------------------------------------------------------------------- #
# 9. Data adapters
# --------------------------------------------------------------------------- #
def rates_to_df(rates, volume_field: str = "tick_volume") -> pd.DataFrame:
    """MetaTrader5 copy_rates_* output -> DataFrame. Position 0 of copy_rates_from_pos is the LIVE bar."""
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["volume"] = df[volume_field]
    return df.set_index("time")[["open", "high", "low", "close", "volume"]]


def ohlcv_to_df(rows) -> pd.DataFrame:
    """ccxt / Binance / Bybit klines [[ms, o, h, l, c, v], ...] -> DataFrame."""
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time").astype(float)


if __name__ == "__main__":
    # Quick demo on synthetic gold-like data (not real market data).
    rng = np.random.default_rng(7)
    n = 260
    close = 2650 + np.cumsum(rng.normal(0, 1.2, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.5, n)
    vol = rng.integers(200, 2000, n).astype(float)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    data = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

    eng = AAVPEngine(AAVPConfig.for_symbol("XAUUSD"))
    print(eng.update(data).to_json(indent=2))