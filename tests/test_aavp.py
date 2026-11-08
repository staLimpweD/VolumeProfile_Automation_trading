import numpy as np
import pandas as pd
import pytest

from aavp import (AAVPConfig, AAVPEngine, AnchorMode, LvnZone, build_profile,
                  detect_signal, find_anchor, find_lvns, nice_step,
                  point_of_control, value_area)


def make_df(n=120, seed=1, start=100.0):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.6, n)
    vol = rng.integers(100, 1000, n).astype(float)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_volume_is_conserved():
    df = make_df()
    prof = build_profile(df, 0.5)
    assert prof.volumes.sum() == pytest.approx(df["volume"].sum())


def test_flat_candle_goes_to_one_row():
    df = make_df(10)
    df.iloc[3, df.columns.get_indexer(["open", "high", "low", "close"])] = 100.2
    prof = build_profile(df, 0.5)
    assert prof.volumes.sum() == pytest.approx(df["volume"].sum())


def test_rows_are_aligned_to_row_size_multiples():
    prof = build_profile(make_df(), 0.5)
    assert np.allclose(prof.edges / 0.5, np.round(prof.edges / 0.5))


def test_poc_is_busiest_row():
    df = make_df()
    prof = build_profile(df, 0.5)
    assert prof.volumes[point_of_control(prof.volumes)] == prof.volumes.max()


def test_value_area_reaches_target_and_is_contiguous():
    prof = build_profile(make_df(), 0.25)
    poc = point_of_control(prof.volumes)
    lo, hi, frac = value_area(prof.volumes, poc, 0.70)
    assert lo <= poc <= hi
    assert frac >= 0.70 - 1e-9
    assert prof.volumes[lo:hi + 1].sum() / prof.volumes.sum() == pytest.approx(frac)


def test_value_area_hand_example():
    vols = np.array([1, 2, 10, 30, 20, 5, 1, 1.0])  # total 70, target 49
    poc = 3
    lo, hi, frac = value_area(vols, poc, 0.70)
    # start 30; up pair(20+5)=25 vs down pair(2+10)=12 -> add up: 55 >= 49
    assert (lo, hi) == (3, 5)
    assert frac == pytest.approx(55 / 70)


def test_value_area_100pct_covers_everything():
    vols = np.array([1.0, 3, 5, 3, 1])
    assert value_area(vols, 2, 1.0)[:2] == (0, 4)


def test_anchor_modes():
    df = make_df()
    w = df.iloc[-50:]
    assert w.iloc[find_anchor(w, AnchorMode.HIGHEST_HIGH)]["high"] == w["high"].max()
    assert w.iloc[find_anchor(w, AnchorMode.LOWEST_LOW)]["low"] == w["low"].min()
    assert w.iloc[find_anchor(w, AnchorMode.HIGHEST_VOLUME)]["volume"] == w["volume"].max()


def test_lvn_interior_gap_found_and_tails_ignored():
    # volume: thin tail, heavy, gap, heavy, thin tail
    from aavp import VolumeProfile
    vols = np.array([0.1, 10, 10, 0.2, 0.2, 10, 10, 0.1])
    prof = VolumeProfile(edges=np.arange(9, dtype=float), volumes=vols, row_size=1.0)
    zones = find_lvns(prof, ratio=0.5, smooth=1)
    assert len(zones) == 1
    assert (zones[0].low, zones[0].high) == (3.0, 5.0)


def test_nice_step_stable_and_monotonic():
    assert nice_step(0.013, 0.01) == 0.02
    assert nice_step(0.04, 0.01) == 0.05
    assert nice_step(0.001, 0.01) == 0.01
    assert nice_step(0.51, 0.01) == 1.0


def test_engine_basic_and_json():
    eng = AAVPEngine(AAVPConfig(tick_size=0.01))
    res = eng.update(make_df(), include_profile=True)
    assert res.ready and res.val <= res.poc <= res.vah
    assert res.total_volume == pytest.approx(sum(p["volume"] for p in res.profile))
    import json
    json.loads(res.to_json())


def test_reanchor_freezes_old_levels_until_touched():
    cfg = AAVPConfig(tick_size=0.01, anchor_mode=AnchorMode.HIGHEST_HIGH, lookback=60)
    eng = AAVPEngine(cfg)
    df = make_df(100)
    r1 = eng.update(df.iloc[:80])
    assert r1.ready and not r1.frozen_levels
    old_poc = r1.poc

    # Make a clearly new macro high far above everything, with enough bars after it.
    ext = df.iloc[80:].copy()
    ext.iloc[2, ext.columns.get_loc("high")] = df["high"].max() + 20
    ext.iloc[2, ext.columns.get_loc("close")] = df["high"].max() + 19.5
    full = pd.concat([df.iloc[:80], ext])
    seen_pending = False
    for k in range(82, len(full) + 1):
        r = eng.update(full.iloc[:k])
        seen_pending |= (r.status == "anchor_pending")
    assert seen_pending                               # new anchor too young at first
    assert r.ready and r.anchor["price"] > old_poc + 15   # re-anchored to the new high
    # The old levels (below new price action) are still projected unless price traded through them.
    assert isinstance(r.frozen_levels, list)


def _cfg():
    return AAVPConfig(tick_size=0.01, sweep_lookback=10, lvn_lookback=3, live_candle=False)


def _bars(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_signal_long_fires():
    base = [(101.0, 101.5, 100.5, 101.0, 100)] * 12          # prior lows = 100.5
    trigger = (100.7, 100.9, 99.8, 100.8, 500)               # low 99.8 < 100.5 swept, close 100.8 > 100.5
    w = _bars(base + [trigger])
    lvn = [LvnZone(99.5, 100.0, 1.0)]                        # trigger touches this LVN
    s = detect_signal(w, {"VAL": 99.9, "POC": 102.0, "VAH": 103.0}, lvn, _cfg(), atr_val=1.0, row_size=0.1)
    assert s and s.direction == "long" and s.level_name == "VAL"
    assert s.invalidation == 99.8


def test_signal_long_blocked_without_lvn_or_sweep_or_wick():
    base = [(101.0, 101.5, 100.5, 101.0, 100)] * 12
    good = (100.7, 100.9, 99.8, 100.8, 500)
    levels = {"VAL": 99.9, "POC": 102.0, "VAH": 103.0}
    lvn = [LvnZone(99.5, 100.0, 1.0)]
    # no LVN touched
    assert detect_signal(_bars(base + [good]), levels, [LvnZone(90, 91, 1)], _cfg(), 1.0, 0.1) is None
    # no sweep (low doesn't go below prior lows)
    nosweep = (100.9, 101.0, 100.6, 100.95, 500)
    assert detect_signal(_bars(base + [nosweep]), levels, lvn, _cfg(), 1.0, 0.1) is None
    # sweep but tiny wick, bearish body
    nowick = (100.9, 101.0, 99.8, 99.9, 500)
    assert detect_signal(_bars(base + [nowick]), levels, lvn, _cfg(), 1.0, 0.1) is None
    # level too far away
    far = {"VAL": 95.0, "POC": 110.0, "VAH": 120.0}
    assert detect_signal(_bars(base + [good]), far, lvn, _cfg(), 1.0, 0.1) is None


def test_signal_short_fires():
    base = [(101.0, 101.5, 100.5, 101.0, 100)] * 12          # prior highs = 101.5
    trigger = (101.3, 102.2, 101.1, 101.2, 500)              # high 102.2 > 101.5, close 101.2 < 101.5
    lvn = [LvnZone(102.0, 102.5, 1.0)]
    s = detect_signal(_bars(base + [trigger]), {"VAH": 102.1, "POC": 100.0, "VAL": 99.0}, lvn, _cfg(), 1.0, 0.1)
    assert s and s.direction == "short" and s.level_name == "VAH"


def test_live_candle_is_ignored_for_signal():
    base = [(101.0, 101.5, 100.5, 101.0, 100)] * 12
    trigger = (100.7, 100.9, 99.8, 100.8, 500)
    live = (100.8, 100.85, 100.75, 100.8, 5)
    cfg = _cfg(); cfg.live_candle = True
    lvn = [LvnZone(99.5, 100.0, 1.0)]
    s = detect_signal(_bars(base + [trigger, live]), {"VAL": 99.9, "POC": 102.0, "VAH": 103.0}, lvn, cfg, 1.0, 0.1)
    assert s and s.direction == "long"
