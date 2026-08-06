"""
Gold Donchian Trend Continuation (GDTC)

Independent of PMTS. Designed from PAXG/USDT sample diagnostics:

What the data showed
--------------------
- Strong year-long gold uptrend (~$3380 → ~$4270 on this sample).
- 1H Donchian(20) upside breakouts had ~55% hit rate and +0.15% mean 12h
  forward return; downside breakouts did NOT show a symmetric edge.
- Median 5m ATR is only ~0.06% of price. Strategies that stop at ~1–1.5× 5m ATR
  are structurally crushed by ~0.30% round-trip costs (~2–5R of friction).
- Range is highest around 13–14 UTC (London/NY); Asian hours are quieter.

Edge hypothesis
---------------
Gold on crypto rails still trends when it breaks multi-day ranges (macro/USD
driven). We take *continuation* after a confirmed 1H channel break, but only
with a stop wide enough that costs are a fraction of 1R, and we skip short
breakouts unless price is already below a slow SMA (regime filter).

Rules
-----
1. On 1H: Donchian channel lookback N (default 20).
2. Breakout long when close > prior Donchian high; short when close < prior low
   AND close < SMA(100) (avoid shorting a bull).
3. Confirm on 5m within the next 3 hours: a close beyond the breakout level
   plus a small pullback that holds the level (no immediate failure).
4. Stop: opposite side of the 1H breakout candle ± buffer, floored at
   min_stop_pct (default 0.50% of price).
5. Target: rr_target × risk (default 2.0).
6. Session: entries 07:00–19:00 UTC only.
7. Cooldown: one trade per direction per 24h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy_pmts import Signal, OrderBlockSetup, atr, Direction


@dataclass
class GoldDonchianParams:
    donchian: int = 20
    sma_period: int = 100
    confirm_hours: int = 3
    rr_target: float = 2.0
    min_stop_pct: float = 0.005  # 0.50%
    stop_buffer_atr: float = 0.25  # 1H ATR fraction beyond breakout bar
    session_start: int = 7
    session_end: int = 19
    allow_shorts_below_sma: bool = True
    long_only: bool = False
    cooldown_hours: int = 24


def generate_gold_donchian_signals(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    params: GoldDonchianParams | None = None,
) -> list[Signal]:
    params = params or GoldDonchianParams()
    h = df_1h.copy().reset_index(drop=True)
    h["atr"] = atr(h, 14)
    h["don_high"] = h["high"].rolling(params.donchian).max().shift(1)
    h["don_low"] = h["low"].rolling(params.donchian).min().shift(1)
    h["sma"] = h["close"].rolling(params.sma_period, min_periods=params.sma_period).mean()

    # Breakout events on 1H close
    events: list[dict] = []
    for i in range(len(h)):
        row = h.iloc[i]
        if np.isnan(row["don_high"]) or np.isnan(row["don_low"]) or np.isnan(row["atr"]):
            continue
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        direction: Direction | None = None
        level = None
        if row["close"] > row["don_high"]:
            direction = "long"
            level = float(row["don_high"])
        elif row["close"] < row["don_low"]:
            if params.long_only:
                continue
            if params.allow_shorts_below_sma:
                if np.isnan(row["sma"]) or row["close"] >= row["sma"]:
                    continue
            direction = "short"
            level = float(row["don_low"])
        if direction is None:
            continue
        events.append(
            {
                "i": i,
                "ts": ts,
                "direction": direction,
                "level": level,
                "bar_high": float(row["high"]),
                "bar_low": float(row["low"]),
                "atr": float(row["atr"]),
                "close": float(row["close"]),
            }
        )

    df5 = df_5m.copy().reset_index(drop=True)
    signals: list[Signal] = []
    last_signal_time: dict[str, pd.Timestamp] = {"long": pd.Timestamp("1970-01-01", tz="UTC"), "short": pd.Timestamp("1970-01-01", tz="UTC")}

    for ev in events:
        # 5m confirmation window
        t0 = ev["ts"]
        t1 = t0 + pd.Timedelta(hours=params.confirm_hours)
        win = df5[(df5["timestamp"] > t0) & (df5["timestamp"] <= t1)]
        if win.empty:
            continue

        direction = ev["direction"]
        level = ev["level"]
        confirmed = False
        entry_row = None
        # Require price to hold breakout: first pullback toward level that doesn't close back through it
        broke = False
        for _, r in win.iterrows():
            ts = pd.Timestamp(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            hour = ts.tz_convert("UTC").hour
            if not (params.session_start <= hour < params.session_end):
                continue

            if direction == "long":
                if r["close"] > level:
                    broke = True
                if broke and r["low"] <= level * 1.001 and r["close"] >= level:
                    # pullback hold
                    confirmed = True
                    entry_row = r
                    break
            else:
                if r["close"] < level:
                    broke = True
                if broke and r["high"] >= level * 0.999 and r["close"] <= level:
                    confirmed = True
                    entry_row = r
                    break

        # Fallback: if no clean retest, take first close beyond level after 30m
        if not confirmed:
            later = win[win["timestamp"] >= t0 + pd.Timedelta(minutes=30)]
            for _, r in later.iterrows():
                ts = pd.Timestamp(r["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                hour = ts.tz_convert("UTC").hour
                if not (params.session_start <= hour < params.session_end):
                    continue
                if direction == "long" and r["close"] > level:
                    confirmed = True
                    entry_row = r
                    break
                if direction == "short" and r["close"] < level:
                    confirmed = True
                    entry_row = r
                    break

        if not confirmed or entry_row is None:
            continue

        ts = pd.Timestamp(entry_row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts - last_signal_time[direction] < pd.Timedelta(hours=params.cooldown_hours):
            continue

        entry = float(entry_row["close"])
        buf = params.stop_buffer_atr * ev["atr"]
        if direction == "long":
            stop = ev["bar_low"] - buf
            stop = min(stop, entry * (1 - params.min_stop_pct))
            risk = entry - stop
            if risk / entry < params.min_stop_pct * 0.95:
                stop = entry * (1 - params.min_stop_pct)
                risk = entry - stop
            tp = entry + params.rr_target * risk
        else:
            stop = ev["bar_high"] + buf
            stop = max(stop, entry * (1 + params.min_stop_pct))
            risk = stop - entry
            if risk / entry < params.min_stop_pct * 0.95:
                stop = entry * (1 + params.min_stop_pct)
                risk = stop - entry
            tp = entry - params.rr_target * risk

        setup = OrderBlockSetup(
            direction=direction,
            impulse_start_idx=ev["i"],
            impulse_end_idx=ev["i"],
            impulse_start_price=level,
            impulse_end_price=ev["close"],
            ob_high=ev["bar_high"],
            ob_low=ev["bar_low"],
            fib_618=level,
            fib_790=level,
            zone_top=max(level, entry),
            zone_bottom=min(level, entry),
            structure_break_time=ev["ts"],
            structure_break_idx=ev["i"],
        )
        signals.append(
            Signal(
                timestamp=ts,
                direction=direction,
                entry=entry,
                stop=float(stop),
                take_profit=float(tp),
                setup=setup,
                pattern="donchian_retest",
                bos_time=ev["ts"],
                meta={"level": level, "donchian": params.donchian},
            )
        )
        last_signal_time[direction] = ts

    return signals


# Backwards-compatible alias used by pipeline during transition
GoldMRParams = GoldDonchianParams
generate_gold_mr_signals = generate_gold_donchian_signals
