"""
UT Bot + MACD + Range Filter confluence strategy.

Optimized defaults (BTCUSDT 30m, costs~0.30% RT, $10k, 365d):
  UT key/ATR     : 3.0 / 20  (buy & sell)
  MACD           : 8 / 21 / 9
  Range Filter   : 50 / 3.0 + slope filter
  Entry mode     : MACD cross same bar as UT cross
  Stops          : UT trail as SL, TP = 6×ATR14
  Cooldown       : 4 bars
  Full-year lab  : ~39 trades, WR~51%, MaxDD~-3.9%, Ret~+6.3%

Original user TV preset (1.5/51, 1.9/41, MACD 3/61/12, RF 51/1.0)
loses heavily on 15m/30m after costs — keep via params if needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


def ut_bot_trail(close: pd.Series, atr_s: pd.Series, key: float) -> pd.Series:
    """ATR trailing stop line (UT Bot)."""
    c = close.to_numpy(dtype=float)
    n_loss = (key * atr_s).to_numpy(dtype=float)
    n = len(c)
    trail = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(n_loss[i]) or np.isnan(c[i]):
            continue
        prev = trail[i - 1] if i > 0 and not np.isnan(trail[i - 1]) else np.nan
        if np.isnan(prev):
            trail[i] = c[i] - n_loss[i]
            continue
        prev_c = c[i - 1]
        if c[i] > prev and prev_c > prev:
            trail[i] = max(prev, c[i] - n_loss[i])
        elif c[i] < prev and prev_c < prev:
            trail[i] = min(prev, c[i] + n_loss[i])
        elif c[i] > prev:
            trail[i] = c[i] - n_loss[i]
        else:
            trail[i] = c[i] + n_loss[i]
    return pd.Series(trail, index=close.index)


def range_filter(close: pd.Series, period: int, mult: float) -> pd.Series:
    """Smooth range filter (period / multiplier)."""
    x = close.astype(float)
    wper = period * 2 - 1
    avrng = ind.ema((x - x.shift(1)).abs(), period)
    smrng = ind.ema(avrng, wper) * mult
    c = x.to_numpy(dtype=float)
    r = smrng.to_numpy(dtype=float)
    n = len(c)
    filt = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(c[i]) or np.isnan(r[i]):
            continue
        if i == 0 or np.isnan(filt[i - 1]):
            filt[i] = c[i]
            continue
        prev = filt[i - 1]
        if c[i] > prev:
            filt[i] = prev if (c[i] - r[i]) < prev else (c[i] - r[i])
        else:
            filt[i] = prev if (c[i] + r[i]) > prev else (c[i] + r[i])
    return pd.Series(filt, index=close.index)


class UtBotMacdRangeStrategy(Strategy):
    name = "ut_bot_macd_range"
    library = "pandas UT Bot + MACD + Range Filter"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        # Optimized profitable preset (BTC 30m)
        return {
            "ut_buy_key": 3.0,
            "ut_buy_atr": 20,
            "ut_sell_key": 3.0,
            "ut_sell_atr": 20,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 9,
            "range_period": 50,
            "range_mult": 3.0,
            "atr_period": 14,
            "stop_atr": 2.5,
            "tp_atr": 6.0,
            "max_hold_bars": 80,
            "entry_mode": "macd_cross",  # or "state"
            "rf_slope": True,
            "cooldown": 4,
            "use_trail_stop": True,  # SL = UT trail; TP = tp_atr × ATR
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[UT+MACD+RF] Defaults tuned on BTC 30m 365d — re-validate OOS before live.",
            "[UT+MACD+RF] Original 15m TV preset loses after ~0.30% RT costs.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        close = df["close"].astype(float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
        c = close.to_numpy(dtype=float)

        atr_buy = ind.atr(df, int(p["ut_buy_atr"]))
        atr_sell = ind.atr(df, int(p["ut_sell_atr"]))
        trail_buy = ut_bot_trail(close, atr_buy, float(p["ut_buy_key"])).to_numpy()
        trail_sell = ut_bot_trail(close, atr_sell, float(p["ut_sell_key"])).to_numpy()

        buy_cross = (c > trail_buy) & (np.roll(c, 1) <= np.roll(trail_buy, 1))
        sell_cross = (c < trail_sell) & (np.roll(c, 1) >= np.roll(trail_sell, 1))
        buy_cross[0] = False
        sell_cross[0] = False

        macd_line, macd_sig, _ = ind.macd(
            close, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
        )
        ml = macd_line.to_numpy(dtype=float)
        ms = macd_sig.to_numpy(dtype=float)
        if str(p.get("entry_mode", "state")) == "macd_cross":
            macd_bull = (ml > ms) & (np.roll(ml, 1) <= np.roll(ms, 1))
            macd_bear = (ml < ms) & (np.roll(ml, 1) >= np.roll(ms, 1))
            macd_bull[0] = False
            macd_bear[0] = False
        else:
            macd_bull = ml > ms
            macd_bear = ml < ms

        rfilt = range_filter(close, int(p["range_period"]), float(p["range_mult"])).to_numpy()
        if bool(p.get("rf_slope", False)):
            above_rf = (c > rfilt) & (rfilt > np.roll(rfilt, 1))
            below_rf = (c < rfilt) & (rfilt < np.roll(rfilt, 1))
            above_rf[0] = False
            below_rf[0] = False
        else:
            above_rf = c > rfilt
            below_rf = c < rfilt

        long_ok = buy_cross & macd_bull & above_rf
        short_ok = sell_cross & macd_bear & below_rf

        atr14 = ind.atr(df, int(p["atr_period"])).to_numpy(dtype=float)
        hold = int(p["max_hold_bars"])
        cooldown = int(p.get("cooldown", 0))
        stop_m = float(p["stop_atr"])
        tp_m = float(p["tp_atr"])
        use_trail = bool(p.get("use_trail_stop", False))

        signals: list[Signal] = []
        in_pos = False
        pos_side = None
        pos_stop = pos_tp = 0.0
        entry_i = -1
        last_exit = -10_000

        for i in range(len(df)):
            if in_pos:
                exited = False
                if pos_side == "long" and (low[i] <= pos_stop or high[i] >= pos_tp):
                    exited = True
                elif pos_side == "short" and (high[i] >= pos_stop or low[i] <= pos_tp):
                    exited = True
                if i - entry_i >= hold:
                    exited = True
                if exited:
                    in_pos = False
                    pos_side = None
                    last_exit = i
                else:
                    continue

            if i - last_exit < cooldown:
                continue

            a = atr14[i]
            if not np.isfinite(a) or a <= 0 or not np.isfinite(c[i]):
                continue
            px = float(c[i])
            tsi = pd.Timestamp(ts[i])

            if long_ok[i]:
                if use_trail:
                    stop = float(trail_buy[i])
                    tp = px + tp_m * float(a)
                    if not np.isfinite(stop) or stop >= px:
                        continue
                else:
                    stop, tp = atr_stop_tp(px, "long", float(a), stop_m, tp_m)
                signals.append(
                    Signal(
                        timestamp=tsi,
                        direction="long",
                        entry=px,
                        stop=stop,
                        take_profit=tp,
                        pattern="ut_macd_rf_long",
                        max_hold_bars=hold,
                    )
                )
                in_pos = True
                pos_side = "long"
                pos_stop, pos_tp, entry_i = stop, tp, i
            elif short_ok[i]:
                if use_trail:
                    stop = float(trail_sell[i])
                    tp = px - tp_m * float(a)
                    if not np.isfinite(stop) or stop <= px:
                        continue
                else:
                    stop, tp = atr_stop_tp(px, "short", float(a), stop_m, tp_m)
                signals.append(
                    Signal(
                        timestamp=tsi,
                        direction="short",
                        entry=px,
                        stop=stop,
                        take_profit=tp,
                        pattern="ut_macd_rf_short",
                        max_hold_bars=hold,
                    )
                )
                in_pos = True
                pos_side = "short"
                pos_stop, pos_tp, entry_i = stop, tp, i

        return signals
