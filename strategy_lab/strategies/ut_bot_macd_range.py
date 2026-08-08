"""
UT Bot + MACD + Range Filter confluence strategy.

UT Bot (QuantNomad-style ATR trailing stop):
  Buy  params: key=1.5, ATR period=51
  Sell params: key=1.9, ATR period=41

MACD: fast=3, slow=61, signal=12

Range Filter (DonovanWall-style): period=51, multiplier=1.0

Long  : UT buy cross + MACD line > signal + close > range filter
Short : UT sell cross + MACD line < signal + close < range filter
Stops : ATR multiples (default 1.5 / 3.0)
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
        return {
            "ut_buy_key": 1.5,
            "ut_buy_atr": 51,
            "ut_sell_key": 1.9,
            "ut_sell_atr": 41,
            "macd_fast": 3,
            "macd_slow": 61,
            "macd_signal": 12,
            "range_period": 51,
            "range_mult": 1.0,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
            "max_hold_bars": 96,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[UT+MACD+RF] Non-standard MACD 3/61/12 + dual UT Bot keys — verify OOS.",
            "[UT+MACD+RF] Confluence of 3 indicators can overfit TV presets.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        close = df["close"].astype(float)
        ts = pd.to_datetime(df["timestamp"], utc=True)

        atr_buy = ind.atr(df, int(p["ut_buy_atr"]))
        atr_sell = ind.atr(df, int(p["ut_sell_atr"]))
        trail_buy = ut_bot_trail(close, atr_buy, float(p["ut_buy_key"]))
        trail_sell = ut_bot_trail(close, atr_sell, float(p["ut_sell_key"]))

        # Crosses (confirmed on closed bar)
        buy_cross = (close > trail_buy) & (close.shift(1) <= trail_buy.shift(1))
        sell_cross = (close < trail_sell) & (close.shift(1) >= trail_sell.shift(1))

        macd_line, macd_sig, _ = ind.macd(
            close, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
        )
        macd_bull = macd_line > macd_sig
        macd_bear = macd_line < macd_sig

        rfilt = range_filter(close, int(p["range_period"]), float(p["range_mult"]))
        above_rf = close > rfilt
        below_rf = close < rfilt

        atr14 = ind.atr(df, int(p["atr_period"]))
        hold = int(p["max_hold_bars"])
        stop_m = float(p["stop_atr"])
        tp_m = float(p["tp_atr"])

        long_ok = buy_cross & macd_bull & above_rf
        short_ok = sell_cross & macd_bear & below_rf

        signals: list[Signal] = []
        in_pos = False
        pos_side = None
        pos_stop = pos_tp = 0.0
        entry_i = -1

        for i in range(len(df)):
            if in_pos:
                hi = float(df["high"].iloc[i])
                lo = float(df["low"].iloc[i])
                exited = False
                if pos_side == "long" and (lo <= pos_stop or hi >= pos_tp):
                    exited = True
                if pos_side == "short" and (hi >= pos_stop or lo <= pos_tp):
                    exited = True
                if i - entry_i >= hold:
                    exited = True
                # Flip on opposite confluence signal
                if pos_side == "long" and bool(short_ok.iloc[i]):
                    exited = True
                if pos_side == "short" and bool(long_ok.iloc[i]):
                    exited = True
                if exited:
                    in_pos = False
                    pos_side = None

            if in_pos:
                continue

            a = float(atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) else np.nan
            if np.isnan(a) or a <= 0:
                continue
            px = float(close.iloc[i])
            tsi = pd.Timestamp(ts.iloc[i])

            if bool(long_ok.iloc[i]):
                stop, tp = atr_stop_tp(px, "long", a, stop_m, tp_m)
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
            elif bool(short_ok.iloc[i]):
                stop, tp = atr_stop_tp(px, "short", a, stop_m, tp_m)
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
