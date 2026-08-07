"""
16. RSI Divergence Filtering — price HH/LL vs RSI non-confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class RsiDivergenceStrategy(Strategy):
    name = "rsi_divergence"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "swing_lookback": 5,  # local pivot left/right
            "max_pivot_gap": 40,  # max bars between pivots
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 2.5,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        lb = int(p["swing_lookback"])

        # Pivot highs / lows
        highs, lows = [], []
        for i in range(lb, len(out) - lb):
            window_h = out["high"].iloc[i - lb : i + lb + 1]
            window_l = out["low"].iloc[i - lb : i + lb + 1]
            if out["high"].iloc[i] >= window_h.max():
                highs.append(i)
            if out["low"].iloc[i] <= window_l.min():
                lows.append(i)

        signals: list[Signal] = []
        # Bearish divergence: price higher high, RSI lower high
        for a, b in zip(highs, highs[1:]):
            if b - a > int(p["max_pivot_gap"]):
                continue
            price_hh = out["high"].iloc[b] > out["high"].iloc[a]
            rsi_lh = out["rsi"].iloc[b] < out["rsi"].iloc[a]
            if price_hh and rsi_lh and not pd.isna(out["atr"].iloc[b]) and out["atr"].iloc[b] > 0:
                entry = float(out["close"].iloc[b])
                atr_v = float(out["atr"].iloc[b])
                stop, tp = atr_stop_tp(entry, "short", atr_v, p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(out["timestamp"].iloc[b]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="rsi_bear_div",
                    )
                )

        # Bullish divergence: price lower low, RSI higher low
        for a, b in zip(lows, lows[1:]):
            if b - a > int(p["max_pivot_gap"]):
                continue
            price_ll = out["low"].iloc[b] < out["low"].iloc[a]
            rsi_hl = out["rsi"].iloc[b] > out["rsi"].iloc[a]
            if price_ll and rsi_hl and not pd.isna(out["atr"].iloc[b]) and out["atr"].iloc[b] > 0:
                entry = float(out["close"].iloc[b])
                atr_v = float(out["atr"].iloc[b])
                stop, tp = atr_stop_tp(entry, "long", atr_v, p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(out["timestamp"].iloc[b]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="rsi_bull_div",
                    )
                )

        signals.sort(key=lambda s: s.timestamp)
        return signals
