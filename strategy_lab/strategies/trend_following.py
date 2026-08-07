"""
1. Trend Following — MA crossover filtered by ADX.

Library: pandas signal gen + shared event-driven engine
(vectorbt would shine for MA grids, but we keep fills identical across families).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "fast_ma": 20,
            "slow_ma": 50,
            "adx_period": 14,
            "adx_threshold": 25,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,  # ~1:2 RR vs stop — textbook, not optimized
            "use_adx_filter": True,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["fast"] = ind.sma(out["close"], int(p["fast_ma"]))
        out["slow"] = ind.sma(out["close"], int(p["slow_ma"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        adx_df = ind.adx(out, int(p["adx_period"]))
        out = out.join(adx_df)

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if pd.isna(row["fast"]) or pd.isna(row["slow"]) or pd.isna(row["atr"]):
                continue
            if row["atr"] <= 0:
                continue
            if p["use_adx_filter"] and (pd.isna(row["adx"]) or row["adx"] < p["adx_threshold"]):
                continue

            # Bullish cross
            if prev["fast"] <= prev["slow"] and row["fast"] > row["slow"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="ma_cross_up",
                    )
                )
            # Bearish cross
            elif prev["fast"] >= prev["slow"] and row["fast"] < row["slow"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="ma_cross_down",
                    )
                )
        return signals
