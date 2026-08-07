"""
13. Regime Detection + Adaptive Parameter Switching.

Detect trending vs ranging (ADX + efficiency ratio), then switch:
  - trending → MA momentum entries (wider stops)
  - ranging → Bollinger mean-reversion entries (tighter stops)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class RegimeAdaptiveStrategy(Strategy):
    name = "regime_adaptive"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "adx_period": 14,
            "adx_trend_threshold": 25,
            "er_period": 20,  # Kaufman efficiency ratio
            "er_trend_threshold": 0.3,  # textbook-ish ER cut
            "fast_ma": 20,
            "slow_ma": 50,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "trend_stop_atr": 1.5,
            "trend_tp_atr": 3.0,
            "range_stop_atr": 1.2,
            "range_tp_atr": 1.8,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["fast"] = ind.sma(out["close"], int(p["fast_ma"]))
        out["slow"] = ind.sma(out["close"], int(p["slow_ma"]))
        bb_l, bb_m, bb_u = ind.bollinger(out["close"], int(p["bb_period"]), float(p["bb_std"]))
        out["bb_l"], out["bb_m"], out["bb_u"] = bb_l, bb_m, bb_u
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        adx_df = ind.adx(out, int(p["adx_period"]))
        out = out.join(adx_df)
        # Efficiency ratio: net change / sum |changes|
        change = out["close"].diff(int(p["er_period"])).abs()
        volatility = out["close"].diff().abs().rolling(int(p["er_period"])).sum()
        out["er"] = change / volatility.replace(0, pd.NA)

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if any(pd.isna(row[c]) for c in ("atr", "adx", "er", "fast", "slow", "bb_l")):
                continue
            trending = row["adx"] >= p["adx_trend_threshold"] and row["er"] >= p["er_trend_threshold"]

            if trending:
                if prev["fast"] <= prev["slow"] and row["fast"] > row["slow"]:
                    entry = float(row["close"])
                    stop, tp = atr_stop_tp(
                        entry, "long", float(row["atr"]), p["trend_stop_atr"], p["trend_tp_atr"]
                    )
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="long",
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            pattern="regime_trend_long",
                        )
                    )
                elif prev["fast"] >= prev["slow"] and row["fast"] < row["slow"]:
                    entry = float(row["close"])
                    stop, tp = atr_stop_tp(
                        entry, "short", float(row["atr"]), p["trend_stop_atr"], p["trend_tp_atr"]
                    )
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="short",
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            pattern="regime_trend_short",
                        )
                    )
            else:
                # Ranging → fade bands
                if row["close"] <= row["bb_l"]:
                    entry = float(row["close"])
                    stop, tp = atr_stop_tp(
                        entry, "long", float(row["atr"]), p["range_stop_atr"], p["range_tp_atr"]
                    )
                    tp = min(tp, float(row["bb_m"])) if row["bb_m"] > entry else tp
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="long",
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            pattern="regime_range_long",
                        )
                    )
                elif row["close"] >= row["bb_u"]:
                    entry = float(row["close"])
                    stop, tp = atr_stop_tp(
                        entry, "short", float(row["atr"]), p["range_stop_atr"], p["range_tp_atr"]
                    )
                    tp = max(tp, float(row["bb_m"])) if row["bb_m"] < entry else tp
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(row["timestamp"]),
                            direction="short",
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            pattern="regime_range_short",
                        )
                    )
        return signals
