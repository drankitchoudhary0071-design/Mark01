"""
2. Mean Reversion — Bollinger Band touch + RSI confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "stop_atr": 1.2,
            "tp_atr": 1.8,  # mean-reversion often < 1:2; not tuned
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        lower, mid, upper = ind.bollinger(out["close"], int(p["bb_period"]), float(p["bb_std"]))
        out["bb_lower"], out["bb_mid"], out["bb_upper"] = lower, mid, upper
        out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            if any(pd.isna(row[c]) for c in ("bb_lower", "bb_upper", "rsi", "atr")):
                continue
            if row["atr"] <= 0:
                continue

            # Long: close below lower band + RSI oversold
            if row["close"] <= row["bb_lower"] and row["rsi"] < p["rsi_oversold"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                # Prefer mid-band as soft target if closer than ATR tp
                mid_tp = float(row["bb_mid"])
                if mid_tp > entry:
                    tp = min(tp, mid_tp)
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="bb_rsi_long",
                    )
                )
            # Short: close above upper band + RSI overbought
            elif row["close"] >= row["bb_upper"] and row["rsi"] > p["rsi_overbought"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                mid_tp = float(row["bb_mid"])
                if mid_tp < entry:
                    tp = max(tp, mid_tp)
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="bb_rsi_short",
                    )
                )
        return signals
