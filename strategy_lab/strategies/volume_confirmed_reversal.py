"""
15. Volume-Confirmed Reversal — extreme move + volume spike filter.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class VolumeConfirmedReversalStrategy(Strategy):
    name = "volume_confirmed_reversal"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "vol_lookback": 20,
            "vol_spike_mult": 1.5,  # volume > 1.5× SMA — textbook, not optimized
            "atr_period": 14,
            "stop_atr": 1.2,
            "tp_atr": 2.0,
            "pin_wick_ratio": 2.0,  # wick >= 2× body for rejection candle
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        out["vol_sma"] = out["volume"].rolling(int(p["vol_lookback"])).mean()
        out["body"] = (out["close"] - out["open"]).abs()
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            if any(pd.isna(row[c]) for c in ("rsi", "atr", "vol_sma")):
                continue
            if row["vol_sma"] <= 0 or row["atr"] <= 0:
                continue
            vol_ok = row["volume"] >= p["vol_spike_mult"] * row["vol_sma"]
            if not vol_ok:
                continue

            # Bullish rejection: RSI oversold + lower wick dominance
            bull_pin = row["body"] > 0 and row["lower_wick"] >= p["pin_wick_ratio"] * row["body"]
            bear_pin = row["body"] > 0 and row["upper_wick"] >= p["pin_wick_ratio"] * row["body"]

            if row["rsi"] < p["rsi_oversold"] and bull_pin:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                stop = min(stop, float(row["low"]) - 0.1 * float(row["atr"]))
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="vol_rev_long",
                    )
                )
            elif row["rsi"] > p["rsi_overbought"] and bear_pin:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                stop = max(stop, float(row["high"]) + 0.1 * float(row["atr"]))
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="vol_rev_short",
                    )
                )
        return signals
