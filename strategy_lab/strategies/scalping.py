"""
5. Scalping — low-timeframe momentum bursts with tight SL/TP.

Intended timeframe: 5m. Under realistic costs, most scalp edges vanish —
that is intentional honesty, not a bug.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class ScalpingStrategy(Strategy):
    name = "scalping"
    library = "pandas+shared_engine (5m)"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "ema_fast": 8,
            "ema_slow": 21,
            "rsi_period": 7,
            "rsi_long_min": 55,
            "rsi_short_max": 45,
            "atr_period": 14,
            "scalp_atr_stop": 0.8,  # CURVE-FIT RISK
            "scalp_rr": 1.2,  # CURVE-FIT RISK
            "min_atr_pct": 0.0005,  # skip dead markets
            "max_hold_bars": 12,  # ~1h on 5m
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["ema_f"] = ind.ema(out["close"], int(p["ema_fast"]))
        out["ema_s"] = ind.ema(out["close"], int(p["ema_slow"]))
        out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))

        signals: list[Signal] = []
        stop_atr = float(p["scalp_atr_stop"])
        tp_atr = stop_atr * float(p["scalp_rr"])

        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if any(pd.isna(row[c]) for c in ("ema_f", "ema_s", "rsi", "atr")):
                continue
            atr_pct = float(row["atr"]) / float(row["close"]) if row["close"] else 0.0
            if atr_pct < p["min_atr_pct"]:
                continue

            # Momentum burst: fast EMA cross + RSI confirmation
            long_cross = prev["ema_f"] <= prev["ema_s"] and row["ema_f"] > row["ema_s"]
            short_cross = prev["ema_f"] >= prev["ema_s"] and row["ema_f"] < row["ema_s"]

            if long_cross and row["rsi"] >= p["rsi_long_min"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), stop_atr, tp_atr)
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="scalp_long",
                        max_hold_bars=int(p["max_hold_bars"]),
                    )
                )
            elif short_cross and row["rsi"] <= p["rsi_short_max"]:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), stop_atr, tp_atr)
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="scalp_short",
                        max_hold_bars=int(p["max_hold_bars"]),
                    )
                )
        return signals
