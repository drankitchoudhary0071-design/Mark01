"""
7. Volatility Squeeze / Breakout (TTM Squeeze style).

When Bollinger Bands are inside Keltner Channels (squeeze), wait for release
and trade in the direction of the momentum impulse.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class VolatilitySqueezeStrategy(Strategy):
    name = "volatility_squeeze"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "squeeze_bb_kc_mult": 1.5,  # Keltner ATR mult
            "mom_period": 12,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
            "min_squeeze_bars": 3,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        bb_l, bb_m, bb_u = ind.bollinger(out["close"], int(p["bb_period"]), float(p["bb_std"]))
        kc_l, kc_m, kc_u = ind.keltner(out, int(p["kc_period"]), float(p["squeeze_bb_kc_mult"]))
        out["bb_l"], out["bb_u"] = bb_l, bb_u
        out["kc_l"], out["kc_u"] = kc_l, kc_u
        out["squeeze"] = (out["bb_l"] > out["kc_l"]) & (out["bb_u"] < out["kc_u"])
        out["mom"] = out["close"] - out["close"].shift(int(p["mom_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))

        # Count consecutive squeeze bars
        sq = out["squeeze"].astype(int)
        groups = (sq != sq.shift()).cumsum()
        out["sq_run"] = sq.groupby(groups).cumsum()

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if pd.isna(row["atr"]) or row["atr"] <= 0 or pd.isna(row["mom"]):
                continue
            # Squeeze release: was in squeeze long enough, now not
            released = (
                bool(prev["squeeze"])
                and not bool(row["squeeze"])
                and int(prev["sq_run"]) >= int(p["min_squeeze_bars"])
            )
            if not released:
                continue

            if row["mom"] > 0:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="squeeze_release_long",
                    )
                )
            elif row["mom"] < 0:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="squeeze_release_short",
                    )
                )
        return signals
