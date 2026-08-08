"""
8. Fair Value Gap + Order Block detection (SMC-lite).

Simplified, rule-based Smart Money Concepts:
- Bullish FVG: candle[i-2].high < candle[i].low (gap up)
- Bearish FVG: candle[i-2].low > candle[i].high (gap down)
- Order block: last opposing candle before an impulsive displace

These heuristics are highly discretionary in discretionary SMC trading;
thresholds are flagged as curve-fit risks. This is a research scaffold,
not a claim of edge.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class SmcFvgObStrategy(Strategy):
    name = "smc_fvg_ob"
    library = "pandas SMC heuristics + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "fvg_min_gap_atr": 0.25,  # CURVE-FIT RISK
            "ob_lookback": 20,  # CURVE-FIT RISK
            "impulse_atr": 1.2,  # body size to call impulse
            "atr_period": 14,
            "stop_atr": 1.0,
            "tp_atr": 2.0,
            "max_mitigation_bars": 30,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        out["body"] = (out["close"] - out["open"]).abs()

        signals: list[Signal] = []
        # Pending FVGs awaiting mitigation (price revisiting the gap)
        pending: list[dict] = []

        for i in range(2, len(out)):
            row = out.iloc[i]
            c0 = out.iloc[i - 2]
            c1 = out.iloc[i - 1]
            if pd.isna(row["atr"]) or row["atr"] <= 0:
                continue
            atr_v = float(row["atr"])

            # Detect new FVGs on bar i (3-candle pattern)
            gap_up = float(c0["high"]) < float(row["low"])
            gap_down = float(c0["low"]) > float(row["high"])
            gap_size_up = float(row["low"]) - float(c0["high"]) if gap_up else 0.0
            gap_size_dn = float(c0["low"]) - float(row["high"]) if gap_down else 0.0

            # Impulse filter on middle candle
            impulse = float(c1["body"]) >= p["impulse_atr"] * atr_v

            if gap_up and gap_size_up >= p["fvg_min_gap_atr"] * atr_v and impulse and c1["close"] > c1["open"]:
                # Bullish FVG zone [c0.high, row.low]; OB = last down candle in lookback
                ob_low, ob_high = self._find_ob(out, i, bullish=True, lookback=int(p["ob_lookback"]))
                pending.append(
                    {
                        "dir": "long",
                        "zone_low": float(c0["high"]),
                        "zone_high": float(row["low"]),
                        "ob_low": ob_low,
                        "ob_high": ob_high,
                        "created": i,
                        "atr": atr_v,
                    }
                )
            if gap_down and gap_size_dn >= p["fvg_min_gap_atr"] * atr_v and impulse and c1["close"] < c1["open"]:
                ob_low, ob_high = self._find_ob(out, i, bullish=False, lookback=int(p["ob_lookback"]))
                pending.append(
                    {
                        "dir": "short",
                        "zone_low": float(row["high"]),
                        "zone_high": float(c0["low"]),
                        "ob_low": ob_low,
                        "ob_high": ob_high,
                        "created": i,
                        "atr": atr_v,
                    }
                )

            # Mitigation: price trades back into zone
            still: list[dict] = []
            for fz in pending:
                if i - fz["created"] > int(p["max_mitigation_bars"]):
                    continue  # expire
                lo = float(row["low"])
                hi = float(row["high"])
                if fz["dir"] == "long":
                    # revisit FVG / OB from above
                    touched = lo <= fz["zone_high"] and hi >= fz["zone_low"]
                    if touched:
                        entry = float(row["close"])
                        stop = min(fz["ob_low"] or entry, entry) - 0.1 * atr_v
                        # Ensure stop below entry
                        if stop >= entry:
                            stop = entry - p["stop_atr"] * atr_v
                        _, tp = atr_stop_tp(entry, "long", atr_v, p["stop_atr"], p["tp_atr"])
                        # Use structural stop if wider
                        stop = min(stop, entry - 0.5 * atr_v)
                        signals.append(
                            Signal(
                                timestamp=pd.Timestamp(row["timestamp"]),
                                direction="long",
                                entry=entry,
                                stop=stop,
                                take_profit=tp,
                                pattern="fvg_ob_long",
                            )
                        )
                        continue  # consumed
                else:
                    touched = hi >= fz["zone_low"] and lo <= fz["zone_high"]
                    if touched:
                        entry = float(row["close"])
                        stop = max(fz["ob_high"] or entry, entry) + 0.1 * atr_v
                        if stop <= entry:
                            stop = entry + p["stop_atr"] * atr_v
                        _, tp = atr_stop_tp(entry, "short", atr_v, p["stop_atr"], p["tp_atr"])
                        stop = max(stop, entry + 0.5 * atr_v)
                        signals.append(
                            Signal(
                                timestamp=pd.Timestamp(row["timestamp"]),
                                direction="short",
                                entry=entry,
                                stop=stop,
                                take_profit=tp,
                                pattern="fvg_ob_short",
                            )
                        )
                        continue
                still.append(fz)
            pending = still

        return signals

    @staticmethod
    def _find_ob(
        df: pd.DataFrame, i: int, bullish: bool, lookback: int
    ) -> tuple[float | None, float | None]:
        start = max(0, i - lookback)
        window = df.iloc[start:i]
        if bullish:
            # last bearish candle
            bears = window[window["close"] < window["open"]]
            if bears.empty:
                return None, None
            candle = bears.iloc[-1]
        else:
            bulls = window[window["close"] > window["open"]]
            if bulls.empty:
                return None, None
            candle = bulls.iloc[-1]
        return float(candle["low"]), float(candle["high"])
