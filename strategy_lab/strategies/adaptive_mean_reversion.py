"""
14. Adaptive Mean-Reversion — dynamic mean length by detected regime.

In quiet/ranging regimes use a shorter mean (faster reversion target);
in trending regimes use a longer mean (avoid fighting the trend) and
only fade extreme z-scores.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class AdaptiveMeanReversionStrategy(Strategy):
    name = "adaptive_mean_reversion"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "fast_mean": 20,
            "slow_mean": 50,
            "adx_period": 14,
            "adx_trend_threshold": 25,
            "z_entry_range": 2.0,
            "z_entry_trend": 2.5,  # stricter when trending
            "atr_period": 14,
            "stop_atr": 1.2,
            "tp_atr": 1.8,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["mean_fast"] = ind.sma(out["close"], int(p["fast_mean"]))
        out["mean_slow"] = ind.sma(out["close"], int(p["slow_mean"]))
        out["std_fast"] = out["close"].rolling(int(p["fast_mean"])).std()
        out["std_slow"] = out["close"].rolling(int(p["slow_mean"])).std()
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        adx_df = ind.adx(out, int(p["adx_period"]))
        out = out.join(adx_df)

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            if any(pd.isna(row[c]) for c in ("mean_fast", "mean_slow", "atr", "adx")):
                continue
            trending = row["adx"] >= p["adx_trend_threshold"]
            if trending:
                mu, sd, z_thr = row["mean_slow"], row["std_slow"], p["z_entry_trend"]
                pattern = "amr_trend_fade"
            else:
                mu, sd, z_thr = row["mean_fast"], row["std_fast"], p["z_entry_range"]
                pattern = "amr_range_fade"
            if pd.isna(sd) or sd <= 0:
                continue
            z = (row["close"] - mu) / sd
            entry = float(row["close"])
            if z <= -z_thr:
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                # Target dynamic mean if closer
                if mu > entry:
                    tp = min(tp, float(mu))
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern=pattern + "_long",
                        meta={"z": float(z), "regime": "trend" if trending else "range"},
                    )
                )
            elif z >= z_thr:
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                if mu < entry:
                    tp = max(tp, float(mu))
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern=pattern + "_short",
                        meta={"z": float(z), "regime": "trend" if trending else "range"},
                    )
                )
        return signals
