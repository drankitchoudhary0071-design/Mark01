"""
10. Combined Momentum + Volatility Regime Filter.

Long/short residual momentum only when realized vol is in a mid-band
(not too quiet, not panic). Combines dual-lookback momentum with a vol gate.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class MomentumVolFilterStrategy(Strategy):
    name = "momentum_vol_filter"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "mom_lookback": 24,
            "mom_lookback_slow": 72,
            "vol_regime_lookback": 48,
            "vol_percentile_low": 0.3,  # trade when vol between p30–p80 of its history
            "vol_percentile_high": 0.8,
            "vol_hist": 200,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 2.5,
            "min_mom": 0.005,  # 0.5% move over fast lookback
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["mom_f"] = out["close"].pct_change(int(p["mom_lookback"]))
        out["mom_s"] = out["close"].pct_change(int(p["mom_lookback_slow"]))
        rets = out["close"].pct_change()
        out["rvol"] = ind.realized_vol(rets, int(p["vol_regime_lookback"]))
        out["vol_pctl"] = out["rvol"].rolling(int(p["vol_hist"])).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        out["atr"] = ind.atr(out, int(p["atr_period"]))

        signals: list[Signal] = []
        prev_side = None
        for i in range(1, len(out)):
            row = out.iloc[i]
            if any(pd.isna(row[c]) for c in ("mom_f", "mom_s", "vol_pctl", "atr")):
                continue
            if not (p["vol_percentile_low"] <= row["vol_pctl"] <= p["vol_percentile_high"]):
                prev_side = None
                continue

            # Agreement of fast & slow momentum
            long_ok = (
                row["mom_f"] > p["min_mom"]
                and row["mom_s"] > 0
            )
            short_ok = (
                row["mom_f"] < -p["min_mom"]
                and row["mom_s"] < 0
            )

            if long_ok and prev_side != "long":
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mom_vol_long",
                    )
                )
                prev_side = "long"
            elif short_ok and prev_side != "short":
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mom_vol_short",
                    )
                )
                prev_side = "short"
            elif not long_ok and not short_ok:
                prev_side = None
        return signals
