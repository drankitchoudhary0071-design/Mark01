"""
6. Event-driven / Regime-filtered momentum.

Trade breakouts only when a volatility + trend regime filter is ON
(high ADX + expanding realized vol). Avoids chop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class RegimeMomentumStrategy(Strategy):
    name = "regime_momentum"
    library = "pandas+shared_engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "donchian_period": 20,
            "adx_period": 14,
            "adx_threshold": 20,  # slightly below textbook 25 so events fire on 1h crypto
            "vol_lookback": 48,
            "vol_expand_mult": 1.0,  # require vol ≥ rolling median (not 1.2× — too sparse)
            "require_vol_filter": True,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
            "mom_lookback": 24,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        lo, hi = ind.donchian(out, int(p["donchian_period"]))
        # Prior-bar channel so a close can break out (inclusive Donchian never breaks)
        out["dc_lo"], out["dc_hi"] = lo.shift(1), hi.shift(1)
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        adx_df = ind.adx(out, int(p["adx_period"]))
        out = out.join(adx_df)
        rets = out["close"].pct_change()
        out["rvol"] = ind.realized_vol(rets, int(p["vol_lookback"]))
        out["rvol_med"] = out["rvol"].rolling(int(p["vol_lookback"]) * 3).median()
        out["mom"] = out["close"].pct_change(int(p["mom_lookback"]))

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            needed = ("dc_hi", "dc_lo", "atr", "adx", "mom")
            if any(pd.isna(row[c]) for c in needed):
                continue
            if row["adx"] < p["adx_threshold"]:
                continue
            if p["require_vol_filter"]:
                if pd.isna(row["rvol"]) or pd.isna(row["rvol_med"]):
                    continue
                if row["rvol"] < p["vol_expand_mult"] * row["rvol_med"]:
                    continue

            # Breakout events (close through Donchian + momentum agreement)
            if prev["close"] <= prev["dc_hi"] and row["close"] > row["dc_hi"] and row["mom"] > 0:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="regime_breakout_long",
                    )
                )
            elif prev["close"] >= prev["dc_lo"] and row["close"] < row["dc_lo"] and row["mom"] < 0:
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="regime_breakout_short",
                    )
                )
        return signals
