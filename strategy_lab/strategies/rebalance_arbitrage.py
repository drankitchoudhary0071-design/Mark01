"""
17. Index/ETF Rebalancing Arbitrage (simulated calendar pressure).

Applicability note
------------------
True index-rebalance arb needs constituent weights, AUM flows, and the
index reconstitution calendar. Spot PAXG/USDT and BTC/USDT are **not**
equity-index constituents, so this module simulates a *proxy*: month-end
/ quarter-end flow pressure (documented in crypto/gold literature as a
weak calendar effect). Results are research-only and flagged as such.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class RebalanceArbitrageStrategy(Strategy):
    name = "rebalance_arbitrage"
    library = "pandas calendar proxy + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "pre_days": 2,  # enter N calendar days before month-end
            "hold_bars": 48,  # ~2 days on 1h
            "quarter_end_only": False,  # if True, only Mar/Jun/Sep/Dec
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 2.0,
            # Direction heuristic: prior 20d return — fade into rebalance (mean-revert flow)
            "mom_lookback": 20 * 24,  # ~20d on 1h
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[APPLICABILITY] Spot crypto/gold is not an equity-index constituent — "
            "this is a month-end calendar PROXY, not true ETF rebalance arb.",
            "[CURVE-FIT RISK] pre_days / hold_bars are research defaults.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["atr"] = ind.atr(out, int(p["atr_period"]))
        out["mom"] = out["close"].pct_change(int(p["mom_lookback"]))
        ts = pd.to_datetime(out["timestamp"], utc=True)
        out["_date"] = ts.dt.date
        out["_month"] = ts.dt.month
        out["_day"] = ts.dt.day

        # Month-end dates present in sample (tz-naive period key to avoid warning)
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        month_ends = (
            pd.DataFrame({"t": ts, "ym": ts_naive.dt.to_period("M")})
            .groupby("ym")["t"]
            .max()
        )
        me_set = {pd.Timestamp(t).tz_convert("UTC").normalize() for t in month_ends}

        signals: list[Signal] = []
        fired_months: set[str] = set()
        for i in range(len(out)):
            row = out.iloc[i]
            if pd.isna(row["atr"]) or row["atr"] <= 0 or pd.isna(row["mom"]):
                continue
            t = pd.Timestamp(row["timestamp"]).tz_convert("UTC")
            if p["quarter_end_only"] and t.month not in (3, 6, 9, 12):
                continue
            # Find this month's end
            ym = f"{t.year}-{t.month:02d}"
            if ym in fired_months:
                continue
            # Candidate month-end in me_set
            candidates = [me for me in me_set if me.year == t.year and me.month == t.month]
            if not candidates:
                continue
            me = candidates[0]
            days_to_me = (me.normalize() - t.normalize()).days
            if days_to_me != int(p["pre_days"]):
                continue
            # Fade prior momentum into rebalance window
            if row["mom"] > 0:
                direction = "short"
            else:
                direction = "long"
            entry = float(row["close"])
            stop, tp = atr_stop_tp(
                entry, direction, float(row["atr"]), p["stop_atr"], p["tp_atr"]
            )
            signals.append(
                Signal(
                    timestamp=t,
                    direction=direction,  # type: ignore[arg-type]
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    pattern="month_end_rebalance_proxy",
                    max_hold_bars=int(p["hold_bars"]),
                )
            )
            fired_months.add(ym)
        return signals
