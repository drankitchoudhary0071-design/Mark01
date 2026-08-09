"""
19. Multi-timeframe Confluence — require HTF + LTF agreement before entry.

Resamples the LTF frame to an HTF (default 4h from 1h) and only emits
LTF entries when HTF trend agrees (EMA slope / position).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    ohlc = x.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    ohlc = ohlc.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in ohlc.columns:
        ohlc = ohlc.rename(columns={ohlc.columns[0]: "timestamp"})
    return ohlc


class MultiTimeframeConfluenceStrategy(Strategy):
    name = "mtf_confluence"
    library = "pandas resample + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "htf_rule": "4h",
            "htf_ema": 20,
            "ltf_fast_ma": 20,
            "ltf_slow_ma": 50,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        ltf = df.copy()
        htf = _resample_ohlcv(ltf, str(p["htf_rule"]))
        if len(htf) < int(p["htf_ema"]) + 5:
            return []

        htf["ema"] = ind.ema(htf["close"], int(p["htf_ema"]))
        htf["htf_bias"] = 0
        htf.loc[htf["close"] > htf["ema"], "htf_bias"] = 1
        htf.loc[htf["close"] < htf["ema"], "htf_bias"] = -1

        # Map HTF bias onto LTF bars (as-of merge — no look-ahead)
        htf_map = htf[["timestamp", "htf_bias"]].rename(columns={"timestamp": "htf_ts"})
        ltf = ltf.copy()
        ltf["timestamp"] = pd.to_datetime(ltf["timestamp"], utc=True)
        htf_map["htf_ts"] = pd.to_datetime(htf_map["htf_ts"], utc=True)
        ltf = pd.merge_asof(
            ltf.sort_values("timestamp"),
            htf_map.sort_values("htf_ts"),
            left_on="timestamp",
            right_on="htf_ts",
            direction="backward",
        )
        ltf["fast"] = ind.sma(ltf["close"], int(p["ltf_fast_ma"]))
        ltf["slow"] = ind.sma(ltf["close"], int(p["ltf_slow_ma"]))
        ltf["atr"] = ind.atr(ltf, int(p["atr_period"]))

        signals: list[Signal] = []
        for i in range(1, len(ltf)):
            row = ltf.iloc[i]
            prev = ltf.iloc[i - 1]
            if any(pd.isna(row[c]) for c in ("fast", "slow", "atr", "htf_bias")):
                continue
            # Long only if HTF bullish + LTF bullish cross
            if (
                row["htf_bias"] > 0
                and prev["fast"] <= prev["slow"]
                and row["fast"] > row["slow"]
            ):
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mtf_long",
                    )
                )
            elif (
                row["htf_bias"] < 0
                and prev["fast"] >= prev["slow"]
                and row["fast"] < row["slow"]
            ):
                entry = float(row["close"])
                stop, tp = atr_stop_tp(entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"])
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mtf_short",
                    )
                )
        return signals
