"""
Multi-Timeframe: Daily trend + 4H RSI mean-reversion.

Distinct from:
  - ``mtf_confluence`` (1h MA cross + 4h EMA bias) — lost under costs
  - ``btc_4h_rsi_mr`` (4h-only RSI, no HTF filter)

Rules (no lookahead — HTF merged as-of):
  HTF 1D : EMA50 vs EMA200 defines bull/bear regime
  LTF 4H : RSI(14) cross up through 30 only in daily bull;
           RSI(14) cross down through 70 only in daily bear
  Exit   : ATR(14) stop 1.5× / TP 3.0× (1:2)

Designed for BTCUSDT and PAXGUSDT with slippage + compounding WF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    out = (
        x.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    out.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return out


def prepare_mtf_frame(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build 4h LTF frame with as-of daily trend columns (no lookahead)."""
    ltf = resample_ohlcv(df_1h, "4h")
    htf = resample_ohlcv(df_1h, "1D")
    ltf["rsi"] = ind.rsi(ltf["close"], 14)
    ltf["atr"] = ind.atr(ltf, 14)
    htf["d_ema50"] = ind.ema(htf["close"], 50)
    htf["d_ema200"] = ind.ema(htf["close"], 200)
    htf["d_trend"] = np.where(
        htf["d_ema50"] > htf["d_ema200"],
        1,
        np.where(htf["d_ema50"] < htf["d_ema200"], -1, 0),
    )
    h = htf[["timestamp", "d_ema50", "d_ema200", "d_trend"]].copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    l = ltf.copy()
    l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True)
    return pd.merge_asof(
        l.sort_values("timestamp"),
        h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


class MtfDaily4hRsiStrategy(Strategy):
    name = "mtf_daily_4h_rsi"
    library = "pandas MTF (1D+4H) + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "daily_ema_fast": 50,
            "daily_ema_slow": 200,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[MTF] 1D EMA50/200 regime + 4H RSI cross — different from mtf_confluence / 4h-only RSI-MR.",
            "[NOTE] Requires 1h (or finer) history so 4h+1d can be built; HTF joined as-of (no lookahead).",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        Accepts either:
          - 1h OHLCV (will resample internally), or
          - pre-built MTF frame with columns rsi, atr, d_trend
        """
        p = self.params
        if "d_trend" in df.columns and "rsi" in df.columns:
            out = df.copy()
        else:
            # Assume 1h (or any LTF denser than 4h)
            out = prepare_mtf_frame(df)
            # recompute with configurable lengths if needed
            if int(p["rsi_period"]) != 14:
                out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
            if int(p["atr_period"]) != 14:
                out["atr"] = ind.atr(out, int(p["atr_period"]))

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if any(pd.isna(row[c]) for c in ("atr", "rsi", "d_trend")):
                continue
            if pd.isna(prev["rsi"]) or row["atr"] <= 0:
                continue

            # Long only with daily bull trend
            if (
                prev["rsi"] < p["rsi_oversold"]
                and row["rsi"] >= p["rsi_oversold"]
                and row["d_trend"] > 0
            ):
                entry = float(row["close"])
                stop, tp = atr_stop_tp(
                    entry, "long", float(row["atr"]), p["stop_atr"], p["tp_atr"]
                )
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="long",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mtf_d_rsi_long",
                    )
                )
            # Short only with daily bear trend
            elif (
                prev["rsi"] > p["rsi_overbought"]
                and row["rsi"] <= p["rsi_overbought"]
                and row["d_trend"] < 0
            ):
                entry = float(row["close"])
                stop, tp = atr_stop_tp(
                    entry, "short", float(row["atr"]), p["stop_atr"], p["tp_atr"]
                )
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(row["timestamp"]),
                        direction="short",
                        entry=entry,
                        stop=stop,
                        take_profit=tp,
                        pattern="mtf_d_rsi_short",
                    )
                )
        return signals
