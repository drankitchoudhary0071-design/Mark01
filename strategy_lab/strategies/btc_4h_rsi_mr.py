"""
BTC 4H RSI Mean-Reversion (TV-aligned, no lookahead).

Why this exists
---------------
The earlier ``rsi_divergence`` lab profit used pivot timestamps at the
pivot bar itself — that is lookahead (pivot is only known after
``swing_lookback`` bars). After fixing confirmation delay, divergence
lost money. This module is the best *honest* single-chart candidate
found under TV-like costs:

  Asset/TF : BTCUSDT 4h
  Entry    : RSI(14) crosses back above 30 (long) / below 70 (short)
  Exit     : ATR(14) stop 1.5× / take-profit 3.0×  (fixed 1:2 RR)
  Costs    : 0.10% commission/side + small spread/slip

Textbook Wilder RSI levels — not optimized on a single OOS window.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


class Btc4hRsiMrStrategy(Strategy):
    name = "btc_4h_rsi_mr"
    library = "pandas+shared_engine (TV-aligned)"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,  # 1:2 vs stop
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[TV-ALIGNED] Enter on RSI cross bar close (no pivot lookahead).",
            "[NOTE] Designed for BTCUSDT 4h. Edge is modest; validate on TradingView "
            "with commission 0.1% before live use.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = df.copy()
        out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = ind.atr(out, int(p["atr_period"]))

        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if pd.isna(row["atr"]) or row["atr"] <= 0 or pd.isna(row["rsi"]) or pd.isna(prev["rsi"]):
                continue

            # Long: RSI crosses UP through oversold
            if prev["rsi"] < p["rsi_oversold"] and row["rsi"] >= p["rsi_oversold"]:
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
                        pattern="rsi4h_long",
                    )
                )
            # Short: RSI crosses DOWN through overbought
            elif prev["rsi"] > p["rsi_overbought"] and row["rsi"] <= p["rsi_overbought"]:
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
                        pattern="rsi4h_short",
                    )
                )
        return signals


def resample_ohlcv_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h (UTC)."""
    x = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    out = (
        x.resample("4h")
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
