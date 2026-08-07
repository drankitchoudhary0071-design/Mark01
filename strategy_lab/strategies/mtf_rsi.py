"""
Multi-Timeframe RSI mean-reversion (parameterized HTF/LTF).

Same logic as ``mtf_daily_4h_rsi``, but HTF/LTF are configurable so we can
compare smaller timeframes (more trades) vs the original 1D+4H setup.

Rules (no lookahead — HTF merged as-of):
  HTF : EMA_fast vs EMA_slow defines bull/bear regime
  LTF : RSI cross up through oversold only in HTF bull;
        RSI cross down through overbought only in HTF bear
  Exit: ATR stop / TP multiples
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


# pandas resample rule ↔ friendly label
TF_RULES = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1D": "1D",
    "D": "1D",
}


def to_rule(tf: str) -> str:
    if tf not in TF_RULES:
        raise ValueError(f"Unsupported TF {tf!r}; choose from {sorted(TF_RULES)}")
    return TF_RULES[tf]


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


def prepare_mtf_frame(
    df: pd.DataFrame,
    *,
    ltf: str = "4h",
    htf: str = "1D",
    rsi_period: int = 14,
    atr_period: int = 14,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> pd.DataFrame:
    """Build LTF frame with as-of HTF trend columns (no lookahead)."""
    ltf_df = resample_ohlcv(df, to_rule(ltf))
    htf_df = resample_ohlcv(df, to_rule(htf))
    ltf_df["rsi"] = ind.rsi(ltf_df["close"], rsi_period)
    ltf_df["atr"] = ind.atr(ltf_df, atr_period)
    htf_df["htf_ema_fast"] = ind.ema(htf_df["close"], ema_fast)
    htf_df["htf_ema_slow"] = ind.ema(htf_df["close"], ema_slow)
    htf_df["htf_trend"] = np.where(
        htf_df["htf_ema_fast"] > htf_df["htf_ema_slow"],
        1,
        np.where(htf_df["htf_ema_fast"] < htf_df["htf_ema_slow"], -1, 0),
    )
    # Keep legacy aliases used by older daily+4h runner / pine port
    htf_df["d_ema50"] = htf_df["htf_ema_fast"]
    htf_df["d_ema200"] = htf_df["htf_ema_slow"]
    htf_df["d_trend"] = htf_df["htf_trend"]

    h = htf_df[
        [
            "timestamp",
            "htf_ema_fast",
            "htf_ema_slow",
            "htf_trend",
            "d_ema50",
            "d_ema200",
            "d_trend",
        ]
    ].copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    l = ltf_df.copy()
    l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True)
    return pd.merge_asof(
        l.sort_values("timestamp"),
        h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


class MtfRsiStrategy(Strategy):
    """Parameterized MTF RSI — default matches original 1D+4H setup."""

    name = "mtf_rsi"
    library = "pandas MTF RSI + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "ltf": "4h",
            "htf": "1D",
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "ema_fast": 50,
            "ema_slow": 200,
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            f"[MTF] {self.params['htf']} EMA{self.params['ema_fast']}/"
            f"{self.params['ema_slow']} + {self.params['ltf']} RSI cross.",
            "[NOTE] HTF joined as-of (no lookahead). Smaller TFs → more trades, more friction.",
        ]
        self.name = (
            f"mtf_{self.params['htf'].lower()}_{self.params['ltf'].lower()}_rsi"
        )

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        if "htf_trend" in df.columns and "rsi" in df.columns:
            out = df.copy()
        elif "d_trend" in df.columns and "rsi" in df.columns:
            out = df.copy()
            out["htf_trend"] = out["d_trend"]
        else:
            out = prepare_mtf_frame(
                df,
                ltf=str(p["ltf"]),
                htf=str(p["htf"]),
                rsi_period=int(p["rsi_period"]),
                atr_period=int(p["atr_period"]),
                ema_fast=int(p["ema_fast"]),
                ema_slow=int(p["ema_slow"]),
            )

        trend_col = "htf_trend" if "htf_trend" in out.columns else "d_trend"
        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if any(pd.isna(row[c]) for c in ("atr", "rsi", trend_col)):
                continue
            if pd.isna(prev["rsi"]) or row["atr"] <= 0:
                continue

            trend = float(row[trend_col])
            if (
                prev["rsi"] < p["rsi_oversold"]
                and row["rsi"] >= p["rsi_oversold"]
                and trend > 0
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
                        pattern="mtf_rsi_long",
                    )
                )
            elif (
                prev["rsi"] > p["rsi_overbought"]
                and row["rsi"] <= p["rsi_overbought"]
                and trend < 0
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
                        pattern="mtf_rsi_short",
                    )
                )
        return signals
