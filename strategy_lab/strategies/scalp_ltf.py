"""
Scalping strategies for 1m–5m charts (TradingView-style indicators).

Designed for first-6m train / next-6m OOS evaluation under costs.
Wider ATR stops (relative to friction) are preferred — tiny stops get eaten.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


def _prep(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["ema_f"] = ind.ema(out["close"], int(p["ema_fast"]))
    out["ema_s"] = ind.ema(out["close"], int(p["ema_slow"]))
    out["ema_trend"] = ind.ema(out["close"], int(p["ema_trend"]))
    out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
    out["atr"] = ind.atr(out, int(p["atr_period"]))
    macd_l, macd_s, macd_h = ind.macd(
        out["close"], int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
    )
    out["macd"] = macd_l
    out["macd_sig"] = macd_s
    out["macd_hist"] = macd_h
    st_k, st_d = ind.stochastic(
        out, int(p["stoch_k"]), int(p["stoch_smooth"]), int(p["stoch_d"])
    )
    out["stoch_k"] = st_k
    out["stoch_d"] = st_d
    bb_lo, bb_mid, bb_hi = ind.bollinger(out["close"], int(p["bb_period"]), float(p["bb_std"]))
    out["bb_lo"] = bb_lo
    out["bb_mid"] = bb_mid
    out["bb_hi"] = bb_hi
    adx_df = ind.adx(out, int(p["adx_period"]))
    out["adx"] = adx_df["adx"]
    out["vol_sma"] = out["volume"].rolling(int(p["vol_sma"])).mean()
    # VWAP proxy: cumulative typical price * vol / cum vol (session-less rolling)
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    roll = int(p["vwap_window"])
    out["vwap"] = (tp * out["volume"]).rolling(roll).sum() / out["volume"].rolling(roll).sum().replace(
        0, np.nan
    )
    return out


def _emit(
    row: pd.Series,
    direction: str,
    p: dict[str, Any],
    pattern: str,
) -> Signal:
    entry = float(row["close"])
    stop, tp = atr_stop_tp(
        entry, direction, float(row["atr"]), float(p["stop_atr"]), float(p["tp_atr"])
    )
    return Signal(
        timestamp=pd.Timestamp(row["timestamp"]),
        direction=direction,  # type: ignore[arg-type]
        entry=entry,
        stop=stop,
        take_profit=tp,
        pattern=pattern,
        max_hold_bars=int(p["max_hold_bars"]),
    )


def _rule_ema_rsi(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """EMA cross + RSI filter + min ATR%."""
    need = ("ema_f", "ema_s", "rsi", "atr", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["ema_f"]):
        return None
    atr_pct = float(row["atr"]) / float(row["close"])
    if atr_pct < float(p["min_atr_pct"]):
        return None
    long_x = prev["ema_f"] <= prev["ema_s"] and row["ema_f"] > row["ema_s"]
    short_x = prev["ema_f"] >= prev["ema_s"] and row["ema_f"] < row["ema_s"]
    if long_x and float(row["rsi"]) >= float(p["rsi_long_min"]):
        return "long"
    if short_x and float(row["rsi"]) <= float(p["rsi_short_max"]):
        return "short"
    return None


def _rule_vwap_rsi(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """Pullback to VWAP with RSI reclaim + trend EMA filter."""
    need = ("vwap", "rsi", "atr", "ema_trend", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["rsi"]) or pd.isna(prev["close"]):
        return None
    atr_pct = float(row["atr"]) / float(row["close"])
    if atr_pct < float(p["min_atr_pct"]):
        return None
    # Long: price above trend EMA, dipped to/below VWAP, RSI crosses up os
    if (
        float(row["close"]) > float(row["ema_trend"])
        and float(prev["close"]) <= float(prev["vwap"])
        and float(row["close"]) > float(row["vwap"])
        and prev["rsi"] < p["rsi_os"]
        and row["rsi"] >= p["rsi_os"]
    ):
        return "long"
    if (
        float(row["close"]) < float(row["ema_trend"])
        and float(prev["close"]) >= float(prev["vwap"])
        and float(row["close"]) < float(row["vwap"])
        and prev["rsi"] > p["rsi_ob"]
        and row["rsi"] <= p["rsi_ob"]
    ):
        return "short"
    return None


def _rule_bb_stoch(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """Bollinger touch + Stochastic cross (mean-reversion scalp)."""
    need = ("bb_lo", "bb_hi", "stoch_k", "stoch_d", "atr", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["stoch_k"]):
        return None
    atr_pct = float(row["atr"]) / float(row["close"])
    if atr_pct < float(p["min_atr_pct"]):
        return None
    k0, k1 = float(prev["stoch_k"]), float(row["stoch_k"])
    d0, d1 = float(prev["stoch_d"]), float(row["stoch_d"])
    if float(row["close"]) <= float(row["bb_lo"]) and k0 <= d0 and k1 > d1 and k1 < 30:
        return "long"
    if float(row["close"]) >= float(row["bb_hi"]) and k0 >= d0 and k1 < d1 and k1 > 70:
        return "short"
    return None


def _rule_macd_ema(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """MACD cross with EMA trend + volume spike."""
    need = ("macd", "macd_sig", "ema_trend", "atr", "volume", "vol_sma", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["macd"]):
        return None
    atr_pct = float(row["atr"]) / float(row["close"])
    if atr_pct < float(p["min_atr_pct"]):
        return None
    if float(row["volume"]) < float(row["vol_sma"]) * float(p["vol_mult"]):
        return None
    m0, m1 = float(prev["macd"]), float(row["macd"])
    s0, s1 = float(prev["macd_sig"]), float(row["macd_sig"])
    if m0 <= s0 and m1 > s1 and float(row["close"]) > float(row["ema_trend"]):
        return "long"
    if m0 >= s0 and m1 < s1 and float(row["close"]) < float(row["ema_trend"]):
        return "short"
    return None


def _rule_adx_ema_rsi(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """ADX trending + EMA pullback RSI bounce."""
    need = ("adx", "ema_f", "ema_s", "rsi", "atr", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["rsi"]):
        return None
    if float(row["adx"]) < float(p["adx_min"]):
        return None
    atr_pct = float(row["atr"]) / float(row["close"])
    if atr_pct < float(p["min_atr_pct"]):
        return None
    # Uptrend: ema_f > ema_s, RSI cross up from os
    if (
        float(row["ema_f"]) > float(row["ema_s"])
        and prev["rsi"] < p["rsi_os"]
        and row["rsi"] >= p["rsi_os"]
    ):
        return "long"
    if (
        float(row["ema_f"]) < float(row["ema_s"])
        and prev["rsi"] > p["rsi_ob"]
        and row["rsi"] <= p["rsi_ob"]
    ):
        return "short"
    return None


RULES: dict[str, Callable] = {
    "ema_rsi": _rule_ema_rsi,
    "vwap_rsi": _rule_vwap_rsi,
    "bb_stoch": _rule_bb_stoch,
    "macd_ema": _rule_macd_ema,
    "adx_ema_rsi": _rule_adx_ema_rsi,
}

VARIANT_META = {
    "ema_rsi": {
        "indicators": ["EMA", "RSI", "ATR"],
        "desc": "EMA cross + RSI momentum scalp",
    },
    "vwap_rsi": {
        "indicators": ["VWAP", "EMA", "RSI", "ATR"],
        "desc": "VWAP reclaim + RSI + trend EMA",
    },
    "bb_stoch": {
        "indicators": ["Bollinger", "Stochastic", "ATR"],
        "desc": "BB touch + Stoch cross mean-reversion",
    },
    "macd_ema": {
        "indicators": ["MACD", "EMA", "Volume", "ATR"],
        "desc": "MACD cross + EMA trend + volume spike",
    },
    "adx_ema_rsi": {
        "indicators": ["ADX", "EMA", "RSI", "ATR"],
        "desc": "ADX trend + EMA + RSI pullback",
    },
}


class ScalpMtfStrategy(Strategy):
    """1m/5m scalping family — pick ``variant``."""

    name = "scalp_ltf"
    library = "pandas scalp 1m-5m + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "variant": "ema_rsi",
            "ema_fast": 8,
            "ema_slow": 21,
            "ema_trend": 50,
            "rsi_period": 7,
            "rsi_long_min": 55,
            "rsi_short_max": 45,
            "rsi_os": 30,
            "rsi_ob": 70,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "stoch_k": 14,
            "stoch_smooth": 3,
            "stoch_d": 3,
            "bb_period": 20,
            "bb_std": 2.0,
            "adx_period": 14,
            "adx_min": 20,
            "vol_sma": 20,
            "vol_mult": 1.2,
            "vwap_window": 78,  # ~6.5h on 5m; ~1.3h on 1m
            "atr_period": 14,
            "min_atr_pct": 0.0004,
            "stop_atr": 1.5,
            "tp_atr": 2.0,
            "max_hold_bars": 24,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        v = str(self.params["variant"])
        if v not in RULES:
            raise ValueError(f"Unknown scalp variant {v}")
        meta = VARIANT_META[v]
        self.name = f"scalp_{v}"
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            f"[SCALP] {meta['desc']}",
            f"[SCALP] Indicators: {', '.join(meta['indicators'])}",
            "[SCALP] Train first 6m / OOS next 6m — costs matter a lot on 1m/5m.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = _prep(df, p)
        rule = RULES[str(p["variant"])]
        signals: list[Signal] = []
        for i in range(1, len(out)):
            side = rule(out.iloc[i - 1], out.iloc[i], p)
            if side is None:
                continue
            signals.append(_emit(out.iloc[i], side, p, f"scalp_{p['variant']}_{side}"))
        return signals
