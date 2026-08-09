"""
Scalping strategies for 1m–5m (vectorized signals for speed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp


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

RULES = list(VARIANT_META.keys())


def _features(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    out["ema_f"] = ind.ema(c, int(p["ema_fast"]))
    out["ema_s"] = ind.ema(c, int(p["ema_slow"]))
    out["ema_trend"] = ind.ema(c, int(p["ema_trend"]))
    out["rsi"] = ind.rsi(c, int(p["rsi_period"]))
    out["atr"] = ind.atr(out, int(p["atr_period"]))
    ml, ms, mh = ind.macd(c, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    out["macd"], out["macd_sig"], out["macd_hist"] = ml, ms, mh
    sk, sd = ind.stochastic(out, int(p["stoch_k"]), int(p["stoch_smooth"]), int(p["stoch_d"]))
    out["stoch_k"], out["stoch_d"] = sk, sd
    lo, mid, hi = ind.bollinger(c, int(p["bb_period"]), float(p["bb_std"]))
    out["bb_lo"], out["bb_mid"], out["bb_hi"] = lo, mid, hi
    out["adx"] = ind.adx(out, int(p["adx_period"]))["adx"]
    out["vol_sma"] = out["volume"].rolling(int(p["vol_sma"])).mean()
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    roll = int(p["vwap_window"])
    out["vwap"] = (tp * out["volume"]).rolling(roll).sum() / out["volume"].rolling(roll).sum().replace(
        0, np.nan
    )
    out["atr_pct"] = out["atr"] / out["close"]
    return out


def _signals_from_masks(
    out: pd.DataFrame, long_m: np.ndarray, short_m: np.ndarray, p: dict, pattern: str
) -> list[Signal]:
    signals: list[Signal] = []
    idx = np.where(long_m | short_m)[0]
    stop_atr = float(p["stop_atr"])
    tp_atr = float(p["tp_atr"])
    hold = int(p["max_hold_bars"])
    for i in idx:
        row = out.iloc[i]
        if pd.isna(row["atr"]) or float(row["atr"]) <= 0:
            continue
        direction = "long" if long_m[i] else "short"
        entry = float(row["close"])
        stop, tp = atr_stop_tp(entry, direction, float(row["atr"]), stop_atr, tp_atr)
        signals.append(
            Signal(
                timestamp=pd.Timestamp(row["timestamp"]),
                direction=direction,  # type: ignore[arg-type]
                entry=entry,
                stop=stop,
                take_profit=tp,
                pattern=f"{pattern}_{direction}",
                max_hold_bars=hold,
            )
        )
    return signals


def gen_ema_rsi(out: pd.DataFrame, p: dict) -> list[Signal]:
    ef, es = out["ema_f"], out["ema_s"]
    long_x = (ef.shift(1) <= es.shift(1)) & (ef > es)
    short_x = (ef.shift(1) >= es.shift(1)) & (ef < es)
    ok = out["atr_pct"] >= float(p["min_atr_pct"])
    long_m = (long_x & (out["rsi"] >= float(p["rsi_long_min"])) & ok).fillna(False).to_numpy()
    short_m = (short_x & (out["rsi"] <= float(p["rsi_short_max"])) & ok).fillna(False).to_numpy()
    # no simultaneous
    short_m = short_m & ~long_m
    return _signals_from_masks(out, long_m, short_m, p, "scalp_ema_rsi")


def gen_vwap_rsi(out: pd.DataFrame, p: dict) -> list[Signal]:
    ok = out["atr_pct"] >= float(p["min_atr_pct"])
    long_m = (
        (out["close"] > out["ema_trend"])
        & (out["close"].shift(1) <= out["vwap"].shift(1))
        & (out["close"] > out["vwap"])
        & (out["rsi"].shift(1) < float(p["rsi_os"]))
        & (out["rsi"] >= float(p["rsi_os"]))
        & ok
    ).fillna(False).to_numpy()
    short_m = (
        (out["close"] < out["ema_trend"])
        & (out["close"].shift(1) >= out["vwap"].shift(1))
        & (out["close"] < out["vwap"])
        & (out["rsi"].shift(1) > float(p["rsi_ob"]))
        & (out["rsi"] <= float(p["rsi_ob"]))
        & ok
    ).fillna(False).to_numpy()
    short_m = short_m & ~long_m
    return _signals_from_masks(out, long_m, short_m, p, "scalp_vwap_rsi")


def gen_bb_stoch(out: pd.DataFrame, p: dict) -> list[Signal]:
    ok = out["atr_pct"] >= float(p["min_atr_pct"])
    k, d = out["stoch_k"], out["stoch_d"]
    long_m = (
        (out["close"] <= out["bb_lo"])
        & (k.shift(1) <= d.shift(1))
        & (k > d)
        & (k < 30)
        & ok
    ).fillna(False).to_numpy()
    short_m = (
        (out["close"] >= out["bb_hi"])
        & (k.shift(1) >= d.shift(1))
        & (k < d)
        & (k > 70)
        & ok
    ).fillna(False).to_numpy()
    short_m = short_m & ~long_m
    return _signals_from_masks(out, long_m, short_m, p, "scalp_bb_stoch")


def gen_macd_ema(out: pd.DataFrame, p: dict) -> list[Signal]:
    ok = out["atr_pct"] >= float(p["min_atr_pct"])
    vol_ok = out["volume"] >= out["vol_sma"] * float(p["vol_mult"])
    m, s = out["macd"], out["macd_sig"]
    long_m = (
        (m.shift(1) <= s.shift(1))
        & (m > s)
        & (out["close"] > out["ema_trend"])
        & vol_ok
        & ok
    ).fillna(False).to_numpy()
    short_m = (
        (m.shift(1) >= s.shift(1))
        & (m < s)
        & (out["close"] < out["ema_trend"])
        & vol_ok
        & ok
    ).fillna(False).to_numpy()
    short_m = short_m & ~long_m
    return _signals_from_masks(out, long_m, short_m, p, "scalp_macd_ema")


def gen_adx_ema_rsi(out: pd.DataFrame, p: dict) -> list[Signal]:
    ok = (out["atr_pct"] >= float(p["min_atr_pct"])) & (out["adx"] >= float(p["adx_min"]))
    long_m = (
        (out["ema_f"] > out["ema_s"])
        & (out["rsi"].shift(1) < float(p["rsi_os"]))
        & (out["rsi"] >= float(p["rsi_os"]))
        & ok
    ).fillna(False).to_numpy()
    short_m = (
        (out["ema_f"] < out["ema_s"])
        & (out["rsi"].shift(1) > float(p["rsi_ob"]))
        & (out["rsi"] <= float(p["rsi_ob"]))
        & ok
    ).fillna(False).to_numpy()
    short_m = short_m & ~long_m
    return _signals_from_masks(out, long_m, short_m, p, "scalp_adx_ema_rsi")


GENERATORS = {
    "ema_rsi": gen_ema_rsi,
    "vwap_rsi": gen_vwap_rsi,
    "bb_stoch": gen_bb_stoch,
    "macd_ema": gen_macd_ema,
    "adx_ema_rsi": gen_adx_ema_rsi,
}


class ScalpMtfStrategy(Strategy):
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
            "vwap_window": 78,
            "atr_period": 14,
            "min_atr_pct": 0.0004,
            "stop_atr": 1.5,
            "tp_atr": 2.0,
            "max_hold_bars": 24,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        v = str(self.params["variant"])
        if v not in GENERATORS:
            raise ValueError(f"Unknown scalp variant {v}")
        meta = VARIANT_META[v]
        self.name = f"scalp_{v}"
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            f"[SCALP] {meta['desc']}",
            f"[SCALP] Indicators: {', '.join(meta['indicators'])}",
            "[SCALP] Train first 6m / OOS next 6m.",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        out = _features(df, p)
        return GENERATORS[str(p["variant"])](out, p)
