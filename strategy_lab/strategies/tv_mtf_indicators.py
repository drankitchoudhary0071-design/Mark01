"""
TradingView-style multi-indicator MTF strategies.

Each variant uses 3–5 classic TV indicators across HTF (regime) + LTF (entry).
Default structure: Daily HTF + 4H LTF (proven cost-tolerant on BTC).

Variants
--------
A  ema_rsi_macd      : HTF EMA50/200 + LTF RSI cross + MACD hist confirm
B  ema_adx_stoch     : HTF EMA + ADX filter + LTF Stochastic cross
C  supertrend_rsi_bb : HTF Supertrend + LTF RSI + Bollinger touch
D  ema_macd_stoch    : HTF EMA + LTF MACD cross + Stochastic filter
E  adx_rsi_macd_bb   : HTF ADX+EMA + LTF RSI + MACD + BB (5 indicators)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy, atr_stop_tp
from strategy_lab.strategies.mtf_rsi import resample_ohlcv, to_rule


def _htf_features(htf: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = htf.copy()
    out["htf_ema_fast"] = ind.ema(out["close"], int(p["ema_fast"]))
    out["htf_ema_slow"] = ind.ema(out["close"], int(p["ema_slow"]))
    out["htf_trend"] = np.where(
        out["htf_ema_fast"] > out["htf_ema_slow"],
        1,
        np.where(out["htf_ema_fast"] < out["htf_ema_slow"], -1, 0),
    )
    adx_df = ind.adx(out, int(p["adx_period"]))
    out["htf_adx"] = adx_df["adx"]
    out["htf_plus_di"] = adx_df["+di"]
    out["htf_minus_di"] = adx_df["-di"]
    _, st_dir = ind.supertrend(out, int(p["st_period"]), float(p["st_mult"]))
    out["htf_st_dir"] = st_dir
    return out


def _ltf_features(ltf: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = ltf.copy()
    out["rsi"] = ind.rsi(out["close"], int(p["rsi_period"]))
    out["atr"] = ind.atr(out, int(p["atr_period"]))
    macd_line, macd_sig, macd_hist = ind.macd(
        out["close"], int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
    )
    out["macd"] = macd_line
    out["macd_sig"] = macd_sig
    out["macd_hist"] = macd_hist
    stoch_k, stoch_d = ind.stochastic(
        out, int(p["stoch_k"]), int(p["stoch_smooth"]), int(p["stoch_d"])
    )
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    bb_lo, bb_mid, bb_hi = ind.bollinger(
        out["close"], int(p["bb_period"]), float(p["bb_std"])
    )
    out["bb_lo"] = bb_lo
    out["bb_mid"] = bb_mid
    out["bb_hi"] = bb_hi
    return out


def prepare_tv_mtf(
    df: pd.DataFrame,
    *,
    ltf: str = "4h",
    htf: str = "1D",
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build LTF frame with as-of HTF columns (no lookahead)."""
    p = {**TvMtfStrategy.default_params(), **(params or {})}
    ltf_df = _ltf_features(resample_ohlcv(df, to_rule(ltf)), p)
    htf_df = _htf_features(resample_ohlcv(df, to_rule(htf)), p)
    h_cols = [
        "timestamp",
        "htf_ema_fast",
        "htf_ema_slow",
        "htf_trend",
        "htf_adx",
        "htf_plus_di",
        "htf_minus_di",
        "htf_st_dir",
    ]
    h = htf_df[h_cols].copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    l = ltf_df.copy()
    l["timestamp"] = pd.to_datetime(l["timestamp"], utc=True)
    return pd.merge_asof(
        l.sort_values("timestamp"),
        h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


# ── Entry rule helpers (operate on consecutive rows) ─────────────────────────

def _cross_up(prev: float, curr: float, level: float) -> bool:
    return prev < level <= curr


def _cross_down(prev: float, curr: float, level: float) -> bool:
    return prev > level >= curr


def _rule_ema_rsi_macd(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """A: Daily EMA trend + 4H RSI cross + MACD hist same-side."""
    if any(pd.isna(row[c]) for c in ("rsi", "atr", "macd_hist", "htf_trend")):
        return None
    if pd.isna(prev["rsi"]) or pd.isna(prev["macd_hist"]):
        return None
    if (
        row["htf_trend"] > 0
        and _cross_up(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_os"]))
        and float(row["macd_hist"]) > 0
    ):
        return "long"
    if (
        row["htf_trend"] < 0
        and _cross_down(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_ob"]))
        and float(row["macd_hist"]) < 0
    ):
        return "short"
    return None


def _rule_ema_adx_stoch(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """B: Daily EMA + ADX strength + Stochastic %K/%D cross in zone."""
    need = ("stoch_k", "stoch_d", "atr", "htf_trend", "htf_adx")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["stoch_k"]) or pd.isna(prev["stoch_d"]):
        return None
    if float(row["htf_adx"]) < float(p["adx_min"]):
        return None
    k0, k1 = float(prev["stoch_k"]), float(row["stoch_k"])
    d0, d1 = float(prev["stoch_d"]), float(row["stoch_d"])
    # %K crosses above %D from oversold
    if (
        row["htf_trend"] > 0
        and k0 <= d0
        and k1 > d1
        and k1 < float(p["stoch_os"])
    ):
        return "long"
    if (
        row["htf_trend"] < 0
        and k0 >= d0
        and k1 < d1
        and k1 > float(p["stoch_ob"])
    ):
        return "short"
    return None


def _rule_supertrend_rsi_bb(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """C: Daily Supertrend + 4H RSI cross + Bollinger touch."""
    need = ("rsi", "atr", "bb_lo", "bb_hi", "htf_st_dir", "close")
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["rsi"]):
        return None
    if (
        float(row["htf_st_dir"]) > 0
        and _cross_up(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_os"]))
        and float(row["close"]) <= float(row["bb_lo"]) * 1.002  # near/below lower BB
    ):
        return "long"
    if (
        float(row["htf_st_dir"]) < 0
        and _cross_down(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_ob"]))
        and float(row["close"]) >= float(row["bb_hi"]) * 0.998
    ):
        return "short"
    return None


def _rule_ema_macd_stoch(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """D: Daily EMA + MACD line/signal cross + Stochastic not extreme opposite."""
    need = ("macd", "macd_sig", "stoch_k", "atr", "htf_trend")
    if any(pd.isna(row[c]) for c in need):
        return None
    if pd.isna(prev["macd"]) or pd.isna(prev["macd_sig"]):
        return None
    m0, m1 = float(prev["macd"]), float(row["macd"])
    s0, s1 = float(prev["macd_sig"]), float(row["macd_sig"])
    if (
        row["htf_trend"] > 0
        and m0 <= s0
        and m1 > s1
        and float(row["stoch_k"]) < float(p["stoch_ob"])
    ):
        return "long"
    if (
        row["htf_trend"] < 0
        and m0 >= s0
        and m1 < s1
        and float(row["stoch_k"]) > float(p["stoch_os"])
    ):
        return "short"
    return None


def _rule_adx_rsi_macd_bb(prev: pd.Series, row: pd.Series, p: dict) -> str | None:
    """E: Daily ADX+EMA + RSI cross + MACD hist + BB mid reclaim."""
    need = (
        "rsi",
        "macd_hist",
        "bb_mid",
        "atr",
        "htf_trend",
        "htf_adx",
        "close",
    )
    if any(pd.isna(row[c]) for c in need) or pd.isna(prev["rsi"]) or pd.isna(prev["close"]):
        return None
    if float(row["htf_adx"]) < float(p["adx_min"]):
        return None
    if (
        row["htf_trend"] > 0
        and _cross_up(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_os"]))
        and float(row["macd_hist"]) > 0
        and float(prev["close"]) < float(prev["bb_mid"])
        and float(row["close"]) >= float(row["bb_mid"])
    ):
        return "long"
    if (
        row["htf_trend"] < 0
        and _cross_down(float(prev["rsi"]), float(row["rsi"]), float(p["rsi_ob"]))
        and float(row["macd_hist"]) < 0
        and float(prev["close"]) > float(prev["bb_mid"])
        and float(row["close"]) <= float(row["bb_mid"])
    ):
        return "short"
    return None


RULES: dict[str, Callable[[pd.Series, pd.Series, dict], str | None]] = {
    "ema_rsi_macd": _rule_ema_rsi_macd,
    "ema_adx_stoch": _rule_ema_adx_stoch,
    "supertrend_rsi_bb": _rule_supertrend_rsi_bb,
    "ema_macd_stoch": _rule_ema_macd_stoch,
    "adx_rsi_macd_bb": _rule_adx_rsi_macd_bb,
}

VARIANT_META = {
    "ema_rsi_macd": {
        "indicators": ["EMA", "RSI", "MACD", "ATR"],
        "n_ind": 4,
        "desc": "HTF EMA50/200 trend + LTF RSI cross + MACD histogram confirm",
    },
    "ema_adx_stoch": {
        "indicators": ["EMA", "ADX", "Stochastic", "ATR"],
        "n_ind": 4,
        "desc": "HTF EMA+ADX strength + LTF Stochastic %K/%D cross in zone",
    },
    "supertrend_rsi_bb": {
        "indicators": ["Supertrend", "RSI", "Bollinger", "ATR"],
        "n_ind": 4,
        "desc": "HTF Supertrend + LTF RSI cross + Bollinger band touch",
    },
    "ema_macd_stoch": {
        "indicators": ["EMA", "MACD", "Stochastic", "ATR"],
        "n_ind": 4,
        "desc": "HTF EMA trend + LTF MACD cross + Stochastic filter",
    },
    "adx_rsi_macd_bb": {
        "indicators": ["EMA", "ADX", "RSI", "MACD", "Bollinger", "ATR"],
        "n_ind": 5,
        "desc": "HTF ADX+EMA + LTF RSI + MACD + BB mid reclaim (5-indicator)",
    },
}


class TvMtfStrategy(Strategy):
    """TradingView multi-indicator MTF — pick variant via ``variant`` param."""

    name = "tv_mtf"
    library = "pandas TV indicators MTF + shared engine"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "variant": "ema_rsi_macd",
            "ltf": "4h",
            "htf": "1D",
            # EMA
            "ema_fast": 50,
            "ema_slow": 200,
            # RSI
            "rsi_period": 14,
            "rsi_os": 30,
            "rsi_ob": 70,
            # MACD
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            # Stochastic
            "stoch_k": 14,
            "stoch_smooth": 3,
            "stoch_d": 3,
            "stoch_os": 20,
            "stoch_ob": 80,
            # ADX / Supertrend / BB
            "adx_period": 14,
            "adx_min": 20,
            "st_period": 10,
            "st_mult": 3.0,
            "bb_period": 20,
            "bb_std": 2.0,
            # Risk
            "atr_period": 14,
            "stop_atr": 1.5,
            "tp_atr": 3.0,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        v = str(self.params["variant"])
        if v not in RULES:
            raise ValueError(f"Unknown variant {v}; choose from {sorted(RULES)}")
        meta = VARIANT_META[v]
        self.name = f"tv_mtf_{v}"
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            f"[TV-MTF] {meta['desc']}",
            f"[TV-MTF] Indicators ({meta['n_ind']}): {', '.join(meta['indicators'])}",
            f"[TF] {self.params['htf']} → {self.params['ltf']} (HTF as-of, no lookahead)",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        if "htf_trend" in df.columns and "rsi" in df.columns and "macd_hist" in df.columns:
            out = df.copy()
        else:
            out = prepare_tv_mtf(
                df, ltf=str(p["ltf"]), htf=str(p["htf"]), params=p
            )
        rule = RULES[str(p["variant"])]
        signals: list[Signal] = []
        for i in range(1, len(out)):
            row = out.iloc[i]
            prev = out.iloc[i - 1]
            if pd.isna(row.get("atr")) or float(row["atr"]) <= 0:
                continue
            side = rule(prev, row, p)
            if side is None:
                continue
            entry = float(row["close"])
            stop, tp = atr_stop_tp(
                entry, side, float(row["atr"]), p["stop_atr"], p["tp_atr"]
            )
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(row["timestamp"]),
                    direction=side,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    pattern=f"tv_{p['variant']}_{side}",
                )
            )
        return signals
