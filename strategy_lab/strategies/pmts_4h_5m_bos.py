"""
PMTS — 4H Fib Golden Zone + 5min BOS Confirmation (v2)

Matches updated TradingView logic:
  HTF 4H: fib ALWAYS from latest confirmed swing high + low (no trailing)
  Pivot filters: swingLen=15 + minMovePct (default 1.5%)
  Direction: close beyond latest opposite pivot (BOS), not 0.7 flip
  Zone 0.5–0.7; SL=0.7 side; TP=anchor extreme
  LTF 5m: arm on zone tap; enter on LTF BOS in HTF direction
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy
from strategy_lab.strategies.mtf_rsi import resample_ohlcv
from strategy_lab.strategies.pmts_fib_trailing import pivot_high, pivot_low


def run_htf_pmts(
    htf: pd.DataFrame,
    swing_len: int = 15,
    min_move_pct: float = 1.5,
) -> pd.DataFrame:
    """HTF structure: latest major swing pair + BOS direction (Pine f_pmts v2)."""
    high = htf["high"].astype(float).to_numpy()
    low = htf["low"].astype(float).to_numpy()
    close = htf["close"].astype(float).to_numpy()
    n = len(htf)

    ph = pivot_high(htf["high"].astype(float), swing_len, swing_len).to_numpy()
    pl = pivot_low(htf["low"].astype(float), swing_len, swing_len).to_numpy()

    direction = np.array([None] * n, dtype=object)
    anchor_high = np.full(n, np.nan)
    anchor_low = np.full(n, np.nan)

    last_sh: Optional[float] = None
    last_sl: Optional[float] = None
    dir_s: Optional[str] = None

    for i in range(n):
        if not np.isnan(ph[i]):
            pv = float(ph[i])
            if last_sh is None or abs(pv - last_sh) / last_sh * 100.0 >= min_move_pct:
                last_sh = pv
        if not np.isnan(pl[i]):
            pv = float(pl[i])
            if last_sl is None or abs(pv - last_sl) / last_sl * 100.0 >= min_move_pct:
                last_sl = pv

        # BOS on close beyond latest opposite pivot (order matches Pine:
        # bear check first, then bull — bull wins if both true same bar)
        if last_sl is not None and close[i] < last_sl:
            dir_s = "bear"
        if last_sh is not None and close[i] > last_sh:
            dir_s = "bull"

        direction[i] = dir_s
        if last_sh is not None:
            anchor_high[i] = last_sh
        if last_sl is not None:
            anchor_low[i] = last_sl

    out = htf[["timestamp"]].copy()
    out["htf_dir"] = direction
    out["htf_anchor_high"] = anchor_high
    out["htf_anchor_low"] = anchor_low
    return out


def map_htf_to_ltf(ltf: pd.DataFrame, htf_state: pd.DataFrame) -> pd.DataFrame:
    """As-of merge of last *closed* HTF bar (lookahead_off)."""
    h = htf_state.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    for col in ("htf_dir", "htf_anchor_high", "htf_anchor_low"):
        h[col] = h[col].shift(1)

    left = pd.DataFrame({"timestamp": pd.to_datetime(ltf["timestamp"], utc=True)})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged


class Pmts4h5mBosStrategy(Strategy):
    """4H fib golden zone + 5m BOS confirmation (major-swing v2)."""

    name = "pmts_4h_5m_bos"
    library = "pandas PMTS 4H zone + 5m BOS v2"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "htf": "4h",
            "htf_swing_len": 15,
            "min_move_pct": 1.5,
            "zone_upper": 0.5,
            "zone_lower": 0.7,
            "ltf_swing_len": 3,
            "max_hold_bars": 12 * 24 * 5,  # ~5 days on 5m
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            "[PMTS 4H+5m v2] Fib from latest major swing pair (len=15, minMove%).",
            "[PMTS 4H+5m v2] HTF dir = close beyond latest pivot (not 0.7 trail flip).",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """``df`` must be LTF (5m) OHLCV."""
        p = self.params
        htf = resample_ohlcv(df, str(p.get("htf", "4h")))
        htf_state = run_htf_pmts(
            htf,
            swing_len=int(p["htf_swing_len"]),
            min_move_pct=float(p["min_move_pct"]),
        )
        mapped = map_htf_to_ltf(df, htf_state)

        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        n = len(df)

        htf_dir = mapped["htf_dir"].to_numpy()
        a_hi = mapped["htf_anchor_high"].to_numpy(dtype=float)
        a_lo = mapped["htf_anchor_low"].to_numpy(dtype=float)

        zone_u = float(p["zone_upper"])
        zone_l = float(p["zone_lower"])
        swing = int(p["ltf_swing_len"])
        hold = int(p["max_hold_bars"])

        ph = pivot_high(df["high"].astype(float), swing, swing).to_numpy()
        pl = pivot_low(df["low"].astype(float), swing, swing).to_numpy()
        ltf_sh = np.full(n, np.nan)
        ltf_sl = np.full(n, np.nan)
        last_sh = np.nan
        last_sl = np.nan
        for i in range(n):
            if not np.isnan(ph[i]):
                last_sh = ph[i]
            if not np.isnan(pl[i]):
                last_sl = pl[i]
            ltf_sh[i] = last_sh
            ltf_sl[i] = last_sl

        bull_bos = np.zeros(n, dtype=bool)
        bear_bos = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if np.isfinite(ltf_sh[i]) and close[i - 1] <= ltf_sh[i] and close[i] > ltf_sh[i]:
                bull_bos[i] = True
            if np.isfinite(ltf_sl[i]) and close[i - 1] >= ltf_sl[i] and close[i] < ltf_sl[i]:
                bear_bos[i] = True

        zone_top = np.full(n, np.nan)
        zone_bot = np.full(n, np.nan)
        sl_px = np.full(n, np.nan)
        tp_px = np.full(n, np.nan)
        for i in range(n):
            d = htf_dir[i]
            if not isinstance(d, str):
                continue
            if not np.isfinite(a_hi[i]) or not np.isfinite(a_lo[i]):
                continue
            rng = a_hi[i] - a_lo[i]
            if rng <= 0:
                continue
            if d == "bull":
                zone_top[i] = a_hi[i] - rng * zone_u
                zone_bot[i] = a_hi[i] - rng * zone_l
                sl_px[i] = zone_bot[i]
                tp_px[i] = a_hi[i]
            elif d == "bear":
                zone_bot[i] = a_lo[i] + rng * zone_u
                zone_top[i] = a_lo[i] + rng * zone_l
                sl_px[i] = zone_top[i]
                tp_px[i] = a_lo[i]

        signals: list[Signal] = []
        armed = False
        armed_dir: Optional[str] = None
        in_pos = False
        pos_side: Optional[str] = None
        pos_stop = pos_tp = 0.0
        entry_i = -1

        for i in range(n):
            d_str = htf_dir[i] if isinstance(htf_dir[i], str) else None

            if in_pos:
                exited = False
                if pos_side == "long" and (low[i] <= pos_stop or high[i] >= pos_tp):
                    exited = True
                elif pos_side == "short" and (high[i] >= pos_stop or low[i] <= pos_tp):
                    exited = True
                if i - entry_i >= hold:
                    exited = True
                if exited:
                    in_pos = False
                    pos_side = None

            if d_str is not None and d_str != armed_dir:
                armed = False
                armed_dir = d_str

            in_zone = (
                np.isfinite(zone_top[i])
                and np.isfinite(zone_bot[i])
                and low[i] <= zone_top[i]
                and high[i] >= zone_bot[i]
            )
            if in_zone:
                armed = True

            if in_pos or not armed or d_str is None:
                continue
            if not np.isfinite(sl_px[i]) or not np.isfinite(tp_px[i]):
                continue

            long_signal = d_str == "bull" and bull_bos[i]
            short_signal = d_str == "bear" and bear_bos[i]

            if long_signal and sl_px[i] < close[i] < tp_px[i]:
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(ts.iloc[i]),
                        direction="long",
                        entry=float(close[i]),
                        stop=float(sl_px[i]),
                        take_profit=float(tp_px[i]),
                        pattern="pmts_4h5m_long",
                        max_hold_bars=hold,
                        meta={
                            "htf_dir": d_str,
                            "zone_top": float(zone_top[i]),
                            "zone_bot": float(zone_bot[i]),
                            "anchor_high": float(a_hi[i]),
                            "anchor_low": float(a_lo[i]),
                        },
                    )
                )
                in_pos = True
                pos_side = "long"
                pos_stop, pos_tp, entry_i = float(sl_px[i]), float(tp_px[i]), i
                armed = False
            elif short_signal and tp_px[i] < close[i] < sl_px[i]:
                signals.append(
                    Signal(
                        timestamp=pd.Timestamp(ts.iloc[i]),
                        direction="short",
                        entry=float(close[i]),
                        stop=float(sl_px[i]),
                        take_profit=float(tp_px[i]),
                        pattern="pmts_4h5m_short",
                        max_hold_bars=hold,
                        meta={
                            "htf_dir": d_str,
                            "zone_top": float(zone_top[i]),
                            "zone_bot": float(zone_bot[i]),
                            "anchor_high": float(a_hi[i]),
                            "anchor_low": float(a_lo[i]),
                        },
                    )
                )
                in_pos = True
                pos_side = "short"
                pos_stop, pos_tp, entry_i = float(sl_px[i]), float(tp_px[i]), i
                armed = False

        return signals
