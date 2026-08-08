"""
PMTS Fib Trailing — port of TradingView Pine:
  Entry at 0.6 fib touch, SL at 0.7 (+ optional point buffer), TP at swing anchor.
  Structure: BOS from swing pivots; close beyond 0.7 flips direction.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import Strategy


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Confirmed pivot high at bar i-right when i is current (like ta.pivothigh)."""
    n = len(high)
    out = np.full(n, np.nan)
    h = high.to_numpy(dtype=float)
    for i in range(left + right, n):
        # candidate pivot bar index
        c = i - right
        val = h[c]
        ok = True
        for j in range(c - left, c + right + 1):
            if j == c:
                continue
            if h[j] >= val:
                ok = False
                break
        if ok:
            out[i] = val  # confirmed on bar i (delayed by `right`)
    return pd.Series(out, index=high.index)


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = np.full(n, np.nan)
    lo = low.to_numpy(dtype=float)
    for i in range(left + right, n):
        c = i - right
        val = lo[c]
        ok = True
        for j in range(c - left, c + right + 1):
            if j == c:
                continue
            if lo[j] <= val:
                ok = False
                break
        if ok:
            out[i] = val
    return pd.Series(out, index=low.index)


class PmtsFibTrailingStrategy(Strategy):
    """Fib trailing structure strategy (PMTS Pine port)."""

    name = "pmts_fib_trailing"
    library = "pandas PMTS fib trailing"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "swing_len": 3,
            "entry_level": 0.6,
            "sl_level": 0.7,
            # SL buffer in price points (gold $): below 0.7 for longs, above for shorts
            "sl_buffer_pts": 3.0,
            # TP: "anchor" = swing extreme; "rr" = entry ± rr_multiple * |entry-sl|
            "tp_mode": "anchor",
            "rr_multiple": 3.0,
            "max_hold_bars": 500,
        }

    def __init__(self, **params: Any):
        super().__init__(**params)
        mode = str(self.params.get("tp_mode", "anchor"))
        rr = self.params.get("rr_multiple", 3.0)
        tp_desc = (
            "TP = swing anchor"
            if mode == "anchor"
            else f"TP = {rr}× SL distance (R-multiple)"
        )
        self.curve_fit_flags = list(self.curve_fit_flags) + [
            f"[PMTS] Entry 0.6 fib touch; SL 0.7 ± buffer pts; {tp_desc}.",
            "[PMTS] Close beyond 0.7 flips structure (wick alone does not).",
        ]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        p = self.params
        swing = int(p["swing_len"])
        entry_lvl = float(p["entry_level"])
        sl_lvl = float(p["sl_level"])
        buf = float(p["sl_buffer_pts"])
        hold = int(p["max_hold_bars"])
        tp_mode = str(p.get("tp_mode", "anchor"))
        rr_mult = float(p.get("rr_multiple", 3.0))

        def resolve_tp(side: str, entry: float, stop: float, anchor_tp: float) -> float:
            if tp_mode == "anchor":
                return anchor_tp
            risk = abs(entry - stop)
            if side == "long":
                return entry + rr_mult * risk
            return entry - rr_mult * risk

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        ts = pd.to_datetime(df["timestamp"], utc=True)

        ph = pivot_high(high, swing, swing)
        pl = pivot_low(low, swing, swing)

        last_sh: Optional[float] = None
        last_sl: Optional[float] = None
        direction: Optional[str] = None  # bull / bear
        anchor_low: Optional[float] = None
        anchor_high: Optional[float] = None

        # Simulate one position at a time like Pine (flat check on entry).
        # Engine also enforces one-trade-at-a-time; we still avoid overlapping signals.
        in_pos = False
        pos_side: Optional[str] = None
        pos_stop = 0.0
        pos_tp = 0.0
        pos_entry_i = -1

        signals: list[Signal] = []

        for i in range(len(df)):
            if not np.isnan(ph.iloc[i]):
                last_sh = float(ph.iloc[i])
            if not np.isnan(pl.iloc[i]):
                last_sl = float(pl.iloc[i])

            hi = float(high.iloc[i])
            lo = float(low.iloc[i])
            cl = float(close.iloc[i])

            # Manage open "virtual" position for signal spacing (stop/tp/time)
            if in_pos:
                exited = False
                if pos_side == "long":
                    if lo <= pos_stop or hi >= pos_tp:
                        exited = True
                else:
                    if hi >= pos_stop or lo <= pos_tp:
                        exited = True
                if i - pos_entry_i >= hold:
                    exited = True
                if exited:
                    in_pos = False
                    pos_side = None

            long_signal = False
            short_signal = False
            entry_px = sl_px = tp_px = None

            if direction is None:
                if last_sh is not None and last_sl is not None:
                    if cl > last_sh:
                        direction = "bull"
                        anchor_low = last_sl
                        anchor_high = hi
                    elif cl < last_sl:
                        direction = "bear"
                        anchor_high = last_sh
                        anchor_low = lo

            elif direction == "bull":
                assert anchor_low is not None and anchor_high is not None
                if hi > anchor_high:
                    anchor_high = hi
                rng = anchor_high - anchor_low
                if rng > 0:
                    entry_px = anchor_high - rng * entry_lvl
                    sl_raw = anchor_high - rng * sl_lvl
                    sl_px = sl_raw - buf  # 2–5 pts below 0.7 for gold long
                    tp_px = resolve_tp("long", entry_px, sl_px, anchor_high)

                    if (not in_pos) and lo <= entry_px:
                        long_signal = True

                    # close-confirmed reversal past 0.7 (use raw 0.7, not buffered SL)
                    if cl < sl_raw:
                        prev_high = anchor_high
                        direction = "bear"
                        anchor_high = prev_high
                        anchor_low = lo

            elif direction == "bear":
                assert anchor_low is not None and anchor_high is not None
                if lo < anchor_low:
                    anchor_low = lo
                rng = anchor_high - anchor_low
                if rng > 0:
                    entry_px = anchor_low + rng * entry_lvl
                    sl_raw = anchor_low + rng * sl_lvl
                    sl_px = sl_raw + buf  # 2–5 pts above 0.7 for gold short
                    tp_px = resolve_tp("short", entry_px, sl_px, anchor_low)

                    if (not in_pos) and hi >= entry_px:
                        short_signal = True

                    if cl > sl_raw:
                        prev_low = anchor_low
                        direction = "bull"
                        anchor_low = prev_low
                        anchor_high = hi

            if long_signal and entry_px is not None and sl_px is not None and tp_px is not None:
                # Sanity: long SL below entry, TP above
                if sl_px < entry_px < tp_px:
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(ts.iloc[i]),
                            direction="long",
                            entry=float(entry_px),  # fill at 0.6 fib (wick touch)
                            stop=float(sl_px),
                            take_profit=float(tp_px),
                            pattern="pmts_fib_long",
                            max_hold_bars=hold,
                            meta={
                                "fib_entry": float(entry_px),
                                "anchor_low": float(anchor_low) if anchor_low else None,
                                "anchor_high": float(anchor_high) if anchor_high else None,
                            },
                        )
                    )
                    in_pos = True
                    pos_side = "long"
                    pos_stop = float(sl_px)
                    pos_tp = float(tp_px)
                    pos_entry_i = i

            if short_signal and entry_px is not None and sl_px is not None and tp_px is not None:
                if tp_px < entry_px < sl_px:
                    signals.append(
                        Signal(
                            timestamp=pd.Timestamp(ts.iloc[i]),
                            direction="short",
                            entry=float(entry_px),  # fill at 0.6 fib (wick touch)
                            stop=float(sl_px),
                            take_profit=float(tp_px),
                            pattern="pmts_fib_short",
                            max_hold_bars=hold,
                            meta={
                                "fib_entry": float(entry_px),
                                "anchor_low": float(anchor_low) if anchor_low else None,
                                "anchor_high": float(anchor_high) if anchor_high else None,
                            },
                        )
                    )
                    in_pos = True
                    pos_side = "short"
                    pos_stop = float(sl_px)
                    pos_tp = float(tp_px)
                    pos_entry_i = i

        return signals
