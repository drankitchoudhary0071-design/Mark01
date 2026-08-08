"""
PMTS Fib Trailing — port of TradingView Pine:
  Entry at fib touch, SL at 0.7 (+ optional point buffer), TP at swing anchor (or R-multiple).
  Structure: BOS from swing pivots; close beyond 0.7 flips direction.

Extras for lab sweeps (optional):
  - min_range_pts / min_planned_rr quality filters
  - soft exit when structure flips against the trade
  - optional HTF bias filter (bull longs / bear shorts only)
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


def _htf_bias_series(df: pd.DataFrame, htf: str, ema_len: int = 50) -> np.ndarray:
    """Map higher-TF EMA bias onto LTF bars. +1 bull, -1 bear, 0 unknown."""
    from strategy_lab.strategies.mtf_rsi import resample_ohlcv

    h = resample_ohlcv(df, htf)
    if h.empty or len(h) < ema_len + 2:
        return np.zeros(len(df), dtype=int)
    ema = h["close"].astype(float).ewm(span=ema_len, adjust=False).mean()
    bias_htf = np.where(h["close"].to_numpy(dtype=float) > ema.to_numpy(dtype=float), 1, -1)
    # Last closed HTF bar only (no lookahead): shift by 1 HTF bar
    h_ts = pd.to_datetime(h["timestamp"], utc=True)
    bias_df = pd.DataFrame({"timestamp": h_ts, "bias": bias_htf}).copy()
    bias_df["bias"] = bias_df["bias"].shift(1)
    bias_df = bias_df.dropna()
    l_ts = pd.to_datetime(df["timestamp"], utc=True)
    left = pd.DataFrame({"timestamp": l_ts})
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        bias_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged["bias"].fillna(0).astype(int).to_numpy()


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
            # Quality / frequency levers
            "min_range_pts": 0.0,  # skip tiny fib ranges (gold $)
            "min_planned_rr": 0.0,  # skip if TP/SL distance ratio too small
            "flip_soft_exit": False,  # exit at close when structure flips
            "htf_bias": "",  # e.g. "4h" — only long in HTF bull / short in HTF bear
            "htf_ema": 50,
            "reentry_cooldown": 0,  # bars after exit before next entry
            # After TP: arm locked entry fib; require leave-zone then retouch for next entry
            "retouch_after_tp": False,
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
            f"[PMTS] Entry fib touch; SL 0.7 ± buffer pts; {tp_desc}.",
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
        min_range = float(p.get("min_range_pts", 0.0))
        min_rr = float(p.get("min_planned_rr", 0.0))
        flip_exit = bool(p.get("flip_soft_exit", False))
        htf = str(p.get("htf_bias", "") or "").strip()
        htf_ema = int(p.get("htf_ema", 50))
        cooldown = int(p.get("reentry_cooldown", 0))
        retouch_after_tp = bool(p.get("retouch_after_tp", False))

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

        bias = _htf_bias_series(df, htf, htf_ema) if htf else np.zeros(len(df), dtype=int)

        last_sh: Optional[float] = None
        last_sl: Optional[float] = None
        direction: Optional[str] = None  # bull / bear
        anchor_low: Optional[float] = None
        anchor_high: Optional[float] = None

        # Simulate one position at a time like Pine (flat check on entry).
        in_pos = False
        pos_side: Optional[str] = None
        pos_stop = 0.0
        pos_tp = 0.0
        pos_entry = 0.0
        pos_entry_i = -1
        last_exit_i = -10_000
        active_sig: Optional[Signal] = None

        # Locked fib retouch after TP
        armed = False
        armed_side: Optional[str] = None
        armed_entry = 0.0
        armed_sl = 0.0
        armed_away = False

        signals: list[Signal] = []

        def clear_arm() -> None:
            nonlocal armed, armed_side, armed_entry, armed_sl, armed_away
            armed = False
            armed_side = None
            armed_entry = 0.0
            armed_sl = 0.0
            armed_away = False

        def emit(side: str, entry: float, stop: float, tp: float, pattern: str) -> None:
            nonlocal in_pos, pos_side, pos_stop, pos_tp, pos_entry, pos_entry_i, active_sig
            if side == "long" and not (stop < entry < tp):
                return
            if side == "short" and not (tp < entry < stop):
                return
            sig = Signal(
                timestamp=tsi,
                direction=side,  # type: ignore[arg-type]
                entry=float(entry),
                stop=float(stop),
                take_profit=float(tp),
                pattern=pattern,
                max_hold_bars=hold,
                meta={
                    "fib_entry": float(entry),
                    "anchor_low": float(anchor_low) if anchor_low else None,
                    "anchor_high": float(anchor_high) if anchor_high else None,
                    "soft_exits": {},
                    "retouch": pattern.endswith("retouch"),
                },
            )
            signals.append(sig)
            in_pos = True
            pos_side = side
            pos_stop = float(stop)
            pos_tp = float(tp)
            pos_entry = float(entry)
            pos_entry_i = i
            active_sig = sig

        for i in range(len(df)):
            if not np.isnan(ph.iloc[i]):
                last_sh = float(ph.iloc[i])
            if not np.isnan(pl.iloc[i]):
                last_sl = float(pl.iloc[i])

            hi = float(high.iloc[i])
            lo = float(low.iloc[i])
            cl = float(close.iloc[i])
            tsi = pd.Timestamp(ts.iloc[i])

            # Manage open "virtual" position for signal spacing (stop/tp/time/flip)
            if in_pos:
                exited = False
                exit_reason = ""
                if pos_side == "long":
                    if lo <= pos_stop:
                        exited, exit_reason = True, "stop"
                    elif hi >= pos_tp:
                        exited, exit_reason = True, "take_profit"
                else:
                    if hi >= pos_stop:
                        exited, exit_reason = True, "stop"
                    elif lo <= pos_tp:
                        exited, exit_reason = True, "take_profit"
                if i - pos_entry_i >= hold:
                    exited, exit_reason = True, "time"
                if exited:
                    if retouch_after_tp and exit_reason == "take_profit" and pos_side:
                        armed = True
                        armed_side = pos_side
                        armed_entry = pos_entry
                        armed_sl = pos_stop
                        # At TP, price is already away from entry toward target
                        armed_away = True
                    else:
                        clear_arm()
                    in_pos = False
                    pos_side = None
                    active_sig = None
                    last_exit_i = i

            long_signal = False
            short_signal = False
            entry_px = sl_px = tp_px = None
            sl_raw = None

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
                    sl_px = sl_raw - buf
                    tp_px = resolve_tp("long", entry_px, sl_px, anchor_high)

                    bias_ok = bias[i] >= 0 if htf else True
                    range_ok = rng >= min_range
                    risk = abs(entry_px - sl_px)
                    planned = (abs(tp_px - entry_px) / risk) if risk > 0 else 0.0
                    rr_ok = planned >= min_rr
                    cd_ok = (i - last_exit_i) >= cooldown

                    # Live fib entry when flat (still allowed while armed for locked retouch)
                    if (
                        (not in_pos)
                        and lo <= entry_px
                        and bias_ok
                        and range_ok
                        and rr_ok
                        and cd_ok
                    ):
                        long_signal = True

                    if cl < sl_raw:
                        if flip_exit and in_pos and pos_side == "long" and active_sig is not None:
                            soft = active_sig.meta.setdefault("soft_exits", {})
                            soft[tsi] = cl
                            in_pos = False
                            pos_side = None
                            active_sig = None
                            last_exit_i = i
                        prev_high = anchor_high
                        direction = "bear"
                        anchor_high = prev_high
                        anchor_low = lo
                        clear_arm()

            elif direction == "bear":
                assert anchor_low is not None and anchor_high is not None
                if lo < anchor_low:
                    anchor_low = lo
                rng = anchor_high - anchor_low
                if rng > 0:
                    entry_px = anchor_low + rng * entry_lvl
                    sl_raw = anchor_low + rng * sl_lvl
                    sl_px = sl_raw + buf
                    tp_px = resolve_tp("short", entry_px, sl_px, anchor_low)

                    bias_ok = bias[i] <= 0 if htf else True
                    range_ok = rng >= min_range
                    risk = abs(entry_px - sl_px)
                    planned = (abs(tp_px - entry_px) / risk) if risk > 0 else 0.0
                    rr_ok = planned >= min_rr
                    cd_ok = (i - last_exit_i) >= cooldown

                    if (
                        (not in_pos)
                        and hi >= entry_px
                        and bias_ok
                        and range_ok
                        and rr_ok
                        and cd_ok
                    ):
                        short_signal = True

                    if cl > sl_raw:
                        if flip_exit and in_pos and pos_side == "short" and active_sig is not None:
                            soft = active_sig.meta.setdefault("soft_exits", {})
                            soft[tsi] = cl
                            in_pos = False
                            pos_side = None
                            active_sig = None
                            last_exit_i = i
                        prev_low = anchor_low
                        direction = "bull"
                        anchor_low = prev_low
                        anchor_high = hi
                        clear_arm()

            # Locked retouch after TP: leave zone then return to locked entry
            # Runs before live emit so a pure retouch is tagged; live still works if flat.
            if retouch_after_tp and armed and (not in_pos) and armed_side is not None:
                if armed_side == "long":
                    if cl > armed_entry:
                        armed_away = True
                    if armed_away and lo <= armed_entry:
                        tp_live = float(anchor_high) if anchor_high is not None else armed_entry + abs(
                            armed_entry - armed_sl
                        )
                        emit("long", armed_entry, armed_sl, tp_live, "pmts_fib_long_retouch")
                        clear_arm()
                        long_signal = False  # already entered
                else:
                    if cl < armed_entry:
                        armed_away = True
                    if armed_away and hi >= armed_entry:
                        tp_live = float(anchor_low) if anchor_low is not None else armed_entry - abs(
                            armed_sl - armed_entry
                        )
                        emit("short", armed_entry, armed_sl, tp_live, "pmts_fib_short_retouch")
                        clear_arm()
                        short_signal = False

            if long_signal and (not in_pos) and entry_px is not None and sl_px is not None and tp_px is not None:
                emit("long", entry_px, sl_px, tp_px, "pmts_fib_long")
                clear_arm()

            if short_signal and (not in_pos) and entry_px is not None and sl_px is not None and tp_px is not None:
                emit("short", entry_px, sl_px, tp_px, "pmts_fib_short")
                clear_arm()

        return signals
