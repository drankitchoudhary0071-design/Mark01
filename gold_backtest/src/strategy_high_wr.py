"""
High Win-Rate Gold Momentum (HWR-GM)
Target: >=50% win rate at fixed 1:2 RR on PAXG/USDT.

Edge
----
Long-only continuation in a confirmed 1H uptrend:
  - SMA50 > SMA100 and price > SMA100 (regime)
  - ADX(14) > 35 and +DI > -DI (strong directional trend)
  - London/NY session only (08:00–17:00 UTC)
  - 15m micro BOS: higher-low bar that closes above prior high
  - RSI(14) on 15m in 50–70 (momentum, not exhaustion)

Risk
----
  - Stop below recent 8-bar swing low (buffered), floored at 1.0% of price
    so ~0.30% round-trip costs stay << 1R
  - Take profit = entry + 2.0 * risk (fixed 1:2)
  - Reject stops wider than 2%
  - Cooldown 8 bars (~2h) between entries

Params (adx_min, min_stop_pct, swing_look, cooldown) are IS-selected only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy_pmts import Signal, OrderBlockSetup, atr


@dataclass
class HighWRParams:
    adx_period: int = 14
    adx_min: float = 35.0
    sma_fast: int = 50
    sma_slow: int = 100
    rsi_period: int = 14
    rsi_lo: float = 50.0
    rsi_hi: float = 70.0
    session_start: int = 8
    session_end: int = 17
    swing_look: int = 8
    min_stop_pct: float = 0.01  # 1.0% — cost-aware
    max_stop_pct: float = 0.02
    rr_target: float = 2.0
    cooldown_bars: int = 8
    stop_atr_buffer: float = 0.15


def _adx_frame(h: pd.DataFrame, period: int) -> pd.DataFrame:
    up = h["high"].diff()
    down = -h["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [
            (h["high"] - h["low"]),
            (h["high"] - h["close"].shift()).abs(),
            (h["low"] - h["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_h = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).rolling(period).mean() / atr_h
    minus_di = 100 * pd.Series(minus_dm, index=h.index).rolling(period).mean() / atr_h
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    h = h.copy()
    h["adx"] = dx.rolling(period).mean()
    h["plus_di"] = plus_di
    h["minus_di"] = minus_di
    return h


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_frame(df_1h: pd.DataFrame, df_15m: pd.DataFrame, params: HighWRParams) -> pd.DataFrame:
    h = df_1h.copy()
    h[f"sma{params.sma_fast}"] = h["close"].rolling(params.sma_fast).mean()
    h[f"sma{params.sma_slow}"] = h["close"].rolling(params.sma_slow).mean()
    h = _adx_frame(h, params.adx_period)
    cols = [
        "timestamp",
        f"sma{params.sma_fast}",
        f"sma{params.sma_slow}",
        "adx",
        "plus_di",
        "minus_di",
    ]
    m = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        h[cols].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).reset_index(drop=True)
    m["atr15"] = atr(m, 14)
    m["rsi15"] = _rsi(m["close"], params.rsi_period)
    m["hour"] = m["timestamp"].dt.hour
    return m


def generate_high_wr_signals(
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    params: HighWRParams | None = None,
) -> list[Signal]:
    params = params or HighWRParams()
    m = build_feature_frame(df_1h, df_15m, params)
    sma_f = f"sma{params.sma_fast}"
    sma_s = f"sma{params.sma_slow}"

    bull = (m["close"] > m[sma_s]) & (m[sma_f] > m[sma_s])
    strong = bull & (m["adx"] > params.adx_min) & (m["plus_di"] > m["minus_di"])
    session = m["hour"].between(params.session_start, params.session_end - 1)
    micro_bos = (m["low"] > m["low"].shift(1)) & (m["close"] > m["high"].shift(1))
    rsi_ok = m["rsi15"].between(params.rsi_lo, params.rsi_hi)
    entry = strong & session & micro_bos & rsi_ok

    signals: list[Signal] = []
    last_i = -10_000
    for i in np.where(entry.fillna(False).to_numpy())[0]:
        if i - last_i < params.cooldown_bars:
            continue
        if i < params.swing_look:
            continue
        row = m.iloc[i]
        if np.isnan(row["atr15"]) or np.isnan(row["adx"]):
            continue

        entry_px = float(row["close"])
        swing_low = float(m["low"].iloc[i - params.swing_look : i + 1].min())
        stop = swing_low - params.stop_atr_buffer * float(row["atr15"])
        stop = min(stop, entry_px * (1 - params.min_stop_pct))
        risk = entry_px - stop
        stop_pct = risk / entry_px
        if stop_pct < params.min_stop_pct * 0.95:
            continue
        if stop_pct > params.max_stop_pct:
            continue

        tp = entry_px + params.rr_target * risk
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        setup = OrderBlockSetup(
            direction="long",
            impulse_start_idx=int(i),
            impulse_end_idx=int(i),
            impulse_start_price=entry_px,
            impulse_end_price=entry_px,
            ob_high=float(row["high"]),
            ob_low=float(stop),
            fib_618=entry_px,
            fib_790=entry_px,
            zone_top=entry_px,
            zone_bottom=float(stop),
            structure_break_time=ts,
            structure_break_idx=int(i),
        )
        signals.append(
            Signal(
                timestamp=ts,
                direction="long",
                entry=entry_px,
                stop=float(stop),
                take_profit=float(tp),
                setup=setup,
                pattern="hl_bos_adx",
                bos_time=ts,
                meta={
                    "adx": float(row["adx"]),
                    "rsi15": float(row["rsi15"]),
                    "plus_di": float(row["plus_di"]),
                    "minus_di": float(row["minus_di"]),
                    "stop_pct": float(stop_pct),
                },
            )
        )
        last_i = i
    return signals


def is_tuned_params() -> HighWRParams:
    """Locked params from IS grid (adx / min_stop / swing / cooldown)."""
    return HighWRParams(
        adx_min=35.0,
        min_stop_pct=0.01,
        swing_look=8,
        cooldown_bars=8,
        rr_target=2.0,
        rsi_lo=50.0,
        rsi_hi=70.0,
    )
