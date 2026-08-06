"""
PMTS — Pure Math Trading System
4-step price-action logic, no external indicators.

Step 1 (1H): Order blocks via displacement → range → structural break
Step 2 (1H): Fibonacci golden zone 61.8%–79% of impulsive move
Step 3 (5m): Break of Structure (BOS) confirming continuation
Step 4 (5m): Candlestick rejection pattern inside golden zone → entry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class OrderBlockSetup:
    direction: Direction
    impulse_start_idx: int
    impulse_end_idx: int
    impulse_start_price: float
    impulse_end_price: float
    ob_high: float
    ob_low: float
    fib_618: float
    fib_790: float
    zone_top: float
    zone_bottom: float
    structure_break_time: pd.Timestamp
    structure_break_idx: int
    valid_until: pd.Timestamp | None = None


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: Direction
    entry: float
    stop: float
    take_profit: float
    setup: OrderBlockSetup
    pattern: str
    bos_time: pd.Timestamp
    meta: dict = field(default_factory=dict)


@dataclass
class PMTSParams:
    # Displacement / order block (1H)
    displacement_body_mult: float = 2.0  # body vs median body of lookback
    displacement_lookback: int = 20
    range_max_atr_frac: float = 0.55  # consolidation width vs ATR
    range_min_bars: int = 3
    range_max_bars: int = 18
    atr_period: int = 14
    structure_lookback: int = 5  # swing for structural break
    # Fib
    fib_low: float = 0.618
    fib_high: float = 0.79
    # 5m BOS
    bos_swing_lookback: int = 5
    # Risk
    rr_target: float = 2.0
    max_stop_atr_mult: float = 3.0  # reject setups with absurd stops
    min_stop_pct: float = 0.0035  # >=0.35% so ~0.30% RT costs < 1R
    setup_max_age_hours: int = 72  # HTF setup expiry
    # Pattern
    pin_wick_ratio: float = 2.0
    # Improved-mode toggles (baseline = False)
    use_london_ny_filter: bool = False
    use_volatility_filter: bool = False
    use_bos_retest: bool = False
    use_tight_fib: bool = False  # 0.618–0.705
    use_htf_trend_filter: bool = False  # only long above SMA, short below
    htf_sma_period: int = 100
    min_impulse_atr: float = 1.5
    vol_atr_percentile: float = 0.0  # if >0, require ATR%ile above this
    # Session hours UTC (London open → NY afternoon)
    session_start_hour: int = 7
    session_end_hour: int = 20
    # Prefer longs in improved mode if True (still needs short book optional)
    long_only: bool = False


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def swing_highs(high: np.ndarray, lookback: int) -> np.ndarray:
    """True at local swing highs (strict)."""
    n = len(high)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = high[i - lookback : i + lookback + 1]
        if high[i] == window.max() and np.sum(window == high[i]) == 1:
            out[i] = True
    return out


def swing_lows(low: np.ndarray, lookback: int) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = low[i - lookback : i + lookback + 1]
        if low[i] == window.min() and np.sum(window == window.min()) == 1:
            out[i] = True
    return out


def detect_order_blocks(df_1h: pd.DataFrame, params: PMTSParams) -> list[OrderBlockSetup]:
    """
    Displacement candle → consolidation range → break of structure.
    Bullish OB: last bearish candle before bullish displacement (demand).
    Bearish OB: last bullish candle before bearish displacement (supply).
    """
    df = df_1h.copy()
    df["atr"] = atr(df, params.atr_period)
    df["body"] = (df["close"] - df["open"]).abs()
    df["median_body"] = df["body"].rolling(params.displacement_lookback, min_periods=5).median()
    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df["timestamp"].values
    atr_v = df["atr"].values
    body = df["body"].values
    med_body = df["median_body"].values
    bullish = df["bullish"].values
    bearish = df["bearish"].values

    setups: list[OrderBlockSetup] = []
    i = params.displacement_lookback + 2
    n = len(df)

    while i < n - params.range_min_bars - 2:
        if np.isnan(atr_v[i]) or np.isnan(med_body[i]) or med_body[i] <= 0:
            i += 1
            continue

        is_disp_bull = (
            bullish[i]
            and body[i] >= params.displacement_body_mult * med_body[i]
            and body[i] >= params.min_impulse_atr * 0.5 * atr_v[i]
        )
        is_disp_bear = (
            bearish[i]
            and body[i] >= params.displacement_body_mult * med_body[i]
            and body[i] >= params.min_impulse_atr * 0.5 * atr_v[i]
        )

        if not (is_disp_bull or is_disp_bear):
            i += 1
            continue

        direction: Direction = "long" if is_disp_bull else "short"
        impulse_start_idx = i
        impulse_end_idx = i
        impulse_start_price = opens[i] if is_disp_bull else opens[i]
        # Extend impulse while consecutive strong closes in same direction (max 3 bars)
        for j in range(i + 1, min(i + 4, n)):
            cont = (bullish[j] and is_disp_bull) or (bearish[j] and is_disp_bear)
            if cont and body[j] >= 0.8 * med_body[j]:
                impulse_end_idx = j
            else:
                break

        impulse_end_price = closes[impulse_end_idx]
        if is_disp_bull:
            impulse_start_price = lows[impulse_start_idx]  # swing low origin
            # Prefer prior swing low within lookback
            look_lo = max(0, impulse_start_idx - params.structure_lookback * 3)
            impulse_start_price = float(np.min(lows[look_lo : impulse_start_idx + 1]))
        else:
            look_hi = max(0, impulse_start_idx - params.structure_lookback * 3)
            impulse_start_price = float(np.max(highs[look_hi : impulse_start_idx + 1]))

        move = abs(impulse_end_price - impulse_start_price)
        if atr_v[impulse_end_idx] > 0 and move < params.min_impulse_atr * atr_v[impulse_end_idx]:
            i += 1
            continue

        # Consolidation after impulse
        range_start = impulse_end_idx + 1
        found_range = False
        range_end = range_start
        for length in range(params.range_min_bars, params.range_max_bars + 1):
            re = range_start + length - 1
            if re >= n:
                break
            rh = float(np.max(highs[range_start : re + 1]))
            rl = float(np.min(lows[range_start : re + 1]))
            width = rh - rl
            ref_atr = atr_v[re]
            if np.isnan(ref_atr) or ref_atr <= 0:
                continue
            if width <= params.range_max_atr_frac * ref_atr * 2.5:
                # also require no new extreme beyond impulse end by much
                if direction == "long" and rh < impulse_end_price + 0.5 * ref_atr:
                    found_range = True
                    range_end = re
                elif direction == "short" and rl > impulse_end_price - 0.5 * ref_atr:
                    found_range = True
                    range_end = re
        if not found_range:
            i += 1
            continue

        # Structural break after range: close beyond range high/low
        broken = False
        break_idx = None
        search_end = min(range_end + 12, n - 1)
        range_high = float(np.max(highs[range_start : range_end + 1]))
        range_low = float(np.min(lows[range_start : range_end + 1]))
        for k in range(range_end + 1, search_end + 1):
            if direction == "long" and closes[k] > range_high:
                broken = True
                break_idx = k
                break
            if direction == "short" and closes[k] < range_low:
                broken = True
                break_idx = k
                break
        if not broken or break_idx is None:
            i += 1
            continue

        # Order block candle: opposite-color candle at start of impulse
        ob_idx = impulse_start_idx
        if direction == "long":
            # last bearish candle at/before impulse start
            for t in range(impulse_start_idx, max(impulse_start_idx - 5, 0) - 1, -1):
                if bearish[t] or (not bullish[t]):
                    ob_idx = t
                    break
            ob_high = float(highs[ob_idx])
            ob_low = float(lows[ob_idx])
        else:
            for t in range(impulse_start_idx, max(impulse_start_idx - 5, 0) - 1, -1):
                if bullish[t] or (not bearish[t]):
                    ob_idx = t
                    break
            ob_high = float(highs[ob_idx])
            ob_low = float(lows[ob_idx])

        # Update impulse end to structural break extreme if stronger
        if direction == "long":
            impulse_end_price = float(np.max(highs[impulse_start_idx : break_idx + 1]))
            impulse_start_price = float(np.min(lows[max(0, impulse_start_idx - 3) : impulse_start_idx + 1]))
        else:
            impulse_end_price = float(np.min(lows[impulse_start_idx : break_idx + 1]))
            impulse_start_price = float(np.max(highs[max(0, impulse_start_idx - 3) : impulse_start_idx + 1]))

        move = abs(impulse_end_price - impulse_start_price)
        if move <= 0:
            i += 1
            continue

        fib_low_r = params.fib_low
        fib_high_r = 0.705 if params.use_tight_fib else params.fib_high

        if direction == "long":
            # Retracement from high (end) toward low (start)
            fib_618 = impulse_end_price - fib_low_r * move
            fib_790 = impulse_end_price - fib_high_r * move
            zone_top = max(fib_618, fib_790)
            zone_bottom = min(fib_618, fib_790)
        else:
            fib_618 = impulse_end_price + fib_low_r * move
            fib_790 = impulse_end_price + fib_high_r * move
            zone_top = max(fib_618, fib_790)
            zone_bottom = min(fib_618, fib_790)

        break_ts = pd.Timestamp(times[break_idx])
        if break_ts.tzinfo is None:
            break_ts = break_ts.tz_localize("UTC")
        else:
            break_ts = break_ts.tz_convert("UTC")
        valid_until = break_ts + pd.Timedelta(hours=params.setup_max_age_hours)

        setups.append(
            OrderBlockSetup(
                direction=direction,
                impulse_start_idx=int(impulse_start_idx),
                impulse_end_idx=int(impulse_end_idx),
                impulse_start_price=float(impulse_start_price),
                impulse_end_price=float(impulse_end_price),
                ob_high=ob_high,
                ob_low=ob_low,
                fib_618=float(fib_618),
                fib_790=float(fib_790),
                zone_top=float(zone_top),
                zone_bottom=float(zone_bottom),
                structure_break_time=break_ts,
                structure_break_idx=int(break_idx),
                valid_until=valid_until,
            )
        )
        # Skip ahead past this structure to reduce duplicate setups
        i = break_idx + 1

    return setups


def _in_session(ts: pd.Timestamp, params: PMTSParams) -> bool:
    if not params.use_london_ny_filter:
        return True
    hour = ts.tz_convert("UTC").hour if ts.tzinfo else ts.hour
    return params.session_start_hour <= hour < params.session_end_hour


def _is_engulfing(o, h, l, c, po, ph, pl, pc, direction: Direction) -> bool:
    if direction == "long":
        return pc < po and c > o and c >= po and o <= pc and (c - o) > (po - pc) * 0.9
    return pc > po and c < o and c <= po and o >= pc and (o - c) > (pc - po) * 0.9


def _is_pin_bar(o, h, l, c, direction: Direction, wick_ratio: float) -> bool:
    body = abs(c - o)
    if body <= 1e-12:
        body = 1e-12
    upper = h - max(c, o)
    lower = min(c, o) - l
    if direction == "long":
        # hammer: long lower wick
        return lower >= wick_ratio * body and upper <= body * 1.2 and c >= o
    # shooting star
    return upper >= wick_ratio * body and lower <= body * 1.2 and c <= o


def _is_rejection(o, h, l, c, direction: Direction, wick_ratio: float) -> str | None:
    if direction == "long":
        if _is_engulfing(o, h, l, c, o, h, l, c, direction):
            pass  # need prior — handled outside
    return None


def detect_pattern(
    row: pd.Series,
    prev: pd.Series,
    direction: Direction,
    params: PMTSParams,
) -> str | None:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
    if _is_engulfing(o, h, l, c, po, ph, pl, pc, direction):
        return "engulfing"
    if _is_pin_bar(o, h, l, c, direction, params.pin_wick_ratio):
        return "pin_bar"
    # Inside-bar breakout rejection: close back into zone after wick through
    body = abs(c - o)
    if direction == "long":
        lower = min(c, o) - l
        if lower >= 1.5 * max(body, 1e-12) and c > o:
            return "rejection_wick"
    else:
        upper = h - max(c, o)
        if upper >= 1.5 * max(body, 1e-12) and c < o:
            return "rejection_wick"
    return None


def generate_signals(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    params: PMTSParams | None = None,
) -> list[Signal]:
    """Full 4-step PMTS signal generation."""
    params = params or PMTSParams()
    setups = detect_order_blocks(df_1h, params)
    if not setups:
        return []

    df5 = df_5m.copy().reset_index(drop=True)
    df5["atr"] = atr(df5, params.atr_period)
    # Volatility filter prep on 1H ATR percentile + optional SMA trend
    df1 = df_1h.copy()
    df1["atr"] = atr(df1, params.atr_period)
    df1["atr_pct"] = df1["atr"].rank(pct=True)
    df1["sma"] = df1["close"].rolling(params.htf_sma_period, min_periods=params.htf_sma_period).mean()

    highs = df5["high"].values
    lows = df5["low"].values
    closes = df5["close"].values
    n = len(df5)
    sh = swing_highs(highs, params.bos_swing_lookback)
    sl = swing_lows(lows, params.bos_swing_lookback)

    signals: list[Signal] = []
    used_setup_ids: set[int] = set()

    for s_i, setup in enumerate(setups):
        if params.long_only and setup.direction != "long":
            continue
        # Map HTF break time into 5m index
        start_mask = df5["timestamp"] >= setup.structure_break_time
        if not start_mask.any():
            continue
        start_i = int(np.argmax(start_mask.to_numpy()))
        end_time = setup.valid_until or (setup.structure_break_time + pd.Timedelta(hours=72))
        end_mask = df5["timestamp"] <= end_time
        end_i = int(np.where(end_mask.to_numpy())[0][-1]) if end_mask.any() else n - 1

        # Optional vol filter at setup time
        nearest = df1.iloc[(df1["timestamp"] - setup.structure_break_time).abs().argmin()]
        if params.use_volatility_filter and params.vol_atr_percentile > 0:
            if float(nearest["atr_pct"]) < params.vol_atr_percentile:
                continue
        if params.use_htf_trend_filter and not np.isnan(nearest["sma"]):
            if setup.direction == "long" and float(nearest["close"]) < float(nearest["sma"]):
                continue
            if setup.direction == "short" and float(nearest["close"]) > float(nearest["sma"]):
                continue

        bos_done = False
        bos_level = None
        bos_time = None
        bos_idx = None
        last_swing = None

        for i in range(start_i + params.bos_swing_lookback, end_i + 1):
            ts = pd.Timestamp(df5.at[i, "timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            if not _in_session(ts, params):
                continue

            # Track swings for BOS
            if setup.direction == "long":
                # find most recent swing high before i
                for j in range(i - 1, max(start_i, i - 80), -1):
                    if sh[j]:
                        last_swing = highs[j]
                        break
                if last_swing is not None and not bos_done and closes[i] > last_swing:
                    bos_done = True
                    bos_level = last_swing
                    bos_time = ts
                    bos_idx = i
                    if params.use_bos_retest:
                        # wait for retest — don't enter on BOS bar
                        continue
            else:
                for j in range(i - 1, max(start_i, i - 80), -1):
                    if sl[j]:
                        last_swing = lows[j]
                        break
                if last_swing is not None and not bos_done and closes[i] < last_swing:
                    bos_done = True
                    bos_level = last_swing
                    bos_time = ts
                    bos_idx = i
                    if params.use_bos_retest:
                        continue

            if not bos_done:
                continue

            # After BOS: look for pattern inside golden zone
            if params.use_bos_retest and bos_idx is not None and i <= bos_idx:
                continue

            price_in_zone = (
                lows[i] <= setup.zone_top and highs[i] >= setup.zone_bottom
            )
            # Prefer close or wick touching zone
            if not price_in_zone:
                continue

            if i < 1:
                continue
            pattern = detect_pattern(df5.iloc[i], df5.iloc[i - 1], setup.direction, params)
            if pattern is None:
                continue

            entry = float(closes[i])
            # Stop beyond order block / zone extreme
            if setup.direction == "long":
                stop = min(setup.ob_low, setup.zone_bottom) - 0.1 * float(df5.at[i, "atr"] or 0)
                # Enforce minimum stop distance (cost-aware)
                stop = min(stop, entry * (1 - params.min_stop_pct))
                risk = entry - stop
                if risk <= 0:
                    continue
                tp = entry + params.rr_target * risk
            else:
                stop = max(setup.ob_high, setup.zone_top) + 0.1 * float(df5.at[i, "atr"] or 0)
                stop = max(stop, entry * (1 + params.min_stop_pct))
                risk = stop - entry
                if risk <= 0:
                    continue
                tp = entry - params.rr_target * risk

            atr5 = float(df5.at[i, "atr"]) if not np.isnan(df5.at[i, "atr"]) else risk
            if atr5 > 0 and risk > params.max_stop_atr_mult * atr5 * 12:  # allow wider HTF stops
                # Compare vs 1H-ish scale: 12 * 5m ATR ≈ 1H
                pass
            if risk / entry < params.min_stop_pct * 0.95:
                continue

            signals.append(
                Signal(
                    timestamp=ts,
                    direction=setup.direction,
                    entry=entry,
                    stop=float(stop),
                    take_profit=float(tp),
                    setup=setup,
                    pattern=pattern,
                    bos_time=bos_time or ts,
                    meta={
                        "bos_level": bos_level,
                        "setup_id": s_i,
                        "zone_top": setup.zone_top,
                        "zone_bottom": setup.zone_bottom,
                    },
                )
            )
            used_setup_ids.add(s_i)
            break  # one entry per HTF setup

    # Deduplicate by timestamp
    signals.sort(key=lambda s: s.timestamp)
    deduped: list[Signal] = []
    last_ts = None
    for sig in signals:
        if last_ts is not None and (sig.timestamp - last_ts) < pd.Timedelta(minutes=30):
            continue
        deduped.append(sig)
        last_ts = sig.timestamp
    return deduped


def improved_params() -> PMTSParams:
    """Concrete improvements based on gold microstructure + cost reality; tuned on IS only."""
    return PMTSParams(
        displacement_body_mult=2.2,
        range_min_bars=3,
        range_max_bars=14,
        fib_low=0.618,
        fib_high=0.79,
        use_tight_fib=True,  # 61.8–70.5 often cleaner on gold
        use_london_ny_filter=True,
        use_volatility_filter=True,
        vol_atr_percentile=0.30,
        use_bos_retest=True,
        use_htf_trend_filter=True,
        htf_sma_period=100,
        min_impulse_atr=1.8,
        rr_target=2.0,
        min_stop_pct=0.004,  # 0.40% min risk distance
        setup_max_age_hours=48,
        pin_wick_ratio=2.2,
        session_start_hour=8,
        session_end_hour=18,  # focus hours with better avg R in diagnosis
        long_only=False,
    )
