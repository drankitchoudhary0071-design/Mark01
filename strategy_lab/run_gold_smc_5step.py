#!/usr/bin/env python3
"""
Gold Scalping — 5-Step SMC Strategy (algorithmic backtest)

Step 1: 1H + 15M trend alignment via Break of Structure (swing pivots)
Step 2: Mark 15M POI (bullish FVG / demand OB, or bearish FVG / supply OB)
        + liquidity at recent 15M swing lows/highs
Step 3: Wait until LTF price reaches POI (no mid-air entries)
Step 4: Entry after liquidity sweep
        - aggressive: enter on sweep confirmation
        - conservative: sweep → LTF market shift → retest new demand/supply
Step 5: TP = nearest logical 15M swing high (long) / swing low (short)
        SL = sweep candle extreme (± buffer)

Data: 5m LTF (resample → 15m, 1H). Gold: PAXGUSDT (Binance) + XAUUSD (Dukascopy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch

# ── adjustable parameters ──────────────────────────────────────────────
swing_len_1h = 3
swing_len_15m = 3
swing_len_5m = 3
impulse_atr_mult = 1.2  # displacement candle body ≥ this × ATR
atr_len = 14
sl_buffer = 0.5  # $ beyond sweep extreme (gold)
poi_touch_pad = 0.0  # allow slight pierce of zone
max_poi_age_bars_15m = 96  # ~4 days of 15m
min_rr = 1.5  # skip entry if nearest valid TP gives RR below this
asia_skip = False  # strategy doc does not require Asia filter

INITIAL_CAPITAL = 10_000.0
FIXED_NOTIONAL = 3_000.0
DAYS = 365

OUT = Path(__file__).resolve().parent / "results" / "gold_smc_5step"
OUT.mkdir(parents=True, exist_ok=True)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    out = (
        x.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    out.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("s")
    return out


def atr_wilder(h, l, c, period: int) -> np.ndarray:
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def pivot_confirm(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Pivot confirmed on bar i (center = i-right). Returns ph/pl price arrays (nan else)."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        val = high[c]
        if all(high[j] < val for j in range(c - left, c + right + 1) if j != c):
            ph[i] = val
        val = low[c]
        if all(low[j] > val for j in range(c - left, c + right + 1) if j != c):
            pl[i] = val
    return ph, pl


def structure_direction(close: np.ndarray, ph: np.ndarray, pl: np.ndarray):
    """BOS direction: close beyond last swing high/low."""
    n = len(close)
    last_sh = np.nan
    last_sl = np.nan
    direction = np.array([None] * n, dtype=object)
    dir_s = None
    for i in range(n):
        if np.isfinite(ph[i]):
            last_sh = ph[i]
        if np.isfinite(pl[i]):
            last_sl = pl[i]
        if np.isfinite(last_sl) and close[i] < last_sl:
            dir_s = "bear"
        if np.isfinite(last_sh) and close[i] > last_sh:
            dir_s = "bull"
        direction[i] = dir_s
    return direction, last_sh, last_sl  # last_* unused; kept for clarity


@dataclass
class POI:
    side: str  # bull demand / bear supply
    z_low: float
    z_high: float
    created_i: int  # 15m index
    kind: str  # fvg | ob


def detect_pois_15m(df15: pd.DataFrame) -> list[list[POI]]:
    """Per-bar list of active POIs (cumulative state snapshot is rebuilt in sim)."""
    h = df15["high"].to_numpy(float)
    l = df15["low"].to_numpy(float)
    o = df15["open"].to_numpy(float)
    c = df15["close"].to_numpy(float)
    atr = atr_wilder(h, l, c, atr_len)
    n = len(df15)
    # We return events: list of (bar_index, POI) creations
    events: list[tuple[int, POI]] = []
    for i in range(2, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        # Bullish FVG: gap between candle i-2 high and candle i low
        if l[i] > h[i - 2] and c[i - 1] > o[i - 1]:
            body = abs(c[i - 1] - o[i - 1])
            if body >= impulse_atr_mult * atr[i]:
                events.append(
                    (
                        i,
                        POI("bull", float(h[i - 2]), float(l[i]), i, "fvg"),
                    )
                )
        # Bearish FVG
        if h[i] < l[i - 2] and c[i - 1] < o[i - 1]:
            body = abs(c[i - 1] - o[i - 1])
            if body >= impulse_atr_mult * atr[i]:
                events.append(
                    (
                        i,
                        POI("bear", float(h[i]), float(l[i - 2]), i, "fvg"),
                    )
                )
        # Demand OB: last down candle before bullish impulse bar i
        if c[i] > o[i] and (c[i] - o[i]) >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, max(i - 8, -1), -1):
                if c[j] < o[j]:
                    events.append(
                        (
                            i,
                            POI("bull", float(l[j]), float(max(o[j], c[j])), i, "ob"),
                        )
                    )
                    break
        # Supply OB: last up candle before bearish impulse
        if c[i] < o[i] and (o[i] - c[i]) >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, max(i - 8, -1), -1):
                if c[j] > o[j]:
                    events.append(
                        (
                            i,
                            POI("bear", float(min(o[j], c[j])), float(h[j]), i, "ob"),
                        )
                    )
                    break
    return events  # type: ignore[return-value]


def nearest_swing_above(level: float, swings: list[float], sl: float, min_rr: float) -> float | None:
    risk = level - sl
    if risk <= 0:
        return None
    cands = sorted(s for s in swings if s > level and (s - level) / risk >= min_rr)
    return cands[0] if cands else None


def nearest_swing_below(level: float, swings: list[float], sl: float, min_rr: float) -> float | None:
    risk = sl - level
    if risk <= 0:
        return None
    cands = sorted((s for s in swings if s < level and (level - s) / risk >= min_rr), reverse=True)
    return cands[0] if cands else None


def load_gold_5m(symbol: str) -> pd.DataFrame:
    if symbol.upper() == "XAUUSD":
        path = Path(__file__).resolve().parent / "cache" / "XAUUSD_5m_365d.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        print(f"Loaded {path} ({len(df)} bars)")
        return df
    return load_or_fetch(symbol, "5m", DAYS)


def run_backtest(symbol: str, mode: str = "aggressive") -> dict:
    """mode: aggressive | conservative"""
    assert mode in ("aggressive", "conservative")

    df5 = load_gold_5m(symbol)
    df5 = df5.copy()
    df5["timestamp"] = pd.to_datetime(df5["timestamp"], utc=True).dt.floor("s")
    df15 = resample_ohlc(df5, "15min")
    df1h = resample_ohlc(df5, "1h")

    # --- Step 1: structure directions ---
    h1, l1, c1 = df1h["high"].to_numpy(float), df1h["low"].to_numpy(float), df1h["close"].to_numpy(float)
    ph1, pl1 = pivot_confirm(h1, l1, swing_len_1h, swing_len_1h)
    dir_1h, _, _ = structure_direction(c1, ph1, pl1)

    h15 = df15["high"].to_numpy(float)
    l15 = df15["low"].to_numpy(float)
    c15 = df15["close"].to_numpy(float)
    ph15, pl15 = pivot_confirm(h15, l15, swing_len_15m, swing_len_15m)
    dir_15, _, _ = structure_direction(c15, ph15, pl15)

    # map closed HTF to 5m (shift 1)
    s1 = pd.DataFrame(
        {
            "timestamp": df1h["timestamp"],
            "dir_1h": dir_1h,
        }
    )
    s1["dir_1h"] = s1["dir_1h"].shift(1)
    s15 = pd.DataFrame(
        {
            "timestamp": df15["timestamp"],
            "dir_15": dir_15,
            "ph": pd.Series(ph15).shift(0),  # confirmed on bar
            "pl": pd.Series(pl15),
        }
    )
    s15["dir_15"] = s15["dir_15"].shift(1)

    left = df5[["timestamp"]].copy()
    m1 = pd.merge_asof(left.sort_values("timestamp"), s1.sort_values("timestamp"), on="timestamp", direction="backward")
    m15 = pd.merge_asof(
        left.sort_values("timestamp"), s15.sort_values("timestamp"), on="timestamp", direction="backward"
    )

    # POI creation events on 15m → map created time
    poi_events = detect_pois_15m(df15)
    # liquidity swings on 15m: running list of confirmed swing highs/lows
    # rebuild chronologically in 5m loop

    # 5m swings for sweep / conservative CISD
    h5 = df5["high"].to_numpy(float)
    l5 = df5["low"].to_numpy(float)
    c5 = df5["close"].to_numpy(float)
    o5 = df5["open"].to_numpy(float)
    ts5 = pd.to_datetime(df5["timestamp"], utc=True)
    ph5, pl5 = pivot_confirm(h5, l5, swing_len_5m, swing_len_5m)

    n = len(df5)
    d1 = m1["dir_1h"].to_numpy()
    d15 = m15["dir_15"].to_numpy()

    # Precompute 15m swing series mapped to 5m (most recent confirmed)
    swing_hi_15 = np.full(n, np.nan)
    swing_lo_15 = np.full(n, np.nan)
    # from merge - rebuild from 15m pivots as-of
    sh_map = pd.merge_asof(
        left.sort_values("timestamp"),
        pd.DataFrame({"timestamp": df15["timestamp"], "ph": ph15, "pl": pl15}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    # running last swing
    last_ph = np.nan
    last_pl = np.nan
    for i in range(n):
        # approximate: walk 15m confirmed pivots via index alignment
        pass

    # Better: build running last swing on 15m then asof
    last_ph_arr = np.full(len(df15), np.nan)
    last_pl_arr = np.full(len(df15), np.nan)
    cur_ph = np.nan
    cur_pl = np.nan
    for i in range(len(df15)):
        if np.isfinite(ph15[i]):
            cur_ph = ph15[i]
        if np.isfinite(pl15[i]):
            cur_pl = pl15[i]
        last_ph_arr[i] = cur_ph
        last_pl_arr[i] = cur_pl
    sw = pd.merge_asof(
        left.sort_values("timestamp"),
        pd.DataFrame(
            {
                "timestamp": df15["timestamp"],
                "last_ph": last_ph_arr,
                "last_pl": last_pl_arr,
            }
        ).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    swing_hi_15 = sw["last_ph"].to_numpy(float)
    swing_lo_15 = sw["last_pl"].to_numpy(float)

    # 5m running swings
    last_sh5 = np.full(n, np.nan)
    last_sl5 = np.full(n, np.nan)
    cur_sh = np.nan
    cur_sl = np.nan
    for i in range(n):
        if np.isfinite(ph5[i]):
            cur_sh = ph5[i]
        if np.isfinite(pl5[i]):
            cur_sl = pl5[i]
        last_sh5[i] = cur_sh
        last_sl5[i] = cur_sl

    # Active POIs managed by 15m time
    active: list[POI] = []
    event_i = 0
    # Map 5m bar → current 15m bar index
    t15 = df15["timestamp"].to_numpy()
    idx15 = np.searchsorted(t15, ts5.to_numpy(), side="right") - 1

    trades = []
    capital = INITIAL_CAPITAL
    equity = [INITIAL_CAPITAL]
    equity_ts = [ts5.iloc[0]]
    in_pos = False
    pos = None

    # Conservative state machine
    # after sweep: wait_ms / wait_retest
    cons = None  # dict

    # Liquidity levels: recent 15m swing lows/highs history
    liq_lows: list[float] = []
    liq_highs: list[float] = []
    seen_pl = set()
    seen_ph = set()

    for i in range(1, n):
        # sync POI creations up to current 15m bar
        i15 = int(idx15[i])
        if i15 < 0:
            continue
        while event_i < len(poi_events) and poi_events[event_i][0] <= i15:
            active.append(poi_events[event_i][1])
            event_i += 1
        # expire old / mitigated POIs
        still = []
        for p in active:
            if i15 - p.created_i > max_poi_age_bars_15m:
                continue
            # mitigate: close through opposite side
            if p.side == "bull" and c5[i] < p.z_low:
                continue
            if p.side == "bear" and c5[i] > p.z_high:
                continue
            still.append(p)
        active = still[-12:]  # keep recent

        # update liquidity from 15m pivots
        if i15 >= 1:
            if np.isfinite(pl15[i15]) and (i15, pl15[i15]) not in seen_pl:
                liq_lows.append(float(pl15[i15]))
                seen_pl.add((i15, pl15[i15]))
                liq_lows = liq_lows[-20:]
            if np.isfinite(ph15[i15]) and (i15, ph15[i15]) not in seen_ph:
                liq_highs.append(float(ph15[i15]))
                seen_ph.add((i15, ph15[i15]))
                liq_highs = liq_highs[-20:]

        # manage open trade
        if in_pos:
            hit = None
            exit_px = None
            if pos["dir"] == "long":
                if l5[i] <= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif h5[i] >= pos["tp"]:
                    hit, exit_px = "TP", pos["tp"]
            else:
                if h5[i] >= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif l5[i] <= pos["tp"]:
                    hit, exit_px = "TP", pos["tp"]
            if hit is not None:
                pnl = (
                    (exit_px - pos["entry"]) * pos["qty"]
                    if pos["dir"] == "long"
                    else (pos["entry"] - exit_px) * pos["qty"]
                )
                capital += pnl
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "direction": pos["dir"],
                        "entry_price": pos["entry"],
                        "exit_time": ts5.iloc[i],
                        "exit_price": float(exit_px),
                        "sl_price": pos["sl"],
                        "tp_price": pos["tp"],
                        "pnl": float(pnl),
                        "exit_hit": hit,
                        "mode": mode,
                        "poi_kind": pos.get("poi_kind"),
                    }
                )
                equity.append(capital)
                equity_ts.append(ts5.iloc[i])
                in_pos = False
                pos = None
                cons = None

        if in_pos:
            continue

        bias = d1[i] if isinstance(d1[i], str) else None
        bias15 = d15[i] if isinstance(d15[i], str) else None
        # Step 1: alignment required
        if bias is None or bias15 is None or bias != bias15:
            cons = None
            continue

        # Step 3: price must be interacting with a POI in bias direction
        # (conservative wait_ms / wait_retest can continue outside original POI)
        pois = [p for p in active if (p.side == "bull" and bias == "bull") or (p.side == "bear" and bias == "bear")]
        in_poi = None
        for p in reversed(pois):
            if l5[i] <= p.z_high + poi_touch_pad and h5[i] >= p.z_low - poi_touch_pad:
                in_poi = p
                break

        if in_poi is None and cons is None:
            continue

        # Step 4: liquidity sweep (only when touching a POI)
        swept = False
        sweep_ext = None
        if in_poi is not None:
            if bias == "bull" and liq_lows:
                lvl = liq_lows[-1]
                if np.isfinite(last_sl5[i - 1]):
                    lvl = min(lvl, float(last_sl5[i - 1]))
                if l5[i] < lvl and c5[i] > lvl:
                    swept = True
                    sweep_ext = float(l5[i])
            elif bias == "bear" and liq_highs:
                lvl = liq_highs[-1]
                if np.isfinite(last_sh5[i - 1]):
                    lvl = max(lvl, float(last_sh5[i - 1]))
                if h5[i] > lvl and c5[i] < lvl:
                    swept = True
                    sweep_ext = float(h5[i])

        if mode == "aggressive":
            if not swept or in_poi is None:
                continue
            if i + 1 >= n:
                continue
            entry = float(o5[i + 1])
            if bias == "bull":
                sl = sweep_ext - sl_buffer
                highs = [float(x) for x in liq_highs]
                if np.isfinite(swing_hi_15[i]):
                    highs.append(float(swing_hi_15[i]))
                tp = nearest_swing_above(entry, highs, sl, min_rr)
                if tp is None or not (sl < entry < tp):
                    continue
                side = "long"
            else:
                sl = sweep_ext + sl_buffer
                lows = [float(x) for x in liq_lows]
                if np.isfinite(swing_lo_15[i]):
                    lows.append(float(swing_lo_15[i]))
                tp = nearest_swing_below(entry, lows, sl, min_rr)
                if tp is None or not (tp < entry < sl):
                    continue
                side = "short"
            qty = FIXED_NOTIONAL / entry
            pos = {
                "dir": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": ts5.iloc[i + 1],
                "poi_kind": in_poi.kind,
            }
            in_pos = True
            continue

        # conservative
        if swept and in_poi is not None:
            cons = {
                "stage": "wait_ms",
                "bias": bias,
                "sweep_ext": sweep_ext,
                "poi": in_poi,
                "ms_level": float(last_sh5[i])
                if bias == "bull" and np.isfinite(last_sh5[i])
                else (float(last_sl5[i]) if bias == "bear" and np.isfinite(last_sl5[i]) else None),
                "zone_low": None,
                "zone_high": None,
            }
            continue

        if cons is None or cons.get("bias") != bias:
            continue

        if cons["stage"] == "wait_ms":
            lvl = cons.get("ms_level")
            if lvl is None:
                if bias == "bull" and np.isfinite(last_sh5[i]):
                    cons["ms_level"] = float(last_sh5[i])
                elif bias == "bear" and np.isfinite(last_sl5[i]):
                    cons["ms_level"] = float(last_sl5[i])
                continue
            shifted = (bias == "bull" and c5[i] > lvl) or (bias == "bear" and c5[i] < lvl)
            if shifted:
                cons["stage"] = "wait_retest"
                cons["zone_low"] = float(l5[i])
                cons["zone_high"] = float(h5[i])
            continue

        if cons["stage"] == "wait_retest":
            zl, zh = cons["zone_low"], cons["zone_high"]
            touched = l5[i] <= zh and h5[i] >= zl
            if not touched:
                continue
            if i + 1 >= n:
                continue
            entry = float(o5[i + 1])
            if bias == "bull":
                sl = min(cons["sweep_ext"], zl) - sl_buffer
                highs = [float(x) for x in liq_highs]
                if np.isfinite(swing_hi_15[i]):
                    highs.append(float(swing_hi_15[i]))
                tp = nearest_swing_above(entry, highs, sl, min_rr)
                if tp is None or not (sl < entry < tp):
                    cons = None
                    continue
                side = "long"
            else:
                sl = max(cons["sweep_ext"], zh) + sl_buffer
                lows = [float(x) for x in liq_lows]
                if np.isfinite(swing_lo_15[i]):
                    lows.append(float(swing_lo_15[i]))
                tp = nearest_swing_below(entry, lows, sl, min_rr)
                if tp is None or not (tp < entry < sl):
                    cons = None
                    continue
                side = "short"
            qty = FIXED_NOTIONAL / entry
            pos = {
                "dir": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": ts5.iloc[i + 1],
                "poi_kind": cons["poi"].kind,
            }
            in_pos = True
            cons = None
            continue

    # EOD
    if in_pos and pos is not None:
        exit_px = float(c5[-1])
        pnl = (
            (exit_px - pos["entry"]) * pos["qty"]
            if pos["dir"] == "long"
            else (pos["entry"] - exit_px) * pos["qty"]
        )
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "direction": pos["dir"],
                "entry_price": pos["entry"],
                "exit_time": ts5.iloc[-1],
                "exit_price": exit_px,
                "sl_price": pos["sl"],
                "tp_price": pos["tp"],
                "pnl": float(pnl),
                "exit_hit": "EOD",
                "mode": mode,
                "poi_kind": pos.get("poi_kind"),
            }
        )
        equity.append(capital)
        equity_ts.append(ts5.iloc[-1])

    tdf = pd.DataFrame(trades)
    # de-dup accidental same-bar double entries
    if not tdf.empty:
        tdf = tdf.drop_duplicates(subset=["entry_time", "direction"], keep="first")
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {"symbol": symbol, "mode": mode, "trades": tdf, "equity": eq, "final": capital}


def summarize(label: str, tdf: pd.DataFrame, eq: pd.Series, final: float) -> str:
    lines = ["=" * 90, f"RESULTS — {label}", "=" * 90]
    if tdf.empty:
        lines.append("No trades.")
        return "\n".join(lines)
    wins = tdf[tdf.pnl > 1e-8]
    losses = tdf[tdf.pnl < -1e-8]
    n = len(tdf)
    wr = len(wins) / n * 100
    gp = wins.pnl.sum() if len(wins) else 0.0
    gl = abs(losses.pnl.sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    avg_w = wins.pnl.mean() if len(wins) else 0.0
    avg_l = losses.pnl.mean() if len(losses) else 0.0
    wl = abs(avg_w / avg_l) if avg_l else float("inf")
    net = final - INITIAL_CAPITAL
    net_pct = net / INITIAL_CAPITAL * 100
    dd = eq - eq.cummax()
    max_dd = float(dd.min())
    lines += [
        f"1) Trades={n}  Wins={len(wins)}  Losses={len(losses)}  WR={wr:.1f}%",
        f"2) PF={pf:.3f}  GP=${gp:.2f}  GL=${gl:.2f}",
        f"3) AvgW=${avg_w:.2f}  AvgL=${avg_l:.2f}  W/L={wl:.3f}",
        f"4) Net=${net:.2f} ({net_pct:+.2f}%)  Final=${final:.2f}",
        f"5) MaxDD=${max_dd:.2f} ({max_dd/INITIAL_CAPITAL*100:.2f}%)",
        "",
        "6) Month-by-month:",
    ]
    t = tdf.copy()
    t["month"] = pd.to_datetime(t["exit_time"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
    monthly = t.groupby("month").agg(trades=("pnl", "count"), pnl=("pnl", "sum"))
    lines.append(f"{'Month':10s} {'Trades':>7s} {'PnL$':>12s}")
    for m, r in monthly.iterrows():
        lines.append(f"{m:10s} {int(r.trades):7d} {r.pnl:12.2f}")
    lines.append("")
    lines.append("7) Long vs Short:")
    for side in ("long", "short"):
        s = tdf[tdf.direction == side]
        if s.empty:
            lines.append(f"   {side}: 0")
            continue
        sw = (s.pnl > 0).sum()
        lines.append(f"   {side}: n={len(s)} WR={sw/len(s)*100:.1f}% PnL=${s.pnl.sum():.2f}")
    if "poi_kind" in tdf.columns:
        lines.append("")
        lines.append("POI kind mix: " + str(tdf["poi_kind"].value_counts().to_dict()))
    return "\n".join(lines)


def plot_eq(label: str, eq: pd.Series, path: Path, net_pct: float):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(eq.index, eq.values, color="#1f6feb", lw=1.3)
    ax.axhline(INITIAL_CAPITAL, color="#888", ls="--", lw=0.8)
    ax.set_title(f"{label} ({net_pct:+.2f}%)")
    ax.set_ylabel("Equity $")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    print("Gold 5-Step SMC Scalping backtest")
    print(
        f"params: swing1h={swing_len_1h} swing15={swing_len_15m} swing5={swing_len_5m} "
        f"impulseATR={impulse_atr_mult} sl_buf=${sl_buffer} minRR={min_rr} notional=${FIXED_NOTIONAL}"
    )
    rows = []
    texts = []
    for sym in ("PAXGUSDT", "XAUUSD"):
        for mode in ("aggressive", "conservative"):
            print(f"\n>>> {sym} [{mode}]")
            try:
                out = run_backtest(sym, mode)
            except FileNotFoundError as e:
                print("SKIP", e)
                continue
            tdf, eq, final = out["trades"], out["equity"], out["final"]
            label = f"{sym} {mode}"
            text = summarize(label, tdf, eq, final)
            print(text)
            texts.append(text)
            net = final - INITIAL_CAPITAL
            net_pct = net / INITIAL_CAPITAL * 100
            stem = f"{sym.lower()}_{mode}"
            if not tdf.empty:
                tdf.to_csv(OUT / f"{stem}_trades.csv", index=False)
            eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
            plot_eq(label, eq, OUT / f"{stem}_equity.png", net_pct)
            rows.append(
                {
                    "symbol": sym,
                    "mode": mode,
                    "trades": len(tdf),
                    "wins": int((tdf.pnl > 0).sum()) if len(tdf) else 0,
                    "losses": int((tdf.pnl < 0).sum()) if len(tdf) else 0,
                    "wr": float((tdf.pnl > 0).mean() * 100) if len(tdf) else 0.0,
                    "pf": (
                        float(tdf.loc[tdf.pnl > 0, "pnl"].sum() / abs(tdf.loc[tdf.pnl < 0, "pnl"].sum()))
                        if len(tdf) and (tdf.pnl < 0).any() and (tdf.pnl > 0).any()
                        else float("nan")
                    ),
                    "net": net,
                    "net_pct": net_pct,
                    "max_dd_pct": float((eq - eq.cummax()).min() / INITIAL_CAPITAL * 100),
                }
            )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "summary.csv", index=False)
    (OUT / "report.txt").write_text("\n\n".join(texts) + "\n")
    (OUT / "params.json").write_text(
        json.dumps(
            {
                "swing_len_1h": swing_len_1h,
                "swing_len_15m": swing_len_15m,
                "swing_len_5m": swing_len_5m,
                "impulse_atr_mult": impulse_atr_mult,
                "sl_buffer": sl_buffer,
                "initial_capital": INITIAL_CAPITAL,
                "fixed_notional": FIXED_NOTIONAL,
                "days": DAYS,
                "notes": "Algorithmic encoding of 5-step gold SMC scalping video rules",
            },
            indent=2,
        )
    )
    print("\nSUMMARY")
    print(rdf.to_string(index=False))
    print(f"Artifacts → {OUT}")


if __name__ == "__main__":
    main()
