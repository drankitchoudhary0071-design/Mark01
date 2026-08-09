#!/usr/bin/env python3
"""
Gold Scalping — 5-Step Strategy (video-faithful algorithmic port)

Video rules (Hindi transcript mapping):
  Step 1 — 1H & 15M same direction via Market Shift / BOS (HH/HL vs LH/LL)
  Step 2 — Mark 15M Demand (bull) / Supply (bear) POI = Order Block or FVG
           + mark liquidity under swing/internal lows (retail stops)
  Step 3 — Do NOTHING until price reaches that 15M POI (no mid-air entries)
  Step 4 — At POI wait for Liquidity Sweep (not blind entry)
           Aggressive: enter on sweep; SL just below sweep candle low
           Conservative: sweep → LTF Market Shift (break internal high) →
                         enter on retest of that new demand zone
  Step 5 — Scalp TP = nearest logical 15M internal/swing high (long) /
           nearest swing low (short). No 1:10 greed.

LTF for entry model = 5m. Gold: PAXGUSDT + XAUUSD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

# ── video-aligned parameters ───────────────────────────────────────────
# Structure swings (BOS / market shift on 1H & 15M)
struct_len_1h = 5
struct_len_15m = 5
# Internal swings (liquidity + LTF market shift) — tighter
internal_len_15m = 2
internal_len_5m = 2
# Logical TP swings on 15M ("nearest logical swing/internal high")
tp_swing_len_15m = 3

impulse_atr_mult = 1.5  # strong displace to define OB origin
atr_len = 14
sl_buffer = 0.3  # just below/above sweep candle (gold $)
max_poi_age_15m = 64  # ~16h of 15m — fresh POIs only
poi_max_active = 3  # only most recent POIs in bias (video: mark clear zones)

INITIAL_CAPITAL = 10_000.0
FIXED_NOTIONAL = 3_000.0
DAYS = 365

OUT = Path(__file__).resolve().parent / "results" / "gold_smc_5step_v2"
OUT.mkdir(parents=True, exist_ok=True)
FLAT = Path(__file__).resolve().parent / "results"


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
    a = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - a) + tr[i] * a
    return atr


def pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        hv = high[c]
        ok_h = True
        for j in range(c - left, c + right + 1):
            if j != c and high[j] >= hv:
                ok_h = False
                break
        if ok_h:
            ph[i] = hv
        lv = low[c]
        ok_l = True
        for j in range(c - left, c + right + 1):
            if j != c and low[j] <= lv:
                ok_l = False
                break
        if ok_l:
            pl[i] = lv
    return ph, pl


def structure_bias(close: np.ndarray, ph: np.ndarray, pl: np.ndarray) -> np.ndarray:
    """
    Step 1 market shift:
      bull when close breaks last swing high (BOS up / HH)
      bear when close breaks last swing low  (BOS down / LL)
    """
    n = len(close)
    bias = np.array([None] * n, dtype=object)
    last_sh = np.nan
    last_sl = np.nan
    cur = None
    for i in range(n):
        if np.isfinite(ph[i]):
            last_sh = ph[i]
        if np.isfinite(pl[i]):
            last_sl = pl[i]
        if np.isfinite(last_sl) and close[i] < last_sl:
            cur = "bear"
        if np.isfinite(last_sh) and close[i] > last_sh:
            cur = "bull"
        bias[i] = cur
    return bias


def running_last(pivot: np.ndarray) -> np.ndarray:
    out = np.full(len(pivot), np.nan)
    last = np.nan
    for i, v in enumerate(pivot):
        if np.isfinite(v):
            last = v
        out[i] = last
    return out


def collect_swing_history(pivot: np.ndarray, maxlen: int = 30) -> list[list[float]]:
    """Per-bar snapshot of recent confirmed swing prices."""
    hist: list[float] = []
    snaps: list[list[float]] = []
    for i, v in enumerate(pivot):
        if np.isfinite(v):
            hist.append(float(v))
            if len(hist) > maxlen:
                hist = hist[-maxlen:]
        snaps.append(list(hist))
    return snaps


@dataclass
class POI:
    side: str  # bull | bear
    z_low: float
    z_high: float
    created: int
    kind: str  # ob | fvg
    mitigated: bool = False


def build_poi_events(o, h, l, c, atr) -> list[tuple[int, POI]]:
    """
    Step 2 POI:
      Demand OB  = last bearish candle before bullish impulse
      Supply OB  = last bullish candle before bearish impulse
      FVG        = 3-candle gap with impulse middle candle
    """
    n = len(c)
    events: list[tuple[int, POI]] = []
    for i in range(2, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        body_mid = abs(c[i - 1] - o[i - 1])
        body_i = abs(c[i] - o[i])

        # --- FVG ---
        if l[i] > h[i - 2] and c[i - 1] > o[i - 1] and body_mid >= impulse_atr_mult * atr[i]:
            events.append((i, POI("bull", float(h[i - 2]), float(l[i]), i, "fvg")))
        if h[i] < l[i - 2] and c[i - 1] < o[i - 1] and body_mid >= impulse_atr_mult * atr[i]:
            events.append((i, POI("bear", float(h[i]), float(l[i - 2]), i, "fvg")))

        # --- Order blocks from impulse bar i ---
        if c[i] > o[i] and body_i >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, max(-1, i - 10), -1):
                if c[j] < o[j]:
                    events.append(
                        (i, POI("bull", float(l[j]), float(max(o[j], c[j])), i, "ob"))
                    )
                    break
        if c[i] < o[i] and body_i >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, max(-1, i - 10), -1):
                if c[j] > o[j]:
                    events.append(
                        (i, POI("bear", float(min(o[j], c[j])), float(h[j]), i, "ob"))
                    )
                    break
    return events


def nearest_above(px: float, levels: list[float]) -> float | None:
    cands = [x for x in levels if x > px * 1.00005]  # strictly above
    return min(cands) if cands else None


def nearest_below(px: float, levels: list[float]) -> float | None:
    cands = [x for x in levels if x < px * 0.99995]
    return max(cands) if cands else None


def load_gold_5m(symbol: str) -> pd.DataFrame:
    if symbol.upper() == "XAUUSD":
        path = Path(__file__).resolve().parent / "cache" / "XAUUSD_5m_365d.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        print(f"Loaded {path} ({len(df)} bars)")
        return df
    return load_or_fetch(symbol, "5m", DAYS)


def asof_map(left_ts: pd.Series, right: pd.DataFrame) -> pd.DataFrame:
    L = pd.DataFrame({"timestamp": pd.to_datetime(left_ts, utc=True).dt.floor("s")})
    R = right.copy()
    R["timestamp"] = pd.to_datetime(R["timestamp"], utc=True).dt.floor("s")
    return pd.merge_asof(L.sort_values("timestamp"), R.sort_values("timestamp"), on="timestamp", direction="backward")


def run_backtest(symbol: str, mode: str) -> dict:
    assert mode in ("aggressive", "conservative")
    df5 = load_gold_5m(symbol).copy()
    df5["timestamp"] = pd.to_datetime(df5["timestamp"], utc=True).dt.floor("s")
    df15 = resample_ohlc(df5, "15min")
    df1h = resample_ohlc(df5, "1h")

    # ===== Step 1: 1H & 15M alignment =====
    h1, l1, c1 = df1h.high.to_numpy(float), df1h.low.to_numpy(float), df1h.close.to_numpy(float)
    ph1, pl1 = pivots(h1, l1, struct_len_1h, struct_len_1h)
    bias_1h = structure_bias(c1, ph1, pl1)

    h15 = df15.high.to_numpy(float)
    l15 = df15.low.to_numpy(float)
    o15 = df15.open.to_numpy(float)
    c15 = df15.close.to_numpy(float)
    ph15_s, pl15_s = pivots(h15, l15, struct_len_15m, struct_len_15m)
    bias_15 = structure_bias(c15, ph15_s, pl15_s)

    # Internal liquidity swings + TP swings on 15M
    ph15_i, pl15_i = pivots(h15, l15, internal_len_15m, internal_len_15m)
    ph15_tp, pl15_tp = pivots(h15, l15, tp_swing_len_15m, tp_swing_len_15m)
    atr15 = atr_wilder(h15, l15, c15, atr_len)
    poi_events = build_poi_events(o15, h15, l15, c15, atr15)

    liq_low_hist = collect_swing_history(pl15_i)
    liq_high_hist = collect_swing_history(ph15_i)
    tp_high_hist = collect_swing_history(ph15_tp)
    tp_low_hist = collect_swing_history(pl15_tp)

    # Map closed HTF to 5m (non-repaint: shift 1)
    m1 = asof_map(
        df5["timestamp"],
        pd.DataFrame({"timestamp": df1h["timestamp"], "bias_1h": pd.Series(bias_1h).shift(1)}),
    )
    m15 = asof_map(
        df5["timestamp"],
        pd.DataFrame(
            {
                "timestamp": df15["timestamp"],
                "bias_15": pd.Series(bias_15).shift(1),
                "i15": np.arange(len(df15)),
            }
        ),
    )

    # 5m internals for sweep refine + conservative MS
    h5 = df5.high.to_numpy(float)
    l5 = df5.low.to_numpy(float)
    o5 = df5.open.to_numpy(float)
    c5 = df5.close.to_numpy(float)
    ts5 = pd.to_datetime(df5["timestamp"], utc=True)
    ph5, pl5 = pivots(h5, l5, internal_len_5m, internal_len_5m)
    last_sh5 = running_last(ph5)
    last_sl5 = running_last(pl5)

    n = len(df5)
    b1 = m1["bias_1h"].to_numpy()
    b15 = m15["bias_15"].to_numpy()
    i15_of = m15["i15"].fillna(-1).to_numpy(int)

    active: list[POI] = []
    ev = 0
    trades = []
    capital = INITIAL_CAPITAL
    equity = [INITIAL_CAPITAL]
    equity_ts = [ts5.iloc[0]]
    in_pos = False
    pos = None
    cons = None  # conservative state

    for i in range(2, n - 1):
        # ---- manage open position (Step 5 exits) ----
        if in_pos:
            hit = exit_px = None
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
            if hit:
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
                        "poi_kind": pos["poi_kind"],
                        "step4": pos["step4"],
                    }
                )
                equity.append(capital)
                equity_ts.append(ts5.iloc[i])
                in_pos = False
                pos = None
                cons = None
            else:
                continue

        # ===== Step 1 gate =====
        bias = b1[i] if isinstance(b1[i], str) else None
        bias_m = b15[i] if isinstance(b15[i], str) else None
        if bias is None or bias_m is None or bias != bias_m:
            cons = None
            continue

        i15 = int(i15_of[i])
        if i15 < 5:
            continue

        # sync POI creations (Step 2)
        while ev < len(poi_events) and poi_events[ev][0] <= i15:
            active.append(poi_events[ev][1])
            ev += 1

        # mitigate / age out
        kept: list[POI] = []
        for p in active:
            if i15 - p.created > max_poi_age_15m:
                continue
            if p.side == "bull" and c5[i] < p.z_low:
                continue  # fully mitigated
            if p.side == "bear" and c5[i] > p.z_high:
                continue
            kept.append(p)
        active = kept
        pois = [p for p in active if p.side == ("bull" if bias == "bull" else "bear")]
        pois = pois[-poi_max_active:]

        # ===== Step 3: must be AT / overlapping POI =====
        in_poi = None
        for p in reversed(pois):
            if l5[i] <= p.z_high and h5[i] >= p.z_low:
                in_poi = p
                break

        # Liquidity levels (Step 2 marks) at current 15m context
        liq_lows = liq_low_hist[i15] if i15 < len(liq_low_hist) else []
        liq_highs = liq_high_hist[i15] if i15 < len(liq_high_hist) else []
        tp_highs = tp_high_hist[i15] if i15 < len(tp_high_hist) else []
        tp_lows = tp_low_hist[i15] if i15 < len(tp_low_hist) else []
        # also use latest 5m internal as nested liquidity
        if np.isfinite(last_sl5[i - 1]):
            liq_lows = liq_lows + [float(last_sl5[i - 1])]
        if np.isfinite(last_sh5[i - 1]):
            liq_highs = liq_highs + [float(last_sh5[i - 1])]

        # ===== Step 4: liquidity sweep inside POI =====
        swept = False
        sweep_low = sweep_high = None
        if in_poi is not None:
            if bias == "bull" and liq_lows:
                # sweep = take out most recent liquidity low then reclaim
                lvl = min(liq_lows[-3:])  # recent pool
                if l5[i] < lvl and c5[i] > lvl:
                    swept = True
                    sweep_low = float(l5[i])
            elif bias == "bear" and liq_highs:
                lvl = max(liq_highs[-3:])
                if h5[i] > lvl and c5[i] < lvl:
                    swept = True
                    sweep_high = float(h5[i])

        # --- Aggressive (video 11:40): enter on sweep; SL under sweep candle ---
        if mode == "aggressive":
            if not (swept and in_poi is not None):
                continue
            entry = float(o5[i + 1])  # next 5m open after sweep candle closes
            if bias == "bull":
                sl = sweep_low - sl_buffer
                tp = nearest_above(entry, tp_highs + liq_highs)
                if tp is None or not (sl < entry < tp):
                    continue
                # scalp: reject absurd SL wider than 1.5× distance to TP (invalid geometry)
                if (entry - sl) > 2.5 * (tp - entry):
                    continue
                side = "long"
            else:
                sl = sweep_high + sl_buffer
                tp = nearest_below(entry, tp_lows + liq_lows)
                if tp is None or not (tp < entry < sl):
                    continue
                if (sl - entry) > 2.5 * (entry - tp):
                    continue
                side = "short"
            pos = {
                "dir": side,
                "entry": entry,
                "sl": sl,
                "tp": float(tp),
                "qty": FIXED_NOTIONAL / entry,
                "entry_time": ts5.iloc[i + 1],
                "poi_kind": in_poi.kind,
                "step4": "aggressive_sweep",
            }
            in_pos = True
            continue

        # --- Conservative (video 13:55–14:56) ---
        # After sweep: wait LTF market shift (break previous internal high),
        # then enter when price retests the new demand from that shift.
        if swept and in_poi is not None:
            # lock internal high/low that must break for MS
            if bias == "bull":
                ms_lvl = float(last_sh5[i]) if np.isfinite(last_sh5[i]) else float(h5[i])
            else:
                ms_lvl = float(last_sl5[i]) if np.isfinite(last_sl5[i]) else float(l5[i])
            cons = {
                "bias": bias,
                "stage": "wait_ms",
                "sweep_ext": sweep_low if bias == "bull" else sweep_high,
                "ms_lvl": ms_lvl,
                "poi_kind": in_poi.kind,
                "zone_low": None,
                "zone_high": None,
            }
            continue

        if cons is None or cons["bias"] != bias:
            continue

        if cons["stage"] == "wait_ms":
            ms_lvl = cons["ms_lvl"]
            # refresh level if a newer internal forms before break
            if bias == "bull" and np.isfinite(last_sh5[i]) and last_sh5[i] < ms_lvl:
                cons["ms_lvl"] = float(last_sh5[i])
                ms_lvl = cons["ms_lvl"]
            if bias == "bear" and np.isfinite(last_sl5[i]) and last_sl5[i] > ms_lvl:
                cons["ms_lvl"] = float(last_sl5[i])
                ms_lvl = cons["ms_lvl"]

            shifted = (bias == "bull" and c5[i] > ms_lvl) or (bias == "bear" and c5[i] < ms_lvl)
            if shifted:
                # new demand/supply = displacement candle that made the shift
                cons["stage"] = "wait_retest"
                cons["zone_low"] = float(l5[i])
                cons["zone_high"] = float(h5[i])
            continue

        if cons["stage"] == "wait_retest":
            zl, zh = cons["zone_low"], cons["zone_high"]
            # price returns to touch new demand/supply
            if not (l5[i] <= zh and h5[i] >= zl):
                continue
            entry = float(o5[i + 1])
            if bias == "bull":
                sl = float(cons["sweep_ext"]) - sl_buffer
                tp = nearest_above(entry, tp_highs + liq_highs)
                if tp is None or not (sl < entry < tp):
                    cons = None
                    continue
                if (entry - sl) > 2.5 * (tp - entry):
                    cons = None
                    continue
                side = "long"
            else:
                sl = float(cons["sweep_ext"]) + sl_buffer
                tp = nearest_below(entry, tp_lows + liq_lows)
                if tp is None or not (tp < entry < sl):
                    cons = None
                    continue
                if (sl - entry) > 2.5 * (entry - tp):
                    cons = None
                    continue
                side = "short"
            pos = {
                "dir": side,
                "entry": entry,
                "sl": sl,
                "tp": float(tp),
                "qty": FIXED_NOTIONAL / entry,
                "entry_time": ts5.iloc[i + 1],
                "poi_kind": cons["poi_kind"],
                "step4": "conservative_ms_retest",
            }
            in_pos = True
            cons = None

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
                "poi_kind": pos["poi_kind"],
                "step4": pos["step4"],
            }
        )
        equity.append(capital)
        equity_ts.append(ts5.iloc[-1])

    tdf = pd.DataFrame(trades)
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
    gp = float(wins.pnl.sum()) if len(wins) else 0.0
    gl = float(abs(losses.pnl.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    avg_w = float(wins.pnl.mean()) if len(wins) else 0.0
    avg_l = float(losses.pnl.mean()) if len(losses) else 0.0
    net = final - INITIAL_CAPITAL
    dd = eq - eq.cummax()
    lines += [
        f"1) Trades={n}  Wins={len(wins)}  Losses={len(losses)}  WR={wr:.1f}%",
        f"2) PF={pf:.3f}  GP=${gp:.2f}  GL=${gl:.2f}",
        f"3) AvgW=${avg_w:.2f}  AvgL=${avg_l:.2f}  W/L={(abs(avg_w/avg_l) if avg_l else float('inf')):.3f}",
        f"4) Net=${net:.2f} ({net/INITIAL_CAPITAL*100:+.2f}%)  Final=${final:.2f}",
        f"5) MaxDD=${float(dd.min()):.2f} ({float(dd.min())/INITIAL_CAPITAL*100:.2f}%)",
        "",
        "6) Month-by-month:",
        f"{'Month':10s} {'Trades':>7s} {'PnL$':>12s}",
    ]
    t = tdf.copy()
    t["month"] = pd.to_datetime(t.exit_time, utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
    for m, r in t.groupby("month").agg(trades=("pnl", "count"), pnl=("pnl", "sum")).iterrows():
        lines.append(f"{m:10s} {int(r.trades):7d} {r.pnl:12.2f}")
    lines.append("")
    lines.append("7) Long vs Short:")
    for side in ("long", "short"):
        s = tdf[tdf.direction == side]
        if s.empty:
            lines.append(f"   {side}: 0")
        else:
            lines.append(
                f"   {side}: n={len(s)} WR={(s.pnl>0).mean()*100:.1f}% PnL=${s.pnl.sum():.2f}"
            )
    if "poi_kind" in tdf.columns:
        lines.append("POI mix: " + str(tdf.poi_kind.value_counts().to_dict()))
    if "step4" in tdf.columns:
        lines.append("Entry mix: " + str(tdf.step4.value_counts().to_dict()))
    return "\n".join(lines)


def plot_eq(label, eq, path, net_pct):
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
    print("Gold 5-Step SMC — video-faithful v2")
    print(
        f"struct 1H/{struct_len_1h} 15M/{struct_len_15m} | internal 15M/{internal_len_15m} 5M/{internal_len_5m} "
        f"| TP swing/{tp_swing_len_15m} | impulseATR={impulse_atr_mult} sl_buf=${sl_buffer}"
    )
    rows = []
    texts = []
    for sym in ("PAXGUSDT", "XAUUSD"):
        for mode in ("aggressive", "conservative"):
            print(f"\n>>> {sym} [{mode}]")
            out = run_backtest(sym, mode)
            tdf, eq, final = out["trades"], out["equity"], out["final"]
            label = f"{sym} {mode} v2"
            text = summarize(label, tdf, eq, final)
            print(text)
            texts.append(text)
            net = final - INITIAL_CAPITAL
            net_pct = net / INITIAL_CAPITAL * 100
            stem = f"{sym.lower()}_{mode}"
            if not tdf.empty:
                tdf.to_csv(OUT / f"{stem}_trades.csv", index=False)
                tdf.to_csv(FLAT / f"gold_smc_v2_{stem}_trades.csv", index=False)
            eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
            plot_eq(label, eq, OUT / f"{stem}_equity.png", net_pct)
            plot_eq(label, eq, FLAT / f"gold_smc_v2_{stem}_equity.png", net_pct)
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
    rdf.to_csv(FLAT / "gold_smc_v2_summary.csv", index=False)
    (OUT / "report.txt").write_text("\n\n".join(texts) + "\n")
    (FLAT / "gold_smc_v2_report.txt").write_text("\n\n".join(texts) + "\n")
    meta = {
        "version": "v2_video_faithful",
        "struct_len_1h": struct_len_1h,
        "struct_len_15m": struct_len_15m,
        "internal_len_15m": internal_len_15m,
        "internal_len_5m": internal_len_5m,
        "tp_swing_len_15m": tp_swing_len_15m,
        "impulse_atr_mult": impulse_atr_mult,
        "sl_buffer": sl_buffer,
        "steps": [
            "1H+15M BOS alignment",
            "15M OB/FVG POI + internal liquidity",
            "wait for price in POI",
            "liquidity sweep → aggressive or conservative MS-retest",
            "TP nearest logical 15M swing; SL sweep candle extreme",
        ],
    }
    (OUT / "params.json").write_text(json.dumps(meta, indent=2))
    (FLAT / "gold_smc_v2_params.json").write_text(json.dumps(meta, indent=2))
    print("\nSUMMARY")
    print(rdf.to_string(index=False))
    print(f"Artifacts → {OUT}")


if __name__ == "__main__":
    main()
