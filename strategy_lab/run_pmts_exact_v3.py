#!/usr/bin/env python3
"""
PMTS 4H Fib Zone + 5m BOS — exact-spec backtest (pandas/numpy only).

SL/TP: swing × (1 ± sl_pct%), anchor × (1 ∓ tp_pct%).
Runs separately on BTCUSDT and PAXGUSDT 5m × 365d.
Fixed notional position sizing (see FIXED_NOTIONAL) with $10,000 capital.
"""

from __future__ import annotations

import json
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
htf_swing_len = 15
atr_len = 14
atr_mult = 1.5
zone_upper = 0.5
zone_lower = 0.7
ltf_swing_len = 6
# SL/TP as % of price (Step 8):
#   long  SL = ltf_swing_low  * (1 - sl_pct/100), TP = anchor_high * (1 - tp_pct/100)
#   short SL = ltf_swing_high * (1 + sl_pct/100), TP = anchor_low  * (1 + tp_pct/100)
sl_pct = 0.09
tp_pct = 0.14

INITIAL_CAPITAL = 10_000.0
# Fixed notional $ per trade (non-compounding). qty = FIXED_NOTIONAL / entry.
FIXED_NOTIONAL = 3_000.0
DAYS = 365
OUT = Path(__file__).resolve().parent / "results" / "pmts_exact_v3"
OUT.mkdir(parents=True, exist_ok=True)


def resample_4h(df5: pd.DataFrame) -> pd.DataFrame:
    x = df5.set_index(pd.to_datetime(df5["timestamp"], utc=True))
    out = (
        x.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    out.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return out


def atr_wilder(df: pd.DataFrame, period: int) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
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


def pivot_extremes(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Return arrays: pivot_high[i]=price when confirmed on bar i, else nan (same for low)."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        # high
        val = high[c]
        ok = True
        for j in range(c - left, c + right + 1):
            if j == c:
                continue
            if high[j] >= val:
                ok = False
                break
        if ok:
            ph[i] = val
        # low
        val = low[c]
        ok = True
        for j in range(c - left, c + right + 1):
            if j == c:
                continue
            if low[j] <= val:
                ok = False
                break
        if ok:
            pl[i] = val
    return ph, pl


def build_htf_state(htf: pd.DataFrame) -> pd.DataFrame:
    high = htf["high"].to_numpy(float)
    low = htf["low"].to_numpy(float)
    close = htf["close"].to_numpy(float)
    n = len(htf)
    ph, pl = pivot_extremes(high, low, htf_swing_len, htf_swing_len)
    atr = atr_wilder(htf, atr_len)

    last_sh = np.nan
    last_sl = np.nan
    direction = np.array([None] * n, dtype=object)
    a_hi = np.full(n, np.nan)
    a_lo = np.full(n, np.nan)
    dir_s = None

    for i in range(n):
        thr = atr[i] * atr_mult if np.isfinite(atr[i]) else np.nan
        if np.isfinite(ph[i]):
            if (not np.isfinite(last_sh)) or (
                np.isfinite(thr) and abs(ph[i] - last_sh) >= thr
            ):
                last_sh = ph[i]
        if np.isfinite(pl[i]):
            if (not np.isfinite(last_sl)) or (
                np.isfinite(thr) and abs(pl[i] - last_sl) >= thr
            ):
                last_sl = pl[i]

        # BOS: bear then bull (bull wins same bar if both)
        if np.isfinite(last_sl) and close[i] < last_sl:
            dir_s = "bear"
        if np.isfinite(last_sh) and close[i] > last_sh:
            dir_s = "bull"

        direction[i] = dir_s
        a_hi[i] = last_sh
        a_lo[i] = last_sl

    out = htf[["timestamp"]].copy()
    out["htf_dir"] = direction
    out["anchor_high"] = a_hi
    out["anchor_low"] = a_lo
    out["atr"] = atr
    return out


def map_closed_htf(ltf: pd.DataFrame, htf_state: pd.DataFrame) -> pd.DataFrame:
    """Use only closed HTF bars (shift 1) then as-of merge."""
    h = htf_state.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    for col in ("htf_dir", "anchor_high", "anchor_low"):
        h[col] = h[col].shift(1)
    left = pd.DataFrame({"timestamp": pd.to_datetime(ltf["timestamp"], utc=True)})
    return pd.merge_asof(
        left.sort_values("timestamp"),
        h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def run_backtest(symbol: str) -> dict:
    df = load_or_fetch(symbol, "5m", DAYS)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    htf = resample_4h(df)
    htf_state = build_htf_state(htf)
    mapped = map_closed_htf(df, htf_state)

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    ts = df["timestamp"].to_numpy()
    n = len(df)

    htf_dir = mapped["htf_dir"].to_numpy()
    a_hi = mapped["anchor_high"].to_numpy(float)
    a_lo = mapped["anchor_low"].to_numpy(float)

    # LTF pivots
    ph, pl = pivot_extremes(high, low, ltf_swing_len, ltf_swing_len)
    ltf_sh = np.full(n, np.nan)
    ltf_sl = np.full(n, np.nan)
    cur_sh = np.nan
    cur_sl = np.nan
    for i in range(n):
        if np.isfinite(ph[i]):
            cur_sh = ph[i]
        if np.isfinite(pl[i]):
            cur_sl = pl[i]
        ltf_sh[i] = cur_sh
        ltf_sl[i] = cur_sl

    bull_bos = np.zeros(n, dtype=bool)
    bear_bos = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.isfinite(ltf_sh[i]) and close[i - 1] <= ltf_sh[i] and close[i] > ltf_sh[i]:
            bull_bos[i] = True
        if np.isfinite(ltf_sl[i]) and close[i - 1] >= ltf_sl[i] and close[i] < ltf_sl[i]:
            bear_bos[i] = True

    # Zone geometry
    zone_top = np.full(n, np.nan)
    zone_bot = np.full(n, np.nan)
    for i in range(n):
        d = htf_dir[i]
        if not isinstance(d, str) or not np.isfinite(a_hi[i]) or not np.isfinite(a_lo[i]):
            continue
        rng = a_hi[i] - a_lo[i]
        if rng <= 0:
            continue
        if d == "bull":
            zone_top[i] = a_hi[i] - rng * zone_upper
            zone_bot[i] = a_hi[i] - rng * zone_lower
        else:
            zone_bot[i] = a_lo[i] + rng * zone_upper
            zone_top[i] = a_lo[i] + rng * zone_lower

    in_zone = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(zone_top[i]) and np.isfinite(zone_bot[i]):
            # overlap with [min(bot,top), max(bot,top)] using Pine: low<=zoneTop and high>=zoneBottom
            in_zone[i] = low[i] <= zone_top[i] and high[i] >= zone_bot[i]

    # Rising-edge arm
    armed_flag = np.zeros(n, dtype=bool)
    armed = False
    armed_dir = None
    for i in range(n):
        d = htf_dir[i] if isinstance(htf_dir[i], str) else None
        if d is not None and d != armed_dir:
            armed = False
            armed_dir = d
        # rising edge into zone
        if in_zone[i] and (i == 0 or not in_zone[i - 1]):
            armed = True
        armed_flag[i] = armed

    trades = []
    equity = [INITIAL_CAPITAL]
    equity_ts = [ts[0]]
    capital = INITIAL_CAPITAL

    in_pos = False
    pos = None
    last_trade_anchors = (None, None)  # (ahi, alo) used in last trade
    # Recompute armed live during sim (disarm on entry not required by spec for next fib)
    armed = False
    armed_dir = None

    for i in range(n):
        d = htf_dir[i] if isinstance(htf_dir[i], str) else None

        # manage open position
        if in_pos:
            hit = None
            exit_px = None
            if pos["dir"] == "long":
                # SL first if both (conservative)
                if low[i] <= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif high[i] >= pos["tp"]:
                    hit, exit_px = "TP", pos["tp"]
            else:
                if high[i] >= pos["sl"]:
                    hit, exit_px = "SL", pos["sl"]
                elif low[i] <= pos["tp"]:
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
                        "exit_time": pd.Timestamp(ts[i]),
                        "exit_price": float(exit_px),
                        "sl_price": pos["sl"],
                        "tp_price": pos["tp"],
                        "qty": pos["qty"],
                        "pnl": float(pnl),
                        "exit_hit": hit,
                        "anchor_high": pos["ahi"],
                        "anchor_low": pos["alo"],
                    }
                )
                equity.append(capital)
                equity_ts.append(ts[i])
                in_pos = False
                pos = None

        # arm state
        if d is not None and d != armed_dir:
            armed = False
            armed_dir = d
        if in_zone[i] and (i == 0 or not in_zone[i - 1]):
            armed = True

        if in_pos or not armed or d is None:
            continue
        if not np.isfinite(a_hi[i]) or not np.isfinite(a_lo[i]):
            continue

        anchors = (float(a_hi[i]), float(a_lo[i]))
        if last_trade_anchors[0] is not None:
            if (
                abs(anchors[0] - last_trade_anchors[0]) < 1e-9
                and abs(anchors[1] - last_trade_anchors[1]) < 1e-9
            ):
                continue  # same fib — no re-trade

        long_ok = d == "bull" and bull_bos[i]
        short_ok = d == "bear" and bear_bos[i]
        if not long_ok and not short_ok:
            continue

        entry = float(close[i])
        qty = FIXED_NOTIONAL / entry
        sl_mult_lo = 1.0 - sl_pct / 100.0
        sl_mult_hi = 1.0 + sl_pct / 100.0
        tp_mult_lo = 1.0 - tp_pct / 100.0
        tp_mult_hi = 1.0 + tp_pct / 100.0
        if long_ok:
            if not np.isfinite(ltf_sl[i]):
                continue
            sl = float(ltf_sl[i]) * sl_mult_lo
            tp = float(a_hi[i]) * tp_mult_lo
            if not (sl < entry < tp):
                continue
            pos = {
                "dir": "long",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": pd.Timestamp(ts[i]),
                "ahi": anchors[0],
                "alo": anchors[1],
            }
        else:
            if not np.isfinite(ltf_sh[i]):
                continue
            sl = float(ltf_sh[i]) * sl_mult_hi
            tp = float(a_lo[i]) * tp_mult_hi
            if not (tp < entry < sl):
                continue
            pos = {
                "dir": "short",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": pd.Timestamp(ts[i]),
                "ahi": anchors[0],
                "alo": anchors[1],
            }
        in_pos = True
        last_trade_anchors = anchors
        # Spec: armed resets on direction change only; keep armed True after entry
        # but same-fib filter blocks re-entry on same anchors.

    # Force-close open at end
    if in_pos and pos is not None:
        exit_px = float(close[-1])
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
                "exit_time": pd.Timestamp(ts[-1]),
                "exit_price": exit_px,
                "sl_price": pos["sl"],
                "tp_price": pos["tp"],
                "qty": pos["qty"],
                "pnl": float(pnl),
                "exit_hit": "EOD",
                "anchor_high": pos["ahi"],
                "anchor_low": pos["alo"],
            }
        )
        equity.append(capital)
        equity_ts.append(ts[-1])

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {"symbol": symbol, "trades": tdf, "equity": eq, "final": capital}


def summarize(symbol: str, tdf: pd.DataFrame, eq: pd.Series, final: float) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(f"RESULTS — {symbol}")
    lines.append("=" * 90)

    if tdf.empty:
        lines.append("No trades.")
        return "\n".join(lines)

    wins = tdf[tdf["pnl"] > 1e-8]
    losses = tdf[tdf["pnl"] < -1e-8]
    be = tdf[tdf["pnl"].abs() <= 1e-8]
    n = len(tdf)
    wr = len(wins) / n * 100 if n else 0
    gp = wins["pnl"].sum() if len(wins) else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    avg_w = wins["pnl"].mean() if len(wins) else 0.0
    avg_l = losses["pnl"].mean() if len(losses) else 0.0
    wl_ratio = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
    net = final - INITIAL_CAPITAL
    net_pct = net / INITIAL_CAPITAL * 100

    peak = eq.cummax()
    dd = eq - peak
    max_dd = float(dd.min())
    max_dd_pct = max_dd / INITIAL_CAPITAL * 100

    lines += [
        f"1) Trades={n}  Wins={len(wins)}  Losses={len(losses)}  Breakevens={len(be)}  WinRate={wr:.1f}%",
        f"2) Profit Factor={pf:.3f}  (gross_profit=${gp:.2f} / gross_loss=${gl:.2f})",
        f"3) Avg Win=${avg_w:.2f}  Avg Loss=${avg_l:.2f}  Win/Loss Ratio={wl_ratio:.3f}",
        f"4) Net Profit=${net:.2f}  ({net_pct:+.2f}%)  Final Equity=${final:.2f}",
        f"   Fixed notional=${FIXED_NOTIONAL:.0f}/trade  Initial=${INITIAL_CAPITAL:.0f}",
        f"5) Max Drawdown=${max_dd:.2f}  ({max_dd_pct:.2f}%)",
        "",
        "6) Month-by-month:",
    ]

    tdf = tdf.copy()
    tdf["month"] = pd.to_datetime(tdf["exit_time"], utc=True).dt.to_period("M")
    monthly = tdf.groupby("month").agg(trades=("pnl", "count"), pnl=("pnl", "sum"))
    lines.append(f"{'Month':10s} {'Trades':>7s} {'PnL$':>12s}")
    for m, r in monthly.iterrows():
        lines.append(f"{str(m):10s} {int(r['trades']):7d} {r['pnl']:12.2f}")

    lines.append("")
    lines.append("7) Long vs Short:")
    for side in ("long", "short"):
        s = tdf[tdf["direction"] == side]
        if s.empty:
            lines.append(f"   {side}: 0 trades")
            continue
        sw = (s["pnl"] > 0).sum()
        lines.append(
            f"   {side}: trades={len(s)} wins={sw} WR={sw/len(s)*100:.1f}% "
            f"PnL=${s['pnl'].sum():.2f}"
        )

    lines.append("")
    lines.append("8) Trade log (also CSV):")
    lines.append(
        f"{'#':>3} {'Dir':5} {'EntryTime':20} {'ExitTime':20} "
        f"{'Entry':>10} {'Exit':>10} {'SL':>10} {'TP':>10} {'PnL':>10} {'Hit':4}"
    )
    for i, r in tdf.iterrows():
        lines.append(
            f"{i+1:3d} {r['direction']:5s} "
            f"{str(r['entry_time'])[:19]:20s} {str(r['exit_time'])[:19]:20s} "
            f"{r['entry_price']:10.2f} {r['exit_price']:10.2f} "
            f"{r['sl_price']:10.2f} {r['tp_price']:10.2f} "
            f"{r['pnl']:10.2f} {r['exit_hit']:4s}"
        )
    return "\n".join(lines)


def plot_equity(symbol: str, eq: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eq.index, eq.values, color="#1f6feb", lw=1.5)
    ax.axhline(INITIAL_CAPITAL, color="#888", ls="--", lw=0.8)
    ax.set_title(f"{symbol} — PMTS 4H+5m Equity (fixed ${FIXED_NOTIONAL:.0f} notional)")
    ax.set_ylabel("Equity $")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    print("PMTS exact v3 backtest")
    print(
        f"params: htf_swing={htf_swing_len} atr={atr_len}×{atr_mult} "
        f"zone={zone_upper}/{zone_lower} ltf_swing={ltf_swing_len} "
        f"sl_pct={sl_pct}% tp_pct={tp_pct}% notional=${FIXED_NOTIONAL}"
    )
    print(
        "Symbols: BTCUSDT/PAXGUSDT = Binance; "
        "XAUUSD = Dukascopy gold spot (OANDA-style FX hours, bid OHLC)"
    )
    all_text = []
    for sym in ("BTCUSDT", "PAXGUSDT", "XAUUSD"):
        print(f"\nRunning {sym}...")
        out = run_backtest(sym)
        tdf = out["trades"]
        eq = out["equity"]
        label = sym
        if sym == "XAUUSD":
            label = "XAUUSD (Dukascopy ≈ OANDA XAU/USD)"
        text = summarize(label, tdf, eq, out["final"])
        print(text)
        all_text.append(text)
        stem = sym.lower()
        if not tdf.empty:
            tdf.to_csv(OUT / f"{stem}_trade_log.csv", index=False)
        eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
        plot_equity(label, eq, OUT / f"{stem}_equity_curve.png")
        print(f"Wrote {OUT / f'{stem}_equity_curve.png'}")

    (OUT / "pmts_exact_v3_report.txt").write_text("\n\n".join(all_text) + "\n")
    meta = {
        "htf_swing_len": htf_swing_len,
        "atr_len": atr_len,
        "atr_mult": atr_mult,
        "zone_upper": zone_upper,
        "zone_lower": zone_lower,
        "ltf_swing_len": ltf_swing_len,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "initial_capital": INITIAL_CAPITAL,
        "fixed_notional": FIXED_NOTIONAL,
        "xauusd_source": "Dukascopy XAUUSD M5 bid (proxy for OANDA XAU/USD; not OANDA API)",
    }
    (OUT / "params.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll artifacts in {OUT}")


if __name__ == "__main__":
    main()
