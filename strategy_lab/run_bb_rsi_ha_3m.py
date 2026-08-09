#!/usr/bin/env python3
"""
Bollinger(20,2) + RSI(14) mean-reversion with 15M Heikin Ashi filter.

Entry TF: 3M | HTF: 15M HA
BUY:  HA bull + 3M touches BB lower + closes back inside + RSI<35
SELL: HA bear + 3M touches BB upper + closes back inside + RSI>65
Skip Asia session 00:00–08:00 UTC.
Entry at next 3M open.

Exits (adjustable; not specified by user — classic mean-reversion defaults):
  TP = BB middle (basis) at signal time
  SL = signal candle low (long) / high (short)
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

# ── parameters ─────────────────────────────────────────────────────────
bb_len = 20
bb_std = 2.0
rsi_len = 14
rsi_os = 35.0
rsi_ob = 65.0
asia_start_h = 0  # inclusive UTC
asia_end_h = 8  # exclusive UTC

INITIAL_CAPITAL = 10_000.0
FIXED_NOTIONAL = 3_000.0
DAYS = 365
OUT = Path(__file__).resolve().parent / "results" / "bb_rsi_ha_3m"
OUT.mkdir(parents=True, exist_ok=True)


def rsi_wilder(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.mean(gain[1 : period + 1])
    avg_l = np.mean(loss[1 : period + 1])
    if avg_l == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        if avg_l == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def bollinger(close: np.ndarray, length: int, nstd: float):
    s = pd.Series(close)
    mid = s.rolling(length).mean().to_numpy()
    sd = s.rolling(length).std(ddof=0).to_numpy()
    upper = mid + nstd * sd
    lower = mid - nstd * sd
    return mid, upper, lower


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    ha_o = np.empty(n)
    ha_c = np.empty(n)
    ha_h = np.empty(n)
    ha_l = np.empty(n)
    ha_c[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_o[0] = (o[0] + c[0]) / 2.0
    ha_h[0] = max(h[0], ha_o[0], ha_c[0])
    ha_l[0] = min(l[0], ha_o[0], ha_c[0])
    for i in range(1, n):
        ha_c[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
        ha_h[i] = max(h[i], ha_o[i], ha_c[i])
        ha_l[i] = min(l[i], ha_o[i], ha_c[i])
    out = df[["timestamp"]].copy()
    out["ha_open"] = ha_o
    out["ha_close"] = ha_c
    out["ha_bull"] = ha_c > ha_o
    out["ha_bear"] = ha_c < ha_o
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    out = (
        x.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    out.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return out


def in_asia(ts: pd.Timestamp) -> bool:
    h = pd.Timestamp(ts).tz_convert("UTC").hour
    return asia_start_h <= h < asia_end_h


def _utc_floor_s(s: pd.Series) -> pd.Series:
    """UTC timestamps floored to seconds — same dtype for merge_asof."""
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("UTC").dt.floor("s")


def run_backtest(symbol: str) -> dict:
    # Prefer native 3m from Binance; fallback resample 1m if needed
    try:
        df3 = load_or_fetch(symbol, "3m", DAYS)
    except Exception:
        df1 = load_or_fetch(symbol, "1m", DAYS)
        df3 = resample_ohlc(df1, "3min")

    df3 = df3.copy()
    df3["timestamp"] = _utc_floor_s(df3["timestamp"])

    # 15M from 3M (or fetch)
    try:
        df15 = load_or_fetch(symbol, "15m", DAYS)
        df15 = df15.copy()
        df15["timestamp"] = _utc_floor_s(df15["timestamp"])
    except Exception:
        df15 = resample_ohlc(df3, "15min")
        df15["timestamp"] = _utc_floor_s(df15["timestamp"])

    ha = heikin_ashi(df15)
    # only closed HTF: shift 1
    ha = ha.copy()
    ha["timestamp"] = _utc_floor_s(ha["timestamp"])
    for col in ("ha_bull", "ha_bear", "ha_open", "ha_close"):
        ha[col] = ha[col].shift(1)

    mapped = pd.merge_asof(
        df3[["timestamp"]].sort_values("timestamp"),
        ha.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    close = df3["close"].to_numpy(float)
    high = df3["high"].to_numpy(float)
    low = df3["low"].to_numpy(float)
    opn = df3["open"].to_numpy(float)
    ts = pd.to_datetime(df3["timestamp"], utc=True)
    n = len(df3)

    mid, upper, lower = bollinger(close, bb_len, bb_std)
    rsi = rsi_wilder(close, rsi_len)
    ha_bull = mapped["ha_bull"].fillna(False).to_numpy(dtype=bool)
    ha_bear = mapped["ha_bear"].fillna(False).to_numpy(dtype=bool)

    # signal on closed bar i → enter at open of i+1
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(lower[i]) or not np.isfinite(rsi[i]):
            continue
        # BUY: touch lower, close back inside (>= lower), RSI os, HA bull
        if (
            ha_bull[i]
            and low[i] <= lower[i]
            and close[i] >= lower[i]
            and rsi[i] < rsi_os
        ):
            buy_sig[i] = True
        if (
            ha_bear[i]
            and high[i] >= upper[i]
            and close[i] <= upper[i]
            and rsi[i] > rsi_ob
        ):
            sell_sig[i] = True

    trades = []
    capital = INITIAL_CAPITAL
    equity = [INITIAL_CAPITAL]
    equity_ts = [ts.iloc[0]]
    in_pos = False
    pos = None

    for i in range(1, n):
        # manage open position on bar i (use high/low)
        if in_pos:
            hit = None
            exit_px = None
            if pos["dir"] == "long":
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
                        "exit_time": ts.iloc[i],
                        "exit_price": float(exit_px),
                        "sl_price": pos["sl"],
                        "tp_price": pos["tp"],
                        "pnl": float(pnl),
                        "exit_hit": hit,
                        "signal_rsi": pos["rsi"],
                    }
                )
                equity.append(capital)
                equity_ts.append(ts.iloc[i])
                in_pos = False
                pos = None

        if in_pos:
            continue

        # entry at open of bar i if signal on bar i-1
        sig_i = i - 1
        entry_ts = ts.iloc[i]
        if in_asia(entry_ts):
            continue

        if buy_sig[sig_i] and np.isfinite(mid[sig_i]):
            entry = float(opn[i])
            sl = float(low[sig_i])
            tp = float(mid[sig_i])
            if not (sl < entry < tp):
                continue
            qty = FIXED_NOTIONAL / entry
            pos = {
                "dir": "long",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": entry_ts,
                "rsi": float(rsi[sig_i]),
            }
            in_pos = True
        elif sell_sig[sig_i] and np.isfinite(mid[sig_i]):
            entry = float(opn[i])
            sl = float(high[sig_i])
            tp = float(mid[sig_i])
            if not (tp < entry < sl):
                continue
            qty = FIXED_NOTIONAL / entry
            pos = {
                "dir": "short",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": entry_ts,
                "rsi": float(rsi[sig_i]),
            }
            in_pos = True

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
                "exit_time": ts.iloc[-1],
                "exit_price": exit_px,
                "sl_price": pos["sl"],
                "tp_price": pos["tp"],
                "pnl": float(pnl),
                "exit_hit": "EOD",
                "signal_rsi": pos["rsi"],
            }
        )
        equity.append(capital)
        equity_ts.append(ts.iloc[-1])

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {"symbol": symbol, "trades": tdf, "equity": eq, "final": capital}


def summarize(symbol: str, tdf: pd.DataFrame, eq: pd.Series, final: float) -> str:
    lines = [
        "=" * 90,
        f"RESULTS — {symbol}  |  BB({bb_len},{bb_std}) RSI({rsi_len}) HA15 / Entry3m",
        f"Exits: TP=BB mid @ signal, SL=signal candle extreme | Asia skip {asia_start_h:02d}-{asia_end_h:02d} UTC",
        "=" * 90,
    ]
    if tdf.empty:
        lines.append("No trades.")
        return "\n".join(lines)

    wins = tdf[tdf["pnl"] > 1e-8]
    losses = tdf[tdf["pnl"] < -1e-8]
    be = tdf[tdf["pnl"].abs() <= 1e-8]
    n = len(tdf)
    wr = len(wins) / n * 100
    gp = wins["pnl"].sum() if len(wins) else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    avg_w = wins["pnl"].mean() if len(wins) else 0.0
    avg_l = losses["pnl"].mean() if len(losses) else 0.0
    wl = abs(avg_w / avg_l) if avg_l != 0 else float("inf")
    net = final - INITIAL_CAPITAL
    net_pct = net / INITIAL_CAPITAL * 100
    dd = eq - eq.cummax()
    max_dd = float(dd.min())
    max_dd_pct = max_dd / INITIAL_CAPITAL * 100

    lines += [
        f"1) Trades={n}  Wins={len(wins)}  Losses={len(losses)}  BE={len(be)}  WR={wr:.1f}%",
        f"2) Profit Factor={pf:.3f}  (GP=${gp:.2f} / GL=${gl:.2f})",
        f"3) AvgWin=${avg_w:.2f}  AvgLoss=${avg_l:.2f}  W/L={wl:.3f}",
        f"4) Net=${net:.2f} ({net_pct:+.2f}%)  Final=${final:.2f}  notional=${FIXED_NOTIONAL:.0f}",
        f"5) MaxDD=${max_dd:.2f} ({max_dd_pct:.2f}%)",
        "",
        "6) Month-by-month:",
        f"{'Month':10s} {'Trades':>7s} {'PnL$':>12s}",
    ]
    t = tdf.copy()
    t["month"] = pd.to_datetime(t["exit_time"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)
    monthly = t.groupby("month").agg(trades=("pnl", "count"), pnl=("pnl", "sum"))
    for m, r in monthly.iterrows():
        lines.append(f"{m:10s} {int(r['trades']):7d} {r['pnl']:12.2f}")

    lines.append("")
    lines.append("7) Long vs Short:")
    for side in ("long", "short"):
        s = tdf[tdf["direction"] == side]
        if s.empty:
            lines.append(f"   {side}: 0")
            continue
        sw = (s["pnl"] > 0).sum()
        lines.append(
            f"   {side}: n={len(s)} wins={sw} WR={sw/len(s)*100:.1f}% PnL=${s['pnl'].sum():.2f}"
        )

    lines.append("")
    lines.append("8) Trade log (CSV has full):")
    lines.append(
        f"{'#':>3} {'Dir':5} {'EntryTime':20} {'ExitTime':20} "
        f"{'Entry':>10} {'Exit':>10} {'SL':>10} {'TP':>10} {'PnL':>10} {'Hit':4}"
    )
    for i, r in tdf.reset_index(drop=True).iterrows():
        lines.append(
            f"{i+1:3d} {r['direction']:5s} "
            f"{str(r['entry_time'])[:19]:20s} {str(r['exit_time'])[:19]:20s} "
            f"{r['entry_price']:10.4f} {r['exit_price']:10.4f} "
            f"{r['sl_price']:10.4f} {r['tp_price']:10.4f} "
            f"{r['pnl']:10.2f} {r['exit_hit']:4s}"
        )
    return "\n".join(lines)


def plot_equity(symbol: str, eq: pd.Series, path: Path, net_pct: float):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(eq.index, eq.values, color="#1f6feb", lw=1.4)
    ax.axhline(INITIAL_CAPITAL, color="#888", ls="--", lw=0.8)
    ax.set_title(f"{symbol} BB+RSI+HA15 / 3m  ({net_pct:+.2f}%)")
    ax.set_ylabel("Equity $")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    print(
        f"BB({bb_len},{bb_std}) RSI({rsi_len} os={rsi_os} ob={rsi_ob}) "
        f"HA15 filter | entry 3m | Asia skip {asia_start_h}-{asia_end_h} UTC"
    )
    print(f"Exits: TP=BB mid, SL=signal extreme | capital={INITIAL_CAPITAL} notional={FIXED_NOTIONAL}")
    all_text = []
    summary_rows = []
    for sym in ("BTCUSDT", "PAXGUSDT"):
        print(f"\nRunning {sym}...")
        out = run_backtest(sym)
        tdf, eq, final = out["trades"], out["equity"], out["final"]
        text = summarize(sym, tdf, eq, final)
        print(text)
        all_text.append(text)
        net = final - INITIAL_CAPITAL
        net_pct = net / INITIAL_CAPITAL * 100
        stem = sym.lower()
        if not tdf.empty:
            tdf.to_csv(OUT / f"{stem}_trades.csv", index=False)
        eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
        plot_equity(sym, eq, OUT / f"{stem}_equity.png", net_pct)
        wins = int((tdf["pnl"] > 0).sum()) if not tdf.empty else 0
        losses = int((tdf["pnl"] < 0).sum()) if not tdf.empty else 0
        summary_rows.append(
            {
                "symbol": sym,
                "trades": len(tdf),
                "wins": wins,
                "losses": losses,
                "wr": wins / len(tdf) * 100 if len(tdf) else 0,
                "net": net,
                "net_pct": net_pct,
                "final": final,
            }
        )

    (OUT / "report.txt").write_text("\n\n".join(all_text) + "\n")
    meta = {
        "bb_len": bb_len,
        "bb_std": bb_std,
        "rsi_len": rsi_len,
        "rsi_os": rsi_os,
        "rsi_ob": rsi_ob,
        "asia_utc": f"{asia_start_h:02d}-{asia_end_h:02d}",
        "entry_tf": "3m",
        "htf": "15m Heikin Ashi (closed bar)",
        "tp": "BB middle at signal",
        "sl": "signal candle extreme",
        "initial_capital": INITIAL_CAPITAL,
        "fixed_notional": FIXED_NOTIONAL,
        "days": DAYS,
    }
    (OUT / "params.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    print(f"\nArtifacts → {OUT}")


if __name__ == "__main__":
    main()
