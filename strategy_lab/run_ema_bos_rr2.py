#!/usr/bin/env python3
"""
9/21 EMA + Swing High Break (BOS) — Long only, 1:2 RR.

Entry (long):
  - Close above 9 EMA AND 21 EMA
  - Close breaks last swing high (BOS)
  - Volume >= 20-bar average (active market)
  - Skip Asian session (trade London/NY 07:00–22:00 UTC only)
Stop:  below last swing low
Target: 1:2 risk-reward (TP = entry + 2 * risk)
Sizing: compounding % of equity + slippage/fees
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.costs import CostModel, apply_entry_price, apply_exit_price

EMA_FAST = 9
EMA_SLOW = 21
PIVOT_LEN = 5
RR = 2.0  # 1:2
VOL_MA_LEN = 20
VOL_MULT = 1.0  # volume must be >= avg * mult

# London ∪ NY (UTC) — skip Asian dead zone ~00:00–07:00
SESSION_START_H = 7
SESSION_END_H = 22

INITIAL_CAPITAL = 10_000.0
POSITION_PCT = 0.05
DAYS = 365

CRYPTO_COSTS = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
XAU_SLIP_PIPS = 0.5
PIP = {"XAUUSD": 0.10, "PAXGUSDT": 0.10}

OUT = Path(__file__).resolve().parent / "results" / "ema_bos_rr2"
FLAT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


def vol_sma(volume: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(volume).rolling(period, min_periods=period).mean().to_numpy()


def in_session(ts: pd.Timestamp) -> bool:
    return SESSION_START_H <= ts.hour < SESSION_END_H


def max_drawdown(eq: pd.Series) -> tuple[float, float]:
    dd = eq - eq.cummax()
    max_dd_usd = float(dd.min()) if len(dd) else 0.0
    peak = float(eq.cummax().loc[dd.idxmin()]) if len(dd) and dd.min() < 0 else float(eq.max()) if len(eq) else INITIAL_CAPITAL
    max_dd_pct = (max_dd_usd / peak * 100) if peak > 0 else 0.0
    return max_dd_usd, max_dd_pct


def running_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Return running last swing high / swing low at each bar."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(left + right, n):
        c = i - right
        hv, lv = high[c], low[c]
        if all(high[j] < hv for j in range(c - left, c + right + 1) if j != c):
            ph[i] = hv
        if all(low[j] > lv for j in range(c - left, c + right + 1) if j != c):
            pl[i] = lv
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    sh, sl = np.nan, np.nan
    for i in range(n):
        if np.isfinite(ph[i]):
            sh = ph[i]
        if np.isfinite(pl[i]):
            sl = pl[i]
        last_sh[i] = sh
        last_sl[i] = sl
    return last_sh, last_sl


def entry_fill(raw: float, direction: str, symbol: str) -> float:
    pip = PIP.get(symbol, 0.10)
    if symbol == "XAUUSD":
        s = XAU_SLIP_PIPS * pip
        return raw + s if direction == "long" else raw - s
    return apply_entry_price(raw, direction, CRYPTO_COSTS)


def exit_fill(raw: float, direction: str, symbol: str) -> float:
    pip = PIP.get(symbol, 0.10)
    if symbol == "XAUUSD":
        s = XAU_SLIP_PIPS * pip
        return raw - s if direction == "long" else raw + s
    return apply_exit_price(raw, direction, CRYPTO_COSTS)


def close_pnl(direction: str, entry: float, exit_raw: float, qty: float, symbol: str) -> tuple[float, float]:
    xf = exit_fill(exit_raw, direction, symbol)
    gross = (xf - entry) * qty if direction == "long" else (entry - xf) * qty
    fees = 0.0 if symbol == "XAUUSD" else (entry + xf) * qty * CRYPTO_COSTS.commission_rate
    return gross - fees, fees


def run_backtest(
    symbol: str,
    tf: str,
    df: pd.DataFrame | None = None,
    *,
    use_volume_filter: bool = False,
    use_session_filter: bool = False,
) -> dict:
    if df is None:
        df = load_or_fetch(symbol, tf, DAYS)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    e9 = ema(c, EMA_FAST)
    e21 = ema(c, EMA_SLOW)
    vma = vol_sma(v, VOL_MA_LEN)
    swing_hi, swing_lo = running_pivots(h, l, PIVOT_LEN, PIVOT_LEN)

    warmup = max(EMA_SLOW, PIVOT_LEN * 2, VOL_MA_LEN) + 5
    capital = INITIAL_CAPITAL
    total_fees = 0.0
    trades: list[dict] = []
    equity = [capital]
    equity_ts = [ts.iloc[warmup]]
    pos = None

    for i in range(warmup, len(df) - 1):
        if pos is not None:
            entry, sl, tp, qty = pos["entry"], pos["sl"], pos["tp"], pos["qty"]
            hit = None
            exit_px = None
            if l[i] <= sl:
                hit, exit_px = "sl", sl
            elif h[i] >= tp:
                hit, exit_px = "tp", tp
            if hit:
                pnl, fees = close_pnl("long", entry, exit_px, qty, symbol)
                total_fees += fees
                capital += pnl
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": ts.iloc[i],
                        "entry": entry,
                        "exit": exit_fill(exit_px, "long", symbol),
                        "sl": sl,
                        "tp": tp,
                        "pnl": pnl,
                        "fees": fees,
                        "equity_after": capital,
                        "exit_hit": hit,
                        "risk": entry - sl,
                        "reward": tp - entry,
                    }
                )
                equity.append(capital)
                equity_ts.append(ts.iloc[i])
                pos = None
            continue

        if use_session_filter and not in_session(ts.iloc[i]):
            continue

        sh, slo = swing_hi[i], swing_lo[i]
        if not (np.isfinite(sh) and np.isfinite(slo)):
            continue
        if not (np.isfinite(e9[i]) and np.isfinite(e21[i])):
            continue
        if use_volume_filter and not (np.isfinite(vma[i]) and v[i] >= vma[i] * VOL_MULT):
            continue

        above_emas = c[i] > e9[i] and c[i] > e21[i]
        sh_level = swing_hi[i]
        bos = np.isfinite(sh_level) and c[i] > sh_level and c[i - 1] <= sh_level

        if above_emas and bos:
            raw_entry = float(o[i + 1])
            fill = entry_fill(raw_entry, "long", symbol)
            sl = float(slo)
            if sl >= fill:
                continue
            risk = fill - sl
            tp = fill + RR * risk
            notional = capital * POSITION_PCT
            qty = notional / fill
            pos = {
                "entry": fill,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "entry_time": ts.iloc[i + 1],
                "equity_at_entry": capital,
            }

    if pos is not None:
        pnl, fees = close_pnl("long", pos["entry"], float(c[-1]), pos["qty"], symbol)
        total_fees += fees
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": ts.iloc[-1],
                "entry": pos["entry"],
                "exit": exit_fill(float(c[-1]), "long", symbol),
                "sl": pos["sl"],
                "tp": pos["tp"],
                "pnl": pnl,
                "fees": fees,
                "equity_after": capital,
                "exit_hit": "eod",
                "risk": pos["entry"] - pos["sl"],
                "reward": pos["tp"] - pos["entry"],
            }
        )
        equity.append(capital)
        equity_ts.append(ts.iloc[-1])

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    max_dd_usd, max_dd_pct = max_drawdown(eq)
    return {
        "symbol": symbol,
        "tf": tf,
        "trades": tdf,
        "equity": eq,
        "final": capital,
        "total_fees": total_fees,
        "max_dd_usd": max_dd_usd,
        "max_dd_pct": max_dd_pct,
        "use_volume_filter": use_volume_filter,
        "use_session_filter": use_session_filter,
    }


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


def summarize_run(out: dict) -> dict:
    tdf, eq, final = out["trades"], out["equity"], out["final"]
    n = len(tdf)
    net = final - INITIAL_CAPITAL
    return {
        "symbol": out["symbol"],
        "timeframe": out["tf"],
        "filter_mode": out.get("filter_mode", ""),
        "volume_filter": out.get("use_volume_filter", False),
        "session_filter": out.get("use_session_filter", False),
        "trades": n,
        "wins": int((tdf.pnl > 0).sum()) if n else 0,
        "wr": float((tdf.pnl > 0).mean() * 100) if n else 0.0,
        "pf": (
            float(tdf.loc[tdf.pnl > 0, "pnl"].sum() / abs(tdf.loc[tdf.pnl < 0, "pnl"].sum()))
            if n and (tdf.pnl < 0).any() and (tdf.pnl > 0).any()
            else float("nan")
        ),
        "net": net,
        "net_pct": net / INITIAL_CAPITAL * 100,
        "final_capital": final,
        "max_dd_usd": out["max_dd_usd"],
        "max_dd_pct": out["max_dd_pct"],
        "total_fees": out["total_fees"],
        "tp_hits": int((tdf.exit_hit == "tp").sum()) if n else 0,
        "sl_hits": int((tdf.exit_hit == "sl").sum()) if n else 0,
    }


FILTER_MODES = [
    ("baseline", False, False),
    ("volume_only", True, False),
    ("session_only", False, True),
    ("both", True, True),
]


def main():
    print(
        f"9/21 EMA BOS — filter comparison | {POSITION_PCT*100:.0f}% compound + slippage\n"
        f"Modes: baseline | volume_only | session_only | both"
    )
    rows = []
    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            df = load_or_fetch(sym, tf, DAYS)
            print(f"\n{'='*60}\n{sym} {tf}")
            for mode_name, vol_f, sess_f in FILTER_MODES:
                out = run_backtest(sym, tf, df, use_volume_filter=vol_f, use_session_filter=sess_f)
                out["filter_mode"] = mode_name
                s = summarize_run(out)
                rows.append(s)
                print(
                    f"  [{mode_name:13s}] trades={s['trades']:4d} WR={s['wr']:5.1f}% PF={s['pf']:5.2f} "
                    f"net={s['net_pct']:+6.2f}% MaxDD={s['max_dd_pct']:6.2f}% (${s['max_dd_usd']:.0f})"
                )
                if mode_name in ("volume_only", "session_only", "both"):
                    stem = f"{sym.lower()}_{tf}_{mode_name}"
                    out["trades"].to_csv(OUT / f"{stem}_trades.csv", index=False)
                    out["equity"].to_csv(OUT / f"{stem}_equity.csv", header=["equity"])

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "filter_compare.csv", index=False)
    rdf.to_csv(FLAT / "ema_bos_rr2_filter_compare.csv", index=False)

    print("\n" + "=" * 90)
    print("FILTER COMPARISON SUMMARY")
    print("=" * 90)
    cols = ["symbol", "timeframe", "filter_mode", "trades", "wr", "pf", "net_pct", "max_dd_usd", "max_dd_pct", "total_fees"]
    print(rdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()
