#!/usr/bin/env python3
"""
EMA BOS best setup (volume_only) — test EMA-based exits.

Exit modes:
  rr2:        SL swing low, TP 1:2 RR (previous best)
  ema_sl_tp:  SL at 21 EMA touch, TP at 9 EMA pullback touch (after price ran up)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.run_ema_bos_rr2 import (
    DAYS,
    EMA_FAST,
    EMA_SLOW,
    FLAT,
    INITIAL_CAPITAL,
    OUT,
    PIVOT_LEN,
    POSITION_PCT,
    RR,
    VOL_MA_LEN,
    VOL_MULT,
    close_pnl,
    ema,
    entry_fill,
    exit_fill,
    max_drawdown,
    running_pivots,
    vol_sma,
)

OUT.mkdir(parents=True, exist_ok=True)


def run_backtest(
    symbol: str,
    tf: str,
    df: pd.DataFrame,
    exit_mode: str = "rr2",
) -> dict:
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
            entry = pos["entry"]
            qty = pos["qty"]
            hit, exit_px, reason = None, None, None

            if exit_mode == "rr2":
                sl, tp = pos["sl"], pos["tp"]
                if l[i] <= sl:
                    hit, exit_px, reason = True, sl, "sl"
                elif h[i] >= tp:
                    hit, exit_px, reason = True, tp, "tp"
            else:  # ema_sl_tp
                e21v, e9v = e21[i], e9[i]
                if not (np.isfinite(e21v) and np.isfinite(e9v)):
                    continue
                # track extension above entry for valid 9 EMA TP (pullback touch)
                if h[i] > pos["max_high"]:
                    pos["max_high"] = h[i]
                if pos["max_high"] > entry and l[i] <= e9v:
                    hit, exit_px, reason = True, e9v, "tp_ema9"
                elif l[i] <= e21v:
                    hit, exit_px, reason = True, e21v, "sl_ema21"

            if hit and exit_px is not None:
                pnl, fees = close_pnl("long", entry, exit_px, qty, symbol)
                total_fees += fees
                capital += pnl
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": ts.iloc[i],
                        "pnl": pnl,
                        "exit_hit": reason,
                        "exit_mode": exit_mode,
                    }
                )
                equity.append(capital)
                equity_ts.append(ts.iloc[i])
                pos = None
            continue

        sh, slo = swing_hi[i], swing_lo[i]
        if not (np.isfinite(sh) and np.isfinite(slo)):
            continue
        if not (np.isfinite(e9[i]) and np.isfinite(e21[i])):
            continue
        if not (np.isfinite(vma[i]) and v[i] >= vma[i] * VOL_MULT):
            continue

        above_emas = c[i] > e9[i] and c[i] > e21[i]
        sh_level = swing_hi[i]
        bos = np.isfinite(sh_level) and c[i] > sh_level and c[i - 1] <= sh_level
        if not (above_emas and bos):
            continue

        raw_entry = float(o[i + 1])
        fill = entry_fill(raw_entry, "long", symbol)
        sl_px = float(slo)
        if exit_mode == "rr2":
            if sl_px >= fill:
                continue
            risk = fill - sl_px
            tp_px = fill + RR * risk
        else:
            if e21[i] >= fill:
                continue
            sl_px = float(e21[i])
            tp_px = float(e9[i])

        notional = capital * POSITION_PCT
        qty = notional / fill
        pos = {
            "entry": fill,
            "sl": sl_px,
            "tp": tp_px,
            "qty": qty,
            "entry_time": ts.iloc[i + 1],
            "max_high": fill,
        }

    if pos is not None:
        pnl, fees = close_pnl("long", pos["entry"], float(c[-1]), pos["qty"], symbol)
        total_fees += fees
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": ts.iloc[-1],
                "pnl": pnl,
                "exit_hit": "eod",
                "exit_mode": exit_mode,
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
        "exit_mode": exit_mode,
        "trades": tdf,
        "equity": eq,
        "final": capital,
        "total_fees": total_fees,
        "max_dd_usd": max_dd_usd,
        "max_dd_pct": max_dd_pct,
    }


def summarize(out: dict) -> dict:
    tdf, final = out["trades"], out["final"]
    n = len(tdf)
    net = final - INITIAL_CAPITAL
    return {
        "symbol": out["symbol"],
        "timeframe": out["tf"],
        "exit_mode": out["exit_mode"],
        "trades": n,
        "wr": float((tdf.pnl > 0).mean() * 100) if n else 0.0,
        "pf": (
            float(tdf.loc[tdf.pnl > 0, "pnl"].sum() / abs(tdf.loc[tdf.pnl < 0, "pnl"].sum()))
            if n and (tdf.pnl < 0).any() and (tdf.pnl > 0).any()
            else float("nan")
        ),
        "net_pct": net / INITIAL_CAPITAL * 100,
        "final_capital": final,
        "max_dd_usd": out["max_dd_usd"],
        "max_dd_pct": out["max_dd_pct"],
        "total_fees": out["total_fees"],
        "exits": tdf.exit_hit.value_counts().to_dict() if n else {},
    }


def main():
    print("EMA BOS volume_only — exit test: RR 1:2 vs SL=21EMA / TP=9EMA touch")
    rows = []
    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            df = load_or_fetch(sym, tf, DAYS)
            print(f"\n{'='*60}\n{sym} {tf}")
            for mode in ("rr2", "ema_sl_tp"):
                out = run_backtest(sym, tf, df, exit_mode=mode)
                s = summarize(out)
                rows.append(s)
                label = "SL swing/TP 1:2" if mode == "rr2" else "SL 21EMA/TP 9EMA"
                print(
                    f"  [{label:22s}] trades={s['trades']:4d} WR={s['wr']:5.1f}% PF={s['pf']:5.2f} "
                    f"net={s['net_pct']:+6.2f}% MaxDD={s['max_dd_pct']:6.2f}% (${s['max_dd_usd']:.0f})"
                )
                if s["exits"]:
                    print(f"    exits: {s['exits']}")

    rdf = pd.DataFrame(rows)
    rdf.drop(columns=["exits"], errors="ignore").to_csv(OUT / "exit_compare.csv", index=False)
    rdf.drop(columns=["exits"], errors="ignore").to_csv(FLAT / "ema_bos_rr2_exit_compare.csv", index=False)
    print("\n" + "=" * 90)
    print(rdf[["symbol", "timeframe", "exit_mode", "trades", "wr", "pf", "net_pct", "max_dd_pct", "max_dd_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
