#!/usr/bin/env python3
"""
EMA BOS best setup (volume_only) + bullish candlestick pattern filter.

Compare: volume_only vs volume_only + candle confirmation.
Patterns: engulfing, hammer, pin bar, strong bullish body.
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
    OUT,
    FLAT,
    INITIAL_CAPITAL,
    POSITION_PCT,
    plot_eq,
    run_backtest,
    summarize_run,
)

CANDLE_PATTERNS = ("engulfing", "hammer", "pin_bar", "strong_bull")


def _parts(o, h, l, c, i):
    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    if rng <= 0:
        return None
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    return body, rng, upper, lower


def pattern_engulfing(o, h, l, c, i) -> bool:
    if i < 1:
        return False
    return c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] > o[i - 1] and o[i] < c[i - 1]


def pattern_hammer(o, h, l, c, i) -> bool:
    p = _parts(o, h, l, c, i)
    if p is None:
        return False
    body, rng, upper, lower = p
    b = max(body, rng * 0.05)
    return lower >= 2 * b and upper <= b


def pattern_pin_bar(o, h, l, c, i) -> bool:
    p = _parts(o, h, l, c, i)
    if p is None:
        return False
    body, rng, upper, lower = p
    return lower >= 0.66 * rng and upper <= 0.25 * rng and c[i] >= o[i] - rng * 0.1


def pattern_strong_bull(o, h, l, c, i) -> bool:
    p = _parts(o, h, l, c, i)
    if p is None:
        return False
    body, rng, _, _ = p
    return c[i] > o[i] and body >= 0.55 * rng


def detect_bullish_pattern(o, h, l, c, i) -> str | None:
    if pattern_engulfing(o, h, l, c, i):
        return "engulfing"
    if pattern_hammer(o, h, l, c, i):
        return "hammer"
    if pattern_pin_bar(o, h, l, c, i):
        return "pin_bar"
    if pattern_strong_bull(o, h, l, c, i):
        return "strong_bull"
    return None


def run_backtest_candle(
    symbol: str,
    tf: str,
    df: pd.DataFrame,
    *,
    pattern: str | None = None,  # None = any pattern
) -> dict:
    """Volume-only base + optional specific candle pattern."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    # Monkey-patch: wrap run_backtest by precomputing pattern mask
    pattern_at = [detect_bullish_pattern(o, h, l, c, i) for i in range(len(df))]

    from strategy_lab.run_ema_bos_rr2 import (
        EMA_FAST,
        EMA_SLOW,
        PIVOT_LEN,
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

    v = df["volume"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    e9 = ema(c, EMA_FAST)
    e21 = ema(c, EMA_SLOW)
    vma = vol_sma(v, VOL_MA_LEN)
    swing_hi, swing_lo = running_pivots(h, l, PIVOT_LEN, PIVOT_LEN)
    warmup = max(EMA_SLOW, PIVOT_LEN * 2, VOL_MA_LEN) + 5
    capital = INITIAL_CAPITAL
    total_fees = 0.0
    trades = []
    equity = [capital]
    equity_ts = [ts.iloc[warmup]]
    pos = None

    for i in range(warmup, len(df) - 1):
        if pos is not None:
            entry, sl, tp, qty = pos["entry"], pos["sl"], pos["tp"], pos["qty"]
            hit, exit_px = None, None
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
                        "pnl": pnl,
                        "exit_hit": hit,
                        "candle_pattern": pos.get("candle_pattern"),
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

        pat = pattern_at[i]
        if pat is None:
            continue
        if pattern is not None and pat != pattern:
            continue

        above_emas = c[i] > e9[i] and c[i] > e21[i]
        sh_level = swing_hi[i]
        bos = np.isfinite(sh_level) and c[i] > sh_level and c[i - 1] <= sh_level
        if not (above_emas and bos):
            continue

        raw_entry = float(o[i + 1])
        fill = entry_fill(raw_entry, "long", symbol)
        sl_px = float(slo)
        if sl_px >= fill:
            continue
        risk = fill - sl_px
        tp_px = fill + RR * risk
        qty = (capital * POSITION_PCT) / fill
        pos = {
            "entry": fill,
            "sl": sl_px,
            "tp": tp_px,
            "qty": qty,
            "entry_time": ts.iloc[i + 1],
            "candle_pattern": pat,
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
                "candle_pattern": pos.get("candle_pattern"),
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
        "use_volume_filter": True,
        "use_session_filter": False,
        "use_candle_filter": True,
        "candle_pattern": pattern or "any",
    }


def main():
    print("EMA BOS — volume_only (best) vs + candlestick pattern | 5% compound + slippage")
    rows = []
    modes = [
        ("volume_only", False),
        ("volume_candle_any", True),
    ]

    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            df = load_or_fetch(sym, tf, DAYS)
            print(f"\n{'='*60}\n{sym} {tf}")
            for mode_name, use_candle in modes:
                if use_candle:
                    out = run_backtest_candle(sym, tf, df, pattern=None)
                else:
                    out = run_backtest(sym, tf, df, use_volume_filter=True, use_session_filter=False)
                out["filter_mode"] = mode_name
                s = summarize_run(out)
                rows.append(s)
                print(
                    f"  [{mode_name:20s}] trades={s['trades']:4d} WR={s['wr']:5.1f}% PF={s['pf']:5.2f} "
                    f"net={s['net_pct']:+6.2f}% MaxDD={s['max_dd_pct']:6.2f}%"
                )

            # Per-pattern breakdown on best TF focus
            if sym == "XAUUSD" and tf == "5m":
                print("  --- candle pattern breakdown (XAU 5m) ---")
                for pat in CANDLE_PATTERNS:
                    out = run_backtest_candle(sym, tf, df, pattern=pat)
                    out["filter_mode"] = f"candle_{pat}"
                    s = summarize_run(out)
                    rows.append(s)
                    print(
                        f"  [{pat:20s}] trades={s['trades']:4d} WR={s['wr']:5.1f}% PF={s['pf']:5.2f} "
                        f"net={s['net_pct']:+6.2f}% MaxDD={s['max_dd_pct']:6.2f}%"
                    )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "candle_compare.csv", index=False)
    rdf.to_csv(FLAT / "ema_bos_rr2_candle_compare.csv", index=False)

    print("\n" + "=" * 90)
    print("CANDLESTICK FILTER COMPARISON")
    print("=" * 90)
    cols = ["symbol", "timeframe", "filter_mode", "trades", "wr", "pf", "net_pct", "max_dd_pct", "max_dd_usd"]
    print(rdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()
