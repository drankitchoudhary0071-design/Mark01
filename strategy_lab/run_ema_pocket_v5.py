#!/usr/bin/env python3
"""
Triple 9 EMA Pocket Scalp v5 — Pine Script port (time-normalized EMA21 angle).

Matches TradingView strategy:
  - EMA21 angle (not EMA9), 60-min lookback auto-scaled to bars
  - Pivot S/R (len=5), chase 8 pips, partial 10 pips, SL buffer 3 pips
  - Recovery pierce within 5 bars, session 07:00-22:00 UTC
  - 100% equity, 0.02% commission, 1-tick slippage
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

# ── Pine v5 defaults ───────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
RSI_LEN = 14
ATR_LEN = 14
PIVOT_LEN = 5

PIP_SIZE = 0.10
CHASE_MAX_PIPS = 8.0
PARTIAL_PIPS = 10.0
MIN_ANGLE_DEG = 10.0
ANGLE_LOOKBACK_MIN = 60.0  # minutes
RECOVERY_BARS = 5
SL_BUFFER_PIPS = 3.0
PARTIAL_FRAC = 0.5

SESSION_START = 7 * 60  # 07:00 UTC minutes
SESSION_END = 22 * 60  # 22:00 UTC

INITIAL_CAPITAL = 10_000.0
POSITION_PCT = 1.0  # Pine: 100% of equity
COMMISSION_RATE = 0.0002  # 0.02%
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.01  # 1 TV tick for gold

DAYS = 365
OUT = Path(__file__).resolve().parent / "results" / "ema_pocket_v5"
FLAT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


def rsi_wilder(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.mean(gain[:period])
    avg_l = np.mean(loss[:period])
    out[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l > 0 else 100.0
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gain[i - 1]) / period
        avg_l = (avg_l * (period - 1) + loss[i - 1]) / period
        out[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l > 0 else 100.0
    return out


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
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


def pivot_series(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Return running last pivot high/low (Pine var resistance/support)."""
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
    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    last_r, last_s = np.nan, np.nan
    for i in range(n):
        if np.isfinite(ph[i]):
            last_r = ph[i]
        if np.isfinite(pl[i]):
            last_s = pl[i]
        res[i] = last_r
        sup[i] = last_s
    return res, sup


def in_session(ts: pd.Timestamp) -> bool:
    mins = ts.hour * 60 + ts.minute
    return SESSION_START <= mins < SESSION_END


def minutes_per_bar(tf: str) -> float:
    return 1.0 if tf == "1m" else 5.0


def fill_entry(raw: float, direction: str) -> float:
    slip = SLIPPAGE_TICKS * TICK_SIZE
    return raw + slip if direction == "long" else raw - slip


def fill_exit(raw: float, direction: str) -> float:
    slip = SLIPPAGE_TICKS * TICK_SIZE
    return raw - slip if direction == "long" else raw + slip


def leg_pnl(direction: str, entry: float, exit_raw: float, qty: float) -> float:
    xf = fill_exit(exit_raw, direction)
    if direction == "long":
        gross = (xf - entry) * qty
    else:
        gross = (entry - xf) * qty
    fees = (entry + xf) * qty * COMMISSION_RATE
    return gross - fees


def run_backtest(symbol: str, tf: str, df: pd.DataFrame | None = None) -> dict:
    pip = PIP_SIZE
    if df is None:
        df = load_or_fetch(symbol, tf, DAYS)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    e9 = ema(c, EMA_FAST)
    e21 = ema(c, EMA_SLOW)
    rsi = rsi_wilder(c, RSI_LEN)
    atr = atr_wilder(h, l, c, ATR_LEN)
    resistance, support = pivot_series(h, l, PIVOT_LEN, PIVOT_LEN)

    mpb = minutes_per_bar(tf)
    angle_lb = max(1, int(round(ANGLE_LOOKBACK_MIN / mpb)))

    warmup = max(EMA_SLOW, RSI_LEN, ATR_LEN, PIVOT_LEN * 2, angle_lb) + 5
    capital = INITIAL_CAPITAL
    trades: list[dict] = []
    equity = [capital]
    equity_ts = [ts.iloc[warmup]]

    pos = None
    pierced_low_bar: int | None = None
    pierced_high_bar: int | None = None

    def close_leg(bar_i, exit_raw, qty, leg, exit_hit):
        nonlocal capital, pos
        if qty <= 0:
            return
        pnl = leg_pnl(pos["dir"], pos["entry"], exit_raw, qty)
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": ts.iloc[bar_i],
                "direction": pos["dir"],
                "leg": leg,
                "pnl": pnl,
                "exit_hit": exit_hit,
            }
        )
        equity.append(capital)
        equity_ts.append(ts.iloc[bar_i])

    for i in range(warmup, len(df)):
        pocket_hi = max(e9[i], e21[i]) if np.isfinite(e9[i]) and np.isfinite(e21[i]) else np.nan
        pocket_lo = min(e9[i], e21[i]) if np.isfinite(e9[i]) and np.isfinite(e21[i]) else np.nan
        in_pocket = np.isfinite(pocket_hi) and pocket_lo <= c[i] <= pocket_hi

        sup = support[i]
        res = resistance[i]
        if np.isfinite(sup) and l[i] < sup - pip:
            pierced_low_bar = i
        if np.isfinite(res) and h[i] > res + pip:
            pierced_high_bar = i

        if pos is not None:
            direction = pos["dir"]
            entry = pos["entry"]
            init_sl = pos["init_sl"]
            partial_done = pos["partial_done"]
            qty_part = pos["qty_part"]
            qty_rem = pos["qty_rem"]

            if direction == "long":
                tp1 = entry + PARTIAL_PIPS * pip
                active_sl = entry if partial_done else init_sl

                # init SL hits full position (before partial)
                if not partial_done and np.isfinite(init_sl) and l[i] <= init_sl:
                    close_leg(i, init_sl, qty_part + qty_rem, "full", "init_sl")
                    pos = None
                    continue

                if not partial_done and h[i] >= tp1:
                    close_leg(i, tp1, qty_part, "partial", "partial_tp")
                    pos["partial_done"] = True

                if partial_done and l[i] <= entry:
                    close_leg(i, entry, qty_rem, "remainder", "breakeven")
                    pos = None
                    continue

                if np.isfinite(e21[i]) and c[i] < e21[i] and qty_rem > 0:
                    close_leg(i, c[i], qty_rem, "remainder", "ema21_close")
                    pos = None
                    continue

                if np.isfinite(res) and h[i] >= res and qty_rem > 0:
                    close_leg(i, res, qty_rem, "remainder", "runner_resistance")
                    pos = None
                    continue

            else:  # short
                tp1 = entry - PARTIAL_PIPS * pip
                if not partial_done and np.isfinite(init_sl) and h[i] >= init_sl:
                    close_leg(i, init_sl, qty_part + qty_rem, "full", "init_sl")
                    pos = None
                    continue

                if not partial_done and l[i] <= tp1:
                    close_leg(i, tp1, qty_part, "partial", "partial_tp")
                    pos["partial_done"] = True

                if partial_done and h[i] >= entry:
                    close_leg(i, entry, qty_rem, "remainder", "breakeven")
                    pos = None
                    continue

                if np.isfinite(e21[i]) and c[i] > e21[i] and qty_rem > 0:
                    close_leg(i, c[i], qty_rem, "remainder", "ema21_close")
                    pos = None
                    continue

                if np.isfinite(sup) and l[i] <= sup and qty_rem > 0:
                    close_leg(i, sup, qty_rem, "remainder", "runner_support")
                    pos = None
                    continue

            continue

        # flat — evaluate entry at bar close, fill next open
        if i >= len(df) - 1:
            continue

        if not in_session(ts.iloc[i]):
            continue

        if not (np.isfinite(atr[i]) and atr[i] > 0 and i >= angle_lb):
            continue

        ema_slope = e21[i] - e21[i - angle_lb]
        angle_deg = float(np.degrees(np.arctan(ema_slope / (atr[i] * angle_lb))))
        angle_up = angle_deg >= MIN_ANGLE_DEG
        angle_down = angle_deg <= -MIN_ANGLE_DEG

        near_sup = np.isfinite(sup) and abs(l[i] - sup) <= CHASE_MAX_PIPS * pip
        near_res = np.isfinite(res) and abs(h[i] - res) <= CHASE_MAX_PIPS * pip

        recovered_long = (
            pierced_low_bar is not None
            and (i - pierced_low_bar) <= RECOVERY_BARS
            and in_pocket
        )
        recovered_short = (
            pierced_high_bar is not None
            and (i - pierced_high_bar) <= RECOVERY_BARS
            and in_pocket
        )

        long_touch = near_sup or recovered_long
        short_touch = near_res or recovered_short
        candle_red = c[i] < o[i]

        long_cond = long_touch and in_pocket and rsi[i] > 50 and angle_up
        short_cond = short_touch and in_pocket and candle_red and rsi[i] < 50 and angle_down

        if long_cond:
            raw = float(o[i + 1])
            fill = fill_entry(raw, "long")
            init_sl = sup - SL_BUFFER_PIPS * pip if np.isfinite(sup) else np.nan
            notional = capital * POSITION_PCT
            qty_total = notional / fill
            pos = {
                "dir": "long",
                "entry": fill,
                "init_sl": init_sl,
                "qty_part": qty_total * PARTIAL_FRAC,
                "qty_rem": qty_total * (1 - PARTIAL_FRAC),
                "partial_done": False,
                "entry_time": ts.iloc[i + 1],
            }
            pierced_low_bar = None
        elif short_cond:
            raw = float(o[i + 1])
            fill = fill_entry(raw, "short")
            init_sl = res + SL_BUFFER_PIPS * pip if np.isfinite(res) else np.nan
            notional = capital * POSITION_PCT
            qty_total = notional / fill
            pos = {
                "dir": "short",
                "entry": fill,
                "init_sl": init_sl,
                "qty_part": qty_total * PARTIAL_FRAC,
                "qty_rem": qty_total * (1 - PARTIAL_FRAC),
                "partial_done": False,
                "entry_time": ts.iloc[i + 1],
            }
            pierced_high_bar = None

    if pos is not None and pos["qty_rem"] > 0:
        close_leg(len(df) - 1, c[-1], pos["qty_rem"], "remainder", "eod")

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {"symbol": symbol, "tf": tf, "trades": tdf, "equity": eq, "final": capital}


def aggregate_trades(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return tdf
    return (
        tdf.groupby(["entry_time", "direction"], as_index=False)
        .agg(
            exit_time=("exit_time", "max"),
            pnl=("pnl", "sum"),
            exit_hit=("exit_hit", lambda x: "|".join(x)),
        )
    )


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
    print("EMA Pocket Scalp v5 (Pine port) — 1m & 5m | 100% equity, 0.02% comm, 1-tick slip")
    rows = []
    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            print(f"\n>>> {sym} {tf}")
            out = run_backtest(sym, tf)
            tdf, eq, final = out["trades"], out["equity"], out["final"]
            agg = aggregate_trades(tdf)
            net = final - INITIAL_CAPITAL
            net_pct = net / INITIAL_CAPITAL * 100
            n = len(agg)
            wr = float((agg.pnl > 0).mean() * 100) if n else 0.0
            pf = (
                float(agg.loc[agg.pnl > 0, "pnl"].sum() / abs(agg.loc[agg.pnl < 0, "pnl"].sum()))
                if n and (agg.pnl < 0).any() and (agg.pnl > 0).any()
                else float("nan")
            )
            dd = float((eq - eq.cummax()).min() / INITIAL_CAPITAL * 100) if len(eq) else 0.0
            print(
                f"Trades={n} WR={wr:.1f}% PF={pf:.2f} Net=${net:.2f} ({net_pct:+.2f}%) "
                f"Final=${final:.2f} MaxDD={dd:.2f}%"
            )
            if not tdf.empty:
                print(f"Exits: {tdf.exit_hit.value_counts().to_dict()}")
            stem = f"{sym.lower()}_{tf}"
            tdf.to_csv(OUT / f"{stem}_legs.csv", index=False)
            agg.to_csv(OUT / f"{stem}_trades.csv", index=False)
            agg.to_csv(FLAT / f"ema_pocket_v5_{stem}_trades.csv", index=False)
            eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
            plot_eq(f"{sym} {tf} v5", eq, OUT / f"{stem}_equity.png", net_pct)
            plot_eq(f"{sym} {tf} v5", eq, FLAT / f"ema_pocket_v5_{stem}_equity.png", net_pct)
            rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "trades": n,
                    "wins": int((agg.pnl > 0).sum()) if n else 0,
                    "wr": wr,
                    "pf": pf,
                    "net": net,
                    "net_pct": net_pct,
                    "final_capital": final,
                    "max_dd_pct": dd,
                }
            )
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "summary.csv", index=False)
    rdf.to_csv(FLAT / "ema_pocket_v5_summary.csv", index=False)
    (OUT / "params.json").write_text(
        json.dumps(
            {
                "version": "v5_pine_port",
                "ema21_angle": True,
                "angle_lookback_min": ANGLE_LOOKBACK_MIN,
                "min_angle_deg": MIN_ANGLE_DEG,
                "pivot_len": PIVOT_LEN,
                "chase_pips": CHASE_MAX_PIPS,
                "partial_pips": PARTIAL_PIPS,
                "position_pct": POSITION_PCT,
                "commission": COMMISSION_RATE,
            },
            indent=2,
        )
    )
    print("\nSUMMARY")
    print(rdf.to_string(index=False))


if __name__ == "__main__":
    main()
