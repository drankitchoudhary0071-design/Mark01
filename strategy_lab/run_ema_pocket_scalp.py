#!/usr/bin/env python3
"""
EMA Pocket Scalp — video-faithful backtest (1m & 5m).

Rules:
  Setup: 9 EMA, 21 EMA, EMA pocket, recent 1h S/R
  Long:  support touch + price in pocket + RSI>50 + candle close entry
  Short: resistance touch + pocket + red candle close + RSI<50
  Recovery: pierce S/R then recover to pocket within 5 bars → still valid
  Filters: London/NY only, 9 EMA angle ≥30°, no chase >5 pips from level
  Exit:   +10 pips → 50% off + SL breakeven; rest until next S/R or 21 EMA close
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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

# ── parameters ─────────────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
RSI_LEN = 14
ATR_LEN = 14

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

RECOVERY_BARS = 5
CHASE_PIPS = 5
PARTIAL_PIPS = 10
PARTIAL_FRAC = 0.5
SL_BUFFER_PIPS = 2

EMA_ANGLE_LOOKBACK = 8
MIN_EMA_ANGLE_DEG = 30.0

# London ∪ NY (UTC): skip Asian dead zone ~00–07
SESSION_START_H = 7
SESSION_END_H = 22

INITIAL_CAPITAL = 10_000.0
POSITION_PCT = 0.05  # 5% of equity per trade (compounding)
DAYS = 365

# Costs: Binance taker for PAXG; pip spread+slip for XAUUSD (OANDA-style)
CRYPTO_COSTS = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
XAU_SLIP_PIPS = 0.5  # adverse spread+slippage per side (~0.5 pip gold)

PIP_SIZE = {
    "XAUUSD": 0.10,
    "PAXGUSDT": 0.10,
    "BTCUSDT": 10.0,
}

OUT = Path(__file__).resolve().parent / "results" / "ema_pocket_scalp"
OUT.mkdir(parents=True, exist_ok=True)
FLAT = Path(__file__).resolve().parent / "results"


def entry_fill(raw: float, direction: str, symbol: str, pip: float) -> float:
    if symbol == "XAUUSD":
        slip = XAU_SLIP_PIPS * pip
        return raw + slip if direction == "long" else raw - slip
    return apply_entry_price(raw, direction, CRYPTO_COSTS)


def exit_fill(raw: float, direction: str, symbol: str, pip: float) -> float:
    if symbol == "XAUUSD":
        slip = XAU_SLIP_PIPS * pip
        return raw - slip if direction == "long" else raw + slip
    return apply_exit_price(raw, direction, CRYPTO_COSTS)


def leg_pnl(
    direction: str,
    entry_px: float,
    exit_raw: float,
    qty: float,
    symbol: str,
    pip: float,
) -> tuple[float, float]:
    """Return (net_pnl, exit_fill) for one closed leg."""
    xf = exit_fill(exit_raw, direction, symbol, pip)
    if direction == "long":
        gross = (xf - entry_px) * qty
    else:
        gross = (entry_px - xf) * qty
    fees = 0.0
    if symbol != "XAUUSD":
        fees = (entry_px + xf) * qty * CRYPTO_COSTS.commission_rate
    return gross - fees, xf


def position_size(capital: float, entry_px: float) -> tuple[float, float, float]:
    """Return (qty_total, qty_partial, qty_remainder) from compounding notional."""
    notional = capital * POSITION_PCT
    qty_total = notional / entry_px
    qty_part = qty_total * PARTIAL_FRAC
    qty_rem = qty_total * (1.0 - PARTIAL_FRAC)
    return qty_total, qty_part, qty_rem


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


def pivot_highs_lows(high: np.ndarray, low: np.ndarray, left: int, right: int):
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
    return ph, pl


def in_pocket(close: float, e9: float, e21: float, low: float | None = None, high: float | None = None) -> bool:
    """Price settled in EMA pocket — close inside, or bar overlaps the band."""
    if not (np.isfinite(e9) and np.isfinite(e21) and np.isfinite(close)):
        return False
    lo, hi = min(e9, e21), max(e9, e21)
    if lo <= close <= hi:
        return True
    if low is not None and high is not None:
        return low <= hi and high >= lo
    return False


def ema_angle_deg(ema9: np.ndarray, i: int, lookback: int, pip: float) -> float:
    """Visual-style EMA slope: pip move over ``lookback`` bars → degrees."""
    if i < lookback or pip <= 0:
        return 0.0
    delta_pips = (ema9[i] - ema9[i - lookback]) / pip
    return float(np.degrees(np.arctan(delta_pips / lookback)))


def session_ok(ts: pd.Timestamp) -> bool:
    h = ts.hour
    return SESSION_START_H <= h < SESSION_END_H


def lookback_bars(tf: str) -> int:
    return 60 if tf == "1m" else 12


@dataclass
class SRLevels:
    support: float
    resistance: float


def recent_sr(
    high: np.ndarray,
    low: np.ndarray,
    ph: np.ndarray,
    pl: np.ndarray,
    i: int,
    lb: int,
) -> SRLevels | None:
    start = max(0, i - lb)
    sup = np.nan
    res = np.nan
    for j in range(start, i + 1):
        if np.isfinite(pl[j]):
            sup = pl[j]
        if np.isfinite(ph[j]):
            res = ph[j]
    if not (np.isfinite(sup) and np.isfinite(res)):
        return None
    return SRLevels(support=float(sup), resistance=float(res))


def next_sr_target(
    high: np.ndarray,
    low: np.ndarray,
    ph: np.ndarray,
    pl: np.ndarray,
    i: int,
    direction: str,
    entry: float,
    lb: int,
) -> float | None:
    start = max(0, i - lb)
    if direction == "long":
        cands = []
        for j in range(start, i + 1):
            if np.isfinite(ph[j]) and ph[j] > entry:
                cands.append(ph[j])
        return float(min(cands)) if cands else None
    cands = []
    for j in range(start, i + 1):
        if np.isfinite(pl[j]) and pl[j] < entry:
            cands.append(pl[j])
    return float(max(cands)) if cands else None


def support_touch_valid(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    i: int,
    support: float,
    e9: float,
    e21: float,
    pip: float,
    recovery: int,
) -> bool:
    tol = pip * 0.5
    if in_pocket(close[i], e9, e21, low[i], high[i]) and low[i] <= support + tol:
        return True
    if not in_pocket(close[i], e9, e21, low[i], high[i]):
        return False
    start = max(0, i - recovery + 1)
    pierced = any(low[j] < support for j in range(start, i + 1))
    near = abs(close[i] - support) <= CHASE_PIPS * pip or low[i] <= support + tol
    return pierced and near


def resistance_touch_valid(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    i: int,
    resistance: float,
    e9: float,
    e21: float,
    pip: float,
    recovery: int,
) -> bool:
    tol = pip * 0.5
    bearish = close[i] < open_[i]
    if not bearish:
        return False
    if in_pocket(close[i], e9, e21, low[i], high[i]) and high[i] >= resistance - tol:
        return True
    if not in_pocket(close[i], e9, e21, low[i], high[i]):
        return False
    start = max(0, i - recovery + 1)
    pierced = any(high[j] > resistance for j in range(start, i + 1))
    near = abs(close[i] - resistance) <= CHASE_PIPS * pip or high[i] >= resistance - tol
    return pierced and near


def run_backtest(
    symbol: str,
    tf: str,
    *,
    partial_pips: float = PARTIAL_PIPS,
    df: pd.DataFrame | None = None,
) -> dict:
    pip = PIP_SIZE.get(symbol, 0.10)
    lb = lookback_bars(tf)
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
    ph, pl = pivot_highs_lows(h, l, PIVOT_LEFT, PIVOT_RIGHT)

    warmup = max(EMA_SLOW, RSI_LEN, lb, PIVOT_LEFT + PIVOT_RIGHT) + 5
    capital = INITIAL_CAPITAL
    trades: list[dict] = []
    equity = [capital]
    equity_ts = [ts.iloc[warmup]]
    total_fees = 0.0

    pos = None

    def record_exit(pos_dict, exit_raw, qty, exit_hit, bar_i, leg):
        nonlocal capital, total_fees
        if qty <= 0:
            return
        pnl, xf = leg_pnl(pos_dict["dir"], pos_dict["entry"], exit_raw, qty, symbol, pip)
        if symbol != "XAUUSD":
            fees = (pos_dict["entry"] + xf) * qty * CRYPTO_COSTS.commission_rate
            total_fees += fees
        capital += pnl
        trades.append(
            {
                "entry_time": pos_dict["entry_time"],
                "exit_time": ts.iloc[bar_i],
                "direction": pos_dict["dir"],
                "leg": leg,
                "entry_price": pos_dict["entry"],
                "exit_price": xf,
                "qty": qty,
                "notional_pct": POSITION_PCT,
                "equity_at_entry": pos_dict["equity_at_entry"],
                "pnl": float(pnl),
                "exit_hit": exit_hit,
            }
        )
        equity.append(capital)
        equity_ts.append(ts.iloc[bar_i])

    for i in range(warmup, len(df) - 1):
        if pos is not None:
            bar_h, bar_l, bar_c = h[i], l[i], c[i]
            entry = pos["entry"]
            qty_rem = pos["qty_rem"]
            sl = pos["sl"]
            direction = pos["dir"]
            partial_done = pos["partial_done"]

            if direction == "long":
                target_partial = entry + partial_pips * pip
                if not partial_done and bar_h >= target_partial:
                    record_exit(pos, target_partial, pos["qty_part"], f"partial_{int(partial_pips)}pip", i, "partial")
                    pos["partial_done"] = True
                    pos["sl"] = entry
                    sl = entry

                if bar_l <= sl:
                    if qty_rem > 0:
                        record_exit(pos, sl, qty_rem, "sl_be" if partial_done else "sl", i, "remainder")
                    pos = None
                    continue

                if np.isfinite(e21[i]) and bar_c < e21[i]:
                    if qty_rem > 0:
                        record_exit(pos, bar_c, qty_rem, "ema21_close", i, "remainder")
                    pos = None
                    continue

                tp_lvl = next_sr_target(h, l, ph, pl, i, "long", entry, lb * 4)
                if tp_lvl is not None and bar_h >= tp_lvl and qty_rem > 0:
                    record_exit(pos, tp_lvl, qty_rem, "next_resistance", i, "remainder")
                    pos = None
                    continue

            else:  # short
                target_partial = entry - partial_pips * pip
                if not partial_done and bar_l <= target_partial:
                    record_exit(pos, target_partial, pos["qty_part"], f"partial_{int(partial_pips)}pip", i, "partial")
                    pos["partial_done"] = True
                    pos["sl"] = entry
                    sl = entry

                if bar_h >= sl:
                    if qty_rem > 0:
                        record_exit(pos, sl, qty_rem, "sl_be" if partial_done else "sl", i, "remainder")
                    pos = None
                    continue

                if np.isfinite(e21[i]) and bar_c > e21[i]:
                    if qty_rem > 0:
                        record_exit(pos, bar_c, qty_rem, "ema21_close", i, "remainder")
                    pos = None
                    continue

                tp_lvl = next_sr_target(h, l, ph, pl, i, "short", entry, lb * 4)
                if tp_lvl is not None and bar_l <= tp_lvl and qty_rem > 0:
                    record_exit(pos, tp_lvl, qty_rem, "next_support", i, "remainder")
                    pos = None
                    continue

            continue

        if not session_ok(ts.iloc[i]):
            continue

        sr = recent_sr(h, l, ph, pl, i, lb)
        if sr is None:
            continue

        angle = ema_angle_deg(e9, i, EMA_ANGLE_LOOKBACK, pip)
        chase_sup = abs(c[i] - sr.support) <= CHASE_PIPS * pip
        chase_res = abs(c[i] - sr.resistance) <= CHASE_PIPS * pip

        if (
            support_touch_valid(l, h, c, i, sr.support, e9[i], e21[i], pip, RECOVERY_BARS)
            and chase_sup
            and angle >= MIN_EMA_ANGLE_DEG
            and np.isfinite(rsi[i])
            and rsi[i] > 50
        ):
            raw_entry = float(o[i + 1])
            fill = entry_fill(raw_entry, "long", symbol, pip)
            sl_raw = min(l[i], sr.support) - SL_BUFFER_PIPS * pip
            if sl_raw < fill:
                _, qty_part, qty_rem = position_size(capital, fill)
                pos = {
                    "dir": "long",
                    "entry": fill,
                    "sl": sl_raw,
                    "qty_part": qty_part,
                    "qty_rem": qty_rem,
                    "partial_done": False,
                    "entry_time": ts.iloc[i + 1],
                    "equity_at_entry": capital,
                }
            continue

        if (
            resistance_touch_valid(h, l, o, c, i, sr.resistance, e9[i], e21[i], pip, RECOVERY_BARS)
            and chase_res
            and angle <= -MIN_EMA_ANGLE_DEG
            and np.isfinite(rsi[i])
            and rsi[i] < 50
        ):
            raw_entry = float(o[i + 1])
            fill = entry_fill(raw_entry, "short", symbol, pip)
            sl_raw = max(h[i], sr.resistance) + SL_BUFFER_PIPS * pip
            if sl_raw > fill:
                _, qty_part, qty_rem = position_size(capital, fill)
                pos = {
                    "dir": "short",
                    "entry": fill,
                    "sl": sl_raw,
                    "qty_part": qty_part,
                    "qty_rem": qty_rem,
                    "partial_done": False,
                    "entry_time": ts.iloc[i + 1],
                    "equity_at_entry": capital,
                }

    if pos is not None:
        qty_rem = pos["qty_rem"]
        if qty_rem > 0:
            record_exit(pos, float(c[-1]), qty_rem, "eod", len(df) - 1, "remainder")

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=pd.to_datetime(equity_ts, utc=True))
    return {
        "symbol": symbol,
        "tf": tf,
        "trades": tdf,
        "equity": eq,
        "final": capital,
        "total_fees": total_fees,
        "partial_pips": partial_pips,
    }


def aggregate_trades(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return tdf
    g = (
        tdf.groupby(["entry_time", "direction"], as_index=False)
        .agg(
            exit_time=("exit_time", "max"),
            pnl=("pnl", "sum"),
            legs=("leg", lambda x: ",".join(sorted(set(x)))),
            exit_hit=("exit_hit", lambda x: "|".join(x)),
        )
    )
    return g


def summarize(label: str, tdf: pd.DataFrame, eq: pd.Series, final: float, fees: float = 0.0) -> str:
    agg = aggregate_trades(tdf)
    lines = ["=" * 90, f"RESULTS — {label}", "=" * 90]
    lines.append(
        f"Capital=${INITIAL_CAPITAL:,.0f} | Position={POSITION_PCT*100:.0f}%/trade (compounding) | "
        f"Slippage: XAU {XAU_SLIP_PIPS}pip/side, PAXG ~{CRYPTO_COSTS.round_trip_friction()*100:.2f}% RT"
    )
    if agg.empty:
        lines.append("No trades.")
        return "\n".join(lines)
    wins = agg[agg.pnl > 1e-8]
    losses = agg[agg.pnl < -1e-8]
    n = len(agg)
    wr = len(wins) / n * 100
    gp = float(wins.pnl.sum()) if len(wins) else 0.0
    gl = float(abs(losses.pnl.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    net = final - INITIAL_CAPITAL
    dd = eq - eq.cummax()
    lines += [
        f"Trades={n}  Wins={len(wins)}  Losses={len(losses)}  WR={wr:.1f}%",
        f"PF={pf:.3f}  GP=${gp:.2f}  GL=${gl:.2f}",
        f"Net=${net:.2f} ({net/INITIAL_CAPITAL*100:+.2f}%)  Final=${final:.2f}",
        f"MaxDD=${float(dd.min()):.2f} ({float(dd.min())/INITIAL_CAPITAL*100:.2f}%)",
        f"Total commission paid=${fees:.2f}",
        f"Leg exits: {tdf.exit_hit.value_counts().to_dict() if not tdf.empty else {}}",
    ]
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
    print(
        f"EMA Pocket Scalp — compounding {POSITION_PCT*100:.0f}%/trade + slippage "
        f"(XAUUSD OANDA-proxy + PAXG Binance)"
    )
    rows = []
    texts = []
    for sym in ("XAUUSD", "PAXGUSDT"):
        for tf in ("1m", "5m"):
            print(f"\n>>> {sym} {tf}")
            out = run_backtest(sym, tf)
            tdf, eq, final, fees = out["trades"], out["equity"], out["final"], out["total_fees"]
            label = f"{sym} {tf}"
            text = summarize(label, tdf, eq, final, fees)
            print(text)
            texts.append(text)
            net = final - INITIAL_CAPITAL
            net_pct = net / INITIAL_CAPITAL * 100
            stem = f"{sym.lower()}_{tf}_compound"
            agg = aggregate_trades(tdf)
            if not tdf.empty:
                tdf.to_csv(OUT / f"{stem}_legs.csv", index=False)
                agg.to_csv(OUT / f"{stem}_trades.csv", index=False)
                agg.to_csv(FLAT / f"ema_pocket_{stem}_trades.csv", index=False)
            eq.to_csv(OUT / f"{stem}_equity.csv", header=["equity"])
            plot_eq(label, eq, OUT / f"{stem}_equity.png", net_pct)
            plot_eq(label, eq, FLAT / f"ema_pocket_{stem}_equity.png", net_pct)
            rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "position_pct": POSITION_PCT,
                    "trades": len(agg),
                    "wins": int((agg.pnl > 0).sum()) if len(agg) else 0,
                    "losses": int((agg.pnl < 0).sum()) if len(agg) else 0,
                    "wr": float((agg.pnl > 0).mean() * 100) if len(agg) else 0.0,
                    "pf": (
                        float(agg.loc[agg.pnl > 0, "pnl"].sum() / abs(agg.loc[agg.pnl < 0, "pnl"].sum()))
                        if len(agg) and (agg.pnl < 0).any() and (agg.pnl > 0).any()
                        else float("nan")
                    ),
                    "net": net,
                    "net_pct": net_pct,
                    "final_capital": final,
                    "total_fees": fees,
                    "max_dd_pct": float((eq - eq.cummax()).min() / INITIAL_CAPITAL * 100) if len(eq) else 0.0,
                }
            )
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "summary_compound.csv", index=False)
    rdf.to_csv(FLAT / "ema_pocket_scalp_compound_summary.csv", index=False)
    (OUT / "report_compound.txt").write_text("\n\n".join(texts) + "\n")
    (FLAT / "ema_pocket_scalp_compound_report.txt").write_text("\n\n".join(texts) + "\n")
    meta = {
        "initial_capital": INITIAL_CAPITAL,
        "position_pct": POSITION_PCT,
        "compounding": True,
        "xau_slip_pips_per_side": XAU_SLIP_PIPS,
        "crypto_costs": {
            "commission_rate": CRYPTO_COSTS.commission_rate,
            "half_spread": CRYPTO_COSTS.half_spread,
            "slippage": CRYPTO_COSTS.slippage,
            "round_trip_pct": CRYPTO_COSTS.round_trip_friction() * 100,
        },
        "ema_fast": EMA_FAST,
        "ema_slow": EMA_SLOW,
        "rsi_len": RSI_LEN,
        "recovery_bars": RECOVERY_BARS,
        "partial_pips": PARTIAL_PIPS,
        "chase_pips": CHASE_PIPS,
        "min_ema_angle_deg": MIN_EMA_ANGLE_DEG,
        "session_utc": f"{SESSION_START_H}:00-{SESSION_END_H}:00",
        "pip_sizes": PIP_SIZE,
    }
    (OUT / "params_compound.json").write_text(json.dumps(meta, indent=2))
    (FLAT / "ema_pocket_scalp_compound_params.json").write_text(json.dumps(meta, indent=2))
    print("\nSUMMARY (compounding + slippage)")
    print(rdf.to_string(index=False))
    print(f"Artifacts → {OUT}")


if __name__ == "__main__":
    main()
