#!/usr/bin/env python3
"""Optimize UT Bot + MACD + Range Filter for profitability after costs.

Search params on 15m/30m/1h for BTC & PAXG.
Train = first 60% of bars, OOS = last 40%.
Require: OOS profitable, PF>1.1, n>=15, MaxDD > -25%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics
from strategy_lab.engine import indicators as ind
from strategy_lab.engine.types import Signal
from strategy_lab.strategies.base import atr_stop_tp
from strategy_lab.strategies.mtf_rsi import resample_ohlcv
from strategy_lab.strategies.ut_bot_macd_range import ut_bot_trail, range_filter

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 10_000.0
DAYS = 365
COSTS = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)


def load_tf(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "30m":
        return resample_ohlcv(load_or_fetch(symbol, "15m", DAYS), "30min")
    if interval == "1h":
        return load_or_fetch(symbol, "1h", DAYS)
    return load_or_fetch(symbol, "15m", DAYS)


def gen_signals(df: pd.DataFrame, p: dict) -> list[Signal]:
    close = df["close"].astype(float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()

    atr_buy = ind.atr(df, int(p["ut_buy_atr"]))
    atr_sell = ind.atr(df, int(p["ut_sell_atr"]))
    trail_buy = ut_bot_trail(close, atr_buy, float(p["ut_buy_key"]))
    trail_sell = ut_bot_trail(close, atr_sell, float(p["ut_sell_key"]))

    c = close.to_numpy(dtype=float)
    tb = trail_buy.to_numpy(dtype=float)
    tsell = trail_sell.to_numpy(dtype=float)
    buy_cross = (c > tb) & (np.roll(c, 1) <= np.roll(tb, 1))
    sell_cross = (c < tsell) & (np.roll(c, 1) >= np.roll(tsell, 1))
    buy_cross[0] = False
    sell_cross[0] = False

    ml, ms, _ = ind.macd(close, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    ml_a = ml.to_numpy(dtype=float)
    ms_a = ms.to_numpy(dtype=float)
    if p["entry_mode"] == "macd_cross":
        macd_bull = (ml_a > ms_a) & (np.roll(ml_a, 1) <= np.roll(ms_a, 1))
        macd_bear = (ml_a < ms_a) & (np.roll(ml_a, 1) >= np.roll(ms_a, 1))
        macd_bull[0] = False
        macd_bear[0] = False
    else:
        macd_bull = ml_a > ms_a
        macd_bear = ml_a < ms_a

    rf = range_filter(close, int(p["range_period"]), float(p["range_mult"])).to_numpy(dtype=float)
    if p.get("rf_slope", False):
        rf_up = rf > np.roll(rf, 1)
        rf_dn = rf < np.roll(rf, 1)
        rf_up[0] = False
        rf_dn[0] = False
        above = (c > rf) & rf_up
        below = (c < rf) & rf_dn
    else:
        above = c > rf
        below = c < rf

    long_ok = buy_cross & macd_bull & above
    short_ok = sell_cross & macd_bear & below

    atr_s = ind.atr(df, int(p["atr_period"])).to_numpy(dtype=float)
    hold = int(p["max_hold_bars"])
    cooldown = int(p.get("cooldown", 0))
    stop_m = float(p["stop_atr"])
    tp_m = float(p["tp_atr"])

    signals: list[Signal] = []
    in_pos = False
    pos_side = None
    pos_stop = pos_tp = 0.0
    entry_i = -1
    last_exit = -10_000
    n = len(df)

    for i in range(n):
        if in_pos:
            exited = False
            if pos_side == "long" and (low[i] <= pos_stop or high[i] >= pos_tp):
                exited = True
            elif pos_side == "short" and (high[i] >= pos_stop or low[i] <= pos_tp):
                exited = True
            if i - entry_i >= hold:
                exited = True
            if exited:
                in_pos = False
                pos_side = None
                last_exit = i
            else:
                continue

        if i - last_exit < cooldown:
            continue
        a = atr_s[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(c[i]):
            continue
        px = float(c[i])
        tsi = pd.Timestamp(ts[i])
        if long_ok[i]:
            stop, tp = atr_stop_tp(px, "long", float(a), stop_m, tp_m)
            signals.append(
                Signal(tsi, "long", px, stop, tp, pattern="ut_opt_long", max_hold_bars=hold)
            )
            in_pos, pos_side, pos_stop, pos_tp, entry_i = True, "long", stop, tp, i
        elif short_ok[i]:
            stop, tp = atr_stop_tp(px, "short", float(a), stop_m, tp_m)
            signals.append(
                Signal(tsi, "short", px, stop, tp, pattern="ut_opt_short", max_hold_bars=hold)
            )
            in_pos, pos_side, pos_stop, pos_tp, entry_i = True, "short", stop, tp, i
    return signals


def eval_slice(df: pd.DataFrame, p: dict) -> dict:
    sigs = gen_signals(df, p)
    res = run_backtest(
        df,
        sigs,
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=COSTS,
        max_hold_bars=int(p["max_hold_bars"]),
        strategy_name="ut_opt",
        symbol="",
    )
    m = compute_metrics(res)
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    return {
        "n": m["total_trades"],
        "wins": wins,
        "losses": losses,
        "wr": m["win_rate"],
        "pf": m["profit_factor"],
        "dd": m["max_drawdown"],
        "ret": m["total_return"],
        "expR": m["expectancy_R"],
        "eq": m["final_equity"],
        "pnl": m["final_equity"] - CAPITAL,
    }


def split_df(df: pd.DataFrame, frac: float = 0.6):
    cut = int(len(df) * frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def param_grid(interval: str) -> list[dict]:
    hold = {"15m": 96, "30m": 64, "1h": 48}[interval]
    # Focused grid: wider UT keys (fewer crosses), stricter entries, better RR
    combos = []
    for bk, sk in ((2.5, 2.5), (3.0, 3.0), (3.5, 3.5), (2.5, 3.0), (3.0, 3.5)):
        for ba, sa in ((20, 20), (30, 30), (50, 40), (50, 50)):
            for mf, ms, mg in ((12, 26, 9), (8, 21, 9), (5, 35, 5)):
                for rp, rm in ((20, 1.5), (30, 2.0), (50, 2.0)):
                    for st, tp in ((2.0, 4.0), (2.0, 5.0), (2.5, 5.0), (1.5, 4.0)):
                        for mode in ("macd_cross",):
                            for cd in ((10, 5, 0) if interval == "15m" else (6, 3, 0)):
                                for rfs in (True, False):
                                    combos.append(
                                        {
                                            "ut_buy_key": bk,
                                            "ut_sell_key": sk,
                                            "ut_buy_atr": ba,
                                            "ut_sell_atr": sa,
                                            "macd_fast": mf,
                                            "macd_slow": ms,
                                            "macd_signal": mg,
                                            "range_period": rp,
                                            "range_mult": rm,
                                            "atr_period": 14,
                                            "stop_atr": st,
                                            "tp_atr": tp,
                                            "entry_mode": mode,
                                            "cooldown": cd,
                                            "rf_slope": rfs,
                                            "max_hold_bars": hold,
                                        }
                                    )
    # Subsample deterministically if still huge
    if len(combos) > 400:
        combos = combos[:: max(1, len(combos) // 350)]
    return combos


def score(oos: dict) -> float:
    if oos["n"] < 10:
        return -1e9
    return (
        oos["ret"] * 100
        + max(oos["pf"] - 1, 0) * 15
        + oos["wr"] * 10
        + oos["dd"] * 40
        + min(oos["n"], 80) * 0.05
        + oos["expR"] * 20
    )


def passes(oos: dict) -> bool:
    return (
        oos["n"] >= 15
        and oos["ret"] > 0.02
        and oos["pf"] > 1.15
        and oos["dd"] >= -0.25
        and oos["expR"] > 0
    )


def main() -> int:
    symbols = ("BTCUSDT", "PAXGUSDT")
    intervals = ("15m", "30m", "1h")
    all_hits = []
    lines = [
        "UT+MACD+RF OPTIMIZATION | $10k | costs~0.30% RT | risk 1% | 365d",
        "Train 60% / OOS 40% | goal: OOS Ret>2%, PF>1.15, n>=15, MaxDD>=-25%",
        "",
    ]

    for sym in symbols:
        for iv in intervals:
            grid = param_grid(iv)
            df = load_tf(sym, iv)
            train, oos = split_df(df, 0.6)
            print(f"\n=== {sym} {iv} | {len(grid)} configs | bars={len(df)} ===")
            best_train = []
            for i, p in enumerate(grid, 1):
                try:
                    tr = eval_slice(train, p)
                except Exception:
                    continue
                # Pre-filter: train must show some edge
                if tr["n"] < 12 or tr["pf"] < 1.05 or tr["ret"] <= 0 or tr["expR"] <= 0:
                    continue
                best_train.append((tr, p))
                if i % 200 == 0:
                    print(f"  scanned {i}/{len(grid)} train-survivors={len(best_train)}")

            # Rank train survivors, evaluate OOS on top N
            best_train.sort(key=lambda x: score(x[0]), reverse=True)
            top = best_train[:80]
            print(f"  train survivors={len(best_train)} → OOS test top {len(top)}")

            hits = []
            for tr, p in top:
                oo = eval_slice(oos, p)
                row_base = {
                    "symbol": sym,
                    "interval": iv,
                    **{k: p[k] for k in p},
                    "train_n": tr["n"],
                    "train_ret": tr["ret"],
                    "train_pf": tr["pf"],
                    "train_dd": tr["dd"],
                    "oos_n": oo["n"],
                    "oos_wins": oo["wins"],
                    "oos_losses": oo["losses"],
                    "oos_wr": oo["wr"],
                    "oos_ret": oo["ret"],
                    "oos_pf": oo["pf"],
                    "oos_dd": oo["dd"],
                    "oos_expR": oo["expR"],
                    "oos_pnl": oo["pnl"],
                }
                if passes(oo):
                    full = eval_slice(df, p)
                    row_base.update(
                        {
                            "full_n": full["n"],
                            "full_wr": full["wr"],
                            "full_ret": full["ret"],
                            "full_pf": full["pf"],
                            "full_dd": full["dd"],
                            "full_pnl": full["pnl"],
                            "full_eq": full["eq"],
                        }
                    )
                    hits.append(row_base)
                    all_hits.append(row_base)

            hits.sort(key=lambda r: score({
                "n": r["oos_n"], "ret": r["oos_ret"], "pf": r["oos_pf"],
                "wr": r["oos_wr"], "dd": r["oos_dd"], "expR": r["oos_expR"],
            }), reverse=True)

            lines.append(f"{sym} {iv}: OOS profitable hits = {len(hits)}")
            for r in hits[:5]:
                lines.append(
                    f"  OOS n={r['oos_n']} WR={r['oos_wr']*100:.1f}% PF={r['oos_pf']:.2f} "
                    f"DD={r['oos_dd']*100:.1f}% Ret={r['oos_ret']*100:+.1f}% PnL=${r['oos_pnl']:+.0f} | "
                    f"FULL n={r['full_n']} Ret={r['full_ret']*100:+.1f}% Eq=${r['full_eq']:.0f} | "
                    f"UT buy {r['ut_buy_key']}/{r['ut_buy_atr']} sell {r['ut_sell_key']}/{r['ut_sell_atr']} "
                    f"MACD {r['macd_fast']}/{r['macd_slow']}/{r['macd_signal']} "
                    f"RF {r['range_period']}/{r['range_mult']} "
                    f"SL{r['stop_atr']}/TP{r['tp_atr']} mode={r['entry_mode']} "
                    f"cd={r['cooldown']} rfSlope={r['rf_slope']}"
                )
            if not hits:
                lines.append("  (none)")
            lines.append("")
            print(f"  OOS hits: {len(hits)}")

    OUT.mkdir(parents=True, exist_ok=True)
    all_hits.sort(key=lambda r: r["oos_ret"], reverse=True)
    pd.DataFrame(all_hits).to_csv(OUT / "ut_bot_macd_range_opt_hits.csv", index=False)

    lines += ["BEST OVERALL (by OOS return)", "-" * 100]
    if all_hits:
        for r in all_hits[:15]:
            lines.append(
                f"{r['symbol']} {r['interval']} | OOS Ret={r['oos_ret']*100:+.1f}% "
                f"n={r['oos_n']} WR={r['oos_wr']*100:.1f}% PF={r['oos_pf']:.2f} DD={r['oos_dd']*100:.1f}% | "
                f"FULL Ret={r['full_ret']*100:+.1f}% n={r['full_n']} Eq=${r['full_eq']:.0f} | "
                f"UT {r['ut_buy_key']}/{r['ut_buy_atr']} & {r['ut_sell_key']}/{r['ut_sell_atr']} "
                f"MACD {r['macd_fast']}/{r['macd_slow']}/{r['macd_signal']} "
                f"RF {r['range_period']}/{r['range_mult']} SL{r['stop_atr']} TP{r['tp_atr']} "
                f"{r['entry_mode']} cd={r['cooldown']} slope={r['rf_slope']}"
            )
        best = all_hits[0]
        lines += [
            "",
            "RECOMMENDED",
            f"  {best['symbol']} {best['interval']}",
            f"  UT buy {best['ut_buy_key']}/{best['ut_buy_atr']} | sell {best['ut_sell_key']}/{best['ut_sell_atr']}",
            f"  MACD {best['macd_fast']}/{best['macd_slow']}/{best['macd_signal']}",
            f"  Range {best['range_period']}/{best['range_mult']} | slope={best['rf_slope']}",
            f"  SL {best['stop_atr']}×ATR | TP {best['tp_atr']}×ATR | mode={best['entry_mode']} | cooldown={best['cooldown']}",
            f"  OOS: n={best['oos_n']} W/L={best['oos_wins']}/{best['oos_losses']} "
            f"WR={best['oos_wr']*100:.1f}% PF={best['oos_pf']:.2f} DD={best['oos_dd']*100:.1f}% "
            f"Ret={best['oos_ret']*100:+.1f}% PnL=${best['oos_pnl']:+.0f}",
            f"  FULL year: n={best['full_n']} WR={best['full_wr']*100:.1f}% "
            f"Ret={best['full_ret']*100:+.1f}% MaxDD={best['full_dd']*100:.1f}% Equity=${best['full_eq']:.0f}",
        ]
    else:
        lines.append("No OOS-profitable config found under constraints.")
        lines.append("Trying relaxed search note: see soft hits in CSV if any.")

    text = "\n".join(lines) + "\n"
    (OUT / "ut_bot_macd_range_opt_report.txt").write_text(text)
    (OUT / "ut_bot_macd_range_opt_hits.json").write_text(json.dumps(all_hits[:50], indent=2, default=str))
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
