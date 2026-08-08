#!/usr/bin/env python3
"""Fast focused search to make UT+MACD+RF profitable after costs."""

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
COSTS = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)


def load_tf(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "30m":
        return resample_ohlcv(load_or_fetch(symbol, "15m", 365), "30min")
    if interval == "1h":
        return load_or_fetch(symbol, "1h", 365)
    return load_or_fetch(symbol, "15m", 365)


def gen_signals(df: pd.DataFrame, p: dict) -> list[Signal]:
    close = df["close"].astype(float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
    c = close.to_numpy(dtype=float)

    tb = ut_bot_trail(close, ind.atr(df, int(p["ut_buy_atr"])), float(p["ut_buy_key"])).to_numpy()
    tsell = ut_bot_trail(close, ind.atr(df, int(p["ut_sell_atr"])), float(p["ut_sell_key"])).to_numpy()
    buy_cross = (c > tb) & (np.roll(c, 1) <= np.roll(tb, 1))
    sell_cross = (c < tsell) & (np.roll(c, 1) >= np.roll(tsell, 1))
    buy_cross[0] = sell_cross[0] = False

    ml, ms, _ = ind.macd(close, int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    ml_a, ms_a = ml.to_numpy(), ms.to_numpy()
    if p["entry_mode"] == "macd_cross":
        macd_bull = (ml_a > ms_a) & (np.roll(ml_a, 1) <= np.roll(ms_a, 1))
        macd_bear = (ml_a < ms_a) & (np.roll(ml_a, 1) >= np.roll(ms_a, 1))
        macd_bull[0] = macd_bear[0] = False
    else:
        macd_bull = ml_a > ms_a
        macd_bear = ml_a < ms_a

    rf = range_filter(close, int(p["range_period"]), float(p["range_mult"])).to_numpy()
    if p.get("rf_slope"):
        above = (c > rf) & (rf > np.roll(rf, 1))
        below = (c < rf) & (rf < np.roll(rf, 1))
        above[0] = below[0] = False
    else:
        above = c > rf
        below = c < rf

    long_ok = buy_cross & macd_bull & above
    short_ok = sell_cross & macd_bear & below
    atr_s = ind.atr(df, 14).to_numpy()
    hold = int(p["max_hold_bars"])
    cd = int(p.get("cooldown", 0))
    stop_m, tp_m = float(p["stop_atr"]), float(p["tp_atr"])

    signals: list[Signal] = []
    in_pos = False
    side = None
    stop = tp = 0.0
    ei = last_exit = -10_000
    for i in range(len(df)):
        if in_pos:
            ex = False
            if side == "long" and (low[i] <= stop or high[i] >= tp):
                ex = True
            elif side == "short" and (high[i] >= stop or low[i] <= tp):
                ex = True
            if i - ei >= hold:
                ex = True
            if ex:
                in_pos = False
                side = None
                last_exit = i
            else:
                continue
        if i - last_exit < cd:
            continue
        a = atr_s[i]
        if not np.isfinite(a) or a <= 0:
            continue
        px = float(c[i])
        tsi = pd.Timestamp(ts[i])
        if long_ok[i]:
            s, t = atr_stop_tp(px, "long", float(a), stop_m, tp_m)
            signals.append(Signal(tsi, "long", px, s, t, pattern="L", max_hold_bars=hold))
            in_pos, side, stop, tp, ei = True, "long", s, t, i
        elif short_ok[i]:
            s, t = atr_stop_tp(px, "short", float(a), stop_m, tp_m)
            signals.append(Signal(tsi, "short", px, s, t, pattern="S", max_hold_bars=hold))
            in_pos, side, stop, tp, ei = True, "short", s, t, i
    return signals


def evaluate(df: pd.DataFrame, p: dict) -> dict:
    res = run_backtest(
        df,
        gen_signals(df, p),
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=COSTS,
        max_hold_bars=int(p["max_hold_bars"]),
    )
    m = compute_metrics(res)
    return {
        "n": m["total_trades"],
        "wins": sum(1 for t in res.trades if t.pnl > 0),
        "losses": sum(1 for t in res.trades if t.pnl < 0),
        "wr": m["win_rate"],
        "pf": m["profit_factor"],
        "dd": m["max_drawdown"],
        "ret": m["total_return"],
        "expR": m["expectancy_R"],
        "eq": m["final_equity"],
        "pnl": m["final_equity"] - CAPITAL,
    }


def split(df: pd.DataFrame, frac=0.6):
    cut = int(len(df) * frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def configs(interval: str) -> list[dict]:
    hold = {"15m": 128, "30m": 80, "1h": 48}[interval]
    cds = {"15m": (12, 20), "30m": (6, 12), "1h": (3, 6)}[interval]
    out = []
    for bk, sk in ((3.0, 3.0), (3.5, 3.5), (4.0, 4.0), (3.0, 3.5), (2.5, 3.0)):
        for atrp in (14, 20, 30):
            for mf, ms, mg in ((12, 26, 9), (8, 21, 9), (10, 30, 9)):
                for rp, rm in ((20, 2.0), (30, 2.5), (40, 2.0)):
                    for st, tp in ((2.0, 5.0), (2.5, 5.0), (2.0, 4.0), (2.5, 6.0)):
                        for mode in ("macd_cross",):
                            for cd in cds:
                                for slope in (True,):
                                    out.append(
                                        dict(
                                            ut_buy_key=bk,
                                            ut_sell_key=sk,
                                            ut_buy_atr=atrp,
                                            ut_sell_atr=atrp,
                                            macd_fast=mf,
                                            macd_slow=ms,
                                            macd_signal=mg,
                                            range_period=rp,
                                            range_mult=rm,
                                            stop_atr=st,
                                            tp_atr=tp,
                                            entry_mode=mode,
                                            cooldown=cd,
                                            rf_slope=slope,
                                            max_hold_bars=hold,
                                        )
                                    )
    # stride sample to ~120
    if len(out) > 120:
        out = out[:: max(1, len(out) // 120)]
    return out[:120]


def main() -> int:
    hits = []
    soft = []
    lines = [
        "UT+MACD+RF FAST OPT | $10k | ~0.30% RT | risk 1% | train60/oos40",
        "Goal OOS: Ret>2% PF>1.15 n>=12 MaxDD>=-25%",
        "",
    ]
    # Prioritize higher TFs where costs hurt less
    jobs = [
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "30m"),
        ("PAXGUSDT", "1h"),
        ("PAXGUSDT", "30m"),
        ("BTCUSDT", "15m"),
        ("PAXGUSDT", "15m"),
    ]
    for sym, iv in jobs:
        grid = configs(iv)
        df = load_tf(sym, iv)
        tr_df, oo_df = split(df)
        print(f"=== {sym} {iv} configs={len(grid)} bars={len(df)} ===", flush=True)
        survivors = []
        for i, p in enumerate(grid, 1):
            tr = evaluate(tr_df, p)
            if tr["n"] >= 10 and tr["ret"] > 0 and tr["pf"] > 1.05 and tr["expR"] > 0:
                survivors.append((tr, p))
            if i % 30 == 0:
                print(f"  {i}/{len(grid)} train-ok={len(survivors)}", flush=True)
        survivors.sort(key=lambda x: x[0]["ret"] * x[0]["pf"], reverse=True)
        top = survivors[:40]
        print(f"  train survivors {len(survivors)} → OOS {len(top)}", flush=True)
        local = []
        for tr, p in top:
            oo = evaluate(oo_df, p)
            row = {
                "symbol": sym,
                "interval": iv,
                **p,
                "train_n": tr["n"],
                "train_ret": tr["ret"],
                "train_pf": tr["pf"],
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
            ok = (
                oo["n"] >= 12
                and oo["ret"] > 0.02
                and oo["pf"] > 1.15
                and oo["dd"] >= -0.25
                and oo["expR"] > 0
            )
            soft_ok = oo["n"] >= 8 and oo["ret"] > 0 and oo["pf"] > 1.05 and oo["dd"] >= -0.35
            if ok or soft_ok:
                full = evaluate(df, p)
                row.update(
                    {
                        "full_n": full["n"],
                        "full_wins": full["wins"],
                        "full_losses": full["losses"],
                        "full_wr": full["wr"],
                        "full_ret": full["ret"],
                        "full_pf": full["pf"],
                        "full_dd": full["dd"],
                        "full_pnl": full["pnl"],
                        "full_eq": full["eq"],
                        "hard": ok,
                    }
                )
                local.append(row)
                if ok:
                    hits.append(row)
                else:
                    soft.append(row)
        local.sort(key=lambda r: r["oos_ret"], reverse=True)
        lines.append(f"{sym} {iv}: hard={sum(1 for r in local if r['hard'])} soft={sum(1 for r in local if not r['hard'])}")
        for r in local[:5]:
            tag = "HARD" if r["hard"] else "soft"
            lines.append(
                f"  [{tag}] OOS n={r['oos_n']} WR={r['oos_wr']*100:.1f}% PF={r['oos_pf']:.2f} "
                f"DD={r['oos_dd']*100:.1f}% Ret={r['oos_ret']*100:+.1f}% | "
                f"FULL n={r['full_n']} Ret={r['full_ret']*100:+.1f}% Eq=${r['full_eq']:.0f} | "
                f"UT {r['ut_buy_key']}/{r['ut_buy_atr']} MACD {r['macd_fast']}/{r['macd_slow']}/{r['macd_signal']} "
                f"RF {r['range_period']}/{r['range_mult']} SL{r['stop_atr']} TP{r['tp_atr']} cd={r['cooldown']}"
            )
        if not local:
            lines.append("  (none)")
        lines.append("")

    pool = hits if hits else soft
    pool.sort(key=lambda r: (r.get("hard", False), r["oos_ret"], r["full_ret"]), reverse=True)
    lines += ["BEST", "-" * 90]
    if pool:
        for r in pool[:12]:
            lines.append(
                f"{r['symbol']} {r['interval']} hard={r.get('hard')} "
                f"OOS {r['oos_ret']*100:+.1f}% n={r['oos_n']} WR={r['oos_wr']*100:.1f}% "
                f"PF={r['oos_pf']:.2f} DD={r['oos_dd']*100:.1f}% | "
                f"FULL {r['full_ret']*100:+.1f}% n={r['full_n']} Eq=${r['full_eq']:.0f}"
            )
        b = pool[0]
        lines += [
            "",
            "RECOMMENDED PARAMS",
            f"  Symbol/TF: {b['symbol']} {b['interval']}",
            f"  UT buy/sell key: {b['ut_buy_key']} / {b['ut_sell_key']}  ATR: {b['ut_buy_atr']}",
            f"  MACD: {b['macd_fast']}/{b['macd_slow']}/{b['macd_signal']}",
            f"  Range: {b['range_period']}/{b['range_mult']} (slope filter ON)",
            f"  SL/TP: {b['stop_atr']} / {b['tp_atr']} ATR | cooldown={b['cooldown']} | mode={b['entry_mode']}",
            f"  OOS: trades={b['oos_n']} W/L={b['oos_wins']}/{b['oos_losses']} WR={b['oos_wr']*100:.1f}% "
            f"PF={b['oos_pf']:.2f} MaxDD={b['oos_dd']*100:.1f}% Ret={b['oos_ret']*100:+.1f}% PnL=${b['oos_pnl']:+.0f}",
            f"  FULL: trades={b['full_n']} W/L={b['full_wins']}/{b['full_losses']} WR={b['full_wr']*100:.1f}% "
            f"PF={b['full_pf']:.2f} MaxDD={b['full_dd']*100:.1f}% Ret={b['full_ret']*100:+.1f}% Equity=${b['full_eq']:.0f}",
        ]
    else:
        lines.append("No profitable OOS config found.")

    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / "ut_bot_macd_range_opt_report.txt").write_text(text)
    pd.DataFrame(pool).to_csv(OUT / "ut_bot_macd_range_opt_hits.csv", index=False)
    (OUT / "ut_bot_macd_range_opt_hits.json").write_text(json.dumps(pool[:30], indent=2, default=str))
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
