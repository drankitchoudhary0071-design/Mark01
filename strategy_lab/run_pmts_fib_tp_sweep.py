#!/usr/bin/env python3
"""PMTS Fib on PAXG: TP modes (anchor / 3R / 5R) × SL buffers × TFs incl 4H.

Capital $10,000. Report trades / wins / losses / max DD / return / PF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_lab.data.fetch import load_or_fetch
from strategy_lab.engine.backtest import run_backtest
from strategy_lab.engine.costs import CostModel
from strategy_lab.engine.metrics import compute_metrics
from strategy_lab.strategies.mtf_rsi import resample_ohlcv
from strategy_lab.strategies.pmts_fib_trailing import PmtsFibTrailingStrategy

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 10_000.0
DAYS = 365
INTERVALS = ("5m", "15m", "1h", "4h")
BUFFERS = (2.0, 3.0, 4.0, 5.0)
# TP configs: (label, tp_mode, rr_multiple)
TP_CFGS = (
    ("anchor", "anchor", 0.0),
    ("3R", "rr", 3.0),
    ("5R", "rr", 5.0),
)


def max_hold(interval: str) -> int:
    return {"5m": 600, "15m": 200, "1h": 72, "4h": 30}[interval]


def load_tf(interval: str) -> pd.DataFrame:
    if interval == "4h":
        df1h = load_or_fetch("PAXGUSDT", "1h", DAYS)
        return resample_ohlcv(df1h, "4h")
    return load_or_fetch("PAXGUSDT", interval, DAYS)


def run_one(
    df: pd.DataFrame,
    interval: str,
    buf: float,
    tp_label: str,
    tp_mode: str,
    rr: float,
    costs: CostModel,
) -> dict:
    params = {
        "swing_len": 3,
        "entry_level": 0.6,
        "sl_level": 0.7,
        "sl_buffer_pts": buf,
        "tp_mode": tp_mode,
        "rr_multiple": rr,
        "max_hold_bars": max_hold(interval),
    }
    strat = PmtsFibTrailingStrategy(**params)
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=params["max_hold_bars"],
        strategy_name=strat.name,
        symbol="PAXGUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    wins = int(round(m["win_rate"] * m["total_trades"])) if m["total_trades"] else 0
    # Prefer exact counts from trades list
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    flats = sum(1 for t in res.trades if t.pnl == 0)
    return {
        "interval": interval,
        "sl_buffer_pts": buf,
        "tp": tp_label,
        "capital": CAPITAL,
        "total_trades": m["total_trades"],
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": m["win_rate"],
        "expectancy_R": m["expectancy_R"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "final_equity": m["final_equity"],
        "net_pnl": m["final_equity"] - CAPITAL,
        "long_trades": m["long_trades"],
        "short_trades": m["short_trades"],
        "tp_exits": m["tp_exits"],
        "stop_exits": m["stop_exits"],
    }


def main() -> int:
    for iv in ("5m", "15m", "1h"):
        load_or_fetch("PAXGUSDT", iv, DAYS)

    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    rows: list[dict] = []

    print("=" * 88)
    print("PMTS Fib | PAXGUSDT Gold | $10,000 | 365d | SL 0.7±buf | TP=anchor / 3R / 5R")
    print(f"Costs RT≈{costs.round_trip_friction()*100:.2f}% | risk 1%/trade")
    print("=" * 88)
    hdr = (
        f"{'TF':4s} {'buf':>3s} {'TP':>6s} {'n':>4s} {'W':>3s} {'L':>3s} "
        f"{'WR':>6s} {'PF':>5s} {'DD':>7s} {'PnL':>10s} {'Ret':>7s} {'Equity':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for iv in INTERVALS:
        df = load_tf(iv)
        print(f"# {iv}: {len(df)} bars  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        for buf in BUFFERS:
            for tp_label, tp_mode, rr in TP_CFGS:
                r = run_one(df, iv, buf, tp_label, tp_mode, rr, costs)
                rows.append(r)
                flag = "★" if r["total_return"] > 0 and r["profit_factor"] > 1 else " "
                print(
                    f"{flag}{iv:4s} {buf:3.0f} {tp_label:>6s} {r['total_trades']:4d} "
                    f"{r['wins']:3d} {r['losses']:3d} {r['win_rate']*100:5.1f}% "
                    f"{r['profit_factor']:5.2f} {r['max_drawdown']*100:6.1f}% "
                    f"{r['net_pnl']:+9.0f} {r['total_return']*100:+6.1f}% "
                    f"{r['final_equity']:10.0f}"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pmts_fib_paxg_tp_sweep.csv", index=False)

    # Best overall + best per TF
    lines = [
        "PMTS Fib Trailing — PAXG Gold | Capital $10,000 | 365 days",
        "Entry: 0.6 fib wick touch | SL: 0.7 fib ± buffer pts | same-bar SL allowed",
        "TP modes: anchor (swing extreme) | 3R | 5R  (= multiple of |entry−SL|)",
        f"Costs ≈ {costs.round_trip_friction()*100:.2f}% RT | risk 1%/trade",
        "",
        "FULL TABLE",
        "-" * 88,
        f"{'TF':4s} {'buf':>3s} {'TP':>6s} {'Trades':>6s} {'Wins':>4s} {'Loss':>4s} "
        f"{'WR%':>6s} {'PF':>5s} {'MaxDD%':>7s} {'NetPnL$':>10s} {'Ret%':>7s} {'Equity$':>10s}",
    ]
    for r in rows:
        lines.append(
            f"{r['interval']:4s} {r['sl_buffer_pts']:3.0f} {r['tp']:>6s} "
            f"{r['total_trades']:6d} {r['wins']:4d} {r['losses']:4d} "
            f"{r['win_rate']*100:6.1f} {r['profit_factor']:5.2f} {r['max_drawdown']*100:7.1f} "
            f"{r['net_pnl']:10.0f} {r['total_return']*100:7.1f} {r['final_equity']:10.0f}"
        )

    lines += ["", "BEST PER TIMEFRAME (by net PnL, require PF>1 & n≥5)", "-" * 88]
    for iv in INTERVALS:
        sub = [r for r in rows if r["interval"] == iv and r["total_trades"] >= 5]
        ok = [r for r in sub if r["profit_factor"] > 1 and r["total_return"] > 0]
        pool = ok if ok else sub
        if not pool:
            lines.append(f"{iv}: no trades")
            continue
        best = max(pool, key=lambda x: x["net_pnl"])
        lines.append(
            f"{iv}: TP={best['tp']} SL_buf={best['sl_buffer_pts']:.0f}pt | "
            f"Trades={best['total_trades']} Wins={best['wins']} Losses={best['losses']} "
            f"WR={best['win_rate']*100:.1f}% PF={best['profit_factor']:.2f} "
            f"MaxDD={best['max_drawdown']*100:.1f}% "
            f"NetPnL=${best['net_pnl']:+,.0f} ({best['total_return']*100:+.1f}%) "
            f"Equity=${best['final_equity']:,.0f}"
        )

    # Best TP mode overall among profitable
    lines += ["", "BEST TP MODE OVERALL (profitable only)", "-" * 88]
    prof = [r for r in rows if r["total_return"] > 0 and r["profit_factor"] > 1 and r["total_trades"] >= 5]
    if not prof:
        lines.append("None profitable.")
    else:
        best = max(prof, key=lambda x: x["net_pnl"])
        lines.append(
            f"WINNER → {best['interval']} | TP={best['tp']} | SL buf={best['sl_buffer_pts']:.0f} pts"
        )
        lines.append(
            f"  Capital $10,000 → ${best['final_equity']:,.0f} | "
            f"Trades={best['total_trades']} | Wins={best['wins']} | Losses={best['losses']} | "
            f"WR={best['win_rate']*100:.1f}% | PF={best['profit_factor']:.2f} | "
            f"MaxDD={best['max_drawdown']*100:.1f}% | Net=${best['net_pnl']:+,.0f}"
        )
        # Compare TP modes at that TF+buf
        same = [
            r
            for r in rows
            if r["interval"] == best["interval"] and r["sl_buffer_pts"] == best["sl_buffer_pts"]
        ]
        lines.append("  Same TF+buffer, TP comparison:")
        for r in same:
            lines.append(
                f"    TP={r['tp']:>6s}: n={r['total_trades']:3d} W/L={r['wins']}/{r['losses']} "
                f"PF={r['profit_factor']:.2f} DD={r['max_drawdown']*100:.1f}% "
                f"PnL=${r['net_pnl']:+,.0f} Ret={r['total_return']*100:+.1f}%"
            )

    report = "\n".join(lines)
    (OUT / "pmts_fib_paxg_tp_sweep_report.txt").write_text(report)
    (OUT / "pmts_fib_paxg_tp_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + report)
    print(f"\nWrote {OUT / 'pmts_fib_paxg_tp_sweep_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
