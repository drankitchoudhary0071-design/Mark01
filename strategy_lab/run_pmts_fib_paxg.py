#!/usr/bin/env python3
"""Backtest PMTS Fib Trailing on PAXG (gold) 5m / 15m / 1h, 365 days.

SL = 0.7 fib ± buffer points (default sweep 2,3,4,5).
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
from strategy_lab.engine.metrics import compute_metrics, format_metrics
from strategy_lab.strategies.pmts_fib_trailing import PmtsFibTrailingStrategy

OUT = Path(__file__).resolve().parent / "results"

INTERVALS = ("5m", "15m", "1h")
BUFFERS = (2.0, 3.0, 4.0, 5.0)  # points below/above 0.7 fib
DAYS = 365


def max_hold(interval: str) -> int:
    # ~2–3 days calendar
    return {"5m": 600, "15m": 200, "1h": 72}[interval]


def run_one(df: pd.DataFrame, interval: str, buf: float, costs: CostModel) -> dict:
    params = {
        "swing_len": 3,
        "entry_level": 0.6,
        "sl_level": 0.7,
        "sl_buffer_pts": buf,
        "max_hold_bars": max_hold(interval),
    }
    strat = PmtsFibTrailingStrategy(**params)
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=10_000,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=params["max_hold_bars"],
        strategy_name=strat.name,
        symbol="PAXGUSDT",
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    return {
        "interval": interval,
        "sl_buffer_pts": buf,
        "n_signals": len(sigs),
        "trades": m["total_trades"],
        "win_rate": m["win_rate"],
        "expectancy_R": m["expectancy_R"],
        "profit_factor": m["profit_factor"],
        "max_dd": m["max_drawdown"],
        "return": m["total_return"],
        "final_equity": m["final_equity"],
        "long_trades": m["long_trades"],
        "short_trades": m["short_trades"],
        "tp_exits": m["tp_exits"],
        "stop_exits": m["stop_exits"],
        "metrics": m,
        "params": params,
    }


def main() -> int:
    # Prefetch
    for iv in INTERVALS:
        load_or_fetch("PAXGUSDT", iv, DAYS)

    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    light = CostModel(commission_rate=0.001, half_spread=0.0, slippage=0.0)

    rows = []
    print("=" * 78)
    print("PMTS Fib Trailing | PAXGUSDT (Gold) | 365d | SL = 0.7 fib ± buffer pts")
    print(f"Full costs RT≈{costs.round_trip_friction()*100:.2f}% | also TV-light 0.1% only")
    print("=" * 78)

    for iv in INTERVALS:
        df = load_or_fetch("PAXGUSDT", iv, DAYS)
        print(f"\n{iv}: {len(df)} bars  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        for buf in BUFFERS:
            for cname, cmodel in (("full", costs), ("tv_light", light)):
                r = run_one(df, iv, buf, cmodel)
                r["cost_model"] = cname
                rows.append(r)
                m = r["metrics"]
                flag = "★" if m["total_return"] > 0 and m["profit_factor"] > 1 else " "
                print(
                    f" {flag} {iv:4s} buf={buf:.0f}pt [{cname:8s}] "
                    f"n={m['total_trades']:4d} WR={m['win_rate']:.1%} "
                    f"ExpR={m['expectancy_R']:+.3f} PF={m['profit_factor']:.2f} "
                    f"DD={m['max_drawdown']:.1%} RET={m['total_return']*100:+.1f}% "
                    f"TP/SL={m['tp_exits']}/{m['stop_exits']}"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "metrics"} for r in rows]
    pd.DataFrame(clean).to_csv(OUT / "pmts_fib_paxg_365d.csv", index=False)

    # Best per interval under full costs
    full = [r for r in rows if r["cost_model"] == "full"]
    lines = [
        "PMTS Fib Trailing — PAXGUSDT (Gold) 365 days",
        "Logic: BOS swings → Entry 0.6 fib touch → SL 0.7±buffer pts → TP swing anchor",
        "Close beyond 0.7 flips structure. Risk 1%/trade. Full costs ≈0.30% RT.",
        "",
        "RESULTS (full costs)",
        "-" * 70,
    ]
    for iv in INTERVALS:
        subset = [r for r in full if r["interval"] == iv]
        lines.append(f"\n{iv}:")
        for r in subset:
            m = r["metrics"]
            lines.append(
                f"  SL buffer {r['sl_buffer_pts']:.0f} pts | "
                f"n={m['total_trades']} WR={m['win_rate']*100:.1f}% "
                f"ExpR={m['expectancy_R']:+.3f} PF={m['profit_factor']:.2f} "
                f"DD={m['max_drawdown']*100:.1f}% RET={m['total_return']*100:+.1f}% "
                f"→ ${m['final_equity']:,.0f}"
            )
        best = max(subset, key=lambda x: x["metrics"]["total_return"])
        bm = best["metrics"]
        lines.append(
            f"  → Best buffer: {best['sl_buffer_pts']:.0f} pts "
            f"(RET={bm['total_return']*100:+.1f}%, PF={bm['profit_factor']:.2f})"
        )

    prof = [
        r
        for r in full
        if r["metrics"]["total_return"] > 0
        and r["metrics"]["profit_factor"] > 1.0
        and r["metrics"]["expectancy_R"] > 0
        and r["metrics"]["total_trades"] >= 10
    ]
    lines += ["", "PROFITABLE (full costs, n≥10)", "-" * 70]
    if not prof:
        lines.append("None under full ~0.30% RT costs.")
    else:
        for r in sorted(prof, key=lambda x: x["metrics"]["total_return"], reverse=True):
            m = r["metrics"]
            lines.append(
                f"★ {r['interval']} buf={r['sl_buffer_pts']:.0f}pt | "
                f"n={m['total_trades']} WR={m['win_rate']*100:.1f}% "
                f"PF={m['profit_factor']:.2f} ExpR={m['expectancy_R']:+.3f} "
                f"RET={m['total_return']*100:+.1f}% DD={m['max_drawdown']*100:.1f}%"
            )
            print(format_metrics(m, f"PAXG {r['interval']} buf={r['sl_buffer_pts']}"))

    # TV-light profitable
    light_rows = [r for r in rows if r["cost_model"] == "tv_light"]
    light_prof = [
        r
        for r in light_rows
        if r["metrics"]["total_return"] > 0
        and r["metrics"]["profit_factor"] > 1.0
        and r["metrics"]["expectancy_R"] > 0
        and r["metrics"]["total_trades"] >= 10
    ]
    lines += ["", "PROFITABLE (TV-light: 0.1% commission only)", "-" * 70]
    if not light_prof:
        lines.append("None.")
    else:
        for r in sorted(light_prof, key=lambda x: x["metrics"]["total_return"], reverse=True)[:12]:
            m = r["metrics"]
            lines.append(
                f"★ {r['interval']} buf={r['sl_buffer_pts']:.0f}pt | "
                f"n={m['total_trades']} WR={m['win_rate']*100:.1f}% "
                f"PF={m['profit_factor']:.2f} RET={m['total_return']*100:+.1f}%"
            )

    report = "\n".join(lines)
    (OUT / "pmts_fib_paxg_365d_report.txt").write_text(report)
    (OUT / "pmts_fib_paxg_365d.json").write_text(json.dumps(clean, indent=2, default=str))
    print("\n" + report)
    print(f"\nWrote {OUT / 'pmts_fib_paxg_365d_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
