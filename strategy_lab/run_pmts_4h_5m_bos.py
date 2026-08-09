#!/usr/bin/env python3
"""Backtest PMTS 4H Fib Zone + 5m BOS on BTC & Gold (PAXG), 365d, $10k."""

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
from strategy_lab.strategies.pmts_4h_5m_bos import Pmts4h5mBosStrategy

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 10_000.0
DAYS = 365
SYMBOLS = ("BTCUSDT", "PAXGUSDT")


def run_one(symbol: str, costs: CostModel, label: str) -> dict:
    df = load_or_fetch(symbol, "5m", DAYS)
    strat = Pmts4h5mBosStrategy()
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=int(strat.params["max_hold_bars"]),
        strategy_name=strat.name,
        symbol=symbol,
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    row = {
        "label": label,
        "symbol": symbol,
        "interval": "5m",
        "htf": "4h",
        "bars": len(df),
        "signals": len(sigs),
        "total_trades": m["total_trades"],
        "wins": wins,
        "losses": losses,
        "win_rate": m["win_rate"],
        "expectancy_R": m["expectancy_R"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "final_equity": m["final_equity"],
        "net_pnl": m["final_equity"] - CAPITAL,
        "long_trades": m.get("long_trades", 0),
        "short_trades": m.get("short_trades", 0),
        "tp_exits": m.get("tp_exits", 0),
        "stop_exits": m.get("stop_exits", 0),
    }
    print(
        f"{symbol:10s} [{label:12s}] n={row['total_trades']:4d} "
        f"W/L={wins}/{losses} WR={row['win_rate']*100:5.1f}% "
        f"PF={row['profit_factor']:.2f} DD={row['max_drawdown']*100:6.1f}% "
        f"PnL=${row['net_pnl']:+8.0f} Ret={row['total_return']*100:+6.1f}% "
        f"Eq=${row['final_equity']:.0f}"
    )
    return row


def main() -> int:
    full = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    light = CostModel(commission_rate=0.001, half_spread=0.0, slippage=0.0)

    print("PMTS 4H Fib Zone + 5m BOS | Capital $10,000 | 365d | risk 1%/trade")
    print("Zone 0.5–0.7 | SL=0.7 fib | TP=HTF anchor | chart 5m / HTF 4h")
    print("=" * 100)

    rows = []
    print("\nFull costs ~0.30% RT:")
    for sym in SYMBOLS:
        rows.append(run_one(sym, full, "full_0.30%"))

    print("\nTV-like 0.1% commission only:")
    for sym in SYMBOLS:
        rows.append(run_one(sym, light, "tv_0.1%"))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "pmts_4h_5m_bos_365d.csv", index=False)

    lines = [
        "PMTS 4H Fib Golden Zone + 5min BOS Confirmation",
        "Capital $10,000 | 365 days | risk 1%/trade | chart 5m + HTF 4h",
        "Zone 0.5–0.7 | SL = 0.7 fib | TP = HTF anchor extreme",
        "Arm on zone tap; entry on LTF BOS in HTF direction; disarm after entry",
        "",
        "RESULTS",
        "-" * 90,
    ]
    for r in rows:
        lines.append(
            f"{r['symbol']:10s} {r['label']:12s} | Trades={r['total_trades']:4d} "
            f"W/L={r['wins']}/{r['losses']} WR={r['win_rate']*100:.1f}% "
            f"PF={r['profit_factor']:.2f} MaxDD={r['max_drawdown']*100:.1f}% "
            f"PnL=${r['net_pnl']:+.0f} Ret={r['total_return']*100:+.1f}% "
            f"Equity=${r['final_equity']:.0f}"
        )
    lines += [
        "",
        "NOTES",
        "- Lab uses risk sizing 1%/trade (Pine script uses fixed qty=1 — equity path differs).",
        "- HTF mapped with closed-bar only (no lookahead), matching request.security lookahead_off.",
        "- Full costs ≈ 0.10% commission/side + half-spread + slippage (~0.30% RT).",
    ]
    report = OUT / "pmts_4h_5m_bos_365d_report.txt"
    report.write_text("\n".join(lines) + "\n")
    (OUT / "pmts_4h_5m_bos_365d.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
