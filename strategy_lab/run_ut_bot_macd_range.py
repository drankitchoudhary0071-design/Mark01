#!/usr/bin/env python3
"""Backtest UT Bot + MACD + Range Filter on 15m / 30m.

Params (user):
  UT Bot buy  1.5 / 51
  UT Bot sell 1.9 / 41
  MACD 3 / 61 / 12
  Range Filter 51 / 1.0

Capital $10,000 | costs ~0.30% RT | risk 1%/trade | 365d
Symbols: BTCUSDT, PAXGUSDT
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
from strategy_lab.strategies.ut_bot_macd_range import UtBotMacdRangeStrategy

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 10_000.0
DAYS = 365
SYMBOLS = ("BTCUSDT", "PAXGUSDT")
INTERVALS = ("15m", "30m")


def load_tf(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "30m":
        return resample_ohlcv(load_or_fetch(symbol, "15m", DAYS), "30min")
    return load_or_fetch(symbol, interval, DAYS)


def max_hold(interval: str) -> int:
    return {"15m": 96, "30m": 64}[interval]  # ~24h / ~32h


def run_one(symbol: str, interval: str, costs: CostModel) -> dict:
    df = load_tf(symbol, interval)
    params = {
        "ut_buy_key": 1.5,
        "ut_buy_atr": 51,
        "ut_sell_key": 1.9,
        "ut_sell_atr": 41,
        "macd_fast": 3,
        "macd_slow": 61,
        "macd_signal": 12,
        "range_period": 51,
        "range_mult": 1.0,
        "atr_period": 14,
        "stop_atr": 1.5,
        "tp_atr": 3.0,
        "max_hold_bars": max_hold(interval),
    }
    strat = UtBotMacdRangeStrategy(**params)
    sigs = strat.generate_signals(df)
    res = run_backtest(
        df,
        sigs,
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=params["max_hold_bars"],
        strategy_name=strat.name,
        symbol=symbol,
        params=strat.params,
        curve_fit_flags=strat.curve_fit_flags,
    )
    m = compute_metrics(res)
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    return {
        "symbol": symbol,
        "interval": interval,
        "bars": len(df),
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


def main() -> int:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    rows = []
    lines = [
        "UT Bot + MACD + Range Filter | Capital $10,000 | 365d | costs ~0.30% RT | risk 1%",
        "UT buy 1.5/51 | UT sell 1.9/41 | MACD 3/61/12 | Range 51/1.0 | SL 1.5×ATR TP 3×ATR",
        "",
        f"{'Sym':10s} {'TF':4s} {'n':>5s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'PF':>6s} "
        f"{'MaxDD%':>7s} {'PnL$':>9s} {'Ret%':>7s} {'Equity$':>9s}",
        "-" * 90,
    ]
    print(lines[0])
    print(lines[1])
    for sym in SYMBOLS:
        for iv in INTERVALS:
            r = run_one(sym, iv, costs)
            rows.append(r)
            line = (
                f"{r['symbol']:10s} {r['interval']:4s} {r['total_trades']:5d} "
                f"{r['wins']:4d} {r['losses']:4d} {r['win_rate']*100:6.1f} "
                f"{r['profit_factor']:6.2f} {r['max_drawdown']*100:7.1f} "
                f"{r['net_pnl']:9.0f} {r['total_return']*100:7.1f} {r['final_equity']:9.0f}"
            )
            print(line)
            lines.append(line)

    lines += [
        "",
        "NOTES",
        "- 30m built by resampling 15m (no lookahead).",
        "- Entry requires all 3: UT cross + MACD side + Range Filter side.",
        "- One trade at a time; opposite confluence can flip/exit early.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "ut_bot_macd_range_15m30m_report.txt"
    report.write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "ut_bot_macd_range_15m30m.csv", index=False)
    (OUT / "ut_bot_macd_range_15m30m.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
