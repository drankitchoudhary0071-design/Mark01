#!/usr/bin/env python3
"""Backtest UT Bot + MACD + Range Filter — original vs optimized presets."""

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

ORIGINAL = {
    "ut_buy_key": 1.5,
    "ut_buy_atr": 51,
    "ut_sell_key": 1.9,
    "ut_sell_atr": 41,
    "macd_fast": 3,
    "macd_slow": 61,
    "macd_signal": 12,
    "range_period": 51,
    "range_mult": 1.0,
    "stop_atr": 1.5,
    "tp_atr": 3.0,
    "entry_mode": "state",
    "rf_slope": False,
    "cooldown": 0,
    "use_trail_stop": False,
}

OPT_BTC_30M = UtBotMacdRangeStrategy.default_params()

OPT_PAXG_1H = {
    "ut_buy_key": 5.0,
    "ut_buy_atr": 10,
    "ut_sell_key": 5.0,
    "ut_sell_atr": 10,
    "macd_fast": 16,
    "macd_slow": 36,
    "macd_signal": 9,
    "range_period": 40,
    "range_mult": 2.0,
    "stop_atr": 2.5,
    "tp_atr": 5.0,
    "entry_mode": "macd_cross",
    "rf_slope": True,
    "cooldown": 4,
    "use_trail_stop": False,
    "max_hold_bars": 48,
}


def load_tf(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "30m":
        return resample_ohlcv(load_or_fetch(symbol, "15m", DAYS), "30min")
    if interval == "1h":
        return load_or_fetch(symbol, "1h", DAYS)
    return load_or_fetch(symbol, interval, DAYS)


def run_one(symbol: str, interval: str, params: dict, costs: CostModel, label: str) -> dict:
    df = load_tf(symbol, interval)
    hold = int(params.get("max_hold_bars") or {"15m": 128, "30m": 80, "1h": 48}[interval])
    p = {**params, "max_hold_bars": hold}
    strat = UtBotMacdRangeStrategy(**p)
    res = run_backtest(
        df,
        strat.generate_signals(df),
        initial_capital=CAPITAL,
        risk_per_trade=0.01,
        costs=costs,
        max_hold_bars=hold,
        strategy_name=strat.name,
        symbol=symbol,
        params=strat.params,
    )
    m = compute_metrics(res)
    wins = sum(1 for t in res.trades if t.pnl > 0)
    losses = sum(1 for t in res.trades if t.pnl < 0)
    row = {
        "label": label,
        "symbol": symbol,
        "interval": interval,
        "total_trades": m["total_trades"],
        "wins": wins,
        "losses": losses,
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "max_drawdown": m["max_drawdown"],
        "total_return": m["total_return"],
        "final_equity": m["final_equity"],
        "net_pnl": m["final_equity"] - CAPITAL,
    }
    print(
        f"{label:22s} {symbol:10s} {interval:4s} n={row['total_trades']:4d} "
        f"W/L={wins}/{losses} WR={row['win_rate']*100:5.1f}% PF={row['profit_factor']:.2f} "
        f"DD={row['max_drawdown']*100:6.1f}% PnL=${row['net_pnl']:+7.0f} "
        f"Ret={row['total_return']*100:+6.1f}% Eq=${row['final_equity']:.0f}"
    )
    return row


def main() -> int:
    costs = CostModel(commission_rate=0.001, half_spread=0.0002, slippage=0.0003)
    print("UT+MACD+RF | $10k | ~0.30% RT | risk 1% | 365d")
    print("=" * 110)
    rows = []
    # Original on requested TFs
    for sym in ("BTCUSDT", "PAXGUSDT"):
        for iv in ("15m", "30m"):
            rows.append(run_one(sym, iv, ORIGINAL, costs, "ORIGINAL"))
    # Optimized
    rows.append(run_one("BTCUSDT", "30m", OPT_BTC_30M, costs, "OPT_BTC_30M"))
    rows.append(run_one("BTCUSDT", "15m", {**OPT_BTC_30M, "cooldown": 12, "max_hold_bars": 128}, costs, "OPT_on_15m"))
    rows.append(run_one("PAXGUSDT", "1h", OPT_PAXG_1H, costs, "OPT_PAXG_1H"))
    rows.append(
        run_one(
            "PAXGUSDT",
            "1h",
            {
                **OPT_PAXG_1H,
                "ut_buy_key": 4.0,
                "ut_sell_key": 4.0,
                "ut_buy_atr": 14,
                "ut_sell_atr": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "range_period": 50,
                "range_mult": 3.0,
                "use_trail_stop": True,
            },
            costs,
            "OPT_PAXG_1H_trail",
        )
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "ut_bot_macd_range_opt_compare.csv", index=False)
    (OUT / "ut_bot_macd_range_opt_compare.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote compare CSV + see ut_bot_macd_range_opt_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
